"""
TTT-Discover Training Example with Group Rollout and PUCT Updates.

This example shows how to:
1. Use AReaL's GroupedRolloutWorkflow for efficient group rollout
2. Process group results externally to update PUCTSampler correctly
3. Implement the PUCT update logic from the paper:
   - m(p) ← max(m(p), y) where y = max(R(child))
   - n(a) ← n(a) + 1 for parent and ancestors
   - Keep top-2 children per parent in archive

Usage:
    python -m areal.experimental.ttt_discover.examples.train_with_group \
        --config-path config.yaml
"""

import sys
from copy import deepcopy

import torch.distributed as dist

from areal import current_platform, PPOTrainer
from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.engine.vllm_remote import RemotevLLMEngine
from areal.utils import seeding, stats_tracker
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger

from areal.experimental.ttt_discover.config import TTTDPPOActorConfig
from areal.experimental.ttt_discover.actor import TTTDActor
from areal.experimental.ttt_discover.dataloader import create_tttd_dataloader
from areal.experimental.ttt_discover.sampler import create_sampler
from areal.experimental.ttt_discover.workflow import TTTDiscoverWorkflow

# Import your environment
# from your_env import YourEnv


def process_group_results_and_update_sampler(
    batch: dict[str, torch.Tensor],
    sampler,
    env,
    group_size: int,
    topk_per_parent: int = 2,
) -> dict[str, torch.Tensor]:
    """
    Process group rollout results and update PUCTSampler.
    
    For each parent state in the batch:
    1. Extract all group results (group_size rollouts)
    2. Find best child: y = max(R(child))
    3. Update PUCT: m(p) ← max(m(p), y), n(a)++ for ancestors
    4. Add top-k children to sampler archive
    
    Args:
        batch: Concatenated batch from GroupedRolloutWorkflow
               Shape: [batch_size * group_size, seq_len]
        sampler: PUCTSampler instance
        env: Environment for creating child states
        group_size: Number of rollouts per parent
        topk_per_parent: Number of top children to keep per parent (default: 2)
    
    Returns:
        Filtered batch for training (e.g., only best results)
    """
    import torch
    
    batch_size = batch["input_ids"].shape[0] // group_size
    
    all_best_trajectories = []
    
    for i in range(batch_size):
        # Extract this parent's group results
        start_idx = i * group_size
        end_idx = (i + 1) * group_size
        
        group_rewards = batch["rewards"][start_idx:end_idx]
        group_trajectories = {
            k: v[start_idx:end_idx] for k, v in batch.items() 
            if k != "_tttd_metadata" and torch.is_tensor(v)
        }
        
        # Get metadata (stored by workflow)
        metadata_list = batch.get("_tttd_metadata", [])[start_idx:end_idx] if "_tttd_metadata" in batch else []
        
        if not metadata_list:
            # No metadata - use all results (fallback)
            all_best_trajectories.append({
                k: v[0:1] for k, v in group_trajectories.items()
            })
            continue
        
        # Find parent state
        parent_state = metadata_list[0].get("parent_state")
        if parent_state is None:
            continue
        
        # Pair results with metadata
        results_with_meta = []
        for j, (reward, meta) in enumerate(zip(group_rewards, metadata_list)):
            if meta.get("is_valid") and meta.get("code"):
                results_with_meta.append({
                    "reward": float(reward),
                    "code": meta["code"],
                    "observation": meta.get("observation", ""),
                    "metadata": meta.get("metadata", {}),
                    "trajectory_idx": j,
                })
        
        if not results_with_meta:
            # No valid results - record failed rollout
            sampler.record_failed_rollout(parent_state)
            continue
        
        # Sort by reward descending
        results_with_meta.sort(key=lambda x: x["reward"], reverse=True)
        
        # Get best reward: y = max(R(child))
        best_reward = results_with_meta[0]["reward"]
        
        # Create child states for top-k results
        children_to_add = []
        for result in results_with_meta[:topk_per_parent]:
            try:
                child_state = env.create_state(
                    parent_state=parent_state,
                    code=result["code"],
                    reward=result["reward"],
                    result=type('obj', (object,), {
                        'reward': result["reward"],
                        'observation': result["observation"],
                        'is_valid': True,
                        'metadata': result["metadata"],
                    })(),
                    timestep=parent_state.timestep + 1,
                )
                children_to_add.append(child_state)
            except Exception as e:
                print(f"Failed to create child state: {e}")
        
        # Update sampler with top-k children
        # This will:
        # 1. Update m(p) ← max(m(p), best_reward)
        # 2. Update n(a) ← n(a) + 1 for parent and ancestors
        # 3. Add children to archive (with topk filtering)
        if children_to_add:
            sampler.update_states(
                children_to_add,
                [parent_state] * len(children_to_add),
                save=False,  # Don't save yet, batch updates
            )
        
        # For training: use best trajectory
        best_idx = results_with_meta[0]["trajectory_idx"]
        best_trajectory = {
            k: v[best_idx:best_idx+1] for k, v in group_trajectories.items()
        }
        all_best_trajectories.append(best_trajectory)
    
    # Concatenate best trajectories for training
    if not all_best_trajectories:
        return {}
    
    from areal.utils.data import concat_padded_tensors
    training_batch = concat_padded_tensors(all_best_trajectories)
    
    return training_batch


def main(args):
    config, _ = load_expr_config(args, TTTDPPOActorConfig)
    
    rank = int(dist.get_rank()) if dist.is_initialized() else 0
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    
    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    
    # Create environment
    # env = YourEnv(...)
    
    # Create PUCTSampler
    # batch_size=8: sample 8 states per iteration
    # group_size=64: PUCT formula parameter G
    sampler = create_sampler(
        sampler_type="puct",
        log_path=config.saver.trial_name,
        env_type="ac1",  # Change to your env type
        batch_size=8,        # 8 states (groups) per batch
        group_size=64,       # PUCT G parameter
        topk_children=2,     # Keep top-2 children per parent
        max_buffer_size=1000, # Global constraint: top-1000 states
        initial_exp_type="random",
    )
    
    # Create dataloader
    train_dataloader = create_tttd_dataloader(
        sampler=sampler,
        rank=rank,
        world_size=1,  # Adjust for distributed
        dataset_config=config.train_dataset,
    )
    
    # Initialize engines
    rollout = RemotevLLMEngine(config.rollout)
    rollout.initialize()
    
    # Create workflow
    workflow = TTTDiscoverWorkflow(
        env=env,  # Your environment
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        enable_thinking=False,
    )
    
    # Training loop with external sampler update
    for global_step in range(config.max_steps):
        # Step 1: Group Rollout
        # group_size=64: each of the 8 states generates 64 rollouts
        # Total: 8 * 64 = 512 rollouts
        batch = rollout.prepare_batch(
            train_dataloader,
            workflow=workflow,
            group_size=64,  # 64 rollouts per state
        )
        
        # batch shape: [8 * 64, seq_len] = [512, seq_len]
        
        # Step 2: Process group results and update PUCTSampler
        # This implements the paper's update logic:
        # - For each parent, find y = max(R(child))
        # - Update m(p) and n(a) for ancestors
        # - Add top-2 children to archive
        training_batch = process_group_results_and_update_sampler(
            batch=batch,
            sampler=sampler,
            env=env,
            group_size=64,
            topk_per_parent=2,  # Keep top-2 children per parent
        )
        
        # Step 3: Flush sampler to disk
        sampler.flush(step=global_step)
        
        # Step 4: Training with best trajectories
        # training_batch now contains only best results: [8, seq_len]
        if training_batch:
            # Compute advantages and update actor
            # ... standard PPO/GRPO training ...
            pass


if __name__ == "__main__":
    main(sys.argv[1:])
