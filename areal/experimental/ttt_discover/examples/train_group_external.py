"""
TTT-Discover Training with External Group Processing and PUCT Updates.

This is the RECOMMENDED approach for TTT-Discover:
1. Use AReaL's GroupedRolloutWorkflow for efficient parallel rollout
2. Process group results externally to correctly implement PUCT updates
3. Only add best/top-k children to sampler (not all group members)

PUCT Update Logic (from paper):
- After expanding parent p with group_size rollouts
- Observe best child reward: y = max(R(child))
- Update m(p) ← max(m(p), y)  [best reward from parent p]
- Update n(a) ← n(a) + 1 for all a ∈ {p} ∪ Anc(p)  [visitation backprop]
- Add top-k children (k=2) to archive
- Maintain global constraint: top-1000 states
"""

import sys
from dataclasses import dataclass
from typing import Any

import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import load_expr_config
from areal.api.engine_api import InferenceEngine
from areal.utils import logging
from areal.utils.hf_utils import load_hf_tokenizer

from areal.experimental.ttt_discover.dataloader import create_tttd_dataloader
from areal.experimental.ttt_discover.envs.env import BaseEnv, EnvResult
from areal.experimental.ttt_discover.sampler import create_sampler, PUCTSampler
from areal.experimental.ttt_discover.state import State
from areal.experimental.ttt_discover.workflow import TTTDiscoverWorkflow

logger = logging.getLogger("TTTDTrain")


@dataclass
class GroupMember:
    """Single member of a group rollout."""
    trajectory: dict[str, torch.Tensor]
    reward: float
    code: str
    observation: str
    is_valid: bool
    metadata: dict[str, Any]


@dataclass  
class GroupResult:
    """Result of a group rollout for one parent state."""
    parent_state: State
    members: list[GroupMember]
    
    def get_best(self) -> GroupMember | None:
        """Get best member by reward."""
        valid = [m for m in self.members if m.is_valid]
        if not valid:
            return None
        return max(valid, key=lambda m: m.reward)
    
    def get_topk(self, k: int = 2) -> list[GroupMember]:
        """Get top-k members by reward."""
        valid = [m for m in self.members if m.is_valid]
        valid.sort(key=lambda m: m.reward, reverse=True)
        return valid[:k]


def split_batch_by_group(
    batch: dict[str, torch.Tensor],
    group_size: int,
    parent_states: list[State],
) -> list[GroupResult]:
    """
    Split concatenated batch into groups by parent.
    
    Args:
        batch: Concatenated batch from GroupedRolloutWorkflow
               Shape: [num_parents * group_size, seq_len]
        group_size: Number of rollouts per parent
        parent_states: List of parent states (length = num_parents)
    
    Returns:
        List of GroupResult, one per parent
    """
    total_size = batch["input_ids"].shape[0]
    num_parents = total_size // group_size
    
    assert len(parent_states) == num_parents, \
        f"Parent states mismatch: {len(parent_states)} vs {num_parents}"
    
    groups = []
    
    for i, parent in enumerate(parent_states):
        start = i * group_size
        end = (i + 1) * group_size
        
        members = []
        for j in range(start, end):
            member = GroupMember(
                trajectory={
                    k: v[j:j+1] for k, v in batch.items()
                    if torch.is_tensor(v) and k != "_tttd_metadata"
                },
                reward=float(batch["rewards"][j]),
                code="",  # Will be filled from metadata if available
                observation="",
                is_valid=True,  # Assume valid if in batch
                metadata={},
            )
            
            # Extract metadata if available
            if "_tttd_metadata" in batch:
                meta = batch["_tttd_metadata"][j]
                member.code = meta.get("code", "")
                member.observation = meta.get("observation", "")
                member.is_valid = meta.get("is_valid", False)
                member.metadata = meta.get("metadata", {})
            
            members.append(member)
        
        groups.append(GroupResult(parent_state=parent, members=members))
    
    return groups


def update_sampler_from_groups(
    groups: list[GroupResult],
    sampler: PUCTSampler,
    env: BaseEnv,
    topk_per_parent: int = 2,
):
    """
    Update PUCTSampler from group results.
    
    Implements the paper's update logic:
    1. For each parent, find best reward y = max(R(child))
    2. Update m(parent) ← max(m(parent), y)
    3. Update n(a) ← n(a) + 1 for parent and all ancestors
    4. Add top-k children to archive
    """
    all_children = []
    all_parents = []
    
    for group in groups:
        parent = group.parent_state
        
        # Get top-k members
        top_members = group.get_topk(k=topk_per_parent)
        
        if not top_members:
            # No valid children - record failed rollout
            sampler.record_failed_rollout(parent)
            continue
        
        # Create child states for top-k members
        for member in top_members:
            try:
                child = env.create_state(
                    parent_state=parent,
                    code=member.code,
                    reward=member.reward,
                    result=EnvResult(
                        reward=member.reward,
                        observation=member.observation,
                        is_valid=member.is_valid,
                        metadata=member.metadata,
                    ),
                    timestep=parent.timestep + 1,
                )
                all_children.append(child)
                all_parents.append(parent)
            except Exception as e:
                logger.warning(f"Failed to create child state: {e}")
    
    # Batch update sampler
    # This updates:
    # - m(p) ← max(m(p), y) for each parent
    # - n(a) ← n(a) + 1 for parent and ancestors
    # - Adds children to archive with top-k filtering
    if all_children:
        sampler.update_states(
            all_children,
            all_parents,
            save=False,  # Don't save yet, batch with flush
        )


def select_best_for_training(groups: list[GroupResult]) -> dict[str, torch.Tensor]:
    """
    Select best trajectory from each group for training.
    
    Returns:
        Batch dict with shape [num_parents, seq_len]
    """
    from areal.utils.data import concat_padded_tensors
    
    best_trajectories = []
    
    for group in groups:
        best = group.get_best()
        if best:
            best_trajectories.append(best.trajectory)
        else:
            # No valid result - use empty trajectory
            best_trajectories.append({
                "input_ids": torch.zeros((1, 1), dtype=torch.int32),
                "loss_mask": torch.zeros((1, 1), dtype=torch.int32),
                "logprobs": torch.zeros((1, 1), dtype=torch.float32),
                "versions": torch.zeros((1, 1), dtype=torch.int32),
                "attention_mask": torch.zeros((1, 1), dtype=torch.bool),
                "rewards": torch.tensor([-1.0], dtype=torch.float32),
            })
    
    return concat_padded_tensors(best_trajectories)


def get_parent_states_from_dataloader(
    dataloader: StatefulDataLoader,
    num_parents: int,
) -> list[State]:
    """
    Get parent states that were used to generate the current batch.
    
    This assumes the dataloader yields data with '_state_obj' field.
    """
    # The states should be stored in the dataloader's iterator
    # or we can extract them from the batch if stored there
    
    # For now, assume we need to track this separately
    # In practice, you might store this in a global or pass through batch
    
    # TODO: Implement based on your dataloader setup
    # Option 1: Store in dataloader.sampler._last_sampled_states
    if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, '_last_sampled_states'):
        return dataloader.sampler._last_sampled_states[:num_parents]
    
    raise ValueError("Cannot determine parent states. Ensure sampler tracks last sampled states.")


def train_with_group_rollout(
    config_path: str,
    env: BaseEnv,
):
    """
    Main training loop with group rollout and external PUCT updates.
    
    Configuration:
    - num_parents = 8: Sample 8 parent states per iteration
    - group_size = 64: Generate 64 rollouts per parent
    - Total rollouts = 8 * 64 = 512
    """
    config, _ = load_expr_config([config_path], TTTDPPOActorConfig)
    
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    
    # Create sampler
    # batch_size=8: maintain 8 best states as parents
    # group_size=64: PUCT parameter G (affects exploration bonus)
    sampler = create_sampler(
        sampler_type="puct",
        log_path=config.saver.trial_name,
        env_type="ac1",  # Change to your env
        batch_size=8,        # Number of parent states
        group_size=64,       # PUCT parameter G
        topk_children=2,     # Keep top-2 children per parent
        max_buffer_size=1000, # Global top-1000 constraint
        initial_exp_type="random",
    )
    
    # Create dataloader
    train_dataloader = create_tttd_dataloader(
        sampler=sampler,
        rank=0,
        world_size=1,
        dataset_config=config.train_dataset,
    )
    
    # Create workflow
    workflow = TTTDiscoverWorkflow(
        env=env,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        enable_thinking=False,
    )
    
    # Initialize inference engine
    from areal.engine.vllm_remote import RemotevLLMEngine
    rollout = RemotevLLMEngine(config.rollout)
    rollout.initialize()
    
    # Training loop
    num_parents = 8
    group_size = 64
    
    for step in range(config.max_steps):
        logger.info(f"Step {step}: Sampling {num_parents} parents...")
        
        # Step 1: Get parent states from sampler
        # These are the states that will be expanded
        parent_states = sampler.sample_states(num_parents)
        
        # Create batch data for each parent
        batch_data = [
            {"_state_obj": state, "prompt": env.get_prompt(state)}
            for state in parent_states
        ]
        
        # Step 2: Group Rollout
        # Each parent generates group_size rollouts
        # Total: num_parents * group_size = 512 rollouts
        logger.info(f"Generating {num_parents * group_size} rollouts...")
        
        all_results = []
        for data in batch_data:
            # Use AReaL's group rollout
            results = rollout.rollout_batch(
                [data],  # Single parent
                workflow=workflow,
                group_size=group_size,
            )
            all_results.extend(results)
        
        # Concatenate all results
        from areal.utils.data import concat_padded_tensors
        batch = concat_padded_tensors(all_results)
        # Shape: [num_parents * group_size, seq_len]
        
        logger.info(f"Batch shape: {batch['input_ids'].shape}")
        
        # Step 3: Split by group and process
        logger.info("Processing group results...")
        
        groups = split_batch_by_group(
            batch=batch,
            group_size=group_size,
            parent_states=parent_states,
        )
        
        # Step 4: Update PUCTSampler
        # Implements paper's update logic:
        # - m(p) ← max(m(p), y) for best reward
        # - n(a) ← n(a) + 1 for ancestors
        # - Add top-2 children per parent
        update_sampler_from_groups(
            groups=groups,
            sampler=sampler,
            env=env,
            topk_per_parent=2,
        )
        
        # Step 5: Flush sampler to disk
        sampler.flush(step=step)
        
        # Step 6: Select best for training
        training_batch = select_best_for_training(groups)
        # Shape: [num_parents, seq_len]
        
        logger.info(f"Training batch shape: {training_batch['input_ids'].shape}")
        
        # Step 7: Training (compute advantages, PPO update, etc.)
        # ... your training code here ...
        
        logger.info(f"Step {step} complete. Sampler has {len(sampler._states)} states.")


if __name__ == "__main__":
    # Example usage
    # from your_env import YourEnv
    # env = YourEnv(...)
    # train_with_group_rollout("config.yaml", env)
    pass
