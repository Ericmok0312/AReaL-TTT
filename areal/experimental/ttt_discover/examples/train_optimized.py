"""
TTT-Discover Training with AReaL Optimization + Correct Group Processing.

This combines:
1. AReaL's prepare_batch for async scheduling and staleness control
2. Proper external group processing for PUCT updates
3. Custom batch collation to preserve group structure

Usage:
    python -m areal.experimental.ttt_discover.examples.train_optimized \
        --config-path config.yaml
"""

import sys
from typing import Any

import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.utils import logging
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.data import concat_padded_tensors

from areal.experimental.ttt_discover.config import TTTDPPOActorConfig
from areal.experimental.ttt_discover.dataloader import TTTDiscoverDataLoader
from areal.experimental.ttt_discover.envs.env import BaseEnv, EnvResult
from areal.experimental.ttt_discover.sampler import create_sampler, PUCTSampler, StateSampler
from areal.experimental.ttt_discover.state import State
from areal.experimental.ttt_discover.workflow import TTTDiscoverWorkflow

logger = logging.getLogger("TTTDOptimized")


class GroupTracker:
    """
    Tracks parent states for group rollout batches.
    
    Since AReaL's prepare_batch processes items asynchronously,
    we need to track which parent state each result belongs to.
    """
    
    def __init__(self):
        self._pending_parents: dict[int, State] = {}  # task_id -> parent
        self._current_batch_parents: list[State] = []
    
    def register_task(self, task_id: int, parent_state: State):
        """Register a task with its parent state."""
        self._pending_parents[task_id] = parent_state
    
    def get_parent(self, task_id: int) -> State | None:
        """Get parent state for a completed task."""
        return self._pending_parents.pop(task_id, None)
    
    def start_new_batch(self, parents: list[State]):
        """Start tracking a new batch."""
        self._current_batch_parents = parents
        self._pending_parents.clear()
    
    def get_batch_parents(self) -> list[State]:
        """Get parents for current batch."""
        return self._current_batch_parents


class GroupBatchCollator:
    """
    Custom collator that preserves group structure and metadata.
    
    Unlike default collation, this keeps track of:
    - Which parent each sample belongs to
    - Group boundaries for post-processing
    """
    
    def __init__(self, group_tracker: GroupTracker, group_size: int):
        self.group_tracker = group_tracker
        self.group_size = group_size
        self._task_id_counter = 0
    
    def __call__(self, batch: list[dict]) -> dict[str, Any]:
        """
        Collate batch while preserving parent metadata.
        
        Args:
            batch: List of data items from dataloader
            
        Returns:
            Collated batch with _parent_states and _group_size fields
        """
        # Extract parent states before collation
        parent_states = []
        for item in batch:
            state = item.get("_state_obj")
            if state:
                parent_states.append(state)
                # Register for tracking
                task_id = self._task_id_counter
                self._task_id_counter += 1
                self.group_tracker.register_task(task_id, state)
        
        # Standard collation (list of dicts -> dict of lists)
        collated = {}
        keys = batch[0].keys()
        
        for key in keys:
            values = [item[key] for item in batch]
            
            if key == "_state_obj":
                # Keep as list (will be expanded by group_size)
                collated[key] = values
            elif isinstance(values[0], torch.Tensor):
                collated[key] = values  # Keep as list for now
            else:
                collated[key] = values
        
        # Add metadata for group processing
        collated["_parent_states"] = parent_states
        collated["_group_size"] = self.group_size
        collated["_num_parents"] = len(parent_states)
        
        return collated


def split_grouped_batch(
    batch: dict[str, torch.Tensor],
    parent_states: list[State],
    group_size: int,
) -> list[tuple[State, list[dict]]]:
    """
    Split a grouped batch into (parent, group_results) pairs.
    
    Args:
        batch: Batch from GroupedRolloutWorkflow
               Shape: [num_parents * group_size, seq_len]
        parent_states: List of parent states (length num_parents)
        group_size: Number of rollouts per parent
        
    Returns:
        List of (parent_state, group_trajectories) tuples
    """
    total_size = batch["input_ids"].shape[0]
    num_parents = len(parent_states)
    
    assert total_size == num_parents * group_size, \
        f"Batch size mismatch: {total_size} != {num_parents} * {group_size}"
    
    groups = []
    
    for i, parent in enumerate(parent_states):
        start = i * group_size
        end = (i + 1) * group_size
        
        # Extract trajectories for this group
        group_trajectories = []
        for j in range(start, end):
            traj = {
                k: v[j] for k, v in batch.items()
                if torch.is_tensor(v) and k not in ["_parent_states", "_group_size", "_num_parents"]
            }
            group_trajectories.append(traj)
        
        groups.append((parent, group_trajectories))
    
    return groups


def extract_code_from_trajectory(traj: dict, workflow: TTTDiscoverWorkflow) -> tuple[str, bool]:
    """
    Extract code from trajectory metadata.
    
    Returns:
        (code, is_valid)
    """
    # Check if metadata was stored
    if "_tttd_metadata" in traj:
        meta = traj["_tttd_metadata"]
        return meta.get("code", ""), meta.get("is_valid", False)
    
    # Fallback: decode and extract
    if "input_ids" in traj:
        # This would require tokenizer, skip for now
        pass
    
    return "", False


def process_groups_and_update_sampler(
    groups: list[tuple[State, list[dict]]],
    sampler: PUCTSampler,
    env: BaseEnv,
    workflow: TTTDiscoverWorkflow,
    topk_per_parent: int = 2,
) -> dict[str, torch.Tensor]:
    """
    Process group results and update PUCTSampler.
    
    Implements:
    - Find best reward y = max(R(child)) for each parent
    - Update m(p) ← max(m(p), y) and n(a) for ancestors
    - Add top-k children to archive
    
    Returns:
        Training batch with best trajectories
    """
    all_children = []
    all_parents = []
    best_trajectories = []
    
    for parent, trajectories in groups:
        # Extract rewards and find best
        rewards = [float(t.get("rewards", -1.0)) for t in trajectories]
        
        # Sort by reward
        indexed_rewards = [(i, r) for i, r in enumerate(rewards)]
        indexed_rewards.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k indices
        topk_indices = [idx for idx, _ in indexed_rewards[:topk_per_parent] if rewards[idx] > -0.5]
        
        if not topk_indices:
            # No valid children
            sampler.record_failed_rollout(parent)
            # Use empty trajectory for training
            best_trajectories.append(trajectories[0] if trajectories else {})
            continue
        
        # Get best reward for m(p) update
        best_reward = rewards[topk_indices[0]]
        
        # Create child states for top-k
        for idx in topk_indices:
            traj = trajectories[idx]
            
            # Extract code and metadata
            code = ""
            observation = ""
            is_valid = False
            metadata = {}
            
            if "_tttd_metadata" in traj:
                meta = traj["_tttd_metadata"]
                code = meta.get("code", "")
                observation = meta.get("observation", "")
                is_valid = meta.get("is_valid", False)
                metadata = meta.get("metadata", {})
            
            if not code or not is_valid:
                continue
            
            try:
                child = env.create_state(
                    parent_state=parent,
                    code=code,
                    reward=rewards[idx],
                    result=EnvResult(
                        reward=rewards[idx],
                        observation=observation,
                        is_valid=is_valid,
                        metadata=metadata,
                    ),
                    timestep=parent.timestep + 1,
                )
                all_children.append(child)
                all_parents.append(parent)
            except Exception as e:
                logger.warning(f"Failed to create child state: {e}")
        
        # Add best trajectory for training
        best_idx = topk_indices[0]
        best_trajectories.append(trajectories[best_idx])
    
    # Update sampler with all children
    if all_children:
        logger.info(f"Updating sampler with {len(all_children)} children from {len(set(p.id for p in all_parents))} parents")
        sampler.update_states(all_children, all_parents, save=False)
    
    # Collate best trajectories
    if not best_trajectories:
        return {}
    
    training_batch = {}
    keys = best_trajectories[0].keys()
    
    for key in keys:
        if key.startswith("_"):
            continue  # Skip metadata
        values = [t[key] for t in best_trajectories if key in t]
        if values and torch.is_tensor(values[0]):
            training_batch[key] = torch.stack(values)
    
    return training_batch


def create_optimized_dataloader(
    sampler: StateSampler,
    rank: int,
    world_size: int,
    batch_size: int,
    group_size: int,
) -> tuple[StatefulDataLoader, GroupTracker]:
    """
    Create optimized dataloader with group tracking.
    
    Args:
        sampler: StateSampler instance
        rank: Process rank
        world_size: Total processes
        batch_size: Number of parents per batch
        group_size: Rollouts per parent (for tracking only)
    
    Returns:
        (dataloader, group_tracker)
    """
    group_tracker = GroupTracker()
    collator = GroupBatchCollator(group_tracker, group_size)
    
    # Simple config object
    class Config:
        def __init__(self):
            self.batch_size = batch_size
            self.num_workers = 0
            self.drop_last = True
    
    dataloader = TTTDiscoverDataLoader(
        state_sampler=sampler,
        rank=rank,
        world_size=world_size,
        dataset_config=Config(),
        collate_fn=collator,
    )
    
    return dataloader, group_tracker


def train_optimized(
    config_path: str,
    env: BaseEnv,
):
    """
    Optimized training loop with AReaL performance + correct PUCT updates.
    
    Configuration:
    - num_parents = 8: Sample 8 parents per iteration  
    - group_size = 64: 64 rollouts per parent
    - Total = 8 * 64 = 512 rollouts
    """
    config, _ = load_expr_config([config_path], TTTDPPOActorConfig)
    
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    
    # Create sampler
    sampler = create_sampler(
        sampler_type="puct",
        log_path=config.saver.trial_name,
        env_type="ac1",
        batch_size=8,        # 8 parents
        group_size=64,       # PUCT G parameter
        topk_children=2,     # Keep top-2 per parent
        max_buffer_size=1000,
        initial_exp_type="random",
    )
    
    # Create optimized dataloader with group tracking
    train_dataloader, group_tracker = create_optimized_dataloader(
        sampler=sampler,
        rank=0,
        world_size=1,
        batch_size=8,    # 8 parents
        group_size=64,   # For tracking
    )
    
    # Create workflow
    workflow = TTTDiscoverWorkflow(
        env=env,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        enable_thinking=False,
    )
    
    # Training with PPOTrainer
    with PPOTrainer(config, train_dataset=None, valid_dataset=None) as trainer:
        # Get inference engine from trainer
        rollout = trainer.rollout  # Or create separately
        
        num_parents = 8
        group_size = 64
        
        for step in range(config.max_steps):
            logger.info(f"Step {step}: Starting group rollout...")
            
            # Step 1: Get parent states and start new batch tracking
            parent_states = sampler.sample_states(num_parents)
            group_tracker.start_new_batch(parent_states)
            
            # Step 2: Use AReaL's optimized prepare_batch
            # This provides:
            # - Async scheduling with staleness control
            # - Automatic GroupedRolloutWorkflow wrapping
            # - Efficient batch processing
            grouped_batch = rollout.prepare_batch(
                train_dataloader,
                workflow=workflow,
                group_size=group_size,
            )
            
            # Shape: [8 * 64, seq_len] = [512, seq_len]
            logger.info(f"Grouped batch shape: {grouped_batch['input_ids'].shape}")
            
            # Step 3: Split by groups using tracked parent states
            parents = group_tracker.get_batch_parents()
            groups = split_grouped_batch(
                batch=grouped_batch,
                parent_states=parents,
                group_size=group_size,
            )
            logger.info(f"Split into {len(groups)} groups")
            
            # Step 4: Process groups and update PUCTSampler
            training_batch = process_groups_and_update_sampler(
                groups=groups,
                sampler=sampler,
                env=env,
                workflow=workflow,
                topk_per_parent=2,
            )
            
            # Step 5: Flush sampler
            sampler.flush(step=step)
            
            # Step 6: Continue with training
            if training_batch:
                logger.info(f"Training batch shape: {training_batch['input_ids'].shape}")
                # ... continue with advantage computation, PPO update, etc.
                # trainer.train_step(training_batch)
            
            logger.info(f"Step {step} complete. Sampler has {len(sampler._states)} states.")


if __name__ == "__main__":
    # Example usage
    # from your_env import YourEnv
    # env = YourEnv(...)
    # train_optimized("config.yaml", env)
    pass
