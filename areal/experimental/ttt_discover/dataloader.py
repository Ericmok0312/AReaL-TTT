# areal/experimental/ttt_discover/dataloader.py
from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import torch
from torch.utils.data import IterableDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader

if TYPE_CHECKING:
    from .sampler import StateSampler, PUCTSampler, GreedySampler
    from .state import State


class _StateSamplerIterableDataset(IterableDataset):
    """
    Internal IterableDataset wrapper for StateSampler.
    Handles distributed sharding and infinite sampling.
    """
    
    def __init__(
        self, 
        state_sampler: "StateSampler",
        rank: int,
        world_size: int,
        local_batch_size: int,
        state_to_prompt_fn: Callable[["State"], str] | None = None,
        only_dp_head: bool = False,
    ):
        super().__init__()
        self.state_sampler = state_sampler
        self.rank = rank
        self.world_size = world_size
        self.local_batch_size = local_batch_size
        self.state_to_prompt_fn = state_to_prompt_fn or self._default_prompt_extractor
        self.only_dp_head = only_dp_head
        self._iteration_count = 0
        
    # TODO: Change to return a Message content, or ensure state_to_prompt_fn is a function returns such pattern 
    def _default_prompt_extractor(self, state: "State") -> str:
        """Extract prompt from state for LLM input."""
        if hasattr(state, 'code') and state.code:
            return state.code
        elif hasattr(state, 'observation') and state.observation:
            return state.observation
        return ""
        
    def __iter__(self):
        """Infinite iterator with distributed sharding."""
        # Enforce single worker (PUCT requires stateful single-process access)
        worker_info = get_worker_info()
        if worker_info is not None and worker_info.num_workers > 0:
            raise RuntimeError(
                "TTTDiscoverDataLoader requires num_workers=0 due to PUCT state management"
            )
        
        # Mode 1: Only DP head (rank 0) produces data
        # This is used when rollout is done only on DP head and broadcast to all ranks
        if self.only_dp_head:
            if self.rank != 0:
                # Non-head ranks: don't produce any data
                # prepare_batch will receive data via broadcast from rank 0
                return
            # Rank 0: produce all data
            while True:
                states = self.state_sampler.sample_states(self.local_batch_size)
                for state in states:
                    yield {
                        "prompt": self.state_to_prompt_fn(state),
                        "state_id": state.id,
                        "state_value": state.value,
                        "state_timestep": state.timestep,
                        "parent_values": state.parent_values,
                        "parents": state.parents,
                        "_state_obj": state,
                    }
                    self._iteration_count += 1
        else:
            # Mode 2: Standard distributed sharding (each rank samples its own shard)
            while True:
                global_batch_size = self.local_batch_size * self.world_size
                states = self.state_sampler.sample_states(global_batch_size)
                
                # Shard for current rank
                start_idx = self.rank * self.local_batch_size
                end_idx = start_idx + self.local_batch_size
                local_states = states[start_idx:end_idx]
                
                # Yield individual samples for collation
                for state in local_states:
                    yield {
                        "prompt": self.state_to_prompt_fn(state),
                        "state_id": state.id,
                        "state_value": state.value,
                        "state_timestep": state.timestep,
                        "parent_values": state.parent_values,
                        "parents": state.parents,
                        "_state_obj": state,
                    }
                    self._iteration_count += 1
                
    def state_dict(self) -> dict:
        return {"iteration_count": self._iteration_count}
        
    def load_state_dict(self, state_dict: dict):
        self._iteration_count = state_dict.get("iteration_count", 0)


class TTTDiscoverDataLoader(StatefulDataLoader):
    """
    StatefulDataLoader for TTT-Discover StateSampler.
    
    TTT-Discover does not use a traditional dataset. Instead, PUCTSampler
    manages states internally and generates initial states automatically.
    
    Usage:
        from areal.experimental.ttt_discover.dataloader import create_tttd_dataloader
        sampler = create_sampler(...)
        dataloader = create_tttd_dataloader(
            sampler, rank, world_size, batch_size=8
        )
        
        # Access PUCT features via dataloader.sampler
        for batch in dataloader:
            # ... training ...
            dataloader.sampler.update_states(new_states, parent_states, step=step)
    """
    
    def __init__(
        self,
        state_sampler: "StateSampler",
        rank: int,
        world_size: int,
        batch_size: int,
        collate_fn: Callable | None = None,
        state_to_prompt_fn: Callable[["State"], str] | None = None,
        drop_last: bool = True,
        only_dp_head: bool = False,
        **kwargs
    ):
        self.sampler = state_sampler  # Expose for update_states, flush, etc.
        self._rank = rank
        self._world_size = world_size
        self._only_dp_head = only_dp_head
        
        if only_dp_head:
            # Only DP head produces data, others yield empty
            # batch_size is the total size (not divided by world_size)
            local_batch_size = batch_size
        else:
            # Standard distributed sharding
            if batch_size % world_size != 0:
                raise ValueError(
                    f"batch_size ({batch_size}) must be divisible by "
                    f"world_size ({world_size})"
                )
            local_batch_size = batch_size // world_size
        
        # Create underlying iterable dataset
        self._dataset = _StateSamplerIterableDataset(
            state_sampler=state_sampler,
            rank=rank,
            world_size=world_size,
            local_batch_size=local_batch_size,
            state_to_prompt_fn=state_to_prompt_fn,
            only_dp_head=only_dp_head,
        )
        
        # Initialize parent StatefulDataLoader
        super().__init__(
            dataset=self._dataset,
            batch_size=local_batch_size,
            collate_fn=collate_fn or (lambda x: x),
            num_workers=0,  # Enforced in dataset
            drop_last=drop_last,
            **kwargs
        )
        
    def state_dict(self) -> dict[str, Any]:
        """State dict including both DataLoader and PUCT sampler state."""
        state = super().state_dict()
        state.update({
            'sampler_step': getattr(self.sampler, '_current_step', 0),
            'sampler_type': type(self.sampler).__name__,
            'rank': self._rank,
            'world_size': self._world_size,
        })
        return state
        
    def load_state_dict(self, state_dict: dict[str, Any], reset_puct_stats: bool = True):
        """Restore DataLoader iteration and PUCT sampler state.
        
        Args:
            state_dict: State dict to load
            reset_puct_stats: If True, reset PUCT stats (_T, _n, _m) to 0 after loading.
                            This prevents _T inflation when starting a new experiment.
        """
        super().load_state_dict(state_dict)
        
        if 'sampler_step' in state_dict:
            step = state_dict['sampler_step']
            if hasattr(self.sampler, 'reload_from_step'):
                self.sampler.reload_from_step(step)
            elif hasattr(self.sampler, '_load'):
                self.sampler._load(step)
            
            # Reset PUCT stats if requested (prevents _T inflation across experiments)
            if reset_puct_stats and hasattr(self.sampler, '_T'):
                import logging
                logger = logging.getLogger("TTTDiscoverDataLoader")
                old_T = self.sampler._T
                self.sampler._T = 0
                self.sampler._n = {}
                self.sampler._m = {}
                logger.info(f"[PUCT Stats] Reset _T from {old_T} to 0, cleared _n and _m")
                
    def __len__(self) -> int:
        """Return large number for training loop compatibility."""
        return 1000000000


def create_tttd_dataloader(
    state_sampler: "StateSampler",
    rank: int,
    world_size: int,
    batch_size: int,
    collate_fn: Callable | None = None,
    drop_last: bool = True,
    only_dp_head: bool = False,
    **kwargs
) -> TTTDiscoverDataLoader:
    """
    Create TTTDiscoverDataLoader for StateSampler (PUCT/Greedy).
    
    TTT-Discover does not require a traditional dataset. PUCTSampler manages
    states internally and creates initial states automatically based on
    initial_exp_type and env_type configuration.
    
    Args:
        state_sampler: PUCTSampler, GreedySampler, or FixedSampler instance
        rank: Process rank for distributed training
        world_size: Total number of processes
        batch_size: Total batch size (number of parent states per step)
        collate_fn: Optional custom collation function
        drop_last: Whether to drop last incomplete batch
        only_dp_head: If True, only rank 0 produces data (for prepare_batch mode
                     where DP head does rollout and broadcasts to all ranks)
        **kwargs: Additional args passed to StatefulDataLoader
        
    Returns:
        TTTDiscoverDataLoader instance (StatefulDataLoader subclass)
        
    Example:
        >>> from areal.experimental.ttt_discover.dataloader import create_tttd_dataloader
        >>> from areal.experimental.ttt_discover.sampler import create_sampler
        >>> 
        >>> sampler = create_sampler("puct", log_path="./logs", env_type="ac1")
        >>> dataloader = create_tttd_dataloader(
        ...     sampler, rank=0, world_size=4, batch_size=8
        ... )
        >>> 
        >>> # Training loop
        >>> for batch in dataloader:
        ...     # Access PUCT features
        ...     dataloader.sampler.update_states(states, parents, step=step)
    """
    return TTTDiscoverDataLoader(
        state_sampler=state_sampler,
        rank=rank,
        world_size=world_size,
        batch_size=batch_size,
        collate_fn=collate_fn,
        drop_last=drop_last,
        only_dp_head=only_dp_head,
        **kwargs
    )


# Optional: Convenience factory for common use case
def create_sampler_and_tttd_dataloader(
    log_path: str,
    env_type: str,
    sampler_type: str = "puct",
    batch_size: int = 32,
    rank: int = 0,
    world_size: int = 1,
    budget_s: int = 1000,
    initial_exp_type: str = "random",
    resume_step: int | None = None,
    collate_fn: Callable | None = None,
    **sampler_kwargs
) -> tuple["StateSampler", TTTDiscoverDataLoader]:
    """
    Convenience factory: creates both sampler and dataloader.
    
    Returns:
        Tuple of (sampler, dataloader) for explicit management
    """
    from .sampler import create_sampler
    
    sampler = create_sampler(
        sampler_type=sampler_type,
        log_path=log_path,
        env_type=env_type,
        budget_s=budget_s,
        initial_exp_type=initial_exp_type,
        batch_size=batch_size,
        resume_step=resume_step,
        **sampler_kwargs
    )
    
    # Dummy config object (compatible with AReaL config)
    class _SimpleConfig:
        def __init__(self):
            self.batch_size = batch_size
            self.num_workers = 0
            self.drop_last = True
            
    config = _SimpleConfig()
    
    dataloader = create_tttd_dataloader(
        state_sampler=sampler,
        rank=rank,
        world_size=world_size,
        dataset_config=config,
        collate_fn=collate_fn,
    )
    
    return sampler, dataloader