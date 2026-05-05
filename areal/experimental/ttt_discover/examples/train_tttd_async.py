#!/usr/bin/env python3
"""
TTT-Discover Async Training Script using AReaL's native PPOTrainer.

This script demonstrates how to use AReaL's native async training framework
with the TTT-Discover workflow. Key features:
- Uses TTTDPPOTrainer (extends PPOTrainer) for TTT-Discover specific logic
- Async rollout with max_head_offpolicyness > 0 for better throughput
- RolloutWorkflow-based workflow for native async integration
- TTT-Discover custom advantage computation (entropic objective)
- Distributed sampler synchronization across DP ranks

Usage:
    python train_tttd_async.py --config conf/fsdp_lora_vllm_ac1_qwen3_8b_async.yaml
"""

import sys
import time
import queue as queue_module
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal import PPOTrainer
from areal.api.alloc_mode import _AllocationMode as AllocationMode
from areal.api.cli_args import (
    PPOActorConfig,
    PPOConfig,
    ClusterSpecConfig,
    load_expr_config,
)
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.infra import (
    current_platform,
)

from areal.utils.environ import is_single_controller

from areal.utils import logging, seeding, stats_tracker
from areal.utils.evaluator import Evaluator
from areal.utils.saver import Saver
from areal.utils.recover import RecoverHandler

from areal.experimental.ttt_discover.config import (
    SamplerConfig, 
    TTTDPPOActorConfig,
    create_env_from_config,
)
from areal.experimental.ttt_discover.sampler import create_sampler_from_config
from areal.experimental.ttt_discover.dataloader import create_tttd_dataloader
from areal.experimental.ttt_discover.actor import TTTDActor
from areal.experimental.ttt_discover.workflow_v2 import TTTDiscoverWorkflowV2
from areal.experimental.ttt_discover.ttt_logger import TTTTrainingLogger
from areal.utils.stats_logger import StatsLogger
from areal.utils.hf_utils import load_hf_processor_and_tokenizer

logger = logging.getLogger("train_tttd_async")



class TTTDPPOTrainer(PPOTrainer):
    """
    Custom PPOTrainer for TTT-Discover with:
    - TTTDActor (custom compute_advantages with entropic objective)
    - Sampler-based dataloader
    - Distributed sampler synchronization
    - TTTTrainingLogger for visualization
    - Original recovery handling
    """
    
    def __init__(self, config: TTTDPPOActorConfig):
        # Initialize basic attributes first
        self.config = config
        rank = int(__import__('os').getenv("RANK", "0"))
        if is_single_controller():
            logging.setup_file_logging(StatsLogger.get_log_path(config.stats_logger))
        
        # Load tokenizer and processor
        self.processor, self.tokenizer = load_hf_processor_and_tokenizer(
            config.tokenizer_path
        )
        
        # Initialize scheduler
        self.scheduler = None
        if is_single_controller():
            self.scheduler = self._init_scheduler()
        
        # Set seed
        seeding.set_random_seed(config.seed, key=f"trainer{rank}")
        
        # Parse allocation mode
        self.allocation_mode = AllocationMode.from_str(config.allocation_mode)
        self._amend_xccl_weight_update_envvar()
        
        # Create sampler first (needed for dataloader)
        # Note: create_sampler_from_config uses config.checkpoint_dir by default.
        # Do NOT override log_path to ensure sampler checkpoints are saved to the
        # configured directory (cluster.fileroot/experiment_name/trial_name/sampler/)
        # 
        # max_version_history should be >= max_head_offpolicyness + 1 to support lazy sampling
        # with all possible policy versions in flight.
        max_head_offpolicyness = getattr(config.rollout, 'max_head_offpolicyness', 2)
        max_version_history = max_head_offpolicyness + 1
        logger.info(f"[SAMPLER_CONFIG] max_head_offpolicyness={max_head_offpolicyness}, "
                   f"max_version_history={max_version_history} (auto-computed)")
        self.sampler = create_sampler_from_config(
            config=config.sampler,
            env_type=getattr(config.sampler, 'env_type', 'ac1'),
            max_version_history=max_version_history,
        )
        
        # Create TTTDActor (no critic - TTT-Discover doesn't use value function)
        # Pass full config (TTTDPPOActorConfig) instead of config.actor to enable
        # entropic advantage computation with adv_estimator settings
        self.actor = self._create_tttd_actor(config)
        # No critic - TTT-Discover uses entropic objective without value function
        self.ref = None
        # Use top-level config.kl_ctl instead of config.actor.kl_ctl
        if config.kl_ctl > 0 and config.ref is not None:
            # ref model only needs PPOActorConfig (no adv_estimator needed)
            self.ref = self._create_tttd_actor(config.ref)
        
        # REMOVED: No longer override max_concurrent_rollouts / max_head_offpolicyness
        # for lazy sampling. We now rely on the config values (e.g. max_concurrent_rollouts=8)
        # to naturally bound the number of in-flight rollouts, which keeps PUCT sampling
        # fresh because at most one batch is executing at a time.
        if config.sampler.lazy_puct_sampling:
            logger.info(
                f"[LAZY_CONFIG] Using configured rollout limits: "
                f"max_concurrent_rollouts={config.rollout.max_concurrent_rollouts}, "
                f"max_head_offpolicyness={config.rollout.max_head_offpolicyness}, "
                f"vllm_concurrency={config.sampler.vllm_concurrency}, "
                f"execution_concurrency={config.sampler.execution_concurrency}"
            )

        # Create dataloaders using sampler (before engine init, only needs process group)
        self.train_dataloader = self._create_tttd_dataloader(
            sampler=self.sampler,
            rank=self.actor.data_parallel_rank,
            world_size=self.actor.data_parallel_world_size,
            batch_size=config.sampler.batch_size,
            lazy_sampling=config.sampler.lazy_puct_sampling,
        )
        self.train_dataset = self.train_dataloader.dataset
        self.valid_dataloader = None
        self.valid_dataset = None
        
        # Initialize inference engines
        self.rollout = self._init_rollout(config.rollout, is_eval=False)
        
        # Initialize models
        self._initialize_engines()
        
        # Connect sampler to actor for distributed synchronization
        # Must be after _initialize_engines() because it uses self.cpu_group
        self.actor.connect_sampler(self.sampler)
        
        # Setup weight update meta
        self._setup_weight_update_meta()
        
        # Setup evaluation, saver, recover, stats logger
        self._setup_utilities()
        
        # Initialize training history logger (TTT-Discover specific)
        self._init_training_history_logger()
        
        # Initialize proxy workers flag
        self._proxy_started = False
        
        # Store workflow kwargs for later use (updated in train method)
        self._workflow_kwargs = {}
    
    def _create_tttd_actor(self, actor_config: TTTDPPOActorConfig):
        """Create TTTDActor with custom compute_advantages."""
        # Always create TTTDActor directly (not as_controller) to avoid RPC issues
        # This matches the behavior of train_tttd_vllm_v2.py
        actor = TTTDActor(config=actor_config)
        actor.create_process_group(parallel_strategy=self.allocation_mode.train)
        return actor
    
    def _create_tttd_dataloader(
        self,
        sampler,
        rank: int,
        world_size: int,
        batch_size: int,
        lazy_sampling: bool = False,
    ) -> StatefulDataLoader:
        """Create TTT-Discover dataloader with sampler."""
        return create_tttd_dataloader(
            state_sampler=sampler,
            rank=rank,
            world_size=world_size,
            batch_size=batch_size,
            lazy_sampling=lazy_sampling,
        )
    
    def _initialize_engines(self):
        """Initialize training engines."""
        # TTT-Discover: max_steps is the actual number of training steps
        max_steps = getattr(self.config, 'max_steps', self.config.total_train_epochs)
        ft_spec = FinetuneSpec(
            total_train_epochs=max_steps,  # Each step is effectively an epoch
            dataset_size=max_steps * self.config.sampler.batch_size,
            train_batch_size=self.config.sampler.batch_size,
        )
        
        engine_init_kwargs = {
            "addr": None,
            "ft_spec": ft_spec,
            "alloc_mode": self.allocation_mode,
        }
        
        self.actor.initialize(**engine_init_kwargs, role="actor")
        if self.ref is not None:
            self.ref.initialize(**engine_init_kwargs, role="ref")
    
    def _setup_weight_update_meta(self):
        """Setup weight update meta and connect to inference engine."""
        config = self.config
        
        # Use top-level config values (same as train_tttd_vllm_v2.py)
        # because TTTDPPOActorConfig inherits these from PPOActorConfig
        if config.weight_update_mode == "disk":
            disk_kwargs = {
                "experiment_name": config.experiment_name,
                "trial_name": config.trial_name,
                "file_root": config.cluster.fileroot,
                "name": "default",
                "clear_checkpoint_after_load": True,
            }
            if config.use_lora:
                disk_kwargs.update({
                    "use_lora": config.use_lora,
                    "lora_name": config.gconfig.lora_name,
                    "lora_int_id": 1,
                    "base_model_name": config.path,
                })
            self.weight_update_meta = WeightUpdateMeta.from_disk(**disk_kwargs)
        elif config.weight_update_mode == "xccl":
            if self.allocation_mode.train_backend == "megatron":
                self.weight_update_meta = WeightUpdateMeta.from_megatron_xccl(
                    self.allocation_mode
                )
            else:
                xccl_kwargs = {"allocation_mode": self.allocation_mode}
                if config.use_lora:
                    xccl_kwargs.update({
                        "use_lora": config.use_lora,
                        "lora_name": config.gconfig.lora_name,
                        "lora_int_id": 1,
                        "base_model_name": config.path,
                    })
                self.weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(**xccl_kwargs)
        else:
            raise ValueError(
                f"Invalid weight update mode: {config.weight_update_mode}"
            )
        
        self.actor.connect_engine(self.rollout, self.weight_update_meta)
    
    def _setup_utilities(self):
        """Setup evaluator, saver, recover handler, and stats logger."""
        config = self.config
        
        # TTT-Discover: Use max_steps for consistent dataset_size calculation
        max_steps = getattr(config, 'max_steps', config.total_train_epochs)
        ft_spec = FinetuneSpec(
            total_train_epochs=max_steps,  # Each step is an epoch
            dataset_size=max_steps * config.sampler.batch_size,
            train_batch_size=config.sampler.batch_size,
        )
        
        self.evaluator = Evaluator(config.evaluator, ft_spec)
        self.saver = Saver(config.saver, ft_spec)
        self.recover_handler = RecoverHandler(config.recover, ft_spec)
        self.stats_logger = StatsLogger(config, ft_spec)
        
        # Load recovery info (original recovery handling)
        self.recover_info = self.recover_handler.load(
            self.actor,
            self.saver,
            self.evaluator,
            self.stats_logger,
            self.train_dataloader,
            inference_engine=self.rollout,
            weight_update_meta=self.weight_update_meta,
        )
        
        # Clear stale state after recovery (same as original)
        self._clear_stale_state_after_recovery()
        
        self._config_perf_tracer()
    
    def _get_dispatcher(self):
        """Get dispatcher from rollout (handles both RolloutController and RemotevLLMEngine)."""
        # Try RolloutController first
        if hasattr(self.rollout, 'dispatcher'):
            return self.rollout.dispatcher
        # Try RemotevLLMEngine
        if hasattr(self.rollout, '_engine') and hasattr(self.rollout._engine, 'workflow_executor'):
            return self.rollout._engine.workflow_executor._dispatcher
        # Try workflow_executor directly
        if hasattr(self.rollout, 'workflow_executor'):
            return self.rollout.workflow_executor._dispatcher
        return None
    
    def _get_workflow_executor(self):
        """Get workflow executor from rollout."""
        if hasattr(self.rollout, '_engine') and hasattr(self.rollout._engine, 'workflow_executor'):
            return self.rollout._engine.workflow_executor
        if hasattr(self.rollout, 'workflow_executor'):
            return self.rollout.workflow_executor
        return None
    
    def _clear_stale_state_after_recovery(self):
        """Clear stale state after recovery - same logic as original train_tttd_vllm_v2.py"""
        is_dp_head = self.actor.rank == 0  # Global rank 0 is the unique head for logging/saving
        
        if self.recover_info:
            start_step = self.recover_info.last_step_info.next().global_step
            
            if is_dp_head:
                logger.info(f"[Resume] Cleaning up stale state for step {start_step}")
            
            dispatcher = self._get_dispatcher()
            workflow_executor = self._get_workflow_executor()
            
            if dispatcher is not None:
                # 1. Clear _pending_inputs
                with dispatcher._input_lock:
                    stale_inputs = len(dispatcher._pending_inputs)
                    if stale_inputs > 0:
                        dispatcher._pending_inputs.clear()
                        if is_dp_head:
                            logger.info(f"[Resume] Cleared {stale_inputs} stale pending inputs")
                
                # 2. Clear AsyncTaskRunner queues
                runner = dispatcher.runner
                queue_cleared = 0
                for q in [runner.input_queue, runner.output_queue]:
                    while not q.empty():
                        try:
                            q.get_nowait()
                            queue_cleared += 1
                        except queue_module.Empty:
                            break
                if is_dp_head and queue_cleared > 0:
                    logger.info(f"[Resume] Cleared {queue_cleared} items from async task queues")
                
                # 3. Fix accepted count for staleness capacity calculation
                sm = dispatcher.staleness_manager
                with sm.lock:
                    consumer_bs = sm.consumer_batch_size
                    completed_steps = start_step
                    expected_accepted = completed_steps * consumer_bs
                    
                    old_accepted = sm.rollout_stat.accepted
                    sm.rollout_stat.running = 0
                    sm.rollout_stat.enqueued = 0
                    
                    if old_accepted != expected_accepted:
                        sm.rollout_stat.accepted = expected_accepted
                        if is_dp_head:
                            logger.info(f"[Resume] Fixed accepted count: {old_accepted} -> {expected_accepted}")
                    
                    version = self.rollout.get_version()
                    max_staleness = sm.max_staleness
                    capacity = (max_staleness + version + 1) * consumer_bs - sm.rollout_stat.accepted
                    if is_dp_head:
                        logger.info(f"[Resume] staleness_capacity = ({max_staleness} + {version} + 1) * {consumer_bs} - {sm.rollout_stat.accepted} = {capacity}")
                
                # 4. Clear _pending_results
                with dispatcher._result_lock:
                    stale_results = len(dispatcher._pending_results)
                    if stale_results > 0:
                        dispatcher._pending_results.clear()
                        dispatcher._active_task_ids.clear()
                        if is_dp_head:
                            logger.info(f"[Resume] Cleared {stale_results} stale pending results")
            
            # 5. Clear data_generator cache
            if workflow_executor is not None and hasattr(workflow_executor, 'data_generator'):
                delattr(workflow_executor, 'data_generator')
                if is_dp_head:
                    logger.info("[Resume] Cleared data generator cache")
            
            # 6. Clear workflow internal state (Scheme 1, staleness tracker, etc.)
            self._clear_workflow_state_after_recovery(start_step)
            
            if is_dp_head:
                logger.info(f"[Resume] Ready to start from step {start_step}")
    
    def _clear_workflow_state_after_recovery(self, start_step: int):
        """Clear workflow internal state after recovery.
        
        This clears all transient state that should not persist across resumes:
        - Scheme 1 batch tracking
        - Staleness tracker (incomplete rollouts from old version)
        - Pending updates
        - Failed parents cache
        - Rollout metadata
        
        Args:
            start_step: The step to resume from (used to reset version)
        """
        is_dp_head = self.actor.rank == 0
        
        # Get workflow from rollout
        workflow_executor = self._get_workflow_executor()
        if workflow_executor is None:
            return
        
        workflow = None
        if hasattr(workflow_executor, 'workflow'):
            workflow = workflow_executor.workflow
        elif hasattr(workflow_executor, '_workflow'):
            workflow = workflow_executor._workflow
        
        if workflow is None:
            return
        
        cleared_items = []
        
        # Clear Scheme 1 batch tracking
        if hasattr(workflow, '_current_batch_parent_ids'):
            stale_count = len(workflow._current_batch_parent_ids)
            if stale_count > 0:
                workflow._current_batch_parent_ids.clear()
                workflow._expected_batch_size = 0
                workflow._expected_n_samples = 0
                cleared_items.append(f"scheme1_batch({stale_count})")
        
        # Clear staleness tracker (incomplete rollouts from old model version)
        if hasattr(workflow, '_staleness_tracker'):
            stale_count = len(workflow._staleness_tracker)
            if stale_count > 0:
                workflow._staleness_tracker.clear()
                cleared_items.append(f"staleness_tracker({stale_count})")
        
        # Clear parent episodes
        if hasattr(workflow, '_parent_episodes'):
            stale_count = len(workflow._parent_episodes)
            if stale_count > 0:
                workflow._parent_episodes.clear()
                cleared_items.append(f"parent_episodes({stale_count})")
        
        # Clear pending updates (async buffer)
        if hasattr(workflow, '_pending_children'):
            stale_count = len(workflow._pending_children)
            if stale_count > 0:
                workflow._pending_children.clear()
                workflow._pending_parents.clear()
                cleared_items.append(f"pending_updates({stale_count})")
        
        # Clear failed parents cache
        if hasattr(workflow, '_failed_parents'):
            stale_count = len(workflow._failed_parents)
            if stale_count > 0:
                workflow._failed_parents.clear()
                cleared_items.append(f"failed_parents({stale_count})")
        
        # Clear rollout metadata
        if hasattr(workflow, '_rollout_metadata'):
            stale_count = len(workflow._rollout_metadata)
            if stale_count > 0:
                workflow._rollout_metadata.clear()
                cleared_items.append(f"rollout_metadata({stale_count})")
        
        # Clear PUCT update log
        if hasattr(workflow, '_puct_update_log'):
            stale_count = len(workflow._puct_update_log)
            if stale_count > 0:
                workflow._puct_update_log.clear()
                cleared_items.append(f"puct_log({stale_count})")
        
        # Clear parent stats
        if hasattr(workflow, '_parent_stats'):
            stale_count = len(workflow._parent_stats)
            if stale_count > 0:
                workflow._parent_stats.clear()
                cleared_items.append(f"parent_stats({stale_count})")
        
        # Clear rollout timing pairs
        if hasattr(workflow, '_rollout_timing_pairs'):
            stale_count = len(workflow._rollout_timing_pairs)
            if stale_count > 0:
                workflow._rollout_timing_pairs.clear()
                cleared_items.append(f"timing_pairs({stale_count})")
        
        # Reset current version to match resumed step
        if hasattr(workflow, '_current_version'):
            old_version = workflow._current_version
            workflow._current_version = start_step
            if old_version != start_step and is_dp_head:
                cleared_items.append(f"version({old_version}->{start_step})")
        
        if cleared_items and is_dp_head:
            logger.info(f"[Resume] Cleared workflow state: {', '.join(cleared_items)}")
    
    def _init_training_history_logger(self):
        """Initialize TTTTrainingLogger for visualization."""
        config = self.config
        
        # Get start step from recovery
        start_step = (
            self.recover_info.last_step_info.next().global_step
            if self.recover_info is not None
            else 0
        )
        max_steps = getattr(config, 'max_steps', config.total_train_epochs)
        
        # Generate save_steps
        all_save_steps = getattr(config, 'save_steps', list(range(0, max_steps)))
        future_save_steps = [s for s in all_save_steps if start_step <= s < max_steps]
        
        is_dp_head = self.actor.rank == 0  # Global rank 0 is the unique head for logging/saving
        
        # Each rank saves its own data independently - no distributed aggregation
        # This avoids all_gather_object timeout issues with large metadata
        rank_suffix = f"_rank{self.actor.dp_rank}" if self.actor.data_parallel_world_size > 1 else ""
        self.history_logger = TTTTrainingLogger(
            save_steps=future_save_steps,
            output_dir=config.saver.fileroot,
            is_dp_head=True,  # Every rank is its own "head" for saving
            filename=f'training_history{rank_suffix}.pkl',
            checkpoint_filename=f'training_history_checkpoint{rank_suffix}.pkl',
            aggregate_distributed=False,  # No aggregation - each rank keeps its own data
        )
        
        if is_dp_head:
            existing_snapshots = len(self.history_logger.history) if self.recover_info else 0
            logger.info(f"[TTTLogger] Initialized for step {start_step} to {max_steps}")
            logger.info(f"[TTTLogger] Future save_steps: {future_save_steps}")
            logger.info(f"[TTTLogger] Existing snapshots from checkpoint: {existing_snapshots}")
            logger.info(f"[TTTLogger] Output directory: {config.saver.fileroot}")
        
        # DEBUG: Print PUCTSampler state after recovery (all ranks)
        if hasattr(self.sampler, '_states'):
            logger.info(
                f"[PUCT_RESUME][Rank {self.actor.dp_rank}] "
                f"States: {len(self.sampler._states)}, "
                f"T: {getattr(self.sampler, '_T', 'N/A')}, "
                f"n: {len(getattr(self.sampler, '_n', {}))}, "
                f"m: {len(getattr(self.sampler, '_m', {}))}, "
                f"Step: {getattr(self.sampler, '_current_step', 'N/A')}"
            )
            if self.sampler._states:
                values = [s.value for s in self.sampler._states if s.value is not None]
                if values:
                    logger.info(
                        f"[PUCT_RESUME][Rank {self.actor.dp_rank}] "
                        f"Buffer values: min={min(values):.4f}, max={max(values):.4f}, "
                        f"mean={sum(values)/len(values):.4f}"
                    )
    
    def train(
        self,
        workflow,
        eval_workflow=None,
        workflow_kwargs: dict = None,
        eval_workflow_kwargs: dict = None,
    ):
        r"""Main training loop with TTT-Discover workflow and logging.
        
        TTT-Discover uses max_steps (not epochs) for training schedule.
        Each step samples from PUCTSampler, performs rollouts, and updates the policy.
        
        """
        config = self.config
        
        # Update workflow kwargs with sampler and DP info
        if workflow_kwargs is not None:
            self._workflow_kwargs = workflow_kwargs.copy()
            # Add sampler reference for lazy sampling
            if config.sampler.lazy_puct_sampling and 'sampler' not in self._workflow_kwargs:
                self._workflow_kwargs['sampler'] = self.sampler
                self._workflow_kwargs['dp_rank'] = self.actor.dp_rank
                self._workflow_kwargs['dp_world_size'] = self.actor.data_parallel_world_size
        
        # Create workflow instance from class and kwargs
        # (needed for calling methods like cleanup_old_versions)
        if isinstance(workflow, type):
            workflow = workflow(**self._workflow_kwargs)
        
        # Save initial PUCT snapshot (version 0) for step 0 sampling
        # This ensures step 0 uses the initial PUCT state, step 1 uses step 0's result, etc.
        if config.sampler.lazy_puct_sampling and hasattr(self.sampler, 'save_version_snapshot'):
            self.sampler.save_version_snapshot(0)
            logger.info(f"[INIT] Saved initial PUCT snapshot version=0 for step 0 sampling")
        
        start_step = (
            self.recover_info.last_step_info.next().global_step
            if self.recover_info is not None
            else 0
        )
        
        # TTT-Discover: Use max_steps directly (not steps_per_epoch * epochs)
        max_steps = getattr(config, 'max_steps', config.total_train_epochs)
        
        is_dp_head = self.actor.rank == 0  # Global rank 0 is the unique head for logging/saving
        batch_size = config.sampler.batch_size
        group_size = config.gconfig.n_samples
        best_reward = float('-inf')
        
        # DEBUG: Log critical values
        logger.info(f"[DEBUG] max_steps={max_steps}, start_step={start_step}, range={start_step}-{max_steps-1}")
        logger.info(f"[DEBUG] config.max_steps={getattr(config, 'max_steps', None)}, config.total_train_epochs={config.total_train_epochs}")
        
        logger.info(f"Starting training from step {start_step}/{max_steps}")
        
        loop_count = 0
        for global_step in range(start_step, max_steps):
            loop_count += 1
            logger.info(f"[DEBUG] Loop #{loop_count}: global_step={global_step}, max_steps={max_steps}")
            
            # TTT-Discover uses max_steps for termination
            max_steps_limit = getattr(config, 'max_steps', None)
            if max_steps_limit is not None and global_step >= max_steps_limit:
                logger.info(f"[DEBUG] Breaking loop at global_step={global_step}, max_steps_limit={max_steps_limit}")
                break
            
            # Clean up old batch caches for lazy sampling
            # Keep max_head_offpolicyness + 1 batches to support old rollouts
            if config.sampler.lazy_puct_sampling and hasattr(workflow, 'cleanup_old_versions'):
                max_history = getattr(config.rollout, 'max_head_offpolicyness', 2) + 1
                workflow.cleanup_old_versions(global_step, max_history=max_history)
            
            # In TTT-Discover, each step is effectively an epoch
            # (we don't use traditional epochs since data comes from PUCTSampler)
            epoch = global_step
            step = global_step
            steps_per_epoch = max_steps
            
            step_info = StepInfo(
                global_step=global_step,
                epoch=epoch,
                epoch_step=step,
                steps_per_epoch=steps_per_epoch,
            )
            
            # === Simplified Timing: only 3 phases ===
            step_start_time = time.perf_counter()
            
            # Note: We intentionally do NOT call workflow.reset() here.
            # In pure async mode, we want to process ALL completed rollouts,
            # including those from previous steps that finished late.
            # get_pending_updates(clear=True) handles cleanup automatically.
            
            # Set current version for staleness tracking
            # Staleness = current training step - sampled_step
            if hasattr(workflow, 'set_current_version'):
                workflow.set_current_version(global_step)
            
            # Scheme 1: Enable batch tracking for sync-like PUCT update
            # Scheme 1 = wait for complete batch before PUCT update (slower, more accurate)
            # Scheme 2 = use available data immediately (faster, default)
            use_scheme_1 = getattr(config, 'use_scheme_1', False)
            if use_scheme_1 and hasattr(workflow, 'set_current_batch_tracking'):
                workflow.set_current_batch_tracking(
                    step=global_step,
                    batch_size=batch_size,
                    n_samples=group_size
                )
                logger.info(f"[SCHEME1][Step {global_step}] Enabled batch tracking")
            
            # === Rollout with async prepare_batch ===
            rollout_start = time.perf_counter()
            with stats_tracker.record_timing("rollout"):
                rollout_batch = self.actor.prepare_batch(
                    self.train_dataloader,
                    workflow=workflow,
                    workflow_kwargs=workflow_kwargs,
                    should_accept_fn=None,
                    group_size=group_size,
                    dynamic_bs=config.dynamic_bs,
                )
            rollout_time = time.perf_counter() - rollout_start
            
            # === Rollout token counting for MFU calculation ===
            rollout_total_tokens = 0
            if "attention_mask" in rollout_batch:
                rollout_total_tokens = int(rollout_batch["attention_mask"].sum().item())
            elif "input_ids" in rollout_batch:
                rollout_total_tokens = rollout_batch["input_ids"].numel()
            
            if dist.is_initialized():
                local_rollout_tokens = torch.tensor([rollout_total_tokens], dtype=torch.long, device=self.actor.device)
                dist.all_reduce(local_rollout_tokens, op=dist.ReduceOp.SUM)
                global_rollout_tokens = int(local_rollout_tokens.item())
            else:
                global_rollout_tokens = rollout_total_tokens
            
            # Get execute tail latency from workflow if available
            execute_tail = 0.0
            timing_stats = {}
            if hasattr(workflow, 'get_execute_tail_latency'):
                execute_tail = workflow.get_execute_tail_latency(clear=True)
                if hasattr(workflow, 'get_timing_stats'):
                    timing_stats = workflow.get_timing_stats(clear=False)
            
            # Scheme 1: Wait for batch completion before getting updates
            scheme_1_wait_time = 0.0
            if use_scheme_1 and hasattr(workflow, 'wait_for_batch_completion'):
                logger.info(f"[SCHEME1][Step {global_step}] Waiting for batch completion...")
                scheme_1_wait_time = workflow.wait_for_batch_completion(batch_step=global_step)
                logger.info(f"[SCHEME1][Step {global_step}] Waited {scheme_1_wait_time:.2f}s")
            
            # === Sampler Synchronization (part of training phase) ===
            # Scheme 1: Use parent_ids filter to get complete batch
            if use_scheme_1 and hasattr(workflow, 'get_current_batch_parent_ids'):
                batch_parent_ids = workflow.get_current_batch_parent_ids()
                puct_update_result = workflow.get_pending_updates(
                    clear=True,
                    current_step=global_step,
                    parent_ids=batch_parent_ids
                )
                workflow.clear_current_batch_tracking()
            else:
                # Scheme 2: Use all available data (default)
                puct_update_result = workflow.get_pending_updates(clear=True, current_step=global_step)
            
            # Handle both old (4-tuple) and new (5-tuple) return formats
            if len(puct_update_result) == 5:
                local_children, local_parents, local_failed, rollout_metadata, cross_batch_info = puct_update_result
            else:
                local_children, local_parents, local_failed, rollout_metadata = puct_update_result
                cross_batch_info = {}
            
            # Log cross-batch contamination prevention (strict mode)
            if cross_batch_info.get('mode') == 'strict' and cross_batch_info.get('n_delayed', 0) > 0:
                logger.info(f"[Step {global_step}] STRICT_PUCT: Prevented cross-batch contamination. "
                           f"This batch: {cross_batch_info['n_this_batch']}, "
                           f"Delayed: {cross_batch_info['n_delayed']}")
            
            # Calculate staleness statistics from metadata
            if rollout_metadata:
                staleness_values = [m['staleness'] for m in rollout_metadata]
                exec_times = [m.get('exec_time_ms', 0) for m in rollout_metadata if m.get('exec_time_ms', 0) > 0]
                
                avg_staleness = sum(staleness_values) / len(staleness_values) if staleness_values else 0
                max_staleness = max(staleness_values) if staleness_values else 0
                min_staleness = min(staleness_values) if staleness_values else 0
                
                if is_dp_head:
                    logger.info(f"[Step {global_step}] STALENESS_BATCH: n={len(staleness_values)}, "
                               f"avg={avg_staleness:.2f}, min={min_staleness}, max={max_staleness}")
                    if exec_times:
                        avg_exec_time = sum(exec_times) / len(exec_times)
                        logger.info(f"[Step {global_step}] EXEC_TIME: n={len(exec_times)}, avg={avg_exec_time:.2f}ms")
            
            # Async monitoring: track expected vs actual rollouts for research analysis
            expected_per_rank = (batch_size * group_size) // self.actor.data_parallel_world_size
            actual_children = len(local_children)
            actual_failed = len(local_failed)
            actual_total = actual_children + actual_failed
            async_overhead = actual_total - expected_per_rank
            
            # Record async metrics for history_logger analysis
            async_metrics = {
                'expected_rollouts': expected_per_rank,
                'actual_children': actual_children,
                'actual_failed': actual_failed,
                'actual_total': actual_total,
                'async_overhead': async_overhead,
            }
            
            # Add Scheme 1 wait time if applicable
            if use_scheme_1 and scheme_1_wait_time > 0:
                async_metrics['scheme_1_wait_time_s'] = scheme_1_wait_time
            
            # Add staleness statistics if available
            if rollout_metadata:
                staleness_values = [m['staleness'] for m in rollout_metadata]
                exec_times = [m.get('exec_time_ms', 0) for m in rollout_metadata if m.get('exec_time_ms', 0) > 0]
                async_metrics['staleness'] = {
                    'n': len(staleness_values),
                    'avg': sum(staleness_values) / len(staleness_values) if staleness_values else 0,
                    'min': min(staleness_values) if staleness_values else 0,
                    'max': max(staleness_values) if staleness_values else 0,
                }
                if exec_times:
                    async_metrics['exec_time_ms'] = {
                        'n': len(exec_times),
                        'avg': sum(exec_times) / len(exec_times),
                    }
            
            if async_overhead != 0:
                logger.info(f"[Async Monitor][Step {global_step}][Rank {self.actor.dp_rank}] "
                           f"expected={expected_per_rank}, actual={actual_total} "
                           f"(children={actual_children}, failed={actual_failed}), "
                           f"overhead={async_overhead:+d}")
            
            # Calculate unique parents and completion rate
            unique_parents = set(p.id for p in local_parents) if local_parents else set()
            n_unique_parents = len(unique_parents)
            avg_children_per_parent = len(local_children) / n_unique_parents if n_unique_parents > 0 else 0
            
            logger.info(f"[Step {global_step}][Rank {self.actor.dp_rank}] Local updates: "
                       f"children={len(local_children)}, parents={len(local_parents)} "
                       f"unique_parents={n_unique_parents}, failed={len(local_failed)} "
                       f"avg_children_per_parent={avg_children_per_parent:.2f}")
            
            # Detailed parent-child mapping for analysis (regex-friendly)
            if local_parents and local_children:
                parent_child_counts = {}
                for parent in local_parents:
                    pid = parent.id
                    parent_child_counts[pid] = parent_child_counts.get(pid, 0) + 1
                
                # Log distribution of children per parent
                for pid, count in sorted(parent_child_counts.items())[:10]:  # Limit to first 10 to avoid log spam
                    logger.info(f"[PARENT_CHILD_MAP] step={global_step} rank={self.actor.dp_rank} "
                               f"parent_id={pid[:8]}... children_count={count}")
            
            # Pre-sync: record _m values before update (for Q-value change analysis)
            prev_m_values = {}
            if hasattr(workflow, 'log_puct_update') and hasattr(self.sampler, '_m'):
                for parent in local_parents:
                    pid = parent.id
                    if pid in self.sampler._m:
                        prev_m_values[pid] = self.sampler._m[pid]
            
            self.actor.sync_sampler(
                local_children=local_children,
                local_parents=local_parents,
                local_failed=local_failed,
                step=global_step
            )
            
            # Record PUCT update events for three-core-metrics analysis
            # This captures _m and _n values after sync_sampler updates them
            if hasattr(workflow, 'log_puct_update') and hasattr(self.sampler, '_m'):
                for parent in local_parents:
                    pid = parent.id
                    if pid in self.sampler._m:
                        m_value = self.sampler._m[pid]
                        n_visits = self.sampler._n.get(pid, 0)
                        
                        # Get children rewards for this parent from local_children
                        children_rewards = [
                            c.value for c in local_children
                            if any(p.get('id') == pid for p in (c.parents or []))
                            and c.value is not None
                        ]
                        
                        # Calculate score (simplified, without scale/prior)
                        score = m_value  # Q-value is the main component
                        
                        try:
                            workflow.log_puct_update(
                                parent_id=pid,
                                update_step=global_step,
                                n_visits=n_visits,
                                m_value=m_value,
                                score=score,
                                children_rewards=children_rewards,
                                prev_m_value=prev_m_values.get(pid),  # Record change
                            )
                        except Exception as e:
                            logger.debug(f"[PUCT_TRACK] Failed to log update for {pid[:8]}: {e}")
            
            # Post-sync verification (lightweight checksum)
            if dist.is_initialized() and self.actor.data_parallel_world_size > 1:
                import hashlib
                if hasattr(self.sampler, '_states'):
                    state_ids = sorted([s.id for s in self.sampler._states])
                    hash_input = ','.join(state_ids).encode('utf-8')
                    state_checksum = int(hashlib.md5(hash_input).hexdigest(), 16) & 0xFFFFFFFF
                else:
                    state_checksum = 0
                
                all_checksums = [0] * self.actor.data_parallel_world_size
                dist.all_gather_object(all_checksums, state_checksum, group=self.actor.data_parallel_group)
                
                if len(set(all_checksums)) != 1:
                    logger.warning(f"[Step {global_step}] Cross-rank sampler state INCONSISTENT!")
            
            # === Training Phase (everything after rollout until next step) ===
            training_start = time.perf_counter()
            
            # Compute reward statistics
            local_rollouts = rollout_batch["rewards"].shape[0]
            step_rewards = rollout_batch["rewards"].cpu().numpy()
            step_max_reward = float(step_rewards.max())
            step_mean_reward = float(step_rewards.mean())
            
            # Global reward statistics (All-Reduce)
            if dist.is_initialized():
                local_max = torch.tensor([step_max_reward], dtype=torch.float32, device=self.actor.device)
                local_sum = torch.tensor([step_rewards.sum()], dtype=torch.float32, device=self.actor.device)
                local_count = torch.tensor([len(step_rewards)], dtype=torch.float32, device=self.actor.device)
                
                dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
                dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
                
                step_max_reward = local_max.item()
                step_mean_reward = (local_sum / local_count).item() if local_count.item() > 0 else 0.0
                global_rollouts = int(local_count.item())
            else:
                global_rollouts = local_rollouts
            
            best_reward = max(best_reward, step_max_reward)
            
            # === Actor policy update token counting for MFU calculation ===
            # loss_mask indicates which tokens actually participate in policy gradient computation
            # Note: compute_advantages() rolls loss_mask by -1, but sum remains the same
            actor_update_tokens = 0
            if "loss_mask" in rollout_batch:
                actor_update_tokens = int(rollout_batch["loss_mask"].sum().item())
            elif "attention_mask" in rollout_batch:
                actor_update_tokens = int(rollout_batch["attention_mask"].sum().item())
            elif "input_ids" in rollout_batch:
                actor_update_tokens = rollout_batch["input_ids"].numel()
            
            if dist.is_initialized():
                local_actor_tokens = torch.tensor([actor_update_tokens], dtype=torch.long, device=self.actor.device)
                dist.all_reduce(local_actor_tokens, op=dist.ReduceOp.SUM)
                global_actor_update_tokens = int(local_actor_tokens.item())
            else:
                global_actor_update_tokens = actor_update_tokens
            
            # Record Training History
            local_step_rewards = step_rewards.tolist()
            current_best_solution = None
            if is_dp_head and hasattr(self.sampler, 'get_best_solution'):
                try:
                    current_best_solution = self.sampler.get_best_solution()
                except Exception as e:
                    logger.warning(f"Failed to get best solution from sampler: {e}")
            
            # Step metrics (without timing, will be added later)
            step_metrics = {
                'batch_parents': batch_size,
                'group_size': group_size,
                'local_rollouts': local_rollouts,
                'global_rollouts': global_rollouts,
                'step_max_reward': step_max_reward,
                'step_mean_reward': step_mean_reward,
                'rollout_total_tokens': global_rollout_tokens,
                'actor_update_tokens': global_actor_update_tokens,
                'local_rollout_total_tokens': rollout_total_tokens,
                'local_actor_update_tokens': actor_update_tokens,
            }
            
            # NOTE: Staleness metrics are now collected from rollout_metadata in async_metrics
            # The _staleness_tracker is cleared by get_pending_updates(), so we can't use it here
            # Staleness data will be in step_metrics['async']['staleness'] after distributed sync
            
            # Add execution time stats for research analysis
            if hasattr(workflow, 'get_exec_time_stats'):
                exec_time_stats = workflow.get_exec_time_stats()
                if exec_time_stats:
                    # Aggregate exec time stats across all parents
                    all_exec_times = []
                    parent_exec_stats = []
                    for pid, stats in exec_time_stats.items():
                        all_exec_times.extend(stats['exec_times_ms'])
                        parent_exec_stats.append({
                            'parent_id': pid[:8],  # Short ID for readability
                            'avg_ms': stats['avg_exec_time_ms'],
                            'total_ms': stats['total_exec_time_ms'],
                            'n_children': stats['n_children'],
                            'parent_timestep': stats['parent_timestep'],
                        })
                    
                    step_metrics['exec_times'] = {
                        'n_parents': len(exec_time_stats),
                        'total_rollouts': len(all_exec_times),
                        'avg_ms': sum(all_exec_times) / len(all_exec_times) if all_exec_times else 0,
                        'min_ms': min(all_exec_times) if all_exec_times else 0,
                        'max_ms': max(all_exec_times) if all_exec_times else 0,
                        'parent_details': parent_exec_stats[:10],  # Limit to first 10 for brevity
                    }
                    
                    # Log for easy regex extraction
                    logger.info(f"[EXEC_TIME_STATS] step={global_step} "
                               f"n_parents={len(exec_time_stats)} "
                               f"total_rollouts={len(all_exec_times)} "
                               f"avg_ms={step_metrics['exec_times']['avg_ms']:.2f} "
                               f"min_ms={step_metrics['exec_times']['min_ms']:.2f} "
                               f"max_ms={step_metrics['exec_times']['max_ms']:.2f}")
            
            logger.info(f"[Rank {self.actor.dp_rank}][Step {global_step}] Rollouts: {local_rollouts} (global: {global_rollouts}), "
                       f"batch parents: {batch_size}, group_size: {group_size}")
            
            # Training computations
            metrics = {
                "rollout/count": global_rollouts,
                "reward/max": step_max_reward,
                "reward/mean": step_mean_reward,
                "reward/best_overall": best_reward,
            }
            
            # FIX: Use config.should_compute_prox_logp() instead of config.actor.should_compute_prox_logp()
            # because the actor config is nested and doesn't inherit top-level settings like
            # use_decoupled_loss and recompute_logprob
            if config.should_compute_prox_logp():
                rollout_batch["prox_logp"] = self.actor.compute_logp(rollout_batch)
                logger.info(f"[Rank {self.actor.dp_rank}][Step {global_step}] compute_logp done, prox_logp shape: {rollout_batch['prox_logp'].shape}, dtype: {rollout_batch['prox_logp'].dtype}")
            else:
                logger.info(f"[Rank {self.actor.dp_rank}][Step {global_step}] should_compute_prox_logp=False, prox_logp not computed")
            
            if self.ref is not None:
                rollout_batch["ref_logp"] = self.ref.compute_logp(rollout_batch)
            
            # Use rollout_batch directly (compute_advantages modifies in-place)
            # This matches train_tttd_vllm_v2.py behavior
            self.actor.compute_advantages(rollout_batch)
            
            # DEBUG: Check prox_logp status before ppo_update
            bs, max_seqlen = rollout_batch['input_ids'].shape
            if 'prox_logp' in rollout_batch and rollout_batch['prox_logp'] is not None:
                prox_logp = rollout_batch['prox_logp']
                logger.info(f"[Rank {self.actor.dp_rank}][Step {global_step}] Before ppo_update: prox_logp shape={prox_logp.shape}, numel={prox_logp.numel()}, expected={bs * max_seqlen}")
            else:
                logger.warning(f"[Rank {self.actor.dp_rank}][Step {global_step}] Before ppo_update: prox_logp is None or missing!")
            
            # Add advantage statistics
            if "advantages" in rollout_batch:
                adv = rollout_batch["advantages"].cpu().numpy()
                metrics.update({
                    "advantage/mean": float(adv.mean()),
                    "advantage/std": float(adv.std()),
                    "advantage/min": float(adv.min()),
                    "advantage/max": float(adv.max()),
                })
            
            # PPO update
            self.actor.ppo_update(rollout_batch)
            self.actor.step_lr_scheduler()
            
            # Add training stats
            stats = self.actor.export_stats()
            metrics.update({
                "train/entropy": stats.get('ppo_actor/update/entropy/avg', 0.0),
                "train/actor_loss": stats.get('ppo_actor/update/actor_loss/avg', 0.0),
                "train/approx_kl": stats.get('ppo_actor/update/approx_kl/avg', 0.0),
                "train/grad_norm": stats.get('ppo_actor/update/grad_norm', 0.0),
                "train/lr": stats.get('ppo_actor/update/lr', 0.0),
                "train/importance_weight": stats.get('ppo_actor/update/importance_weight/avg', 0.0),
                "train/clip_ratio": stats.get('ppo_actor/update/clip_ratio/avg', 0.0),
                "train/advantages/avg": stats.get('ppo_actor/advantages/avg', 0.0),
                "train/advantages/max": stats.get('ppo_actor/advantages/max', 0.0),
                "train/advantages/min": stats.get('ppo_actor/advantages/min', 0.0),
                "rollout/total_tokens": global_rollout_tokens,
                "train/actor_update_tokens": global_actor_update_tokens,
            })
            
            # Log to stats_logger
            if is_dp_head:
                self.stats_logger.commit(
                    epoch=epoch,
                    step=step,
                    global_step=global_step,
                    data=metrics,
                )
                logger.info(
                    f"[Step {global_step}] Reward: max={step_max_reward:.4f}, mean={step_mean_reward:.4f}, best={best_reward:.4f} | "
                    f"Loss: {metrics['train/actor_loss']:.4f}, KL: {metrics['train/approx_kl']:.4f}, "
                    f"Entropy: {metrics['train/entropy']:.4f}, GradNorm: {metrics['train/grad_norm']:.4f}, LR: {metrics['train/lr']:.6f} | "
                    f"Tokens: rollout={global_rollout_tokens}, actor_update={global_actor_update_tokens}"
                )
            
            # Update weights and save (all part of training phase)
            logger.info(f"[DEBUG][Step {global_step}] Before rollout.pause")
            self.rollout.pause()
            logger.info(f"[DEBUG][Step {global_step}] After rollout.pause")
            self.actor.update_weights(self.weight_update_meta)
            self.actor.set_version(global_step + 1)
            self.rollout.set_version(global_step + 1)
            
            logger.info(f"[DEBUG][Step {global_step}] Before _save_hf")
            self._save_hf(epoch=epoch, epoch_step=step, global_step=global_step)
            logger.info(f"[DEBUG][Step {global_step}] After _save_hf")
            logger.info(f"[DEBUG][Step {global_step}] Before _save_recover_checkpoint")
            self._save_recover_checkpoint(epoch=epoch, epoch_step=step, global_step=global_step)
            logger.info(f"[DEBUG][Step {global_step}] After _save_recover_checkpoint")
            
            logger.info(f"[DEBUG][Step {global_step}] Before dist.barrier")
            dist.barrier(group=self.actor.cpu_group)
            logger.info(f"[DEBUG][Step {global_step}] After dist.barrier")
            current_platform.synchronize()
            logger.info(f"[DEBUG][Step {global_step}] After platform.synchronize")
            self.rollout.resume()
            logger.info(f"[DEBUG][Step {global_step}] After rollout.resume")
            
            training_time = time.perf_counter() - training_start
            step_total = time.perf_counter() - step_start_time
            
            # === Simplified Timing Output ===
            # Add timing to step_metrics so it's saved in history_logger
            step_metrics['timing'] = {
                'rollout': rollout_time,
                'execute_tail': execute_tail,
                'training': training_time,
                'total': step_total,
            }
            
            # Add detailed timing stats if available
            if timing_stats:
                step_metrics['timing_stats'] = timing_stats
            
            # Add async metrics for research analysis
            step_metrics['async'] = async_metrics
            
            # Get PUCT analysis data for three-core-metrics analysis
            puct_analysis_data = None
            if hasattr(workflow, 'get_puct_analysis_data'):
                try:
                    puct_analysis_data = workflow.get_puct_analysis_data(clear=False)
                except Exception as e:
                    logger.warning(f"[PUCT_TRACK] Failed to get analysis data: {e}")
            
            # All-gather rewards across ranks for complete distribution (lightweight)
            # Metadata remains per-rank to avoid all_gather_object timeout
            if dist.is_initialized() and self.actor.data_parallel_world_size > 1:
                world_size = self.actor.data_parallel_world_size
                all_rewards_list = [None] * world_size
                dist.all_gather_object(all_rewards_list, local_step_rewards)
                # Only rank0 uses aggregated rewards for complete distribution
                if is_dp_head:
                    aggregated_rewards = []
                    for r in all_rewards_list:
                        if r is not None:
                            aggregated_rewards.extend(r)
                    rewards_to_record = aggregated_rewards
                else:
                    rewards_to_record = local_step_rewards
            else:
                rewards_to_record = local_step_rewards
            
            # Each rank records its own data:
            # - rank0: aggregated rewards (complete distribution) + local metadata
            # - others: local rewards + local metadata
            self.history_logger.record_step(
                step=global_step,
                rewards=rewards_to_record,
                best_solution=current_best_solution,
                additional_metrics=step_metrics,
                rollout_metadata=rollout_metadata if rollout_metadata else None,
                puct_analysis_data=puct_analysis_data,
            )
            
            if is_dp_head:
                # Log detailed timing breakdown if available
                if timing_stats:
                    logger.info(
                        f"[TIMING][Step {global_step}] "
                        f"rollout={rollout_time:.2f}s | "
                        f"exec_tail={execute_tail:.2f}s | "
                        f"training={training_time:.2f}s | "
                        f"total={step_total:.2f}s | "
                        f"reward={step_max_reward:.4f} | "
                        f"n={timing_stats.get('n_rollouts', 0)}"
                    )
                else:
                    logger.info(
                        f"[TIMING][Step {global_step}] "
                        f"rollout={rollout_time:.2f}s | "
                        f"exec_tail={execute_tail:.2f}s | "
                        f"training={training_time:.2f}s | "
                        f"total={step_total:.2f}s | "
                        f"reward={step_max_reward:.4f}"
                    )
            
            # Save Training History Checkpoint AFTER recording step data
            # This ensures the current step is included in the checkpoint
            logger.info(f"[DEBUG][Step {global_step}] Before save_checkpoint")
            checkpoint_path = self.history_logger.save_checkpoint()
            if checkpoint_path:
                logger.debug(f"[TTTLogger] Saved checkpoint to {checkpoint_path}")
            logger.info(f"[DEBUG][Step {global_step}] After save_checkpoint")
            
            # DEBUG: End of loop iteration
            logger.info(f"[DEBUG] Loop #{loop_count} (global_step={global_step}) completed successfully")
        
        # DEBUG: Loop ended
        logger.info(f"[DEBUG] Loop ended after {loop_count} iterations. global_step={global_step if 'global_step' in locals() else 'N/A'}")
        
        # === Save Final Training History ===
        # Each rank saves its own data (files are suffixed with rank id)
        history_path = self.history_logger.save(also_save_json=True)
        if history_path:
            logger.info(f"[TTTLogger][Rank {self.actor.dp_rank}] Final training history saved to {history_path}")
            
            summary = self.history_logger.get_summary()
            logger.info(
                f"[TTTLogger][Rank {self.actor.dp_rank}] Training Summary:\n"
                f"  - Recorded snapshots: {summary['num_snapshots']}\n"
                f"  - Recorded steps: {summary['recorded_steps']}\n"
                f"  - Overall best reward: {summary['overall_best_reward']:.4f} (step {summary['overall_best_step']})\n"
                f"  - Has Best-of-N baseline: {summary['has_best_of_n']}"
            )
            
            # Only rank 0 shows the visualization command (to avoid log spam)
            if self.actor.dp_rank == 0:
                logger.info(
                    f"\n[TTTLogger] To generate visualization, run:\n"
                    f"  python areal/experimental/ttt_discover/generate_plot.py \\\n"
                    f"    --history_path {history_path} \\\n"
                    f"    --benchmark_value 2.635983\n"
                    f"\n[TTTLogger] Note: For multi-rank analysis, process each rank's file separately:\n"
                    f"  for f in {self.config.saver.fileroot}/training_history_rank*.pkl; do\n"
                    f"    python areal/experimental/ttt_discover/generate_plot.py \\\n"
                    f"      --history_path $f --benchmark_value 2.635983\n"
                    f"  done"
                )

    def _save_hf(self, epoch: int, epoch_step: int, global_step: int):
        """
        Override parent _save_hf to remove extra barrier.
        
        FIXME: Workaround for freq_ctl.check() inconsistency across ranks.
        """
        self.saver.save(
            self.actor,
            epoch,
            epoch_step,
            global_step,
            tokenizer=self.tokenizer,
            processor=getattr(self, 'processor', None),
        )
        # NOTE: No extra barrier - saver.save() already has FSDP sync.

    def _save_recover_checkpoint(self, epoch: int, epoch_step: int, global_step: int):
        """
        Override parent _save_recover_checkpoint to remove extra barrier.
        
        FIXME: Workaround for freq_ctl.check() inconsistency in recover_handler.dump().
        """
        from areal.api.io_struct import StepInfo
        
        to_save = dict(default=self.actor)
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=epoch_step,
            steps_per_epoch=1,  # TTT-Discover: each step is effectively an epoch
        )
        
        # Directly dump without freq_ctl check to avoid sync issues
        self.recover_handler._save_checkpoint(
            self.actor,
            name="default",
            tokenizer=self.tokenizer,
            processor=getattr(self, 'processor', None),
        )
        
        # Update last_step_info
        self.recover_handler.last_step_info = step_info
        
        # Save recover info metadata
        from areal.utils.recover import RecoverInfo
        recover_info = RecoverInfo(
            last_step_info=step_info,
            saver_info=self.saver.state_dict(),
            evaluator_info=self.evaluator.state_dict(),
            stats_logger_info=self.stats_logger.state_dict(),
            dataloader_info=self.train_dataloader.state_dict(),
            checkpoint_info=self.recover_handler.freq_ctl.state_dict(),
        )
        recover_info_path = self.recover_handler.recover_info_path(
            self.config.experiment_name,
            self.config.trial_name,
            self.config.recover.fileroot,
        )
        recover_info.dump(recover_info_path)
        # NOTE: No extra barrier - _save_checkpoint already handles sync.

    def close(self):
        """Cleanup resources. Overrides parent to handle missing eval_rollout."""
        self.stats_logger.close()
        # TTT-Discover doesn't use eval_rollout
        if hasattr(self, 'rollout') and self.rollout is not None:
            self.rollout.destroy()
        if hasattr(self, 'ref') and self.ref is not None:
            self.ref.destroy()
        if hasattr(self, 'actor') and self.actor is not None:
            self.actor.destroy()
        from areal.utils import perf_tracer
        perf_tracer.save(force=True)


def main(args):
    """Main training function."""
    import json
    import os
    from areal.utils import logging
    
    config, _ = load_expr_config(args, TTTDPPOActorConfig)
    
    # Ensure stop tokens are set
    if config.tokenizer_path:
        from areal.utils.hf_utils import load_hf_tokenizer
        tokenizer = load_hf_tokenizer(config.tokenizer_path)
        if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
            config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
        if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
            config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    
    # ============================================================
    # Verify LoRA adapter exists (same as train_tttd_vllm_v2.py)
    # ============================================================
    if config.use_lora and not config.skip_lora_check:
        lora_output_path = "./lora_init"
        # Support both dict (legacy) and vLLMConfig dataclass
        if hasattr(config, 'vllm'):
            if isinstance(config.vllm, dict):
                lora_modules_str = config.vllm.get('lora_modules', '')
            else:
                # vLLMConfig dataclass
                lora_modules_str = getattr(config.vllm, 'lora_modules', '') or ''
            if lora_modules_str:
                try:
                    lora_modules = json.loads(lora_modules_str)
                    if isinstance(lora_modules, dict):
                        lora_output_path = lora_modules.get('path', lora_output_path)
                except json.JSONDecodeError:
                    pass
        
        lora_output_path = os.path.abspath(lora_output_path)
        adapter_config_path = os.path.join(lora_output_path, "adapter_config.json")
        
        if not os.path.exists(adapter_config_path):
            error_msg = (
                f"\n{'='*80}\n"
                f"ERROR: Initial LoRA adapter not found at {lora_output_path}\n"
                f"{'='*80}\n\n"
                f"The LoRA adapter must be created BEFORE starting training because\n"
                f"vLLM loads it at startup (before this training script runs).\n\n"
                f"To fix this, run the preparation script first:\n"
                f"  python areal/experimental/ttt_discover/examples/prepare_lora_init.py \\\n"
                f"    --config-path <your_config.yaml>\n\n"
                f"If you have already initialized the LoRA adapter elsewhere, you can\n"
                f"skip this check by adding '+skip_lora_check=true' to your command.\n"
                f"{'='*80}\n"
            )
            raise RuntimeError(error_msg)
        
        logger.info(f"[LoRA Check] ✓ LoRA adapter verified at {lora_output_path}")
    
    # Create environment
    env = create_env_from_config(config)
    
    # DEBUG: Check lazy_puct_sampling value
    logger.info(f"[DEBUG] config.sampler.lazy_puct_sampling = {config.sampler.lazy_puct_sampling}")
    
    # Calculate local batch size (per-rank)
    # NOTE: batch_size here is LOCAL batch_size (per-rank), not global
    # Parse allocation mode to get train world size
    from areal.api.alloc_mode import _AllocationMode as AllocationMode
    alloc_mode = AllocationMode.from_str(config.allocation_mode)
    train_world_size = alloc_mode.train.world_size
    local_batch_size = config.sampler.batch_size // train_world_size
    group_size = config.gconfig.n_samples
    
    # Workflow kwargs - will be used to create workflow instance in the engine
    # This ensures lazy sampling parameters are properly applied
    workflow_kwargs = dict(
        env=env,
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=config.enable_thinking,
        max_prompt_thinking_tokens=config.max_prompt_thinking_tokens,
        max_reward_workers=64,
        batch_size=local_batch_size,  # Local batch_size per rank
        group_size=group_size,  # Number of rollouts per parent
        lazy_sampling=config.sampler.lazy_puct_sampling,
        vllm_concurrency=config.sampler.vllm_concurrency,
        execution_concurrency=config.sampler.execution_concurrency,
    )
    
    # Run training
    # Pass the class (not instance) so workflow_kwargs are used to create workflow
    with TTTDPPOTrainer(config) as trainer:
        trainer.train(
            workflow=TTTDiscoverWorkflowV2,  # Pass class, not instance
            workflow_kwargs=workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
