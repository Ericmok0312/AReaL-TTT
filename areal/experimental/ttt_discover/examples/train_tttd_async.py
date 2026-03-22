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
from areal.api.alloc_mode import AllocationMode
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
        self.sampler = create_sampler_from_config(
            config=config.sampler,
            log_path=config.saver.fileroot,
            env_type=getattr(config.sampler, 'env_type', 'ac1'),
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
        
        # Create dataloaders using sampler (before engine init, only needs process group)
        self.train_dataloader = self._create_tttd_dataloader(
            sampler=self.sampler,
            rank=self.actor.data_parallel_rank,
            world_size=self.actor.data_parallel_world_size,
            batch_size=config.sampler.batch_size,
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
    ) -> StatefulDataLoader:
        """Create TTT-Discover dataloader with sampler."""
        return create_tttd_dataloader(
            state_sampler=sampler,
            rank=rank,
            world_size=world_size,
            batch_size=batch_size,
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
        is_dp_head = self.actor.is_data_parallel_head() if hasattr(self.actor, 'is_data_parallel_head') else (int(__import__('os').getenv("RANK", "0")) == 0)
        
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
            
            if is_dp_head:
                logger.info(f"[Resume] Ready to start from step {start_step}")
    
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
        
        is_dp_head = self.actor.is_data_parallel_head() if hasattr(self.actor, 'is_data_parallel_head') else (int(__import__('os').getenv("RANK", "0")) == 0)
        
        self.history_logger = TTTTrainingLogger(
            save_steps=future_save_steps,
            output_dir=config.saver.fileroot,
            is_dp_head=is_dp_head,
            filename='training_history.pkl',
            checkpoint_filename='training_history_checkpoint.pkl',
            aggregate_distributed=True,
        )
        
        if is_dp_head:
            existing_snapshots = len(self.history_logger.history) if self.recover_info else 0
            logger.info(f"[TTTLogger] Initialized for step {start_step} to {max_steps}")
            logger.info(f"[TTTLogger] Future save_steps: {future_save_steps}")
            logger.info(f"[TTTLogger] Existing snapshots from checkpoint: {existing_snapshots}")
            logger.info(f"[TTTLogger] Output directory: {config.saver.fileroot}")
    
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
        
        start_step = (
            self.recover_info.last_step_info.next().global_step
            if self.recover_info is not None
            else 0
        )
        
        # TTT-Discover: Use max_steps directly (not steps_per_epoch * epochs)
        max_steps = getattr(config, 'max_steps', config.total_train_epochs)
        
        is_dp_head = self.actor.is_data_parallel_head() if hasattr(self.actor, 'is_data_parallel_head') else (int(__import__('os').getenv("RANK", "0")) == 0)
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
            
            # Get execute tail latency from workflow if available
            # Tail latency = time from last GPU done to last execute done
            execute_tail = 0.0
            timing_stats = {}
            if hasattr(workflow, 'get_execute_tail_latency'):
                execute_tail = workflow.get_execute_tail_latency(clear=True)
                if hasattr(workflow, 'get_timing_stats'):
                    timing_stats = workflow.get_timing_stats(clear=False)  # Don't clear, already cleared above
            
            # === Sampler Synchronization (part of training phase) ===
            local_children, local_parents, local_failed = workflow.get_pending_updates(clear=True)
            
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
            
            if async_overhead != 0:
                logger.info(f"[Async Monitor][Step {global_step}][Rank {self.actor.dp_rank}] "
                           f"expected={expected_per_rank}, actual={actual_total} "
                           f"(children={actual_children}, failed={actual_failed}), "
                           f"overhead={async_overhead:+d}")
            
            logger.info(f"[Step {global_step}][Rank {self.actor.dp_rank}] Local updates: "
                       f"children={len(local_children)}, parents={len(local_parents)}, failed={len(local_failed)}")
            
            self.actor.sync_sampler(
                local_children=local_children,
                local_parents=local_parents,
                local_failed=local_failed,
                step=global_step
            )
            
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
            }
            
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
                    f"Entropy: {metrics['train/entropy']:.4f}, GradNorm: {metrics['train/grad_norm']:.4f}, LR: {metrics['train/lr']:.6f}"
                )
            
            logger.info(f"[DEBUG][Step {global_step}] Before save_checkpoint, is_dp_head={is_dp_head}")
            # Save Training History Checkpoint
            if is_dp_head:
                checkpoint_path = self.history_logger.save_checkpoint()
                if checkpoint_path:
                    logger.debug(f"[TTTLogger] Saved checkpoint to {checkpoint_path}")
            logger.info(f"[DEBUG][Step {global_step}] After save_checkpoint")
            
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
            
            # Record to history logger (now includes timing)
            self.history_logger.record_step(
                step=global_step,
                rewards=local_step_rewards,
                best_solution=current_best_solution,
                additional_metrics=step_metrics
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
            
            # DEBUG: End of loop iteration
            logger.info(f"[DEBUG] Loop #{loop_count} (global_step={global_step}) completed successfully")
        
        # DEBUG: Loop ended
        logger.info(f"[DEBUG] Loop ended after {loop_count} iterations. global_step={global_step if 'global_step' in locals() else 'N/A'}")
        
        # === Save Final Training History ===
        if is_dp_head:
            history_path = self.history_logger.save(also_save_json=True)
            if history_path:
                logger.info(f"[TTTLogger] Final training history saved to {history_path}")
                
                summary = self.history_logger.get_summary()
                logger.info(
                    f"[TTTLogger] Training Summary:\n"
                    f"  - Recorded snapshots: {summary['num_snapshots']}\n"
                    f"  - Recorded steps: {summary['recorded_steps']}\n"
                    f"  - Overall best reward: {summary['overall_best_reward']:.4f} (step {summary['overall_best_step']})\n"
                    f"  - Has Best-of-N baseline: {summary['has_best_of_n']}"
                )
                
                logger.info(
                    f"\n[TTTLogger] To generate visualization, run:\n"
                    f"  python areal/experimental/ttt_discover/generate_plot.py \\\n"
                    f"    --history_path {history_path} \\\n"
                    f"    --benchmark_value 2.635983"
                )

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
    
    # Create workflow
    workflow = TTTDiscoverWorkflowV2(
        env=env,
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=config.enable_thinking,
        max_prompt_thinking_tokens=config.max_prompt_thinking_tokens,
        max_reward_workers=64
    )
    
    # Workflow kwargs
    workflow_kwargs = dict(
        env=env,
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=config.enable_thinking,
        max_prompt_thinking_tokens=config.max_prompt_thinking_tokens,
    )
    
    # Run training
    with TTTDPPOTrainer(config) as trainer:
        trainer.train(
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
