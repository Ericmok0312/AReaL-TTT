#!/usr/bin/env python3
"""
TTT-Discover Distillation Training Script.

This script distills a teacher model's policy into a student model with LoRA:
1. Load teacher model checkpoint from a given path
2. Load teacher PUCTSampler state from another given path
3. Student is a new model with LoRA
4. Training: sample from teacher's PUCTSampler, student rollout (no verification)
5. Reward = negative KL divergence between student and teacher
6. Run N distill steps and save student checkpoint

Usage:
    python train_tttd_distill.py --config conf/distill_lora_vllm_cp_qwen3_8b.yaml
"""

import copy
import json
import os
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal import PPOTrainer
from areal.api.alloc_mode import ModelAllocation
from areal.api.alloc_mode import _AllocationMode as AllocationMode
from areal.api.cli_args import (
    load_expr_config,
)
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.experimental.ttt_discover.actor import TTTDActor

# Native AReaL KDRL uses teacher_logp in ppo_update; no manual KL estimator needed.
from areal.experimental.ttt_discover.config import (
    TTTDDistillConfig,
    TTTDPPOActorConfig,
    create_env_from_config,
)
from areal.experimental.ttt_discover.dataloader import create_tttd_dataloader
from areal.experimental.ttt_discover.envs.env import EnvResult
from areal.experimental.ttt_discover.sampler import (
    _find_latest_sampler_step,
    create_sampler_from_config,
)
from areal.experimental.ttt_discover.ttt_logger import TTTTrainingLogger
from areal.experimental.ttt_discover.workflow_v2 import TTTDiscoverWorkflowV2
from areal.infra import current_platform
from areal.utils import logging, seeding, stats_tracker
from areal.utils.environ import is_single_controller
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger

logger = logging.getLogger("train_tttd_distill")


# =============================================================================
# Dummy reward function for distill steps (no verification)
# =============================================================================
def dummy_reward_fn(prompt, completions, prompt_ids, completion_ids, **data):
    """No-op reward function that skips verification.
    
    Returns reward=0.0 so that the real reward (negative KL) can be injected
    after rollout in the training loop.
    """
    result = EnvResult(reward=0.0, is_valid=True, observation="", metadata={})
    return 0.0, result, "", 0.0




# =============================================================================
# Distillation Trainer
# =============================================================================
class TTTDDistillTrainer(PPOTrainer):
    """Custom trainer for TTT-Discover distillation.
    
    Key differences from standard TTTDPPOTrainer:
    - Loads a frozen teacher model from checkpoint
    - Loads teacher PUCTSampler state
    - Student uses LoRA
    - Distill steps: no verification, KL divergence as reward
    - Eval step: verification enabled, records results
    """

    def __init__(self, config: TTTDDistillConfig):
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
        self.actor_alloc = ModelAllocation.from_str(config.actor.backend, name="actor")
        self.rollout_alloc = ModelAllocation.from_str(config.rollout.backend, name="rollout")
        self._amend_xccl_weight_update_envvar()

        # =====================================================================
        # Load teacher PUCTSampler from teacher checkpoint
        # =====================================================================
        # max_version_history controls how many historical PUCT snapshots the
        # sampler keeps. In async mode (max_head_offpolicyness > 0), old
        # rollouts may return after several training steps, so we need to keep
        # snapshots of past PUCT states. In sync mode (max_head_offpolicyness=0),
        # rollout and training are synchronized, so only 1 version is needed.
        max_head_offpolicyness = getattr(config.rollout, 'max_head_offpolicyness', 2)
        max_version_history = max_head_offpolicyness + 1

        # Detect sync mode and disable lazy sampling automatically
        self.is_sync_mode = (max_head_offpolicyness == 0)
        if self.is_sync_mode and config.sampler.lazy_puct_sampling:
            logger.info(
                "[Distill] Sync mode detected (max_head_offpolicyness=0). "
                "Disabling lazy_puct_sampling."
            )
            config.sampler.lazy_puct_sampling = False

        # Create sampler using student's config but teacher's checkpoint dir
        # We use the teacher's checkpoint directory so we can load its state
        teacher_sampler_config = copy.deepcopy(config.sampler)
        if config.teacher_sampler_checkpoint:
            teacher_sampler_config.checkpoint_dir = config.teacher_sampler_checkpoint

        self.sampler = create_sampler_from_config(
            config=teacher_sampler_config,
            env_type=getattr(config.sampler, 'env_type', 'ac1'),
            max_version_history=max_version_history,
        )

        # Load the latest teacher sampler state
        if config.teacher_sampler_checkpoint:
            latest_step = _find_latest_sampler_step(
                config.teacher_sampler_checkpoint,
                getattr(config.sampler, 'type', 'puct')
            )
            if latest_step is not None:
                logger.info(f"[TeacherSampler] Loading latest checkpoint at step {latest_step}")
                self.sampler._load(latest_step)
                # Reset the sampler's internal step counter to avoid confusion
                self.sampler._current_step = 0
            else:
                logger.warning(
                    f"[TeacherSampler] No checkpoint found in {config.teacher_sampler_checkpoint}. "
                    f"Using fresh sampler state."
                )

        logger.info(
            f"[TeacherSampler] Loaded {len(self.sampler._states)} states, "
            f"T={self.sampler._T}, n_entries={len(self.sampler._n)}, "
            f"m_entries={len(self.sampler._m)}"
        )

        # =====================================================================
        # Create student actor (with LoRA)
        # =====================================================================
        self.actor = self._create_tttd_actor(config.actor)
        self.ref = None  # No ref model needed (kl_ctl=0)

        # =====================================================================
        # Create teacher model (native AReaL path)
        # =====================================================================
        self.teacher = None
        if config.teacher is not None:
            teacher_alloc = ModelAllocation.from_str(config.teacher.backend, name="teacher")
            self.teacher = self._create_train_engine(config.teacher, teacher_alloc)

        # =====================================================================
        # Create dataloaders using teacher sampler
        # =====================================================================
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

        # Initialize models
        self._initialize_engines()

        # =====================================================================
        # Save evaluation checkpoints for base, teacher, and student
        # These are used in the final eval phase to compare three models
        # =====================================================================
        eval_ckpt_dir = os.path.join(config.saver.fileroot, "eval_checkpoints")
        os.makedirs(eval_ckpt_dir, exist_ok=True)

        from areal.api.io_struct import SaveLoadMeta

        # 1. Save base model (student's initial LoRA, before distillation)
        self.base_eval_path = os.path.join(eval_ckpt_dir, "base")
        os.makedirs(self.base_eval_path, exist_ok=True)
        base_meta = SaveLoadMeta(
            path=self.base_eval_path,
            weight_format="hf",
            with_optim=False,
            tokenizer=None,
            processor=None,
        )
        if is_single_controller() or self.actor.rank == 0:
            logger.info(f"[EvalSetup] Saving base model (initial LoRA) to {self.base_eval_path}")
        self.actor.save(base_meta)

        # 2. Save teacher model for eval
        self.teacher_eval_path = os.path.join(eval_ckpt_dir, "teacher")
        os.makedirs(self.teacher_eval_path, exist_ok=True)
        teacher_meta = SaveLoadMeta(
            path=self.teacher_eval_path,
            weight_format="hf",
            with_optim=False,
            tokenizer=None,
            processor=None,
        )
        if is_single_controller() or self.actor.rank == 0:
            logger.info(f"[EvalSetup] Saving teacher model to {self.teacher_eval_path}")
        self.teacher.save(teacher_meta)

        # 3. Student eval path (will be saved after training)
        self.student_eval_path = os.path.join(eval_ckpt_dir, "student")
        os.makedirs(self.student_eval_path, exist_ok=True)

        # Connect sampler to actor (for API compatibility, though we won't sync)
        self.actor.connect_sampler(self.sampler)

        # Initialize inference engines
        self.rollout = self._init_rollout(config.rollout, is_eval=False)

        # Setup weight update meta
        self._setup_weight_update_meta()

        # Setup evaluation, saver, recover, stats logger
        self._setup_utilities()

        # Initialize proxy workers flag
        self._proxy_started = False

        # Store workflow kwargs for later use
        self._workflow_kwargs = {}

        # Initialize logger for dynamic metrics (overlap ratio, advantage, entropy gap)
        self._init_dynamic_metrics_logger()

    def _init_dynamic_metrics_logger(self):
        """Initialize TTTTrainingLogger to persist dynamic metrics to JSON."""
        config = self.config
        max_steps = getattr(config, 'max_steps', config.total_train_epochs)

        # Default: save every step. Can override via config.dynamic_metric_save_steps
        all_save_steps = getattr(config, 'dynamic_metric_save_steps', list(range(0, max_steps)))
        start_step = 0
        if hasattr(self, 'recover_info') and self.recover_info is not None:
            start_step = self.recover_info.last_step_info.next().global_step

        future_save_steps = [s for s in all_save_steps if start_step <= s < max_steps]

        rank_suffix = f"_rank{self.actor.dp_rank}" if self.actor.data_parallel_world_size > 1 else ""
        output_dir = os.path.join(config.saver.fileroot, "dynamic_metrics")
        os.makedirs(output_dir, exist_ok=True)

        self.dynamic_metrics_logger = TTTTrainingLogger(
            save_steps=future_save_steps,
            output_dir=output_dir,
            is_dp_head=True,
            filename=f'dynamic_metrics{rank_suffix}.pkl',
            checkpoint_filename=f'dynamic_metrics_checkpoint{rank_suffix}.pkl',
            aggregate_distributed=False,
        )

        if self.actor.rank == 0:
            logger.info(
                f"[DynamicMetrics] Logger initialized: steps={len(future_save_steps)}, "
                f"output_dir={output_dir}"
            )

    def _create_tttd_actor(self, actor_config: TTTDPPOActorConfig):
        """Create student TTTDActor."""
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
        max_steps = self.config.max_steps
        ft_spec = FinetuneSpec(
            total_train_epochs=max_steps,
            dataset_size=max_steps * self.config.sampler.batch_size,
            train_batch_size=self.config.sampler.batch_size,
        )

        engine_init_kwargs = {
            "addr": None,
            "ft_spec": ft_spec,
            "alloc_mode": self.allocation_mode,
        }

        self.actor.initialize(**engine_init_kwargs, role="actor")
        if self.teacher is not None:
            self.teacher.initialize(**engine_init_kwargs, role="teacher")

    def _setup_weight_update_meta(self):
        """Setup weight update meta and connect to inference engine."""
        config = self.config

        if config.actor.weight_update_mode == "disk":
            disk_kwargs = {
                "experiment_name": config.experiment_name,
                "trial_name": config.trial_name,
                "file_root": config.cluster.fileroot,
                "name": "default",
                "clear_checkpoint_after_load": True,
            }
            if config.actor.use_lora:
                disk_kwargs.update({
                    "use_lora": config.actor.use_lora,
                    "lora_name": config.gconfig.lora_name,
                    "lora_int_id": 1,
                    "base_model_name": config.actor.path,
                })
            self.weight_update_meta = WeightUpdateMeta.from_disk(**disk_kwargs)
        elif config.actor.weight_update_mode == "xccl":
            if self.allocation_mode.train_backend == "megatron":
                self.weight_update_meta = WeightUpdateMeta.from_megatron_xccl(
                    self.allocation_mode
                )
            else:
                xccl_kwargs = {"gen_allocation": self.rollout_alloc}
                if config.actor.use_lora:
                    xccl_kwargs.update({
                        "use_lora": config.actor.use_lora,
                        "lora_name": config.gconfig.lora_name,
                        "lora_int_id": 1,
                        "base_model_name": config.actor.path,
                    })
                self.weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(**xccl_kwargs)
        else:
            raise ValueError(f"Invalid weight update mode: {config.actor.weight_update_mode}")

        self.actor.connect_engine(self.rollout, self.weight_update_meta)

    def _setup_utilities(self):
        """Setup evaluator, saver, recover handler, and stats logger."""
        config = self.config

        max_steps = config.max_steps
        ft_spec = FinetuneSpec(
            total_train_epochs=max_steps,
            dataset_size=max_steps * config.sampler.batch_size,
            train_batch_size=config.sampler.batch_size,
        )

        self.evaluator = Evaluator(config.evaluator, ft_spec)
        self.saver = Saver(config.saver, ft_spec)
        self.recover_handler = RecoverHandler(config.recover, ft_spec)
        self.stats_logger = StatsLogger(config, ft_spec)

        # Load recovery info
        self.recover_info = self.recover_handler.load(
            self.actor,
            self.saver,
            self.evaluator,
            self.stats_logger,
            self.train_dataloader,
            inference_engine=self.rollout,
            weight_update_meta=self.weight_update_meta,
        )

        self._config_perf_tracer()

    def _get_workflow_executor(self):
        """Get workflow executor from rollout."""
        if hasattr(self.rollout, '_engine') and hasattr(self.rollout._engine, 'workflow_executor'):
            return self.rollout._engine.workflow_executor
        if hasattr(self.rollout, 'workflow_executor'):
            return self.rollout.workflow_executor
        return None

    def _clear_workflow_cache(self):
        """Clear workflow executor cache so a new workflow can be used."""
        workflow_executor = self._get_workflow_executor()
        if workflow_executor is not None and hasattr(workflow_executor, 'data_generator'):
            delattr(workflow_executor, 'data_generator')
            logger.info("[Distill] Cleared workflow executor cache")

        if hasattr(self.rollout, 'data_generator'):
            delattr(self.rollout, 'data_generator')
            logger.info("[Distill] Cleared rollout controller data_generator")

    def _normalize_rollout_batch(self, rollout_batch) -> dict[str, Any]:
        """Ensure rollout_batch is a single dict for TTTDActor methods.

        prepare_batch may return list[dict] depending on the backend path;
        concat if necessary so that TTTDActor.compute_advantages (dict input)
        and the KDRL path below work uniformly.
        """
        if isinstance(rollout_batch, dict):
            return rollout_batch
        if isinstance(rollout_batch, list):
            if len(rollout_batch) == 1:
                return rollout_batch[0]
            from areal.utils.data import concat_batch
            batched, _meta = concat_batch(rollout_batch)
            return batched
        raise TypeError(f"Unexpected rollout_batch type: {type(rollout_batch)}")

    def train(
        self,
        workflow,
        eval_workflow=None,
        workflow_kwargs: dict = None,
        eval_workflow_kwargs: dict = None,
    ):
        """Main distillation loop.
        
        Phase 1: Distill steps (no verification, KL reward)
        Phase 2: Eval step (verification, record results)
        """
        config = self.config

        # Update workflow kwargs with sampler and DP info
        if workflow_kwargs is not None:
            self._workflow_kwargs = workflow_kwargs.copy()
            # Sync lazy_sampling to the potentially modified config value
            # (TTTDDistillTrainer.__init__ may disable lazy_puct_sampling in sync mode)
            self._workflow_kwargs['lazy_sampling'] = config.sampler.lazy_puct_sampling
            if config.sampler.lazy_puct_sampling and 'sampler' not in self._workflow_kwargs:
                self._workflow_kwargs['sampler'] = self.sampler
                self._workflow_kwargs['dp_rank'] = self.actor.dp_rank
                self._workflow_kwargs['dp_world_size'] = self.actor.data_parallel_world_size

        is_dp_head = self.actor.rank == 0
        batch_size = config.sampler.batch_size
        group_size = config.gconfig.n_samples

        # Validate total rollouts
        total_rollouts = batch_size * group_size
        if total_rollouts != config.total_rollouts_per_step:
            logger.warning(
                f"[Distill] Total rollouts {total_rollouts} != "
                f"configured {config.total_rollouts_per_step}. "
                f"Using actual: batch_size={batch_size} * group_size={group_size} = {total_rollouts}"
            )

        logger.info(
            f"[Distill] Starting distillation: "
            f"max_steps={config.max_steps}, "
            f"total_rollouts_per_step={total_rollouts}, "
            f"distill_loss_weight={config.teacher.distill_loss_weight if config.teacher else 'N/A'}"
        )

        # =====================================================================
        # Phase 1: Distillation steps (no verification)
        # =====================================================================
        for step_idx in range(config.max_steps):
            global_step = step_idx
            logger.info(f"[Distill][Step {global_step}] Starting distill step")

            step_start_time = time.perf_counter()

            # Create distill workflow with dummy reward (no verification)
            distill_kwargs = self._workflow_kwargs.copy()
            distill_kwargs['reward_fn'] = dummy_reward_fn
            distill_kwargs['max_reward_workers'] = 1  # Dummy reward is fast
            distill_workflow = workflow(**distill_kwargs)

            # Set current version for tracking
            if hasattr(distill_workflow, 'set_current_version'):
                distill_workflow.set_current_version(global_step)

            # Clear workflow cache to ensure the new workflow instance is used
            self._clear_workflow_cache()

            # Rollout
            rollout_start = time.perf_counter()
            with stats_tracker.record_timing("rollout"):
                rollout_batch = self.actor.prepare_batch(
                    self.train_dataloader,
                    workflow=distill_workflow,
                    workflow_kwargs=None,  # Already resolved workflow instance
                    should_accept_fn=None,
                    group_size=group_size,
                    dynamic_bs=config.dynamic_bs,
                )
            rollout_time = time.perf_counter() - rollout_start

            # Normalize batch to dict for TTTDActor methods
            rollout_batch = self._normalize_rollout_batch(rollout_batch)

            # Compute global reward statistics
            local_rollouts = rollout_batch["rewards"].shape[0]
            step_rewards = rollout_batch["rewards"].cpu().numpy()
            step_max_reward = float(step_rewards.max())
            step_mean_reward = float(step_rewards.mean())
            step_min_reward = float(step_rewards.min())

            if dist.is_initialized():
                local_max = torch.tensor([step_max_reward], dtype=torch.float32, device=self.actor.device)
                local_min = torch.tensor([step_min_reward], dtype=torch.float32, device=self.actor.device)
                local_sum = torch.tensor([step_rewards.sum()], dtype=torch.float32, device=self.actor.device)
                local_count = torch.tensor([len(step_rewards)], dtype=torch.float32, device=self.actor.device)

                dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
                dist.all_reduce(local_min, op=dist.ReduceOp.MIN)
                dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

                step_max_reward = local_max.item()
                step_min_reward = local_min.item()
                step_mean_reward = (local_sum / local_count).item() if local_count.item() > 0 else 0.0
                global_rollouts = int(local_count.item())
            else:
                global_rollouts = local_rollouts

            logger.info(
                f"[Distill][Step {global_step}] Rollouts: {local_rollouts} (global: {global_rollouts}), "
                f"Reward: max={step_max_reward:.4f}, mean={step_mean_reward:.4f}, min={step_min_reward:.4f}"
            )

            # Training computations
            step_info = StepInfo(
                global_step=global_step,
                epoch=global_step,
                epoch_step=global_step,
                steps_per_epoch=config.max_steps,
            )

            # Compute teacher logp (native AReaL KDRL path)
            if self.teacher is not None:
                with torch.no_grad():
                    teacher_logps = self.teacher.compute_logp([rollout_batch])
                rollout_batch["teacher_logp"] = teacher_logps[0]
                rollout_batch["rl_loss_weight"] = self.config.teacher.rl_loss_weight
                rollout_batch["distill_loss_weight"] = self.config.teacher.distill_loss_weight

            # Compute prox_logp if needed (wrap dict in list for PPOActor.compute_logp)
            if config.actor.should_compute_prox_logp():
                prox_logps = self.actor.compute_logp([rollout_batch])
                rollout_batch["prox_logp"] = prox_logps[0]

            # =====================================================================
            # Compute dynamic metrics (overlap ratio, overlap-token advantage, entropy gap)
            # =====================================================================
            dynamic_metrics = {}
            if self.teacher is not None:
                try:
                    dynamic_metrics = self._compute_dynamic_metrics(rollout_batch, k=50)
                except Exception as e:
                    logger.warning(f"[Distill][Step {global_step}] Failed to compute dynamic metrics: {e}")

            # All-reduce dynamic metrics across DP ranks for global averages
            if dist.is_initialized() and dynamic_metrics:
                dp_world_size = self.actor.data_parallel_world_size
                for key in list(dynamic_metrics.keys()):
                    val = torch.tensor([dynamic_metrics[key]], dtype=torch.float32, device=self.actor.device)
                    dist.all_reduce(val, op=dist.ReduceOp.SUM)
                    dynamic_metrics[key] = (val / dp_world_size).item()

            # Record dynamic metrics to history logger (saved as JSON)
            if hasattr(self, 'dynamic_metrics_logger') and self.dynamic_metrics_logger is not None:
                try:
                    self.dynamic_metrics_logger.record_step(
                        step=global_step,
                        rewards=step_rewards.tolist(),
                        additional_metrics=dynamic_metrics,
                    )
                except Exception as e:
                    logger.warning(f"[Distill][Step {global_step}] Failed to record dynamic metrics to logger: {e}")

            # Compute advantages using TTTDActor (native AReaL logic)
            rollout_batch = self.actor.compute_advantages(rollout_batch)

            # PPO update: automatically handles KD loss when teacher_logp is present
            self.actor.ppo_update([rollout_batch])
            self.actor.step_lr_scheduler()

            # Export training stats
            stats = self.actor.export_stats()
            metrics = {
                "rollout/count": global_rollouts,
                "reward/max": step_max_reward,
                "reward/mean": step_mean_reward,
                "reward/min": step_min_reward,
                "train/actor_loss": stats.get('ppo_actor/update/actor_loss/avg', 0.0),
                "train/approx_kl": stats.get('ppo_actor/update/approx_kl/avg', 0.0),
                "train/entropy": stats.get('ppo_actor/update/entropy/avg', 0.0),
                "train/grad_norm": stats.get('ppo_actor/update/grad_norm', 0.0),
                "train/lr": stats.get('ppo_actor/update/lr', 0.0),
            }
            metrics.update(dynamic_metrics)

            if is_dp_head:
                self.stats_logger.commit(
                    epoch=global_step,
                    step=global_step,
                    global_step=global_step,
                    data=metrics,
                )
                log_msg = (
                    f"[Distill][Step {global_step}] "
                    f"Loss: {metrics['train/actor_loss']:.4f}, "
                    f"KL: {metrics['train/approx_kl']:.4f}, "
                    f"Entropy: {metrics['train/entropy']:.4f}, "
                    f"GradNorm: {metrics['train/grad_norm']:.4f}, "
                    f"LR: {metrics['train/lr']:.6f}"
                )
                if dynamic_metrics:
                    log_msg += (
                        f", Overlap: {metrics.get('distill/overlap_ratio', 0.0):.4f}, "
                        f"Adv: {metrics.get('distill/overlap_token_advantage', 0.0):.4f}, "
                        f"EntGap: {metrics.get('distill/entropy_gap', 0.0):.4f}"
                    )
                logger.info(log_msg)

            # Update weights and save
            logger.info(f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] Before rollout.pause")
            self.rollout.pause()
            logger.info(f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] After rollout.pause")

            logger.info(f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] Before actor.update_weights")
            new_version = global_step + 1
            versioned_meta = self.weight_update_meta.with_version(new_version)
            logger.info(f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] Using versioned_meta version={versioned_meta.version}")
            self.actor.update_weights(versioned_meta)
            logger.info(f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] After actor.update_weights")

            self.actor.set_version(new_version)
            self.rollout.set_version(new_version)
            logger.info(f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] After set_version")

            logger.info(f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] Before _save_hf")
            self._save_hf(epoch=global_step, epoch_step=global_step, global_step=global_step)
            logger.info(f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] After _save_hf")

            logger.info(f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] Before _save_recover_checkpoint")
            self._save_recover_checkpoint(epoch=global_step, epoch_step=global_step, global_step=global_step)
            logger.info(f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] After _save_recover_checkpoint")

            logger.info(f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] Before dist.barrier")
            dist.barrier(group=self.actor.cpu_group)
            logger.info(f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] After dist.barrier")

            logger.info(f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] Before current_platform.synchronize")
            current_platform.synchronize()
            logger.info(f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] After current_platform.synchronize")

            logger.info(f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] Before rollout.resume")
            self.rollout.resume()
            logger.info(f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] After rollout.resume")

            step_total = time.perf_counter() - step_start_time
            logger.info(
                f"[Distill][Step {global_step}] Completed in {step_total:.2f}s "
                f"(rollout={rollout_time:.2f}s)"
            )

        # =====================================================================
        # Save final student checkpoint for separate evaluation
        # =====================================================================
        from areal.api.io_struct import SaveLoadMeta
        student_meta = SaveLoadMeta(
            path=self.student_eval_path,
            weight_format="hf",
            with_optim=False,
            tokenizer=None,
            processor=None,
        )
        self.actor.save(student_meta)
        if is_dp_head:
            logger.info(f"[Distill] Saved student checkpoint to {self.student_eval_path}")

        # Save dynamic metrics history (JSON + PKL)
        if hasattr(self, 'dynamic_metrics_logger') and self.dynamic_metrics_logger is not None:
            history_path = self.dynamic_metrics_logger.save(also_save_json=True)
            if history_path:
                logger.info(f"[DynamicMetrics] Saved history to {history_path}")

        if dist.is_initialized():
            dist.barrier(group=self.actor.cpu_group)

    def _save_hf(self, epoch: int, epoch_step: int, global_step: int):
        """Override parent _save_hf to remove extra barrier."""
        self.saver.save(
            self.actor,
            epoch,
            epoch_step,
            global_step,
            tokenizer=self.tokenizer,
            processor=getattr(self, 'processor', None),
        )

    def _save_recover_checkpoint(self, epoch: int, epoch_step: int, global_step: int):
        """Override parent _save_recover_checkpoint to remove extra barrier."""
        from areal.api.io_struct import StepInfo

        to_save = dict(default=self.actor)
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=epoch_step,
            steps_per_epoch=1,
        )

        self.recover_handler._save_checkpoint(
            self.actor,
            name="default",
            tokenizer=self.tokenizer,
            processor=getattr(self, 'processor', None),
        )

        self.recover_handler.last_step_info = step_info

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

    def _compute_dynamic_metrics(self, rollout_batch: dict[str, Any], k: int = 50) -> dict[str, float]:
        """Compute dynamic metrics from the TTT-Discover paper.

        Metrics:
        - overlap_ratio: M_overlap = E_t[ |S_t^(p) ∩ S_t^(q)| / k ]
        - overlap_token_advantage: M_adv = E_t[ 1/|∩| Σ_{v∈∩} A_t(v) ]
        - entropy_gap: ΔH_t = |H(q_t) - H(p_t)|

        To control memory, we chunk the batch and only keep top-k indices/logprobs
        instead of the full vocab distribution.
        """
        chunk_size = getattr(self.config, "metric_chunk_size", 4)

        input_ids = rollout_batch["input_ids"]
        attention_mask = rollout_batch["attention_mask"]
        loss_mask = rollout_batch["loss_mask"]
        n, seqlen = input_ids.shape

        def _get_topk_and_entropy(engine, device):
            engine.model.eval()
            topk_indices = []
            topk_logps = []
            entropies = []

            for i in range(0, n, chunk_size):
                end = min(i + chunk_size, n)
                ids = input_ids[i:end].to(device)
                mask = attention_mask[i:end].to(device)
                lmask = loss_mask[i:end].to(device)

                with torch.no_grad():
                    out = engine.model(input_ids=ids, attention_mask=mask)
                    logits = out.logits  # [chunk, seqlen, vocab]

                log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
                entropy = -(log_probs.exp() * log_probs).sum(-1)  # [chunk, seqlen]

                active = lmask > 0
                if active.any():
                    entropies.append(entropy[active].cpu())
                    active_log_probs = log_probs[active]  # [n_active, vocab]
                    tk_logp, tk_idx = torch.topk(active_log_probs, k, dim=-1)
                    topk_logps.append(tk_logp.cpu())
                    topk_indices.append(tk_idx.cpu())

            if not entropies:
                return None, None, None
            return torch.cat(topk_indices), torch.cat(topk_logps), torch.cat(entropies)

        s_idx, s_logp, s_ent = _get_topk_and_entropy(self.actor, self.actor.device)
        t_idx, t_logp, t_ent = _get_topk_and_entropy(self.teacher, self.teacher.device)

        if s_idx is None or t_idx is None:
            return {}

        # Entropy metrics (GPU/CPU tensor ops)
        entropy_gap = (t_ent - s_ent).abs().mean().item()

        # Overlap metrics (CPU numpy for simplicity and speed with small k)
        s_idx_np = s_idx.numpy()
        t_idx_np = t_idx.numpy()
        s_logp_np = s_logp.numpy()
        t_logp_np = t_logp.numpy()

        n_active = s_idx_np.shape[0]
        overlap_ratios = np.zeros(n_active, dtype=np.float64)
        adv_values = np.zeros(n_active, dtype=np.float64)

        for i in range(n_active):
            s_set = set(s_idx_np[i])
            t_set = set(t_idx_np[i])
            inter = s_set & t_set
            if not inter:
                continue

            overlap_ratios[i] = len(inter) / k

            p_vals = []
            q_vals = []
            for tok in inter:
                s_pos = int(np.where(s_idx_np[i] == tok)[0][0])
                t_pos = int(np.where(t_idx_np[i] == tok)[0][0])
                p_vals.append(s_logp_np[i][s_pos])
                q_vals.append(t_logp_np[i][t_pos])

            p = np.exp(np.array(p_vals, dtype=np.float64))
            q = np.exp(np.array(q_vals, dtype=np.float64))
            p = p / (p.sum() + 1e-12)
            q = q / (q.sum() + 1e-12)

            A = p * (np.log(q + 1e-12) - np.log(p + 1e-12))
            adv_values[i] = A.sum() / len(inter)

        return {
            "distill/overlap_ratio": float(overlap_ratios.mean()),
            "distill/overlap_token_advantage": float(adv_values.mean()),
            "distill/entropy_gap": entropy_gap,
            "distill/student_entropy": s_ent.mean().item(),
            "distill/teacher_entropy": t_ent.mean().item(),
        }

    def close(self):
        """Cleanup resources."""
        # Save dynamic metrics checkpoint before cleanup
        if hasattr(self, 'dynamic_metrics_logger') and self.dynamic_metrics_logger is not None:
            try:
                checkpoint_path = self.dynamic_metrics_logger.save_checkpoint()
                if checkpoint_path:
                    logger.info(f"[DynamicMetrics] Saved checkpoint to {checkpoint_path}")
            except Exception as e:
                logger.warning(f"[DynamicMetrics] Failed to save checkpoint: {e}")

        self.stats_logger.close()
        if hasattr(self, 'rollout') and self.rollout is not None:
            self.rollout.destroy()
        if hasattr(self, 'teacher') and self.teacher is not None:
            self.teacher.destroy()
        if hasattr(self, 'actor') and self.actor is not None:
            self.actor.destroy()
        from areal.utils import perf_tracer
        perf_tracer.save(force=True)


# =============================================================================
# Main
# =============================================================================
def main(args):
    """Main training function."""
    import os


    config, _ = load_expr_config(args, TTTDDistillConfig)

    # Validate teacher config
    if config.teacher is None:
        raise ValueError(
            "teacher config block must be provided for distillation. "
            "Add teacher: {...} to your YAML."
        )
    if not config.teacher_sampler_checkpoint:
        raise ValueError(
            "teacher_sampler_checkpoint must be provided. "
            "Add +teacher_sampler_checkpoint=<path> to your command."
        )

    # Ensure stop tokens are set
    if config.tokenizer_path:
        from areal.utils.hf_utils import load_hf_tokenizer
        tokenizer = load_hf_tokenizer(config.tokenizer_path)
        if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
            config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
        if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
            config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)

    # Verify LoRA adapter exists (required for SPMD mode vLLM pre-loading)
    if config.actor.use_lora and not config.skip_lora_check:
        lora_output_path = "./lora_init"
        if hasattr(config, 'vllm'):
            if isinstance(config.vllm, dict):
                lora_modules_str = config.vllm.get('lora_modules', '')
            else:
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
                f"Run the preparation script first:\n"
                f"  python areal/experimental/ttt_discover/examples/prepare_lora_init.py"
                f" --config-path <your_config.yaml>\n\n"
                f"Or skip this check with +skip_lora_check=true\n"
                f"{'='*80}\n"
            )
            raise RuntimeError(error_msg)

        logger.info(f"[LoRA Check] LoRA adapter verified at {lora_output_path}")

    # Create environment (needed for workflow)
    env = create_env_from_config(config)

    # Calculate local batch size
    from areal.api.alloc_mode import _AllocationMode as AllocationMode
    alloc_mode = AllocationMode.from_str(config.allocation_mode)
    train_world_size = alloc_mode.train.world_size
    local_batch_size = config.sampler.batch_size // train_world_size
    group_size = config.gconfig.n_samples

    # Validate total rollouts
    total_rollouts = config.sampler.batch_size * group_size
    if total_rollouts != config.total_rollouts_per_step:
        logger.warning(
            f"[Main] Configured total_rollouts_per_step={config.total_rollouts_per_step} "
            f"but actual rollouts will be {total_rollouts} "
            f"(batch_size={config.sampler.batch_size} * group_size={group_size})"
        )

    # Workflow kwargs
    workflow_kwargs = dict(
        env=env,
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=config.enable_thinking,
        max_prompt_thinking_tokens=config.max_prompt_thinking_tokens,
        batch_size=local_batch_size,
        group_size=group_size,
        lazy_sampling=config.sampler.lazy_puct_sampling,
        vllm_concurrency=config.sampler.vllm_concurrency,
        execution_concurrency=config.sampler.execution_concurrency,
    )

    # Run distillation
    with TTTDDistillTrainer(config) as trainer:
        trainer.train(
            workflow=TTTDiscoverWorkflowV2,
            workflow_kwargs=workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
