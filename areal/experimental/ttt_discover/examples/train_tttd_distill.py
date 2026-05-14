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
    python -m areal.infra.launcher.local \
        areal/experimental/ttt_discover/examples/conf/fsdp_lora_vllm_ac1_qwen3_8b_distill.yaml
"""

import copy
import json
import os
import queue
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
        # Create TWO samplers:
        #   - student_sampler: fresh, only initial states (for student rollout)
        #   - teacher_sampler: loaded from teacher checkpoint (for privileged OPD)
        # =====================================================================
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

        # --- Teacher sampler: loaded from checkpoint, used for privileged OPD ---
        teacher_sampler_config = copy.deepcopy(config.sampler)
        if config.teacher_sampler_checkpoint:
            teacher_sampler_config.checkpoint_dir = config.teacher_sampler_checkpoint

        self.teacher_sampler = create_sampler_from_config(
            config=teacher_sampler_config,
            env_type=getattr(config.sampler, 'env_type', 'ac1'),
            max_version_history=max_version_history,
        )

        if config.teacher_sampler_checkpoint:
            latest_step = _find_latest_sampler_step(
                config.teacher_sampler_checkpoint,
                getattr(config.sampler, 'type', 'puct')
            )
            if latest_step is not None:
                logger.info(f"[TeacherSampler] Loading latest checkpoint at step {latest_step}")
                self.teacher_sampler._load(latest_step)
                self.teacher_sampler._current_step = 0
            else:
                logger.warning(
                    f"[TeacherSampler] No checkpoint found in {config.teacher_sampler_checkpoint}. "
                    f"Using fresh sampler state."
                )

        # Teacher sampler strategy for privileged OPD (configurable)
        self.teacher_sampler.sampling_strategy = config.teacher_sampler_strategy
        logger.info(
            f"[TeacherSampler] Loaded {len(self.teacher_sampler._states)} states, "
            f"T={self.teacher_sampler._T}, strategy=puct (for privileged OPD)"
        )

        # --- Student sampler: fresh, only initial states (for student rollout) ---
        student_sampler_config = copy.deepcopy(config.sampler)
        # Ensure student sampler uses its own checkpoint dir (not teacher's)
        # so it doesn't accidentally load teacher states
        if not getattr(student_sampler_config, 'checkpoint_dir', None):
            student_sampler_config.checkpoint_dir = os.path.join(
                config.saver.fileroot,
                config.experiment_name,
                config.trial_name,
                "student_sampler",
            )

        self.sampler = create_sampler_from_config(
            config=student_sampler_config,
            env_type=getattr(config.sampler, 'env_type', 'ac1'),
            max_version_history=1,  # Student doesn't need version history
        )
        # Student sampler uses the config's default strategy (usually parent_pool
        # or initial) so student sees only initial/random states
        self.sampler.sampling_strategy = getattr(
            config.sampler, 'sampling_strategy', 'parent_pool'
        )
        logger.info(
            f"[StudentSampler] Fresh sampler with {len(self.sampler._states)} states, "
            f"strategy={self.sampler.sampling_strategy} (for student rollout)"
        )

        # --- Optionally inherit teacher sampler's state pool into student sampler ---
        if config.student_sampler_inherit_teacher_pool:
            self.sampler._states = copy.deepcopy(self.teacher_sampler._states)
            self.sampler._initial_states = copy.deepcopy(self.teacher_sampler._initial_states)
            self.sampler._n = copy.deepcopy(self.teacher_sampler._n)
            self.sampler._m = copy.deepcopy(self.teacher_sampler._m)
            self.sampler._T = self.teacher_sampler._T
            logger.info(
                f"[StudentSampler] Inherited teacher pool: {len(self.sampler._states)} states, "
                f"T={self.sampler._T}, n_entries={len(self.sampler._n)}, "
                f"m_entries={len(self.sampler._m)}"
            )

        # =====================================================================
        # Create environment (needed for privileged prompt construction)
        # =====================================================================
        self.env = create_env_from_config(config)

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
        eval_ckpt_dir = os.path.join(
            config.saver.fileroot,
            config.experiment_name,
            config.trial_name,
            "eval_checkpoints",
        )
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
        output_dir = os.path.join(
            config.saver.fileroot,
            config.experiment_name,
            config.trial_name,
            "dynamic_metrics",
        )
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

        # Clear stale state after recovery (critical for correct staleness capacity)
        self._clear_stale_state_after_recovery()

    def _get_dispatcher(self):
        """Get dispatcher from rollout (handles both RolloutController and RemotevLLMEngine)."""
        if hasattr(self.rollout, 'dispatcher'):
            return self.rollout.dispatcher
        if hasattr(self.rollout, '_engine') and hasattr(self.rollout._engine, 'workflow_executor'):
            return self.rollout._engine.workflow_executor._dispatcher
        if hasattr(self.rollout, 'workflow_executor'):
            return self.rollout.workflow_executor._dispatcher
        return None

    def _clear_stale_state_after_recovery(self):
        """Clear stale state after recovery to fix staleness manager capacity calculation."""
        is_dp_head = self.actor.rank == 0

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
                        except queue.Empty:
                            break
                if is_dp_head and queue_cleared > 0:
                    logger.info(f"[Resume] Cleared {queue_cleared} items from async task queues")

                # 3. CRITICAL: Fix accepted count for staleness capacity calculation
                # Normal training: accepted = step * consumer_batch_size at each step start
                # Recovery: version = start_step, so accepted should be start_step * consumer_batch_size
                sm = dispatcher.staleness_manager
                with sm.lock:
                    consumer_bs = sm.consumer_batch_size
                    completed_steps = start_step
                    expected_accepted = completed_steps * consumer_bs

                    old_accepted = sm.rollout_stat.accepted
                    old_running = sm.rollout_stat.running
                    old_enqueued = sm.rollout_stat.enqueued

                    sm.rollout_stat.running = 0
                    sm.rollout_stat.enqueued = 0

                    if old_accepted != expected_accepted:
                        sm.rollout_stat.accepted = expected_accepted
                        if is_dp_head:
                            logger.info(
                                f"[Resume] Fixed accepted count: {old_accepted} -> {expected_accepted} "
                                f"(expected after {completed_steps} completed steps, consumer_batch_size={consumer_bs})"
                            )
                    elif is_dp_head:
                        logger.info(f"[Resume] accepted count is correct: {old_accepted}")

                    version = self.rollout.get_version()
                    max_staleness = sm.max_staleness
                    capacity = (max_staleness + version + 1) * consumer_bs - sm.rollout_stat.accepted
                    if is_dp_head:
                        logger.info(
                            f"[Resume] staleness_capacity = ({max_staleness} + {version} + 1) * {consumer_bs} - "
                            f"{sm.rollout_stat.accepted} = {capacity}"
                        )

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

            # 6. Clear workflow internal state (staleness tracker, etc.)
            self._clear_workflow_state_after_recovery(start_step)

            if is_dp_head:
                logger.info(f"[Resume] Ready to start from step {start_step}")

    def _clear_workflow_state_after_recovery(self, start_step: int):
        """Clear workflow internal state after recovery.

        This clears all transient state that should not persist across resumes:
        - Scheme 1 batch tracking
        - Staleness tracker (incomplete rollouts from old version)
        - Parent episodes
        """
        is_dp_head = self.actor.rank == 0

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

        if hasattr(workflow, '_current_batch_parent_ids'):
            stale_count = len(workflow._current_batch_parent_ids)
            if stale_count > 0:
                workflow._current_batch_parent_ids.clear()
                workflow._expected_batch_size = 0
                workflow._expected_n_samples = 0
                cleared_items.append(f"scheme1_batch({stale_count})")

        if hasattr(workflow, '_staleness_tracker'):
            stale_count = len(workflow._staleness_tracker)
            if stale_count > 0:
                workflow._staleness_tracker.clear()
                cleared_items.append(f"staleness_tracker({stale_count})")

        if hasattr(workflow, '_parent_episodes'):
            stale_count = len(workflow._parent_episodes)
            if stale_count > 0:
                workflow._parent_episodes.clear()
                cleared_items.append(f"parent_episodes({stale_count})")

        if cleared_items and is_dp_head:
            logger.info(f"[Resume] Cleared workflow state: {', '.join(cleared_items)}")

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

        # Determine starting step from recovery info
        start_step = (
            self.recover_info.last_step_info.next().global_step
            if getattr(self, 'recover_info', None) is not None
            else 0
        )
        if start_step > 0:
            logger.info(f"[Distill] Resuming from step {start_step} (recovered)")

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
            f"max_steps={config.max_steps}, start_step={start_step}, "
            f"total_rollouts_per_step={total_rollouts}, "
            f"distill_loss_weight={config.teacher.distill_loss_weight if config.teacher else 'N/A'}"
        )

        # =====================================================================
        # Phase 1: Distillation steps (no verification)
        # =====================================================================
        for global_step in range(start_step, config.max_steps):
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

            # Compute teacher logp
            if self.teacher is not None:
                torch.cuda.empty_cache()
                with torch.no_grad():
                    if config.use_privileged_teacher_logp:
                        teacher_logps = self._compute_privileged_teacher_logp(rollout_batch)
                    else:
                        # Teacher and student see the same prompts (no privileged OPD)
                        teacher_logps_list = self.teacher.compute_logp([rollout_batch])
                        teacher_logps = teacher_logps_list[0]
                rollout_batch["teacher_logp"] = teacher_logps
                rollout_batch["rl_loss_weight"] = self.config.teacher.rl_loss_weight
                rollout_batch["distill_loss_weight"] = self.config.teacher.distill_loss_weight

            # Compute prox_logp if needed
            if config.actor.should_compute_prox_logp():
                torch.cuda.empty_cache()
                prox_logps = self.actor.compute_logp([rollout_batch])
                rollout_batch["prox_logp"] = prox_logps[0]

            # Compute advantages using TTTDActor (native AReaL logic)
            rollout_batch = self.actor.compute_advantages(rollout_batch)

            # Clear cached memory before backward pass to prevent OOM
            torch.cuda.empty_cache()

            # PPO update: automatically handles KD loss when teacher_logp is present
            self.actor.ppo_update([rollout_batch])
            self.actor.step_lr_scheduler()

            # =====================================================================
            # =====================================================================
            # Compute dynamic metrics AFTER ppo_update so OOM here doesn't block training
            # Head (first step) and tail (last step) are always computed.
            # Middle steps use 1-based modulo: step 4, 9, 14, ... (i.e. (step+1)%5==0)
            # =====================================================================
            dynamic_metrics = {}
            is_first_step = (global_step == start_step)
            is_last_step = (global_step == config.max_steps - 1)
            is_middle_5th = ((global_step + 1) % 5 == 0)
            if self.teacher is not None and (is_first_step or is_last_step or is_middle_5th):
                try:
                    torch.cuda.empty_cache()
                    dynamic_metrics = self._compute_dynamic_metrics(rollout_batch, k=16)
                except Exception as e:
                    logger.warning(
                        f"[Distill][Step {global_step}] Failed to compute dynamic metrics: {e}"
                    )

            # All-reduce dynamic metrics across DP ranks (weighted by token count)
            if dist.is_initialized() and dynamic_metrics:
                count = dynamic_metrics.pop("distill/_count", 0)
                count_t = torch.tensor([float(count)], dtype=torch.float32, device=self.actor.device)
                dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
                total_count = int(count_t.item())

                for key in list(dynamic_metrics.keys()):
                    val = torch.tensor([dynamic_metrics[key]], dtype=torch.float32, device=self.actor.device)
                    dist.all_reduce(val, op=dist.ReduceOp.SUM)
                    dynamic_metrics[key] = (val / total_count).item() if total_count > 0 else 0.0

            # Record dynamic metrics to history logger (saved as JSON)
            if hasattr(self, 'dynamic_metrics_logger') and self.dynamic_metrics_logger is not None:
                try:
                    # Use actual rollout rewards instead of dummy rewards
                    actual_rewards = rollout_batch["rewards"].cpu().numpy().tolist()
                    if not isinstance(actual_rewards, list):
                        actual_rewards = [actual_rewards]
                    self.dynamic_metrics_logger.record_step(
                        step=global_step,
                        rewards=actual_rewards,
                        additional_metrics=dynamic_metrics,
                    )
                except Exception as e:
                    logger.warning(f"[Distill][Step {global_step}] Failed to record dynamic metrics to logger: {e}")

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

            # Save dynamic metrics checkpoint after each step (for crash recovery)
            if hasattr(self, 'dynamic_metrics_logger') and self.dynamic_metrics_logger is not None:
                try:
                    self.dynamic_metrics_logger.save_checkpoint()
                except Exception as e:
                    logger.warning(f"[Distill][Step {global_step}] Failed to save dynamic metrics checkpoint: {e}")

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

    def _compute_privileged_teacher_logp(
        self, rollout_batch: dict[str, Any]
    ) -> torch.Tensor:
        """Compute teacher logp on privileged prompts + student completions (group-level).

        We sample one privileged state per group (not per rollout) for stable
        teacher signal. All rollouts within the same group share the same
        privileged context.

        In distributed training (DP > 1), we follow the same global-sample +
        local-slice pattern as the dataloader: sample `global_num_groups` states
        on every rank, then each rank takes its `rank * local_num_groups` slice.
        This ensures different ranks receive distinct privileged states.

        Steps:
        1. Sample privileged states: one per group from teacher PUCTSampler.
        2. Tokenize privileged prompts via env.get_prompt(state).
        3. Extract student completions from rollout_batch.
        4. Build teacher sequences: repeat each privileged prompt group_size
           times and concatenate with the corresponding student completions.
        5. Call teacher.compute_logp() and align completion logps back to
           student sequence positions using the verified offset:
           teacher_comp_start = priv_len - 1, student_comp_start = prompt_len - 1.

        Returns
        -------
        aligned_teacher_logp : torch.Tensor
            Shape [batch_size, student_seq_len], with teacher completion logps
            placed at the same positions as student completion tokens.
        """
        input_ids = rollout_batch["input_ids"]
        attention_mask = rollout_batch["attention_mask"]
        loss_mask = rollout_batch["loss_mask"]
        batch_size, student_seqlen = input_ids.shape
        device = input_ids.device
        group_size = self.config.gconfig.n_samples

        # ------------------------------------------------------------------
        # 1. Sample privileged states (one per group) with DP sharding
        # ------------------------------------------------------------------
        local_num_groups = batch_size // group_size
        if local_num_groups * group_size != batch_size:
            logger.warning(
                f"[PrivilegedOPD] batch_size ({batch_size}) is not divisible by "
                f"group_size ({group_size}). Falling back to sample-level sampling."
            )
            local_num_groups = batch_size
            group_size = 1

        dp_world_size = self.actor.data_parallel_world_size
        dp_rank = self.actor.data_parallel_rank
        global_num_groups = local_num_groups * dp_world_size

        # All ranks sample the same global pool (deterministic for puct strategy)
        all_privileged_states = self.teacher_sampler.sample_states(global_num_groups)

        # Take this rank's slice so each rank gets distinct privileged states
        start_idx = dp_rank * local_num_groups
        privileged_states = all_privileged_states[start_idx:start_idx + local_num_groups]

        # Log sampled privileged state values for debugging
        for i, state in enumerate(privileged_states):
            logger.info(
                f"[PrivilegedOPD] Rank {dp_rank} state {i}/{local_num_groups} "
                f"(global {start_idx + i}/{global_num_groups}): "
                f"value={state.value:.4f}, timestep={state.timestep}, "
                f"id={state.id[:8] if hasattr(state.id, '__len__') and len(state.id) > 8 else state.id}"
            )

        # ------------------------------------------------------------------
        # 2. Build privileged prompts and tokenize
        # ------------------------------------------------------------------
        privileged_prompt_ids_list = []
        for state in privileged_states:
            prompt = self.env.get_prompt(state)
            messages = [{"role": "user", "content": prompt}]
            ids = list(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=getattr(self.config, "enable_thinking", False),
                )
            )
            privileged_prompt_ids_list.append(ids)

        # Expand each privileged prompt to group_size copies
        privileged_prompt_ids_expanded = []
        for ids in privileged_prompt_ids_list:
            privileged_prompt_ids_expanded.extend([ids] * group_size)

        # ------------------------------------------------------------------
        # 3. Extract student completions from rollout_batch
        # ------------------------------------------------------------------
        # prompt_len = # of positions where attention_mask=1 and loss_mask=0
        # comp_len   = # of positions where loss_mask=1
        prompt_lens = (attention_mask & (loss_mask == 0)).sum(dim=1).cpu().numpy()
        comp_lens = loss_mask.sum(dim=1).cpu().numpy()

        # ------------------------------------------------------------------
        # 4. Build teacher sequences: privileged_prompt + student_completion
        # ------------------------------------------------------------------
        teacher_seqs = []
        teacher_loss_masks = []
        for i in range(batch_size):
            prompt_len = int(prompt_lens[i])
            comp_len = int(comp_lens[i])
            # Student completion tokens
            student_comp = input_ids[i, prompt_len : prompt_len + comp_len].cpu().tolist()
            # Privileged prompt tokens
            priv_prompt = privileged_prompt_ids_expanded[i]
            priv_len = len(priv_prompt)
            # Concatenate
            teacher_seq = priv_prompt + student_comp
            teacher_seqs.append(torch.tensor(teacher_seq, dtype=torch.int32))
            # Loss mask: 0 for prompt, 1 for completion
            teacher_loss_masks.append(
                torch.tensor([0] * priv_len + [1] * comp_len, dtype=torch.int32)
            )

        # Pad teacher sequences to max length
        max_teacher_len = max(len(seq) for seq in teacher_seqs)
        pad_id = self.tokenizer.pad_token_id or 0
        teacher_input_ids = torch.stack(
            [
                torch.nn.functional.pad(seq, (0, max_teacher_len - len(seq)), value=pad_id)
                for seq in teacher_seqs
            ]
        )
        teacher_attention_mask = torch.stack(
            [
                torch.nn.functional.pad(
                    torch.ones(len(seq), dtype=torch.bool),
                    (0, max_teacher_len - len(seq)),
                    value=False,
                )
                for seq in teacher_seqs
            ]
        )
        teacher_loss_mask = torch.stack(
            [
                torch.nn.functional.pad(
                    mask, (0, max_teacher_len - len(mask)), value=0
                )
                for mask in teacher_loss_masks
            ]
        )

        teacher_batch = {
            "input_ids": teacher_input_ids,
            "attention_mask": teacher_attention_mask,
            "loss_mask": teacher_loss_mask,
        }

        # ------------------------------------------------------------------
        # 5. Call teacher.compute_logp()
        # ------------------------------------------------------------------
        with torch.no_grad():
            teacher_logps_list = self.teacher.compute_logp([teacher_batch])
        teacher_logps_full = teacher_logps_list[0]  # [batch_size, max_teacher_len]

        # ------------------------------------------------------------------
        # 6. Align completion logps back to student sequence positions
        # ------------------------------------------------------------------
        aligned_teacher_logp = torch.zeros(
            (batch_size, student_seqlen), dtype=torch.float32, device=device
        )
        for i in range(batch_size):
            prompt_len = int(prompt_lens[i])
            comp_len = int(comp_lens[i])
            priv_len = len(privileged_prompt_ids_expanded[i])
            if comp_len == 0:
                continue
            # Teacher completion logps:
            #   compute_logp returns logprobs[j] = log p(input_ids[j+1] | ...)
            #   So completion logps span [priv_len-1, priv_len+comp_len-1)
            t_start = priv_len - 1
            t_end = priv_len + comp_len - 1
            teacher_comp_logps = teacher_logps_full[i, t_start:t_end]

            # Student completion positions:
            #   Same offset: [prompt_len-1, prompt_len+comp_len-1)
            s_start = prompt_len - 1
            s_end = prompt_len + comp_len - 1
            aligned_teacher_logp[i, s_start:s_end] = teacher_comp_logps.to(
                device=device, dtype=torch.float32
            )

        return aligned_teacher_logp

    def _compute_dynamic_metrics(
        self, rollout_batch: dict[str, Any], k: int = 16
    ) -> dict[str, float]:
        """Compute dynamic metrics (overlap ratio, advantage, entropy gap).

        Lightweight wrapper around _compute_dynamic_metrics_and_logps that
        discards the per-token logp tensors and returns only the scalar metrics.
        """
        metrics, _t_logp, _s_logp = self._compute_dynamic_metrics_and_logps(
            rollout_batch, k=k
        )
        return metrics

    def _compute_dynamic_metrics_and_logps(
        self, rollout_batch: dict[str, Any], k: int = 16
    ) -> tuple[dict[str, float], torch.Tensor | None, torch.Tensor | None]:
        """Compute dynamic metrics and extract per-token logps in one pass.

        Returns (metrics_dict, teacher_logp, student_logp).
        teacher_logp and student_logp have shape [batch, seq_len] matching
        compute_logp() output, so caller can skip redundant forward passes.
        """
        seq_chunk_size = getattr(self.config, "metric_seq_chunk_size", 512)

        input_ids = rollout_batch["input_ids"]
        attention_mask = rollout_batch["attention_mask"]
        loss_mask = rollout_batch["loss_mask"]
        n, seqlen = input_ids.shape

        def _get_topk_entropy_and_logp(engine, device, input_ids_full):
            engine.model.eval()
            topk_indices = []
            topk_logps = []
            entropies = []
            all_token_logps = []

            for i in range(n):
                ids = input_ids[i : i + 1].to(device)
                mask = attention_mask[i : i + 1].to(device)
                lmask = loss_mask[i : i + 1].to(device)

                with torch.no_grad():
                    out = engine.model(input_ids=ids, attention_mask=mask)
                    logits = out.logits.squeeze(0)  # [seqlen, vocab]

                active = lmask.squeeze(0) > 0
                # Build token logps on CPU to avoid holding GPU memory across sequences
                seq_token_logps = torch.zeros(seqlen, dtype=torch.float32)

                if not active.any():
                    del out, logits
                    all_token_logps.append(seq_token_logps)
                    continue

                active_logits = logits[active]  # [n_active, vocab]
                n_active = active_logits.shape[0]
                token_logps = []

                for j in range(0, n_active, seq_chunk_size):
                    chunk_logits = active_logits[j : j + seq_chunk_size]
                    log_probs = torch.nn.functional.log_softmax(
                        chunk_logits.float(), dim=-1
                    )

                    # Extract per-token logp for actual tokens (reuse for compute_logp)
                    chunk_input_ids = input_ids_full[i][active][j : j + seq_chunk_size]
                    chunk_token_logp = log_probs.gather(
                        dim=-1, index=chunk_input_ids.unsqueeze(-1)
                    ).squeeze(-1)
                    token_logps.append(chunk_token_logp.cpu())

                    entropy = -(log_probs.exp() * log_probs).sum(-1)
                    entropies.append(entropy.cpu())

                    tk_logp, tk_idx = torch.topk(log_probs, k, dim=-1)
                    topk_logps.append(tk_logp.cpu())
                    topk_indices.append(tk_idx.cpu())

                    del chunk_logits, log_probs, entropy, tk_logp, tk_idx

                active_positions = torch.where(active)[0].cpu()
                seq_token_logps[active_positions] = torch.cat(token_logps)
                all_token_logps.append(seq_token_logps)

                del out, logits, active_logits, active

            if not entropies:
                return None, None, None, None
            return (
                torch.cat(topk_indices),
                torch.cat(topk_logps),
                torch.cat(entropies),
                torch.stack(all_token_logps).to(device),  # [n, seqlen]
            )

        s_idx, s_logp, s_ent, s_token_logp = _get_topk_entropy_and_logp(
            self.actor, self.actor.device, input_ids
        )
        torch.cuda.empty_cache()
        t_idx, t_logp, t_ent, t_token_logp = _get_topk_entropy_and_logp(
            self.teacher, self.teacher.device, input_ids
        )

        if s_idx is None or t_idx is None or s_token_logp is None or t_token_logp is None:
            return {}, None, None

        # Overlap metrics (CPU numpy)
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

        metrics = {
            "distill/overlap_ratio": float(overlap_ratios.sum()),
            "distill/overlap_token_advantage": float(adv_values.sum()),
            "distill/entropy_gap": float((t_ent - s_ent).abs().sum()),
            "distill/student_entropy": float(s_ent.sum()),
            "distill/teacher_entropy": float(t_ent.sum()),
            "distill/_count": int(n_active),
        }
        return metrics, t_token_logp, s_token_logp

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
