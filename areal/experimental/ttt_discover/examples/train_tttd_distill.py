#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

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
from areal.experimental.ttt_discover.ale_bench_eval import (
    combine_ale_bench_results,
    evaluate_problem_subset,
    list_ale_bench_problem_ids,
)
from areal.experimental.ttt_discover.config import (
    MultiTeacherConfig,
    TTTDDistillConfig,
    TTTDPPOActorConfig,
    create_env_from_config,
)
from areal.experimental.ttt_discover.dataloader import (
    create_multi_problem_tttd_dataloader,
    create_tttd_dataloader,
)
from areal.experimental.ttt_discover.envs.ale_bench import (
    create_initial_state_ale_bench,
)
from areal.experimental.ttt_discover.envs.env import BaseEnv, EnvResult
from areal.experimental.ttt_discover.reward import tttd_reward_fn
from areal.experimental.ttt_discover.sampler import (
    StateSampler,
    _find_latest_sampler_step,
    create_sampler_from_config,
)
from areal.experimental.ttt_discover.ttt_logger import TTTTrainingLogger
from areal.experimental.ttt_discover.workflow_v2 import (
    MultiProblemTTTDiscoverWorkflowV2,
    TTTDiscoverWorkflowV2,
)
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
        rank = int(__import__("os").getenv("RANK", "0"))
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
        self.rollout_alloc = ModelAllocation.from_str(
            config.rollout.backend, name="rollout"
        )
        self._amend_xccl_weight_update_envvar()

        # =====================================================================
        # Single-teacher vs. multi-teacher setup
        # =====================================================================
        self.is_multi_teacher = bool(config.multi_teacher)
        self.problem_envs: dict[str, BaseEnv] = {}
        self.problem_samplers: dict[str, StateSampler] = {}
        self.teacher_samplers: dict[str, StateSampler] = {}
        self.teacher_lora_paths: dict[str, str] = {}
        self.student_samplers: dict[str, StateSampler] = {}
        self._multi_teacher_hint_counters: dict[str, int] = {}

        # ALE-Bench full-corpus evaluation state
        self._ale_bench_eval_enabled = getattr(config, "ale_bench_eval_enabled", False)
        self._ale_bench_eval_output_dir = ""
        self._ale_bench_eval_problem_ids: list[str] = []
        self._ale_bench_eval_envs: dict[str, BaseEnv] = {}

        max_head_offpolicyness = getattr(config.rollout, "max_head_offpolicyness", 2)
        max_version_history = max_head_offpolicyness + 1

        # Detect sync mode and disable lazy sampling automatically
        self.is_sync_mode = max_head_offpolicyness == 0
        if self.is_sync_mode and config.sampler.lazy_puct_sampling:
            logger.info(
                "[Distill] Sync mode detected (max_head_offpolicyness=0). "
                "Disabling lazy_puct_sampling."
            )
            config.sampler.lazy_puct_sampling = False

        if self.is_multi_teacher:
            self._init_multi_teacher(config, max_version_history)
        else:
            self._init_single_teacher(config, max_version_history)

        # =====================================================================
        # Create environment (needed for privileged prompt construction)
        # =====================================================================
        if self.is_multi_teacher:
            # Multi-teacher mode already created per-problem envs; keep the first
            # problem's env as the default fallback.
            pass
        else:
            self.env = create_env_from_config(config)

        # =====================================================================
        # Load milestone hints for whole-path teacher distillation (optional)
        # =====================================================================
        self._milestone_hints_data = None
        if getattr(config, "milestone_hints", None):
            hints_path = config.milestone_hints
            if os.path.isfile(hints_path):
                with open(hints_path) as f:
                    self._milestone_hints_data = json.load(f)
                logger.info(
                    f"[MilestoneHints] Loaded from {hints_path}: "
                    f"{len(self._milestone_hints_data.get('paths', []))} paths"
                )
            else:
                logger.warning(f"[MilestoneHints] Path not found: {hints_path}")

        # =====================================================================
        # Create student actor (with LoRA)
        # =====================================================================
        # Propagate standard GRPO settings from top-level config to actor config
        config.actor.use_standard_grpo = getattr(config, "use_standard_grpo", False)
        config.actor.best_reward_anchor = getattr(config, "best_reward_anchor", False)
        # Set group_size for correct GRPO grouping (one group per prompt)
        config.actor.group_size = config.gconfig.n_samples
        self.actor = self._create_tttd_actor(config.actor)
        self.ref = None  # No ref model needed (kl_ctl=0)

        # Setup ALE-Bench eval envs now that self.actor is available.  Only the DP
        # head creates per-problem envs because eval runs on a single rank.
        if self._ale_bench_eval_enabled:
            self._setup_ale_bench_eval(config)

        # =====================================================================
        # Create teacher model (native AReaL path)
        # =====================================================================
        self.teacher = None
        if config.teacher is not None:
            if self.is_multi_teacher:
                # The shared teacher engine uses the base model with LoRA structure so
                # we can load each problem's adapter before computing teacher_logp.
                config.teacher.path = config.actor.path
                config.teacher.use_lora = True
                logger.info(
                    f"[MultiTeacher] Teacher engine using base model {config.teacher.path} "
                    f"with LoRA adapters for per-problem loading"
                )
            teacher_alloc = ModelAllocation.from_str(
                config.teacher.backend, name="teacher"
            )
            self.teacher = self._create_train_engine(config.teacher, teacher_alloc)

        # =====================================================================
        # =====================================================================
        # Create dataloaders using teacher sampler
        # =====================================================================
        if self.is_multi_teacher:
            # NOTE: The dataloader's default state_to_prompt_fn returns state.code or
            # state.observation, which is not the actual LLM prompt. workflow_v2
            # regenerates the prompt via env.get_prompt_distill(state), so the
            # distillation prompt remains stateless and consistent across problems.
            self.train_dataloader = create_multi_problem_tttd_dataloader(
                problem_samplers=self.student_samplers,
                rank=self.actor.data_parallel_rank,
                world_size=self.actor.data_parallel_world_size,
                batch_size=config.sampler.batch_size,
                lazy_sampling=config.sampler.lazy_puct_sampling,
            )
        else:
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
            logger.info(
                f"[EvalSetup] Saving base model (initial LoRA) to {self.base_eval_path}"
            )
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
        if self.is_multi_teacher:
            if is_single_controller() or self.actor.rank == 0:
                logger.info(
                    "[EvalSetup] Skipping single teacher eval save in multi-teacher mode; "
                    "per-problem adapters are configured separately"
                )
        else:
            if is_single_controller() or self.actor.rank == 0:
                logger.info(
                    f"[EvalSetup] Saving teacher model to {self.teacher_eval_path}"
                )
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
        max_steps = getattr(config, "max_steps", config.total_train_epochs)

        # Default: save every step. Can override via config.dynamic_metric_save_steps
        all_save_steps = getattr(
            config, "dynamic_metric_save_steps", list(range(0, max_steps))
        )
        start_step = 0
        if hasattr(self, "recover_info") and self.recover_info is not None:
            start_step = self.recover_info.last_step_info.next().global_step

        future_save_steps = [s for s in all_save_steps if start_step <= s < max_steps]

        rank_suffix = (
            f"_rank{self.actor.dp_rank}"
            if self.actor.data_parallel_world_size > 1
            else ""
        )
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
            filename=f"dynamic_metrics{rank_suffix}.pkl",
            checkpoint_filename=f"dynamic_metrics_checkpoint{rank_suffix}.pkl",
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

    def _init_single_teacher(self, config: TTTDDistillConfig, max_version_history: int):
        """Initialize the original single-teacher sampler pair."""
        # --- Teacher sampler: loaded from checkpoint, used for privileged OPD ---
        teacher_sampler_config = copy.deepcopy(config.sampler)
        if config.teacher_sampler_checkpoint:
            teacher_sampler_config.checkpoint_dir = config.teacher_sampler_checkpoint

        self.teacher_sampler = create_sampler_from_config(
            config=teacher_sampler_config,
            env_type=getattr(config.sampler, "env_type", "ac1"),
            max_version_history=max_version_history,
        )

        if config.teacher_sampler_checkpoint:
            latest_step = _find_latest_sampler_step(
                config.teacher_sampler_checkpoint,
                getattr(config.sampler, "type", "puct"),
            )
            if latest_step is not None:
                logger.info(
                    f"[TeacherSampler] Loading latest checkpoint at step {latest_step}"
                )
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
        if not getattr(student_sampler_config, "checkpoint_dir", None):
            student_sampler_config.checkpoint_dir = os.path.join(
                config.saver.fileroot,
                config.experiment_name,
                config.trial_name,
                "student_sampler",
            )

        self.sampler = create_sampler_from_config(
            config=student_sampler_config,
            env_type=getattr(config.sampler, "env_type", "ac1"),
            max_version_history=1,  # Student doesn't need version history
        )
        self.sampler.sampling_strategy = getattr(
            config.sampler, "sampling_strategy", "parent_pool"
        )
        logger.info(
            f"[StudentSampler] Fresh sampler with {len(self.sampler._states)} states, "
            f"strategy={self.sampler.sampling_strategy} (for student rollout)"
        )

        # --- Optionally inherit teacher sampler's state pool into student sampler ---
        if config.student_sampler_inherit_teacher_pool:
            self.sampler._states = copy.deepcopy(self.teacher_sampler._states)
            self.sampler._initial_states = copy.deepcopy(
                self.teacher_sampler._initial_states
            )
            self.sampler._n = copy.deepcopy(self.teacher_sampler._n)
            self.sampler._m = copy.deepcopy(self.teacher_sampler._m)
            self.sampler._T = self.teacher_sampler._T
            logger.info(
                f"[StudentSampler] Inherited teacher pool: {len(self.sampler._states)} states, "
                f"T={self.sampler._T}, n_entries={len(self.sampler._n)}, "
                f"m_entries={len(self.sampler._m)}"
            )

    def _init_multi_teacher(
        self,
        config: TTTDDistillConfig,
        max_version_history: int,
    ):
        """Initialize per-problem envs, samplers, and teacher LoRA paths.

        One teacher is trained per ALE-Bench problem.  The student sees a mixed
        batch with one group per problem; each group uses the problem's own
        PUCTSampler for privileged hints and the problem's own LoRA adapter for
        teacher log probabilities.
        """
        if config.sampler.env_type != "ale_bench":
            logger.warning(
                f"[MultiTeacher] Forcing env_type from '{config.sampler.env_type}' to 'ale_bench' for current version"
            )
            config.sampler.env_type = "ale_bench"

        n_problems = len(config.multi_teacher)
        # One group per problem per global step.
        config.sampler.batch_size = n_problems
        config.total_rollouts_per_step = n_problems * config.gconfig.n_samples
        logger.info(
            f"[MultiTeacher] Configuring mixed batch: {n_problems} problems x "
            f"{config.gconfig.n_samples} rollouts = {config.total_rollouts_per_step} total"
        )

        original_problem_id = getattr(config.sampler, "problem_id", "")

        for teacher_cfg in config.multi_teacher:
            if not isinstance(teacher_cfg, MultiTeacherConfig):
                raise ValueError(
                    f"multi_teacher entries must be MultiTeacherConfig, got {type(teacher_cfg)}"
                )
            problem_id = teacher_cfg.problem_id
            if not problem_id:
                raise ValueError("Every multi_teacher entry must have a problem_id")
            if not teacher_cfg.sampler_checkpoint:
                raise ValueError(f"sampler_checkpoint missing for problem {problem_id}")
            if not teacher_cfg.lora_path:
                raise ValueError(f"lora_path missing for problem {problem_id}")
            if not os.path.isdir(teacher_cfg.sampler_checkpoint):
                raise ValueError(
                    f"Teacher sampler checkpoint not found for {problem_id}: "
                    f"{teacher_cfg.sampler_checkpoint}"
                )
            if not os.path.isdir(teacher_cfg.lora_path):
                raise ValueError(
                    f"Teacher LoRA path not found for {problem_id}: {teacher_cfg.lora_path}"
                )

            logger.info(
                f"[MultiTeacher] Registering problem={problem_id}, "
                f"sampler={teacher_cfg.sampler_checkpoint}, lora={teacher_cfg.lora_path}"
            )

            # Per-problem env
            config.sampler.problem_id = problem_id
            env = create_env_from_config(config)
            self.problem_envs[problem_id] = env

            # Teacher sampler: loaded from checkpoint, never updated
            teacher_sampler_config = copy.deepcopy(config.sampler)
            teacher_sampler_config.checkpoint_dir = teacher_cfg.sampler_checkpoint
            teacher_sampler_config.problem_id = problem_id
            teacher_sampler_config.env_type = "ale_bench"
            teacher_sampler = create_sampler_from_config(
                config=teacher_sampler_config,
                env_type="ale_bench",
                max_version_history=max_version_history,
            )
            latest_step = _find_latest_sampler_step(
                teacher_cfg.sampler_checkpoint, getattr(config.sampler, "type", "puct")
            )
            if latest_step is not None:
                logger.info(
                    f"[MultiTeacher-{problem_id}] Loading sampler step {latest_step}"
                )
                teacher_sampler._load(latest_step)
                teacher_sampler._current_step = 0
            else:
                logger.warning(
                    f"[MultiTeacher-{problem_id}] No sampler checkpoint found; using fresh state"
                )
            teacher_sampler.sampling_strategy = config.teacher_sampler_strategy
            self.teacher_samplers[problem_id] = teacher_sampler
            self.teacher_lora_paths[problem_id] = teacher_cfg.lora_path
            self._multi_teacher_hint_counters[problem_id] = 0

            # Student sampler: fresh initial state, never updated.
            # In distillation the prompt is stateless (env.get_prompt_distill ignores
            # the state). We keep a minimal State object only because the dataloader
            # and workflow infrastructure expect a StateSampler/State.
            student_sampler_config = copy.deepcopy(config.sampler)
            student_sampler_config.checkpoint_dir = os.path.join(
                config.saver.fileroot,
                config.experiment_name,
                config.trial_name,
                "student_sampler",
                problem_id,
            )
            student_sampler_config.problem_id = problem_id
            student_sampler_config.env_type = "ale_bench"
            student_sampler = create_sampler_from_config(
                config=student_sampler_config,
                env_type="ale_bench",
                max_version_history=1,
            )
            student_sampler.sampling_strategy = getattr(
                config.sampler, "sampling_strategy", "parent_pool"
            )
            self.student_samplers[problem_id] = student_sampler
            self.problem_samplers[problem_id] = student_sampler

            logger.info(
                f"[MultiTeacher-{problem_id}] teacher_states={len(teacher_sampler._states)}, "
                f"student_states={len(student_sampler._states)}"
            )

        # Restore original problem_id so the rest of the config stays clean
        config.sampler.problem_id = original_problem_id

        # Use the first problem's env as the default fallback env
        first_problem = config.multi_teacher[0].problem_id
        self.env = self.problem_envs[first_problem]

        # Dummy single-teacher sampler for code paths that expect self.sampler
        self.sampler = next(iter(self.student_samplers.values()))
        self.teacher_sampler = None

    def _setup_ale_bench_eval(self, config: TTTDDistillConfig):
        """Discover all ALE-Bench problems and create per-problem envs for eval."""
        try:
            self._ale_bench_eval_problem_ids = list_ale_bench_problem_ids(
                lite_version=config.ale_bench_eval_lite_version
            )
        except Exception as e:
            logger.warning(
                f"[AleBenchEval] Failed to list ALE-Bench problems: {e}. Disabling eval."
            )
            self._ale_bench_eval_enabled = False
            return

        logger.info(
            f"[AleBenchEval] Discovered {len(self._ale_bench_eval_problem_ids)} problems "
            f"(lite={config.ale_bench_eval_lite_version})"
        )

        output_dir = config.ale_bench_eval_output_dir
        if not output_dir:
            output_dir = os.path.join(
                config.saver.fileroot,
                config.experiment_name,
                config.trial_name,
                "ale_bench_eval",
            )
        self._ale_bench_eval_output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Only global rank 0 needs per-problem envs now that eval runs on a single rank.
        if self.actor.rank != 0:
            logger.info("[AleBenchEval] Skipping env creation on non-zero ranks")
            return

        original_problem_id = getattr(config.sampler, "problem_id", "")
        for problem_id in self._ale_bench_eval_problem_ids:
            config.sampler.problem_id = problem_id
            config.sampler.env_type = "ale_bench"
            try:
                env = create_env_from_config(config)
                self._ale_bench_eval_envs[problem_id] = env
            except Exception as e:
                logger.warning(
                    f"[AleBenchEval] Failed to create env for {problem_id}: {e}"
                )
        config.sampler.problem_id = original_problem_id

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
                disk_kwargs.update(
                    {
                        "use_lora": config.actor.use_lora,
                        "lora_name": config.gconfig.lora_name,
                        "lora_int_id": 1,
                        "base_model_name": config.actor.path,
                    }
                )
            self.weight_update_meta = WeightUpdateMeta.from_disk(**disk_kwargs)
        elif config.actor.weight_update_mode == "xccl":
            if self.allocation_mode.train_backend == "megatron":
                self.weight_update_meta = WeightUpdateMeta.from_megatron_xccl(
                    self.allocation_mode
                )
            else:
                xccl_kwargs = {"gen_allocation": self.rollout_alloc}
                if config.actor.use_lora:
                    xccl_kwargs.update(
                        {
                            "use_lora": config.actor.use_lora,
                            "lora_name": config.gconfig.lora_name,
                            "lora_int_id": 1,
                            "base_model_name": config.actor.path,
                        }
                    )
                self.weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(**xccl_kwargs)
        else:
            raise ValueError(
                f"Invalid weight update mode: {config.actor.weight_update_mode}"
            )

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
        if hasattr(self.rollout, "dispatcher"):
            return self.rollout.dispatcher
        if hasattr(self.rollout, "_engine") and hasattr(
            self.rollout._engine, "workflow_executor"
        ):
            return self.rollout._engine.workflow_executor._dispatcher
        if hasattr(self.rollout, "workflow_executor"):
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
                            logger.info(
                                f"[Resume] Cleared {stale_inputs} stale pending inputs"
                            )

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
                    logger.info(
                        f"[Resume] Cleared {queue_cleared} items from async task queues"
                    )

                # 3. CRITICAL: Fix accepted count for staleness capacity calculation
                # Normal training: accepted = step * consumer_batch_size at each step start
                # Recovery: version = start_step, so accepted should be start_step * consumer_batch_size
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
                            logger.info(
                                f"[Resume] Fixed accepted count: {old_accepted} -> {expected_accepted} "
                                f"(expected after {completed_steps} completed steps, consumer_batch_size={consumer_bs})"
                            )
                    elif is_dp_head:
                        logger.info(
                            f"[Resume] accepted count is correct: {old_accepted}"
                        )

                    version = self.rollout.get_version()
                    max_staleness = sm.max_staleness
                    capacity = (
                        max_staleness + version + 1
                    ) * consumer_bs - sm.rollout_stat.accepted
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
                            logger.info(
                                f"[Resume] Cleared {stale_results} stale pending results"
                            )

            # 5. Clear data_generator cache
            if workflow_executor is not None and hasattr(
                workflow_executor, "data_generator"
            ):
                delattr(workflow_executor, "data_generator")
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
        if hasattr(workflow_executor, "workflow"):
            workflow = workflow_executor.workflow
        elif hasattr(workflow_executor, "_workflow"):
            workflow = workflow_executor._workflow

        if workflow is None:
            return

        cleared_items = []

        if hasattr(workflow, "_current_batch_parent_ids"):
            stale_count = len(workflow._current_batch_parent_ids)
            if stale_count > 0:
                workflow._current_batch_parent_ids.clear()
                workflow._expected_batch_size = 0
                workflow._expected_n_samples = 0
                cleared_items.append(f"scheme1_batch({stale_count})")

        if hasattr(workflow, "_staleness_tracker"):
            stale_count = len(workflow._staleness_tracker)
            if stale_count > 0:
                workflow._staleness_tracker.clear()
                cleared_items.append(f"staleness_tracker({stale_count})")

        if hasattr(workflow, "_parent_episodes"):
            stale_count = len(workflow._parent_episodes)
            if stale_count > 0:
                workflow._parent_episodes.clear()
                cleared_items.append(f"parent_episodes({stale_count})")

        if cleared_items and is_dp_head:
            logger.info(f"[Resume] Cleared workflow state: {', '.join(cleared_items)}")

    def _get_workflow_executor(self):
        """Get workflow executor from rollout."""
        if hasattr(self.rollout, "_engine") and hasattr(
            self.rollout._engine, "workflow_executor"
        ):
            return self.rollout._engine.workflow_executor
        if hasattr(self.rollout, "workflow_executor"):
            return self.rollout.workflow_executor
        return None

    def _clear_workflow_cache(self):
        """Clear workflow executor cache so a new workflow can be used."""
        workflow_executor = self._get_workflow_executor()
        if workflow_executor is not None and hasattr(
            workflow_executor, "data_generator"
        ):
            delattr(workflow_executor, "data_generator")
            logger.info("[Distill] Cleared workflow executor cache")

        if hasattr(self.rollout, "data_generator"):
            delattr(self.rollout, "data_generator")
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

            # Manually flat-concat _student_prompts because all_gather_tensor_container
            # turns lists into tuples via list(zip(*data)), and concat_padded_tensors
            # only flat-concats lists (tuples are treated as scalars and only the
            # first element is kept).
            student_prompts: list[str] = []
            breakthrough_parents: list[Any] = []
            breakthrough_children: list[Any] = []
            problem_ids: list[str] = []
            for d in rollout_batch:
                sp = d.pop("_student_prompts", None)
                if isinstance(sp, (list, tuple)):
                    student_prompts.extend(sp)
                elif sp is not None:
                    student_prompts.append(sp)

                bp = d.pop("_breakthrough_parent", None)
                if isinstance(bp, (list, tuple)):
                    breakthrough_parents.extend(bp)
                elif bp is not None:
                    breakthrough_parents.append(bp)

                bc = d.pop("_breakthrough_child", None)
                if isinstance(bc, (list, tuple)):
                    breakthrough_children.extend(bc)
                elif bc is not None:
                    breakthrough_children.append(bc)

                pids = d.pop("_problem_ids", None)
                if isinstance(pids, (list, tuple)):
                    problem_ids.extend(pids)
                elif pids is not None:
                    problem_ids.append(pids)

            batched, _meta = concat_batch(rollout_batch)
            if student_prompts:
                batched["_student_prompts"] = student_prompts
            if breakthrough_parents:
                batched["_breakthrough_parent"] = breakthrough_parents
            if breakthrough_children:
                batched["_breakthrough_child"] = breakthrough_children
            if problem_ids:
                batched["_problem_ids"] = problem_ids
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
            if getattr(self, "recover_info", None) is not None
            else 0
        )
        if start_step > 0:
            logger.info(f"[Distill] Resuming from step {start_step} (recovered)")

        # Update workflow kwargs with sampler and DP info
        if workflow_kwargs is not None:
            self._workflow_kwargs = workflow_kwargs.copy()
            # Sync lazy_sampling to the potentially modified config value
            # (TTTDDistillTrainer.__init__ may disable lazy_puct_sampling in sync mode)
            self._workflow_kwargs["lazy_sampling"] = config.sampler.lazy_puct_sampling
            if (
                config.sampler.lazy_puct_sampling
                and "sampler" not in self._workflow_kwargs
            ):
                self._workflow_kwargs["sampler"] = self.sampler
                self._workflow_kwargs["dp_rank"] = self.actor.dp_rank
                self._workflow_kwargs["dp_world_size"] = (
                    self.actor.data_parallel_world_size
                )
            # Enable from-scratch prompt mode for student rollout if configured
            if getattr(config, "distill_from_scratch", False):
                self._workflow_kwargs["distill_mode"] = True
                logger.info(
                    "[Distill] Student rollout using from-scratch prompts (distill_mode=True)"
                )

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
        # Baseline ALE-Bench evaluation before training
        # =====================================================================
        if self._ale_bench_eval_enabled and config.ale_bench_eval_before_train:
            self._run_ale_bench_eval(global_step=start_step)

        # =====================================================================
        # Phase 1: Distillation steps (no verification)
        # =====================================================================
        for global_step in range(start_step, config.max_steps):
            logger.info(f"[Distill][Step {global_step}] Starting distill step")

            step_start_time = time.perf_counter()

            # Create distill workflow (dummy or real reward)
            distill_kwargs = self._workflow_kwargs.copy()
            if config.use_real_reward:
                distill_kwargs["reward_fn"] = tttd_reward_fn
                # Use default max_reward_workers from workflow_kwargs
            else:
                distill_kwargs["reward_fn"] = dummy_reward_fn
                distill_kwargs["max_reward_workers"] = 1  # Dummy reward is fast
            distill_workflow = workflow(**distill_kwargs)

            # Set current version for tracking
            if hasattr(distill_workflow, "set_current_version"):
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
                local_max = torch.tensor(
                    [step_max_reward], dtype=torch.float32, device=self.actor.device
                )
                local_min = torch.tensor(
                    [step_min_reward], dtype=torch.float32, device=self.actor.device
                )
                local_sum = torch.tensor(
                    [step_rewards.sum()], dtype=torch.float32, device=self.actor.device
                )
                local_count = torch.tensor(
                    [len(step_rewards)], dtype=torch.float32, device=self.actor.device
                )

                dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
                dist.all_reduce(local_min, op=dist.ReduceOp.MIN)
                dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

                step_max_reward = local_max.item()
                step_min_reward = local_min.item()
                step_mean_reward = (
                    (local_sum / local_count).item() if local_count.item() > 0 else 0.0
                )
                global_rollouts = int(local_count.item())
            else:
                global_rollouts = local_rollouts

            logger.info(
                f"[Distill][Step {global_step}] Rollouts: {local_rollouts} (global: {global_rollouts}), "
                f"Reward: max={step_max_reward:.4f}, mean={step_mean_reward:.4f}, min={step_min_reward:.4f}"
            )

            # Compute teacher logp
            if self.teacher is not None:
                torch.cuda.empty_cache()
                with torch.no_grad():
                    if config.use_privileged_teacher_logp:
                        (
                            teacher_logps,
                            teacher_input_ids,
                            teacher_attention_mask,
                            teacher_loss_mask,
                        ) = self._compute_privileged_teacher_logp(rollout_batch)
                        rollout_batch["privileged_teacher_input_ids"] = (
                            teacher_input_ids
                        )
                        rollout_batch["privileged_teacher_attention_mask"] = (
                            teacher_attention_mask
                        )
                        rollout_batch["privileged_teacher_loss_mask"] = (
                            teacher_loss_mask
                        )
                    else:
                        # Teacher and student see the same prompts (no privileged OPD)
                        teacher_logps_list = self.teacher.compute_logp([rollout_batch])
                        teacher_logps = teacher_logps_list[0]
                rollout_batch["teacher_logp"] = teacher_logps
                rollout_batch["rl_loss_weight"] = self.config.teacher.rl_loss_weight
                rollout_batch["distill_loss_weight"] = (
                    self.config.teacher.distill_loss_weight
                )

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
            is_first_step = global_step == start_step
            is_last_step = global_step == config.max_steps - 1
            is_middle_5th = (global_step + 1) % 5 == 0
            if self.teacher is not None and (
                is_first_step or is_last_step or is_middle_5th
            ):
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
                count_t = torch.tensor(
                    [float(count)], dtype=torch.float32, device=self.actor.device
                )
                dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
                total_count = int(count_t.item())

                for key in list(dynamic_metrics.keys()):
                    val = torch.tensor(
                        [dynamic_metrics[key]],
                        dtype=torch.float32,
                        device=self.actor.device,
                    )
                    dist.all_reduce(val, op=dist.ReduceOp.SUM)
                    dynamic_metrics[key] = (
                        (val / total_count).item() if total_count > 0 else 0.0
                    )

            # Record dynamic metrics to history logger (saved as JSON)
            if (
                hasattr(self, "dynamic_metrics_logger")
                and self.dynamic_metrics_logger is not None
            ):
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
                    logger.warning(
                        f"[Distill][Step {global_step}] Failed to record dynamic metrics to logger: {e}"
                    )

            # Export training stats
            stats = self.actor.export_stats()
            metrics = {
                "rollout/count": global_rollouts,
                "reward/max": step_max_reward,
                "reward/mean": step_mean_reward,
                "reward/min": step_min_reward,
                "train/actor_loss": stats.get("ppo_actor/update/actor_loss/avg", 0.0),
                "train/approx_kl": stats.get("ppo_actor/update/approx_kl/avg", 0.0),
                "train/entropy": stats.get("ppo_actor/update/entropy/avg", 0.0),
                "train/grad_norm": stats.get("ppo_actor/update/grad_norm", 0.0),
                "train/lr": stats.get("ppo_actor/update/lr", 0.0),
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
                    if "distill/entropy_gap_privileged" in metrics:
                        log_msg += f", EntGapP: {metrics.get('distill/entropy_gap_privileged', 0.0):.4f}"
                logger.info(log_msg)

            # Update weights and save
            logger.info(
                f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] Before rollout.pause"
            )
            self.rollout.pause()
            logger.info(
                f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] After rollout.pause"
            )

            logger.info(
                f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] Before actor.update_weights"
            )
            new_version = global_step + 1
            versioned_meta = self.weight_update_meta.with_version(new_version)
            logger.info(
                f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] Using versioned_meta version={versioned_meta.version}"
            )
            self.actor.update_weights(versioned_meta)
            logger.info(
                f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] After actor.update_weights"
            )

            self.actor.set_version(new_version)
            self.rollout.set_version(new_version)
            logger.info(
                f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] After set_version"
            )

            logger.info(
                f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] Before _save_hf"
            )
            self._save_hf(
                epoch=global_step, epoch_step=global_step, global_step=global_step
            )
            logger.info(
                f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] After _save_hf"
            )

            logger.info(
                f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] Before _save_recover_checkpoint"
            )
            self._save_recover_checkpoint(
                epoch=global_step, epoch_step=global_step, global_step=global_step
            )
            logger.info(
                f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] After _save_recover_checkpoint"
            )

            # Save dynamic metrics checkpoint after each step (for crash recovery)
            if (
                hasattr(self, "dynamic_metrics_logger")
                and self.dynamic_metrics_logger is not None
            ):
                try:
                    self.dynamic_metrics_logger.save_checkpoint()
                except Exception as e:
                    logger.warning(
                        f"[Distill][Step {global_step}] Failed to save dynamic metrics checkpoint: {e}"
                    )

            logger.info(
                f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] Before dist.barrier"
            )
            dist.barrier(group=self.actor.cpu_group)
            logger.info(
                f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] After dist.barrier"
            )

            logger.info(
                f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] Before current_platform.synchronize"
            )
            current_platform.synchronize()
            logger.info(
                f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] After current_platform.synchronize"
            )

            logger.info(
                f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] Before rollout.resume"
            )
            self.rollout.resume()
            logger.info(
                f"[HANG-DEBUG][Step {global_step}][Rank {dist.get_rank()}] After rollout.resume"
            )

            # Periodic ALE-Bench full-corpus evaluation
            if self._ale_bench_eval_enabled and (
                (global_step + 1) % config.ale_bench_eval_freq_steps == 0
                or global_step == config.max_steps - 1
            ):
                self._run_ale_bench_eval(global_step=global_step + 1)

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
            logger.info(
                f"[Distill] Saved student checkpoint to {self.student_eval_path}"
            )

        # Save dynamic metrics history (JSON + PKL)
        if (
            hasattr(self, "dynamic_metrics_logger")
            and self.dynamic_metrics_logger is not None
        ):
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
            processor=getattr(self, "processor", None),
        )

    def _save_recover_checkpoint(self, epoch: int, epoch_step: int, global_step: int):
        """Override parent _save_recover_checkpoint to remove extra barrier."""

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
            processor=getattr(self, "processor", None),
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

    def _hint_code_fence(self) -> str:
        """Return the Markdown code-fence language for hint code blocks."""
        language = getattr(self.env, "code_language", "cpp20").lower()
        if language.startswith("cpp") or language.startswith("c++"):
            return "cpp"
        if language.startswith("py"):
            return "python"
        return language

    def _build_hint(self, privileged_state) -> str:
        """Build a short hint from privileged state for teacher prompt."""
        fence = self._hint_code_fence()
        hint_parts = []
        if privileged_state.code and privileged_state.code.strip():
            hint_parts.append(f"```{fence}\n{privileged_state.code.strip()}\n```")
        raw_score = getattr(privileged_state, "raw_score", None)
        if raw_score is None and privileged_state.value is not None:
            # AC1 uses value = -raw_score; ALE-Bench stores raw_score directly.
            raw_score = -privileged_state.value
        if raw_score is not None:
            # AleBenchState stores the average raw score per public case.
            score_label = (
                "average raw score per public case"
                if hasattr(privileged_state, "parent_raw_scores")
                else "score"
            )
            hint_parts.append(
                f"This reference approach achieves a {score_label} of {raw_score:.6f}."
            )
        hint_text = "\n".join(hint_parts)
        return (
            f"\n\n[Reference Approach]\n{hint_text}\n\n"
            "Study the reference approach above. Reason about why it succeeds on "
            "this problem, identify the underlying principles, and try to generalize "
            "them to related problems or variants. Then generate your own independent "
            "C++20 solution. Do not copy it verbatim; write a fresh implementation "
            "based on your generalized understanding.\n"
        )

    def _build_best_worst_hint(self, best_states, worst_states) -> str:
        """Build a combined hint showing strong and weak reference approaches.

        Accepts either single states or lists of states, allowing few-shot
        best/worst demonstrations when `multi_teacher_hint_k > 1`.
        """
        fence = self._hint_code_fence()
        hint_parts = []

        def _display_score(state):
            raw = getattr(state, "raw_score", None)
            if raw is None and state.value is not None:
                raw = -state.value
            label = (
                "average raw score per public case"
                if hasattr(state, "parent_raw_scores")
                else "score"
            )
            return raw, label

        # Normalize to lists for uniform handling.
        if not isinstance(best_states, (list, tuple)):
            best_states = [best_states]
        if not isinstance(worst_states, (list, tuple)):
            worst_states = [worst_states]

        if best_states:
            hint_parts.append(
                f"Below are {len(best_states)} strong reference approach(es):"
            )
            for idx, state in enumerate(best_states, start=1):
                if state.code and state.code.strip():
                    hint_parts.append(f"Strong example {idx}:")
                    hint_parts.append(f"```{fence}\n{state.code.strip()}\n```")
                score, score_label = _display_score(state)
                if score is not None:
                    hint_parts.append(
                        f"This approach achieves a {score_label} of {score:.6f}."
                    )
                hint_parts.append("")

        if worst_states:
            hint_parts.append(
                f"\nBelow are {len(worst_states)} weaker but still valid reference approach(es):"
            )
            for idx, state in enumerate(worst_states, start=1):
                if state.code and state.code.strip():
                    hint_parts.append(f"Weak example {idx}:")
                    hint_parts.append(f"```{fence}\n{state.code.strip()}\n```")
                score, score_label = _display_score(state)
                if score is not None:
                    hint_parts.append(
                        f"This approach achieves a {score_label} of {score:.6f}."
                    )
                hint_parts.append("")

        hint_text = "\n".join(hint_parts).strip()
        return (
            f"\n\n[Reference Approaches]\n{hint_text}\n\n"
            "Study all the approaches above. Understand why the strong approaches succeed "
            "and why the weaker ones perform less well. Identify the underlying principles, "
            "and try to generalize them. Then generate your own independent C++20 solution "
            "based on your understanding.\n"
        )

    def _build_breakthrough_hint(self, parent_state, child_state) -> str:
        """Build a breakthrough hint showing parent -> child improvement.

        This gives the teacher privileged access to a "breakthrough transition"
        — a pair where the child significantly outperforms its parent.
        The teacher can then better evaluate whether the student's code
        captures the key algorithmic insight behind the breakthrough.
        """
        fence = self._hint_code_fence()
        hint_parts = []

        def _display_score(state):
            raw = getattr(state, "raw_score", None)
            if raw is None and state.value is not None:
                # AC1 uses value = -raw_score
                raw = -state.value
            label = (
                "average raw score per public case"
                if hasattr(state, "parent_raw_scores")
                else "score"
            )
            return raw, label

        # Parent info (the starting point)
        parent_display, parent_label = _display_score(parent_state)
        if parent_display is not None:
            hint_parts.append(
                f"A previous approach achieved {parent_label} {parent_display:.6f}."
            )

        # Child info (the breakthrough)
        child_display, child_label = _display_score(child_state)
        if child_display is not None:
            improvement = child_display - (
                parent_display if parent_display is not None else 0
            )
            hint_parts.append(
                f"A breakthrough approach achieves {child_label} {child_display:.6f} "
                f"(improvement: +{improvement:.6f})."
            )

        # Child code — the key algorithmic insight
        if child_state.code and child_state.code.strip():
            code = child_state.code.strip()
            # Truncate very long code to avoid blowing up the prompt
            max_code_len = 4000
            if len(code) > max_code_len:
                code = code[:max_code_len] + "\n... (truncated)"
            hint_parts.append(f"```{fence}\n{code}\n```")

        # Construction metadata
        if hasattr(child_state, "construction") and child_state.construction:
            hint_parts.append(
                f"This breakthrough uses a sequence of length {len(child_state.construction)}."
            )

        hint_text = "\n".join(hint_parts)
        return (
            f"\n\n[Breakthrough Transition]\n{hint_text}\n\n"
            "Study the transition above. Understand why the second approach "
            "succeeds compared to the first, identify the underlying principles, "
            "and try to generalize them. Then generate your own independent "
            "C++20 solution based on that generalized insight.\n"
        )

    def _build_milestone_hint(self, state) -> str:
        """Build a whole-path milestone hint from pre-extracted PUCT tree paths.

        Uses deterministic round-robin based on state id to select a path,
        then returns the full milestone sequence (baseline → best leaf).
        This gives the teacher privileged access to the *evolution* of search
        strategies, not just a single good state.
        """
        if self._milestone_hints_data is None:
            return ""
        paths = self._milestone_hints_data.get("paths", [])
        if not paths:
            return ""
        # Rotate through all paths to ensure every path is seen by teacher.
        # Relying on state-id hashing is bad because teacher_sampler.sample_states()
        # is deterministic and only covers a small subset of states, causing
        # some paths to never be selected.
        idx = getattr(self, "_milestone_hint_counter", 0) % len(paths)
        self._milestone_hint_counter = idx + 1
        path = paths[idx]
        milestones = path.get("milestones", [])
        if not milestones:
            return ""
        fence = self._hint_code_fence()
        lines = ["\n=== Strategy Evolution Hints ==="]
        lines.append(
            "Below are key phases discovered during search. "
            "Study each phase, reason about why the later phases succeed, "
            "identify the underlying principles, and try to generalize them. "
            "Then generate your own independent C++20 solution.\n"
        )
        for i, ms in enumerate(milestones):
            phase_label = (
                ["Baseline", "Phase 1", "Phase 2", "Phase 3"][i]
                if i < 4
                else f"Phase {i}"
            )
            value = ms.get("value")
            value_str = f"{value:.4f}" if isinstance(value, (int, float)) else "N/A"
            lines.append(f"--- {phase_label} (value={value_str}) ---")
            code = ms.get("code", "")
            if code:
                lines.append(f"```{fence}\n{code}\n```")
            lines.append("")
        lines.append("=== End Hints ===\n")
        return "\n".join(lines)

    def _load_peft_lora_adapter(self, engine, path: str):
        """Load a PEFT LoRA adapter checkpoint into the FSDP-wrapped engine.

        AReaL does not expose a dedicated ``load_adapter`` API for LoRA-only
        checkpoints, so we do it manually with the correct key mapping:

        1. PEFT ``save_pretrained`` strips the ``base_model.model.`` prefix.
        2. PEFT ``save_pretrained`` strips the ``.default`` adapter suffix.
        3. FSDP2 ``set_model_state_dict`` expects the full param names as they
           appear in the model, i.e. ``base_model.model....lora_A.default.weight``.
        """
        from safetensors.torch import load_file
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )

        adapter_path = os.path.join(path, "adapter_model.safetensors")
        if not os.path.isfile(adapter_path):
            raise ValueError(
                f"LoRA adapter not found at {adapter_path}. "
                f"Expected a PEFT checkpoint with adapter_model.safetensors."
            )

        logger.info(f"[LoadAdapter] Loading LoRA adapter from {path}")

        if dist.get_rank() == 0:
            raw_state = load_file(adapter_path)
            fixed_state = {}
            for k, v in raw_state.items():
                if not k.startswith("base_model.model."):
                    k = f"base_model.model.{k}"
                if ".lora_A.weight" in k:
                    k = k.replace(".lora_A.weight", ".lora_A.default.weight")
                elif ".lora_B.weight" in k:
                    k = k.replace(".lora_B.weight", ".lora_B.default.weight")
                fixed_state[k] = v
        else:
            fixed_state = {}

        options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=False,
            broadcast_from_rank0=True,
            strict=False,
        )
        set_model_state_dict(engine.model, fixed_state, options=options)

        if dist.get_rank() == 0:
            expected_keys = set(fixed_state.keys())
            model_keys = set(name for name, _ in engine.model.named_parameters())
            matched = expected_keys & model_keys
            unmatched = expected_keys - model_keys
            logger.info(
                f"[LoadAdapter] Matched {len(matched)}/{len(expected_keys)} keys "
                f"into teacher model"
            )
            if unmatched:
                logger.warning(
                    f"[LoadAdapter] {len(unmatched)} keys could not be matched: "
                    f"{list(unmatched)[:10]}"
                )
        logger.info("[LoadAdapter] LoRA adapter loaded successfully")

    def _sample_hint_for_problem(
        self,
        problem_id: str,
        teacher_sampler,
    ) -> tuple[str, str] | None:
        """Return (hint_type, hint_text) for one problem, cycling hint pool.

        Combined modes (``best_worst_combined`` and ``diverse_best_worst``)
        always return a single hint containing both best and worst examples.
        The counter only cycles when the selected mode yields multiple
        standalone hints, e.g. ``best`` or ``diverse_best`` with ``k > 1``.
        """
        mode = self.config.multi_teacher_hint_mode
        min_improvement = self.config.multi_teacher_hint_min_improvement
        deterministic = getattr(self.config, "multi_teacher_hint_deterministic", False)
        hint_k = getattr(self.config, "multi_teacher_hint_k", 1)
        hints = teacher_sampler.get_hint_states(
            mode=mode,
            k=hint_k,
            min_improvement=min_improvement,
            deterministic=deterministic,
        )
        if not hints:
            return None

        counter = self._multi_teacher_hint_counters.get(problem_id, 0)
        hint = hints[counter % len(hints)]
        self._multi_teacher_hint_counters[problem_id] = counter + 1

        label, payload = hint
        if (
            label == "breakthrough"
            and isinstance(payload, (tuple, list))
            and len(payload) == 2
        ):
            return (label, self._build_breakthrough_hint(payload[0], payload[1]))
        if (
            label in ("best_worst_combined", "diverse_best_worst")
            and isinstance(payload, (tuple, list))
            and len(payload) == 2
        ):
            return (label, self._build_best_worst_hint(payload[0], payload[1]))
        return (label, self._build_hint(payload))

    def _compute_privileged_teacher_logp(
        self, rollout_batch: dict[str, Any]
    ) -> torch.Tensor:
        """Compute teacher logp with hint-based privileged prompts.

        Reads student prompts from the rollout batch (_student_prompts field)
        so the ordering is correct regardless of async execution or DP
        redistribution.  Appends a [Hint] with privileged information
        (better state code/value) to each student prompt.

        Parameters
        ----------
        rollout_batch : dict
            Standard rollout batch with input_ids, loss_mask, attention_mask,
            plus _student_prompts (list[str]) attached by the workflow.

        Returns
        -------
        aligned_teacher_logp : torch.Tensor
            Shape [batch_size, student_seq_len].
        """
        if self.is_multi_teacher:
            return self._compute_privileged_teacher_logp_multi(rollout_batch)

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
                f"[PrivilegedOPD] batch_size ({batch_size}) not divisible by "
                f"group_size ({group_size}). Using sample-level."
            )
            local_num_groups = batch_size
            group_size = 1

        dp_world_size = self.actor.data_parallel_world_size
        dp_rank = self.actor.data_parallel_rank
        global_num_groups = local_num_groups * dp_world_size

        # ------------------------------------------------------------------
        # 1. Check for Breakthrough-Aware OPD data from workflow
        # ------------------------------------------------------------------
        use_breakthrough_opd = getattr(self.config, "use_breakthrough_opd", False)
        breakthrough_parents = rollout_batch.get("_breakthrough_parent", [])
        breakthrough_children = rollout_batch.get("_breakthrough_child", [])
        has_breakthrough_data = (
            use_breakthrough_opd
            and len(breakthrough_parents) > 0
            and len(breakthrough_children) > 0
        )

        if not has_breakthrough_data:
            # Standard OPD: sample privileged states directly
            all_privileged_states = self.teacher_sampler.sample_states(
                global_num_groups
            )
            start_idx = dp_rank * local_num_groups
            privileged_states = all_privileged_states[
                start_idx : start_idx + local_num_groups
            ]

            for i, state in enumerate(privileged_states):
                logger.info(
                    f"[PrivilegedOPD] Rank {dp_rank} priv_state {i}/{local_num_groups} "
                    f"value={state.value:.4f}, id={state.id[:8] if hasattr(state.id, '__len__') and len(state.id) > 8 else state.id}"
                )

            # Expand privileged states to match rollouts (group_size copies per group)
            privileged_states_expanded = []
            for state in privileged_states:
                privileged_states_expanded.extend([state] * group_size)
        else:
            # Breakthrough-Aware OPD: parent/child pairs come from workflow (aligned)
            logger.info(
                f"[BreakthroughOPD] Rank {dp_rank} using aligned breakthrough data: "
                f"parents={len(breakthrough_parents)} children={len(breakthrough_children)} "
                f"batch_size={batch_size}"
            )
            # Validate length alignment
            if (
                len(breakthrough_parents) != batch_size
                or len(breakthrough_children) != batch_size
            ):
                logger.warning(
                    f"[BreakthroughOPD] LENGTH_MISMATCH! parents={len(breakthrough_parents)} "
                    f"children={len(breakthrough_children)} batch_size={batch_size}"
                )
            # Log first few items for verification (with IDs for cross-reference)
            for i in range(min(5, batch_size)):
                parent = (
                    breakthrough_parents[i] if i < len(breakthrough_parents) else None
                )
                child = (
                    breakthrough_children[i] if i < len(breakthrough_children) else None
                )
                if parent is not None and child is not None:
                    improvement = (child.value if child.value is not None else 0) - (
                        parent.value if parent.value is not None else 0
                    )
                    logger.info(
                        f"[BreakthroughOPD] item {i}: parent_id={parent.id[:8]} "
                        f"parent_value={parent.value:.4f} child_id={child.id[:8]} "
                        f"child_value={child.value:.4f} improvement={improvement:.6f}"
                    )
                elif parent is not None:
                    logger.info(
                        f"[BreakthroughOPD] item {i}: parent_id={parent.id[:8]} "
                        f"parent_value={parent.value:.4f} child=None"
                    )
            privileged_states_expanded = None

        # ------------------------------------------------------------------
        # 2. Build teacher prompts from student prompts + hint
        # ------------------------------------------------------------------
        student_prompts = rollout_batch.get("_student_prompts", [])
        if len(student_prompts) != batch_size:
            logger.warning(
                f"[PrivilegedOPD] _student_prompts length ({len(student_prompts)}) != "
                f"batch_size ({batch_size}). Using available prompts cyclically."
            )

        # Determine prompt mode
        prompt_mode = getattr(self.config, "privileged_prompt_mode", "continuation")

        teacher_prompt_ids_list = []
        for i in range(batch_size):
            if has_breakthrough_data:
                # Breakthrough-Aware OPD: aligned parent/child from workflow
                parent_state = (
                    breakthrough_parents[i] if i < len(breakthrough_parents) else None
                )
                child_state = (
                    breakthrough_children[i] if i < len(breakthrough_children) else None
                )
                student_prompt = (
                    student_prompts[i % len(student_prompts)] if student_prompts else ""
                )
                if not student_prompt:
                    logger.warning(
                        f"[BreakthroughOPD] Empty student prompt for item {i}"
                    )

                if parent_state is not None and child_state is not None:
                    hint = self._build_breakthrough_hint(parent_state, child_state)
                elif child_state is not None:
                    # Fallback: no parent, just use child as regular hint
                    hint = self._build_hint(child_state)
                else:
                    # No breakthrough data for this item, fallback to standard
                    hint = ""
                teacher_prompt = student_prompt + hint
            else:
                # Standard OPD
                priv_state = privileged_states_expanded[i]

                if prompt_mode == "hint":
                    # Hint mode: student prompt + [Hint] privileged info
                    student_prompt = (
                        student_prompts[i % len(student_prompts)]
                        if student_prompts
                        else ""
                    )
                    if not student_prompt:
                        logger.warning(
                            f"[PrivilegedOPD] Empty student prompt for item {i}"
                        )
                    # Whole-path milestone hint takes precedence if available
                    if self._milestone_hints_data is not None:
                        hint = self._build_milestone_hint(priv_state)
                        if hint:
                            logger.info(
                                f"[MilestoneHint] item {i}: using milestone hint for state "
                                f"id={priv_state.id[:8] if hasattr(priv_state.id, '__len__') and len(priv_state.id) > 8 else priv_state.id}"
                            )
                    else:
                        hint = self._build_hint(priv_state)
                    teacher_prompt = student_prompt + hint
                else:
                    # Continuation mode (original Setup A behavior):
                    # Use env.get_prompt(privileged_state) directly
                    teacher_prompt = self.env.get_prompt(priv_state)

            messages = [{"role": "user", "content": teacher_prompt}]
            ids = list(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=getattr(self.config, "enable_thinking", False),
                )
            )
            teacher_prompt_ids_list.append(ids)

        # ------------------------------------------------------------------
        # 3. Extract student completions from rollout_batch
        # ------------------------------------------------------------------
        prompt_lens = (attention_mask & (loss_mask == 0)).sum(dim=1).cpu().numpy()
        comp_lens = loss_mask.sum(dim=1).cpu().numpy()

        # ------------------------------------------------------------------
        # 4. Build teacher sequences: teacher_prompt + student_completion
        # ------------------------------------------------------------------
        teacher_seqs = []
        teacher_loss_masks = []
        for i in range(batch_size):
            prompt_len = int(prompt_lens[i])
            comp_len = int(comp_lens[i])
            student_comp = (
                input_ids[i, prompt_len : prompt_len + comp_len].cpu().tolist()
            )
            teacher_prompt_ids = teacher_prompt_ids_list[i]
            teacher_prompt_len = len(teacher_prompt_ids)
            teacher_seq = teacher_prompt_ids + student_comp
            teacher_seqs.append(torch.tensor(teacher_seq, dtype=torch.int32))
            teacher_loss_masks.append(
                torch.tensor(
                    [0] * teacher_prompt_len + [1] * comp_len, dtype=torch.int32
                )
            )

        # Pad to max length
        max_teacher_len = max(len(seq) for seq in teacher_seqs)
        pad_id = self.tokenizer.pad_token_id or 0
        teacher_input_ids = torch.stack(
            [
                torch.nn.functional.pad(
                    seq, (0, max_teacher_len - len(seq)), value=pad_id
                )
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
                torch.nn.functional.pad(mask, (0, max_teacher_len - len(mask)), value=0)
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
        teacher_logps_full = teacher_logps_list[0]

        # ------------------------------------------------------------------
        # 6. Align completion logps back to student sequence positions
        # ------------------------------------------------------------------
        aligned_teacher_logp = torch.zeros(
            (batch_size, student_seqlen), dtype=torch.float32, device=device
        )
        for i in range(batch_size):
            prompt_len = int(prompt_lens[i])
            comp_len = int(comp_lens[i])
            teacher_prompt_len = len(teacher_prompt_ids_list[i])
            if comp_len == 0:
                continue
            # Teacher completion logps: [teacher_prompt_len-1, teacher_prompt_len+comp_len-1)
            t_start = teacher_prompt_len - 1
            t_end = teacher_prompt_len + comp_len - 1
            teacher_comp_logps = teacher_logps_full[i, t_start:t_end]

            # Student completion positions: [prompt_len-1, prompt_len+comp_len-1)
            s_start = prompt_len - 1
            s_end = prompt_len + comp_len - 1
            aligned_teacher_logp[i, s_start:s_end] = teacher_comp_logps.to(
                device=device, dtype=torch.float32
            )

        return (
            aligned_teacher_logp,
            teacher_input_ids,
            teacher_attention_mask,
            teacher_loss_mask,
        )

    def _compute_privileged_teacher_logp_multi(
        self, rollout_batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-problem teacher logp for multi-teacher distillation.

        Each problem's rollouts are identified by ``_problem_ids``.  For every
        problem we (1) load that problem's teacher LoRA adapter into the shared
        teacher engine, (2) sample a privileged hint from that problem's
        PUCTSampler, (3) build teacher prompts from the student prompts + hint,
        and (4) align the resulting completion logps back to the student
        sequence positions.
        """
        input_ids = rollout_batch["input_ids"]
        attention_mask = rollout_batch["attention_mask"]
        loss_mask = rollout_batch["loss_mask"]
        batch_size, student_seqlen = input_ids.shape
        device = input_ids.device

        problem_ids = rollout_batch.get("_problem_ids", [])
        student_prompts = rollout_batch.get("_student_prompts", [])

        if len(problem_ids) != batch_size:
            logger.warning(
                f"[MultiTeacher-Logp] _problem_ids length ({len(problem_ids)}) != "
                f"batch_size ({batch_size}). Falling back to contiguous group order."
            )
            group_size = self.config.gconfig.n_samples
            problem_keys = list(self.teacher_samplers.keys())
            problem_ids = [
                problem_keys[(i // group_size) % len(problem_keys)]
                for i in range(batch_size)
            ]

        # Group rollout indices by problem while preserving original order
        problem_to_indices: dict[str, list[int]] = {}
        for idx, pid in enumerate(problem_ids):
            problem_to_indices.setdefault(pid, []).append(idx)

        prompt_lens = (attention_mask & (loss_mask == 0)).sum(dim=1).cpu().numpy()
        comp_lens = loss_mask.sum(dim=1).cpu().numpy()

        aligned_teacher_logp = torch.zeros(
            (batch_size, student_seqlen), dtype=torch.float32, device=device
        )

        # Buffers for the privileged teacher inputs (used by dynamic metrics)
        teacher_input_ids_rows: dict[int, torch.Tensor] = {}
        teacher_attention_mask_rows: dict[int, torch.Tensor] = {}
        teacher_loss_mask_rows: dict[int, torch.Tensor] = {}
        global_max_teacher_len = 0

        enable_thinking = getattr(self.config, "enable_thinking", False)
        pad_id = self.tokenizer.pad_token_id or 0

        for problem_id, indices in problem_to_indices.items():
            if problem_id not in self.teacher_samplers:
                raise ValueError(
                    f"[MultiTeacher-Logp] Unknown problem_id in rollout batch: {problem_id}"
                )

            lora_path = self.teacher_lora_paths[problem_id]
            teacher_sampler = self.teacher_samplers[problem_id]

            # Load problem-specific teacher LoRA
            self._load_peft_lora_adapter(self.teacher, lora_path)

            # Sample a privileged hint for this problem
            hint_result = self._sample_hint_for_problem(problem_id, teacher_sampler)
            hint_type = hint_result[0] if hint_result else "none"
            hint_text = hint_result[1] if hint_result else ""

            logger.info(
                f"[MultiTeacher-Logp] problem={problem_id} rollouts={len(indices)} "
                f"hint_type={hint_type} lora={lora_path}"
            )

            # Build teacher sequences for every rollout of this problem
            problem_teacher_seqs: list[torch.Tensor] = []
            problem_teacher_loss_masks: list[torch.Tensor] = []
            problem_teacher_prompt_lens: list[int] = []
            problem_comp_lens: list[int] = []

            for idx in indices:
                prompt_len = int(prompt_lens[idx])
                comp_len = int(comp_lens[idx])
                student_comp = (
                    input_ids[idx, prompt_len : prompt_len + comp_len].cpu().tolist()
                )

                student_prompt = (
                    student_prompts[idx]
                    if idx < len(student_prompts) and student_prompts[idx]
                    else ""
                )
                teacher_prompt = student_prompt + hint_text

                messages = [{"role": "user", "content": teacher_prompt}]
                teacher_prompt_ids = list(
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=enable_thinking,
                    )
                )

                teacher_seq = teacher_prompt_ids + student_comp
                problem_teacher_seqs.append(
                    torch.tensor(teacher_seq, dtype=torch.int32)
                )
                problem_teacher_loss_masks.append(
                    torch.tensor(
                        [0] * len(teacher_prompt_ids) + [1] * comp_len,
                        dtype=torch.int32,
                    )
                )
                problem_teacher_prompt_lens.append(len(teacher_prompt_ids))
                problem_comp_lens.append(comp_len)

            if not problem_teacher_seqs:
                continue

            # Pad problem batch to its own max length
            problem_max_len = max(len(seq) for seq in problem_teacher_seqs)
            problem_input_ids = torch.stack(
                [
                    torch.nn.functional.pad(
                        seq, (0, problem_max_len - len(seq)), value=pad_id
                    )
                    for seq in problem_teacher_seqs
                ]
            )
            problem_attention_mask = torch.stack(
                [
                    torch.nn.functional.pad(
                        torch.ones(len(seq), dtype=torch.bool),
                        (0, problem_max_len - len(seq)),
                        value=False,
                    )
                    for seq in problem_teacher_seqs
                ]
            )
            problem_loss_mask = torch.stack(
                [
                    torch.nn.functional.pad(
                        mask, (0, problem_max_len - len(mask)), value=0
                    )
                    for mask in problem_teacher_loss_masks
                ]
            )

            problem_batch = {
                "input_ids": problem_input_ids,
                "attention_mask": problem_attention_mask,
                "loss_mask": problem_loss_mask,
            }

            with torch.no_grad():
                teacher_logps_list = self.teacher.compute_logp([problem_batch])
            teacher_logps_full = teacher_logps_list[0]

            # Align logps back to the original rollout positions
            for j, idx in enumerate(indices):
                comp_len = problem_comp_lens[j]
                if comp_len == 0:
                    continue
                teacher_prompt_len = problem_teacher_prompt_lens[j]
                t_start = teacher_prompt_len - 1
                t_end = teacher_prompt_len + comp_len - 1
                teacher_comp_logps = teacher_logps_full[j, t_start:t_end]

                prompt_len = int(prompt_lens[idx])
                s_start = prompt_len - 1
                s_end = prompt_len + comp_len - 1
                aligned_teacher_logp[idx, s_start:s_end] = teacher_comp_logps.to(
                    device=device, dtype=torch.float32
                )

                # Store privileged teacher inputs for dynamic metrics
                teacher_input_ids_rows[idx] = problem_input_ids[j]
                teacher_attention_mask_rows[idx] = problem_attention_mask[j]
                teacher_loss_mask_rows[idx] = problem_loss_mask[j]
                global_max_teacher_len = max(global_max_teacher_len, problem_max_len)

        # Stack privileged teacher inputs, padding each row to global max length
        if global_max_teacher_len > 0:
            teacher_input_ids = torch.stack(
                [
                    torch.nn.functional.pad(
                        teacher_input_ids_rows[i],
                        (0, global_max_teacher_len - len(teacher_input_ids_rows[i])),
                        value=pad_id,
                    )
                    for i in range(batch_size)
                ]
            )
            teacher_attention_mask = torch.stack(
                [
                    torch.nn.functional.pad(
                        teacher_attention_mask_rows[i],
                        (
                            0,
                            global_max_teacher_len
                            - len(teacher_attention_mask_rows[i]),
                        ),
                        value=False,
                    )
                    for i in range(batch_size)
                ]
            )
            teacher_loss_mask = torch.stack(
                [
                    torch.nn.functional.pad(
                        teacher_loss_mask_rows[i],
                        (0, global_max_teacher_len - len(teacher_loss_mask_rows[i])),
                        value=0,
                    )
                    for i in range(batch_size)
                ]
            )
        else:
            teacher_input_ids = torch.empty((batch_size, 0), dtype=torch.int32)
            teacher_attention_mask = torch.empty((batch_size, 0), dtype=torch.bool)
            teacher_loss_mask = torch.empty((batch_size, 0), dtype=torch.int32)

        return (
            aligned_teacher_logp,
            teacher_input_ids,
            teacher_attention_mask,
            teacher_loss_mask,
        )

    def _generate_ale_bench_eval_candidates(
        self, global_step: int
    ) -> dict[str, list[str]]:
        """Generate candidate responses for ALE-Bench evaluation.

        To avoid cross-rank synchronization issues while debugging, candidate
        generation runs only on the DP head.  vLLM internally load-balances
        requests across its engines, so this keeps the implementation simple.
        """
        config = self.config
        n_candidates = config.ale_bench_eval_n_candidates

        if self.actor.rank != 0:
            return {}

        dp_rank = getattr(self.actor, "data_parallel_rank", 0)
        dp_world_size = getattr(self.actor, "data_parallel_world_size", 1)
        eval_problem_ids = self._ale_bench_eval_problem_ids

        local_candidates: dict[str, list[str]] = {}
        if eval_problem_ids:
            # Use a separate generation config for eval so that the workflow's
            # expected_children matches the requested number of candidates.
            eval_gconfig = config.gconfig.new(n_samples=n_candidates)
            eval_workflow_kwargs = dict(
                env=self.env,
                problem_envs=self._ale_bench_eval_envs,
                gconfig=eval_gconfig,
                tokenizer=self.tokenizer,
                enable_thinking=config.enable_thinking,
                max_prompt_thinking_tokens=config.max_prompt_thinking_tokens,
                batch_size=1,
                group_size=n_candidates,
                lazy_sampling=False,
                reward_fn=dummy_reward_fn,
                max_reward_workers=1,
                # Use the stateless distillation prompt for eval, matching training.
                distill_mode=True,
            )
            # Pass the workflow class (not an instance) so that the rollout engine
            # respects group_size.  This matches the standard AReaL evaluator pattern.
            eval_workflow_cls = MultiProblemTTTDiscoverWorkflowV2

            data_list: list[dict[str, Any]] = []
            for problem_id in eval_problem_ids:
                env = self._ale_bench_eval_envs.get(problem_id)
                if env is None:
                    continue
                state = create_initial_state_ale_bench(problem_id=problem_id)
                # Match the stateless distillation prompt used by the eval workflow.
                prompt = (
                    env.get_prompt_distill(state)
                    if hasattr(env, "get_prompt_distill")
                    else env.get_prompt(state)
                )
                data_list.append(
                    {
                        "prompt": prompt,
                        "state_id": state.id,
                        "_state_obj": state,
                        "_problem_id": problem_id,
                    }
                )

            if data_list:
                logger.info(
                    f"[AleBenchEval][Step {global_step}] "
                    f"DP head generating {n_candidates} candidates "
                    f"for {len(data_list)} problems (dp={dp_rank}/{dp_world_size})"
                )

                logger.info(
                    f"[AleBenchEval][Step {global_step}] Submitting {len(data_list)} "
                    f"problems x {n_candidates} candidates to rollout"
                )
                eval_rollout = self.eval_rollout or self.rollout
                for data in data_list:
                    eval_rollout.submit(
                        data,
                        eval_workflow_cls,
                        workflow_kwargs=eval_workflow_kwargs,
                        group_size=n_candidates,
                        is_eval=True,
                    )

                logger.info(
                    f"[AleBenchEval][Step {global_step}] "
                    f"Waiting for {len(data_list)} grouped rollout results"
                )
                results = eval_rollout.wait(len(data_list), timeout=None)
                logger.info(
                    f"[AleBenchEval][Step {global_step}] Got {len(results)} rollout results"
                )
                batch = self._normalize_rollout_batch(results)

                # Decode completions and extract code per problem.
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                loss_mask = batch["loss_mask"]
                problem_ids = batch.get("_problem_ids", [])

                prompt_lens = (
                    (attention_mask & (loss_mask == 0)).sum(dim=1).cpu().numpy()
                )
                comp_lens = loss_mask.sum(dim=1).cpu().numpy()

                for i in range(input_ids.shape[0]):
                    problem_id = problem_ids[i] if i < len(problem_ids) else ""
                    env = self._ale_bench_eval_envs.get(problem_id)
                    if env is None:
                        continue
                    prompt_len = int(prompt_lens[i])
                    comp_len = int(comp_lens[i])
                    completion_ids = (
                        input_ids[i, prompt_len : prompt_len + comp_len].cpu().tolist()
                    )
                    completion_text = self.tokenizer.decode(completion_ids)
                    code = env.extract_code(completion_text)
                    local_candidates.setdefault(problem_id, []).append(
                        code if code is not None else ""
                    )

                generated_count = sum(len(v) for v in local_candidates.values())
                logger.info(
                    f"[AleBenchEval][Step {global_step}] "
                    f"Rank {dp_rank} generated {generated_count} candidate codes "
                    f"across {len(local_candidates)} problems"
                )

        total_count = sum(len(v) for v in local_candidates.values())
        logger.info(
            f"[AleBenchEval][Step {global_step}] DP head has "
            f"{total_count} candidates across {len(local_candidates)} problems"
        )
        return local_candidates

    def _run_ale_bench_eval(self, global_step: int):
        """Run full ALE-Bench public→private evaluation and save results."""
        if not self._ale_bench_eval_enabled:
            return

        config = self.config
        dp_rank = getattr(self.actor, "data_parallel_rank", 0)
        dp_world_size = getattr(self.actor, "data_parallel_world_size", 1)

        if self.actor.rank != 0:
            logger.info(
                f"[AleBenchEval][Step {global_step}] "
                f"Rank {dp_rank}/{dp_world_size} skipping eval, only rank 0 runs"
            )
            return

        # Generate candidates and evaluate only on the DP head.
        logger.info(f"[AleBenchEval][Step {global_step}] Starting candidate generation")
        candidates_by_problem = self._generate_ale_bench_eval_candidates(global_step)
        logger.info(f"[AleBenchEval][Step {global_step}] Candidate generation done")

        eval_problem_ids = self._ale_bench_eval_problem_ids
        logger.info(
            f"[AleBenchEval][Step {global_step}] "
            f"DP head evaluating {len(eval_problem_ids)} problems"
        )

        local_results = evaluate_problem_subset(
            eval_problem_ids,
            candidates_by_problem,
            lite_version=config.ale_bench_eval_lite_version,
            session_duration_hours=4.0,
            ale_bench_num_workers=config.ale_bench_eval_num_workers,
            n_parallel_problems=config.ale_bench_eval_n_parallel_problems,
        )
        logger.info(
            f"[AleBenchEval][Step {global_step}] "
            f"DP head evaluation done, {len(local_results)} results"
        )

        ordered_results: list[dict[str, Any]] = local_results

        training_problem_ids = {mt.problem_id for mt in config.multi_teacher}
        output = combine_ale_bench_results(
            ordered_results,
            self._ale_bench_eval_problem_ids,
            training_problem_ids,
        )
        output.update(
            {
                "n_candidates": config.ale_bench_eval_n_candidates,
                "lite_version": config.ale_bench_eval_lite_version,
                "session_duration_hours": 4.0,
                "ale_bench_num_workers": config.ale_bench_eval_num_workers,
                "n_parallel_problems": config.ale_bench_eval_n_parallel_problems,
            }
        )

        output_path = os.path.join(
            self._ale_bench_eval_output_dir,
            f"ale_bench_eval_results_step_{global_step:06d}.json",
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        avg_all = output.get("average_all", {})
        avg_oot = output.get("average_out_of_training", {})
        logger.info(
            f"[AleBenchEval][Step {global_step}] Saved results to {output_path}"
        )
        logger.info(
            f"[AleBenchEval][Step {global_step}] "
            f"all_abs={avg_all.get('absolute_score', 0.0):.2f} "
            f"oot_abs={avg_oot.get('absolute_score', 0.0):.2f} "
            f"success={avg_all.get('count', 0)}/{len(self._ale_bench_eval_problem_ids)}"
        )

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

        def _get_topk_entropy_and_logp(
            engine, device, input_ids_full, attn_mask_full=None, lmask_full=None
        ):
            engine.model.eval()
            topk_indices = []
            topk_logps = []
            entropies = []
            all_token_logps = []
            _attn = attn_mask_full if attn_mask_full is not None else attention_mask
            _lmask = lmask_full if lmask_full is not None else loss_mask
            _seqlen = input_ids_full.shape[1]

            # Ensure input_ids_full is on the correct device for indexing
            if input_ids_full.device != device:
                input_ids_full = input_ids_full.to(device)
            if _attn.device != device:
                _attn = _attn.to(device)
            if _lmask.device != device:
                _lmask = _lmask.to(device)

            for i in range(n):
                ids = input_ids_full[i : i + 1]
                mask = _attn[i : i + 1]
                lmask = _lmask[i : i + 1]

                with torch.no_grad():
                    out = engine.model(input_ids=ids, attention_mask=mask)
                    logits = out.logits.squeeze(0)  # [seqlen, vocab]

                active = lmask.squeeze(0) > 0
                # Build token logps on CPU to avoid holding GPU memory across sequences
                seq_token_logps = torch.zeros(_seqlen, dtype=torch.float32)

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

        # Privileged teacher entropy (with milestone hint) for fair entropy gap
        tp_ent = None
        if "privileged_teacher_input_ids" in rollout_batch:
            priv_input_ids = rollout_batch["privileged_teacher_input_ids"]
            priv_attn_mask = rollout_batch["privileged_teacher_attention_mask"]
            priv_loss_mask = rollout_batch["privileged_teacher_loss_mask"]
            _tp_idx, _tp_logp, tp_ent, _tp_token_logp = _get_topk_entropy_and_logp(
                self.teacher,
                self.teacher.device,
                priv_input_ids,
                attn_mask_full=priv_attn_mask,
                lmask_full=priv_loss_mask,
            )
            torch.cuda.empty_cache()

        if (
            s_idx is None
            or t_idx is None
            or s_token_logp is None
            or t_token_logp is None
        ):
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
        if tp_ent is not None:
            metrics["distill/entropy_gap_privileged"] = float(
                (tp_ent - s_ent).abs().sum()
            )
            metrics["distill/teacher_entropy_privileged"] = float(tp_ent.sum())
        return metrics, t_token_logp, s_token_logp

    def close(self):
        """Cleanup resources."""
        # Save dynamic metrics checkpoint before cleanup
        if (
            hasattr(self, "dynamic_metrics_logger")
            and self.dynamic_metrics_logger is not None
        ):
            try:
                checkpoint_path = self.dynamic_metrics_logger.save_checkpoint()
                if checkpoint_path:
                    logger.info(
                        f"[DynamicMetrics] Saved checkpoint to {checkpoint_path}"
                    )
            except Exception as e:
                logger.warning(f"[DynamicMetrics] Failed to save checkpoint: {e}")

        self.stats_logger.close()
        if hasattr(self, "rollout") and self.rollout is not None:
            self.rollout.destroy()
        if hasattr(self, "teacher") and self.teacher is not None:
            self.teacher.destroy()
        if hasattr(self, "actor") and self.actor is not None:
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
    is_multi_teacher = bool(config.multi_teacher)
    if not is_multi_teacher and not config.teacher_sampler_checkpoint:
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
        if hasattr(config, "vllm"):
            if isinstance(config.vllm, dict):
                lora_modules_str = config.vllm.get("lora_modules", "")
            else:
                lora_modules_str = getattr(config.vllm, "lora_modules", "") or ""
            if lora_modules_str:
                try:
                    lora_modules = json.loads(lora_modules_str)
                    if isinstance(lora_modules, dict):
                        lora_output_path = lora_modules.get("path", lora_output_path)
                except json.JSONDecodeError:
                    pass

        lora_output_path = os.path.abspath(lora_output_path)
        adapter_config_path = os.path.join(lora_output_path, "adapter_config.json")

        if not os.path.exists(adapter_config_path):
            error_msg = (
                f"\n{'=' * 80}\n"
                f"ERROR: Initial LoRA adapter not found at {lora_output_path}\n"
                f"{'=' * 80}\n\n"
                f"Run the preparation script first:\n"
                f"  python areal/experimental/ttt_discover/examples/prepare_lora_init.py"
                f" --config-path <your_config.yaml>\n\n"
                f"Or skip this check with +skip_lora_check=true\n"
                f"{'=' * 80}\n"
            )
            raise RuntimeError(error_msg)

        logger.info(f"[LoRA Check] LoRA adapter verified at {lora_output_path}")

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

    # Run distillation first so per-problem envs/samplers are loaded.
    with TTTDDistillTrainer(config) as trainer:
        # Create environment(s) after trainer has set up multi-teacher state
        if trainer.is_multi_teacher:
            env = trainer.env
            workflow_cls = MultiProblemTTTDiscoverWorkflowV2
            workflow_kwargs = dict(
                env=env,
                problem_envs=trainer.problem_envs,
                problem_samplers=trainer.student_samplers,
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
        else:
            env = create_env_from_config(config)
            workflow_cls = TTTDiscoverWorkflowV2
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

        trainer.train(
            workflow=workflow_cls,
            workflow_kwargs=workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
