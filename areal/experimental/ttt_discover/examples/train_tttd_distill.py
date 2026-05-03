#!/usr/bin/env python3
"""
TTT-Discover Distillation Training Script.

This script distills a teacher model's policy into a student model with LoRA:
1. Load teacher model checkpoint from a given path
2. Load teacher PUCTSampler state from another given path
3. Student is a new model with LoRA
4. Training: sample from teacher's PUCTSampler, student rollout (no verification)
5. Reward = negative KL divergence between student and teacher
6. Run N distill steps, then 1 eval step with verification

Usage:
    python train_tttd_distill.py --config conf/distill_lora_vllm_cp_qwen3_8b.yaml
"""

import sys
import time
import copy
import json
import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal import PPOTrainer
from areal.api.alloc_mode import AllocationMode, ParallelStrategy
from areal.api.cli_args import (
    PPOActorConfig,
    PPOConfig,
    ClusterSpecConfig,
    load_expr_config,
)
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.infra import current_platform
from areal.utils.environ import is_single_controller
from areal.utils import logging, seeding, stats_tracker
from areal.utils.evaluator import Evaluator
from areal.utils.saver import Saver
from areal.utils.recover import RecoverHandler
from areal.utils.data import KLEstimator

from areal.experimental.ttt_discover.config import (
    SamplerConfig,
    TTTDPPOActorConfig,
    create_env_from_config,
)
from areal.experimental.ttt_discover.sampler import (
    create_sampler_from_config,
    _find_latest_sampler_step,
)
from areal.experimental.ttt_discover.dataloader import create_tttd_dataloader
from areal.experimental.ttt_discover.actor import TTTDActor
from areal.experimental.ttt_discover.workflow_v2 import TTTDiscoverWorkflowV2
from areal.experimental.ttt_discover.reward import tttd_reward_fn
from areal.experimental.ttt_discover.envs.env import EnvResult
from areal.utils.stats_logger import StatsLogger
from areal.utils.hf_utils import load_hf_processor_and_tokenizer

logger = logging.getLogger("train_tttd_distill")


# =============================================================================
# Helper: Sampler that only returns initial states (for evaluation)
# =============================================================================
class _InitialStateSampler:
    """Simple state sampler that cycles through initial states only.
    
    Used for evaluation to ensure all models see the same starting states.
    """
    def __init__(self, initial_states):
        self._initial_states = initial_states
        self._idx = 0
        self._states = initial_states  # for compatibility
    
    def sample_states(self, num_states: int):
        result = []
        for _ in range(num_states):
            result.append(self._initial_states[self._idx % len(self._initial_states)])
            self._idx += 1
        return result


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
# Distillation Config
# =============================================================================
@dataclass
class TTTDDistillConfig(TTTDPPOActorConfig):
    """Extended config for TTT-Discover distillation."""
    
    # Teacher model settings
    teacher_path: str = field(
        default="",
        metadata={"help": "Path to teacher model checkpoint (HF format or DCP)"}
    )
    teacher_sampler_checkpoint: str = field(
        default="",
        metadata={"help": "Path to teacher PUCTSampler checkpoint directory"}
    )
    teacher_weight_format: str = field(
        default="hf",
        metadata={"help": "Teacher checkpoint format: 'hf' or 'dcp'", "choices": ["hf", "dcp"]}
    )
    
    # Distillation settings
    distill_steps: int = field(
        default=3,
        metadata={"help": "Number of distillation steps (no verification)"}
    )
    run_eval_step: bool = field(
        default=True,
        metadata={"help": "Run evaluation step with verification after distillation"}
    )
    kl_reward_scale: float = field(
        default=1.0,
        metadata={"help": "Scale factor for KL-based reward"}
    )
    kl_estimator_type: str = field(
        default="k1",
        metadata={"help": "KL estimator: k1, k2, or k3", "choices": ["k1", "k2", "k3"]}
    )
    
    # Total rollouts per step (will validate batch_size * n_samples == this)
    total_rollouts_per_step: int = field(
        default=512,
        metadata={"help": "Total number of rollouts per step across all ranks"}
    )
    
    # Use mean_baseline for advantage to get relative KL signal
    adv_estimator: str = field(
        default="mean_baseline",
        metadata={"help": "Advantage estimator for distillation"}
    )
    
    # Disable KL penalty since KL is the reward itself
    kl_ctl: float = field(
        default=0.0,
        metadata={"help": "KL penalty coefficient (should be 0 for distillation)"}
    )


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
                f"[Distill] Sync mode detected (max_head_offpolicyness=0). "
                f"Disabling lazy_puct_sampling."
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
        self.actor = self._create_tttd_actor(config)
        self.ref = None  # No ref model needed (kl_ctl=0)
        
        # =====================================================================
        # Create and initialize teacher model
        # =====================================================================
        self.teacher = self._create_teacher_actor(config)
        
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
        
        # Initialize inference engines
        self.rollout = self._init_rollout(config.rollout, is_eval=False)
        
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
        
        # Setup weight update meta
        self._setup_weight_update_meta()
        
        # Setup evaluation, saver, recover, stats logger
        self._setup_utilities()
        
        # Initialize proxy workers flag
        self._proxy_started = False
        
        # Store workflow kwargs for later use
        self._workflow_kwargs = {}
        
        # KL estimator for distill steps
        self.kl_estimator = KLEstimator(
            kl_estimator=config.kl_estimator_type,
            apply_clamp=True,
        )

    def _load_hf_checkpoint(self, engine, path: str, model_name: str = ""):
        """Load HF checkpoint, handling PEFT LoRA adapter key conversion.
        
        PEFT save_pretrained() strips '.default' suffix from adapter keys,
        but FSDP-wrapped PEFT models expect it. We fix the keys here.
        """
        from areal.api.io_struct import SaveLoadMeta
        from safetensors.torch import load_file
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )
        
        adapter_path = os.path.join(path, "adapter_model.safetensors")
        is_lora_adapter = os.path.isfile(adapter_path)
        
        if not is_lora_adapter:
            # Standard HF checkpoint (full model)
            meta = SaveLoadMeta(
                path=path,
                weight_format="hf",
                with_optim=False,
                tokenizer=None,
                processor=None,
            )
            engine.load(meta)
            return
        
        # LoRA adapter: manually load and fix keys
        logger.info(f"[Load-{model_name}] Loading LoRA adapter from {path}")
        if dist.get_rank() == 0:
            lora_state = load_file(adapter_path)
            fixed_state = {}
            for k, v in lora_state.items():
                if "lora_A" in k or "lora_B" in k:
                    k = k.replace(".lora_A.weight", ".lora_A.default.weight")
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
        logger.info(f"[Load-{model_name}] Loaded LoRA adapter from {path}")
    
    def _run_model_eval(self, model_path, model_name, workflow_class, initial_states, group_size):
        """Evaluate a single model on initial states with verification."""
        
        logger.info(f"[Eval-{model_name}] Loading weights from {model_path}")
        
        # 1. Load model weights into actor
        self._load_hf_checkpoint(self.actor, model_path, model_name)
        
        # 2. Push to vLLM
        self.rollout.pause()
        self.actor.update_weights(self.weight_update_meta)
        self.rollout.resume()
        
        # Give vLLM time to load new LoRA
        time.sleep(2)
        
        # 3. Create temp dataloader with initial states only
        temp_sampler = _InitialStateSampler(initial_states)
        eval_batch_size = min(len(initial_states), self.config.sampler.batch_size)
        eval_batch_size = max(eval_batch_size, 1)
        
        temp_dataloader = create_tttd_dataloader(
            temp_sampler,
            rank=self.actor.data_parallel_rank,
            world_size=self.actor.data_parallel_world_size,
            batch_size=eval_batch_size,
            lazy_sampling=False,
        )
        
        # 4. Create eval workflow with verification
        eval_kwargs = self._workflow_kwargs.copy()
        eval_kwargs['reward_fn'] = tttd_reward_fn
        eval_workflow = workflow_class(**eval_kwargs)
        
        self._clear_workflow_cache()
        
        # 5. Run rollout
        eval_start = time.perf_counter()
        try:
            with stats_tracker.record_timing(f"eval_rollout_{model_name}"):
                eval_batch = self.actor.prepare_batch(
                    temp_dataloader,
                    workflow=eval_workflow,
                    workflow_kwargs=None,
                    should_accept_fn=None,
                    group_size=group_size,
                    dynamic_bs=False,
                )
        except Exception as e:
            logger.error(f"[Eval-{model_name}] prepare_batch failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        eval_rollout_time = time.perf_counter() - eval_start
        
        # 6. Gather results (with all_gather for full reward distribution)
        if "rewards" not in eval_batch:
            logger.error(f"[Eval-{model_name}] eval_batch missing 'rewards' key. Keys: {list(eval_batch.keys())}")
            raise KeyError(f"eval_batch missing 'rewards' key")
        local_rollouts = eval_batch["rewards"].shape[0]
        eval_rewards = eval_batch["rewards"].cpu().numpy()
        eval_max_reward = float(eval_rewards.max())
        eval_mean_reward = float(eval_rewards.mean())
        local_rewards_list = eval_rewards.tolist()
        
        if dist.is_initialized():
            local_max = torch.tensor([eval_max_reward], dtype=torch.float32, device=self.actor.device)
            local_sum = torch.tensor([eval_rewards.sum()], dtype=torch.float32, device=self.actor.device)
            local_count = torch.tensor([len(eval_rewards)], dtype=torch.float32, device=self.actor.device)
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
            eval_max_reward = local_max.item()
            eval_mean_reward = (local_sum / local_count).item() if local_count.item() > 0 else 0.0
            global_rollouts = int(local_count.item())
            
            # All-gather raw rewards from all ranks for full distribution analysis
            world_size = self.actor.data_parallel_world_size
            all_rewards_gathered = [None] * world_size
            dist.all_gather_object(all_rewards_gathered, local_rewards_list)
            all_rewards_list = [r for rank_rewards in all_rewards_gathered for r in rank_rewards]
        else:
            global_rollouts = local_rollouts
            all_rewards_list = local_rewards_list
        
        # 7. Get pending updates (verification details)
        eval_updates = eval_workflow.get_pending_updates(clear=True)
        if len(eval_updates) == 5:
            eval_children, eval_parents, eval_failed, eval_metadata, _ = eval_updates
        else:
            eval_children, eval_parents, eval_failed, eval_metadata = eval_updates
        
        result = {
            "model": model_name,
            "global_rollouts": global_rollouts,
            "max_reward": eval_max_reward,
            "mean_reward": eval_mean_reward,
            "all_rewards": all_rewards_list,
            "n_children": len(eval_children),
            "n_parents": len(eval_parents),
            "n_failed": len(eval_failed),
            "rollout_time_s": eval_rollout_time,
            "children": [],
        }
        
        for child in eval_children:
            result["children"].append({
                "id": child.id,
                "timestep": child.timestep,
                "value": child.value,
                "code": getattr(child, 'code', None),
            })
        
        logger.info(
            f"[Eval-{model_name}] max={eval_max_reward:.4f}, mean={eval_mean_reward:.4f}, "
            f"children={len(eval_children)}, failed={len(eval_failed)}"
        )
        
        return result
    
    def _create_tttd_actor(self, actor_config: TTTDDistillConfig):
        """Create student TTTDActor."""
        actor = TTTDActor(config=actor_config)
        actor.create_process_group(parallel_strategy=self.allocation_mode.train)
        return actor
    
    def _create_teacher_actor(self, config: TTTDDistillConfig):
        """Create and initialize frozen teacher TTTDActor from checkpoint.
        
        Supports:
        - HF LoRA adapter (PEFT format with adapter_config.json)
        - HF full model (standard transformers format)
        - DCP checkpoint (AReaL distributed checkpoint)
        """
        if not config.teacher_path:
            raise ValueError("teacher_path must be provided for distillation")
        
        teacher_config = copy.deepcopy(config)
        
        # Detect checkpoint format
        adapter_config_path = os.path.join(config.teacher_path, "adapter_config.json")
        is_lora_adapter = os.path.isfile(adapter_config_path)
        
        if is_lora_adapter:
            # ================================================================
            # Teacher is a LoRA adapter (from TTT-Discover training)
            # ================================================================
            logger.info(f"[Teacher] Detected LoRA adapter at {config.teacher_path}")
            
            # Read base model from adapter config
            with open(adapter_config_path, "r") as f:
                adapter_cfg = json.load(f)
            base_model = adapter_cfg.get("base_model_name_or_path", config.path)
            
            # If base_model is a HF Hub ID (not a local path), use student's base model
            # or the teacher checkpoint dir itself (which has config.json/tokenizer)
            if not base_model or not os.path.isdir(base_model):
                # Prefer teacher checkpoint dir if it has config.json
                if os.path.isfile(os.path.join(config.teacher_path, "config.json")):
                    base_model = config.teacher_path
                    logger.info(f"[Teacher] Using teacher checkpoint dir as base model: {base_model}")
                else:
                    base_model = config.path
                    logger.info(f"[Teacher] Using student's base model as fallback: {base_model}")
            
            teacher_config.path = base_model
            teacher_config.use_lora = True
            # Inherit LoRA params from student config (assumes same architecture)
            
            logger.info(
                f"[Teacher] Will initialize LoRA teacher: base={base_model}, "
                f"rank={teacher_config.lora_rank}, alpha={teacher_config.lora_alpha}"
            )
        else:
            # ================================================================
            # Teacher is a full model (HF or DCP)
            # ================================================================
            teacher_config.path = config.teacher_path
            teacher_config.use_lora = False
            logger.info(f"[Teacher] Detected full model checkpoint at {config.teacher_path}")
        
        # Create teacher actor
        teacher = TTTDActor(config=teacher_config)
        teacher.create_process_group(parallel_strategy=self.allocation_mode.train)
        
        ft_spec = FinetuneSpec(
            total_train_epochs=1,
            dataset_size=1,
            train_batch_size=1,
        )
        teacher.initialize(addr=None, ft_spec=ft_spec, alloc_mode=self.allocation_mode, role="ref")
        
        # Load weights
        if is_lora_adapter:
            self._load_hf_checkpoint(teacher, config.teacher_path, "Teacher")
        elif config.teacher_weight_format == "dcp":
            from areal.api.io_struct import SaveLoadMeta
            meta = SaveLoadMeta(
                path=config.teacher_path,
                weight_format="dcp",
                with_optim=False,
                tokenizer=None,
                processor=None,
                base_model_path=None,
            )
            teacher.load(meta)
            logger.info(f"[Teacher] Loaded DCP weights from {config.teacher_path}")
        else:
            logger.info(f"[Teacher] Using HF weights loaded during initialize() from {teacher_config.path}")
        
        # Freeze teacher parameters
        for param in teacher.model.parameters():
            param.requires_grad = False
        
        logger.info("[Teacher] Teacher model created and frozen")
        return teacher
    
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
        max_steps = self.config.distill_steps + (1 if self.config.run_eval_step else 0)
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
    
    def _setup_weight_update_meta(self):
        """Setup weight update meta and connect to inference engine."""
        config = self.config
        
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
            raise ValueError(f"Invalid weight update mode: {config.weight_update_mode}")
        
        self.actor.connect_engine(self.rollout, self.weight_update_meta)
    
    def _setup_utilities(self):
        """Setup evaluator, saver, recover handler, and stats logger."""
        config = self.config
        
        max_steps = config.distill_steps + (1 if config.run_eval_step else 0)
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
    
    def _compute_kl_reward(self, rollout_batch: dict[str, Any]) -> torch.Tensor:
        """Compute KL divergence reward for a rollout batch.
        
        Returns sequence-level negative KL scaled by kl_reward_scale.
        """
        device = self.actor.device
        
        # Compute teacher logprobs for the rollout sequences
        with torch.no_grad():
            teacher_logp = self.teacher.compute_logp(rollout_batch)
        
        # Get student logprobs (use prox_logp if available, otherwise logprobs)
        if "prox_logp" in rollout_batch and rollout_batch["prox_logp"] is not None:
            student_logp = rollout_batch["prox_logp"]
        else:
            student_logp = rollout_batch["logprobs"]
        
        # Ensure same shape and device
        if teacher_logp is None:
            logger.warning("[Distill] Teacher logp is None, using zero reward")
            return torch.zeros(rollout_batch["rewards"].shape[0], device=device)
        
        teacher_logp = teacher_logp.to(device)
        student_logp = student_logp.to(device)
        
        # Compute per-token KL using estimator
        # kl_estimator returns log_ratio = log p_student - log p_teacher
        # This is a sampled estimate of KL(student || teacher)
        kl_per_token = self.kl_estimator(student_logp, teacher_logp)  # [bs, seqlen]
        
        # Apply loss mask to sum only over valid tokens
        loss_mask = rollout_batch["loss_mask"].float().to(device)
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)
        kl_per_token = kl_per_token * loss_mask
        
        # Sequence-level KL (sum over valid tokens)
        seq_kl = kl_per_token.sum(dim=1)  # [bs]
        
        # Reward = negative KL (we want to minimize KL)
        reward = -seq_kl * self.config.kl_reward_scale
        
        return reward
    
    def _compute_distill_advantages(self, rollout_batch: dict[str, Any]) -> dict[str, Any]:
        """Simple advantage computation for distillation.
        
        Bypasses TTTDActor's entropic objective and computes plain advantages:
            A = reward - mean(reward)
        
        This is much simpler than TTT-Discover's w_beta - 1 entropic weighting.
        """
        device = rollout_batch["input_ids"].device
        bs, max_seqlen = rollout_batch["input_ids"].shape
        
        # Sequence-level rewards (already -KL)
        reward_score = rollout_batch["rewards"].squeeze(-1)  # [bs]
        rollout_batch["rewards"] = reward_score
        
        # Optional mean baseline across all ranks
        if dist.is_initialized() and self.actor.data_parallel_world_size > 1:
            local_sum = reward_score.sum()
            local_count = torch.tensor(bs, dtype=torch.float32, device=device)
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
            mean_reward = local_sum / local_count
        else:
            mean_reward = reward_score.mean()
        
        # Simple advantage: reward - mean(reward)
        advantages_seq = reward_score - mean_reward  # [bs]
        
        # Broadcast to token level
        advantages = advantages_seq.unsqueeze(-1).expand(-1, max_seqlen)  # [bs, seq_len]
        
        # Apply loss mask
        loss_mask = rollout_batch["loss_mask"].float()
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)
        advantages = advantages * loss_mask
        
        # Store in batch
        rollout_batch["advantages"] = advantages
        rollout_batch["returns"] = advantages
        rollout_batch["loss_mask"] = loss_mask
        
        # ppo_update expects kl_rewards and tot_rewards for logging
        rollout_batch["kl_rewards"] = torch.zeros_like(advantages)
        rollout_batch["tot_rewards"] = reward_score.unsqueeze(-1).expand(-1, max_seqlen) * loss_mask
        
        return rollout_batch
    
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
            f"distill_steps={config.distill_steps}, "
            f"eval_step={config.run_eval_step}, "
            f"total_rollouts_per_step={total_rollouts}, "
            f"kl_scale={config.kl_reward_scale}"
        )
        
        # =====================================================================
        # Phase 1: Distillation steps (no verification)
        # =====================================================================
        for step_idx in range(config.distill_steps):
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
            
            # Compute KL-based reward
            kl_reward = self._compute_kl_reward(rollout_batch)
            
            # Replace batch rewards with KL reward
            rollout_batch["rewards"] = kl_reward.unsqueeze(-1)  # [bs, 1]
            
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
                f"KL reward: max={step_max_reward:.4f}, mean={step_mean_reward:.4f}, min={step_min_reward:.4f}"
            )
            
            # Training computations
            step_info = StepInfo(
                global_step=global_step,
                epoch=global_step,
                epoch_step=global_step,
                steps_per_epoch=config.distill_steps,
            )
            
            # Compute prox_logp if needed
            if config.should_compute_prox_logp():
                rollout_batch["prox_logp"] = self.actor.compute_logp(rollout_batch)
            
            # Compute simple advantages (bypass TTT-Discover entropic objective)
            self._compute_distill_advantages(rollout_batch)
            self.actor.ppo_update(rollout_batch)
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
            
            if is_dp_head:
                self.stats_logger.commit(
                    epoch=global_step,
                    step=global_step,
                    global_step=global_step,
                    data=metrics,
                )
                logger.info(
                    f"[Distill][Step {global_step}] "
                    f"Loss: {metrics['train/actor_loss']:.4f}, "
                    f"KL: {metrics['train/approx_kl']:.4f}, "
                    f"Entropy: {metrics['train/entropy']:.4f}, "
                    f"GradNorm: {metrics['train/grad_norm']:.4f}, "
                    f"LR: {metrics['train/lr']:.6f}"
                )
            
            # Update weights and save
            self.rollout.pause()
            self.actor.update_weights(self.weight_update_meta)
            self.actor.set_version(global_step + 1)
            self.rollout.set_version(global_step + 1)
            
            self._save_hf(epoch=global_step, epoch_step=global_step, global_step=global_step)
            self._save_recover_checkpoint(epoch=global_step, epoch_step=global_step, global_step=global_step)
            
            dist.barrier(group=self.actor.cpu_group)
            current_platform.synchronize()
            self.rollout.resume()
            
            step_total = time.perf_counter() - step_start_time
            logger.info(
                f"[Distill][Step {global_step}] Completed in {step_total:.2f}s "
                f"(rollout={rollout_time:.2f}s)"
            )
        
        # =====================================================================
        # Phase 2: Evaluation step (with verification)
        # =====================================================================
        if config.run_eval_step:
            eval_step = config.distill_steps
            logger.info(f"[Eval][Step {eval_step}] Starting 3-model comparison on initial states")
            
            # Get initial states
            initial_states = self.sampler._initial_states
            if not initial_states:
                initial_states = [s for s in self.sampler._states if getattr(s, 'timestep', 0) == 0]
            if not initial_states:
                initial_states = self.sampler._states[:1]
                logger.warning("[Eval] No initial states found, using first available state")
            
            logger.info(f"[Eval] Using {len(initial_states)} initial states for evaluation")
            
            # Save current student weights before switching
            from areal.api.io_struct import SaveLoadMeta
            student_backup_meta = SaveLoadMeta(
                path=self.student_eval_path,
                weight_format="hf",
                with_optim=False,
                tokenizer=None,
                processor=None,
            )
            self.actor.save(student_backup_meta)
            
            # Evaluate three models
            all_results = {}
            for label, path in [
                ("base", self.base_eval_path),
                ("teacher", self.teacher_eval_path),
                ("student", self.student_eval_path),
            ]:
                if dist.is_initialized():
                    dist.barrier()
                all_results[label] = self._run_model_eval(
                    path, label, workflow, initial_states, group_size
                )
            
            # Restore student weights
            logger.info("[Eval] Restoring student weights")
            self._load_hf_checkpoint(self.actor, self.student_eval_path, "StudentRestore")
            self.rollout.pause()
            self.actor.update_weights(self.weight_update_meta)
            self.rollout.resume()
            
            # Save comparison results
            if is_dp_head:
                comparison_path = os.path.join(
                    config.saver.fileroot,
                    f"eval_comparison_step_{eval_step}.json"
                )
                with open(comparison_path, 'w') as f:
                    json.dump(all_results, f, indent=2, default=str)
                logger.info(f"[Eval] Comparison results saved to {comparison_path}")
                
                # Print summary table
                logger.info("\n" + "="*70)
                logger.info("EVALUATION COMPARISON (Initial States)")
                logger.info("="*70)
                for label in ["base", "teacher", "student"]:
                    r = all_results[label]
                    logger.info(
                        f"{label:10s} | max_reward={r['max_reward']:.4f} | "
                        f"mean_reward={r['mean_reward']:.4f} | "
                        f"rollouts={r['global_rollouts']} | "
                        f"children={r['n_children']} | failed={r['n_failed']}"
                    )
                logger.info("="*70)
                
            # Ensure all ranks finish evaluation before returning
            if dist.is_initialized():
                dist.barrier()
    
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
    
    def close(self):
        """Cleanup resources."""
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
    import json
    import os
    from areal.utils import logging
    
    config, _ = load_expr_config(args, TTTDDistillConfig)
    
    # Validate teacher paths
    if not config.teacher_path:
        raise ValueError("teacher_path must be provided. Add +teacher_path=<path> to your command.")
    if not config.teacher_sampler_checkpoint:
        raise ValueError("teacher_sampler_checkpoint must be provided. Add +teacher_sampler_checkpoint=<path> to your command.")
    
    # Ensure stop tokens are set
    if config.tokenizer_path:
        from areal.utils.hf_utils import load_hf_tokenizer
        tokenizer = load_hf_tokenizer(config.tokenizer_path)
        if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
            config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
        if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
            config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    
    # Verify LoRA adapter exists
    if config.use_lora and not config.skip_lora_check:
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
    
    # Create environment (needed for eval workflow)
    env = create_env_from_config(config)
    
    # Calculate local batch size
    from areal.api.alloc_mode import AllocationMode
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
