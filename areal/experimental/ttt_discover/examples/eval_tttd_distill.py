#!/usr/bin/env python3
"""
TTT-Discover Distillation Evaluation Script.

Evaluates three models (base, teacher, student) on initial states with verification.
Assumes distillation training has already been run and checkpoints exist at:
    <saver.fileroot>/eval_checkpoints/{base,teacher,student}

Usage:
    python eval_tttd_distill.py --config conf/distill_lora_vllm_cp_qwen3_8b.yaml
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
from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import (
    PPOActorConfig,
    PPOConfig,
    ClusterSpecConfig,
    load_expr_config,
)
from areal.api.io_struct import FinetuneSpec, WeightUpdateMeta
from areal.infra import current_platform
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
from areal.experimental.ttt_discover.sampler import (
    create_sampler_from_config,
    _find_latest_sampler_step,
)
from areal.experimental.ttt_discover.dataloader import create_tttd_dataloader
from areal.experimental.ttt_discover.actor import TTTDActor
from areal.experimental.ttt_discover.workflow_v2 import TTTDiscoverWorkflowV2
from areal.experimental.ttt_discover.reward import tttd_reward_fn
from areal.utils.stats_logger import StatsLogger
from areal.utils.hf_utils import load_hf_processor_and_tokenizer

logger = logging.getLogger("eval_tttd_distill")


# =============================================================================
# Helper: Sampler that only returns initial states
# =============================================================================
class _InitialStateSampler:
    """Simple state sampler that cycles through initial states only."""
    def __init__(self, initial_states):
        self._initial_states = initial_states
        self._idx = 0
        self._states = initial_states

    def sample_states(self, num_states: int):
        result = []
        for _ in range(num_states):
            result.append(self._initial_states[self._idx % len(self._initial_states)])
            self._idx += 1
        return result


# =============================================================================
# Eval Config (reuse distillation config)
# =============================================================================
@dataclass
class TTTDDistillConfig(TTTDPPOActorConfig):
    """Config for TTT-Discover distillation and evaluation."""

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
    distill_steps: int = field(
        default=3,
        metadata={"help": "Number of distillation steps (unused in eval)"}
    )
    kl_reward_scale: float = field(default=1.0, metadata={"help": "Scale factor for KL-based reward"})
    kl_estimator_type: str = field(
        default="k1",
        metadata={"help": "KL estimator: k1, k2, or k3", "choices": ["k1", "k2", "k3"]}
    )
    total_rollouts_per_step: int = field(
        default=512,
        metadata={"help": "Total number of rollouts per step across all ranks"}
    )
    adv_estimator: str = field(default="mean_baseline", metadata={"help": "Advantage estimator"})
    kl_ctl: float = field(default=0.0, metadata={"help": "KL penalty coefficient"})
    run_eval_step: bool = field(default=True, metadata={"help": "Run evaluation with real verification after distillation"})


# =============================================================================
# Evaluator
# =============================================================================
class TTTDDistillEvaluator:
    """Evaluates base, teacher, and student models on initial states."""

    def __init__(self, config: TTTDDistillConfig):
        self.config = config
        rank = int(os.getenv("RANK", "0"))
        if is_single_controller():
            logging.setup_file_logging(StatsLogger.get_log_path(config.stats_logger))

        self.processor, self.tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)
        self.scheduler = None
        if is_single_controller():
            self.scheduler = self._init_scheduler()

        seeding.set_random_seed(config.seed, key=f"eval{rank}")
        self.allocation_mode = AllocationMode.from_str(config.allocation_mode)
        self._amend_xccl_weight_update_envvar()

        # Load teacher sampler to get initial states
        max_head_offpolicyness = getattr(config.rollout, 'max_head_offpolicyness', 2)
        max_version_history = max_head_offpolicyness + 1

        self.is_sync_mode = (max_head_offpolicyness == 0)
        if self.is_sync_mode and config.sampler.lazy_puct_sampling:
            config.sampler.lazy_puct_sampling = False

        teacher_sampler_config = copy.deepcopy(config.sampler)
        if config.teacher_sampler_checkpoint:
            teacher_sampler_config.checkpoint_dir = config.teacher_sampler_checkpoint

        self.sampler = create_sampler_from_config(
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
                self.sampler._load(latest_step)
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

        # Create student actor (we will load different weights into it)
        self.actor = self._create_tttd_actor(config)
        self.ref = None

        # Create dataloaders
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

        # Setup weight update meta
        self._setup_weight_update_meta()

        # Setup utilities (saver/evaluator for path resolution)
        self._setup_utilities()

        # Eval checkpoint paths
        eval_ckpt_dir = os.path.join(config.saver.fileroot, "eval_checkpoints")
        self.base_eval_path = os.path.join(eval_ckpt_dir, "base")
        self.teacher_eval_path = os.path.join(eval_ckpt_dir, "teacher")
        self.student_eval_path = os.path.join(eval_ckpt_dir, "student")

        self._proxy_started = False
        self._workflow_kwargs = {}

    def _create_tttd_actor(self, actor_config: TTTDDistillConfig):
        actor = TTTDActor(config=actor_config)
        actor.create_process_group(parallel_strategy=self.allocation_mode.train)
        return actor

    def _create_tttd_dataloader(self, sampler, rank, world_size, batch_size, lazy_sampling=False):
        return create_tttd_dataloader(
            state_sampler=sampler,
            rank=rank,
            world_size=world_size,
            batch_size=batch_size,
            lazy_sampling=lazy_sampling,
        )

    def _initialize_engines(self):
        ft_spec = FinetuneSpec(
            total_train_epochs=1,
            dataset_size=self.config.sampler.batch_size,
            train_batch_size=self.config.sampler.batch_size,
        )
        self.actor.initialize(addr=None, ft_spec=ft_spec, alloc_mode=self.allocation_mode, role="actor")

    def _setup_weight_update_meta(self):
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
                self.weight_update_meta = WeightUpdateMeta.from_megatron_xccl(self.allocation_mode)
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
        config = self.config
        ft_spec = FinetuneSpec(
            total_train_epochs=1,
            dataset_size=config.sampler.batch_size,
            train_batch_size=config.sampler.batch_size,
        )
        self.evaluator = Evaluator(config.evaluator, ft_spec)
        self.saver = Saver(config.saver, ft_spec)
        self.recover_handler = RecoverHandler(config.recover, ft_spec)
        self.stats_logger = StatsLogger(config, ft_spec)

    def _init_scheduler(self):
        from areal.infra import LocalScheduler
        return LocalScheduler(exp_config=self.config)

    def _init_rollout(self, rollout_config, is_eval=False):
        from areal.infra import RolloutController
        from areal.engine.vllm_remote import RemotevLLMEngine
        from copy import deepcopy
        from areal.api.cli_args import InferenceEngineConfig, SchedulingStrategy, SchedulingStrategyType, vLLMConfig

        config = deepcopy(rollout_config)
        if is_eval:
            config.max_head_offpolicyness = int(1e12)
            config.scheduling_strategy = SchedulingStrategy(
                type=SchedulingStrategyType.colocation, target="rollout"
            )
            for spec in config.scheduling_spec:
                spec.gpu = 0

        if self.allocation_mode.gen_backend == "vllm":
            engine_cls = RemotevLLMEngine
            server_args = vLLMConfig.build_args(
                vllm_config=self.config.vllm,
                tp_size=self.allocation_mode.gen.tp_size,
                pp_size=self.allocation_mode.gen.pp_size,
            )
        else:
            raise ValueError(f"Unsupported gen backend: {self.allocation_mode.gen_backend}")

        if self.scheduler is None:
            self.scheduler = self._init_scheduler()

        controller = RolloutController(engine_cls, config, self.scheduler)
        controller.initialize(
            role="rollout",
            alloc_mode=self.allocation_mode,
            server_args=server_args,
        )
        return controller

    def _amend_xccl_weight_update_envvar(self):
        """Ensure XCCL weight update environment variables are set correctly."""
        pass

    def _load_hf_checkpoint(self, engine, path: str, model_name: str = ""):
        """Load HF checkpoint, handling PEFT LoRA adapter key conversion."""
        from areal.api.io_struct import SaveLoadMeta
        from safetensors.torch import load_file
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )

        adapter_path = os.path.join(path, "adapter_model.safetensors")
        is_lora_adapter = os.path.isfile(adapter_path)

        if not is_lora_adapter:
            meta = SaveLoadMeta(
                path=path,
                weight_format="hf",
                with_optim=False,
                tokenizer=None,
                processor=None,
            )
            engine.load(meta)
            return

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

    def _clear_workflow_cache(self):
        """Clear workflow executor cache so a new workflow can be used."""
        workflow_executor = None
        if hasattr(self.rollout, '_engine') and hasattr(self.rollout._engine, 'workflow_executor'):
            workflow_executor = self.rollout._engine.workflow_executor
        elif hasattr(self.rollout, 'workflow_executor'):
            workflow_executor = self.rollout.workflow_executor

        if workflow_executor is not None and hasattr(workflow_executor, 'data_generator'):
            delattr(workflow_executor, 'data_generator')
            logger.info("[Eval] Cleared workflow executor cache")

        if hasattr(self.rollout, 'data_generator'):
            delattr(self.rollout, 'data_generator')
            logger.info("[Eval] Cleared rollout controller data_generator")

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

        # 6. Gather results
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

            world_size = self.actor.data_parallel_world_size
            all_rewards_gathered = [None] * world_size
            dist.all_gather_object(all_rewards_gathered, local_rewards_list)
            all_rewards_list = [r for rank_rewards in all_rewards_gathered for r in rank_rewards]
        else:
            global_rollouts = local_rollouts
            all_rewards_list = local_rewards_list

        # 7. Get pending updates
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

    def run(self, workflow, workflow_kwargs=None):
        """Run evaluation on base, teacher, and student models."""
        config = self.config

        if workflow_kwargs is not None:
            self._workflow_kwargs = workflow_kwargs.copy()
            self._workflow_kwargs['lazy_sampling'] = config.sampler.lazy_puct_sampling
            if config.sampler.lazy_puct_sampling and 'sampler' not in self._workflow_kwargs:
                self._workflow_kwargs['sampler'] = self.sampler
                self._workflow_kwargs['dp_rank'] = self.actor.dp_rank
                self._workflow_kwargs['dp_world_size'] = self.actor.data_parallel_world_size

        is_dp_head = self.actor.rank == 0
        group_size = config.gconfig.n_samples

        # Get initial states
        initial_states = self.sampler._initial_states
        if not initial_states:
            initial_states = [s for s in self.sampler._states if getattr(s, 'timestep', 0) == 0]
        if not initial_states:
            initial_states = self.sampler._states[:1]
            logger.warning("[Eval] No initial states found, using first available state")

        logger.info(f"[Eval] Using {len(initial_states)} initial states for evaluation")

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

        # Save comparison results
        if is_dp_head:
            comparison_path = os.path.join(
                config.saver.fileroot,
                "eval_comparison.json"
            )
            with open(comparison_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            logger.info(f"[Eval] Comparison results saved to {comparison_path}")

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

        if dist.is_initialized():
            dist.barrier()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        if exc_type is not None:
            raise exc_value

    def close(self):
        self.stats_logger.close()
        if hasattr(self, 'rollout') and self.rollout is not None:
            self.rollout.destroy()
        if hasattr(self, 'actor') and self.actor is not None:
            self.actor.destroy()
        from areal.utils import perf_tracer
        perf_tracer.save(force=True)


# =============================================================================
# Main
# =============================================================================
def main(args):
    """Main evaluation function."""
    import os
    from areal.utils import logging

    config, _ = load_expr_config(args, TTTDDistillConfig)

    if not config.teacher_path:
        raise ValueError("teacher_path must be provided. Add +teacher_path=<path> to your command.")
    if not config.teacher_sampler_checkpoint:
        raise ValueError("teacher_sampler_checkpoint must be provided. Add +teacher_sampler_checkpoint=<path> to your command.")

    if config.tokenizer_path:
        from areal.utils.hf_utils import load_hf_tokenizer
        tokenizer = load_hf_tokenizer(config.tokenizer_path)
        if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
            config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
        if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
            config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)

    env = create_env_from_config(config)

    from areal.api.alloc_mode import AllocationMode
    alloc_mode = AllocationMode.from_str(config.allocation_mode)
    train_world_size = alloc_mode.train.world_size
    local_batch_size = config.sampler.batch_size // train_world_size
    group_size = config.gconfig.n_samples

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

    with TTTDDistillEvaluator(config) as evaluator:
        evaluator.run(
            workflow=TTTDiscoverWorkflowV2,
            workflow_kwargs=workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
