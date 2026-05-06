#!/usr/bin/env python3
"""
TTT-Discover Multi-Model Evaluation (v2).

Evaluates multiple models in a SINGLE vLLM process by pre-loading all LoRA
adapters at vLLM startup time, then switching between them via ``set_version()``
(HTTP ``model`` field). This completely avoids ``update_weights()``, which
causes hangs with vLLM 0.17.0 V1 engine.

Usage (via launcher, same as training):
    python -m areal.infra.launcher.local \
        eval_tttd_multi_v2.py \
        --config conf/fsdp_lora_vllm_ac1_qwen3_8b_distill.yaml \
        [+eval.models=teacher,student]
"""

import sys
import time
import copy
import json
import os
from typing import Any

import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal import PPOTrainer
from areal.api.alloc_mode import _AllocationMode as AllocationMode, ModelAllocation
from areal.api.cli_args import load_expr_config
from areal.api.io_struct import FinetuneSpec
from areal.infra import current_platform
from areal.utils.environ import is_single_controller
from areal.utils import logging, seeding, stats_tracker
from areal.utils.stats_logger import StatsLogger
from areal.utils.hf_utils import load_hf_processor_and_tokenizer

from areal.experimental.ttt_discover.config import (
    TTTDDistillConfig,
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

logger = logging.getLogger("eval_tttd_multi_v2")


class _InitialStateSampler:
    """Simple state sampler that cycles through initial states only."""
    def __init__(self, initial_states):
        self._initial_states = initial_states
        self._idx = 0

    def sample_states(self, num_states: int):
        result = []
        for _ in range(num_states):
            result.append(self._initial_states[self._idx % len(self._initial_states)])
            self._idx += 1
        return result


class TTTDMultiEvalTrainer(PPOTrainer):
    """Multi-model evaluation trainer.

    Mirrors ``TTTDDistillTrainer`` initialization but:
    - No training loop
    - No teacher actor
    - Skips ``connect_engine`` / ``update_weights`` entirely
    - Pre-loads all target LoRAs into vLLM at startup via ``lora_modules``
    - Switches model at inference time using ``set_version()`` only
    """

    def __init__(self, config: TTTDDistillConfig):
        self.config = config
        rank = int(os.getenv("RANK", "0"))
        if is_single_controller():
            logging.setup_file_logging(StatsLogger.get_log_path(config.stats_logger))

        # Load tokenizer and processor
        self.processor, self.tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)

        # Scheduler (skip if null)
        self.scheduler = None
        if is_single_controller():
            sched_type = getattr(config.scheduler, 'type', None)
            if sched_type is not None and sched_type != 'null':
                self.scheduler = self._init_scheduler()

        # Set seed
        seeding.set_random_seed(config.seed, key=f"eval{rank}")

        # Parse allocation mode
        self.allocation_mode = AllocationMode.from_str(config.allocation_mode)
        self.actor_alloc = ModelAllocation.from_str(config.actor.backend, name="actor")
        self.rollout_alloc = ModelAllocation.from_str(config.rollout.backend, name="rollout")
        self._amend_xccl_weight_update_envvar()

        # Create sampler from teacher checkpoint
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

        # Create actor (no critic/ref - eval only)
        self.actor = self._create_tttd_actor(config.actor)
        self.ref = None

        # Create dataloader
        self.train_dataloader = create_tttd_dataloader(
            state_sampler=self.sampler,
            rank=self.actor.data_parallel_rank,
            world_size=self.actor.data_parallel_world_size,
            batch_size=config.sampler.batch_size,
            lazy_sampling=config.sampler.lazy_puct_sampling,
        )
        self.train_dataset = self.train_dataloader.dataset
        self.valid_dataloader = None
        self.valid_dataset = None

        # ------------------------------------------------------------------
        # Pre-configure vLLM lora_modules so vLLM loads ALL adapters at startup.
        # Format: JSON object strings, one per adapter.
        # The "name" must match get_versioned_lora_name(config.gconfig.lora_name, version).
        # ------------------------------------------------------------------
        self._setup_vllm_lora_modules(config)

        # Initialize inference engines (vLLM will now load all LoRAs at startup)
        self.rollout = self._init_rollout(config.rollout, is_eval=False)

        # Initialize models
        self._initialize_engines()

        # Connect sampler to actor for distributed synchronization
        self.actor.connect_sampler(self.sampler)

        # ------------------------------------------------------------------
        # Skip weight update meta / connect_engine entirely.
        # We do NOT call self.actor.connect_engine(self.rollout, ...)
        # because we never use update_weights().
        # ------------------------------------------------------------------
        logger.info("[MultiEval] Skipping connect_engine / weight_update_meta (no update_weights)")

        # Setup stats logger only
        self._setup_stats_logger()

        # Store workflow kwargs for later use
        self._workflow_kwargs = {}

    def _create_tttd_actor(self, actor_config):
        """Create TTTDActor."""
        actor = TTTDActor(config=actor_config)
        actor.create_process_group(parallel_strategy=self.allocation_mode.train)
        return actor

    def _initialize_engines(self):
        """Initialize training engines."""
        ft_spec = FinetuneSpec(
            total_train_epochs=1,
            dataset_size=self.config.sampler.batch_size,
            train_batch_size=self.config.sampler.batch_size,
        )
        self.actor.initialize(addr=None, ft_spec=ft_spec, alloc_mode=self.allocation_mode, role="actor")

    def _setup_stats_logger(self):
        """Setup stats logger only (no saver/recover/evaluator for eval)."""
        config = self.config
        ft_spec = FinetuneSpec(
            total_train_epochs=1,
            dataset_size=config.sampler.batch_size,
            train_batch_size=config.sampler.batch_size,
        )
        self.stats_logger = StatsLogger(config, ft_spec)

    def _setup_vllm_lora_modules(self, config: TTTDDistillConfig):
        """Configure vLLM ``lora_modules`` so it pre-loads all evaluation adapters.

        Reads ``eval_models`` from config (or defaults to teacher + student) and
        builds a YAML-compatible list of JSON object strings.
        """
        # Determine which models to evaluate
        eval_models = getattr(config, 'eval_models', None)
        if eval_models is None:
            eval_models = {
                "teacher": getattr(config, 'teacher_lora_path', config.teacher_path),
                "student": getattr(config, 'student_lora_path', None),
            }
        # Filter out missing paths
        eval_models = {k: v for k, v in eval_models.items() if v}
        if not eval_models:
            raise ValueError("No evaluation models found. Set teacher_path / student_lora_path or eval_models.")

        self._eval_models = eval_models
        lora_name = config.gconfig.lora_name
        base_model = config.actor.path

        lora_modules = []
        for idx, (label, path) in enumerate(eval_models.items()):
            resolved = os.path.expanduser(path)
            if not os.path.isdir(resolved):
                raise ValueError(f"Eval model path for '{label}' does not exist: {resolved}")
            # Name must match get_versioned_lora_name(lora_name, version)
            adapter_name = f"{lora_name}-v{idx}"
            lora_modules.append(
                json.dumps({"name": adapter_name, "path": resolved, "base_model_name": base_model})
            )
            logger.info(f"[MultiEval] Will pre-load LoRA '{adapter_name}' -> {resolved}")

        config.vllm.lora_modules = lora_modules

    def _clear_workflow_cache(self):
        """Clear workflow executor cache so a new workflow can be used."""
        workflow_executor = None
        if hasattr(self.rollout, '_engine') and hasattr(self.rollout._engine, 'workflow_executor'):
            workflow_executor = self.rollout._engine.workflow_executor
        elif hasattr(self.rollout, 'workflow_executor'):
            workflow_executor = self.rollout.workflow_executor

        if workflow_executor is not None and hasattr(workflow_executor, 'data_generator'):
            delattr(workflow_executor, 'data_generator')
            logger.info("[MultiEval] Cleared workflow executor cache")

        if hasattr(self.rollout, 'data_generator'):
            delattr(self.rollout, 'data_generator')
            logger.info("[MultiEval] Cleared rollout controller data_generator")

    def _run_single_model_eval(self, label: str, version: int, workflow_class, initial_states, group_size):
        """Evaluate a single model on initial states with verification."""
        logger.info(f"[MultiEval-{label}] Setting version={version} and starting rollout")
        self.rollout.set_version(version)

        # Create temp dataloader with initial states only
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
        logger.info(f"[MultiEval-{label}] Created temporary dataloader with batch size {eval_batch_size}")

        # Create eval workflow with verification
        eval_kwargs = self._workflow_kwargs.copy()
        eval_kwargs['reward_fn'] = tttd_reward_fn
        eval_workflow = workflow_class(**eval_kwargs)

        self._clear_workflow_cache()

        # Run rollout
        eval_start = time.perf_counter()
        try:
            with stats_tracker.record_timing(f"eval_rollout_{label}"):
                eval_batch = self.actor.prepare_batch(
                    temp_dataloader,
                    workflow=eval_workflow,
                    workflow_kwargs=None,
                    should_accept_fn=None,
                    group_size=group_size,
                    dynamic_bs=False,
                )
        except Exception as e:
            logger.error(f"[MultiEval-{label}] prepare_batch failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        eval_rollout_time = time.perf_counter() - eval_start

        # Gather results
        if "rewards" not in eval_batch:
            logger.error(f"[MultiEval-{label}] eval_batch missing 'rewards' key. Keys: {list(eval_batch.keys())}")
            raise KeyError("eval_batch missing 'rewards' key")
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

        # Get pending updates
        eval_updates = eval_workflow.get_pending_updates(clear=True)
        if len(eval_updates) == 5:
            eval_children, eval_parents, eval_failed, eval_metadata, _ = eval_updates
        else:
            eval_children, eval_parents, eval_failed, eval_metadata = eval_updates

        result = {
            "model": label,
            "lora_path": self._eval_models[label],
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
            f"[MultiEval-{label}] max={eval_max_reward:.4f}, mean={eval_mean_reward:.4f}, "
            f"children={len(eval_children)}, failed={len(eval_failed)}"
        )

        return result

    def run_eval(self, workflow_class, workflow_kwargs=None):
        """Run evaluation on all pre-loaded models."""
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
            logger.warning("[MultiEval] No initial states found, using first available state")

        logger.info(f"[MultiEval] Using {len(initial_states)} initial states for evaluation")

        # Evaluate each model sequentially
        all_results = {}
        for idx, (label, path) in enumerate(self._eval_models.items()):
            if dist.is_initialized():
                dist.barrier()
            all_results[label] = self._run_single_model_eval(
                label, idx, workflow_class, initial_states, group_size
            )

        # Save comparison results
        if is_dp_head:
            comparison_path = os.path.join(
                config.saver.fileroot,
                "eval_comparison.json"
            )
            os.makedirs(config.saver.fileroot, exist_ok=True)
            with open(comparison_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            logger.info(f"[MultiEval] Comparison results saved to {comparison_path}")

            logger.info("\n" + "="*70)
            logger.info("EVALUATION COMPARISON (Initial States)")
            logger.info("="*70)
            for label in self._eval_models:
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

    def close(self):
        """Cleanup resources."""
        self.stats_logger.close()
        if hasattr(self, 'rollout') and self.rollout is not None:
            self.rollout.destroy()
        if hasattr(self, 'actor') and self.actor is not None:
            self.actor.destroy()
        from areal.utils import perf_tracer
        perf_tracer.save(force=True)


def main(args):
    """Main evaluation function."""
    config, _ = load_expr_config(args, TTTDDistillConfig)

    if not config.teacher_sampler_checkpoint:
        raise ValueError(
            "teacher_sampler_checkpoint must be provided. "
            "Add +teacher_sampler_checkpoint=<path> to your command."
        )

    if config.tokenizer_path:
        from areal.utils.hf_utils import load_hf_tokenizer
        tokenizer = load_hf_tokenizer(config.tokenizer_path)
        if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
            config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
        if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
            config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)

    env = create_env_from_config(config)

    from areal.api.alloc_mode import _AllocationMode as AllocationMode
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

    with TTTDMultiEvalTrainer(config) as trainer:
        trainer.run_eval(
            workflow_class=TTTDiscoverWorkflowV2,
            workflow_kwargs=workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
