#!/usr/bin/env python3
"""
TTT-Discover Multi-Model Evaluation (v2).

Evaluates multiple models sequentially by loading each LoRA adapter into the
actor and pushing it to vLLM via ``update_weights()`` (same path as training).

Usage (same as training):
    python areal/experimental/ttt_discover/examples/eval_tttd_multi_v2.py \
        --config areal/experimental/ttt_discover/examples/conf/fsdp_lora_vllm_ac1_qwen3_8b_distill.yaml
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
from areal.api.engine_api import WeightUpdateMeta
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


class TTTDMultiEvalTrainer(PPOTrainer):
    """Multi-model evaluation trainer.

    Mirrors ``TTTDDistillTrainer`` initialization but:
    - No training loop, no teacher actor, no saver/evaluator/recover
    - Loads each LoRA adapter on demand and pushes via ``update_weights()``
    - Runs a single rollout step per model and aggregates rewards
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

        # Create sampler from initial state (no teacher checkpoint loading)
        max_head_offpolicyness = getattr(config.rollout, 'max_head_offpolicyness', 2)
        max_version_history = max_head_offpolicyness + 1
        self.is_sync_mode = (max_head_offpolicyness == 0)
        if self.is_sync_mode and config.sampler.lazy_puct_sampling:
            config.sampler.lazy_puct_sampling = False

        teacher_sampler_config = copy.deepcopy(config.sampler)
        import tempfile
        teacher_sampler_config.checkpoint_dir = tempfile.mkdtemp(prefix="eval_fresh_sampler_")

        self.sampler = create_sampler_from_config(
            config=teacher_sampler_config,
            env_type=getattr(config.sampler, 'env_type', 'ac1'),
            max_version_history=max_version_history,
        )
        

        logger.info(
            f"[TeacherSampler] Loaded {len(self.sampler._states)} states, "
            f"T={self.sampler._T}, n_entries={len(self.sampler._n)}, "
            f"m_entries={len(self.sampler._m)}"
        )

        # Create actor (no critic/ref - eval only)
        self.actor = self._create_tttd_actor(config.actor)
        self.ref = None

        # Create dataloader (same as training)
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

        # Initialize rollout engine (same as training)
        self.rollout = self._init_rollout(config.rollout, is_eval=False)

        # Determine which models to evaluate
        eval_models = getattr(config, 'eval_models', None)
        if eval_models is None:
            eval_ckpt_dir = os.path.join(config.saver.fileroot, "eval_checkpoints")
            teacher_path = getattr(config, 'teacher_lora_path', None)
            if teacher_path is None:
                teacher_path = config.teacher_path
            student_path = getattr(config, 'student_lora_path', None)
            if student_path is None and os.path.isdir(os.path.join(eval_ckpt_dir, "student")):
                student_path = os.path.join(eval_ckpt_dir, "student")
            eval_models = {
                "baseline": None,  # Pure base model (no LoRA adapter)
                "teacher": teacher_path,
                "student": student_path,
            }
        # Keep baseline even when its path is None/empty; filter out other null paths
        eval_models = {k: v for k, v in eval_models.items() if v is not None or k == "baseline"}
        if not eval_models:
            raise ValueError("No evaluation models found. Set teacher_path / student_lora_path or eval_models.")
        self._eval_models = eval_models

        # Initialize models
        self._initialize_engines()

        # Connect sampler to actor for distributed synchronization
        self.actor.connect_sampler(self.sampler)

        # Setup weight update meta and connect to inference engine
        self._setup_weight_update_meta()

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
                self.weight_update_meta = WeightUpdateMeta.from_megatron_xccl(self.allocation_mode)
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
        logger.info(f"[Rank {dist.get_rank()}] connect_engine done")

    def _setup_stats_logger(self):
        """Setup stats logger only (no saver/recover/evaluator for eval)."""
        config = self.config
        ft_spec = FinetuneSpec(
            total_train_epochs=1,
            dataset_size=config.sampler.batch_size,
            train_batch_size=config.sampler.batch_size,
        )
        self.stats_logger = StatsLogger(config, ft_spec)

    def _zero_lora_weights(self, engine):
        """Zero out all LoRA parameters so the model behaves like the pure base model."""
        logger.info("[LoadAdapter] Zeroing out LoRA weights for base model evaluation")
        n_zeroed = 0
        for name, param in engine.model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.data.zero_()
                n_zeroed += 1
        logger.info(f"[LoadAdapter] Zeroed {n_zeroed} LoRA parameters")

    def _load_peft_lora_adapter(self, engine, path: str):
        """Load a PEFT LoRA adapter checkpoint into the FSDP-wrapped actor."""
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
                f"into actor model"
            )
            if unmatched:
                logger.warning(
                    f"[LoadAdapter] {len(unmatched)} keys could not be matched: "
                    f"{list(unmatched)[:10]}"
                )
        logger.info("[LoadAdapter] LoRA adapter loaded into actor successfully")

    def _clear_workflow_cache(self):
        """Clear ALL workflow and data generator caches across every layer."""
        targets = []

        # rollout._engine (RolloutController)
        if hasattr(self.rollout, '_engine'):
            if hasattr(self.rollout._engine, 'data_generator'):
                delattr(self.rollout._engine, 'data_generator')
                targets.append("rollout._engine.data_generator")
            if hasattr(self.rollout._engine, 'workflow_executor'):
                we = self.rollout._engine.workflow_executor
                if hasattr(we, 'data_generator'):
                    delattr(we, 'data_generator')
                    targets.append("workflow_executor.data_generator")

        # rollout directly
        if hasattr(self.rollout, 'data_generator'):
            delattr(self.rollout, 'data_generator')
            targets.append("rollout.data_generator")

        if targets:
            logger.info(f"[MultiEval] Cleared caches: {', '.join(targets)}")

    def _normalize_eval_batch(self, eval_batch) -> dict[str, Any]:
        """Ensure eval_batch is a single dict for downstream processing."""
        if isinstance(eval_batch, dict):
            return eval_batch
        if isinstance(eval_batch, list):
            if len(eval_batch) == 1:
                return eval_batch[0]
            from areal.utils.data import concat_batch
            batched, _meta = concat_batch(eval_batch)
            return batched
        raise TypeError(f"Unexpected eval_batch type: {type(eval_batch)}")

    def _run_single_model_eval(self, label: str, version: int, workflow_class, group_size):
        """Evaluate a single model with verification (single step, no training)."""
        lora_path = self._eval_models[label]

        if not lora_path:
            logger.info(f"[MultiEval-{label}] Evaluating pure base model (no LoRA adapter)")
            self._zero_lora_weights(self.actor)
        else:
            logger.info(f"[MultiEval-{label}] Loading LoRA from {lora_path}")
            self._load_peft_lora_adapter(self.actor, lora_path)

        # Push to vLLM via update_weights (same as training)
        logger.info(f"[MultiEval-{label}] Pushing weights to vLLM via update_weights()...")
        self.rollout.pause()
        versioned_meta = self.weight_update_meta.with_version(version)
        self.actor.update_weights(versioned_meta)
        self.actor.set_version(version)
        self.rollout.set_version(version)
        if dist.is_initialized():
            dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()
        self.rollout.resume()
        logger.info(f"[MultiEval-{label}] Weights pushed and rollout resumed.")

        # Create eval workflow with verification (same as training)
        eval_kwargs = self._workflow_kwargs.copy()
        eval_kwargs['reward_fn'] = tttd_reward_fn
        eval_workflow = workflow_class(**eval_kwargs)

        self._clear_workflow_cache()

        # Run rollout using the SAME dataloader and call signature as training
        eval_start = time.perf_counter()
        try:
            with stats_tracker.record_timing(f"eval_rollout_{label}"):
                eval_batch = self.actor.prepare_batch(
                    self.train_dataloader,
                    workflow=eval_workflow,
                    workflow_kwargs=None,
                    should_accept_fn=None,
                    group_size=group_size,
                    dynamic_bs=self.config.dynamic_bs,
                )
        except Exception as e:
            logger.error(f"[MultiEval-{label}] prepare_batch failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        eval_rollout_time = time.perf_counter() - eval_start

        # Normalize batch
        eval_batch = self._normalize_eval_batch(eval_batch)

        # Gather results
        if not isinstance(eval_batch, dict):
            logger.error(f"[MultiEval-{label}] eval_batch is not a dict (type={type(eval_batch).__name__}).")
            raise TypeError(f"eval_batch must be dict, got {type(eval_batch).__name__}")
        if "rewards" not in eval_batch:
            logger.error(f"[MultiEval-{label}] eval_batch missing 'rewards' key. Keys: {list(eval_batch.keys())}")
            raise KeyError("eval_batch missing 'rewards' key")

        local_rollouts = eval_batch["rewards"].shape[0]
        eval_rewards = eval_batch["rewards"].cpu().numpy()
        eval_max_reward = float(eval_rewards.max())
        eval_mean_reward = float(eval_rewards.mean())
        local_rewards_list = eval_rewards.tolist()

        # Pause rollout and sync GPU before PyTorch NCCL collectives to avoid
        # NCCL/GIL deadlock with vLLM backend threads.
        self.rollout.pause()
        current_platform.synchronize()
        torch.cuda.synchronize()

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

            # Write per-rank rewards to disk instead of all_gather_object
            # to avoid NCCL hang with large Python objects.
            rank = self.actor.data_parallel_rank
            local_rewards_path = os.path.join(
                self.config.saver.fileroot,
                f"eval_rewards_{label}_rank{rank}.json"
            )
            os.makedirs(self.config.saver.fileroot, exist_ok=True)
            with open(local_rewards_path, 'w') as f:
                json.dump(local_rewards_list, f)

            all_rewards_list = local_rewards_list  # rank 0 merges from files later
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

        self.rollout.resume()
        return result

    def run_eval(self, workflow_class, workflow_kwargs=None):
        """Run evaluation on all configured models."""
        config = self.config

        # Setup workflow kwargs (same as training)
        if workflow_kwargs is not None:
            self._workflow_kwargs = workflow_kwargs.copy()
            self._workflow_kwargs['lazy_sampling'] = config.sampler.lazy_puct_sampling
            if config.sampler.lazy_puct_sampling and 'sampler' not in self._workflow_kwargs:
                self._workflow_kwargs['sampler'] = self.sampler
                self._workflow_kwargs['dp_rank'] = self.actor.dp_rank
                self._workflow_kwargs['dp_world_size'] = self.actor.data_parallel_world_size

        is_dp_head = self.actor.rank == 0
        group_size = config.gconfig.n_samples

        logger.info(f"[MultiEval] Starting evaluation of {len(self._eval_models)} models: {list(self._eval_models.keys())}")

        # Evaluate each model sequentially
        all_results = {}
        for idx, (label, path) in enumerate(self._eval_models.items()):
            if dist.is_initialized():
                dist.barrier()
            all_results[label] = self._run_single_model_eval(
                label, idx , workflow_class, group_size
            )

        # Merge per-rank reward files on rank 0
        if is_dp_head and dist.is_initialized():
            for label in self._eval_models:
                merged_rewards = []
                for rank in range(self.actor.data_parallel_world_size):
                    rewards_path = os.path.join(
                        config.saver.fileroot,
                        f"eval_rewards_{label}_rank{rank}.json"
                    )
                    if os.path.exists(rewards_path):
                        with open(rewards_path, 'r') as f:
                            merged_rewards.extend(json.load(f))
                all_results[label]["all_rewards"] = merged_rewards

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
            logger.info("EVALUATION COMPARISON")
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
