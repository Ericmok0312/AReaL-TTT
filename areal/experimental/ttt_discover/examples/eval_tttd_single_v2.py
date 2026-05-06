#!/usr/bin/env python3
"""
TTT-Discover Single-Model Evaluation (v2).

Reuses the training infrastructure (PPOTrainer, TTTDActor, rollout engine)
but skips all training logic. Loads a specified LoRA adapter at init time,
pushes it to vLLM once via update_weights() (exactly like training does),
then runs a single evaluation rollout.

Usage (via launcher, same as training):
    python -m areal.infra.launcher.local \
        eval_tttd_single_v2.py \
        --lora-path <adapter_dir> \
        --config conf/fsdp_lora_vllm_ac1_qwen3_8b_distill.yaml
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
from areal.api.io_struct import FinetuneSpec, WeightUpdateMeta
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

logger = logging.getLogger("eval_tttd_single_v2")


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


class TTTDSingleEvalTrainer(PPOTrainer):
    """Single-model evaluation trainer.

    Mirrors ``TTTDDistillTrainer`` initialization but:
    - No teacher actor (no KL computation)
    - No training loop
    - Loads an external LoRA adapter via the corrected PEFT key mapping
    - Calls update_weights() exactly once (same pattern as training)
    """

    def __init__(self, config: TTTDDistillConfig, eval_lora_path: str):
        self.config = config
        self.eval_lora_path = eval_lora_path
        rank = int(os.getenv("RANK", "0"))
        if is_single_controller():
            logging.setup_file_logging(StatsLogger.get_log_path(config.stats_logger))

        # Load tokenizer and processor
        self.processor, self.tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)

        # Scheduler (skip if null, same guard as training)
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

        # Initialize inference engines (same as training, no lora_path pre-loading).
        # Force lora_modules to None to avoid vLLM startup errors if the YAML
        # contains a mis-formatted JSON string. We rely entirely on update_weights().
        config.vllm.lora_modules = None
        self.rollout = self._init_rollout(config.rollout, is_eval=False)

        # Initialize models
        self._initialize_engines()

        # Connect sampler to actor for distributed synchronization
        self.actor.connect_sampler(self.sampler)

        # Setup weight update meta
        self._setup_weight_update_meta()

        # Setup stats logger only (no saver/recover/evaluator for eval)
        self._setup_stats_logger()

        # Store workflow kwargs for later use
        self._workflow_kwargs = {}

        # ------------------------------------------------------------------
        # Load evaluation LoRA adapter into actor
        # ------------------------------------------------------------------
        self._load_peft_lora_adapter(self.actor, eval_lora_path)

        # ------------------------------------------------------------------
        # Push to vLLM once (exact same sequence as training's update_weights)
        # ------------------------------------------------------------------
        logger.info("[SingleEval] Pushing LoRA weights to vLLM via update_weights()...")
        self.rollout.pause()
        self.actor.update_weights(self.weight_update_meta)
        self.actor.set_version(1)
        self.rollout.set_version(1)
        dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()
        self.rollout.resume()
        logger.info("[SingleEval] Weights pushed and rollout resumed.")

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

    def _load_peft_lora_adapter(self, engine, path: str):
        """Load a PEFT LoRA adapter checkpoint into the FSDP-wrapped actor.

        AReaL does not expose a dedicated ``load_adapter`` API for LoRA-only
        checkpoints, so we do it manually with the correct key mapping:

        1. PEFT ``save_pretrained`` strips the ``base_model.model.`` prefix.
        2. PEFT ``save_pretrained`` strips the ``.default`` adapter suffix.
        3. FSDP2 ``set_model_state_dict`` expects the full param names as they
           appear in the model, i.e. ``base_model.model....lora_A.default.weight``.

        This is the corrected version of the helper that used to live in
        ``eval_tttd_distill.py``.
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
                # (1) Restore base_model.model. prefix if PEFT stripped it
                if not k.startswith("base_model.model."):
                    k = f"base_model.model.{k}"
                # (2) Restore .default adapter suffix (PEFT default adapter name)
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
            strict=False,  # LoRA is a subset; keep False to be safe
        )
        set_model_state_dict(engine.model, fixed_state, options=options)

        # Verify loading succeeded
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
        """Clear workflow executor cache so a new workflow can be used."""
        workflow_executor = None
        if hasattr(self.rollout, '_engine') and hasattr(self.rollout._engine, 'workflow_executor'):
            workflow_executor = self.rollout._engine.workflow_executor
        elif hasattr(self.rollout, 'workflow_executor'):
            workflow_executor = self.rollout.workflow_executor

        if workflow_executor is not None and hasattr(workflow_executor, 'data_generator'):
            delattr(workflow_executor, 'data_generator')
            logger.info("[SingleEval] Cleared workflow executor cache")

        if hasattr(self.rollout, 'data_generator'):
            delattr(self.rollout, 'data_generator')
            logger.info("[SingleEval] Cleared rollout controller data_generator")

    def _normalize_eval_batch(self, eval_batch) -> dict[str, Any]:
        """Ensure eval_batch is a single dict for downstream processing.

        prepare_batch may return list[dict] depending on the backend path;
        concat if necessary so that evaluation metrics work uniformly.
        """
        if isinstance(eval_batch, dict):
            return eval_batch
        if isinstance(eval_batch, list):
            if len(eval_batch) == 1:
                return eval_batch[0]
            from areal.utils.data import concat_batch
            batched, _meta = concat_batch(eval_batch)
            return batched
        raise TypeError(f"Unexpected eval_batch type: {type(eval_batch)}")

    def run_eval(self, workflow_class, workflow_kwargs=None):
        """Run evaluation on the single pre-loaded model."""
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
            logger.warning("[SingleEval] No initial states found, using first available state")

        logger.info(f"[SingleEval] Using {len(initial_states)} initial states for evaluation")

        # Create temp dataloader with initial states only
        temp_sampler = _InitialStateSampler(initial_states)
        eval_batch_size = min(len(initial_states), config.sampler.batch_size)
        eval_batch_size = max(eval_batch_size, 1)

        temp_dataloader = create_tttd_dataloader(
            temp_sampler,
            rank=self.actor.data_parallel_rank,
            world_size=self.actor.data_parallel_world_size,
            batch_size=eval_batch_size,
            lazy_sampling=False,
        )
        logger.info(f"[SingleEval] Created temporary dataloader with batch size {eval_batch_size}")

        # Create eval workflow with verification
        eval_kwargs = self._workflow_kwargs.copy()
        eval_kwargs['reward_fn'] = tttd_reward_fn
        eval_workflow = workflow_class(**eval_kwargs)

        self._clear_workflow_cache()

        # Run rollout
        eval_start = time.perf_counter()
        try:
            with stats_tracker.record_timing("eval_rollout"):
                eval_batch = self.actor.prepare_batch(
                    temp_dataloader,
                    workflow=eval_workflow,
                    workflow_kwargs=None,
                    should_accept_fn=None,
                    group_size=group_size,
                    dynamic_bs=False,
                )
        except Exception as e:
            logger.error(f"[SingleEval] prepare_batch failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        eval_rollout_time = time.perf_counter() - eval_start

        # Normalize batch (prepare_batch may return list[dict] on some backends)
        eval_batch = self._normalize_eval_batch(eval_batch)

        # Gather results
        if "rewards" not in eval_batch:
            logger.error(f"[SingleEval] eval_batch missing 'rewards' key. Keys: {list(eval_batch.keys())}")
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

        model_label = os.path.basename(self.eval_lora_path.rstrip("/"))
        result = {
            "model": model_label,
            "lora_path": self.eval_lora_path,
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
            f"[SingleEval] {model_label}: max={eval_max_reward:.4f}, mean={eval_mean_reward:.4f}, "
            f"children={len(eval_children)}, failed={len(eval_failed)}"
        )

        # Save results
        if is_dp_head:
            output_path = os.path.join(
                config.saver.fileroot,
                f"eval_single_{model_label}.json"
            )
            os.makedirs(config.saver.fileroot, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"[SingleEval] Results saved to {output_path}")

        if dist.is_initialized():
            dist.barrier()

        return result

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
    lora_path = os.environ.get("EVAL_LORA_PATH")
    if lora_path is None:
        raise ValueError(
            "Environment variable EVAL_LORA_PATH must be set.\n"
            "Example: EVAL_LORA_PATH=./outputs/teacher python -m areal.infra.launcher.local "
            "eval_tttd_single_v2.py --config conf/fsdp_lora_vllm_ac1_qwen3_8b_distill.yaml"
        )

    if not os.path.isdir(lora_path):
        raise ValueError(f"EVAL_LORA_PATH does not exist or is not a directory: {lora_path}")

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

    with TTTDSingleEvalTrainer(config, lora_path) as trainer:
        trainer.run_eval(
            workflow_class=TTTDiscoverWorkflowV2,
            workflow_kwargs=workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
