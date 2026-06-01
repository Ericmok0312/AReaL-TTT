#!/usr/bin/env python3
"""
TTT-Discover Teacher-with-Hint Evaluation.

Evaluates the teacher model with privileged hint information appended to prompts.
This diagnoses whether the teacher can actually leverage hints to generate
high-reward code.

Usage:
    python areal/experimental/ttt_discover/examples/eval_teacher_with_hint.py \
        --config areal/experimental/ttt_discover/examples/conf/fsdp_lora_vllm_ac1_qwen3_8b_distill_H.yaml
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

logger = logging.getLogger("eval_teacher_with_hint")


class TeacherWithHintEvalTrainer(PPOTrainer):
    """Evaluate teacher model with privileged hints.
    
    Mirrors eval_tttd_multi_v2.py but:
    - Only evaluates the teacher model
    - Wraps env with HintEnvWrapper to append hints to prompts
    """

    def __init__(self, config: TTTDDistillConfig):
        self.config = config
        rank = int(os.getenv("RANK", "0"))
        if is_single_controller():
            logging.setup_file_logging(StatsLogger.get_log_path(config.stats_logger))

        # Load tokenizer and processor
        self.processor, self.tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)

        # Scheduler
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

        # Create base env
        self.base_env = create_env_from_config(config)

        # Create hint sampler (loaded from teacher checkpoint with trained states)
        max_head_offpolicyness = getattr(config.rollout, 'max_head_offpolicyness', 2)
        max_version_history = max_head_offpolicyness + 1
        self.is_sync_mode = (max_head_offpolicyness == 0)
        if self.is_sync_mode and config.sampler.lazy_puct_sampling:
            config.sampler.lazy_puct_sampling = False

        teacher_sampler_config = copy.deepcopy(config.sampler)
        if config.teacher_sampler_checkpoint:
            teacher_sampler_config.checkpoint_dir = config.teacher_sampler_checkpoint

        self.hint_sampler = create_sampler_from_config(
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
                logger.info(f"[HintSampler] Loading latest checkpoint at step {latest_step}")
                self.hint_sampler._load(latest_step)
                self.hint_sampler._current_step = 0
            else:
                logger.warning(
                    f"[HintSampler] No checkpoint found in {config.teacher_sampler_checkpoint}. "
                    f"Using fresh sampler state."
                )

        logger.info(
            f"[HintSampler] Loaded {len(self.hint_sampler._states)} states, "
            f"T={self.hint_sampler._T}"
        )

        # Store base env directly (hint will be applied via workflow hint_fn)
        self.env = self.base_env

        # Determine eval prompt mode
        self.eval_prompt_mode = getattr(config, 'eval_prompt_mode', 'hint')
        logger.info(f"[Eval] Prompt mode: {self.eval_prompt_mode}")

        if self.eval_prompt_mode == 'continuation':
            # Continuation mode: use hint_sampler directly for both prompt and states
            # No need for fresh sampler
            self.sampler = self.hint_sampler
        else:
            # Hint mode: fresh initial-state sampler for eval
            import tempfile
            fresh_log_path = tempfile.mkdtemp(prefix="eval_fresh_sampler_")
            from areal.experimental.ttt_discover.sampler import create_sampler
            self.sampler = create_sampler(
                sampler_type=getattr(config.sampler, 'type', 'puct'),
                log_path=fresh_log_path,
                env_type=getattr(config.sampler, 'env_type', 'ac1'),
                budget_s=getattr(config.sampler, 'save_freq', 100),
                initial_exp_type=getattr(config.sampler, 'initial_exp_type', 'best_available'),
                batch_size=getattr(config.sampler, 'batch_size', 8),
                resume_step=None,
                c_puct=getattr(config.sampler, 'c_puct', 1.5),
                gamma=getattr(config.sampler, 'gamma', 0.95),
                max_children=getattr(config.sampler, 'max_children', 100),
                max_states=getattr(config.sampler, 'max_states', 10000),
                top_k=getattr(config.sampler, 'top_k', 1000),
                temperature=getattr(config.sampler, 'temperature', 1.0),
                max_version_history=max_version_history,
                sampling_strategy=getattr(config.sampler, 'sampling_strategy', 'puct'),
            )

        # Create actor
        self.actor = self._create_tttd_actor(config.actor)
        self.ref = None

        # Create dataloader
        self.train_dataloader = create_tttd_dataloader(
            state_sampler=self.sampler,
            rank=self.actor.data_parallel_rank,
            world_size=self.actor.data_parallel_world_size,
            batch_size=config.eval_batch_size,
            lazy_sampling=config.sampler.lazy_puct_sampling,
        )
        self.train_dataset = self.train_dataloader.dataset
        self.valid_dataloader = None
        self.valid_dataset = None

        config.rollout.consumer_batch_size = config.eval_batch_size
        self.rollout = self._init_rollout(config.rollout, is_eval=False)

        # Only evaluate teacher
        teacher_path = getattr(config, 'teacher_lora_path', None)
        if teacher_path is None:
            teacher_path = config.teacher.path
        self._teacher_path = teacher_path
        logger.info(f"[Eval] Will evaluate teacher with hint: {teacher_path}")

        # Initialize
        self._initialize_engines()
        self.actor.connect_sampler(self.sampler)
        self._setup_weight_update_meta()
        self._setup_stats_logger()
        self._workflow_kwargs = {}

    def _build_hint(self, privileged_state) -> str:
        """Build hint from privileged state (same logic as training)."""
        hint_parts = []
        if privileged_state.code and privileged_state.code.strip():
            hint_parts.append(f"```python\n{privileged_state.code.strip()}\n```")
        if privileged_state.value is not None:
            raw_score = -privileged_state.value
            hint_parts.append(f"This achieves a score of {raw_score:.6f}.")
        hint_text = "\n".join(hint_parts)
        return (
            f"\n\n[Hint] A known good approach for this problem:\n"
            f"{hint_text}\n"
            f"Your task is to reuse the above algorithm with minimal modifications. "
            f"Do NOT invent a completely different approach. "
            f"The hint code is proven to work — your job is to adapt it, not replace it.\n"
        )

    def _create_tttd_actor(self, actor_config):
        actor = TTTDActor(config=actor_config)
        actor.create_process_group(parallel_strategy=self.allocation_mode.train)
        return actor

    def _initialize_engines(self):
        ft_spec = FinetuneSpec(
            total_train_epochs=1,
            dataset_size=self.config.eval_batch_size,
            train_batch_size=self.config.eval_batch_size,
        )
        self.actor.initialize(addr=None, ft_spec=ft_spec, alloc_mode=self.allocation_mode, role="actor")

    def _setup_weight_update_meta(self):
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
        config = self.config
        ft_spec = FinetuneSpec(
            total_train_epochs=1,
            dataset_size=config.eval_batch_size,
            train_batch_size=config.eval_batch_size,
        )
        self.stats_logger = StatsLogger(config, ft_spec)

    def _load_peft_lora_adapter(self, engine, path: str):
        from safetensors.torch import load_file
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )
        adapter_path = os.path.join(path, "adapter_model.safetensors")
        if not os.path.isfile(adapter_path):
            raise ValueError(f"LoRA adapter not found at {adapter_path}")
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
        logger.info("[LoadAdapter] LoRA adapter loaded into actor successfully")

    def _clear_workflow_cache(self):
        targets = []
        if hasattr(self.rollout, '_engine'):
            if hasattr(self.rollout._engine, 'data_generator'):
                delattr(self.rollout._engine, 'data_generator')
                targets.append("rollout._engine.data_generator")
            if hasattr(self.rollout._engine, 'workflow_executor'):
                we = self.rollout._engine.workflow_executor
                if hasattr(we, 'data_generator'):
                    delattr(we, 'data_generator')
                    targets.append("workflow_executor.data_generator")
        if hasattr(self.rollout, 'data_generator'):
            delattr(self.rollout, 'data_generator')
            targets.append("rollout.data_generator")
        if targets:
            logger.info(f"[Eval] Cleared caches: {', '.join(targets)}")

    def run(self):
        """Run teacher-with-hint evaluation."""
        config = self.config

        # Load teacher LoRA
        logger.info(f"[Eval] Loading teacher LoRA from {self._teacher_path}")
        self._load_peft_lora_adapter(self.actor, self._teacher_path)

        # Push to vLLM
        logger.info("[Eval] Pushing weights to vLLM...")
        self.rollout.pause()
        versioned_meta = self.weight_update_meta.with_version(0)
        self.actor.update_weights(versioned_meta)
        self.actor.set_version(0)
        self.rollout.set_version(0)
        if dist.is_initialized():
            dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()
        self.rollout.resume()
        logger.info("[Eval] Weights pushed and rollout resumed.")

        # Create workflow with hint_fn hook
        config = self.config
        local_batch_size = config.eval_batch_size // self.actor.data_parallel_world_size
        group_size = config.gconfig.n_samples
        
        if self.eval_prompt_mode == 'continuation':
            # Continuation mode: no hint, prompt is based on privileged state directly
            workflow_kwargs = dict(
                env=self.env,
                gconfig=config.gconfig,
                tokenizer=self.tokenizer,
                enable_thinking=getattr(config, 'enable_thinking', False),
                max_prompt_thinking_tokens=getattr(config, 'max_prompt_thinking_tokens', 26000),
                batch_size=local_batch_size,
                group_size=group_size,
                lazy_sampling=config.sampler.lazy_puct_sampling,
                vllm_concurrency=getattr(config.sampler, 'vllm_concurrency', None),
                execution_concurrency=getattr(config.sampler, 'execution_concurrency', 64),
                reward_fn=tttd_reward_fn,
                # No hint_fn
            )
        else:
            # Hint mode: append hint to prompt
            def hint_fn(state):
                hint_states = self.hint_sampler.sample_states(1)
                if hint_states:
                    return self._build_hint(hint_states[0])
                return ""
            
            workflow_kwargs = dict(
                env=self.env,
                gconfig=config.gconfig,
                tokenizer=self.tokenizer,
                enable_thinking=getattr(config, 'enable_thinking', False),
                max_prompt_thinking_tokens=getattr(config, 'max_prompt_thinking_tokens', 26000),
                batch_size=local_batch_size,
                group_size=group_size,
                lazy_sampling=config.sampler.lazy_puct_sampling,
                vllm_concurrency=getattr(config.sampler, 'vllm_concurrency', None),
                execution_concurrency=getattr(config.sampler, 'execution_concurrency', 64),
                reward_fn=tttd_reward_fn,
                hint_fn=hint_fn,
            )
        eval_workflow = TTTDiscoverWorkflowV2(**workflow_kwargs)

        self._clear_workflow_cache()

        group_size = config.gconfig.n_samples
        num_steps = getattr(config, 'eval_steps', 4)  # 4 steps * batch_size * group_size rollouts

        all_rewards = []
        step_rewards = []

        for step in range(num_steps):
            logger.info(f"[Eval] Step {step + 1}/{num_steps}")
            
            # Log sampled states before rollout
            preview_states = self.sampler.sample_states(config.eval_batch_size)
            state_values = []
            for i, s in enumerate(preview_states):
                val = getattr(s, 'value', None)
                raw_score = -val if val is not None else None
                state_values.append(f"state{i}: value={val:.4f}, raw_score={raw_score:.4f}" if val is not None else f"state{i}: N/A")
            logger.info(f"[Eval] Sampled states for step {step+1}: {', '.join(state_values)}")
            
            if self.eval_prompt_mode == 'hint':
                # Also log hint states
                hint_states = self.hint_sampler.sample_states(config.eval_batch_size)
                hint_values = []
                for i, s in enumerate(hint_states):
                    val = getattr(s, 'value', None)
                    raw_score = -val if val is not None else None
                    hint_values.append(f"hint{i}: value={val:.4f}, raw_score={raw_score:.4f}" if val is not None else f"hint{i}: N/A")
                logger.info(f"[Eval] Hint states for step {step+1}: {', '.join(hint_values)}")
            
            eval_start = time.perf_counter()
            try:
                with stats_tracker.record_timing("eval_rollout_teacher_hint"):
                    eval_batch = self.rollout.prepare_batch(
                        self.train_dataloader,
                        workflow=eval_workflow,
                        workflow_kwargs=None,
                        should_accept_fn=None,
                        group_size=group_size,
                        dynamic_bs=self.config.dynamic_bs,
                    )
            except Exception as e:
                logger.error(f"[Eval] prepare_batch failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise

            # Normalize batch
            if isinstance(eval_batch, list):
                if len(eval_batch) == 1:
                    eval_batch = eval_batch[0]
                else:
                    from areal.utils.data import concat_batch
                    eval_batch, _ = concat_batch(eval_batch)

            if "rewards" not in eval_batch:
                raise KeyError("eval_batch missing 'rewards' key")

            eval_rewards = eval_batch["rewards"].cpu().numpy()
            step_max = float(eval_rewards.max())
            step_mean = float(eval_rewards.mean())
            step_rewards.append((step_max, step_mean))
            all_rewards.extend(eval_rewards.tolist())

            logger.info(
                f"[Eval] Step {step + 1}: max_reward={step_max:.4f} | "
                f"mean_reward={step_mean:.4f} | rollouts={len(eval_rewards)}"
            )

            self.rollout.pause()
            current_platform.synchronize()
            torch.cuda.synchronize()

        # Final aggregation
        all_rewards_arr = torch.tensor(all_rewards, dtype=torch.float32)
        if dist.is_initialized():
            # Gather from all ranks
            local_count = torch.tensor([len(all_rewards)], dtype=torch.int32, device=self.actor.device)
            local_max = torch.tensor([all_rewards_arr.max().item()], dtype=torch.float32, device=self.actor.device)
            local_sum = torch.tensor([all_rewards_arr.sum().item()], dtype=torch.float32, device=self.actor.device)
            
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
            
            global_max = local_max.item()
            global_mean = (local_sum / local_count).item() if local_count.item() > 0 else 0.0
            global_count = int(local_count.item())
        else:
            global_max = all_rewards_arr.max().item()
            global_mean = all_rewards_arr.mean().item()
            global_count = len(all_rewards)

        # Print results
        logger.info("=" * 60)
        if self.eval_prompt_mode == 'continuation':
            logger.info("TEACHER CONTINUATION - EVALUATION RESULTS")
        else:
            logger.info("TEACHER WITH HINT - EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Teacher path: {self._teacher_path}")
        logger.info(f"Prompt mode: {self.eval_prompt_mode}")
        logger.info(f"Total rollouts: {global_count}")
        logger.info(f"Max reward: {global_max:.4f}")
        logger.info(f"Mean reward: {global_mean:.4f}")
        logger.info("-" * 60)
        for i, (smax, smean) in enumerate(step_rewards):
            logger.info(f"  Step {i + 1}: max={smax:.4f} | mean={smean:.4f}")
        logger.info("=" * 60)

        # Save rewards to disk
        if is_single_controller():
            output_dir = os.path.join(
                config.saver.fileroot,
                config.experiment_name,
                config.trial_name,
            )
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "eval_teacher_hint_rewards.json")
            with open(output_path, 'w') as f:
                json.dump({
                    "teacher_path": self._teacher_path,
                    "num_rollouts": global_count,
                    "max_reward": global_max,
                    "mean_reward": global_mean,
                    "step_results": [
                        {"step": i + 1, "max": smax, "mean": smean}
                        for i, (smax, smean) in enumerate(step_rewards)
                    ],
                    "all_rewards": all_rewards,
                }, f, indent=2)
            logger.info(f"[Eval] Results saved to {output_path}")


def main(args):
    config, _ = load_expr_config(args, TTTDDistillConfig)
    
    if config.tokenizer_path:
        from areal.utils.hf_utils import load_hf_tokenizer
        tokenizer = load_hf_tokenizer(config.tokenizer_path)
        if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
            config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
        if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
            config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    
    # Override eval batch size if needed
    if not hasattr(config, 'eval_batch_size'):
        config.eval_batch_size = getattr(config.sampler, 'batch_size', 8)
    if not hasattr(config, 'eval_steps'):
        config.eval_steps = 4  # 4 steps default

    trainer = TeacherWithHintEvalTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main(sys.argv[1:])
