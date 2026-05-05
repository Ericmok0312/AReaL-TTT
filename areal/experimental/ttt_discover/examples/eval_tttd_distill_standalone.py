#!/usr/bin/env python3
"""
Standalone TTT-Discover Distillation Evaluation using vLLM directly.

This script bypasses AReaL's distributed training engine and inference weight-update
mechanism (which can hang during LoRA adapter switching) by using vLLM's native LLM
class directly for generation.  It reuses TTTDiscoverWorkflowV2 for prompt
construction, teacher-forcing, code verification, reward computation, and child-state
creation.

**Multi-GPU via multiprocessing** – the script automatically detects available GPUs
and launches one independent vLLM instance per GPU using ``spawn`` (so each worker
starts from a clean Python interpreter, avoiding the fork-NCCL deadlock that breaks
torchrun).  Results are gathered in the main process and saved to JSON.

Usage::

    python -m areal.experimental.ttt_discover.examples.eval_tttd_distill_standalone \
        --config areal/experimental/ttt_discover/examples/conf/fsdp_lora_vllm_ac1_qwen3_8b_distill.yaml

Output::

    ${saver.fileroot}/eval_comparison_standalone.json
"""

import os
import sys

# vLLM V1 engine defaults to fork() on Linux, which copies the parent's
# CUDA/NCCL state and deadlocks when PyTorch DDP or other NCCL users are present.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import copy
import json
import math
import time
import multiprocessing as mp
from typing import Any

import torch

from areal.api.cli_args import parse_cli_args, to_structured_cfg
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.utils import logging

from areal.experimental.ttt_discover.config import (
    TTTDDistillConfig,
    create_env_from_config,
)
from areal.experimental.ttt_discover.sampler import (
    create_sampler_from_config,
    _find_latest_sampler_step,
)
from areal.experimental.ttt_discover.state import state_from_dict

from omegaconf import OmegaConf

logger = logging.getLogger("eval_tttd_distill_standalone")


def _shard_states(states: list, rank: int, world_size: int) -> list:
    """Simple contiguous shard."""
    n = len(states)
    per_rank = n // world_size
    rem = n % world_size
    start = rank * per_rank + min(rank, rem)
    end = start + per_rank + (1 if rank < rem else 0)
    return states[start:end]


def _is_lora_adapter(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "adapter_model.safetensors"))


def _worker_main(
    worker_id: int,
    n_workers: int,
    config_dict: dict,
    states_dicts: list[dict],
    models_to_eval: list[tuple[str, str]],
    batch_size: int,
    group_size: int,
    result_queue: mp.Queue,
):
    """Run evaluation in an isolated child process (one per GPU)."""
    # Isolate this worker to a single GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)

    # Re-import everything inside the spawned process
    import asyncio
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    from areal.api.cli_args import to_structured_cfg
    from areal.utils.hf_utils import load_hf_processor_and_tokenizer
    from areal.experimental.ttt_discover.config import TTTDDistillConfig, create_env_from_config
    from areal.experimental.ttt_discover.workflow_v2 import TTTDiscoverWorkflowV2
    from areal.experimental.ttt_discover.reward import tttd_reward_fn
    from areal.api.io_struct import ModelRequest, ModelResponse
    from areal.utils import logging as areal_logging

    worker_logger = areal_logging.getLogger(f"worker{worker_id}")

    # Restore config from plain dict
    raw_cfg = OmegaConf.create(config_dict)
    config = OmegaConf.to_object(to_structured_cfg(raw_cfg, TTTDDistillConfig))

    # Tokenizer & env
    _processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)

    env = create_env_from_config(config)

    # Restore states
    states = [state_from_dict(d) for d in states_dicts]

    # vLLM settings
    vllm_cfg = config_dict.get("vllm", {})
    base_model_path = config.actor.path

    worker_logger.info(f"[Worker {worker_id}] Loading vLLM on GPU {worker_id}")
    llm = LLM(
        model=base_model_path,
        tokenizer=config.tokenizer_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.85),
        max_model_len=vllm_cfg.get("max_model_len", 32768),
        enforce_eager=vllm_cfg.get("enforce_eager", False),
        enable_lora=vllm_cfg.get("enable_lora", True),
        max_lora_rank=vllm_cfg.get("max_lora_rank", getattr(config.actor, "lora_rank", 64)),
        trust_remote_code=True,
    )
    worker_logger.info(f"[Worker {worker_id}] vLLM ready")

    # Simple engine wrapper (same as main script)
    class _SimpleEngine:
        def __init__(self, llm, tokenizer, lora_request=None):
            self.llm = llm
            self.tokenizer = tokenizer
            self.lora_request = lora_request
            self._version = 0

        def get_version(self):
            return self._version

        def set_version(self, v):
            self._version = v

        def set_lora(self, lora_request):
            self.lora_request = lora_request

        async def agenerate(self, req: ModelRequest) -> ModelResponse:
            from vllm import SamplingParams
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._generate_sync, req, SamplingParams)

        def _generate_sync(self, req, SamplingParamsCls):
            temperature = 0.0 if req.gconfig.greedy else req.gconfig.temperature
            sp = SamplingParamsCls(
                temperature=temperature,
                top_p=req.gconfig.top_p,
                top_k=-1 if req.gconfig.top_k >= 1e7 else req.gconfig.top_k,
                max_tokens=req.gconfig.max_new_tokens,
                min_tokens=req.gconfig.min_new_tokens,
                stop_token_ids=req.gconfig.stop_token_ids or [],
                ignore_eos=req.gconfig.ignore_eos,
            )
            # vLLM 0.17.0+ (V1 engine) removed the legacy prompt_token_ids kwarg.
            # Use TokensPrompt via the new ``inputs`` parameter instead.
            try:
                from vllm.inputs import TokensPrompt
                inputs = TokensPrompt(prompt_token_ids=req.input_ids)
                outputs = self.llm.generate(
                    inputs,
                    sampling_params=sp,
                    lora_request=self.lora_request,
                )
            except Exception:
                # Fallback for older vLLM versions
                outputs = self.llm.generate(
                    prompts=None,
                    sampling_params=sp,
                    prompt_token_ids=[req.input_ids],
                    lora_request=self.lora_request,
                )
            out = outputs[0].outputs[0]
            output_tokens = list(out.token_ids)
            stop_reason = "length" if out.finish_reason == "length" else "stop"
            return ModelResponse(
                input_tokens=req.input_ids,
                output_tokens=output_tokens,
                output_logprobs=[0.0] * len(output_tokens),
                output_versions=[self._version] * len(output_tokens),
                stop_reason=stop_reason,
                tokenizer=self.tokenizer,
            )

    # Evaluate each model sequentially (same LLM, swap LoRA)
    per_model_results: dict[str, dict] = {}

    for model_name, model_path in models_to_eval:
        worker_logger.info(f"[Worker {worker_id}] Evaluating {model_name}")

        engine = _SimpleEngine(llm, tokenizer)
        if model_path and _is_lora_adapter(model_path):
            engine.set_lora(LoRARequest(model_name, 1, model_path))
            worker_logger.info(f"[Worker {worker_id}] Loaded LoRA {model_path}")

        workflow = TTTDiscoverWorkflowV2(
            env=env,
            gconfig=config.gconfig,
            tokenizer=tokenizer,
            sampler=None,
            reward_fn=tttd_reward_fn,
            enable_thinking=getattr(config, "enable_thinking", False),
            max_prompt_thinking_tokens=getattr(config, "max_prompt_thinking_tokens", 26000),
            batch_size=batch_size,
            group_size=group_size,
            lazy_sampling=False,
            dp_rank=worker_id,
            dp_world_size=n_workers,
        )
        workflow.reset()
        workflow.set_current_version(0)

        eval_batch_size = min(len(states), batch_size)
        eval_batch_size = max(eval_batch_size, 1)
        num_batches = (len(states) + eval_batch_size - 1) // eval_batch_size

        rewards: list[float] = []
        eval_start = time.perf_counter()

        for batch_idx in range(num_batches):
            batch_start = batch_idx * eval_batch_size
            batch_end = min(batch_start + eval_batch_size, len(states))
            batch_states = states[batch_start:batch_end]

            batch_begin = time.perf_counter()
            for state in batch_states:
                for _ in range(group_size):
                    data = {"_state_obj": state}
                    try:
                        trajectory = asyncio.run(workflow.arun_episode(engine, data))
                    except Exception as e:
                        worker_logger.error(f"arun_episode failed: {e}")
                        trajectory = None
                    if trajectory is None:
                        continue
                    reward = float(trajectory["rewards"][0])
                    rewards.append(reward)
            batch_elapsed = time.perf_counter() - batch_begin
            worker_logger.info(
                f"[Worker {worker_id}] {model_name} batch {batch_idx + 1}/{num_batches} | "
                f"parents={len(batch_states)} rollouts={len(batch_states) * group_size} "
                f"time={batch_elapsed:.1f}s"
            )

        rollout_time = time.perf_counter() - eval_start

        updates = workflow.get_pending_updates(clear=True)
        if len(updates) == 5:
            children, parents, failed, _metadata, _ = updates
        else:
            children, parents, failed, _metadata = updates

        per_model_results[model_name] = {
            "rewards": rewards,
            "n_children": len(children),
            "n_parents": len(parents),
            "n_failed": len(failed),
            "rollout_time_s": rollout_time,
        }
        workflow.shutdown()

    result_queue.put({"worker_id": worker_id, "results": per_model_results})


def main(args):
    # ------------------------------------------------------------------
    # 1. Parse config in main process
    # ------------------------------------------------------------------
    raw_cfg, config_file = parse_cli_args(args)
    config = OmegaConf.to_object(to_structured_cfg(raw_cfg, TTTDDistillConfig))

    if not config.teacher_path:
        raise ValueError("teacher_path must be provided.")
    if not config.teacher_sampler_checkpoint:
        raise ValueError("teacher_sampler_checkpoint must be provided.")

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        n_gpus = 1
    logger.info(f"Detected {n_gpus} GPUs")

    # ------------------------------------------------------------------
    # 2. Load sampler & initial states
    # ------------------------------------------------------------------
    teacher_sampler_config = copy.deepcopy(config.sampler)
    if config.teacher_sampler_checkpoint:
        teacher_sampler_config.checkpoint_dir = config.teacher_sampler_checkpoint

    sampler = create_sampler_from_config(
        config=teacher_sampler_config,
        env_type=getattr(config.sampler, "env_type", "ac1"),
        max_version_history=2,
    )

    if config.teacher_sampler_checkpoint:
        latest_step = _find_latest_sampler_step(
            config.teacher_sampler_checkpoint,
            getattr(config.sampler, "type", "puct"),
        )
        if latest_step is not None:
            logger.info(f"[TeacherSampler] Loading latest checkpoint at step {latest_step}")
            sampler._load(latest_step)
            sampler._current_step = 0
        else:
            logger.warning("[TeacherSampler] No checkpoint found. Using fresh sampler state.")

    initial_states = sampler._initial_states
    if not initial_states:
        initial_states = [s for s in sampler._states if getattr(s, "timestep", 0) == 0]
    if not initial_states:
        initial_states = sampler._states[:1]
        logger.warning("[Eval] No initial states found, using first available state")

    logger.info(f"[Eval] {len(initial_states)} initial states total, GPUs={n_gpus}")

    # Serialize states for pickling across spawn
    states_dicts = [s.to_dict() for s in initial_states]
    shards = [_shard_states(states_dicts, i, n_gpus) for i in range(n_gpus)]

    # ------------------------------------------------------------------
    # 3. Read extra YAML fields (teacher/student lora, vllm settings)
    # ------------------------------------------------------------------
    config_dict = OmegaConf.to_container(raw_cfg, resolve=True)

    # Fix: YAML writes vllm.lora_modules as a JSON string, but vLLMConfig
    # expects a list. Parse it so to_structured_cfg doesn't crash in workers.
    vllm_section = config_dict.get("vllm", {})
    lora_modules_raw = vllm_section.get("lora_modules")
    if isinstance(lora_modules_raw, str):
        import json as _json
        try:
            vllm_section["lora_modules"] = _json.loads(lora_modules_raw)
        except Exception:
            vllm_section["lora_modules"] = None

    teacher_lora_path = config_dict.get("teacher_lora_path", config.teacher_path)
    student_lora_path = config_dict.get("student_lora_path", "")
    models_to_eval = [("teacher", teacher_lora_path), ("student", student_lora_path)]

    batch_size = config.sampler.batch_size
    group_size = config.gconfig.n_samples
    logger.info(f"[EvalConfig] batch_size={batch_size}, group_size={group_size}")

    # ------------------------------------------------------------------
    # 4. Launch one worker process per GPU (spawn -> clean CUDA state)
    # ------------------------------------------------------------------
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []

    for i in range(n_gpus):
        p = ctx.Process(
            target=_worker_main,
            args=(
                i,
                n_gpus,
                config_dict,
                shards[i],
                models_to_eval,
                batch_size,
                group_size,
                result_queue,
            ),
        )
        p.start()
        processes.append(p)

    # ------------------------------------------------------------------
    # 5. Gather results
    # ------------------------------------------------------------------
    worker_outputs = [result_queue.get() for _ in range(n_gpus)]

    for p in processes:
        p.join()

    # Merge per-model results across workers
    all_results: dict[str, dict] = {}
    for model_name, _ in models_to_eval:
        all_rewards: list[float] = []
        n_children = 0
        n_parents = 0
        n_failed = 0
        max_rollout_time = 0.0

        for out in worker_outputs:
            res = out["results"][model_name]
            all_rewards.extend(res["rewards"])
            n_children += res["n_children"]
            n_parents += res["n_parents"]
            n_failed += res["n_failed"]
            max_rollout_time = max(max_rollout_time, res["rollout_time_s"])

        all_results[model_name] = {
            "model": model_name,
            "global_rollouts": len(all_rewards),
            "max_reward": max(all_rewards) if all_rewards else 0.0,
            "mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
            "all_rewards": all_rewards,
            "n_children": n_children,
            "n_parents": n_parents,
            "n_failed": n_failed,
            "rollout_time_s": max_rollout_time,
            "children": [],
        }

    # ------------------------------------------------------------------
    # 6. Save JSON
    # ------------------------------------------------------------------
    comparison_path = os.path.join(config.saver.fileroot, "eval_comparison_standalone.json")
    os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
    with open(comparison_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"[Eval] Results saved to {comparison_path}")

    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPARISON (Initial States) – Standalone vLLM Multi-GPU")
    logger.info("=" * 70)
    for label in ["teacher", "student"]:
        r = all_results[label]
        logger.info(
            f"{label:10s} | max_reward={r['max_reward']:.4f} | "
            f"mean_reward={r['mean_reward']:.4f} | "
            f"rollouts={r['global_rollouts']} | "
            f"children={r['n_children']} | failed={r['n_failed']}"
        )
    logger.info("=" * 70)
    logger.info("[Eval] Done.")


if __name__ == "__main__":
    main(sys.argv[1:])
