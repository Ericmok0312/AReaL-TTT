#!/usr/bin/env python3
"""
Standalone TTT-Discover Distillation Evaluation using vLLM directly.

This script bypasses AReaL's distributed training engine and inference weight-update
mechanism (which can hang during LoRA adapter switching) by using vLLM's native LLM
class directly for generation.  It reuses TTTDiscoverWorkflowV2 for prompt
construction, teacher-forcing, code verification, reward computation, and child-state
creation.

**Multi-DP support** – launch with ``torchrun`` to get multiple independent vLLM
instances (one per rank), exactly like AReaL training::

    torchrun --nproc_per_node=4 \
        -m areal.experimental.ttt_discover.examples.eval_tttd_distill_standalone \
        --config areal/experimental/ttt_discover/examples/conf/fsdp_lora_vllm_ac1_qwen3_8b_distill.yaml

When ``WORLD_SIZE > 1`` each rank sees only its own GPU via ``CUDA_VISIBLE_DEVICES``,
creates a single-GPU vLLM instance, evaluates its shard of the initial states, and
finally rank 0 gathers the results.
"""

import os
import sys

# ---------------------------------------------------------------------------
# CRITICAL: set GPU visibility BEFORE any CUDA init (torch / vllm imports).
# ---------------------------------------------------------------------------
_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
_world_size = int(os.environ.get("WORLD_SIZE", "1"))
if _world_size > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_local_rank)

import asyncio
import copy
import json
import time
from typing import Any

import torch
import torch.distributed as dist

from areal.api.cli_args import parse_cli_args, to_structured_cfg
from areal.api.io_struct import ModelRequest, ModelResponse
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
from areal.experimental.ttt_discover.workflow_v2 import TTTDiscoverWorkflowV2
from areal.experimental.ttt_discover.reward import tttd_reward_fn

from omegaconf import OmegaConf
from vllm import LLM

logger = logging.getLogger("eval_tttd_distill_standalone")


class SimplevLLMEngine:
    """Minimal vLLM wrapper compatible with TTTDiscoverWorkflowV2."""

    def __init__(self, llm, tokenizer, lora_request=None):
        self.llm = llm
        self.tokenizer = tokenizer
        self.lora_request = lora_request
        self._version = 0

    def get_version(self) -> int:
        return self._version

    def set_version(self, v: int) -> None:
        self._version = v

    def set_lora(self, lora_request) -> None:
        self.lora_request = lora_request

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_sync, req)

    def _generate_sync(self, req: ModelRequest) -> ModelResponse:
        from vllm import SamplingParams

        temperature = 0.0 if req.gconfig.greedy else req.gconfig.temperature
        sp = SamplingParams(
            temperature=temperature,
            top_p=req.gconfig.top_p,
            top_k=-1 if req.gconfig.top_k >= 1e7 else req.gconfig.top_k,
            max_tokens=req.gconfig.max_new_tokens,
            min_tokens=req.gconfig.min_new_tokens,
            stop_token_ids=req.gconfig.stop_token_ids or [],
            ignore_eos=req.gconfig.ignore_eos,
        )

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


def _is_lora_adapter(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "adapter_model.safetensors"))


def _shard_states(states: list, rank: int, world_size: int) -> list:
    """Simple contiguous shard – same convention AReaL dataloader uses."""
    n = len(states)
    per_rank = n // world_size
    rem = n % world_size
    start = rank * per_rank + min(rank, rem)
    end = start + per_rank + (1 if rank < rem else 0)
    return states[start:end]


async def evaluate_model(
    llm,
    tokenizer,
    workflow: TTTDiscoverWorkflowV2,
    initial_states: list,
    batch_size: int,
    group_size: int,
    model_name: str,
    model_path: str | None,
) -> dict[str, Any]:
    """Evaluate a single model on the LOCAL shard of initial states.

    Batching follows AReaL conventions:
      - batch_size = number of parent states per batch (from config.sampler.batch_size)
      - group_size = n_samples per parent     (from config.gconfig.n_samples)
    """
    from vllm.lora.request import LoRARequest

    logger.info(f"[Eval-{model_name}] path={model_path}")

    engine = SimplevLLMEngine(llm, tokenizer)
    if model_path and _is_lora_adapter(model_path):
        lora_req = LoRARequest(model_name, 1, model_path)
        engine.set_lora(lora_req)
        logger.info(f"[Eval-{model_name}] Loaded LoRA adapter from {model_path}")
    elif model_path:
        logger.warning(
            f"[Eval-{model_name}] {model_path} does not look like a LoRA adapter. "
            "Evaluating with the base model weights."
        )
    else:
        logger.warning(f"[Eval-{model_name}] No path provided; using base model.")

    workflow.reset()
    workflow.set_current_version(0)

    # AReaL-style batch sizing: never exceed actual number of states
    eval_batch_size = min(len(initial_states), batch_size)
    eval_batch_size = max(eval_batch_size, 1)
    num_batches = (len(initial_states) + eval_batch_size - 1) // eval_batch_size

    logger.info(
        f"[Eval-{model_name}] batch_size={eval_batch_size}, group_size={group_size}, "
        f"states={len(initial_states)}, batches={num_batches}"
    )

    eval_start = time.perf_counter()
    rewards: list[float] = []

    for batch_idx in range(num_batches):
        batch_start = batch_idx * eval_batch_size
        batch_end = min(batch_start + eval_batch_size, len(initial_states))
        batch_states = initial_states[batch_start:batch_end]

        batch_begin_time = time.perf_counter()
        for state in batch_states:
            for _ in range(group_size):
                data = {"_state_obj": state}
                trajectory = await workflow.arun_episode(engine, data)
                if trajectory is None:
                    continue
                reward = float(trajectory["rewards"][0])
                rewards.append(reward)
        batch_elapsed = time.perf_counter() - batch_begin_time

        logger.info(
            f"[Eval-{model_name}] Batch {batch_idx + 1}/{num_batches} done | "
            f"parents={len(batch_states)} rollouts={len(batch_states) * group_size} "
            f"time={batch_elapsed:.1f}s"
        )

    eval_rollout_time = time.perf_counter() - eval_start

    updates = workflow.get_pending_updates(clear=True)
    if len(updates) == 5:
        eval_children, eval_parents, eval_failed, _eval_metadata, _ = updates
    else:
        eval_children, eval_parents, eval_failed, _eval_metadata = updates

    return {
        "rewards": rewards,
        "n_children": len(eval_children),
        "n_parents": len(eval_parents),
        "n_failed": len(eval_failed),
        "rollout_time_s": eval_rollout_time,
    }


def _gather_eval_results(local: dict, rank: int, world_size: int) -> dict:
    """All-gather / all-reduce evaluation stats across DP ranks."""
    if world_size <= 1:
        rewards = local["rewards"]
        return {
            "global_rollouts": len(rewards),
            "max_reward": max(rewards) if rewards else 0.0,
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "all_rewards": rewards,
            "n_children": local["n_children"],
            "n_parents": local["n_parents"],
            "n_failed": local["n_failed"],
            "rollout_time_s": local["rollout_time_s"],
        }

    # 1. all_gather_object rewards lists
    all_rewards_gathered = [None] * world_size
    dist.all_gather_object(all_rewards_gathered, local["rewards"])
    all_rewards = [r for rank_list in all_rewards_gathered for r in rank_list]

    # 2. scalar reductions
    def _red(val, op):
        t = torch.tensor([val], dtype=torch.float32, device="cuda")
        dist.all_reduce(t, op=op)
        return t.item()

    max_reward = _red(max(local["rewards"]) if local["rewards"] else 0.0, dist.ReduceOp.MAX)
    sum_reward = _red(sum(local["rewards"]), dist.ReduceOp.SUM)
    count_reward = _red(len(local["rewards"]), dist.ReduceOp.SUM)
    mean_reward = sum_reward / count_reward if count_reward > 0 else 0.0

    n_children = int(_red(local["n_children"], dist.ReduceOp.SUM))
    n_parents = int(_red(local["n_parents"], dist.ReduceOp.SUM))
    n_failed = int(_red(local["n_failed"], dist.ReduceOp.SUM))
    rollout_time = _red(local["rollout_time_s"], dist.ReduceOp.MAX)  # wall-clock

    return {
        "global_rollouts": len(all_rewards),
        "max_reward": max_reward,
        "mean_reward": mean_reward,
        "all_rewards": all_rewards,
        "n_children": n_children,
        "n_parents": n_parents,
        "n_failed": n_failed,
        "rollout_time_s": rollout_time,
    }


async def main_async(args):
    rank = _local_rank
    world_size = _world_size

    # ------------------------------------------------------------------
    # 0. Init distributed (no-op when world_size == 1)
    # ------------------------------------------------------------------
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES already restricted to 1 GPU
        logger.info(f"[Rank {rank}/{world_size}] Distributed init done")

    # ------------------------------------------------------------------
    # 1. Parse config (structured + raw for extra keys)
    # ------------------------------------------------------------------
    raw_cfg, _config_file = parse_cli_args(args)
    config = OmegaConf.to_object(to_structured_cfg(raw_cfg, TTTDDistillConfig))

    if not config.teacher_path:
        raise ValueError("teacher_path must be provided.")
    if not config.teacher_sampler_checkpoint:
        raise ValueError("teacher_sampler_checkpoint must be provided.")

    # ------------------------------------------------------------------
    # 2. Tokenizer & env (all ranks do it – they need the same objects)
    # ------------------------------------------------------------------
    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)

    env = create_env_from_config(config)

    # ------------------------------------------------------------------
    # 3. Sampler – only rank 0 loads from disk, then broadcast
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
            if rank == 0:
                logger.info(f"[TeacherSampler] Loading latest checkpoint at step {latest_step}")
            sampler._load(latest_step)
            sampler._current_step = 0
        else:
            if rank == 0:
                logger.warning(f"[TeacherSampler] No checkpoint found. Using fresh sampler state.")

    initial_states = sampler._initial_states
    if not initial_states:
        initial_states = [s for s in sampler._states if getattr(s, "timestep", 0) == 0]
    if not initial_states:
        initial_states = sampler._states[:1]
        if rank == 0:
            logger.warning("[Eval] No initial states found, using first available state")

    if rank == 0:
        logger.info(f"[Eval] {len(initial_states)} initial states total, DP={world_size}")

    # Shard states per rank
    my_states = _shard_states(initial_states, rank, world_size)
    if rank == 0:
        logger.info(f"[Eval] Rank 0 shard size = {len(my_states)}")

    # ------------------------------------------------------------------
    # 4. Read vLLM / GPU settings from raw YAML
    # ------------------------------------------------------------------
    vllm_cfg = OmegaConf.to_container(raw_cfg.get("vllm", {}), resolve=True)
    teacher_lora_path = raw_cfg.get("teacher_lora_path", config.teacher_path)
    student_lora_path = raw_cfg.get("student_lora_path", "")

    # ------------------------------------------------------------------
    # 5. Each rank creates its OWN vLLM instance (TP=1, bound to 1 GPU)
    # ------------------------------------------------------------------
    base_model_path = config.actor.path
    if rank == 0:
        logger.info(f"[vLLM] Loading base model {base_model_path} on {world_size} ranks")

    llm = LLM(
        model=base_model_path,
        tokenizer=config.tokenizer_path,
        tensor_parallel_size=1,  # one GPU per instance
        gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.85),
        max_model_len=vllm_cfg.get("max_model_len", 32768),
        enforce_eager=vllm_cfg.get("enforce_eager", False),
        enable_lora=vllm_cfg.get("enable_lora", True),
        max_lora_rank=vllm_cfg.get("max_lora_rank", getattr(config.actor, "lora_rank", 64)),
        trust_remote_code=True,
    )
    logger.info(f"[Rank {rank}] vLLM instance ready")

    # ------------------------------------------------------------------
    # 6. Workflow – batch_size / group_size read exactly like AReaL
    # ------------------------------------------------------------------
    batch_size = config.sampler.batch_size
    group_size = config.gconfig.n_samples

    if rank == 0:
        logger.info(
            f"[EvalConfig] sampler.batch_size={batch_size}, "
            f"gconfig.n_samples={group_size}"
        )

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
        dp_rank=rank,
        dp_world_size=world_size,
    )

    # ------------------------------------------------------------------
    # 7. Evaluate teacher & student on local shard
    # ------------------------------------------------------------------
    models_to_eval = [
        ("teacher", teacher_lora_path),
        ("student", student_lora_path),
    ]

    all_results = {}
    for label, path in models_to_eval:
        local_result = await evaluate_model(
            llm=llm,
            tokenizer=tokenizer,
            workflow=workflow,
            initial_states=my_states,
            batch_size=batch_size,
            group_size=group_size,
            model_name=label,
            model_path=path,
        )
        # Synchronize across ranks
        gathered = _gather_eval_results(local_result, rank, world_size)
        all_results[label] = gathered

    # ------------------------------------------------------------------
    # 8. Rank 0 writes JSON & prints summary
    # ------------------------------------------------------------------
    if rank == 0:
        comparison_path = os.path.join(config.saver.fileroot, "eval_comparison_standalone.json")
        os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
        with open(comparison_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"[Eval] Results saved to {comparison_path}")

        logger.info("\n" + "=" * 70)
        logger.info("EVALUATION COMPARISON (Initial States) – Standalone vLLM Multi-DP")
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

    # Cleanup
    workflow.shutdown()
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        logger.info("[Eval] Done.")


def main(args):
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main(sys.argv[1:])
