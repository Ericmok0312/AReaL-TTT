#!/usr/bin/env python3
"""
Standalone TTT-Discover Distillation Evaluation using vLLM directly.

This script bypasses AReaL's distributed training engine and inference weight-update
mechanism (which can hang during LoRA adapter switching) by using vLLM's native LLM
class directly for generation.  It reuses TTTDiscoverWorkflowV2 for prompt
construction, teacher-forcing, code verification, reward computation, and child-state
creation.

It reads the **same YAML config** as the original eval script
(e.g. ``fsdp_lora_vllm_ac1_qwen3_8b_distill.yaml``) and respects the
``vllm.*``, ``cluster.n_gpus_per_node``, ``teacher_lora_path`` and
``student_lora_path`` fields defined there.

Usage:
    python -m areal.experimental.ttt_discover.examples.eval_tttd_distill_standalone \
        --config areal/experimental/ttt_discover/examples/conf/fsdp_lora_vllm_ac1_qwen3_8b_distill.yaml

Output:
    ``${saver.fileroot}/eval_comparison_standalone.json``
"""

import asyncio
import json
import os
import sys
import time
from typing import Any

import torch

from areal.api.cli_args import (
    load_expr_config,
    parse_cli_args,
    to_structured_cfg,
)
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

logger = logging.getLogger("eval_tttd_distill_standalone")


class SimplevLLMEngine:
    """Minimal vLLM wrapper compatible with TTTDiscoverWorkflowV2.

    Only implements the two methods that WorkflowV2 actually calls:
    - ``agenerate(req) -> ModelResponse``
    - ``get_version() -> int``
    """

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

        # vLLM 0.4.x+ uses SamplingParams; map AReaL gconfig fields
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


async def evaluate_model(
    llm,
    tokenizer,
    workflow: TTTDiscoverWorkflowV2,
    initial_states: list,
    group_size: int,
    model_name: str,
    model_path: str | None,
) -> dict[str, Any]:
    """Evaluate a single model on the given initial states."""
    from vllm.lora.request import LoRARequest

    logger.info(f"[Eval-{model_name}] path={model_path}")

    # Setup LoRA if needed
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

    # Reset workflow buffers so each model starts fresh
    workflow.reset()
    workflow.set_current_version(0)

    eval_start = time.perf_counter()
    rewards: list[float] = []

    for state in initial_states:
        for _ in range(group_size):
            data = {"_state_obj": state}
            trajectory = await workflow.arun_episode(engine, data)
            if trajectory is None:
                continue
            # trajectory["rewards"] is a 1-D torch.Tensor of shape [1]
            reward = float(trajectory["rewards"][0])
            rewards.append(reward)

    eval_rollout_time = time.perf_counter() - eval_start

    # Gather pending updates for statistics (children, failed, etc.)
    updates = workflow.get_pending_updates(clear=True)
    if len(updates) == 5:
        eval_children, eval_parents, eval_failed, eval_metadata, _ = updates
    else:
        eval_children, eval_parents, eval_failed, eval_metadata = updates

    max_reward = max(rewards) if rewards else 0.0
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0

    result = {
        "model": model_name,
        "global_rollouts": len(rewards),
        "max_reward": max_reward,
        "mean_reward": mean_reward,
        "all_rewards": rewards,
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
            "code": getattr(child, "code", None),
        })

    logger.info(
        f"[Eval-{model_name}] max={max_reward:.4f}, mean={mean_reward:.4f}, "
        f"rollouts={len(rewards)}, children={len(eval_children)}, failed={len(eval_failed)}"
    )
    return result


async def main_async(args):
    # ------------------------------------------------------------------
    # 0. Parse config – keep both structured config and raw DictConfig so
    #    we can read extra keys (teacher_lora_path, student_lora_path, vllm, …)
    #    that are not declared in TTTDDistillConfig.
    # ------------------------------------------------------------------
    from omegaconf import OmegaConf

    raw_cfg, _config_file = parse_cli_args(args)
    config = OmegaConf.to_object(to_structured_cfg(raw_cfg, TTTDDistillConfig))

    if not config.teacher_path:
        raise ValueError("teacher_path must be provided. Add +teacher_path=<path> to your command.")
    if not config.teacher_sampler_checkpoint:
        raise ValueError("teacher_sampler_checkpoint must be provided.")

    # ------------------------------------------------------------------
    # 1. Load tokenizer & env
    # ------------------------------------------------------------------
    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)

    env = create_env_from_config(config)

    # ------------------------------------------------------------------
    # 2. Load sampler (teacher checkpoint)
    # ------------------------------------------------------------------
    import copy
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
            logger.warning(f"[TeacherSampler] No checkpoint found. Using fresh sampler state.")

    initial_states = sampler._initial_states
    if not initial_states:
        initial_states = [s for s in sampler._states if getattr(s, "timestep", 0) == 0]
    if not initial_states:
        initial_states = sampler._states[:1]
        logger.warning("[Eval] No initial states found, using first available state")

    logger.info(f"[Eval] Using {len(initial_states)} initial states for evaluation")

    # ------------------------------------------------------------------
    # 3. Read vLLM / GPU settings from the raw YAML config
    # ------------------------------------------------------------------
    vllm_cfg = OmegaConf.to_container(raw_cfg.get("vllm", {}), resolve=True)
    cluster_cfg = OmegaConf.to_container(raw_cfg.get("cluster", {}), resolve=True)

    # GPU count – default to all visible GPUs if not specified in YAML
    n_gpus = cluster_cfg.get("n_gpus_per_node", torch.cuda.device_count())
    if n_gpus <= 0:
        n_gpus = 1

    # Teacher / student LoRA paths – read from raw config (may be omitted in TTTDDistillConfig)
    teacher_lora_path = raw_cfg.get("teacher_lora_path", config.teacher_path)
    student_lora_path = raw_cfg.get("student_lora_path", "")

    logger.info(
        f"[Config] GPUs={n_gpus}, teacher_lora={teacher_lora_path}, "
        f"student_lora={student_lora_path}"
    )

    # ------------------------------------------------------------------
    # 4. Initialize vLLM
    # ------------------------------------------------------------------
    from vllm import LLM

    base_model_path = config.actor.path
    logger.info(f"[vLLM] Loading base model from {base_model_path} (TP={n_gpus})")

    llm = LLM(
        model=base_model_path,
        tokenizer=config.tokenizer_path,
        tensor_parallel_size=n_gpus,
        gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.85),
        max_model_len=vllm_cfg.get("max_model_len", 32768),
        enforce_eager=vllm_cfg.get("enforce_eager", False),
        enable_lora=vllm_cfg.get("enable_lora", True),
        max_lora_rank=vllm_cfg.get("max_lora_rank", getattr(config.actor, "lora_rank", 64)),
        trust_remote_code=True,
    )
    logger.info("[vLLM] Base model loaded")

    # ------------------------------------------------------------------
    # 5. Create workflow (eager mode, no lazy sampling)
    # ------------------------------------------------------------------
    group_size = config.gconfig.n_samples

    workflow = TTTDiscoverWorkflowV2(
        env=env,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        sampler=None,
        reward_fn=tttd_reward_fn,
        enable_thinking=getattr(config, "enable_thinking", False),
        max_prompt_thinking_tokens=getattr(config, "max_prompt_thinking_tokens", 26000),
        batch_size=len(initial_states),
        group_size=group_size,
        lazy_sampling=False,
        dp_rank=0,
        dp_world_size=1,
    )

    # ------------------------------------------------------------------
    # 6. Evaluate teacher & student
    # ------------------------------------------------------------------
    models_to_eval = [
        ("teacher", teacher_lora_path),
        ("student", student_lora_path),
    ]

    all_results = {}
    for label, path in models_to_eval:
        all_results[label] = await evaluate_model(
            llm=llm,
            tokenizer=tokenizer,
            workflow=workflow,
            initial_states=initial_states,
            group_size=group_size,
            model_name=label,
            model_path=path,
        )

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    comparison_path = os.path.join(config.saver.fileroot, "eval_comparison_standalone.json")
    with open(comparison_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"[Eval] Comparison results saved to {comparison_path}")

    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPARISON (Initial States) – Standalone vLLM")
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
    logger.info("[Eval] Done.")


def main(args):
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main(sys.argv[1:])
