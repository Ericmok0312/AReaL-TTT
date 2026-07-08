#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""ALE-Bench GRPO baseline using AReaL's standard PPOTrainer + RLVRWorkflow.

This script trains a full Qwen3-8B on the 8 ALE-Bench teacher problems with
standard group-relative policy optimization (GRPO).  It is intentionally simple:
each dataset example is one problem prompt, the model generates a C++20 solution,
and the reward is the ALE-Bench public evaluation score.

Usage
-----
    python -m areal.infra.launcher.local \
        areal/experimental/ttt_discover/examples/train_ale_bench_grpo_baseline.py \
        --config areal/experimental/ttt_discover/examples/conf/fsdp_full_vllm_ale_bench_qwen3_8b_grpo_baseline.yaml

References
----------
* ``areal/experimental/ttt_discover/examples/train_tttd_async.py``
* ``areal/experimental/ttt_discover/examples/train_tttd_distill.py``
* ``areal/experimental/ttt_discover/examples/conf/fsdp_lora_vllm_ale_bench_qwen3_8b_distill_multi_diverse_best_combined.yaml``
"""

import os
import sys
from dataclasses import dataclass, field

from datasets import Dataset

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.experimental.ttt_discover.ale_bench_baseline_reward import (
    AleBenchBaselineRewardFn,
    build_ale_bench_prompt,
)
from areal.utils.hf_utils import load_hf_tokenizer


# ALE-Bench session creation can take >15s on the first evaluation, but
# AReaL's RLVRWorkflow hard-codes AsyncRewardWrapper(timeout_seconds=15).
# Monkey-patch the class seen by RLVRWorkflow to use a generous timeout.
def _patch_async_reward_wrapper() -> None:
    from areal.api.reward_api import AsyncRewardWrapper

    class _AsyncRewardWrapperWithTimeout(AsyncRewardWrapper):
        def __init__(
            self,
            reward_fn,
            timeout_seconds: float = 1200,
            max_workers: int | None = None,
            max_retries: int = 3,
        ):
            # ALE-Bench starts one Docker-backed session per worker/problem.
            # The default heuristic can spawn too many workers and overwhelm
            # Docker, causing repeated session creation.  Cap at 4 workers.
            if max_workers is None:
                max_workers = 4
            super().__init__(
                reward_fn,
                timeout_seconds=timeout_seconds,
                max_workers=max_workers,
                max_retries=max_retries,
            )

    import areal.workflow.rlvr as _rlvr_mod

    _rlvr_mod.AsyncRewardWrapper = _AsyncRewardWrapperWithTimeout


_patch_async_reward_wrapper()


@dataclass
class AleBenchGRPOConfig(GRPOConfig):
    """GRPO config extended with ALE-Bench baseline knobs."""

    ale_bench_problem_ids: list[str] = field(
        default_factory=lambda: [
            "ahc025",
            "ahc026",
            "ahc027",
            "ahc039",
            "ahc046",
            "ahc011",
            "ahc015",
            "ahc016",
        ]
    )
    ale_bench_lite_version: bool = False
    ale_bench_num_workers: int = 2
    ale_bench_n_repeats: int = 10
    enable_thinking: bool = True


def build_ale_bench_dataset(
    problem_ids: list[str],
    lite_version: bool,
    n_repeats: int,
) -> Dataset:
    """Build a small HuggingFace ``Dataset`` with one row per problem repeat.

    Each row contains ``messages`` (the user prompt) and ``problem_id`` (used by
    the reward function to select the correct ALE-Bench session).
    """
    messages_col: list[list[dict[str, str]]] = []
    problem_id_col: list[str] = []

    for _ in range(n_repeats):
        for problem_id in problem_ids:
            prompt = build_ale_bench_prompt(problem_id, lite_version=lite_version)
            messages = [{"role": "user", "content": prompt}]
            messages_col.append(messages)
            problem_id_col.append(problem_id)

    return Dataset.from_dict(
        {
            "messages": messages_col,
            "problem_id": problem_id_col,
        }
    )


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, AleBenchGRPOConfig)
    os.makedirs(config.cluster.fileroot, exist_ok=True)

    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = build_ale_bench_dataset(
        problem_ids=config.ale_bench_problem_ids,
        lite_version=config.ale_bench_lite_version,
        n_repeats=config.ale_bench_n_repeats,
    )
    valid_dataset = build_ale_bench_dataset(
        problem_ids=config.ale_bench_problem_ids,
        lite_version=config.ale_bench_lite_version,
        n_repeats=1,
    )

    reward_fn = AleBenchBaselineRewardFn(
        lite_version=config.ale_bench_lite_version,
        log_dir=config.cluster.fileroot,
        num_cpus=config.ale_bench_num_workers,
    )

    workflow_kwargs = dict(
        reward_fn=reward_fn,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        enable_thinking=config.enable_thinking,
    )

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.rlvr.RLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow=None,
            eval_workflow_kwargs=None,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
