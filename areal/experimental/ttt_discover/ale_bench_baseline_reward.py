# SPDX-License-Identifier: Apache-2.0

"""Standalone reward function + prompt builder for an ALE-Bench GRPO baseline.

This module is meant to be used with AReaL's standard ``PPOTrainer`` and
``RLVRWorkflow`` (not the TTT-Discover sampler).  It exposes:

* :func:`build_ale_bench_prompt` – builds a concise code-generation prompt from
  an ALE-Bench problem ID.
* :class:`AleBenchBaselineRewardFn` – a picklable callable that evaluates a
  generated C++ program with ALE-Bench's public evaluator and returns a scalar
  reward (higher is better).
"""

from __future__ import annotations

from typing import Any

from areal.experimental.ttt_discover.envs.ale_bench import AleBenchEnv
from areal.utils import logging

logger = logging.getLogger("AleBenchBaselineReward")

# Per-worker environment cache.  Each ProcessPoolExecutor worker keeps its own
# cache, so ALE-Bench sessions are reused across rollouts for the same problem.
_ENV_CACHE: dict[tuple[str, bool, str, int], AleBenchEnv] = {}


def build_ale_bench_prompt(problem_id: str, lite_version: bool = False) -> str:
    """Build the concise ALE-Bench code-generation prompt used by the baseline.

    This is kept byte-for-byte identical to ``AleBenchEnv.get_prompt_distill``
    so that distillation and RL baselines share the same prompt distribution.
    """
    from ale_bench.data import load_problem

    problem, *_ = load_problem(problem_id=problem_id, lite_version=lite_version)
    time_limit = problem.constraints.time_limit
    memory_limit = problem.constraints.memory_limit // 1024 // 1024
    problem_statement = problem.statement

    return f"""There is a problem statement below. First, analyze the problem carefully. Think about the essential points of the problem and possible algorithms to achieve a higher rank in the contest.

Next, implement your solution in C++20 (GNU G++17 / C++20 compatible). Your solution code must be written inside a single ```cpp ... ``` code block.
You can use the following external libraries:
- AC Library@1.5.1
- Boost@1.82.0

[Problem statement]
Execution time limit: {time_limit} sec / Memory limit: {memory_limit} MiB

{problem_statement}
"""


def _get_env(
    problem_id: str, lite_version: bool, log_dir: str, num_cpus: int
) -> AleBenchEnv:
    """Return a cached ``AleBenchEnv`` for the given problem."""
    key = (problem_id, lite_version, log_dir, num_cpus)
    if key not in _ENV_CACHE:
        logger.info(
            f"[AleBenchBaselineReward] Creating env for {problem_id} "
            f"(lite={lite_version}, log_dir={log_dir}, num_cpus={num_cpus})"
        )
        _ENV_CACHE[key] = AleBenchEnv(
            problem_id=problem_id,
            lite_version=lite_version,
            log_dir=log_dir,
            num_cpus=num_cpus,
        )
    return _ENV_CACHE[key]


class AleBenchBaselineRewardFn:
    """Picklable reward callable for ALE-Bench GRPO baseline.

    Parameters
    ----------
    lite_version
        Whether to use ALE-Bench lite sessions.
    log_dir
        Scratch directory passed to ``AleBenchEnv``.
    num_cpus
        Number of ALE-Bench workers used per public evaluation.
    """

    def __init__(
        self,
        lite_version: bool = False,
        log_dir: str = "./outputs_ale_bench",
        num_cpus: int = 2,
    ):
        self.lite_version = lite_version
        self.log_dir = log_dir
        self.num_cpus = num_cpus

    def __call__(
        self,
        prompt: str,
        completions: str,
        prompt_ids: list[int],
        completion_ids: list[int],
        problem_id: str,
        **kwargs: Any,
    ) -> float:
        """Evaluate ``completions`` and return a scalar reward.

        The reward is produced by ``AleBenchEnv.execute``, which already maps
        both maximization and minimization problems to a higher-is-better scale.
        """
        try:
            env = _get_env(problem_id, self.lite_version, self.log_dir, self.num_cpus)
            code = env.extract_code(completions)
            if code is None or not code.strip():
                logger.warning(
                    f"[AleBenchBaselineReward][{problem_id}] No code extracted"
                )
                return 0.0

            result = env.execute(code, state=None)
            metadata = result.metadata or {}
            logger.info(
                f"[AleBenchBaselineReward][{problem_id}] public_eval done: "
                f"reward={float(result.reward):.4f} "
                f"avg_raw_score={metadata.get('avg_raw_score', float('nan')):.4f} "
                f"total_raw_score={metadata.get('total_raw_score', float('nan')):.4f} "
                f"num_accepted={metadata.get('num_accepted', 0)}/{metadata.get('num_cases', 0)} "
                f"maximize={env.maximize}"
            )
            return float(result.reward)
        except Exception:
            logger.warning(
                f"[AleBenchBaselineReward][{problem_id}] Reward computation failed",
                exc_info=True,
            )
            return 0.0
