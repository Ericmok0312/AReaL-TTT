# SPDX-License-Identifier: Apache-2.0

"""
ALE-Bench environment for TTT-Discover.

Wraps the official ALE-Bench evaluation toolkit so that TTT-Discover can be
trained on AtCoder Heuristic Contest problems. Each training run targets a
single problem_id; launch multiple runs for ALE-Bench lite's 10 problems.

This module imports the official ``ale_bench`` package (installed from
``https://github.com/SakanaAI/ALE-Bench``). It relies on Docker to compile
and run C++/Rust judge code, so a working Docker daemon is required.
"""

from __future__ import annotations

from typing import Any

from ale_bench.code_language import CodeLanguage, JudgeVersion
from ale_bench.result import JudgeResult
from ale_bench.start import start

from areal.experimental.ttt_discover.envs.env import BaseEnv, EnvResult
from areal.experimental.ttt_discover.state import AleBenchState, State
from areal.utils import logging

logger = logging.getLogger("AleBenchEnv")


# Cache sessions per problem to avoid rebuilding Rust tools repeatedly.
# The ALE-Bench Session closes its temp tool_dir on process exit.
_ALE_SESSIONS: dict[tuple[str, bool, str], Any] = {}


def _get_session(
    problem_id: str, lite_version: bool, log_dir: str, num_workers: int = 1
):
    """Get or create an ALE-Bench session for the problem."""
    key = (problem_id, lite_version, log_dir, num_workers)
    if key not in _ALE_SESSIONS:
        session = start(
            problem_id=problem_id,
            lite_version=lite_version,
            use_same_time_scale=False,
            session_duration=None,
            num_workers=num_workers,
            run_visualization_server=False,
        )
        _ALE_SESSIONS[key] = session
    return _ALE_SESSIONS[key]


class AleBenchEnv(BaseEnv):
    """ALE-Bench environment for TTT-Discover.

    Args:
        problem_id: ALE-Bench problem ID, e.g. ``"ahc039"``.
        lite_version: Whether to use ALE-Bench lite seeds (default: True).
        eval_timeout: Not used directly; ALE-Bench uses problem time limits.
            Kept for API compatibility with other envs.
        log_dir: Directory for ALE-Bench scratch files (must be writable).
        num_cpus: Number of CPUs requested per evaluation. Mapped to
            ALE-Bench ``num_workers`` for parallel case evaluation.
        reward_scale: Divide raw absolute score by this value to obtain reward.
            If None, auto-computed from the problem standings (top score).
        maximize: If True, higher score is better. If None, inferred from the
            problem's ``score_type``.
        code_language: Language string expected from the model, e.g. ``"cpp20"``.
    """

    def __init__(
        self,
        problem_id: str,
        lite_version: bool = True,
        eval_timeout: int = 600,
        log_dir: str = "./outputs_ale_bench",
        num_cpus: int = 2,
        reward_scale: float | None = None,
        maximize: bool | None = None,
        code_language: str = "cpp20",
    ):
        self.problem_id = problem_id
        self.lite_version = lite_version
        self.eval_timeout = eval_timeout
        self.log_dir = log_dir
        self.num_cpus = num_cpus
        self.code_language = code_language

        self.session = _get_session(
            problem_id, lite_version, log_dir, num_workers=num_cpus
        )
        self.problem = self.session.problem
        self.standings = self.session._standings

        # Infer optimization direction from problem metadata.
        if maximize is None:
            self.maximize = self.problem.metadata.score_type.value == "maximize"
        else:
            self.maximize = maximize

        # Compute a reward scale so that typical rewards are O(1).
        if reward_scale is not None:
            self.reward_scale = reward_scale
        else:
            scores = [s for _, s in self.standings.standings_scores if s > 0]
            self.reward_scale = max(scores) if scores else 1.0
            if self.reward_scale <= 0:
                self.reward_scale = 1.0

        # Target for prompt construction (normalized raw-score units).
        # State.to_prompt negates self.value when maximize=False, so target_score
        # stays in the raw-score domain for both directions.
        positive_scores = [s for _, s in self.standings.standings_scores if s > 0]
        best_raw_score = positive_scores[0] if positive_scores else 0.0
        worst_raw_score = positive_scores[-1] if positive_scores else 0.0
        self.target_score = (
            best_raw_score / self.reward_scale
            if self.maximize
            else worst_raw_score / self.reward_scale
        )

        logger.info(
            f"[AleBenchEnv] problem_id={problem_id} lite_version={lite_version} "
            f"score_type={self.problem.metadata.score_type.value} "
            f"maximize={self.maximize} reward_scale={self.reward_scale:.4f} "
            f"target_score={self.target_score:.4f}"
        )

    def get_prompt(self, state: State) -> str:
        """Build a prompt for the current ALE-Bench problem and state."""
        problem_statement = self.problem.statement
        tool_readme = self.problem.tool_readme
        example_input = getattr(self.problem, "example_input", "")
        example_output = getattr(self.problem, "example_output", "")

        value_context = state.to_prompt(
            target=self.target_score,
            metric_name="score",
            maximize=self.maximize,
            language=self.code_language,
        )

        example_section = ""
        if example_input.strip() or example_output.strip():
            example_section = "\n--- Example Input/Output ---\n"
            if example_input.strip():
                example_section += f"Input:\n```\n{example_input.strip()}\n```\n"
            if example_output.strip():
                example_section += f"Output:\n```\n{example_output.strip()}\n```\n"

        return f"""You are a world-class algorithm engineer participating in an AtCoder Heuristic Contest.

Below is the full problem statement. Read it carefully and write a complete C++20 program that solves it.

--- Problem Statement ---
{problem_statement}{example_section}

--- Tool README ---
{tool_readme}

{value_context}

Rules:
- You must use C++20 (GNU++17/C++20 compatible) to solve the problem.
- Define all of your code in one final ```cpp ... ``` block.
- In your final response, you should only output the code of your program. Do not include any other text, explanations, or markdown outside the code block.
- Your program must read from stdin and write to stdout exactly as described in the statement.
- Make efficient use of the allowed time limit. Think outside the box and try diverse approaches.
"""

    def execute(self, code: str, state: State) -> EnvResult:
        """Evaluate the generated C++ code on the ALE-Bench public cases."""
        try:
            result = self.session.case_eval(
                input_str=self.session._public_inputs,
                code=code,
                code_language=CodeLanguage.CPP20,
                judge_version=JudgeVersion.V202301,
                time_limit=self.problem.constraints.time_limit,
                memory_limit=self.problem.constraints.memory_limit,
                skip_local_visualization=True,
            )
        except Exception as e:
            logger.warning(f"[AleBenchEnv] case_eval failed: {e}", exc_info=True)
            return self.get_failure_result(
                state=state,
                fail_type="execution_error",
                error_msg=str(e),
            )

        case_results = result.case_results
        num_cases = len(case_results)
        num_accepted = sum(
            1
            for c in case_results
            if getattr(c, "judge_result", None) == JudgeResult.ACCEPTED
        )
        total_raw_score = sum(
            float(getattr(c, "absolute_score", 0.0)) for c in case_results
        )
        avg_raw_score = total_raw_score / num_cases if num_cases > 0 else 0.0
        if self.maximize:
            reward = avg_raw_score / self.reward_scale
        else:
            # Negate so higher reward always means better performance.
            reward = -avg_raw_score / self.reward_scale

        observation = (
            f"Evaluated on {num_cases} public cases. "
            f"Passed: {num_accepted}/{num_cases}. "
            f"Avg raw score: {avg_raw_score:.4f}. "
            f"Overall judge: {result.overall_judge_result.value if hasattr(result, 'overall_judge_result') else 'N/A'}."
        )

        return EnvResult(
            reward=float(reward),
            observation=observation,
            is_valid=num_accepted == num_cases,
            metadata={
                "problem_id": self.problem_id,
                "num_cases": num_cases,
                "num_accepted": num_accepted,
                "avg_raw_score": avg_raw_score,
                "total_raw_score": total_raw_score,
                "reward_scale": self.reward_scale,
                "overall_judge_result": getattr(result, "overall_judge_result", None),
            },
        )

    def create_state(
        self,
        parent_state: State,
        code: str,
        reward: float,
        result: EnvResult,
        timestep: int,
    ) -> AleBenchState:
        """Create a new AleBenchState child from the execution result."""
        return AleBenchState(
            timestep=timestep,
            code=code,
            value=float(reward),
            observation=result.observation,
            parent_values=[parent_state.value]
            if parent_state.value is not None
            else [],
            parents=[{"id": parent_state.id, "timestep": parent_state.timestep}],
        )

    def extract_code(self, completion: str) -> str | None:
        """Extract C++ code from markdown blocks.

        ALE-Bench expects a single self-contained C++ file, so we strip
        surrounding explanations and keep only the code inside ```cpp ... ```.
        """
        import re

        # Prefer explicit cpp blocks.
        for pattern in (r"```cpp\s+([\s\S]*?)\s*```", r"```c\+\+\s+([\s\S]*?)\s*```"):
            match = re.search(pattern, completion, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Fallback to generic code block.
        match = re.search(r"```\s+([\s\S]*?)\s*```", completion)
        if match:
            return match.group(1).strip()

        # Last resort: take the whole completion if it looks like code.
        if "int main" in completion:
            return completion.strip()

        return None

    def get_failure_result(
        self,
        state: State | None,
        fail_type: str,
        error_msg: str = "",
    ) -> EnvResult:
        """Return a failure result worse than the worst valid score.

        The base implementation uses ``reward=0.0`` for most failures, which is
        appropriate for maximization problems but misleading for minimization
        (where ``0.0`` corresponds to the best possible raw score). We override
        the reward so that failures are always worse than the worst valid score,
        regardless of the optimization direction.
        """
        result = super().get_failure_result(state, fail_type, error_msg)
        result.reward = -max(1.0, abs(self.target_score)) * 2.0
        return result


def create_initial_state_ale_bench(problem_id: str) -> AleBenchState:
    """Create an empty initial state for an ALE-Bench problem."""
    return AleBenchState(
        timestep=-1,
        code="",
        value=None,
        observation=f"Initial state for ALE-Bench problem {problem_id}",
    )
