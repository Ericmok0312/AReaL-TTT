# SPDX-License-Identifier: Apache-2.0

"""
ALE-Bench environment for TTT-Discover.

Wraps the official ALE-Bench evaluation toolkit so that TTT-Discover can be
trained on AtCoder Heuristic Contest problems. Each training run targets a
single problem_id; launch multiple runs for ALE-Bench lite's 10 problems.

This module imports the official ``ale_bench`` package (installed from
``https://github.com/SakanaAI/ALE-Bench``). It relies on Docker to compile
and run C++/Rust judge code, so a working Docker daemon is required.

ALE-Bench score terminology used in this module:

- ``performance`` / ``relative score`` / ``standing score``: A normalized score
  where higher is always better, regardless of the original problem type. This
  is what ALE-Bench uses for historical rankings and for ``reward_scale``.
- ``absolute raw score``: The raw per-case score returned by the judge. For
  minimization problems lower is better; for maximization problems higher is
  better. This is what we use to compute the training reward.

In other words, the reward follows the original problem semantics
(minimization = lower absolute raw score is better), while the standing-based
values are only used as a normalization reference.
"""

from __future__ import annotations

import datetime as dt
import threading
from typing import Any

from ale_bench.code_language import CodeLanguage, JudgeVersion
from ale_bench.error import AleBenchError
from ale_bench.result import JudgeResult
from ale_bench.start import start

from areal.experimental.ttt_discover.envs.env import BaseEnv, EnvResult
from areal.experimental.ttt_discover.state import AleBenchState, State
from areal.utils import logging

logger = logging.getLogger("AleBenchEnv")


# Cache sessions per problem to avoid rebuilding Rust tools repeatedly.
# The ALE-Bench Session closes its temp tool_dir on process exit.
_ALE_SESSIONS: dict[tuple[str, bool, str, int, float], Any] = {}

# Lock protecting cache access and in-process session recreation. Process-based
# executors (e.g. AsyncRewardWrapper's ProcessPoolExecutor) each have their own
# module state, so this lock coordinates threads within one process.
_ALE_SESSIONS_LOCK = threading.Lock()

# Very long session duration (100 years in seconds) to avoid ALE-Bench's default
# problem-defined session time limit during long-running RL training.
_DEFAULT_SESSION_DURATION_SECONDS: float = 100.0 * 365.25 * 24.0 * 3600.0


def _get_session(
    problem_id: str,
    lite_version: bool,
    log_dir: str,
    num_workers: int = 1,
    session_duration_seconds: float = _DEFAULT_SESSION_DURATION_SECONDS,
):
    """Get or create an ALE-Bench session for the problem."""
    if session_duration_seconds is None:
        session_duration_seconds = _DEFAULT_SESSION_DURATION_SECONDS
    key = (problem_id, lite_version, log_dir, num_workers, session_duration_seconds)
    with _ALE_SESSIONS_LOCK:
        session = _ALE_SESSIONS.get(key)
        if session is not None:
            return session
    # Create the session outside the lock so other threads are not blocked by the
    # potentially slow tool-build step.
    session = start(
        problem_id=problem_id,
        lite_version=lite_version,
        use_same_time_scale=False,
        session_duration=dt.timedelta(seconds=session_duration_seconds),
        num_workers=num_workers,
        run_visualization_server=False,
    )
    with _ALE_SESSIONS_LOCK:
        # If another thread already created one while we were building, prefer it.
        if key in _ALE_SESSIONS:
            return _ALE_SESSIONS[key]
        _ALE_SESSIONS[key] = session
        return session


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
        session_duration_seconds: ALE-Bench session duration in seconds. Defaults
            to ~100 years to avoid the problem-defined contest time limit during
            long RL training runs.
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
        session_duration_seconds: float = _DEFAULT_SESSION_DURATION_SECONDS,
    ):
        self.problem_id = problem_id
        self.lite_version = lite_version
        self.eval_timeout = eval_timeout
        self.log_dir = log_dir
        self.num_cpus = num_cpus
        self.code_language = code_language
        self.session_duration_seconds = (
            session_duration_seconds
            if session_duration_seconds is not None
            else _DEFAULT_SESSION_DURATION_SECONDS
        )

        # Map string code_language to ale_bench CodeLanguage enum.
        self._code_language_enum = getattr(CodeLanguage, code_language.upper(), None)
        if self._code_language_enum is None:
            raise ValueError(
                f"Unsupported code_language={code_language} for ALE-Bench. "
                f"Available: {[x.lower() for x in dir(CodeLanguage) if not x.startswith('_')]}."
            )

        self.session = _get_session(
            problem_id,
            lite_version,
            log_dir,
            num_workers=num_cpus,
            session_duration_seconds=session_duration_seconds,
        )
        self.problem = self.session.problem
        self.standings = self.session._standings

        # DEBUG: log the directories ALE-Bench actually uses for compilation/running.
        import os as _os

        for attr in ("tool_dir", "work_dir", "_tool_dir", "_work_dir", "log_dir"):
            val = getattr(self.session, attr, None)
            if val:
                try:
                    st = _os.stat(val)
                    logger.info(
                        f"[AleBenchEnv][DEBUG] session.{attr}={val} "
                        f"mode={oct(st.st_mode)} uid={st.st_uid} gid={st.st_gid}"
                    )
                except Exception as e:
                    logger.info(
                        f"[AleBenchEnv][DEBUG] session.{attr}={val} stat failed: {e}"
                    )

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

        # Target for prompt construction and failure-reward scaling.
        # Standings are ordered best-to-worst, so positive_scores[0] is the best
        # raw score regardless of optimization direction.
        positive_scores = [s for _, s in self.standings.standings_scores if s > 0]
        best_raw_score = positive_scores[0] if positive_scores else 0.0
        worst_raw_score = positive_scores[-1] if positive_scores else 0.0
        # NOTE: standings scores are what the original contest used for ranking.
        # They are already "higher is better" by construction:
        #   - maximize problems: sum of absolute per-case scores
        #   - minimize problems: sum of relative per-case scores
        #   - special problems (e.g. ahc025): sum of rank scores
        # The values returned by ALE-Bench as "overall_absolute_score" are these
        # standing scores, not a normalized performance metric.
        self.best_standings_score = best_raw_score
        self.best_normalized_score = best_raw_score / self.reward_scale

        logger.info(
            f"[AleBenchEnv] problem_id={problem_id} lite_version={lite_version} "
            f"score_type={self.problem.metadata.score_type.value} "
            f"maximize={self.maximize} reward_scale={self.reward_scale:.4f} "
            f"best_standings_score={self.best_standings_score:.4f} "
            f"worst_standings_score={worst_raw_score:.4f} "
            f"best_normalized_score={self.best_normalized_score:.4f}"
        )

    def get_prompt(self, state: State) -> str:
        """Build a prompt for the current ALE-Bench problem and state."""
        problem_statement = self.problem.statement
        tool_readme = self.problem.tool_readme
        example_input = getattr(self.problem, "example_input", "")
        example_output = getattr(self.problem, "example_output", "")

        value_context = state.to_prompt(
            target=None,  # Do not show the global target/gap; let the model focus
            # on incremental improvement instead of a potentially overwhelming gap.
            metric_name="score",
            maximize=self.maximize,
            language=self.code_language,
        )

        time_limit = getattr(self.problem.constraints, "time_limit", None)
        memory_limit = getattr(self.problem.constraints, "memory_limit", None)
        constraints_section = ""
        if time_limit is not None or memory_limit is not None:
            constraints_lines = ["\n--- Constraints ---\n"]
            if time_limit is not None:
                constraints_lines.append(
                    f"- Time limit: {time_limit} seconds per test case"
                )
            if memory_limit is not None:
                constraints_lines.append(
                    f"- Memory limit: {memory_limit / 1024 / 1024:.0f} MB"
                )
            constraints_section = "\n".join(constraints_lines)

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
{problem_statement}{constraints_section}{example_section}

--- Tool README ---
{tool_readme}

{value_context}

Rules:
- You must use C++20 (GNU++17/C++20 compatible) to solve the problem.
- Reason about your approach before writing code, and write clean, well-structured code with comments.
- Define all of your code in one final ```cpp ... ``` block.
- Your program must read from stdin and write to stdout exactly as described in the statement.
- Make efficient use of the allowed time limit. Think outside the box and try diverse approaches.
- Make sure you /think carefully before coding by learning from previous code and feedback.
"""

    def _recreate_session(self) -> None:
        """Drop the cached ALE-Bench session and create a fresh one.

        ALE-Bench sessions can finish when their time or resource budget is
        exhausted. Recreating gives the trainer a fresh budget while keeping the
        same problem, standings, and reward-scale configuration.

        This method is atomic with respect to other threads in the same process
        that are also recreating or fetching the cached session.
        """
        key = (
            self.problem_id,
            self.lite_version,
            self.log_dir,
            self.num_cpus,
            self.session_duration_seconds,
        )
        with _ALE_SESSIONS_LOCK:
            _ALE_SESSIONS.pop(key, None)
            # Build the new session while holding the lock so concurrent
            # recreations of the same key do not waste work.
            session = start(
                problem_id=self.problem_id,
                lite_version=self.lite_version,
                use_same_time_scale=False,
                session_duration=dt.timedelta(seconds=self.session_duration_seconds),
                num_workers=self.num_cpus,
                run_visualization_server=False,
            )
            _ALE_SESSIONS[key] = session
        self.session = session
        self.problem = self.session.problem
        self.standings = self.session._standings
        logger.info(
            f"[AleBenchEnv] recreated session for {self.problem_id} "
            f"(duration={self.session_duration_seconds:.0f}s)"
        )

    def _eval_code(self, code: str) -> Any:
        """Run ALE-Bench evaluation, recreating the session if it has finished."""
        max_retries = 1
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                if hasattr(self.session, "public_eval"):
                    return self.session.public_eval(
                        code=code,
                        code_language=self._code_language_enum,
                    )
                else:
                    return self.session.case_eval(
                        input_str=self.session._public_inputs,
                        code=code,
                        code_language=self._code_language_enum,
                        judge_version=JudgeVersion.V202301,
                        time_limit=self.problem.constraints.time_limit,
                        memory_limit=self.problem.constraints.memory_limit,
                        skip_local_visualization=True,
                    )
            except AleBenchError as e:
                last_error = e
                error_msg = str(e)
                if "session is finished" in error_msg.lower() and attempt < max_retries:
                    logger.warning(
                        f"[AleBenchEnv] session finished on attempt {attempt + 1}, "
                        f"recreating session and retrying"
                    )
                    self._recreate_session()
                else:
                    break
        raise (
            last_error
            if last_error is not None
            else RuntimeError("_eval_code failed without raising an exception")
        )

    def execute(self, code: str, state: State) -> EnvResult:
        """Evaluate the generated C++ code on the ALE-Bench public cases."""
        # ALE-Bench's official API is session.public_eval(code, code_language=...).
        # Fallback to the lower-level case_eval if public_eval is unavailable.
        try:
            result = self._eval_code(code)
        except Exception as e:
            logger.warning(f"[AleBenchEnv] evaluation failed: {e}", exc_info=True)
            self._log_failed_code(code, fail_type="execution_error", error_msg=str(e))
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
        # ALE-Bench's official aggregate score is the sum over all public cases.
        # We expose the per-case average as both ``raw_score`` and the reward
        # basis, matching the original TTT-Discover AHC implementation.
        total_raw_score = float(getattr(result, "overall_absolute_score", 0.0))
        avg_raw_score = total_raw_score / num_cases if num_cases > 0 else 0.0
        normalized_avg_raw_score = avg_raw_score / self.reward_scale
        if self.maximize:
            reward = normalized_avg_raw_score
        else:
            # Reciprocal so higher reward always means better performance.
            # Valid rewards are strictly positive; failure reward=0 is then
            # worse than any valid solution.
            # For partial case failures (num_accepted < num_cases) avg_raw_score
            # can be 0, which would make the reciprocal explode. Use 0 for those.
            if num_accepted < num_cases:
                reward = 0.0
            else:
                reward = 1.0 / (1e-8 + normalized_avg_raw_score)

        # Build a detailed observation for the LLM. Include per-case results and
        # the first few failure messages so the model can iterate on bugs.
        lines = [
            f"Evaluated on {num_cases} public cases.",
            f"Passed: {num_accepted}/{num_cases}.",
            f"Total raw score: {total_raw_score:.4f}.",
            f"Avg raw score per case: {avg_raw_score:.4f}.",
            f"Overall judge: {getattr(result, 'overall_judge_result', None) and result.overall_judge_result.value or 'N/A'}.",
        ]

        non_ac_cases = [
            (i, c)
            for i, c in enumerate(case_results)
            if getattr(c, "judge_result", None) != JudgeResult.ACCEPTED
        ]
        if non_ac_cases:
            lines.append("Failed cases (showing up to 3):")
            for case_idx, case in non_ac_cases[:3]:
                judge_value = getattr(
                    case.judge_result, "value", str(case.judge_result)
                )
                message = getattr(case, "message", "") or ""
                lines.append(f"  Case {case_idx}: {judge_value} - {message}")
                error_str = (getattr(case, "error_str", "") or "").strip()
                if error_str:
                    if len(error_str) > 500:
                        error_str = error_str[:500] + "\n...(truncated)..."
                    lines.append(f"    stderr:\n{error_str}")

        observation = "\n".join(lines)

        if num_accepted < num_cases:
            logger.info(
                f"[AleBenchEnv][{self.problem_id}] partial case results: "
                f"passed={num_accepted}/{num_cases}, observation=\n{observation}"
            )
            self._log_failed_code(
                code,
                fail_type="case_failed",
                error_msg=observation,
                num_accepted=num_accepted,
                num_cases=num_cases,
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
                "raw_score": avg_raw_score,
                "reward_scale": self.reward_scale,
                "overall_judge_result": getattr(result, "overall_judge_result", None),
            },
        )

    def _log_failed_code(
        self,
        code: str,
        fail_type: str,
        error_msg: str,
        num_accepted: int | None = None,
        num_cases: int | None = None,
    ) -> None:
        """Log a failed rollout's code and diagnostic info for debugging."""
        # Failure details are verbose and bloat trainer logs when many rollouts
        # fail (common during early training). Keep everything at debug level so
        # the info is available on demand without spamming the default output.
        header = f"[AleBenchEnv][FAILED ROLLOUT][{self.problem_id}] type={fail_type}"
        if num_accepted is not None and num_cases is not None:
            header += f" passed={num_accepted}/{num_cases}"
        logger.debug(header)

        # Print the extracted code (truncated if extremely long).
        code_lines = code.splitlines()
        preview_lines = code_lines[:80]
        preview = "\n".join(preview_lines)
        if len(code_lines) > 80:
            preview += "\n...(truncated)..."
        logger.debug(f"Extracted code:\n```cpp\n{preview}\n```")

        logger.debug(f"Diagnostic:\n{error_msg}")

    def create_state(
        self,
        parent_state: State,
        code: str,
        reward: float,
        result: EnvResult,
        timestep: int,
    ) -> AleBenchState:
        """Create a new AleBenchState child from the execution result."""
        parent_raw_score = getattr(parent_state, "raw_score", None)
        return AleBenchState(
            timestep=timestep,
            code=code,
            value=float(reward),
            raw_score=result.metadata.get("raw_score") if result.metadata else None,
            parent_raw_scores=[parent_raw_score]
            if parent_raw_score is not None
            else [],
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

        Valid rewards are non-negative for both maximization and minimization
        (minimization uses a positive reciprocal). Therefore ``reward=0.0`` is
        always worse than any valid solution, and we keep the base implementation
        unchanged.
        """
        return super().get_failure_result(state, fail_type, error_msg)


def create_initial_state_ale_bench(problem_id: str) -> AleBenchState:
    """Create an empty initial state for an ALE-Bench problem."""
    return AleBenchState(
        timestep=-1,
        code="",
        value=None,
        raw_score=None,
        observation=f"Initial state for ALE-Bench problem {problem_id}",
    )
