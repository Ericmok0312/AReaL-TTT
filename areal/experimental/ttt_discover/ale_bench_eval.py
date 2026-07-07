# SPDX-License-Identifier: Apache-2.0

"""ALE-Bench full-corpus evaluation helpers for TTT-Discover distillation.

This module implements the Self-refine x1 protocol described in the ALE-Bench
paper:

1. Generate ``n_candidates`` responses for every ALE-Bench problem.
2. Run public evaluation (50 local test cases) for each candidate.
3. Pick the candidate with the highest median public-case score.
4. Run private evaluation on the selected candidate.
5. Save per-problem results plus overall averages.
"""

from __future__ import annotations

import atexit
import json
import os
import traceback
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import numpy as np

from areal.utils import logging

logger = logging.getLogger("AleBenchEval")


def list_ale_bench_problem_ids(lite_version: bool = False) -> list[str]:
    """Return the official list of ALE-Bench problem IDs."""
    import ale_bench

    return ale_bench.list_problem_ids(lite_version=lite_version)


def _case_absolute_score(case) -> float:
    """Extract the absolute/raw score from a case result robustly."""
    for attr in ("absolute_score", "score", "raw_score"):
        val = getattr(case, attr, None)
        if val is not None:
            return float(val)
    return 0.0


def _reset_private_eval_counter(session: Any) -> None:
    """Reset ALE-Bench's per-session private-eval counter.

    ``start()`` hard-codes ``num_call_private_eval=1``.  Replacing the resource
    usage object with an equivalent one whose private-eval count is zero lets us
    run multiple ``private_eval`` calls on the same session without rebuilding
    Rust tools.  Other resource counters are preserved.
    """
    from ale_bench.result import ResourceUsage

    old = session._current_resource_usage
    session._current_resource_usage = ResourceUsage(
        num_case_gen=old.num_case_gen,
        num_case_eval=old.num_case_eval,
        num_call_public_eval=old.num_call_public_eval,
        num_call_private_eval=0,
        execution_time_case_eval=old.execution_time_case_eval,
    )


def _judge_result_is_accepted(judge_result: Any) -> bool:
    """Return True if an ALE-Bench judge result represents ACCEPTED.

    ALE-Bench returns ``JudgeResult`` enum instances.  When stored as strings they
    may look like ``"ACCEPTED"`` or ``"JudgeResult.ACCEPTED"`` depending on how
    they were converted.  This helper normalizes all common forms.
    """
    if judge_result is None:
        return False
    if isinstance(judge_result, str):
        upper = judge_result.upper()
        return upper == "ACCEPTED" or upper.endswith(".ACCEPTED")
    try:
        from ale_bench.result import JudgeResult

        return judge_result == JudgeResult.ACCEPTED
    except Exception:
        return False


def _get_problem_score_type(problem_id: str, lite_version: bool) -> str:
    """Load ALE-Bench problem metadata and return its score type.

    Returns ``"minimize"`` or ``"maximize"``. Falls back to ``"minimize"`` on
    error.
    """
    try:
        from ale_bench.data import load_problem

        problem, *_ = load_problem(problem_id=problem_id, lite_version=lite_version)
        return str(problem.metadata.score_type.value)
    except Exception as e:
        logger.warning(
            f"[{problem_id}] Failed to load problem metadata: {e}. "
            f"Falling back to minimize semantics."
        )
        return "minimize"


def _select_best_candidate_index(
    candidate_results: list[dict[str, Any]],
    selection_method: str = "median",
    score_type: str = "minimize",
) -> int:
    """Select the best candidate index using the chosen strategy.

    Args:
        candidate_results: List of candidate result dicts. Each dict must contain
            a ``public`` entry with ``overall_absolute_score`` (official median)
            or ``median_case_score`` (legacy).
        selection_method: ``"median"`` for the official ALE-Bench leaderboard
            protocol (closest to median of ``overall_absolute_score``),
            ``"median_case_score"`` for the legacy highest per-candidate median
            case score, or ``"best_public"`` for the candidate with the best
            public ``overall_absolute_score`` according to the problem's score
            type.
        score_type: ``"minimize"`` or ``"maximize"``. Used by ``"best_public"``
            and ``"median_case_score"`` to determine selection direction. Prefer
            ACCEPTED candidates when ``judge_result`` is available.

    Returns:
        Index of the selected candidate in ``candidate_results``.
    """
    if not candidate_results:
        return 0

    is_minimize = str(score_type).lower().strip() == "minimize"

    def _ac_indices() -> list[int] | None:
        ac = [
            i
            for i, c in enumerate(candidate_results)
            if _judge_result_is_accepted(c.get("public", {}).get("judge_result"))
        ]
        return ac if ac else None

    if selection_method == "median_case_score":
        medians = [
            c.get("public", {}).get("median_case_score", float("nan"))
            for c in candidate_results
        ]
        valid_indices = [i for i, s in enumerate(medians) if not np.isnan(s)]
        if not valid_indices:
            return 0
        ac = _ac_indices()
        if ac is not None:
            valid_indices = [i for i in valid_indices if i in ac]
            if not valid_indices:
                return 0
        best_fn = np.argmin if is_minimize else np.argmax
        valid_medians = [medians[i] for i in valid_indices]
        return int(valid_indices[int(best_fn(valid_medians))])

    if selection_method == "best_public":
        scores = [
            c.get("public", {}).get("overall_absolute_score", float("nan"))
            for c in candidate_results
        ]
        valid_indices = [i for i, s in enumerate(scores) if not np.isnan(s)]
        if not valid_indices:
            return _select_best_candidate_index(
                candidate_results, "median_case_score", score_type
            )
        ac = _ac_indices()
        if ac is not None:
            valid_indices = [i for i in valid_indices if i in ac]
            if not valid_indices:
                return _select_best_candidate_index(
                    candidate_results, "median_case_score", score_type
                )
        valid_scores = [scores[i] for i in valid_indices]
        if is_minimize:
            best_sub = int(np.argmin(valid_scores))
        else:
            best_sub = int(np.argmax(valid_scores))
        return int(valid_indices[best_sub])

    # Official ALE-Bench "median" selection: from the repeated samples,
    # pick the candidate whose overall_absolute_score is closest to the median
    # of all candidates' overall_absolute_scores.
    scores = [
        c.get("public", {}).get("overall_absolute_score", float("nan"))
        for c in candidate_results
    ]
    valid_scores = [s for s in scores if not np.isnan(s)]
    if not valid_scores:
        # Fall back to legacy behavior if overall_absolute_score is missing.
        return _select_best_candidate_index(
            candidate_results,
            selection_method="median_case_score",
            score_type=score_type,
        )

    median_score = float(np.median(valid_scores))
    distances = [
        abs(s - median_score) if not np.isnan(s) else float("inf") for s in scores
    ]
    return int(np.argmin(distances))


# Module-level session cache used by ale_bench_public_reward_fn.  Each worker
# process (AsyncRewardWrapper ProcessPoolExecutor) keeps its own cache, keyed by
# the full evaluation configuration so that sessions are reused across candidates
# of the same problem.
_ale_bench_sessions: dict[tuple[str, bool, float, int], Any] = {}


def close_all_cached_ale_bench_sessions() -> None:
    """Close all ALE-Bench sessions cached by this module in this process.

    Registered with :mod:`atexit` for normal interpreter shutdown.  Callers that
    need cleanup on ``SIGINT`` / ``SIGTERM`` should install a signal handler that
    invokes this function.
    """
    sessions = list(_ale_bench_sessions.values())
    _ale_bench_sessions.clear()
    for session in sessions:
        try:
            if session is not None and not getattr(session, "_closed", False):
                session.close()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[AleBenchEval] Failed to close cached session: {e}")


atexit.register(close_all_cached_ale_bench_sessions)


def _get_ale_bench_session(
    problem_id: str,
    lite_version: bool,
    session_duration_hours: float,
    ale_bench_num_workers: int,
) -> Any:
    """Get or create a cached ALE-Bench session for the given problem."""
    import time

    from ale_bench import start

    key = (problem_id, lite_version, session_duration_hours, ale_bench_num_workers)
    session = _ale_bench_sessions.get(key)
    if session is None:
        logger.info(
            f"[_get_ale_bench_session][{problem_id}] Cache miss, calling start() "
            f"(lite={lite_version}, workers={ale_bench_num_workers})"
        )
        start_ts = time.time()
        session = start(
            problem_id=problem_id,
            lite_version=lite_version,
            use_same_time_scale=False,
            session_duration=timedelta(hours=session_duration_hours),
            num_workers=ale_bench_num_workers,
            run_visualization_server=False,
        )
        elapsed = time.time() - start_ts
        logger.info(
            f"[_get_ale_bench_session][{problem_id}] start() returned in {elapsed:.1f}s"
        )
        _ale_bench_sessions[key] = session
    else:
        logger.info(f"[_get_ale_bench_session][{problem_id}] Cache hit")
    return session


def ale_bench_public_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    lite_version: bool = False,
    session_duration_hours: float = 12.0,
    ale_bench_num_workers: int = 1,
    **_kwargs: Any,
) -> tuple[float, Any, str, float]:
    """Reward function that runs ALE-Bench public evaluation.

    Designed to be called from ``TTTDiscoverWorkflowV2`` during rollout so that
    GPU generation and CPU/Docker public evaluation overlap.  The public median
    case score is returned as the reward, and the full public result is stored
    in ``EnvResult.metadata`` so the trainer can later pick the best candidate
    and run the private evaluation.

    Args:
        prompt: Prompt text (unused, kept for AReaL reward signature).
        completions: Generated completion text.
        prompt_ids: Prompt token IDs (unused).
        completion_ids: Completion token IDs (unused).
        lite_version: ALE-Bench lite mode flag.
        session_duration_hours: Time budget passed to ``ale_bench.start``.
        ale_bench_num_workers: ``num_workers`` passed to ``ale_bench.start``.
        **_kwargs: Extra data from the workflow, including ``_problem_id`` and
            ``_env``.

    Returns:
        ``(reward, EnvResult, extracted_code, exec_time_ms)`` tuple.  ``reward``
        is the public median case score.  ``EnvResult.metadata`` contains the
        public evaluation details needed for private-eval selection.
    """
    import time

    from areal.experimental.ttt_discover.envs.env import EnvResult

    start_time = time.time()
    problem_id = _kwargs.get("_problem_id", "")
    env = _kwargs.get("_env")

    if env is not None:
        code = env.extract_code(completions)
    else:
        # Fallback: try to extract a code block heuristically.
        import re

        match = re.search(
            r"```(?:cpp|c\+\+|python)?\n(.*?)\n```", completions, re.DOTALL
        )
        code = match.group(1) if match else completions

    if code is None or not code.strip():
        logger.warning(f"[ale_bench_public_reward_fn][{problem_id}] No code extracted")
        result = EnvResult(
            reward=0.0,
            is_valid=False,
            observation="",
            metadata={"error": "no_code", "problem_id": problem_id},
            fail_type="code_extraction_failed",
        )
        return 0.0, result, "", 0.0

    try:
        from ale_bench.session import CodeLanguage

        # Prefer the env's already-built session; creating a fresh session inside
        # a ProcessPoolExecutor worker re-extracts the problem data and rebuilds
        # the Rust tools, which is very slow.
        if env is not None and hasattr(env, "session") and env.session is not None:
            session = env.session
            logger.info(
                f"[ale_bench_public_reward_fn][{problem_id}] Reusing env.session"
            )
        else:
            logger.info(
                f"[ale_bench_public_reward_fn][{problem_id}] Getting session "
                f"(lite={lite_version}, workers={ale_bench_num_workers})"
            )
            session = _get_ale_bench_session(
                problem_id=problem_id,
                lite_version=lite_version,
                session_duration_hours=session_duration_hours,
                ale_bench_num_workers=ale_bench_num_workers,
            )
        logger.info(
            f"[ale_bench_public_reward_fn][{problem_id}] Running public_eval "
            f"(code_len={len(code)}, session_workers={ale_bench_num_workers})"
        )
        public_result = session.public_eval(
            code=code,
            code_language=CodeLanguage.CPP20,
        )
        case_scores = [_case_absolute_score(c) for c in public_result.case_results]
        median_score = float(np.median(case_scores)) if case_scores else 0.0
        public_rank = -1
        public_perf = -1
        if hasattr(public_result, "rank"):
            public_rank = int(public_result.rank)
        if hasattr(public_result, "performance"):
            public_perf = int(public_result.performance)

        logger.info(
            f"[ale_bench_public_reward_fn][{problem_id}] public_eval done: "
            f"median={median_score:.2f} "
            f"abs={getattr(public_result, 'overall_absolute_score', 0.0):.2f} "
            f"judge={getattr(public_result, 'overall_judge_result', 'UNKNOWN')}"
        )

        result = EnvResult(
            reward=median_score,
            is_valid=True,
            observation="",
            metadata={
                "problem_id": problem_id,
                "code": code,
                "public_median": median_score,
                "public_overall_absolute": float(
                    getattr(public_result, "overall_absolute_score", 0.0)
                ),
                "public_overall_relative": float(
                    getattr(public_result, "overall_relative_score", 0.0) or 0.0
                ),
                "public_judge_result": str(
                    getattr(public_result, "overall_judge_result", "UNKNOWN")
                ),
                "public_num_cases": len(case_scores),
                "public_rank": public_rank,
                "public_performance": public_perf,
            },
        )
        exec_time_ms = (time.time() - start_time) * 1000.0
        return median_score, result, code, exec_time_ms
    except Exception as e:
        logger.error(
            f"[ale_bench_public_reward_fn][{problem_id}] public_eval failed: "
            f"{type(e).__name__}: {e}"
        )
        result = EnvResult(
            reward=0.0,
            is_valid=False,
            observation="",
            metadata={
                "error": f"{type(e).__name__}: {e}",
                "problem_id": problem_id,
                "traceback": traceback.format_exc(),
            },
            fail_type="execution_error",
        )
        return 0.0, result, code if code is not None else "", 0.0


def _private_eval_one_problem(
    problem_id: str,
    best_code: str,
    candidate_results: list[dict[str, Any]],
    lite_version: bool,
    session_duration_hours: float,
    ale_bench_num_workers: int,
    session: Any | None = None,
    selection_method: str = "median",
    score_type: str | None = None,
) -> dict[str, Any]:
    """Run only the private evaluation for the best candidate of a problem.

    ``candidate_results`` must already contain the public evaluation scores
    (e.g. produced by ``ale_bench_public_reward_fn``).  The best candidate is
    selected according to ``selection_method``.

    If a session is provided (e.g. an env session shared with public eval), it
    is reused for the single ``private_eval`` call.  Otherwise a cached session
    is obtained via ``_get_ale_bench_session``.  Provided sessions are owned by
    the caller and are not closed here.

    Args:
        session: Optional pre-built ALE-Bench session to reuse. If provided,
            the caller retains ownership and this function will not close it.
        selection_method: ``"median"`` for official ALE-Bench median selection
            (closest to median of ``overall_absolute_score``),
            ``"median_case_score"`` for legacy highest per-candidate median, or
            ``"best_public"`` for the best public overall absolute score.
        score_type: ``"minimize"`` or ``"maximize"``. If not provided, loaded
            from the problem metadata.
    """
    import time

    from ale_bench.session import CodeLanguage

    result: dict[str, Any] = {
        "problem_id": problem_id,
        "candidates": candidate_results,
        "best_candidate_idx": None,
        "error": None,
    }

    if not candidate_results:
        result["error"] = "no candidates"
        return result

    if score_type is None:
        score_type = _get_problem_score_type(problem_id, lite_version)

    best_idx = _select_best_candidate_index(
        candidate_results, selection_method, score_type
    )
    result["best_candidate_idx"] = best_idx
    result["selection_method"] = selection_method
    result["score_type"] = score_type
    best_code = candidate_results[best_idx].get("code", best_code)

    owns_session = session is None
    step_ts = time.time()
    try:
        logger.info(
            f"[_private_eval_one_problem][{problem_id}] "
            f"private_eval best candidate {best_idx}"
        )
        if session is None:
            logger.info(
                f"[_private_eval_one_problem][{problem_id}] "
                f"No provided session, using _get_ale_bench_session"
            )
            session = _get_ale_bench_session(
                problem_id=problem_id,
                lite_version=lite_version,
                session_duration_hours=session_duration_hours,
                ale_bench_num_workers=ale_bench_num_workers,
            )
        else:
            logger.info(
                f"[_private_eval_one_problem][{problem_id}] Reusing provided session"
            )
        eval_ts = time.time()
        logger.info(f"[_private_eval_one_problem][{problem_id}] Calling private_eval()")
        private_result, rank, performance = session.private_eval(
            code=best_code,
            code_language=CodeLanguage.CPP20,
        )
        logger.info(
            f"[_private_eval_one_problem][{problem_id}] private_eval() returned "
            f"after {time.time() - eval_ts:.1f}s"
        )
        result["private"] = {
            "absolute_score": float(
                getattr(private_result, "overall_absolute_score", 0.0)
            ),
            "relative_score": float(
                getattr(private_result, "overall_relative_score", 0.0) or 0.0
            ),
            "judge_result": str(
                getattr(private_result, "overall_judge_result", "UNKNOWN")
            ),
            "rank": int(rank),
            "performance": int(performance),
        }
        logger.info(
            f"[_private_eval_one_problem][{problem_id}] done: "
            f"abs={result['private']['absolute_score']:.2f} "
            f"rank={rank} perf={performance}"
        )
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        logger.error(
            f"[_private_eval_one_problem][{problem_id}] failed: {result['error']}"
        )
    finally:
        if owns_session and session is not None:
            logger.info(
                f"[_private_eval_one_problem][{problem_id}] "
                f"Closing owned session (total {time.time() - step_ts:.1f}s)"
            )
            try:
                session.close()
            except Exception:
                pass

    return result


def _eval_one_problem(
    problem_id: str,
    candidate_codes: list[str],
    lite_version: bool,
    session_duration_hours: float,
    ale_bench_num_workers: int,
) -> dict[str, Any]:
    """Evaluate all candidates for one problem and return the best private result.

    This function is self-contained so it can run in a ProcessPoolExecutor
    worker without carrying unpicklable trainer state.
    """
    from ale_bench import start
    from ale_bench.session import CodeLanguage

    result: dict[str, Any] = {
        "problem_id": problem_id,
        "candidates": [],
        "best_candidate_idx": None,
        "error": None,
    }

    session = None
    try:
        logger.info(
            f"[_eval_one_problem][{problem_id}] Starting eval with "
            f"{len(candidate_codes)} candidates (lite={lite_version}, "
            f"num_workers={ale_bench_num_workers})"
        )
        session = start(
            problem_id=problem_id,
            lite_version=lite_version,
            use_same_time_scale=False,
            session_duration=timedelta(hours=session_duration_hours),
            num_workers=ale_bench_num_workers,
            run_visualization_server=False,
        )

        candidate_medians: list[float] = []
        for idx, code in enumerate(candidate_codes):
            logger.info(
                f"[_eval_one_problem][{problem_id}] public_eval candidate {idx}/{len(candidate_codes)}"
            )
            public_result = session.public_eval(
                code=code,
                code_language=CodeLanguage.CPP20,
            )
            case_scores = [_case_absolute_score(c) for c in public_result.case_results]
            median_score = float(np.median(case_scores)) if case_scores else 0.0
            candidate_medians.append(median_score)

            public_rank = -1
            public_perf = -1
            if hasattr(public_result, "rank"):
                public_rank = int(public_result.rank)
            if hasattr(public_result, "performance"):
                public_perf = int(public_result.performance)
            result["candidates"].append(
                {
                    "idx": idx,
                    "public": {
                        "median_case_score": median_score,
                        "overall_absolute_score": float(
                            getattr(public_result, "overall_absolute_score", 0.0)
                        ),
                        "overall_relative_score": float(
                            getattr(public_result, "overall_relative_score", 0.0) or 0.0
                        ),
                        "judge_result": str(
                            getattr(public_result, "overall_judge_result", "UNKNOWN")
                        ),
                        "num_cases": len(case_scores),
                        "rank": public_rank,
                        "performance": public_perf,
                    },
                }
            )
            logger.info(
                f"[_eval_one_problem][{problem_id}] public candidate {idx}: "
                f"median={median_score:.2f} abs={getattr(public_result, 'overall_absolute_score', 0.0):.2f} "
                f"judge={getattr(public_result, 'overall_judge_result', 'UNKNOWN')} "
                f"rank={public_rank} perf={public_perf}"
            )

        best_idx = int(np.argmax(candidate_medians))
        result["best_candidate_idx"] = best_idx
        best_code = candidate_codes[best_idx]

        logger.info(
            f"[_eval_one_problem][{problem_id}] private_eval best candidate {best_idx}"
        )
        private_result, rank, performance = session.private_eval(
            code=best_code,
            code_language=CodeLanguage.CPP20,
        )
        logger.info(
            f"[_eval_one_problem][{problem_id}] eval done: "
            f"private_abs={getattr(private_result, 'overall_absolute_score', 0.0):.2f}"
        )
        result["private"] = {
            "absolute_score": float(
                getattr(private_result, "overall_absolute_score", 0.0)
            ),
            "relative_score": float(
                getattr(private_result, "overall_relative_score", 0.0) or 0.0
            ),
            "judge_result": str(
                getattr(private_result, "overall_judge_result", "UNKNOWN")
            ),
            "rank": int(rank),
            "performance": int(performance),
        }
        logger.info(
            f"[_eval_one_problem][{problem_id}] private best {best_idx}: "
            f"abs={getattr(private_result, 'overall_absolute_score', 0.0):.2f} "
            f"rel={getattr(private_result, 'overall_relative_score', 0.0) or 0.0:.2f} "
            f"rank={rank} perf={performance}"
        )
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        logger.error(
            f"[_eval_one_problem][{problem_id}] eval failed: {result['error']}"
        )
    finally:
        if session is not None:
            try:
                session.close()
                logger.info(f"[_eval_one_problem][{problem_id}] session closed")
            except Exception:
                pass

    return result


def _average_private_scores(
    results: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """Compute average private absolute/relative scores over successful results."""
    abs_scores: list[float] = []
    rel_scores: list[float] = []
    performances: list[int] = []
    for r in results:
        if r.get("error") or "private" not in r:
            continue
        abs_scores.append(r["private"]["absolute_score"])
        rel_scores.append(r["private"]["relative_score"])
        performances.append(r["private"]["performance"])

    if not abs_scores:
        return {
            "count": 0,
            "absolute_score": 0.0,
            "relative_score": 0.0,
            "performance": 0.0,
        }

    return {
        "count": len(abs_scores),
        "absolute_score": float(np.mean(abs_scores)),
        "relative_score": float(np.mean(rel_scores)),
        "performance": float(np.mean(performances)),
    }


@dataclass
class AleBenchEvalSpec:
    """Specification for a full ALE-Bench evaluation pass."""

    problem_ids: list[str]
    n_candidates: int = 15
    lite_version: bool = False
    session_duration_hours: float = 4.0
    ale_bench_num_workers: int = 1
    n_parallel_problems: int = 1
    training_problem_ids: set[str] = field(default_factory=set)
    output_path: str = ""


def _log_problem_result(result: dict[str, Any]) -> None:
    """Log a concise summary of one problem's evaluation result."""
    problem_id = result.get("problem_id", "unknown")
    if result.get("error"):
        logger.info(f"[Result][{problem_id}] error={result['error']}")
        return
    candidates = result.get("candidates", [])
    public_summary = " ".join(
        f"c{i}[median={c.get('public', {}).get('median_case_score', 0.0):.2f} "
        f"abs={c.get('public', {}).get('overall_absolute_score', 0.0):.2f} "
        f"perf={c.get('public', {}).get('performance', -1)}]"
        for i, c in enumerate(candidates)
    )
    private = result.get("private", {})
    logger.info(
        f"[Result][{problem_id}] best_idx={result.get('best_candidate_idx')} "
        f"public={public_summary} "
        f"private[abs={private.get('absolute_score', 0.0):.2f} "
        f"rank={private.get('rank', -1)} perf={private.get('performance', -1)}]"
    )


def evaluate_problem_subset(
    problem_ids: list[str],
    candidates_by_problem: dict[str, list[str]],
    lite_version: bool,
    session_duration_hours: float,
    ale_bench_num_workers: int,
    n_parallel_problems: int = 1,
) -> list[dict[str, Any]]:
    """Run public→private evaluation on a subset of ALE-Bench problems.

    This is the unit of work that can be distributed across data-parallel
    ranks: each rank evaluates the problems assigned to it and the caller
    gathers the per-rank results afterwards.

    Args:
        problem_ids: Problems to evaluate on this rank.
        candidates_by_problem: Mapping from problem_id to candidate codes.
        lite_version: Whether to use ALE-Bench lite sessions.
        session_duration_hours: Time budget passed to ``ale_bench.start``.
        ale_bench_num_workers: ``num_workers`` passed to ``ale_bench.start``.
        n_parallel_problems: Number of problems to evaluate concurrently in a
            local process pool.

    Returns:
        List of per-problem result dictionaries in the same order as
        ``problem_ids``.
    """
    total_candidates = sum(
        len(candidates_by_problem.get(pid, [])) for pid in problem_ids
    )
    logger.info(
        f"[evaluate_problem_subset] Starting {len(problem_ids)} problems, "
        f"{total_candidates} total candidates, n_parallel={n_parallel_problems}"
    )
    if n_parallel_problems > 1:
        with ProcessPoolExecutor(max_workers=n_parallel_problems) as executor:
            futures = {
                executor.submit(
                    _eval_one_problem,
                    problem_id,
                    candidates_by_problem.get(problem_id, []),
                    lite_version,
                    session_duration_hours,
                    ale_bench_num_workers,
                ): problem_id
                for problem_id in problem_ids
            }
            results = []
            for future in futures:
                problem_id = futures[future]
                logger.info(
                    f"[evaluate_problem_subset] Waiting for problem {problem_id}"
                )
                result = future.result()
                _log_problem_result(result)
                results.append(result)
                logger.info(f"[evaluate_problem_subset] Finished problem {problem_id}")
        logger.info("[evaluate_problem_subset] All parallel problems done")
        return results

    results = []
    for problem_id in problem_ids:
        logger.info(f"[evaluate_problem_subset] Evaluating problem {problem_id}")
        result = _eval_one_problem(
            problem_id,
            candidates_by_problem.get(problem_id, []),
            lite_version,
            session_duration_hours,
            ale_bench_num_workers,
        )
        _log_problem_result(result)
        results.append(result)
        logger.info(f"[evaluate_problem_subset] Finished problem {problem_id}")
    logger.info("[evaluate_problem_subset] All problems done")
    return results


def evaluate_problem_subset_with_public_scores(
    problem_ids: list[str],
    public_results_by_problem: dict[str, list[dict[str, Any]]],
    lite_version: bool,
    session_duration_hours: float,
    ale_bench_num_workers: int,
    n_parallel_problems: int = 1,
    problem_sessions: dict[str, Any] | None = None,
    selection_method: str = "median",
    problem_score_types: dict[str, str] | None = None,
    per_problem_timeout: float | None = None,
    executor: ProcessPoolExecutor | None = None,
) -> list[dict[str, Any]]:
    """Run only private evaluation after public scores are already known.

    This is used when public evaluation is performed inside the rollout reward
    function.  The best candidate per problem is selected from the pre-computed
    public scores and only the private evaluation is run here.

    Args:
        problem_ids: Problems to evaluate on this rank.
        public_results_by_problem: Mapping from problem_id to list of public
            result dicts (one per candidate).  Each dict must contain a
            ``public`` entry with ``overall_absolute_score`` (for ``median``)
            or ``median_case_score`` (for ``median_case_score``).
        lite_version: Whether to use ALE-Bench lite sessions.
        session_duration_hours: Time budget passed to ``ale_bench.start``.
        ale_bench_num_workers: ``num_workers`` passed to ``ale_bench.start``.
        n_parallel_problems: Number of problems to evaluate concurrently in a
            local process pool.
        problem_sessions: Deprecated. ALE-Bench sessions can only perform one
            ``private_eval`` each, so this mapping is no longer used.  A fresh
            session is created for every candidate.
        selection_method: ``"median"`` for official ALE-Bench median selection,
            ``"median_case_score"`` for legacy highest per-candidate median, or
            ``"best_public"`` for the best public overall absolute score.
        problem_score_types: Optional mapping from problem_id to ``"minimize"``
            or ``"maximize"``. Used for ``"best_public"`` and
            ``"median_case_score"`` selection. If a problem is missing, the
            score type is loaded from problem metadata.
        per_problem_timeout: Optional timeout in seconds for each problem's
            private evaluation. If None, wait indefinitely.
        executor: Optional external ProcessPoolExecutor to use. If provided, the
            caller is responsible for shutting it down. If None, a temporary
            executor is created and destroyed inside this function.

    Returns:
        List of per-problem result dictionaries in the same order as
        ``problem_ids``.
    """
    import time

    overall_ts = time.time()
    logger.info(
        f"[evaluate_problem_subset_with_public_scores] "
        f"Starting private eval for {len(problem_ids)} problems, "
        f"n_parallel={n_parallel_problems}, selection={selection_method}"
    )

    problem_sessions = problem_sessions or {}
    problem_score_types = problem_score_types or {}

    results_by_problem: dict[str, dict[str, Any]] = {}
    total = len(problem_ids)

    if n_parallel_problems > 1:
        external_executor = executor is not None
        if executor is None:
            logger.info(
                f"[evaluate_problem_subset_with_public_scores] "
                f"Creating ProcessPoolExecutor with {n_parallel_problems} workers"
            )
            executor = ProcessPoolExecutor(max_workers=n_parallel_problems)
        try:
            submit_ts = time.time()
            futures = {
                executor.submit(
                    _private_eval_one_problem,
                    problem_id,
                    "",
                    public_results_by_problem.get(problem_id, []),
                    lite_version,
                    session_duration_hours,
                    ale_bench_num_workers,
                    problem_sessions.get(problem_id),
                    selection_method,
                    problem_score_types.get(problem_id),
                ): problem_id
                for problem_id in problem_ids
            }
            logger.info(
                f"[evaluate_problem_subset_with_public_scores] "
                f"Submitted {len(futures)} tasks in {time.time() - submit_ts:.1f}s"
            )
            completed = 0
            for future in as_completed(futures):
                problem_id = futures[future]
                wait_ts = time.time()
                result = future.result(timeout=per_problem_timeout)
                results_by_problem[problem_id] = result
                completed += 1
                private = result.get("private", {})
                logger.info(
                    f"[PrivateEval][{completed}/{total}] {problem_id} done "
                    f"(waited {time.time() - wait_ts:.1f}s): "
                    f"abs={private.get('absolute_score', 0.0):.2f} "
                    f"rank={private.get('rank', -1)} perf={private.get('performance', -1)}"
                )
        finally:
            if not external_executor:
                logger.info(
                    "[evaluate_problem_subset_with_public_scores] "
                    "Shutting down ProcessPoolExecutor"
                )
                executor.shutdown(wait=True)
        logger.info(
            "[evaluate_problem_subset_with_public_scores] All parallel problems done"
        )
    else:
        for idx, problem_id in enumerate(problem_ids, start=1):
            logger.info(f"[PrivateEval][{idx}/{total}] Evaluating problem {problem_id}")
            result = _private_eval_one_problem(
                problem_id,
                "",
                public_results_by_problem.get(problem_id, []),
                lite_version,
                session_duration_hours,
                ale_bench_num_workers,
                problem_sessions.get(problem_id),
                selection_method,
                problem_score_types.get(problem_id),
            )
            results_by_problem[problem_id] = result
            private = result.get("private", {})
            logger.info(
                f"[PrivateEval][{idx}/{total}] {problem_id} done: "
                f"abs={private.get('absolute_score', 0.0):.2f} "
                f"rank={private.get('rank', -1)} perf={private.get('performance', -1)}"
            )
        logger.info("[evaluate_problem_subset_with_public_scores] All problems done")

    logger.info(
        f"[evaluate_problem_subset_with_public_scores] "
        f"Finished {len(problem_ids)} problems in {time.time() - overall_ts:.1f}s"
    )

    # Preserve the caller's problem order.
    return [results_by_problem[pid] for pid in problem_ids]


def _compute_private_stats(
    private_results: list[dict[str, Any]], score_type: str
) -> dict[str, Any]:
    """Compute statistics over a list of per-candidate private results."""
    successful = [r for r in private_results if "error" not in r]
    count = len(private_results)
    count_accepted = sum(
        1 for r in successful if _judge_result_is_accepted(r.get("judge_result"))
    )
    stats: dict[str, Any] = {
        "count": count,
        "count_successful": len(successful),
        "count_accepted": count_accepted,
    }
    if not successful:
        for metric in ("absolute_score", "relative_score", "performance"):
            stats[metric] = {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p25": 0.0,
                "p75": 0.0,
                "skew": 0.0,
                "kurtosis": 0.0,
                "bimodality_coefficient": 0.0,
            }
        stats["best_candidate_idx"] = None
        stats["best_absolute_score"] = 0.0
        return stats

    n = len(successful)
    for metric in ("absolute_score", "relative_score", "performance"):
        vals = [r[metric] for r in successful]
        arr = np.array(vals, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        skew = 0.0
        kurt = 0.0
        bc = 0.0
        if std > 1e-9:
            z = (arr - mean) / std
            skew = float(np.mean(z**3))
            # Excess kurtosis (Fisher), 0 for normal.
            kurt = float(np.mean(z**4) - 3.0)
            # Sarle's bimodality coefficient. BC > 0.55 suggests bimodality.
            denom_term = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
            denom = kurt + denom_term
            if abs(denom) > 1e-9:
                bc = (skew**2 + 1.0) / denom
        stats[metric] = {
            "mean": mean,
            "median": float(np.median(arr)),
            "std": std,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "skew": skew,
            "kurtosis": kurt,
            "bimodality_coefficient": bc,
        }

    is_minimize = str(score_type).lower().strip() == "minimize"
    abs_vals = [r["absolute_score"] for r in successful]
    best_local_idx = (
        int(np.argmin(abs_vals)) if is_minimize else int(np.argmax(abs_vals))
    )
    best_result = successful[best_local_idx]
    stats["best_candidate_idx"] = best_result["idx"]
    stats["best_absolute_score"] = best_result["absolute_score"]
    return stats


def average_all_candidates_stats(
    stats_list: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Average per-problem all-candidate statistics across multiple problems."""
    if not stats_list:
        return {}

    avg_stats: dict[str, Any] = {}
    for key in ("count", "count_successful", "count_accepted"):
        vals = [s.get(key, 0) for s in stats_list]
        avg_stats[key] = float(np.mean(vals))

    for metric in ("absolute_score", "relative_score", "performance"):
        agg: dict[str, float] = {}
        for subkey in ("mean", "median", "std", "min", "max"):
            vals = [s.get(metric, {}).get(subkey, 0.0) for s in stats_list]
            agg[subkey] = float(np.mean(vals))
        avg_stats[metric] = agg

    return avg_stats


def derive_selection_outputs_from_all_candidates(
    public_results_by_problem: dict[str, list[dict[str, Any]]],
    all_candidates_results: Sequence[dict[str, Any]],
    problem_ids: Sequence[str],
    training_problem_ids: set[str],
    selection_methods: Sequence[str],
    problem_score_types: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Derive median/best_public outputs from already-completed private evals.

    When every candidate has been evaluated privately, there is no need to run
    additional private evals for the selected candidate. Instead, select the
    candidate using the public scores and look up its private result from the
    all-candidates pool.
    """
    results_by_problem = {r["problem_id"]: r for r in all_candidates_results}
    outputs: dict[str, dict[str, Any]] = {}

    for sel_method in selection_methods:
        per_problem_results: list[dict[str, Any]] = []
        for pid in problem_ids:
            result = results_by_problem.get(pid)
            public_candidates = public_results_by_problem.get(pid, [])
            score_type = problem_score_types.get(pid, "minimize")

            if result is None or result.get("error"):
                per_problem_results.append(
                    {
                        "problem_id": pid,
                        "best_candidate_idx": None,
                        "error": result.get("error") if result else "missing result",
                    }
                )
                continue

            private_results = result.get("private_results", [])
            best_idx = _select_best_candidate_index(
                public_candidates, sel_method, score_type
            )
            private_result = next(
                (r for r in private_results if r.get("idx") == best_idx), None
            )
            if private_result is None or "error" in private_result:
                per_problem_results.append(
                    {
                        "problem_id": pid,
                        "best_candidate_idx": best_idx,
                        "error": "selected candidate private eval missing",
                    }
                )
                continue

            per_problem_results.append(
                {
                    "problem_id": pid,
                    "best_candidate_idx": best_idx,
                    "selection_method": sel_method,
                    "candidates": public_candidates,
                    "private": {
                        "absolute_score": private_result["absolute_score"],
                        "relative_score": private_result["relative_score"],
                        "judge_result": str(
                            private_result.get("judge_result", "UNKNOWN")
                        ),
                        "rank": int(private_result.get("rank", -1)),
                        "performance": int(private_result.get("performance", -1)),
                    },
                }
            )

        combined = combine_ale_bench_results(
            per_problem_results, problem_ids, training_problem_ids
        )
        combined["selection_method"] = sel_method
        outputs[sel_method] = combined

    return outputs


def _private_eval_all_candidates_one_problem(
    problem_id: str,
    candidate_results: list[dict[str, Any]],
    lite_version: bool,
    session_duration_hours: float,
    ale_bench_num_workers: int,
    session: Any | None = None,
    score_type: str | None = None,
) -> dict[str, Any]:
    """Run private evaluation on *every* candidate for one problem.

    ALE-Bench hard-codes ``maximum_resource_usage.num_call_private_eval=1``.
    To evaluate multiple candidates on the same session without rebuilding Rust
    tools, this function resets the session's private-eval counter before each
    candidate.  Sessions are not shared across processes; each worker creates
    and caches its own session via ``_get_ale_bench_session``.

    Returns a dict with per-candidate private results and aggregated statistics.
    """
    from ale_bench.session import CodeLanguage

    result: dict[str, Any] = {
        "problem_id": problem_id,
        "candidates": candidate_results,
        "private_results": [],
        "private_stats": {},
        "error": None,
    }

    if not candidate_results:
        result["error"] = "no candidates"
        return result

    if score_type is None:
        score_type = _get_problem_score_type(problem_id, lite_version)
    result["score_type"] = score_type

    try:
        session = _get_ale_bench_session(
            problem_id=problem_id,
            lite_version=lite_version,
            session_duration_hours=session_duration_hours,
            ale_bench_num_workers=ale_bench_num_workers,
        )

        private_results: list[dict[str, Any]] = []
        n_total = len(candidate_results)
        for cand in candidate_results:
            code = cand.get("code", "")
            cand_idx = cand.get("idx", len(private_results))
            progress = f"{len(private_results) + 1}/{n_total}"
            try:
                logger.info(
                    f"[_private_eval_all_candidates_one_problem][{problem_id}] "
                    f"candidate {cand_idx} [{progress}] private_eval starting"
                )
                _reset_private_eval_counter(session)
                private_result, rank, performance = session.private_eval(
                    code=code,
                    code_language=CodeLanguage.CPP20,
                )
                absolute_score = float(
                    getattr(private_result, "overall_absolute_score", 0.0)
                )
                relative_score = float(
                    getattr(private_result, "overall_relative_score", 0.0) or 0.0
                )
                judge_result = str(
                    getattr(private_result, "overall_judge_result", "UNKNOWN")
                )
                private_results.append(
                    {
                        "idx": cand_idx,
                        "absolute_score": absolute_score,
                        "relative_score": relative_score,
                        "judge_result": judge_result,
                        "rank": int(rank),
                        "performance": int(performance),
                    }
                )
                logger.info(
                    f"[_private_eval_all_candidates_one_problem][{problem_id}] "
                    f"candidate {cand_idx} [{progress}] done: "
                    f"abs={absolute_score:.2f} rank={rank} perf={performance} "
                    f"judge={judge_result}"
                )
            except Exception as e:
                logger.warning(
                    f"[_private_eval_all_candidates_one_problem][{problem_id}] "
                    f"candidate {cand_idx} [{progress}] failed: {e}"
                )
                private_results.append(
                    {
                        "idx": cand_idx,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )

        # Leave the session in a usable state (not "finished") so the caller can
        # keep using it for public eval or another round of private eval.
        _reset_private_eval_counter(session)

        result["private_results"] = private_results
        result["private_stats"] = _compute_private_stats(private_results, score_type)
        logger.info(
            f"[_private_eval_all_candidates_one_problem][{problem_id}] done: "
            f"evaluated {len(private_results)} candidates, "
            f"best_idx={result['private_stats'].get('best_candidate_idx')}, "
            f"mean_abs={result['private_stats'].get('absolute_score', {}).get('mean', 0.0):.2f}"
        )
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        logger.error(
            f"[_private_eval_all_candidates_one_problem][{problem_id}] failed: {result['error']}"
        )

    return result


def evaluate_all_candidates_private(
    problem_ids: list[str],
    public_results_by_problem: dict[str, list[dict[str, Any]]],
    lite_version: bool,
    session_duration_hours: float,
    ale_bench_num_workers: int,
    n_parallel_problems: int = 1,
    problem_sessions: dict[str, Any] | None = None,
    problem_score_types: dict[str, str] | None = None,
    per_problem_timeout: float | None = None,
    executor: ProcessPoolExecutor | None = None,
) -> list[dict[str, Any]]:
    """Run private evaluation on every candidate for a subset of problems.

    This mirrors ``evaluate_problem_subset_with_public_scores`` but evaluates
    all candidates instead of selecting one, and returns per-problem statistics.

    Args:
        problem_ids: Problems to evaluate on this rank.
        public_results_by_problem: Mapping from problem_id to list of public
            result dicts (one per candidate).  Each dict must contain a
            ``code`` field and a ``public`` entry.
        lite_version: Whether to use ALE-Bench lite sessions.
        session_duration_hours: Time budget passed to ``ale_bench.start``.
        ale_bench_num_workers: ``num_workers`` passed to ``ale_bench.start``.
        n_parallel_problems: Number of problems to evaluate concurrently in a
            local process pool.
        problem_sessions: Deprecated and ignored. ALE-Bench sessions are not
            safe to share across forked workers. Each worker creates and caches
            its own session via ``_get_ale_bench_session``.
        problem_score_types: Optional mapping from problem_id to ``"minimize"``
            or ``"maximize"``.  Used for private-score statistics.
        per_problem_timeout: Optional timeout in seconds for each problem's
            private evaluation. If None, wait indefinitely.
        executor: Optional external ProcessPoolExecutor to use. If provided, the
            caller is responsible for shutting it down. If None, a temporary
            executor is created and destroyed inside this function.

    Returns:
        List of per-problem result dictionaries in the same order as
        ``problem_ids``.
    """
    logger.info(
        f"[evaluate_all_candidates_private] Starting private eval for "
        f"{len(problem_ids)} problems, n_parallel={n_parallel_problems}"
    )

    if problem_sessions:
        logger.warning(
            "[evaluate_all_candidates_private] "
            "problem_sessions is deprecated and ignored; each worker creates its own session."
        )
    problem_score_types = problem_score_types or {}
    results_by_problem: dict[str, dict[str, Any]] = {}
    total = len(problem_ids)

    if n_parallel_problems > 1:
        external_executor = executor is not None
        if executor is None:
            executor = ProcessPoolExecutor(max_workers=n_parallel_problems)
        try:
            futures = {
                executor.submit(
                    _private_eval_all_candidates_one_problem,
                    problem_id,
                    public_results_by_problem.get(problem_id, []),
                    lite_version,
                    session_duration_hours,
                    ale_bench_num_workers,
                    None,
                    problem_score_types.get(problem_id),
                ): problem_id
                for problem_id in problem_ids
            }
            completed = 0
            for future in as_completed(futures):
                problem_id = futures[future]
                result = future.result(timeout=per_problem_timeout)
                results_by_problem[problem_id] = result
                completed += 1
                stats = result.get("private_stats", {})
                logger.info(
                    f"[AllCandidatesPrivate][{completed}/{total}] {problem_id} done: "
                    f"count={stats.get('count', 0)} "
                    f"accepted={stats.get('count_accepted', 0)} "
                    f"mean_abs={stats.get('absolute_score', {}).get('mean', 0.0):.2f}"
                )
        finally:
            if not external_executor:
                executor.shutdown(wait=True)
        logger.info("[evaluate_all_candidates_private] All parallel problems done")
    else:
        for idx, problem_id in enumerate(problem_ids, start=1):
            logger.info(
                f"[AllCandidatesPrivate][{idx}/{total}] Evaluating problem {problem_id}"
            )
            result = _private_eval_all_candidates_one_problem(
                problem_id,
                public_results_by_problem.get(problem_id, []),
                lite_version,
                session_duration_hours,
                ale_bench_num_workers,
                None,
                problem_score_types.get(problem_id),
            )
            results_by_problem[problem_id] = result
            stats = result.get("private_stats", {})
            logger.info(
                f"[AllCandidatesPrivate][{idx}/{total}] {problem_id} done: "
                f"count={stats.get('count', 0)} "
                f"accepted={stats.get('count_accepted', 0)} "
                f"mean_abs={stats.get('absolute_score', {}).get('mean', 0.0):.2f}"
            )
        logger.info("[evaluate_all_candidates_private] All problems done")

    # Preserve the caller's problem order.
    return [results_by_problem[pid] for pid in problem_ids]


def combine_ale_bench_results(
    results: Sequence[dict[str, Any]],
    problem_ids: Sequence[str],
    training_problem_ids: set[str],
) -> dict[str, Any]:
    """Combine per-problem results into the final aggregated report.

    Args:
        results: Per-problem result dictionaries collected from all ranks.
        problem_ids: Full ordered list of problem IDs that were evaluated.
        training_problem_ids: Set of problem IDs seen during training.

    Returns:
        Result dictionary with ``average_all`` and
        ``average_out_of_training`` aggregates plus the ordered per-problem
        ``results`` list.
    """
    results_by_problem = {r["problem_id"]: r for r in results}
    ordered_results = [
        results_by_problem.get(
            pid,
            {
                "problem_id": pid,
                "candidates": [],
                "best_candidate_idx": None,
                "error": "missing result",
            },
        )
        for pid in problem_ids
    ]

    avg_all = _average_private_scores(ordered_results)
    out_of_training = [
        r for r in ordered_results if r["problem_id"] not in training_problem_ids
    ]
    avg_out_of_training = _average_private_scores(out_of_training)

    return {
        "problem_ids": list(problem_ids),
        "training_problem_ids": sorted(training_problem_ids),
        "average_all": avg_all,
        "average_out_of_training": avg_out_of_training,
        "results": ordered_results,
    }


def run_ale_bench_eval(
    candidates_by_problem: dict[str, list[str]],
    spec: AleBenchEvalSpec,
) -> dict[str, Any]:
    """Run public→private evaluation on all problems and save results.

    Args:
        candidates_by_problem: Mapping from problem_id to list of candidate codes.
        spec: Evaluation specification.

    Returns:
        Full results dictionary (also written to ``spec.output_path``).
    """
    if not spec.output_path:
        raise ValueError("output_path must be set in AleBenchEvalSpec")
    os.makedirs(os.path.dirname(spec.output_path), exist_ok=True)

    # Ensure every configured problem has the expected number of candidates.
    for problem_id in spec.problem_ids:
        codes = candidates_by_problem.get(problem_id)
        if codes is None:
            candidates_by_problem[problem_id] = [""] * spec.n_candidates
        elif len(codes) < spec.n_candidates:
            candidates_by_problem[problem_id] = codes + [""] * (
                spec.n_candidates - len(codes)
            )

    problem_results = evaluate_problem_subset(
        spec.problem_ids,
        candidates_by_problem,
        spec.lite_version,
        spec.session_duration_hours,
        spec.ale_bench_num_workers,
        spec.n_parallel_problems,
    )

    output = combine_ale_bench_results(
        problem_results, spec.problem_ids, spec.training_problem_ids
    )
    output.update(
        {
            "n_candidates": spec.n_candidates,
            "lite_version": spec.lite_version,
            "session_duration_hours": spec.session_duration_hours,
            "ale_bench_num_workers": spec.ale_bench_num_workers,
            "n_parallel_problems": spec.n_parallel_problems,
        }
    )

    with open(spec.output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output
