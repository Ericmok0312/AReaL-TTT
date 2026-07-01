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

import json
import logging
import os
import traceback
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import numpy as np

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
