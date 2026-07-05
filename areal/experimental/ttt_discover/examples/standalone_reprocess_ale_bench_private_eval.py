#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Standalone re-run of ALE-Bench private evaluation from saved public results.

This script loads a previously saved ``ale_bench_eval_results_*.json``,
re-selects the best candidate per problem, and runs private evaluation in
parallel using a simple ``ProcessPoolExecutor``.

Each worker builds **one** ALE-Bench session per problem and evaluates **all**
candidates on that session by resetting the private-eval counter before each
``private_eval``.  This avoids rebuilding the Rust tools for every candidate
and matches the session-reuse logic used by the main evaluation pipeline.

No LLM generation is performed.

Usage:
    python areal/experimental/ttt_discover/examples/standalone_reprocess_ale_bench_private_eval.py \
        --input path/to/ale_bench_eval_results_baseline_step_000000.json \
        --selection_method median \
        --n_parallel_problems 8 \
        --per_problem_timeout 600
"""

import argparse
import json
import os
import signal
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from typing import Any

import numpy as np

from areal.experimental.ttt_discover.ale_bench_eval import _compute_private_stats
from areal.utils import logging

logger = logging.getLogger("standalone_reprocess_ale_bench_private_eval")

# Global state used by the SIGINT/SIGTERM handler to cancel pending work.
_shutdown_requested = False
_current_executor: ProcessPoolExecutor | None = None


def _signal_handler(signum, frame) -> None:
    """Cancel pending futures and exit on Ctrl+C or pkill."""
    global _shutdown_requested
    _shutdown_requested = True
    sig_name = signal.Signals(signum).name
    logger.warning(
        f"[Main] Received {sig_name} (signal {signum}), shutting down pending private evals."
    )
    if _current_executor is not None:
        try:
            _current_executor.shutdown(wait=False, cancel_futures=True)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[Main] Failed to shutdown executor: {e}")
    sys.exit(1)


def _get_problem_score_type(problem_id: str, lite_version: bool) -> str:
    """Load ALE-Bench problem metadata and return its score type.

    Returns "minimize" or "maximize". Falls back to "minimize" on error.
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


def _case_absolute_score(case: Any) -> float:
    """Extract the absolute/raw score from a case result robustly."""
    for attr in ("absolute_score", "score", "raw_score"):
        val = getattr(case, attr, None)
        if val is not None:
            return float(val)
    return 0.0


def _is_accepted(candidate: dict[str, Any]) -> bool:
    """Return True if the candidate's public judge result is ACCEPTED.

    ALE-Bench returns ``JudgeResult`` enum instances. When stored as strings they
    may look like ``"ACCEPTED"`` or ``"JudgeResult.ACCEPTED"`` depending on how
    they were converted, so we normalise both forms.
    """
    judge_result = candidate.get("public", {}).get("judge_result")
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


def _reset_private_eval_counter(session: Any) -> None:
    """Reset ALE-Bench's per-session private-eval counter.

    ``ale_bench.start()`` hard-codes ``num_call_private_eval=1``.  Replacing the
    resource usage object with an equivalent one whose private-eval count is
    zero lets us run multiple ``private_eval`` calls on the same session without
    rebuilding Rust tools.
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


def _select_best_candidate_index(
    candidate_results: list[dict[str, Any]],
    selection_method: str = "median",
    score_type: str = "minimize",
) -> int:
    """Select the best candidate index using the chosen strategy.

    Args:
        candidate_results: List of candidate result dicts with public scores.
        selection_method: "median", "median_case_score", or "best_public".
        score_type: "minimize" or "maximize". Both ``best_public`` and
            ``median_case_score`` operate on raw absolute scores, so their
            selection direction follows the original problem semantics: lower
            is better for ``minimize``, higher is better for ``maximize``.
            Non-AC candidates are excluded when ``judge_result`` is available.
    """
    if not candidate_results:
        return 0

    is_minimize = str(score_type).lower().strip() == "minimize"

    def _ac_indices() -> list[int] | None:
        ac = [i for i, c in enumerate(candidate_results) if _is_accepted(c)]
        return ac if ac else None

    if selection_method == "median_case_score":
        # ``median_case_score`` is the median of per-case *raw* scores, so the
        # direction matches the original problem semantics.
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
        # ``overall_absolute_score`` is the sum of raw per-case absolute scores.
        # It is *not* a normalized standing score: lower is better for minimize
        # and higher is better for maximize. Prefer AC candidates.
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

    scores = [
        c.get("public", {}).get("overall_absolute_score", float("nan"))
        for c in candidate_results
    ]
    valid_scores = [s for s in scores if not np.isnan(s)]
    if not valid_scores:
        return _select_best_candidate_index(
            candidate_results, "median_case_score", score_type
        )

    median_score = float(np.median(valid_scores))
    distances = [
        abs(s - median_score) if not np.isnan(s) else float("inf") for s in scores
    ]
    return int(np.argmin(distances))


def _extract_public_results_by_problem(
    saved: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Build the public-results dict from a saved JSON."""
    public_results_by_problem: dict[str, list[dict[str, Any]]] = {}
    for result in saved.get("results", []):
        problem_id = result.get("problem_id")
        if not problem_id:
            continue
        candidates = result.get("candidates", [])
        public_results_by_problem[problem_id] = [
            {
                "idx": c.get("idx", i),
                "code": c.get("code", ""),
                "public": c.get("public", {}),
            }
            for i, c in enumerate(candidates)
        ]
    return public_results_by_problem


def _build_session(
    problem_id: str,
    lite_version: bool,
    session_duration_hours: float,
    ale_bench_num_workers: int,
) -> Any:
    """Build an ALE-Bench session with visible progress logging."""
    from ale_bench import start

    logger.info(f"[{problem_id}] Starting ALE-Bench session build")
    session_start = time.time()
    session = start(
        problem_id=problem_id,
        lite_version=lite_version,
        use_same_time_scale=False,
        session_duration=timedelta(hours=session_duration_hours),
        num_workers=ale_bench_num_workers,
        run_visualization_server=False,
    )
    logger.info(f"[{problem_id}] Session built in {time.time() - session_start:.1f}s")
    return session


def _run_all_candidates_private_for_problem(
    problem_id: str,
    candidate_results: list[dict[str, Any]],
    lite_version: bool,
    session_duration_hours: float,
    ale_bench_num_workers: int,
) -> dict[str, Any]:
    """Run private evaluation for *all* candidates of one problem.

    One ALE-Bench session is created per problem and reused for every candidate
    by resetting ``num_call_private_eval`` before each ``private_eval``.  The
    session is closed before the worker returns.
    """
    from ale_bench.session import CodeLanguage

    result: dict[str, Any] = {
        "problem_id": problem_id,
        "candidates": candidate_results,
        "private_results": [],
        "private_stats": {},
        "score_type": _get_problem_score_type(problem_id, lite_version),
        "error": None,
    }

    if not candidate_results:
        result["error"] = "no candidates"
        return result

    session = None
    try:
        session = _build_session(
            problem_id=problem_id,
            lite_version=lite_version,
            session_duration_hours=session_duration_hours,
            ale_bench_num_workers=ale_bench_num_workers,
        )
        private_results: list[dict[str, Any]] = []
        n_total = len(candidate_results)
        for i, cand in enumerate(candidate_results):
            code = cand.get("code", "")
            cand_idx = cand.get("idx", i)
            progress = f"{i + 1}/{n_total}"
            try:
                logger.info(
                    f"[{problem_id}] candidate {cand_idx} [{progress}] private_eval starting"
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
                    f"[{problem_id}] candidate {cand_idx} [{progress}] done: "
                    f"abs={absolute_score:.2f} rank={rank} perf={performance} judge={judge_result}"
                )
            except Exception as e:
                logger.warning(
                    f"[{problem_id}] candidate {cand_idx} [{progress}] failed: {e}"
                )
                private_results.append(
                    {
                        "idx": cand_idx,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )

        _reset_private_eval_counter(session)
        result["private_results"] = private_results
        result["private_stats"] = _compute_private_stats(
            private_results, result["score_type"]
        )
        logger.info(
            f"[{problem_id}] all-candidates private eval done: "
            f"evaluated {len(private_results)} candidates, "
            f"accepted={result['private_stats'].get('count_accepted', 0)}, "
            f"mean_abs={result['private_stats'].get('absolute_score', {}).get('mean', 0.0):.2f}"
        )
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        logger.error(
            f"[{problem_id}] failed to build session or run private evals: {result['error']}"
        )
    finally:
        if session is not None:
            try:
                session.close()
                logger.info(f"[{problem_id}] Session closed")
            except Exception:
                pass

    return result


def _derive_selection_results(
    all_results: list[dict[str, Any]],
    selection_methods: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Derive per-problem results for each selection method from private results."""
    outputs: dict[str, list[dict[str, Any]]] = {sel: [] for sel in selection_methods}
    for result in all_results:
        problem_id = result["problem_id"]
        candidate_results = result.get("candidates", [])
        private_results = result.get("private_results", [])
        score_type = result.get("score_type", "minimize")
        first_error = result.get("error")

        for sel in selection_methods:
            if first_error:
                outputs[sel].append(
                    {
                        "problem_id": problem_id,
                        "best_candidate_idx": None,
                        "error": first_error,
                    }
                )
                continue

            best_idx = _select_best_candidate_index(candidate_results, sel, score_type)
            private_result = next(
                (r for r in private_results if r.get("idx") == best_idx), None
            )
            if private_result is None or "error" in private_result:
                outputs[sel].append(
                    {
                        "problem_id": problem_id,
                        "best_candidate_idx": best_idx,
                        "error": "selected candidate private eval missing",
                    }
                )
                continue

            outputs[sel].append(
                {
                    "problem_id": problem_id,
                    "best_candidate_idx": best_idx,
                    "selection_method": sel,
                    "candidates": candidate_results,
                    "private": {
                        "absolute_score": private_result["absolute_score"],
                        "relative_score": private_result["relative_score"],
                        "judge_result": private_result["judge_result"],
                        "rank": private_result["rank"],
                        "performance": private_result["performance"],
                    },
                }
            )
    return outputs


def _average_private_scores(
    results: list[dict[str, Any]],
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


def _combine_results(
    results: list[dict[str, Any]],
    problem_ids: list[str],
    training_problem_ids: set[str],
) -> dict[str, Any]:
    """Combine per-problem results into the final aggregated report."""
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


def reprocess_single_file(
    input_path: str,
    output_path: str | None,
    selection_methods: list[str],
    ale_bench_num_workers: int,
    n_parallel_problems: int,
    session_duration_hours: float,
    per_problem_timeout: float,
) -> dict[str, Any]:
    """Re-run private eval for all candidates of one saved result file."""
    file_start = time.time()
    logger.info(f"Loading saved results from {input_path}")
    with open(input_path) as f:
        saved = json.load(f)

    problem_ids = saved.get("problem_ids", [])
    if not problem_ids:
        problem_ids = [
            r.get("problem_id") for r in saved.get("results", []) if r.get("problem_id")
        ]

    training_problem_ids = set(saved.get("training_problem_ids", []))
    public_results_by_problem = _extract_public_results_by_problem(saved)
    eval_problem_ids = [pid for pid in problem_ids if pid in public_results_by_problem]
    if len(eval_problem_ids) != len(problem_ids):
        logger.warning(
            f"Dropping {len(problem_ids) - len(eval_problem_ids)} problems "
            f"with no saved candidate public scores"
        )

    lite_version = bool(saved.get("lite_version", False))
    total_candidates = sum(len(v) for v in public_results_by_problem.values())
    logger.info(
        f"Loaded {len(eval_problem_ids)} problems, {total_candidates} total candidates"
    )
    logger.info(
        f"Re-running all-candidates private eval: "
        f"selection_methods={selection_methods}, "
        f"n_parallel={n_parallel_problems}, workers={ale_bench_num_workers}, "
        f"timeout={per_problem_timeout}s"
    )

    all_results: list[dict[str, Any]] = []
    completed = 0
    failed = 0

    if n_parallel_problems <= 1:
        # Sequential mode (useful for debugging).
        for i, problem_id in enumerate(eval_problem_ids, start=1):
            if _shutdown_requested:
                logger.warning("Shutdown requested, stopping sequential processing")
                break
            logger.info(f"[{i}/{len(eval_problem_ids)}] Processing {problem_id}")
            result = _run_all_candidates_private_for_problem(
                problem_id=problem_id,
                candidate_results=public_results_by_problem.get(problem_id, []),
                lite_version=lite_version,
                session_duration_hours=session_duration_hours,
                ale_bench_num_workers=ale_bench_num_workers,
            )
            all_results.append(result)
            if result.get("error"):
                failed += 1
            else:
                completed += 1
    else:
        # Parallel mode: each worker builds one session per problem, evaluates
        # all candidates by resetting the private-eval counter, and closes the
        # session.  We use as_completed so one hung problem does not block the
        # rest from being logged/saved.
        logger.info(
            f"Submitting {len(eval_problem_ids)} problems to {n_parallel_problems} workers"
        )
        global _current_executor
        with ProcessPoolExecutor(max_workers=n_parallel_problems) as executor:
            _current_executor = executor
            futures = {
                executor.submit(
                    _run_all_candidates_private_for_problem,
                    problem_id,
                    public_results_by_problem.get(problem_id, []),
                    lite_version,
                    session_duration_hours,
                    ale_bench_num_workers,
                ): problem_id
                for problem_id in eval_problem_ids
            }
            for future in as_completed(futures):
                problem_id = futures[future]
                try:
                    result = future.result(timeout=per_problem_timeout)
                except TimeoutError:
                    result = {
                        "problem_id": problem_id,
                        "candidates": public_results_by_problem.get(problem_id, []),
                        "private_results": [],
                        "private_stats": {},
                        "error": f"timeout after {per_problem_timeout}s",
                    }
                    logger.error(
                        f"[{problem_id}] Timed out after {per_problem_timeout}s"
                    )
                except Exception as e:
                    result = {
                        "problem_id": problem_id,
                        "candidates": public_results_by_problem.get(problem_id, []),
                        "private_results": [],
                        "private_stats": {},
                        "error": f"{type(e).__name__}: {e}",
                    }
                    logger.error(f"[{problem_id}] Worker failed: {e}")

                all_results.append(result)
                if result.get("error"):
                    failed += 1
                else:
                    completed += 1

                stats = result.get("private_stats", {})
                logger.info(
                    f"[{completed + failed}/{len(eval_problem_ids)}] {problem_id} done: "
                    f"accepted={stats.get('count_accepted', 0)}/{stats.get('count', 0)} "
                    f"mean_abs={stats.get('absolute_score', {}).get('mean', 0.0):.2f} "
                    f"(completed={completed}, failed={failed})"
                )

        _current_executor = None

    private_elapsed = time.time() - file_start
    logger.info(
        f"All problems finished in {private_elapsed:.1f}s "
        f"({private_elapsed / max(len(eval_problem_ids), 1):.1f}s per problem); "
        f"completed={completed}, failed={failed}"
    )

    selection_results = _derive_selection_results(all_results, selection_methods)

    outputs: dict[str, Any] = {}
    for sel in selection_methods:
        output = _combine_results(
            selection_results[sel], eval_problem_ids, training_problem_ids
        )
        output.update(
            {
                "model": saved.get("model", "unknown"),
                "lora_path": saved.get("lora_path", ""),
                "n_candidates": saved.get("n_candidates", 0),
                "lite_version": lite_version,
                "session_duration_hours": session_duration_hours,
                "ale_bench_num_workers": ale_bench_num_workers,
                "n_parallel_problems": n_parallel_problems,
                "selection_method": sel,
                "reprocessed_from": os.path.abspath(input_path),
            }
        )
        outputs[sel] = output

        avg_all = output.get("average_all", {})
        avg_oot = output.get("average_out_of_training", {})
        logger.info(
            f"Selection={sel}: "
            f"all_abs={avg_all.get('absolute_score', 0.0):.2f} "
            f"all_perf={avg_all.get('performance', 0.0):.2f} "
            f"oot_abs={avg_oot.get('absolute_score', 0.0):.2f} "
            f"oot_perf={avg_oot.get('performance', 0.0):.2f} "
            f"success={avg_all.get('count', 0)}/{len(eval_problem_ids)}"
        )

    if len(selection_methods) == 1:
        output = outputs[selection_methods[0]]
        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_standalone_reprocessed{ext}"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved reprocessed results to {output_path}")
        return output

    # Multiple selection methods: write each selection to its own file.
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_dir = f"{base}_standalone_reprocessed"
    else:
        output_dir = output_path
    os.makedirs(output_dir, exist_ok=True)

    for sel, output in outputs.items():
        sel_path = os.path.join(
            output_dir,
            os.path.basename(input_path).replace(
                ".json", f"_standalone_reprocessed_{sel}.json"
            ),
        )
        with open(sel_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved reprocessed results for {sel} to {sel_path}")

    return outputs


def _find_result_files(path: str) -> list[str]:
    """Return JSON result files to reprocess for a file or directory path."""
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise ValueError(f"Input path does not exist: {path}")

    files = []
    for entry in sorted(os.listdir(path)):
        if not entry.endswith(".json"):
            continue
        if "ale_bench_eval_results_" not in entry:
            continue
        if "_standalone_reprocessed" in entry:
            continue
        files.append(os.path.join(path, entry))
    return files


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Standalone re-run of ALE-Bench private eval from saved public results."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to a saved ale_bench_eval_results_*.json file or a directory "
        "containing such files.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output path (file or directory). If not given, a '_standalone_reprocessed' "
        "suffix is appended next to each input file.",
    )
    parser.add_argument(
        "--selection_method",
        "-s",
        default="median",
        choices=["median", "median_case_score", "best_public", "all"],
        help="Candidate selection method. 'median' matches the official "
        "ALE-Bench leaderboard protocol. 'best_public' selects the candidate "
        "with the best overall_absolute_score according to the problem's "
        "score_type (minimize/maximize). 'all' writes both 'median' and "
        "'best_public' result files.",
    )
    parser.add_argument(
        "--ale_bench_num_workers",
        "-w",
        type=int,
        default=8,
        help="Number of ALE-Bench workers per private-eval session.",
    )
    parser.add_argument(
        "--n_parallel_problems",
        "-p",
        type=int,
        default=8,
        help="Number of problems to evaluate concurrently. Set to 1 for sequential mode.",
    )
    parser.add_argument(
        "--session_duration_hours",
        type=float,
        default=4.0,
        help="Time budget passed to ale_bench.start.",
    )
    parser.add_argument(
        "--per_problem_timeout",
        type=float,
        default=600.0,
        help="Maximum seconds to wait for one problem's private eval before marking it timed out.",
    )
    args = parser.parse_args(argv)

    input_paths = _find_result_files(args.input)
    if not input_paths:
        logger.error(f"No ale_bench_eval_results_*.json files found in {args.input}")
        sys.exit(1)

    logger.info(f"Found {len(input_paths)} result file(s) to reprocess")

    output_is_dir = args.output is not None and os.path.isdir(args.output)
    if args.output is not None and len(input_paths) > 1 and not output_is_dir:
        os.makedirs(args.output, exist_ok=True)
        output_is_dir = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if args.selection_method == "all":
        selection_methods = ["median", "best_public"]
    else:
        selection_methods = [args.selection_method]

    for i, input_path in enumerate(input_paths, start=1):
        if _shutdown_requested:
            logger.warning("Shutdown requested, stopping file processing")
            break
        logger.info(f"[{i}/{len(input_paths)}] Reprocessing {input_path}")
        if output_is_dir:
            output_path = os.path.join(
                args.output,
                os.path.basename(input_path).replace(
                    ".json",
                    f"_standalone_reprocessed_{args.selection_method}.json",
                ),
            )
        else:
            output_path = args.output

        reprocess_single_file(
            input_path=input_path,
            output_path=output_path,
            selection_methods=selection_methods,
            ale_bench_num_workers=args.ale_bench_num_workers,
            n_parallel_problems=args.n_parallel_problems,
            session_duration_hours=args.session_duration_hours,
            per_problem_timeout=args.per_problem_timeout,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
