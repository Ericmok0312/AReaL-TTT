#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Re-run ALE-Bench private evaluation from saved public results.

This script loads a previously saved ``ale_bench_eval_results_*.json`` file
(produced by ``eval_tttd_multi_ale_bench.py``), re-selects the best candidate
per problem using the official ALE-Bench median selection, re-runs only the
private evaluation, and writes a new result file.

No LLM generation is performed; only the CPU/Docker private-eval phase is
re-executed.

Usage:
    python areal/experimental/ttt_discover/examples/reprocess_ale_bench_private_eval.py \
        --input path/to/ale_bench_eval_results_baseline_step_000000.json \
        --selection_method median \
        --n_parallel_problems 8 \
        --ale_bench_num_workers 8

Or reprocess every result file in a directory:

    python areal/experimental/ttt_discover/examples/reprocess_ale_bench_private_eval.py \
        --input path/to/ale_bench_eval/ \
        --selection_method median \
        --n_parallel_problems 8
"""

import argparse
import json
import os
import sys
import time
from typing import Any

from areal.experimental.ttt_discover.ale_bench_eval import (
    _get_ale_bench_session,
    combine_ale_bench_results,
    evaluate_problem_subset_with_public_scores,
)
from areal.utils import logging

logger = logging.getLogger("reprocess_ale_bench_private_eval")


def _extract_public_results_by_problem(
    saved: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Build the public-results dict expected by private eval from a saved JSON."""
    public_results_by_problem: dict[str, list[dict[str, Any]]] = {}
    for result in saved.get("results", []):
        problem_id = result.get("problem_id")
        if not problem_id:
            continue
        candidates = result.get("candidates", [])
        # Ensure every candidate has the public block needed for selection.
        public_results_by_problem[problem_id] = [
            {
                "idx": c.get("idx", i),
                "code": c.get("code", ""),
                "public": c.get("public", {}),
            }
            for i, c in enumerate(candidates)
        ]
    return public_results_by_problem


def reprocess_single_file(
    input_path: str,
    output_path: str | None,
    selection_method: str,
    ale_bench_num_workers: int,
    n_parallel_problems: int,
    session_duration_hours: float = 4.0,
) -> dict[str, Any]:
    """Re-run private eval for one saved result file.

    Args:
        input_path: Path to the saved ``ale_bench_eval_results_*.json``.
        output_path: Where to write the new JSON. If None, a ``_reprocessed``
            suffix is appended to the input file name.
        selection_method: ``"median"`` for official median-of-overall-score
            selection, or ``"median_case_score"`` for legacy selection.
        ale_bench_num_workers: Workers passed to ``ale_bench.start`` per
            private-eval session.
        n_parallel_problems: Number of problems to run concurrently.
        session_duration_hours: Time budget for ALE-Bench sessions.

    Returns:
        The new combined output dictionary.
    """
    file_start = time.time()
    logger.info(f"Loading saved results from {input_path}")
    with open(input_path) as f:
        saved = json.load(f)

    problem_ids = saved.get("problem_ids", [])
    if not problem_ids:
        # Older files stored results without the ordered problem_ids list;
        # fall back to the order in which results appear.
        problem_ids = [
            r.get("problem_id") for r in saved.get("results", []) if r.get("problem_id")
        ]

    training_problem_ids = set(saved.get("training_problem_ids", []))
    public_results_by_problem = _extract_public_results_by_problem(saved)

    # Only reprocess problems for which we have candidate public scores.
    eval_problem_ids = [pid for pid in problem_ids if pid in public_results_by_problem]
    total_candidates = sum(len(v) for v in public_results_by_problem.values())
    if len(eval_problem_ids) != len(problem_ids):
        logger.warning(
            f"Dropping {len(problem_ids) - len(eval_problem_ids)} problems "
            f"with no saved candidate public scores"
        )

    lite_version = bool(saved.get("lite_version", False))

    logger.info(
        f"Loaded {len(eval_problem_ids)} problems, {total_candidates} total candidates"
    )
    logger.info(
        f"Re-running private eval: selection={selection_method}, "
        f"n_parallel={n_parallel_problems}, workers={ale_bench_num_workers}"
    )

    # Pre-build sessions in the main process so compilation logs are visible
    # and workers do not fight over Docker/toolchain resources.
    logger.info(
        f"Pre-building ALE-Bench sessions for {len(eval_problem_ids)} problems..."
    )
    problem_sessions: dict[str, Any] = {}
    session_build_start = time.time()
    for i, problem_id in enumerate(eval_problem_ids, start=1):
        logger.info(
            f"[{i}/{len(eval_problem_ids)}] Building ALE-Bench session for {problem_id}"
        )
        try:
            problem_sessions[problem_id] = _get_ale_bench_session(
                problem_id=problem_id,
                lite_version=lite_version,
                session_duration_hours=session_duration_hours,
                ale_bench_num_workers=ale_bench_num_workers,
            )
        except Exception:
            logger.exception(
                f"Failed to build session for {problem_id}; continuing without it"
            )
    logger.info(
        f"Built {len(problem_sessions)}/{len(eval_problem_ids)} sessions in "
        f"{time.time() - session_build_start:.1f}s"
    )

    private_start = time.time()
    local_results = evaluate_problem_subset_with_public_scores(
        eval_problem_ids,
        public_results_by_problem,
        lite_version=lite_version,
        session_duration_hours=session_duration_hours,
        ale_bench_num_workers=ale_bench_num_workers,
        n_parallel_problems=n_parallel_problems,
        problem_sessions=problem_sessions,
        selection_method=selection_method,
    )
    private_elapsed = time.time() - private_start
    logger.info(
        f"Private eval finished in {private_elapsed:.1f}s "
        f"({private_elapsed / max(len(eval_problem_ids), 1):.1f}s per problem)"
    )

    # Log the selected candidate per problem so the user can verify the new
    # selection method took effect.
    for r in local_results:
        problem_id = r.get("problem_id", "unknown")
        best_idx = r.get("best_candidate_idx")
        private = r.get("private", {})
        if best_idx is not None:
            logger.info(
                f"[{problem_id}] selected candidate {best_idx}, "
                f"private_abs={private.get('absolute_score', 0.0):.2f} "
                f"rank={private.get('rank', -1)} perf={private.get('performance', -1)}"
            )
        elif r.get("error"):
            logger.warning(f"[{problem_id}] private eval failed: {r['error']}")

    output = combine_ale_bench_results(
        local_results, eval_problem_ids, training_problem_ids
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
            "selection_method": selection_method,
            "reprocessed_from": os.path.abspath(input_path),
        }
    )

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_reprocessed{ext}"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    avg_all = output.get("average_all", {})
    avg_oot = output.get("average_out_of_training", {})
    total_elapsed = time.time() - file_start
    logger.info(f"Saved reprocessed results to {output_path}")
    logger.info(
        f"all_abs={avg_all.get('absolute_score', 0.0):.2f} "
        f"all_perf={avg_all.get('performance', 0.0):.2f} "
        f"oot_abs={avg_oot.get('absolute_score', 0.0):.2f} "
        f"oot_perf={avg_oot.get('performance', 0.0):.2f} "
        f"success={avg_all.get('count', 0)}/{len(eval_problem_ids)} "
        f"total_time={total_elapsed:.1f}s"
    )
    return output


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
        # Skip previously reprocessed files to avoid re-reprocessing.
        if "_reprocessed" in entry:
            continue
        files.append(os.path.join(path, entry))
    return files


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Re-run ALE-Bench private eval from saved public results."
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
        help="Output path (file or directory). If not given, a '_reprocessed' "
        "suffix is appended next to each input file.",
    )
    parser.add_argument(
        "--selection_method",
        "-s",
        default="median",
        choices=["median", "median_case_score"],
        help="Candidate selection method. 'median' matches the official "
        "ALE-Bench leaderboard protocol.",
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
        help="Number of problems to evaluate concurrently.",
    )
    parser.add_argument(
        "--session_duration_hours",
        type=float,
        default=4.0,
        help="Time budget passed to ale_bench.start.",
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

    for i, input_path in enumerate(input_paths, start=1):
        logger.info(f"[{i}/{len(input_paths)}] Reprocessing {input_path}")
        if output_is_dir:
            output_path = os.path.join(
                args.output,
                os.path.basename(input_path).replace(".json", "_reprocessed.json"),
            )
        else:
            output_path = args.output

        reprocess_single_file(
            input_path=input_path,
            output_path=output_path,
            selection_method=args.selection_method,
            ale_bench_num_workers=args.ale_bench_num_workers,
            n_parallel_problems=args.n_parallel_problems,
            session_duration_hours=args.session_duration_hours,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
