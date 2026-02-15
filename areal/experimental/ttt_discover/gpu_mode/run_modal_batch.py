#!/usr/bin/env python3
"""Batch-run many submissions on Modal.

This is similar to `src/run_trimul_modal.py`, but:
- Takes a directory containing `submission_1.py`, `submission_2.py`, ...
- Runs them in batches
- Each batch is submitted in parallel using a process pool
- Writes each submission's full printed output to `output_dir/test_<i>.out`

Supported tasks:
  - trimul: BioML TriMul task (bioml/trimul/task.yml)
  - nvfp4_gemm: NVIDIA FP4 GEMM task (nvidia/nvfp4_gemm/task.yml)

Usage (from project root):
    uv run python src/run_trimul_modal_batch.py \
        --submissions-dir src/bioml/trimul/submission_a100 \
        --task trimul \
        --gpu A100 \
        --mode leaderboard \
        --batch-size 8 \
        --workers 8 \
        --output-dir src/results/A100/batch

You must:
  1. Be authenticated with Modal (`modal token new`)
  2. Have deployed the Modal app:
         cd src/runners
         modal deploy modal_runner_archs.py
"""

import argparse
import asyncio
import contextlib
import io
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from run_modal import print_result, run_on_modal


_SUBMISSION_RE = re.compile(r"^submission_(\\d+)\\.py$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run many submissions on Modal.",
    )
    parser.add_argument(
        "--submissions-dir",
        required=True,
        type=Path,
        help="Directory containing submission_*.py files.",
    )
    parser.add_argument(
        "--task",
        "-t",
        default="trimul",
        choices=["trimul", "nvfp4_gemm"],
        help="Task to run (default: trimul).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write outputs (default: <submissions-dir>/outputs)",
    )
    parser.add_argument(
        "--pattern",
        default="*.py",
        help="Glob pattern inside submissions-dir (default: submission_*.py)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="How many submissions per batch.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Process pool size (per batch).",
    )
    parser.add_argument(
        "--gpu",
        "-g",
        default="H100",
        help="Modal GPU type (e.g. T4, L4, A100, H100, B200, L4x4).",
    )
    parser.add_argument(
        "--mode",
        "-m",
        default="leaderboard",
        help="Submission mode (test, benchmark, leaderboard, profile, private).",
    )
    parser.add_argument(
        "--app-name",
        "-a",
        default="discord-bot-runner",
        choices=["discord-bot-runner", "discord-bot-runner-nvfp4"],
        help="Modal app name to use (default: discord-bot-runner).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip outputs that already exist.",
    )
    return parser.parse_args()


def _submission_id_from_path(path: Path) -> Optional[int]:
    m = _SUBMISSION_RE.match(path.name)
    if not m:
        return None
    return int(m.group(1))


def _iter_submission_files(submissions_dir: Path, pattern: str) -> list[Path]:
    files = [p for p in submissions_dir.glob(pattern) if p.is_file()]

    def key(p: Path):
        sid = _submission_id_from_path(p)
        return (0, sid) if sid is not None else (1, p.name)

    return sorted(files, key=key)


def _chunked(items: list[Path], batch_size: int) -> list[list[Path]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _run_one_submission_worker(
    submission_path_str: str,
    gpu: str,
    mode: str,
    task_name: str,
    app_name: str,
) -> tuple[str, str]:
    """Worker entrypoint. Must be top-level for ProcessPool pickling."""
    # Resolve to absolute path to ensure it works regardless of worker's working directory
    submission_path = Path(submission_path_str).resolve()

    buf = io.StringIO()
    started = time.time()
    with contextlib.redirect_stdout(buf):
        print(f"submission: {submission_path}")
        print(f"task: {task_name}")
        print(f"gpu: {gpu}")
        print(f"mode: {mode}")
        print(f"app-name: {app_name}")
        print("-")

        submission_code = submission_path.read_text()

        # Run on Modal (async API) and print result using existing helpers.
        result, task = asyncio.run(
            run_on_modal(
                submission_code=submission_code,
                gpu_type=gpu,
                mode=mode,
                task_name=task_name,
                app_name=app_name,
            )
        )
        print_result(result, task)

        elapsed = time.time() - started
        print("\n-")
        print(f"wall_time_seconds: {elapsed:.3f}")
    
    return submission_path_str, buf.getvalue()


def main() -> int:
    args = _parse_args()

    submissions_dir: Path = args.submissions_dir.resolve()
    if not submissions_dir.exists() or not submissions_dir.is_dir():
        raise FileNotFoundError(f"submissions-dir does not exist or is not a directory: {submissions_dir}")

    output_dir: Path = args.output_dir or (submissions_dir / "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_submission_files(submissions_dir, args.pattern)
    if not files:
        raise FileNotFoundError(f"No files matched {args.pattern} in {submissions_dir}")

    batches = _chunked(files, args.batch_size)
    print(f"Found {len(files)} submissions in {submissions_dir}")
    print(f"Writing outputs to {output_dir}")
    print(f"Batches: {len(batches)} (batch_size={args.batch_size}, workers={args.workers})")

    for batch_idx, batch in enumerate(batches, start=1):
        print(f"\n=== Batch {batch_idx}/{len(batches)} ({len(batch)} submissions) ===")

        # Filter skipped outputs.
        work: list[Path] = []
        for p in batch:
            sid = _submission_id_from_path(p)
            if sid is None:
                # Fall back to a stable, readable filename.
                out_path = output_dir / f"test_{p.stem}.out"
            else:
                out_path = output_dir / f"test_{sid}.out"

            if args.skip_existing and out_path.exists():
                print(f"skip existing: {out_path}")
                continue
            work.append(p)

        if not work:
            print("(nothing to do in this batch)")
            continue

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_run_one_submission_worker, str(p.resolve()), args.gpu, args.mode, args.task, args.app_name): p for p in work
            }

            for fut in as_completed(futures):
                submission_path = futures[fut]
                sid = _submission_id_from_path(submission_path)
                if sid is None:
                    out_path = output_dir / f"test_{submission_path.stem}.out"
                else:
                    out_path = output_dir / f"test_{sid}.out"

                try:
                    _, output_text = fut.result()
                except BaseException as e:
                    output_text = f"Worker failed for {submission_path}:\n{e!r}\n"

                out_path.write_text(output_text)
                print(f"wrote: {out_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
