#!/usr/bin/env python3
"""Pre-download ALE-Bench problem data so eval doesn't hang on first use.

Usage:
    export ALE_BENCH_DATA=/tmp/data/ALE-Bench
    ./predownload_ale_bench.py

If ALE_BENCH_DATA is not set, the script will clone the HuggingFace dataset
into ~/.cache/areal/ale-bench and use it automatically.
"""

import os
import subprocess
from datetime import timedelta
from pathlib import Path


def ensure_ale_bench_data() -> str:
    """Return path to local ALE-Bench data, cloning from HF if needed."""
    data_dir = os.environ.get("ALE_BENCH_DATA")
    if data_dir and Path(data_dir).exists():
        print(f"Using existing ALE_BENCH_DATA={data_dir}")
        return data_dir

    target = Path.home() / ".cache" / "areal" / "ale-bench"
    if target.exists():
        print(f"Using existing local clone: {target}")
        return str(target)

    print("ALE_BENCH_DATA not set. Cloning from HuggingFace...")
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "lfs", "install"],
        check=True,
        cwd=str(target.parent),
    )
    subprocess.run(
        ["git", "clone", "https://huggingface.co/datasets/SakanaAI/ALE-Bench", str(target)],
        check=True,
        cwd=str(target.parent),
    )
    print(f"Clone complete: {target}")
    return str(target)


def predownload(problem_ids: list[str], data_dir: str, lite_version: bool = False):
    # Import inside function so ALE_BENCH_DATA is set first.
    from ale_bench import start

    for i, problem_id in enumerate(problem_ids):
        print(f"[{i+1}/{len(problem_ids)}] Pre-loading {problem_id} ...")
        try:
            start(
                problem_id=problem_id,
                lite_version=lite_version,
                use_same_time_scale=False,
                session_duration=timedelta(hours=4.0),
                num_workers=1,
                run_visualization_server=False,
            )
            print(f"  -> {problem_id} OK")
        except Exception as e:
            print(f"  -> {problem_id} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    from ale_bench import list_problem_ids

    data_dir = ensure_ale_bench_data()
    os.environ["ALE_BENCH_DATA"] = data_dir

    problem_ids = list_problem_ids(lite_version=False)
    print(f"Found {len(problem_ids)} problems")
    predownload(problem_ids, data_dir, lite_version=False)
