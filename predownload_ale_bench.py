#!/usr/bin/env python3
"""Pre-download ALE-Bench problem data so eval doesn't hang on first use."""

from datetime import timedelta
from ale_bench import start, list_problem_ids


def predownload(problem_ids: list[str], lite_version: bool = False):
    for i, problem_id in enumerate(problem_ids):
        print(f"[{i+1}/{len(problem_ids)}] Pre-downloading {problem_id} ...")
        try:
            session = start(
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
    # 全量 40 题
    problem_ids = list_problem_ids(lite_version=False)
    print(f"Found {len(problem_ids)} problems")
    predownload(problem_ids, lite_version=False)

    # 如果要同时下载 lite 版本，取消下面注释
    # lite_ids = list_problem_ids(lite_version=True)
    # predownload(lite_ids, lite_version=True)
