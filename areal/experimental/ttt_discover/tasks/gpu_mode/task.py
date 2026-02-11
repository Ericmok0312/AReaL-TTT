from gpu_mode.libkernelbot.submission import compute_score
from gpu_mode.run_modal import run_on_modal


def get_gpu_mode_error(msg):
    return {
        "score": 0.0,
        "msg": msg,
        "correctness": 0.0,
        "performance": -1_000_000,
    }


async def run_gpu_mode_task(generation: str, gpu_type: str, task_name: str, score_scale: float, app_name: str):

    # Run modal
    result, task = await run_on_modal(
        submission_code=generation,
        gpu_type=gpu_type,
        mode="leaderboard",
        task_name=task_name,
        app_name=app_name,
    )

    if not result.success:
        return get_gpu_mode_error(f"Error: Failed to run test: {result.error}.")
    
    # Unexpected
    if "test" not in result.runs:
        return get_gpu_mode_error(f"Unexpected result: Failed to find test results.")

    test_results = result.runs["test"]

    # Probably compile error
    if not test_results.run.success:
        return get_gpu_mode_error(f"Failed to run tests: {test_results.run.stderr}")

    # Failed test cases
    if not test_results.run.passed:
        return get_gpu_mode_error(f"Failed to pass test cases.")

    if task is not None and "leaderboard" in result.runs:
        try:
            score_seconds = compute_score(result, task, submission_id=-1)
            score_us = score_seconds * 1_000_000
            msg = f"\nOverall leaderboard score (microseconds, {task.ranking_by.value}): {score_us} us"
        except Exception as e:
            return get_gpu_mode_error(f"Could not compute leaderboard score: {e}")

    score = score_scale / score_us

    return {
        "score": score,
        "msg": msg,
        "correctness": 1.0,
        "performance": -score_us,
    }
