
import asyncio
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path

import ray

import ale_bench.constants
from ale_bench.code_language import CodeLanguage, JudgeVersion
from ale_bench.data import load_local_problem
from ale_bench.result import JudgeResult
from ale_bench.tool_wrappers.case_runner import run_cases
from ale_bench.utils import get_cache_dir


def get_ale_bench_error(msg):
    return {
        "score": 0.0,
        "msg": msg,
        "correctness": 0.0,
        "performance": 0.0,
    }


def load_cached_public_inputs(problem_id: str, lite_version: bool = True) -> list[str]:
    """Load cached public inputs for a problem.
    
    Args:
        problem_id: Problem ID
        lite_version: Preferred lite version (will fallback to other version if not found)
        
    Returns:
        List of input strings
        
    Raises:
        FileNotFoundError: If cached inputs not found
    """
    cache_dir = get_cache_dir() / "public_inputs_150"
    
    # Find all cache files for this problem
    cache_files = list(cache_dir.glob(f"{problem_id}_*.json"))
    if not cache_files:
        raise FileNotFoundError(
            f"No cached public inputs found for {problem_id}. "
            f"Run cache_public_inputs.py first. Cache dir: {cache_dir}"
        )
    
    # Try to find a cache file matching the requested lite_version
    matching_file = None
    fallback_file = None
    
    for cache_file in cache_files:
        with open(cache_file, "r") as f:
            cached_data = json.load(f)
        
        # Verify problem_id matches
        if cached_data.get("problem_id") != problem_id:
            continue
        
        # Check if lite_version matches
        if cached_data.get("lite_version") == lite_version:
            matching_file = (cache_file, cached_data)
            break
        else:
            # Keep track of the first file with opposite lite_version as fallback
            if fallback_file is None:
                fallback_file = (cache_file, cached_data)
    
    # Use matching file if found, otherwise use fallback
    if matching_file:
        cache_file, cached_data = matching_file
        print(f"Using cached inputs with lite_version={lite_version} from {cache_file}")
    elif fallback_file:
        cache_file, cached_data = fallback_file
        actual_lite_version = cached_data.get("lite_version")
        print(f"Warning: Requested lite_version={lite_version} not found, using lite_version={actual_lite_version} from {cache_file}")
    else:
        raise ValueError(
            f"No valid cache file found for {problem_id}. "
            f"Found files but none matched problem_id. Cache dir: {cache_dir}"
        )
    
    return cached_data["inputs"]


def setup_cached_tester_dir(problem_id: str, base_dir: Path) -> Path:
    """Return the path to the cached tester binary on NFS.
    
    The tester binary is already on NFS in the cache directory, so we just
    return the cache directory. The volume functions will use the tester path directly.
    
    IMPORTANT: This returns the CACHE directory itself, NOT a temporary directory.
    The cache directory should NEVER be deleted or cleaned up.
    
    Args:
        problem_id: Problem ID
        base_dir: Base directory (unused, kept for compatibility)
        
    Returns:
        Path to cache directory containing tester binaries
        
    Raises:
        FileNotFoundError: If cached tester not found
    """
    from ale_bench.utils import get_cache_dir
    
    cache_dir = get_cache_dir() / "tester_binaries"
    tester_cache_file = cache_dir / f"{problem_id}_tester"
    
    # CRITICAL: Verify cache directory exists and is a directory (not deleted)
    if not cache_dir.exists():
        raise RuntimeError(
            f"CRITICAL: Cache directory {cache_dir} does not exist! "
            f"This may indicate the cache was deleted or the path is incorrect. "
            f"Please run cache_public_inputs.py to recreate it, or check that ALE_BENCH_CACHE is set correctly."
        )
    if not cache_dir.is_dir():
        raise RuntimeError(
            f"CRITICAL: Cache directory {cache_dir} exists but is not a directory! "
            f"This is unexpected and may indicate filesystem issues."
        )
    
    # Verify tester file exists
    if not tester_cache_file.exists():
        raise FileNotFoundError(
            f"No cached tester binary found for {problem_id}. "
            f"Run cache_public_inputs.py first. Expected: {tester_cache_file}. "
            f"Cache directory exists: {cache_dir.exists()}"
        )
    
    # Just return the cache directory - the volume functions will use the tester path directly
    # WARNING: This is the CACHE directory - it should NEVER be deleted!
    return cache_dir


def run_cases_remote(
    code_path: str,
    problem_data_path: str,
    tool_dir_path: str,
    problem_id: str,
    lite_version: bool,
    results_path: str,
):
    """Remote function to run cases using NFS for data transfer.
    
    Args:
        code_path: Path to file containing code string
        problem_data_path: Path to file containing pickled problem data dict
        tool_dir_path: Path to tool directory (on NFS)
        problem_id: Problem ID for loading cached inputs
        lite_version: Whether using lite version for loading cached inputs
        results_path: Path where results will be written (pickled list of CaseResult)
    
    Returns:
        Path to results file
    """
    import json
    import pickle
    from pathlib import Path
    from ale_bench.code_language import CodeLanguage, JudgeVersion
    from ale_bench.tool_wrappers.case_runner import run_cases
    from ale_bench.utils import get_cache_dir
    
    # Read code from file
    with open(code_path, "r") as f:
        code = f.read()
    
    # Load cached public inputs directly on the ray worker
    # (duplicated logic from load_cached_public_inputs to avoid circular import)
    cache_dir = get_cache_dir() / "public_inputs_150"
    cache_files = list(cache_dir.glob(f"{problem_id}_*.json"))
    if not cache_files:
        raise FileNotFoundError(
            f"No cached public inputs found for {problem_id}. "
            f"Run cache_public_inputs.py first. Cache dir: {cache_dir}"
        )
    
    matching_file = None
    fallback_file = None
    for cache_file in cache_files:
        with open(cache_file, "r") as f:
            cached_data = json.load(f)
        if cached_data.get("problem_id") != problem_id:
            continue
        if cached_data.get("lite_version") == lite_version:
            matching_file = (cache_file, cached_data)
            break
        else:
            if fallback_file is None:
                fallback_file = (cache_file, cached_data)
    
    if matching_file:
        cache_file, cached_data = matching_file
    elif fallback_file:
        cache_file, cached_data = fallback_file
    else:
        raise ValueError(
            f"No valid cache file found for {problem_id}. "
            f"Found files but none matched problem_id. Cache dir: {cache_dir}"
        )
    
    inputs = cached_data["inputs"]
    
    # Read problem data from file
    with open(problem_data_path, "rb") as f:
        problem_data = pickle.load(f)
    
    # Extract problem data
    time_limit = problem_data["time_limit"]
    memory_limit = problem_data["memory_limit"]
    problem_type = problem_data["problem_type"]
    code_lang = problem_data["code_language"]
    judge_version = problem_data["judge_version"]
    
    # Convert enums
    code_lang_enum = CodeLanguage(code_lang) if isinstance(code_lang, str) else code_lang
    judge_version_enum = JudgeVersion(judge_version) if isinstance(judge_version, str) else judge_version
    
    # Run cases
    # Ensure tool_dir_path is absolute
    tool_dir_abs = Path(tool_dir_path).resolve()
    # Get base_dir from results_path parent (should be on NFS)
    base_dir = Path(results_path).parent
    case_results = run_cases(
        inputs=inputs,
        code=code,
        code_language=code_lang_enum,
        judge_version=judge_version_enum,
        time_limit=time_limit,
        memory_limit=memory_limit,
        problem_id=problem_id,
        problem_type=problem_type,
        tool_dir=tool_dir_abs,
        return_details=False,
        skip_local_visualization=True,
        num_workers=150,
        base_dir=base_dir,  # Use NFS directory for temp files
    )
    
    # Write results to file
    with open(results_path, "wb") as f:
        pickle.dump(case_results, f)
    
    return results_path


# Initialize ray remote function (module-level)
_exec_fn = None

def _get_exec_fn(num_cpus: int = 2):
    """Get or create the ray remote execution function."""
    global _exec_fn
    if _exec_fn is None:
        _exec_fn = ray.remote(num_cpus=num_cpus, max_calls=1)(run_cases_remote)
    return _exec_fn


def run_ale_bench_task(generation: str, problem_id: str | None = None, lite_version: bool = True, log_dir: str | None = None, num_cpus_per_task: int = 2) -> dict:
    """Evaluate a code sample on public test cases using cached inputs and tester binaries.
    
    Args:
        generation: Code string to evaluate
        problem_id: Problem ID (if None, tries to get from ALE_BENCH_PROBLEM_ID env var)
        lite_version: Whether using lite version (default: True)
        
    Returns:
        Dictionary with score, msg, correctness, and performance
    """
    # Get problem_id from parameter or environment variable
    if problem_id is None:
        problem_id = os.environ.get("ALE_BENCH_PROBLEM_ID")
        if problem_id is None:
            return get_ale_bench_error("problem_id must be provided or set ALE_BENCH_PROBLEM_ID environment variable")
    
    # Validate code
    if not generation or not generation.strip():
        return get_ale_bench_error("Invalid code: empty or missing")
    
    # Basic validation: check for main function in C++ code
    code_lower = generation.lower()
    if "int main" not in code_lower and "void main" not in code_lower:
        return get_ale_bench_error("Invalid code: missing main function")
    
    # IMPORTANT: tool_dir is the CACHE directory (get_cache_dir() / "tester_binaries")
    # It should NEVER be deleted or cleaned up. It's not a temporary directory.
    tool_dir = None  # This is the cache directory, NOT a temp directory - do NOT delete it
    code_path = None
    problem_data_path = None
    results_path = None
    
    try:
        # Set up NFS directory for file-based data transfer
        # log_dir must be provided and must be on NFS for Ray workers to access files
        if log_dir is None:
            # Try to get from environment variable as fallback
            log_dir = os.environ.get("ALE_BENCH_LOG_DIR")
            if log_dir is None:
                raise ValueError(
                    "log_dir must be provided or set ALE_BENCH_LOG_DIR environment variable. "
                    "This must be a path on NFS that is accessible from all Ray workers."
                )
        tmp_dir = Path(log_dir) / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        
        # CRITICAL SAFETY CHECK: Ensure tmp_dir is NOT the cache directory
        # This prevents accidental deletion of cache if log_dir is misconfigured
        from ale_bench.utils import get_cache_dir
        cache_dir = get_cache_dir()
        tmp_dir_resolved = tmp_dir.resolve()
        cache_dir_resolved = cache_dir.resolve()
        if cache_dir_resolved in tmp_dir_resolved.parents or tmp_dir_resolved == cache_dir_resolved:
            raise RuntimeError(
                f"CRITICAL: tmp_dir ({tmp_dir_resolved}) is inside or equal to cache directory ({cache_dir_resolved})! "
                f"This would cause the cache to be deleted. Please use a different log_dir that is not in the cache directory."
            )

        # Load problem data from local problems directory
        problem, seeds, standings, rank_performance_map, data_root = load_local_problem(
            problem_id, lite_version
        )
        
        # Set up cached tester binary (on NFS so Ray workers can access it)
        # IMPORTANT: tool_dir is the CACHE directory, NOT a temporary directory.
        # It points to get_cache_dir() / "tester_binaries" and should NEVER be deleted.
        tool_dir = setup_cached_tester_dir(problem_id, tmp_dir)
        # Safety assertion: verify tool_dir is actually the cache directory
        from ale_bench.utils import get_cache_dir
        expected_cache_dir = (get_cache_dir() / "tester_binaries").resolve()
        tool_dir_resolved = Path(tool_dir).resolve()
        if tool_dir_resolved != expected_cache_dir:
            raise RuntimeError(
                f"CRITICAL: tool_dir ({tool_dir_resolved}) is not the expected cache directory ({expected_cache_dir}). "
                f"This would cause the cache to be deleted! Aborting to prevent data loss."
            )
        
        # Convert code_language string to enum (default: cpp20)
        code_lang = CodeLanguage.CPP20
        judge_version = JudgeVersion.V202301  # Default judge version
        
        # Write code to NFS file
        with tempfile.NamedTemporaryFile(
            suffix=".cpp",
            delete=False,
            mode="w",
            dir=str(tmp_dir),
        ) as f:
            code_path = f.name
            f.write(generation)
        
        # Write problem data to NFS file (pickled)
        problem_data = {
            "time_limit": problem.constraints.time_limit,
            "memory_limit": problem.constraints.memory_limit,
            "problem_id": problem_id,
            "problem_type": problem.metadata.problem_type,
            "code_language": code_lang.value,  # CodeLanguage is a string enum
            "judge_version": judge_version.value,  # JudgeVersion is a string enum
        }
        with tempfile.NamedTemporaryFile(
            suffix=".pkl",
            delete=False,
            mode="wb",
            dir=str(tmp_dir),
        ) as f:
            problem_data_path = f.name
            pickle.dump(problem_data, f)
        
        # Compute expected results path
        results_path = str(Path(code_path).with_suffix(".results.pkl"))
        
        # Launch remote execution
        exec_fn = _get_exec_fn(num_cpus_per_task)
        result_path_future = exec_fn.options(scheduling_strategy="SPREAD").remote(
            code_path,
            problem_data_path,
            str(tool_dir),
            problem_id,
            lite_version,
            results_path,
        )
        
        # Wait for results (no timeout to allow for scheduling delays)
        returned_results_path = ray.get(result_path_future)
        
        if not returned_results_path or not os.path.exists(returned_results_path):
            raise RuntimeError(f"Results file does not exist: {returned_results_path}")
        
        # Load results from NFS file
        with open(returned_results_path, "rb") as f:
            case_results = pickle.load(f)
        
        # Calculate statistics (matching evaluate_public_cached_jsonl.py style)
        num_public_cases = len(case_results)
        num_accepted = sum(
            1 for case in case_results 
            if case.judge_result == JudgeResult.ACCEPTED
        )
        
        # Calculate total score and mean score (matching evaluate_public_cached_jsonl.py)
        total_score = sum(case_result.absolute_score for case_result in case_results)
        mean_score = total_score / num_public_cases if num_public_cases > 0 else 0.0
        
        # Check if all test cases passed (all ACCEPTED)
        all_passed = num_accepted == num_public_cases
        
        # Determine correctness: 1.0 if all passed, 0.0 otherwise
        correctness = 1.0 if all_passed else 0.0
        
        # Performance is the mean score (matching evaluate_public_cached_jsonl.py)
        performance = mean_score
        
        # Score is the same as performance
        score = performance / 1500.0

        if problem_id == "ahc058":
            score = score / 2000.0 # Different score range for AHC058
        
        # Build message
        msg = f"Evaluated on {num_public_cases} public test cases. Passed: {num_accepted}/{num_public_cases}. Performance: {performance:.4f}"
        
        return {
            "score": score,  # Use performance as the score (matching evaluate_private_cached.py)
            "msg": msg,
            "correctness": correctness,
            "performance": performance,
        }
        
    except FileNotFoundError as e:
        return get_ale_bench_error(f"Cache error: {str(e)}")
    except ray.exceptions.GetTimeoutError:
        return get_ale_bench_error("Evaluation timed out")
    except Exception as e:
        error_msg = str(e)
        # Check if it's a compilation error
        is_compile_error = any(keyword in error_msg.lower() for keyword in [
            "undefined reference to `main'",
            "undefined reference to `main\"",
            "no such file or directory",
            "compilation error",
            "compile",
        ])
        if is_compile_error:
            return get_ale_bench_error(f"Compilation error: {error_msg}")
        else:
            return get_ale_bench_error(f"Evaluation error: {error_msg}")
    finally:
        # Cleanup NFS files (but NEVER delete cache directories or files)
        # Safety check: ensure we're not accidentally deleting cache files
        from ale_bench.utils import get_cache_dir
        cache_dir = get_cache_dir()
        
        for file_path in [code_path, problem_data_path, results_path]:
            if file_path is not None:
                try:
                    file_path_obj = Path(file_path)
                    if file_path_obj.exists():
                        # Safety check: don't delete anything in cache directories
                        file_path_resolved = file_path_obj.resolve()
                        cache_dir_resolved = cache_dir.resolve()
                        # If file is inside cache_dir or any cache subdirectory, skip deletion
                        if cache_dir_resolved in file_path_resolved.parents:
                            continue  # Skip deletion of cache files
                        # Also check specific cache subdirectories
                        for cache_subdir in ["tester_binaries", "public_inputs_150"]:
                            cache_subdir_path = (cache_dir / cache_subdir).resolve()
                            if cache_subdir_path.exists() and cache_subdir_path in file_path_resolved.parents:
                                continue  # Skip deletion of cache files
                        os.unlink(file_path)
                except (FileNotFoundError, OSError):
                    pass
