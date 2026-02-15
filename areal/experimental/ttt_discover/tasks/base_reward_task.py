"""
Base reward task for TTT-Discover.

Uses AReaL standard subprocess-based code execution.
No external discover dependencies.
"""

import subprocess
import sys
import pickle
import tempfile
import os
import time
import signal
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from contextlib import nullcontext
from enum import Enum
from typing import Any, List

import numpy as np

from areal.utils import logging

logger = logging.getLogger("BaseRewardTask")


def _timer(name: str, metrics_dict: dict):
    """Simple timing context manager."""
    class TimerContext:
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        def __exit__(self, *args):
            elapsed = time.perf_counter() - self.start
            metrics_dict[name] = elapsed
    return TimerContext()


class RewardType(str, Enum):
    LINEAR = "linear"
    NEG_LINEAR = "neg_linear"
    EXP_CF = "exp_cf"
    RECIPROCAL_CF = "reciprocal_cf"
    SCALED_RECIPROCAL_CF = "scaled_reciprocal_cf"


def run_with_timeout(program_path: str, function_name: str, timeout_seconds: int = 20, num_cpus: int = 1):
    """
    Run the target program file in a separate Python process with a strict timeout.
    
    Uses AReaL standard subprocess approach
    """
    # Create the injected runner script
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as temp_file:
        injected = r'''
import sys
import os
import pickle
import traceback
import importlib.util as _il

# Thread caps for BLAS libs
os.environ.setdefault("OMP_NUM_THREADS", "__MAX_CPUS__")
os.environ.setdefault("MKL_NUM_THREADS", "__MAX_CPUS__")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "__MAX_CPUS__")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "__MAX_CPUS__")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "__MAX_CPUS__")
os.environ.setdefault("BLIS_NUM_THREADS", "__MAX_CPUS__")

# ---------- Add the target module directory to sys.path ----------
_target_program_path = "__PROGRAM_PATH__"
_target_function_name = "__FUNCTION_NAME__"
_results_path = "__RESULTS_PATH__"

sys.path.insert(0, os.path.dirname(_target_program_path))

try:
    spec = _il.spec_from_file_location("program", _target_program_path)
    program = _il.module_from_spec(spec)
    spec.loader.exec_module(program)
    sys.modules["program"] = program

    func = getattr(program, _target_function_name)
    result = func()

    with open(_results_path, "wb") as f:
        pickle.dump(result, f)

except Exception as e:
    try:
        with open(_results_path, "wb") as f:
            pickle.dump({"error": str(e)}, f)
    except Exception:
        pass
    traceback.print_exc()
'''
        temp_file.write(injected)
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"

    # Fill placeholders
    with open(temp_file_path, "r+", encoding="utf-8") as f:
        s = f.read()
        s = s.replace("__PROGRAM_PATH__", program_path)
        s = s.replace("__FUNCTION_NAME__", function_name)
        s = s.replace("__RESULTS_PATH__", results_path)
        s = s.replace("__MAX_CPUS__", str(max(1, num_cpus)))
        f.seek(0)
        f.write(s)
        f.truncate()

    # Thread caps for BLAS libs in the child
    env = os.environ.copy()
    t = str(max(1, num_cpus))
    env.setdefault("OMP_NUM_THREADS", t)
    env.setdefault("MKL_NUM_THREADS", t)
    env.setdefault("OPENBLAS_NUM_THREADS", t)
    env.setdefault("NUMEXPR_NUM_THREADS", t)
    env.setdefault("VECLIB_MAXIMUM_THREADS", t)
    env.setdefault("BLIS_NUM_THREADS", t)

    def _kill_process_tree(p, pgid, hard=False):
        """Kill process tree."""
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL if hard else signal.SIGTERM)
            except Exception:
                pass
        if shutil.which("pkill"):
            try:
                subprocess.run(
                    ["pkill", "-KILL" if hard else "-TERM", "-P", str(p.pid)],
                    check=False
                )
            except Exception:
                pass

    process = None
    try:
        # Start subprocess in its own session/process group
        process = subprocess.Popen(
            [sys.executable, temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )

        # Capture PGID
        try:
            pgid = os.getpgid(process.pid)
        except Exception:
            pgid = None

        stdout, stderr = process.communicate(timeout=timeout_seconds)
        exit_code = process.returncode

        # Soft sweep first, then hard sweep
        _kill_process_tree(process, pgid, hard=False)
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
        _kill_process_tree(process, pgid, hard=True)

        # Write stdout for debugging
        stdout_path = program_path + ".stdout"
        try:
            with open(stdout_path, "w") as sf:
                sf.write(stdout.decode(errors="ignore"))
        except Exception:
            pass

        if exit_code != 0:
            if stderr:
                sys.stderr.write(stderr.decode(errors="ignore"))
            raise RuntimeError(f"Process exited with code {exit_code}")

        if os.path.exists(results_path):
            with open(results_path, "rb") as f:
                results = pickle.load(f)
            if isinstance(results, dict) and "error" in results:
                raise RuntimeError(f"Program execution failed: {results['error']}")
            return results
        else:
            raise RuntimeError("Results file not found")

    except subprocess.TimeoutExpired:
        # TERM -> brief wait -> KILL
        try:
            pgid = os.getpgid(process.pid) if process else None
        except Exception:
            pgid = None
        _kill_process_tree(process, pgid, hard=False)
        try:
            process.wait(timeout=1.0)
        except Exception:
            pass
        _kill_process_tree(process, pgid, hard=True)
        try:
            process.wait(timeout=0.5)
        except Exception:
            pass
        raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")

    finally:
        # Cleanup temp files
        for path in [temp_file_path, results_path]:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass


class BaseRewardTask(ABC):
    """Abstract base class for tasks with static methods.
    
    Uses AReaL standard subprocess execution - no Ray required.
    """

    worst_perf_log: float
    reward_type: RewardType
    eval_timeout: int
    fail_score: float
    n_item: int
    num_cpus_per_task: int
    log_dir: str

    def __init__(self, config, log_dir: str):
        self.num_cpus_per_task = config.ttt_rm.num_cpus_per_task
        assert self.num_cpus_per_task > 0, "Must allow 1 cpu per task"

        reward_type = config.ttt_rm.rew_type
        self.reward_type = RewardType(reward_type)
        self.fail_score = config.ttt_rm.fail_score
        self.eval_timeout = config.ttt_rm.eval_timeout
        self.worst_perf_log = config.ttt_rm.worst_perf_log
        self.n_item = config.ttt_rm.n_item
        self.log_dir = log_dir

        tmp_dir = Path(self.log_dir) / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        if self.reward_type == RewardType.NEG_LINEAR:
            assert self.fail_score < 0, (
                f"Fail score should be much less than 0 when using rew_type {RewardType.NEG_LINEAR.value}, "
                f"found fail_score={self.fail_score}."
            )

    def preprocess_generation(self, generation: str, *args, **kwargs) -> str:
        """Preprocess generation before execution. Override in subclass if needed."""
        return generation

    @abstractmethod
    def get_reward(self, result) -> float:
        """Parse the result and extract the reward."""
        raise NotImplementedError("You must implement 'get_reward' for a RewardTask.")

    @abstractmethod
    def verify(self, result, *args, **kwargs) -> bool:
        """Verify input/output correctness. Returns True/False."""
        raise NotImplementedError("You must implement 'verify' for a RewardTask.")

    @abstractmethod
    def get_function_name(self) -> str:
        """Return the function name to call in the generated code."""
        raise NotImplementedError("You must implement 'get_function_name' for a RewardTask.")

    def compute_score(self, solution_str: str, *args, **kwargs) -> dict[str, Any]:
        """Compute score for a solution string."""
        # Parse python code for solution
        code = self._extract_code(solution_str)
        if code is None:
            return self._get_failure_entry('cannot extract python code from model response')

        # Any task specific modifications to the code
        code = self.preprocess_generation(code, *args, **kwargs)

        # Eval task
        metrics = {}
        with _timer("propose_candidate_time", metrics):
            try:
                result = self.run_eval_code(code)
            except TimeoutError:
                return self._get_failure_entry(f'Evaluation timed out after {self.eval_timeout} seconds.')
            except Exception as e:
                return self._get_failure_entry(f'Evaluation failed: {e}')

        # Validate results
        try:
            is_valid = self.verify(result, *args, **kwargs)
        except Exception as e:
            logger.warning(f"Verification failed: {e}")
            return self._get_failure_entry(f'Program results failed to execute verification, {e}.')

        if not is_valid:
            return self._get_failure_entry('Program results failed to pass verification.')

        # Extract proper reward
        reward = self.get_reward(result)

        # Shape reward and return, include result_construction for state updates
        out = self._transform_reward(reward)
        out["result_construction"] = list(result) if hasattr(result, '__iter__') else result
        out["stdout"] = getattr(self, '_last_stdout', '')
        return out

    def _transform_reward(self, value: float) -> dict[str, Any]:
        """Transform raw reward value based on reward_type."""
        match self.reward_type:
            case RewardType.LINEAR:
                score = value
                performance = value
            case RewardType.EXP_CF:
                score = np.exp(-value)
                performance = -value
            case RewardType.RECIPROCAL_CF:
                score = 1 / (1e-8 + value)
                performance = -value
            case RewardType.SCALED_RECIPROCAL_CF:
                score = 5 / (1e-8 + value)
                performance = -value
            case RewardType.NEG_LINEAR:
                score = -value
                performance = -value
            case _:
                raise ValueError(f"'{self.reward_type.value}' is not supported for reward type.")

        return dict(
            msg=f"success; bound={value}",
            correctness=1.0,
            score=score,
            performance=performance
        )

    def run_eval_code(self, code_str: str):
        """
        Execute code string and return results.
        
        Uses standard subprocess execution - no Ray required.
        """
        import re
        
        tmp_dir = Path(self.log_dir) / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        code_path = None
        results_path = None

        # Write code to temp file
        with tempfile.NamedTemporaryFile(
            suffix=".py",
            delete=False,
            mode="w",
            dir=str(tmp_dir),
        ) as f:
            code_path = f.name
            f.write(code_str)

        # Compute expected stdout path
        expected_stdout_path = Path(code_path).with_suffix(".pkl.stdout")

        try:
            # Direct subprocess execution
            result = run_with_timeout(
                code_path,
                self.get_function_name(),
                timeout_seconds=self.eval_timeout,
                num_cpus=self.num_cpus_per_task,
            )

            # Save results
            results_path = str(Path(code_path).with_suffix(".pkl"))
            with open(results_path, "wb") as f:
                pickle.dump(result, f)

            # Load stdout if available
            stdout_path = str(results_path) + ".stdout"
            try:
                if os.path.exists(stdout_path):
                    with open(stdout_path, "r") as sf:
                        self._last_stdout = sf.read()
                else:
                    self._last_stdout = ""
            except Exception:
                self._last_stdout = ""

            return result

        except Exception:
            # On failure, still try to load stdout for debugging
            try:
                if os.path.exists(expected_stdout_path):
                    with open(expected_stdout_path, "r") as sf:
                        self._last_stdout = sf.read()
            except Exception:
                pass
            raise

        finally:
            # Cleanup temp artifacts
            for path in [code_path, results_path]:
                if path is not None:
                    try:
                        if os.path.exists(path):
                            os.unlink(path)
                    except (FileNotFoundError, OSError):
                        pass

            try:
                os.unlink(expected_stdout_path)
            except (FileNotFoundError, OSError):
                pass

    def _extract_code(self, response: str) -> str | None:
        """Extract Python code from markdown response."""
        import re
        m = re.search(r"```python\s+([\s\S]*?)\s*```", response)
        return m.group(1).strip() if m is not None else None

    def _get_failure_entry(self, msg: str) -> dict[str, Any]:
        """Return a failure entry."""
        return dict(
            score=self.fail_score,
            msg=msg,
            correctness=0.0,
            performance=self.worst_perf_log,
            stdout=getattr(self, '_last_stdout', ''),
        )


if __name__ == "__main__":
    pass
