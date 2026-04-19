"""
Circle Packing environment for TTT-Discover.

Maximizes sum of radii for packing n circles in a unit square.
Adapted from the original TTT-Discover paper source code.

This implementation is compatible with AReaL's BaseEnv interface
and does NOT depend on external tasks/alphaevolve_cp modules.
"""

import inspect
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from areal.experimental.ttt_discover.envs.env import BaseEnv, EnvResult
from areal.experimental.ttt_discover.state import CirclePackingState, State
from areal.utils import logging

import torch
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger("CirclePackingEnv")


# =============================================================================
# Verifier functions (from original paper source code)
# =============================================================================

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n,) with radius of each circle

    Returns:
        True if valid, False otherwise
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        return False, "NaN values detected in circle centers"

    if np.isnan(radii).any():
        return False, "NaN values detected in circle radii"

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            return False, f"Circle {i} has negative radius {radii[i]}"
        elif np.isnan(radii[i]):
            return False, f"Circle {i} has nan radius"

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False, f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square"

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:  # Allow for tiny numerical errors
                return False, (
                    f"Circles {i} and {j} overlap: "
                    f"dist={dist}, r1+r2={radii[i] + radii[j]}"
                )

    return True, ""


def check_packing_correctness(centers, radii, num_circles: int) -> bool:
    shape_valid = centers.shape == (num_circles, 2) and radii.shape == (num_circles,)
    if not shape_valid:
        return False
    valid, _ = validate_packing(centers, radii)
    return valid


# =============================================================================
# Prompt template (adapted from original paper source code)
# =============================================================================

def _get_cp_prompt_template(n_item: int, validator_src: str, state_ctx: str) -> str:
    """Generate the Circle Packing prompt from the original paper."""
    target = 2.636 if n_item == 26 else 2.940
    return f"""You are an expert mathematician specializing in circle packing problems and computational geometry.

Your task is to pack {n_item} circles in a unit square [0,1]×[0,1] to maximize the sum of radii.

We will run the below validation function (read-only, do not modify this):
```python
{validator_src}
```

{state_ctx}

Reason about how you could further improve this packing. Consider:
- Are circles placed optimally near boundaries and corners?
- Could a different arrangement (hexagonal, nested, hybrid) yield better results?
- Are there gaps that could be filled with repositioned or resized circles?
- Could optimization parameters or methods be improved?

Rules:
- You must define the run_packing function: def run_packing() -> tuple
- Returns (centers, radii, sum_radii) where centers has shape ({n_item}, 2) and radii has shape ({n_item},).
- You can use scientific libraries like scipy, numpy, cvxpy, math.
- Centers must lie within [0,1]^2 and radii must be nonnegative.
- The pair (centers, radii) must satisfy non-overlap and boundary constraints.
- Make all helper functions top level and have no closures from function nesting. Don't use any lambda functions.
- No filesystem or network IO.
- You need to get really creative and think from first principles.

Make sure to /think step by step, first give your strategy between <strategy> and </strategy> tags, then finally return the final program between ```python and ```.
"""


# =============================================================================
# Environment class
# =============================================================================

class CirclePackingEnv(BaseEnv):
    """
    Environment for circle packing optimization.

    Task: Pack n circles in [0,1]×[0,1] to maximize the sum of radii.

    Example:
        >>> env = CirclePackingEnv(n_item=26, eval_timeout=530)
        >>> state = create_initial_state_cp(n=26, initial_exp_type="best_available")
        >>> prompt = env.get_prompt(state)
    """

    def __init__(
        self,
        n_item: int = 26,
        eval_timeout: int = 530,
        log_dir: str = "/tmp/ttt_logs",
        num_cpus: int = 1,
        max_memory_mb: int = 4096,
    ):
        """
        Args:
            n_item: Number of circles to pack (26 or 32)
            eval_timeout: Timeout for code execution (seconds)
            log_dir: Directory for temporary code files
            num_cpus: Number of CPUs per task
            max_memory_mb: Maximum memory per subprocess (MB)
        """
        self.n_item = n_item
        self.eval_timeout = eval_timeout
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.num_cpus = num_cpus
        self.max_memory_mb = max_memory_mb
        self.entrypoint = "run_packing"
        self.memory_threshold = 0.95

    def get_prompt(self, state: CirclePackingState) -> str:
        """Generate improvement prompt for the given state."""
        validator_src = inspect.getsource(validate_packing)
        target = 2.636 if self.n_item == 26 else 2.940

        # Use state.to_prompt for value context (code, before/after, stdout)
        state_ctx = state.to_prompt(
            target=target,
            metric_name="sum of radii",
            maximize=True,
            language="python",
        )

        return _get_cp_prompt_template(self.n_item, validator_src, state_ctx)

    def _execute_code(self, code: str, state: CirclePackingState) -> tuple[Any, str]:
        """
        Execute the generated code in a sandboxed subprocess.
        Returns (result, error_msg).
        """
        # Preprocess code: inject verifier and numpy
        numpy_import = "import numpy as np"
        verifier_src = inspect.getsource(validate_packing)
        verifier_src += "\n\n" + inspect.getsource(check_packing_correctness) + "\n"

        base = numpy_import + "\n\n" + verifier_src + "\n\n"

        full_code = base + code

        # Write to temp file
        with tempfile.NamedTemporaryFile(
            suffix=".py",
            delete=False,
            mode="w",
            dir=str(self.log_dir),
        ) as f:
            code_path = f.name
            f.write(full_code)

        # Create runner script
        with tempfile.NamedTemporaryFile(
            suffix=".py",
            delete=False,
            mode="w",
            dir=str(self.log_dir),
        ) as f:
            runner_path = f.name
            runner_code = f'''
import sys
import os
import traceback
import importlib.util as _il

# Thread caps
os.environ.setdefault("OMP_NUM_THREADS", "{max(1, self.num_cpus)}")
os.environ.setdefault("MKL_NUM_THREADS", "{max(1, self.num_cpus)}")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "{max(1, self.num_cpus)}")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "{max(1, self.num_cpus)}")

sys.path.insert(0, "{str(self.log_dir)}")

try:
    spec = _il.spec_from_file_location("program", "{code_path}")
    program = _il.module_from_spec(spec)
    spec.loader.exec_module(program)
    sys.modules["program"] = program

    func = getattr(program, "{self.entrypoint}")
    result = func()

    # Serialize numpy arrays to lists for safe eval
    centers, radii, sum_radii = result
    if hasattr(centers, "tolist"):
        centers = centers.tolist()
    if hasattr(radii, "tolist"):
        radii = radii.tolist()

    print(f"REWARD_RESULT: {{(centers, radii, sum_radii)!r}}")

except Exception as e:
    print(f"REWARD_ERROR: {{e}}")
    traceback.print_exc()
'''
            f.write(runner_code)

        # Monitor RAM before starting new execution
        mem = psutil.virtual_memory()
        while mem.percent >= self.memory_threshold * 100:
            logger.warning(
                f"System memory usage is {mem.percent:.1f}%, "
                f"exceeds threshold {self.memory_threshold * 100:.0f}%. "
                f"Pausing execution until memory drops..."
            )
            time.sleep(5)
            mem = psutil.virtual_memory()

        process = None
        try:
            env = os.environ.copy()
            t = str(max(1, self.num_cpus))
            env.setdefault("OMP_NUM_THREADS", t)
            env.setdefault("MKL_NUM_THREADS", t)
            env.setdefault("OPENBLAS_NUM_THREADS", t)
            env.setdefault("NUMEXPR_NUM_THREADS", t)

            def _limit_memory():
                if self.max_memory_mb > 0:
                    max_bytes = self.max_memory_mb * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))

            process = subprocess.Popen(
                [sys.executable, runner_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                start_new_session=True,
                preexec_fn=_limit_memory,
            )

            try:
                stdout, stderr = process.communicate(timeout=self.eval_timeout)
                stdout = stdout.decode("utf-8", errors="replace")
                stderr = stderr.decode("utf-8", errors="replace")

                # Parse result
                for line in stdout.split("\n"):
                    if line.startswith("REWARD_RESULT:"):
                        result_str = line.split(":", 1)[1].strip()
                        try:
                            result = eval(result_str)
                            return result, ""
                        except Exception as e:
                            return None, f"Failed to parse result: {result_str} ({e})"
                    elif line.startswith("REWARD_ERROR:"):
                        error_msg = line.split(":", 1)[1].strip()
                        return None, error_msg

                # No result found
                if process.returncode != 0:
                    is_memory_error = (
                        "MemoryError" in stderr
                        or process.returncode == -9
                        or process.returncode == -11
                    )
                    if is_memory_error:
                        return None, f"Memory limit exceeded (max {self.max_memory_mb}MB)"
                    return None, f"Process failed: {stderr[:500]}"

                return None, "No result returned"

            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except Exception:
                    pass
                try:
                    process.wait(timeout=5)
                except Exception:
                    pass
                return None, f"Timeout after {self.eval_timeout}s"

        finally:
            if process is not None:
                if process.poll() is None:
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except Exception:
                        pass
                try:
                    process.wait(timeout=5)
                except Exception:
                    pass

            try:
                os.unlink(code_path)
                os.unlink(runner_path)
            except Exception:
                pass

    def execute(self, code: str, state: CirclePackingState) -> EnvResult:
        """Execute the generated code and compute reward."""
        try:
            output, error_msg = self._execute_code(code, state)

            if error_msg:
                is_timeout = "timeout" in error_msg.lower()
                if is_timeout:
                    return EnvResult(
                        reward=0.0,
                        observation=error_msg,
                        is_valid=False,
                        fail_type="timeout",
                        metadata={"timeout": True, "error": error_msg},
                    )
                return EnvResult(
                    reward=0.0,
                    observation=error_msg,
                    is_valid=False,
                    fail_type="execution_error",
                    metadata={"error": error_msg},
                )

            # Extract output
            centers, radii, _ = output
            if not isinstance(centers, np.ndarray):
                centers = np.array(centers)
            if not isinstance(radii, np.ndarray):
                radii = np.array(radii)

            # Check if packing is valid
            if not check_packing_correctness(centers, radii, self.n_item):
                return EnvResult(
                    reward=0.0,
                    observation="Packing is not valid.",
                    is_valid=False,
                    fail_type="execution_error",
                    metadata={"error": "Packing is not valid."},
                )

            # Final reward is sum of radii
            sum_of_radii = float(np.sum(radii))

            # Convert construction to circles format (x, y, r)
            circles = [
                [float(centers[i, 0]), float(centers[i, 1]), float(radii[i])]
                for i in range(len(radii))
            ]

            return EnvResult(
                reward=sum_of_radii,
                observation=f"Success; raw_score={sum_of_radii}",
                is_valid=True,
                metadata={
                    "circles": circles,
                    "sum_radii": sum_of_radii,
                    "raw_score": sum_of_radii,
                },
            )

        except Exception as e:
            logger.warning(f"Exception in execute: {e}", exc_info=True)
            return EnvResult(
                reward=0.0,
                observation=str(e),
                is_valid=False,
                fail_type="execution_error",
                metadata={"error": str(e)},
            )

    def create_state(
        self,
        parent_state: CirclePackingState,
        code: str,
        reward: float,
        result: EnvResult,
        timestep: int,
    ) -> CirclePackingState:
        """Create new CirclePackingState with proper parent tracking."""
        circles = None
        if result.metadata and "circles" in result.metadata:
            circles = result.metadata["circles"]

        parent_values = []
        parents = []
        if parent_state.value is not None:
            parent_values.append(parent_state.value)
            parents.append({"id": parent_state.id, "timestep": parent_state.timestep})
        if parent_state.parent_values:
            parent_values.extend(parent_state.parent_values)
        if parent_state.parents:
            parents.extend(parent_state.parents)

        return CirclePackingState(
            timestep=timestep,
            construction=circles,
            code=code,
            value=reward,
            parent_values=parent_values,
            parents=parents,
            observation=result.observation,
        )

    def get_failure_result(
        self,
        state: State | None,
        fail_type: str,
        error_msg: str = "",
    ) -> EnvResult:
        """Custom failure handling for Circle Packing."""
        if fail_type == "timeout":
            return EnvResult(
                reward=0.0,
                observation=f"Execution timeout after {self.eval_timeout}s",
                is_valid=False,
                fail_type=fail_type,
                metadata={"timeout": True},
            )
        elif fail_type == "code_extraction_failed":
            return EnvResult(
                reward=0.0,
                observation="No valid Python code block found in response",
                is_valid=False,
                fail_type=fail_type,
            )
        elif fail_type == "execution_error":
            return EnvResult(
                reward=0.0,
                observation=f"Execution failed: {error_msg}" if error_msg else "Execution failed",
                is_valid=False,
                fail_type=fail_type,
            )
        else:
            return super().get_failure_result(state, fail_type, error_msg)

    def create_failed_trajectory(
        self,
        state: State | None,
        input_ids: list[int],
        tokenizer: PreTrainedTokenizerFast,
        fail_type: str,
        error_msg: str = "",
    ) -> dict[str, torch.Tensor]:
        """Create failed trajectory for Circle Packing."""
        result = self.get_failure_result(state, fail_type, error_msg)

        seq = input_ids + [tokenizer.eos_token_id or 0]

        return {
            "input_ids": torch.tensor(seq, dtype=torch.int32).unsqueeze(0),
            "loss_mask": torch.ones(len(seq), dtype=torch.int32).unsqueeze(0),
            "logprobs": torch.zeros(len(seq), dtype=torch.float32).unsqueeze(0),
            "versions": torch.full((len(seq),), -1, dtype=torch.int32).unsqueeze(0),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
            "rewards": torch.tensor([result.reward], dtype=torch.float32),
        }


def create_initial_state_cp(
    n: int = 26,
    initial_exp_type: str = "best_available",
    **kwargs
) -> CirclePackingState:
    """
    Create initial state for circle packing.

    Following the original TTT-Discover paper source code, the initial state
    is a blank default state with no code provided. The model must generate
    the packing solution from scratch.

    Args:
        n: Number of circles (26 or 32) — unused, kept for API compatibility
        initial_exp_type: One of "best_available", "none", "random" — all
            return the same blank state in Circle Packing
        **kwargs: Additional arguments

    Returns:
        Initial CirclePackingState with empty code and value=0.0
    """
    return CirclePackingState(
        timestep=-1,
        construction=None,
        code="",
        value=0.0,
    )
