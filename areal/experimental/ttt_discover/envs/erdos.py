"""
Erdos Min Overlap environment for TTT-Discover.

Implements the latest ttt-discover ErdosMinOverlapEnv / ErdosMinOverlapRewardEvaluator
logic inside the AReaL BaseEnv interface.
"""

import sys
import os
import tempfile
import subprocess
import pickle
import signal
import shutil
import inspect
from pathlib import Path
from typing import Any

import numpy as np

from areal.experimental.ttt_discover.envs.env import BaseEnv, EnvResult
from areal.experimental.ttt_discover.state import ErdosState, State
from areal.utils import logging

import torch
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger("ErdosEnv")


### VERIFIER FUNCTIONS (from latest ttt-discover codebase) ###


def verify_c5_solution(h_values: np.ndarray, c5_achieved: float, n_points: int):
    if not isinstance(h_values, np.ndarray):
        try:
            h_values = np.array(h_values, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert h_values to numpy array: {e}")

    if len(h_values.shape) != 1:
        raise ValueError(f"h_values must be 1D array, got shape {h_values.shape}")

    if h_values.shape[0] != n_points:
        raise ValueError(f"Expected h shape ({n_points},), got {h_values.shape}")

    if not np.all(np.isfinite(h_values)):
        raise ValueError("h_values contain NaN or inf values")

    if np.any(h_values < 0) or np.any(h_values > 1):
        raise ValueError(f"h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")

    n = n_points
    target_sum = n / 2.0
    current_sum = np.sum(h_values)

    if current_sum != target_sum:
        h_values = h_values * (target_sum / current_sum)
        if np.any(h_values < 0) or np.any(h_values > 1):
            raise ValueError(
                f"After normalization, h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]"
            )

    dx = 2.0 / n_points

    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    computed_c5 = np.max(correlation)

    if not np.isfinite(computed_c5):
        raise ValueError(f"Computed C5 is not finite: {computed_c5}")

    if not np.isclose(computed_c5, c5_achieved, atol=1e-4):
        raise ValueError(f"C5 mismatch: reported {c5_achieved:.6f}, computed {computed_c5:.6f}")

    return computed_c5


def evaluate_erdos_solution(h_values: np.ndarray, c5_bound: float, n_points: int) -> float:
    verify_c5_solution(h_values, c5_bound, n_points)
    return float(c5_bound)


def verify_erdos_solution(result: tuple) -> bool:
    try:
        h_values, c5_bound, n_points = result
        c5_bound = evaluate_erdos_solution(h_values, c5_bound, n_points)
        if c5_bound <= 0 or np.isnan(c5_bound) or np.isinf(c5_bound):
            return False
    except Exception:
        return False
    return True


### ENVIRONMENT CLASS ###


class ErdosEnv(BaseEnv):
    """
    Environment for Erdos minimum overlap problem.

    Task: Find a construction that minimizes the C5 overlap bound.
    """

    def __init__(
        self,
        n: int = 200,
        budget_s: int = 1000,
        eval_timeout: int = 1100,
        log_dir: str = "/tmp/ttt_logs",
        num_cpus: int = 2,
    ):
        self.n = n
        self.budget_s = budget_s
        self.eval_timeout = eval_timeout
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.num_cpus = num_cpus
        self.entrypoint = "run"

    def get_prompt(self, state: ErdosState) -> str:
        """Generate prompt matching the latest ttt-discover ErdosMinOverlapEnv."""
        state_ctx = state.to_prompt(
            target=0.3808, metric_name="C₅ bound", maximize=False, language="python"
        )

        # Construction section
        construction_section = ""
        construction = getattr(state, "construction", None)
        if construction is not None and len(construction) > 0:
            construction_section = (
                f"\nYou may want to start your search from the current construction, "
                f"which you can access through the `initial_h_values` global variable "
                f"(n={len(construction)} samples).\n"
                f"You are encouraged to explore solutions that use other starting points "
                f"to prevent getting stuck in a local optimum."
            )

        # Code section
        if state.code and state.code.strip():
            code_section = (
                "Reason about how you could further improve this construction.\n"
                "Ideally, try to do something different than the above algorithm. "
                "Could be using different algorithmic ideas, adjusting your heuristics, "
                "adjusting / sweeping your hyperparameters, etc.\n"
                "Unless you make a meaningful improvement, you will not be rewarded."
            )
        else:
            code_section = "Write code to optimize this construction."

        return f"""You are an expert in harmonic analysis, numerical optimization, and mathematical discovery.
Your task is to find an improved upper bound for the Erdős minimum overlap problem constant C₅.

## Problem

Find a step function h: [0, 2] → [0, 1] that **minimizes** the overlap integral:

$$C_5 = \\max_k \\int h(x)(1 - h(x+k)) dx$$

**Constraints**:
1. h(x) ∈ [0, 1] for all x
2. ∫₀² h(x) dx = 1

**Discretization**: Represent h as n_points samples over [0, 2].
With dx = 2.0 / n_points:
- 0 ≤ h[i] ≤ 1 for all i
- sum(h) * dx = 1 (equivalently: sum(h) == n_points / 2 exactly)

The evaluation computes: C₅ = max(np.correlate(h, 1-h, mode="full") * dx)

Smaller sequences with less than 1k samples are preferred - they are faster to optimize and evaluate.

**Lower C₅ values are better** - they provide tighter upper bounds on the Erdős constant.

## Budget & Resources
- **Time budget**: {self.budget_s}s for your code to run
- **CPUs**: {self.num_cpus} available

## Rules
- Define `run(seed=42, budget_s={self.budget_s}, **kwargs)` that returns `(h_values, c5_bound, n_points)`
- Use scipy, numpy, cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS,ECOS], math
- Make all helper functions top level, no closures or lambdas
- No filesystem or network IO
- `evaluate_erdos_solution()` and `initial_h_values` (an initial construction, if available) are pre-imported
- Your function must complete within budget_s seconds and return the best solution found

**Lower is better**. Current record: C₅ ≤ 0.38092. Our goal is to find a construction that shows C₅ ≤ 0.38080.
{state_ctx}
{construction_section}
{code_section}
"""

    def _preprocess_generation(self, generation: str, state: ErdosState) -> str:
        """Preprocess generation exactly like ErdosMinOverlapRewardEvaluator."""
        verifier_src = inspect.getsource(evaluate_erdos_solution)
        numpy_import = "import numpy as np"

        base = numpy_import + "\n\n" + verifier_src + "\n\n"

        if state is None:
            raise ValueError(
                "state is required for preprocess_generation. "
                "Use ExperienceSampler to provide initial state with construction."
            )

        construction = getattr(state, "construction", None)
        if construction is not None:
            initial_h_values = f"initial_h_values = np.array({list(construction)!r})"
            base += initial_h_values + "\n\n"

        return base + generation

    def _execute_code(self, code: str, state: ErdosState) -> tuple[Any, str, str]:
        """
        Execute the generated code in a sandboxed subprocess.
        Returns (result, error_msg, stdout).
        """
        full_code = self._preprocess_generation(code, state)

        tmp_dir = self.log_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Write code to temp file
        with tempfile.NamedTemporaryFile(
            suffix=".py",
            delete=False,
            mode="w",
            dir=str(tmp_dir),
        ) as f:
            code_path = f.name
            f.write(full_code)

        # Create runner script
        with tempfile.NamedTemporaryFile(
            suffix=".py",
            delete=False,
            mode="w",
            dir=str(tmp_dir),
        ) as f:
            runner_path = f.name
            injected = r"""
import sys
import os
import pickle
import traceback
import importlib.util as _il

os.environ.setdefault("OMP_NUM_THREADS", "__MAX_CPUS__")
os.environ.setdefault("MKL_NUM_THREADS", "__MAX_CPUS__")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "__MAX_CPUS__")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "__MAX_CPUS__")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "__MAX_CPUS__")
os.environ.setdefault("BLIS_NUM_THREADS", "__MAX_CPUS__")

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
"""
            t = str(max(1, self.num_cpus))
            injected = injected.replace("__MAX_CPUS__", t)
            injected = injected.replace("__PROGRAM_PATH__", code_path)
            injected = injected.replace("__FUNCTION_NAME__", self.entrypoint)
            results_path = f"{runner_path}.results"
            injected = injected.replace("__RESULTS_PATH__", results_path)
            f.write(injected)

        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", t)
        env.setdefault("MKL_NUM_THREADS", t)
        env.setdefault("OPENBLAS_NUM_THREADS", t)
        env.setdefault("NUMEXPR_NUM_THREADS", t)
        env.setdefault("VECLIB_MAXIMUM_THREADS", t)
        env.setdefault("BLIS_NUM_THREADS", t)

        process = None
        try:
            process = subprocess.Popen(
                [sys.executable, runner_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                start_new_session=True,
            )

            stdout_bytes, stderr_bytes = process.communicate(timeout=self.eval_timeout)
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            exit_code = process.returncode

            # Save stdout for debugging
            stdout_path = str(Path(code_path).with_suffix(".pkl.stdout"))
            try:
                with open(stdout_path, "w") as sf:
                    sf.write(stdout)
            except Exception:
                pass

            if exit_code != 0:
                return None, f"Process exited with code {exit_code}: {stderr[:500]}", stdout

            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)
                if isinstance(results, dict) and "error" in results:
                    return None, f"Program execution failed: {results['error']}", stdout
                return results, "", stdout
            else:
                return None, "Results file not found", stdout

        except subprocess.TimeoutExpired:
            try:
                pgid = os.getpgid(process.pid) if process else None
            except Exception:
                pgid = None
            if pgid is not None:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except Exception:
                    pass
            if shutil.which("pkill"):
                try:
                    subprocess.run(["pkill", "-KILL", "-P", str(process.pid)], check=False)
                except Exception:
                    pass
            try:
                process.wait(timeout=1.0)
            except Exception:
                pass
            return None, f"Timeout after {self.eval_timeout}s", ""

        finally:
            for path in [code_path, runner_path, results_path]:
                if path is not None:
                    try:
                        if os.path.exists(path):
                            os.unlink(path)
                    except (FileNotFoundError, OSError):
                        pass

            stdout_path = str(Path(code_path).with_suffix(".pkl.stdout")) if code_path is not None else None
            if stdout_path is not None:
                try:
                    if os.path.exists(stdout_path):
                        os.unlink(stdout_path)
                except (FileNotFoundError, OSError):
                    pass

    def execute(self, code: str, state: ErdosState) -> EnvResult:
        """Execute the generated code and compute reward."""
        try:
            output, error_msg, stdout = self._execute_code(code, state)

            if error_msg:
                is_timeout = "timeout" in error_msg.lower() or "timed out" in error_msg.lower()
                return EnvResult(
                    reward=0.0,
                    observation=error_msg,
                    is_valid=False,
                    fail_type="timeout" if is_timeout else "execution_error",
                    metadata={"timeout": is_timeout, "error": error_msg},
                )

            if not verify_erdos_solution(output):
                return EnvResult(
                    reward=0.0,
                    observation="Invalid solution.",
                    is_valid=False,
                    fail_type="execution_error",
                    metadata={"error": "Invalid solution."},
                )

            h_values, c5_bound, n_points = output
            c5_bound = evaluate_erdos_solution(h_values, c5_bound, n_points)

            reward = float(1.0 / (1e-8 + c5_bound))

            return EnvResult(
                reward=reward,
                observation=stdout or "Success",
                is_valid=True,
                metadata={
                    "construction": list(h_values),
                    "c5_bound": c5_bound,
                    "n_points": n_points,
                    "raw_score": c5_bound,
                    "msg": f"C5 bound: {c5_bound}",
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
        parent_state: ErdosState,
        code: str,
        reward: float,
        result: EnvResult,
        timestep: int,
    ) -> ErdosState:
        """Create new ErdosState with parent tracking.

        Store the display metric (-c5_bound) in value for prompt/UCT compatibility,
        matching the latest discover convention.
        """
        construction = None
        c5_bound = None
        if result.metadata:
            construction = result.metadata.get("construction")
            c5_bound = result.metadata.get("c5_bound")

        parent_values = []
        parents = []
        if parent_state.value is not None:
            parent_values.append(parent_state.value)
            parents.append({"id": parent_state.id, "timestep": parent_state.timestep})
        if parent_state.parent_values:
            parent_values.extend(parent_state.parent_values)
        if parent_state.parents:
            parents.extend(parent_state.parents)

        # Use display metric for state.value (negative for minimization problems)
        display_value = -c5_bound if c5_bound is not None else reward

        return ErdosState(
            timestep=timestep,
            code=code,
            value=display_value,
            c5_bound=c5_bound,
            construction=construction,
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
        """Custom failure handling."""
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
        """Create failed trajectory with reward=0."""
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


def create_initial_state_erdos(
    n: int = 200,
    initial_exp_type: str = "best_available",
    budget_s: int = 1000,
    **kwargs,
) -> ErdosState:
    """Create initial state for Erdos problem (matching latest discover code)."""
    rng = np.random.default_rng()
    n_points = rng.integers(40, 100)
    construction = np.ones(n_points) * 0.5
    perturbation = rng.uniform(-0.4, 0.4, n_points)
    perturbation = perturbation - np.mean(perturbation)
    construction = construction + perturbation
    dx = 2.0 / n_points
    correlation = np.correlate(construction, 1 - construction, mode="full") * dx
    c5_bound = float(np.max(correlation))

    return ErdosState(
        timestep=-1,
        code="",
        value=-c5_bound,
        c5_bound=c5_bound,
        construction=list(construction),
    )
