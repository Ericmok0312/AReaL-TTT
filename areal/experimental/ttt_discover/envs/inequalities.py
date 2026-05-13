"""
Inequalities (AlphaEvolve AC1/AC2) environment for TTT-Discover.

Based on latest ttt-discover codebase - simplified architecture without task layer.
"""

import sys
import os
import resource
import tempfile
import subprocess
import pickle
import signal
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from areal.experimental.ttt_discover.envs.env import BaseEnv, EnvResult
from areal.experimental.ttt_discover.state import InequalitiesState, State
from areal.utils import logging

import torch
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger("InequalitiesEnv")


### VERIFIER FUNCTIONS (from latest codebase) ###

def evaluate_sequence_ac1(sequence: list[float]) -> float:
    """
    Evaluates a sequence of coefficients for AC1 (minimize upper bound).
    Returns np.inf if the input is invalid.
    """
    # Verify that the input is a list
    if not isinstance(sequence, list):
        return np.inf

    # Reject empty lists
    if not sequence:
        return np.inf

    # Check each element in the list for validity
    for x in sequence:
        # Reject boolean types and other non-numeric types
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return np.inf
        # Reject NaN and infinity values
        if np.isnan(x) or np.isinf(x):
            return np.inf

    # Convert all elements to float for consistency
    sequence = [float(x) for x in sequence]

    # Protect against negative numbers
    sequence = [max(0, x) for x in sequence]

    # Protect against numbers that are too large
    sequence = [min(1000.0, x) for x in sequence]

    n = len(sequence)
    b_sequence = np.convolve(sequence, sequence)
    max_b = max(b_sequence)
    sum_a = np.sum(sequence)

    # Protect against the case where the sum is too close to zero
    if sum_a < 0.01:
        return np.inf

    return float(2 * n * max_b / (sum_a**2))


def evaluate_sequence_ac2(sequence: list[float]) -> float:
    """
    Evaluates a sequence of coefficients for AC2 (maximize lower bound).
    Returns -np.inf if the input is invalid.
    """
    # Verify that the input is a list
    if not isinstance(sequence, list):
        return -np.inf

    # Reject empty lists
    if not sequence:
        return -np.inf

    # Check each element in the list for validity
    for x in sequence:
        # Reject boolean types and other non-numeric types
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return -np.inf
        # Reject NaN and infinity values
        if np.isnan(x) or np.isinf(x):
            return -np.inf

    # Convert all elements to float for consistency
    sequence = [float(x) for x in sequence]

    # Protect against negative numbers
    sequence = [max(0, x) for x in sequence]

    # Check if sum of sequence will be too close to zero
    if np.sum(sequence) < 0.01:
        return -np.inf
    
    # Protect against numbers that are too large
    sequence = [min(1000.0, x) for x in sequence]

    convolution_2 = np.convolve(sequence, sequence)
    
    # Calculate the 2-norm squared: ||f*f||_2^2
    num_points = len(convolution_2)
    x_points = np.linspace(-0.5, 0.5, num_points + 2)
    x_intervals = np.diff(x_points)
    y_points = np.concatenate(([0], convolution_2, [0]))
    l2_norm_squared = 0.0
    for i in range(len(convolution_2) + 1):
        y1 = y_points[i]
        y2 = y_points[i+1]
        h = x_intervals[i]
        interval_l2_squared = (h / 3) * (y1**2 + y1*y2 + y2**2)
        l2_norm_squared += interval_l2_squared

    # Calculate the 1-norm: ||f*f||_1
    norm_1 = np.sum(np.abs(convolution_2)) / (len(convolution_2) + 1)

    # Calculate the infinity-norm: ||f*f||_inf
    norm_inf = np.max(np.abs(convolution_2))
    
    if norm_1 <= 0 or norm_inf <= 0:
        return -np.inf
        
    C_lower_bound = l2_norm_squared / (norm_1 * norm_inf)
    return C_lower_bound


### PROMPT TEMPLATES (from latest codebase) ###

AC1_LITERATURE = r"""A previous state of the art used the following approach. You can use it as inspiration, but you are not required to use it, and you are encouraged to explore.
```latex
Starting from a nonnegative step function $f=(a_0,\dots,a_{n-1})$ normalized so that $\sum_j a_j=\sqrt{2n}$, set $M=\|f*f\|_\infty$. Next compute $g_0=(b_0,\dots,b_{n-1})$ by solving a linear program, i.e.\ maximizing $\sum_j b_j$ subject to $b_j\ge0$ and $\|f*g_0\|_\infty\le M$; as is standard, the optimum is attained at an extreme point determined by an active set of binding inequalities, here corresponding to important constraints where the convolution bound $(f*g_0)(x)\le M$ is tight and limiting. Rescale $g_0$ to match the normalization, $g=\frac{\sqrt{2n}}{\sum_j b_j}g_0$, and update $f\leftarrow (1-t)f+t g$ for a small $t>0$. Repeating this step produces a sequence with nonincreasing $\|f*f\|_\infty$, and the iteration is continued until it stabilizes.
```"""

AC1_EVAL_FUNCTION = '''```python
import numpy as np

def evaluate_sequence(sequence: list[float]) -> float:
    """
    Evaluates a sequence of coefficients with enhanced security checks.
    Returns np.inf if the input is invalid.
    """
    # Verify that the input is a list
    if not isinstance(sequence, list):
        return np.inf

    # Reject empty lists
    if not sequence:
        return np.inf

    # Check each element in the list for validity
    for x in sequence:
        # Reject boolean types and other non-numeric types
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return np.inf
        # Reject NaN and infinity values
        if np.isnan(x) or np.isinf(x):
            return np.inf

    # Convert all elements to float for consistency
    sequence = [float(x) for x in sequence]

    # Protect against negative numbers
    sequence = [max(0, x) for x in sequence]

    # Protect against numbers that are too large
    sequence = [min(1000.0, x) for x in sequence]

    n = len(sequence)
    b_sequence = np.convolve(sequence, sequence)
    max_b = max(b_sequence)
    sum_a = np.sum(sequence)

    # Protect against the case where the sum is too close to zero
    if sum_a < 0.01:
        return np.inf

    return float(2 * n * max_b / (sum_a**2))
```'''


def get_ac1_prompt(budget_s: int, value_context: str) -> str:
    """Generate AC1 prompt from template."""
    return f'''Act as an expert software developer and inequality specialist specializing in creating step functions with certain properties.

Your task is to generate the sequence of non-negative heights of a step function, that minimizes the following evaluation function:

{AC1_EVAL_FUNCTION}

{AC1_LITERATURE}

Your task is to write a search function that searches for the best sequence of coefficients. Your function will have {budget_s} seconds to run, and after that it has to have returned the best sequence it found. If after {budget_s} seconds it has not returned anything, it will be terminated with negative infinity points. All numbers in your sequence have to be positive or zero. Larger sequences with 1000s of items often have better attack surface, but too large sequences with 100s of thousands of items may be too slow to search.

You may code up any search method you want, and you are allowed to call the evaluate_sequence() function as many times as you want. You have access to it, you don't need to code up the evaluate_sequence() function.

{value_context}

You may want to start your search from one of the constructions we have found so far, which you can access through the 'height_sequence_1' global variable. 
However, you are encouraged to explore solutions that use other starting points to prevent getting stuck in a local minimum.

Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparemeters, etc. 
Unless you make a meaningful improvement, you will not be rewarded.

Rules:
- You must define the `propose_candidate` function as this is what will be invoked.
- You can use scientific libraries like scipy, numpy, cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS,ECOS], math.
- You can use up to 2 CPUs.
- Make all helper functions top level and have no closures from function nesting. Don't use any lambda functions.
- No filesystem or network IO.
- Do not import evaluate_sequence yourself. Assume it will already be imported and can be directly invoked.
- **Print statements**: Use `print()` to log progress, intermediate bounds, timing info, etc. Your output will be shown back to you.
- Include a short docstring at the top summarizing your algorithm.

Make sure to think and return the final program between ```python and ```.'''


def get_example_program_random_init(num_seconds: int) -> str:
    """Example program with random initialization."""
    return f'''
import numpy as np
import time
from scipy import optimize
linprog = optimize.linprog


def get_good_direction_to_move_into(sequence):
    """Returns a better direction using LP to find g with larger sum while keeping conv bounded."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)
    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]
    rhs = np.max(np.convolve(normalized_sequence, normalized_sequence))
    g_fun = solve_convolution_lp(normalized_sequence, rhs)
    if g_fun is None:
        return None
    sum_g = np.sum(g_fun)
    normalized_g_fun = [x * np.sqrt(2 * n) / sum_g for x in g_fun]
    t = 0.01
    new_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)]
    return new_sequence


def solve_convolution_lp(f_sequence, rhs):
    """Solves the LP: maximize sum(b) s.t. conv(f, b) <= rhs, b >= 0."""
    n = len(f_sequence)
    c = -np.ones(n)
    a_ub = []
    b_ub = []
    for k in range(2 * n - 1):
        row = np.zeros(n)
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                row[j] = f_sequence[i]
        a_ub.append(row)
        b_ub.append(rhs)
    a_ub_nonneg = -np.eye(n)
    b_ub_nonneg = np.zeros(n)
    a_ub = np.vstack([a_ub, a_ub_nonneg])
    b_ub = np.hstack([b_ub, b_ub_nonneg])
    result = linprog(c, A_ub=a_ub, b_ub=b_ub, options={{
        'time_limit': 10.0,
        'disp': False,
    }})
    if result.success:
        return result.x
    return None


def propose_candidate(seed=42, budget_s={num_seconds}, **kwargs):
    np.random.seed(seed)
    deadline = time.time() + budget_s - 10
        
    if np.random.rand() < 0.5:
        # Start from the SOTA sequence
        best_sequence = list(height_sequence_1)
    else:
        # Start from random initialization
        best_sequence = [np.random.random()] * np.random.randint(100, 1000)
    curr_sequence = best_sequence.copy()
    best_score = evaluate_sequence(best_sequence)
    
    while time.time() < deadline:
        h_function = get_good_direction_to_move_into(curr_sequence)
        if h_function is None:
            # Random perturbation if LP fails
            idx = np.random.randint(len(curr_sequence))
            curr_sequence[idx] = max(0, curr_sequence[idx] + np.random.randn() * 0.01)
        else:
            curr_sequence = h_function
        
        try:
            curr_score = evaluate_sequence(curr_sequence)
            if curr_score < best_score:
                best_score = curr_score
                best_sequence = curr_sequence.copy()
        except:
            pass
    
    return best_sequence
'''


### ENVIRONMENT CLASS ###

class InequalitiesEnv(BaseEnv):
    """
    Environment for inequalities (AC1) optimization.
    
    Simplified architecture: directly executes code and evaluates results
    without going through task layer.
    """
    
    construction_length_limits = (1000, 100000)
    
    def __init__(
        self,
        problem_type: str = "ac1",  # "ac1" or "ac2"
        budget_s: int = 1000,
        eval_timeout: int = 600,
        log_dir: str = "/tmp/ttt_logs",
        num_cpus: int = 2,
        memory_threshold: float = 0.60,
        max_memory_mb: int = 4096,
    ):
        self.problem_type = problem_type
        self.budget_s = budget_s
        self.eval_timeout = eval_timeout
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.num_cpus = num_cpus
        self.memory_threshold = memory_threshold
        self.max_memory_mb = max_memory_mb
        
        # Select verifier based on problem type
        if problem_type == "ac1":
            self._evaluate = evaluate_sequence_ac1
            self.is_maximize = False
            self.entrypoint = "propose_candidate"
        elif problem_type == "ac2":
            self._evaluate = evaluate_sequence_ac2
            self.is_maximize = True
            self.entrypoint = "construct_function"
        else:
            raise ValueError(f"Unknown problem_type: {problem_type}. Must be 'ac1' or 'ac2'")
    
    def get_prompt(self, state: InequalitiesState) -> str:
        """Generate improvement prompt using state.to_prompt."""
        if self.problem_type == "ac1":
            metric_name = "upper bound"
            target = 1.5030
            is_maximize = False
        else:
            metric_name = "lower bound"
            target = 0.97
            is_maximize = True

        value_ctx = state.to_prompt(
            target=target, metric_name=metric_name, maximize=is_maximize, language="python"
        )

        # Show construction length if available
        if state.construction:
            value_ctx += f"\nLength of the construction: {len(state.construction)}"

        if self.problem_type == "ac1":
            return get_ac1_prompt(self.budget_s, value_ctx)
        else:
            return get_ac2_prompt(self.budget_s, value_ctx, self.num_cpus)
    
    def _execute_code(self, code: str, state: InequalitiesState) -> tuple[Any, str]:
        """
        Execute the generated code in a sandboxed subprocess.
        Returns (result, error_msg).
        """
        # Preprocess code: inject verifier and construction
        import inspect
        if self.problem_type == "ac1":
            verifier_src = inspect.getsource(evaluate_sequence_ac1)
            # Add alias so generated code can call evaluate_sequence()
            verifier_src += "\n\n# Alias for generated code compatibility\nevaluate_sequence = evaluate_sequence_ac1\n"
        else:
            verifier_src = inspect.getsource(evaluate_sequence_ac2)
            verifier_src += "\n\n# Alias for generated code compatibility\nevaluate_sequence = evaluate_sequence_ac2\n"
        
        numpy_import = "import numpy as np"
        scipy_import = "from scipy.optimize import minimize"
        
        base = numpy_import + "\n" + scipy_import + "\n\n" + verifier_src + "\n\n"
        
        # Inject construction from state
        if state.construction is not None:
            sota_sequence = f"height_sequence_1 = np.array({list(state.construction)!r})"
            base += sota_sequence + "\n\n"
        
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
import pickle
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
    
    print(f"REWARD_RESULT: {{result}}")
    
except Exception as e:
    import io as _io
    _tb_buf = _io.StringIO()
    traceback.print_exc(file=_tb_buf)
    _tb_str = _tb_buf.getvalue()
    print(f"REWARD_ERROR: {{e}}")
    print(f"REWARD_ERROR_TYPE: {{type(e).__name__}}")
    print(f"REWARD_TRACEBACK: {{_tb_str}}")
'''
            f.write(runner_code)
        
        # Monitor RAM before starting new execution
        mem = psutil.virtual_memory()
        while mem.percent >= self.memory_threshold * 100:
            logger.warning(
                f"System memory usage is {mem.percent:.1f}%, "
                f"exceeds threshold {self.memory_threshold * 100:.0f}%. "
                f"Pausing execution of temp.py until memory drops..."
            )
            time.sleep(5)
            mem = psutil.virtual_memory()

        process = None
        try:
            # Run in subprocess with timeout
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
                error_msg = None
                error_type = None
                error_traceback = None
                for line in stdout.split("\n"):
                    if line.startswith("REWARD_RESULT:"):
                        result_str = line.split(":", 1)[1].strip()
                        try:
                            result = eval(result_str)
                            return result, ""
                        except:
                            return None, f"Failed to parse result: {result_str}"
                    elif line.startswith("REWARD_ERROR:"):
                        error_msg = line.split(":", 1)[1].strip()
                    elif line.startswith("REWARD_ERROR_TYPE:"):
                        error_type = line.split(":", 1)[1].strip()
                    elif line.startswith("REWARD_TRACEBACK:"):
                        error_traceback = line.split(":", 1)[1].strip()
                
                if error_msg is not None:
                    parts = [f"[{error_type}]" if error_type else "", error_msg]
                    if error_traceback:
                        parts.append(f"Traceback: {error_traceback[:600]}")
                    if stderr:
                        parts.append(f"Stderr: {stderr[:400]}")
                    full_error = " ".join([p for p in parts if p])
                    return None, full_error
                
                # No result found
                if process.returncode != 0:
                    # Detect memory-related crashes
                    is_memory_error = (
                        "MemoryError" in stderr
                        or process.returncode == -9   # SIGKILL (OOM killer)
                        or process.returncode == -11  # SIGSEGV (failed malloc)
                    )
                    if is_memory_error:
                        logger.warning(
                            f"Memory limit exceeded (max {self.max_memory_mb}MB) "
                            f"for subprocess PID {process.pid}. "
                            f"Return code: {process.returncode}. "
                            f"Stderr preview: {stderr[:200]!r}"
                        )
                        return None, f"Memory limit exceeded (max {self.max_memory_mb}MB)"
                    return None, f"Process failed: {stderr[:500]}"
                
                return None, "No result returned"
                
            except subprocess.TimeoutExpired:
                # Kill process tree
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except:
                    pass
                # CRITICAL: Must wait() to reap zombie process
                try:
                    process.wait(timeout=5)
                except:
                    pass
                return None, f"Timeout after {self.eval_timeout}s"
            
        finally:
            # CRITICAL: Ensure subprocess is always cleaned up to prevent zombie processes
            if process is not None:
                if process.poll() is None:
                    # Process is still running, kill it
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except:
                        pass
                # Always wait to reap zombie, even if process already terminated
                try:
                    process.wait(timeout=5)
                except:
                    pass
            
            # Cleanup temp files
            try:
                os.unlink(code_path)
                os.unlink(runner_path)
            except:
                pass
    
    def execute(self, code: str, state: InequalitiesState) -> EnvResult:
        """Execute the generated code and compute reward (synchronous version)."""
        try:
            output, error_msg = self._execute_code(code, state)
            
            if error_msg:
                is_timeout = "timeout" in error_msg.lower()
                is_memory_error = "memory limit exceeded" in error_msg.lower()
                if is_timeout:
                    fail_type = "timeout"
                elif is_memory_error:
                    fail_type = "memory_error"
                else:
                    fail_type = "execution_error"
                    # Log generated code snippet for debugging runtime errors
                    code_preview = "\n".join(code.splitlines()[:20])
                    logger.warning(
                        f"[ExecutionError] {error_msg}\n"
                        f"--- Generated code (first 20 lines) ---\n{code_preview}\n"
                        f"--- End preview ---"
                    )
                return EnvResult(
                    reward=0.0,
                    observation=error_msg,
                    is_valid=False,
                    fail_type=fail_type,
                    metadata={"timeout": is_timeout, "memory_error": is_memory_error, "error": error_msg},
                )
            
            # Verify and compute reward
            try:
                raw_score = self._evaluate(output)
                
                # Check for invalid results
                if self.problem_type == "ac1" and raw_score == np.inf:
                    return EnvResult(
                        reward=0.0,
                        observation="Invalid solution",
                        is_valid=False,
                        fail_type="execution_error",
                        metadata={"error": "Invalid solution"},
                    )
                elif self.problem_type == "ac2" and raw_score == -np.inf:
                    return EnvResult(
                        reward=0.0,
                        observation="Invalid solution",
                        is_valid=False,
                        fail_type="execution_error",
                        metadata={"error": "Invalid solution"},
                    )
                
                # Convert to reward (higher = better)
                if self.problem_type == "ac1":
                    reward = 1.0 / (1e-8 + raw_score)  # Reciprocal for minimization
                else:
                    reward = raw_score
                
                return EnvResult(
                    reward=reward,
                    observation="Success",
                    is_valid=True,
                    metadata={
                        "construction": output,
                        "raw_score": raw_score,
                    },
                )
                
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
                return EnvResult(
                    reward=0.0,
                    observation=str(e),
                    is_valid=False,
                    fail_type="execution_error",
                    metadata={"error": str(e)},
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
        parent_state: InequalitiesState,
        code: str,
        reward: float,
        result: EnvResult,
        timestep: int,
    ) -> InequalitiesState:
        """Create new InequalitiesState with parent tracking.

        Store the display metric in value for prompt/UCT compatibility,
        matching the latest discover convention.
        """
        construction = None
        raw_score = None
        if result.metadata:
            construction = result.metadata.get("construction")
            raw_score = result.metadata.get("raw_score")

        parent_values = []
        parents = []
        if parent_state.value is not None:
            parent_values.append(parent_state.value)
            parents.append({"id": parent_state.id, "timestep": parent_state.timestep})
        if parent_state.parent_values:
            parent_values.extend(parent_state.parent_values)
        if parent_state.parents:
            parents.extend(parent_state.parents)

        # Use display metric for state.value
        if self.problem_type == "ac1":
            display_value = -raw_score if raw_score is not None else reward
        elif self.problem_type == "ac2":
            display_value = raw_score if raw_score is not None else reward
        else:
            display_value = reward

        return InequalitiesState(
            timestep=timestep,
            construction=construction,
            code=code,
            value=display_value,
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
        elif fail_type == "memory_error":
            return EnvResult(
                reward=0.0,
                observation=f"Memory limit exceeded: {error_msg}" if error_msg else "Memory limit exceeded",
                is_valid=False,
                fail_type=fail_type,
                metadata={"memory_error": True, "max_memory_mb": self.max_memory_mb},
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


def create_initial_state_ac1(
    initial_exp_type: str = "best_available",
    budget_s: int = 1000,
    **kwargs
) -> InequalitiesState:
    """Create initial state for AC1."""
    timestep = -1
    rng = np.random.default_rng(12345)
    construction = [rng.random()] * rng.integers(1000, 8000)
    initial_bound = evaluate_sequence_ac1(construction)
    initial_value = -initial_bound

    if initial_exp_type == "best_available" or initial_exp_type == "random":
        code = "```python\n" + get_example_program_random_init(budget_s) + "\n```"
        return InequalitiesState(
            timestep=timestep,
            construction=construction,
            code=code,
            value=initial_value,
        )

    elif initial_exp_type == "none":
        code = "```python\n" + get_example_program_random_init(budget_s) + "\n```"
        return InequalitiesState(
            timestep=timestep,
            construction=None,
            code=code,
            value=0.0,
        )

    elif initial_exp_type == "random_no_code":
        return InequalitiesState(
            timestep=timestep,
            construction=construction,
            code="",
            value=initial_value,
        )

    else:
        raise ValueError(f"Unknown initial_exp_type: {initial_exp_type}")


# ============================================================================
# AC2 (AlphaEvolve AC2) Support - Lower Bound Maximization
# ============================================================================

AC2_LITERATURE = r"""A previous state of the art used the following approach. You can use it as inspiration, but you are not required to use it, and you are encoraged to explore.
```latex
Their procedure is a coarse-to-fine optimization of the score. It starts with a stochastic global search that repeatedly perturbs the current best candidate and keeps the perturbation whenever it improves (Q), with the perturbation scale gradually reduced over time. Once a good basin is found, they switch to a deterministic local improvement step, performing projected gradient ascent (move in the gradient direction and project back to the feasible region). To reach higher resolution, they lift a good low-resolution solution to a higher-dimensional one by simply repeating its entries and then rerun the local refinement. Iterating this explore–refine–upscale cycle yields their final high-resolution maximizer and the improved lower bound.
```"""

ae_verifier_program = '''
import numpy as np
def evaluate_sequence(sequence: list[float]) -> float:
    # Verify that the input is a list
    if not isinstance(sequence, list):
        raise ValueError("Invalid sequence type")

    # Reject empty lists
    if not sequence:
        raise ValueError("Empty sequence")

    # Check each element in the list for validity
    for x in sequence:
        # Reject boolean types (as they are a subclass of int) and
        # any other non-integer/non-float types (like strings or complex numbers).
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError("Invalid sequence element type")

        # Reject Not-a-Number (NaN) and infinity values.
        if np.isnan(x) or np.isinf(x):
            raise ValueError("Invalid sequence element value")

    # Convert all elements to float for consistency
    sequence = [float(x) for x in sequence]

    # Protect against negative numbers
    sequence = [max(0, x) for x in sequence]

    # Check if sum of sequence will be too close to zero
    if np.sum(sequence) < 0.01:
        raise ValueError("Sum of sequence is too close to zero.")
    
    # Protect against numbers that are too large
    sequence = [min(1000.0, x) for x in sequence]

    convolution_2 = np.convolve(sequence, sequence)
    # --- Security Checks ---

    # Calculate the 2-norm squared: ||f*f||_2^2
    num_points = len(convolution_2)
    x_points = np.linspace(-0.5, 0.5, num_points + 2)
    x_intervals = np.diff(x_points) # Width of each interval
    y_points = np.concatenate(([0], convolution_2, [0]))
    l2_norm_squared = 0.0
    for i in range(len(convolution_2) + 1):  # Iterate through intervals
        y1 = y_points[i]
        y2 = y_points[i+1]
        h = x_intervals[i]
        # Integral of (mx + c)^2 = h/3 * (y1^2 + y1*y2 + y2^2) where m = (y2-y1)/h, c = y1 - m*x1, interval is [x1, x2], y1 = mx1+c, y2=mx2+c
        interval_l2_squared = (h / 3) * (y1**2 + y1 * y2 + y2**2)
        l2_norm_squared += interval_l2_squared

    # Calculate the 1-norm: ||f*f||_1
    norm_1 = np.sum(np.abs(convolution_2)) / (len(convolution_2) + 1)

    # Calculate the infinity-norm: ||f*f||_inf
    norm_inf = np.max(np.abs(convolution_2))
    C_lower_bound = l2_norm_squared / (norm_1 * norm_inf)
    return C_lower_bound
'''

thetaevolve_initial_program_prev_init = '''
import numpy as np
from typing import Tuple
from tqdm import trange


def _simpson_l2sq(conv: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute ||f*f||_2^2 via Simpson-like piecewise-linear rule with endpoint zeros,
    and return its gradient w.r.t conv (same length as conv).

    l2_sq = sum_{i=0..M} (dx/3) * (y_i^2 + y_i*y_{i+1} + y_{i+1}^2),
    where y = [0, conv, 0], dx = 1/(M+1).

    d l2_sq / d y_j = (dx/3) * (4*y_j + y_{j-1} + y_{j+1})
    => restrict to indices j=1..M (these correspond to conv entries)
    """
    M = conv.size
    if M == 0:
        return 0.0, np.zeros_like(conv)

    dx = 1.0 / (M + 1)

    # pad endpoints with zeros
    y = np.empty(M + 2, dtype=conv.dtype)
    y[0] = 0.0
    y[1:-1] = conv
    y[-1] = 0.0

    # l2 value
    lhs = y[:-1]
    rhs = y[1:]
    l2_sq = (dx / 3.0) * np.sum(lhs * lhs + lhs * rhs + rhs * rhs)

    # gradient wrt conv (positions 1..M of y)
    # d/d y_j : (dx/3)*(4*y_j + y_{j-1} + y_{j+1})
    grad_y = (dx / 3.0) * (4.0 * y + np.roll(y, 1) + np.roll(y, -1))
    grad_conv = grad_y[1:-1]  # strip padding

    return float(l2_sq), grad_conv


def _l1(conv: np.ndarray) -> Tuple[float, np.ndarray]:
    """ ||f*f||_1 = dx * sum(conv); gradient is dx * ones """
    M = conv.size
    dx = 1.0 / (M + 1) if M > 0 else 1.0
    val = dx * float(np.sum(conv)) if M > 0 else 0.0
    grad = np.full_like(conv, dx)
    return val, grad


def _linf(conv: np.ndarray) -> Tuple[float, np.ndarray]:
    """ ||f*f||_inf = max(conv); subgradient: uniform over argmax set """
    if conv.size == 0:
        return 0.0, np.zeros_like(conv)
    m = float(np.max(conv))
    mask = (conv == m)
    count = int(mask.sum())
    if count == 0 or m <= 0.0:
        return m, np.zeros_like(conv)
    grad = mask.astype(conv.dtype) / count
    return m, grad


def _objective_and_grad_conv(conv: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute C = l2_sq / (l1 * linf) and gradient dC/d(conv) using quotient rule.
    """
    l2_sq, g_l2 = _simpson_l2sq(conv)
    l1, g_l1 = _l1(conv)
    linf, g_linf = _linf(conv)

    if l1 <= 0.0 or linf <= 0.0:
        return 0.0, np.zeros_like(conv)

    denom = l1 * linf
    C = l2_sq / denom

    # dC = (g_l2 * denom - l2_sq * (g_l1 * linf + l1 * g_linf)) / denom^2
    num_grad = g_l2 * denom - l2_sq * (g_l1 * linf + l1 * g_linf)
    g_conv = num_grad / (denom * denom)

    return float(C), g_conv


def _grad_h_from_conv_grad(h: np.ndarray, g_conv: np.ndarray) -> np.ndarray:
    """
    Given dC/d(conv) and conv = h * h (full convolution),
    dC/dh = 2 * (g_conv convolved with reverse(h)) in 'valid' mode (length N).
    """
    h_rev = h[::-1]
    # length(g_conv)=2N-1, length(h_rev)=N; 'valid' output length is N
    g_h = np.convolve(g_conv, h_rev, mode="valid")
    return 2.0 * g_h


class _Adam:
    """Lightweight Adam optimizer for numpy arrays (per-candidate)."""
    def __init__(self, shape, lr=3e-2, beta1=0.9, beta2=0.999, eps=1e-8, dtype=np.float32):
        self.m = np.zeros(shape, dtype=dtype)
        self.v = np.zeros(shape, dtype=dtype)
        self.t = 0
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.dtype = dtype

    def step(self, params, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * (grad * grad)
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return params + self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset_like(self, params):
        self.m[...] = 0.0
        self.v[...] = 0.0
        self.t = 0


def _batch_objective(h_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized evaluation over a batch of candidates.
    Returns:
        C_vals: (B,) objective values
        conv_grads: list/array of per-candidate dC/d(conv) for backprop
    """
    B, N = h_batch.shape
    C_vals = np.zeros(B, dtype=np.float32)
    conv_grads = [None] * B
    for b in range(B):
        h = np.clip(h_batch[b], 0.0, None)
        conv = np.convolve(h, h, mode="full")
        Cb, g_conv = _objective_and_grad_conv(conv)
        C_vals[b] = Cb
        conv_grads[b] = g_conv
    return C_vals, conv_grads


def _phase_update(h_batch, opt_list, lr, add_noise=False, t=0, eta=1e-3, gamma=0.55):
    """
    One optimization step for the whole batch.
    """
    B, N = h_batch.shape
    # compute dC/dh for each candidate
    C_vals, conv_grads = _batch_objective(h_batch)
    grads = np.zeros_like(h_batch, dtype=h_batch.dtype)
    for b in range(B):
        g_h = _grad_h_from_conv_grad(np.clip(h_batch[b], 0.0, None), conv_grads[b])
        grads[b] = g_h

    if add_noise:
        sigma = eta / ((t + 1) ** gamma)
        grads = grads + sigma * np.random.normal(size=grads.shape).astype(grads.dtype)

    # apply Adam update + project to nonnegativity
    for b in range(B):
        opt = opt_list[b]
        opt.lr = lr
        h_new = opt.step(h_batch[b], grads[b].astype(h_batch.dtype))
        h_batch[b] = np.clip(h_new, 0.0, None)

    return h_batch, C_vals


def _elitist_respawn(h_batch, C_vals, keep_frac, init_sampler, opt_list):
    """
    Keep top frac, respawn the rest with fresh random samples, reset their Adam states.
    """
    B = h_batch.shape[0]
    K = max(1, int(B * keep_frac))
    idx = np.argsort(C_vals)[-K:]  # top K
    survivors = h_batch[idx].copy()

    fresh = init_sampler(B - K)
    new_batch = np.concatenate([survivors, fresh], axis=0)

    # reorder optimizers to match new batch; reset the respawned ones
    new_opts = []
    for _ in range(K):
        new_opts.append(opt_list[idx[_]])  # keep state for survivors
    for _ in range(B - K):
        opt = _Adam(shape=h_batch.shape[1:], lr=opt_list[0].lr, dtype=h_batch.dtype)
        new_opts.append(opt)

    return new_batch, new_opts


def _upsample_1d(h: np.ndarray) -> np.ndarray:
    """Linear ×2 upsampling on [-1/4,1/4] grid (index-space)."""
    N = h.shape[0]
    x_old = np.linspace(-0.5, 0.5, N)
    x_new = np.linspace(-0.5, 0.5, 2 * N)
    return np.interp(x_new, x_old, h)


def _single_candidate_finetune(h0: np.ndarray, lr=3e-3, steps=50_000, log_every=2_000) -> Tuple[np.ndarray, float]:
    """
    Pure exploitation (no noise) on a single vector with Adam + projection.
    """
    h = h0.astype(np.float32).copy()
    opt = _Adam(h.shape, lr=lr, dtype=h.dtype)
    last_C = 0.0
    for t in trange(steps, desc="upsampled-finetune", leave=False):
        # objective and gradient
        conv = np.convolve(np.clip(h, 0.0, None), np.clip(h, 0.0, None), mode="full")
        C, g_conv = _objective_and_grad_conv(conv)
        g_h = _grad_h_from_conv_grad(np.clip(h, 0.0, None), g_conv)

        h = np.clip(opt.step(h, g_h.astype(h.dtype)), 0.0, None)
        last_C = C
        if log_every and (t + 1) % log_every == 0:
            pass  # tqdm
    return h, float(last_C)


def construct_function():
    """
    Use the paper's 4-phase gradient-based search to maximize
        R(f) = ||f*f||_2^2 / (||f*f||_1 * ||f*f||_inf).
    Returns (heights, r_value).
    """
    # ---------- Hyperparameters (close to paper, but conservatively smaller by default) ----------
    N = 256 # search resolution (paper: 768; will truncate zeros automatically)
    B = 64                 # batch size (paper used batch; you can raise this)
    ITER = 10_000          # total iterations
    EXPLORE_STEPS = 30_000 # phase split (paper: 30k)
    DROP_EVERY = 10_000    # respawn period (paper: 20k)
    KEEP_FRAC = 0.5        # keep top fraction (paper: kappa%)
    LR_EXPLORE = 3e-2      # Adam lr for exploration (paper)
    LR_EXPLOIT = 5e-3      # Adam lr for exploitation (paper)
    ETA, GAMMA = 1e-3, 0.55  # gradient noise schedule (paper used ~0.65; 0.55 is slightly gentler)
    dtype = np.float32

    h_prev_best = np.array(height_sequence_1, dtype=dtype)
    h_prev_best = np.clip(h_prev_best, 0.0, None)

    # resize if length doesn't match N
    if h_prev_best.shape[0] != N:
        x_old = np.linspace(-0.5, 0.5, h_prev_best.shape[0])
        x_new = np.linspace(-0.5, 0.5, N)
        h_prev_best = np.interp(x_new, x_old, h_prev_best).astype(dtype)

    rng = np.random.default_rng()

    def init_sampler(m):
        out = rng.uniform(0.0, 1.0, size=(m, N)).astype(dtype)
        if m > 0:
            out[0] = h_prev_best
        return out

    # init population and per-candidate Adam
    h_batch = init_sampler(B)
    opt_list = [_Adam(shape=(N,), lr=LR_EXPLORE, dtype=dtype) for _ in range(B)]
    best_h = h_batch.copy()
    best_C = np.full(B, -np.inf, dtype=dtype)

    for t in trange(ITER, desc="optimizing", leave=False):
        if t < EXPLORE_STEPS:
            # exploration phase: higher LR + noise
            h_batch, C_vals = _phase_update(
                h_batch, opt_list, lr=LR_EXPLORE, add_noise=True, t=t, eta=ETA, gamma=GAMMA
            )
        else:
            # exploitation phase: lower LR, no noise
            h_batch, C_vals = _phase_update(
                h_batch, opt_list, lr=LR_EXPLOIT, add_noise=False, t=t, eta=ETA, gamma=GAMMA
            )

        # update per-candidate bests
        improved = C_vals > best_C
        best_C = np.where(improved, C_vals, best_C)
        best_h[improved] = h_batch[improved]

        # periodic elitist respawn
        if (t + 1) % DROP_EVERY == 0:
            h_batch, opt_list = _elitist_respawn(
                h_batch, C_vals, keep_frac=KEEP_FRAC, init_sampler=init_sampler, opt_list=opt_list
            )

    # pick the best candidate
    idx = int(np.argmax(best_C))
    h_star = np.clip(best_h[idx].astype(np.float32), 0.0, None)

    # ---------- Phase 4: Upsampling + exploitation ----------
    # 2× upsample then fine-tune
    h_up1 = _upsample_1d(h_star)
    h_up1, _ = _single_candidate_finetune(h_up1, lr=3e-3, steps=40_000, log_every=2_000)

    h_up2 = _upsample_1d(h_up1)
    h_up2, _ = _single_candidate_finetune(h_up2, lr=3e-3, steps=40_000, log_every=2_000)

    heights = np.clip(h_up2, 0.0, None)
    r_value = evaluate_sequence(heights.tolist())
    print("This gets a C2 lower bound of", r_value)
    return heights.tolist()
'''


def get_ac2_prompt(budget_s: int, value_context: str, num_cpus: int = 2) -> str:
    """Generate AC2 prompt from template."""
    return f'''Act as an expert software developer and inequality specialist specializing in creating step functions with certain properties.

Your task is to generate the sequence of non-negative heights of a step functions, that maximizes the following evaluation function:

```python
{ae_verifier_program}
```

{AC2_LITERATURE}

Your task is to write a search function, `construct_function()`, that searches for the best sequence of coefficients. Your function will have {budget_s} seconds to run, and after that it has to have returned the best sequence it found. If after {budget_s} seconds it has not returned anything, it will be terminated with negative infinity points. All numbers in your sequence have to be positive or zero. Larger sequences with 1000s of items often have better attack surface, but too large sequences with 100s of thousands of items may be too slow to search.

You may code up any search method you want, and you are allowed to call the evaluate_sequence() function as many times as you want. You have access to it, you don't need to code up the evaluate_sequence() function.

{value_context}

You may want to start your search from one of the constructions we have found so far, which you can access through the 'height_sequence_1' global variable. 
However, you are encouraged to explore solutions that use other starting points to prevent getting stuck in a local minimum.

Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparemeters, etc. 
Unless you make a meaningful improvement, you will not be rewarded, if you are stuck you should think about how to get unstuck.

Rules:
- You must define the `construct_function` function as this is what will be invoked.
- You can use scientific libraries like scipy, numpy, cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS,ECOS], math.
- You can use up to {num_cpus} CPUs.
- Make all helper functions top level and have no closures from function nesting. Don't use any lambda functions.
- No filesystem or network IO.
- Do not import evaluate_sequence yourself. Assume it will already be imported and can be directly invoked. Do not import height_sequence_1 yourself; it will already be available.
- **Print statements**: Use `print()` to log progress, intermediate bounds, timing info, etc. Your output will be shown back to you.
- Include a short docstring at the top summarizing your algorithm.

Make sure to think and return the final program between ```python and ```.'''


def create_initial_state_ac2(
    initial_exp_type: str = "best_available",
    budget_s: int = 1000,
    **kwargs
) -> InequalitiesState:
    """Create initial state for AC2 (lower bound maximization)."""
    timestep = -1
    rng = np.random.default_rng(12345)
    construction = [rng.random() for _ in range(rng.integers(1000, 8000))]
    initial_value = evaluate_sequence_ac2(construction)

    if initial_exp_type == "best_available" or initial_exp_type == "random":
        code = "```python\n" + thetaevolve_initial_program_prev_init + "\n```"
        return InequalitiesState(
            timestep=timestep,
            construction=construction,
            code=code,
            value=initial_value,
        )

    elif initial_exp_type == "none":
        code = "```python\n# No initial code available. Start from scratch.\n```"
        return InequalitiesState(
            timestep=timestep,
            construction=None,
            code=code,
            value=0.0,
        )

    elif initial_exp_type == "random_no_code":
        return InequalitiesState(
            timestep=timestep,
            construction=construction,
            code="",
            value=initial_value,
        )

    else:
        raise ValueError(f"Unknown initial_exp_type: {initial_exp_type}")
