"""
Inequalities (AlphaEvolve AC1/AC2) environment for TTT-Discover.

Based on latest ttt-discover codebase - simplified architecture without task layer.
"""

import sys
import os
import tempfile
import subprocess
import pickle
import signal
import shutil
from pathlib import Path
from typing import Any

import numpy as np

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


def get_ac1_prompt(budget_s: int, last_code: str, value_context: str) -> str:
    """Generate AC1 prompt from template."""
    return f'''Act as an expert software developer and inequality specialist specializing in creating step functions with certain properties.

Your task is to generate the sequence of non-negative heights of a step function, that minimizes the following evaluation function:

{AC1_EVAL_FUNCTION}

{AC1_LITERATURE}

Your task is to write a search function that searches for the best sequence of coefficients. Your function will have {budget_s} seconds to run, and after that it has to have returned the best sequence it found. If after {budget_s} seconds it has not returned anything, it will be terminated with negative infinity points. All numbers in your sequence have to be positive or zero. Larger sequences with 1000s of items often have better attack surface, but too large sequences with 100s of thousands of items may be too slow to search.

You may code up any search method you want, and you are allowed to call the evaluate_sequence() function as many times as you want. You have access to it, you don't need to code up the evaluate_sequence() function.

Here is the last code we ran:
{last_code}

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
    ):
        self.problem_type = problem_type
        self.budget_s = budget_s
        self.eval_timeout = eval_timeout
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.num_cpus = num_cpus
        
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
        """Generate improvement prompt."""
        has_code = state.code and state.code.strip()
        
        # Value context
        if self.problem_type == "ac1":
            metric_name = "upper bound"
            target = 1.5030
            is_maximize = False
        else:
            metric_name = "lower bound"
            target = 0.97
            is_maximize = True
        
        # Build value context string
        # NOTE: state.value stores reward (1/bound for AC1, bound for AC2)
        # Prompt should show the actual metric (bound for AC1)
        def reward_to_bound(reward):
            """Convert reward back to bound for display."""
            if reward is None or reward <= 0:
                return float('inf')
            return 1.0 / reward
        
        if state.parent_values and state.value is not None:
            if is_maximize:
                # AC2: value is already the bound
                before_val = state.parent_values[0]
                after_val = state.value
            else:
                # AC1: value is 1/bound, convert back
                before_val = reward_to_bound(state.parent_values[0])
                after_val = reward_to_bound(state.value)
            value_ctx = (
                f"\nHere are the {metric_name}s before and after running the code above "
                f"({'higher' if is_maximize else 'lower'} is better): "
                f"{before_val:.6f} -> {after_val:.6f}"
            )
            value_ctx += f"\nTarget: {target}. Further improvements will be generously rewarded."
        elif state.value is not None:
            if is_maximize:
                current_val = state.value
            else:
                # AC1: convert reward back to bound
                current_val = reward_to_bound(state.value)
            value_ctx = f"\nCurrent {metric_name} ({'higher' if is_maximize else 'lower'} is better): {current_val:.6f}"
            value_ctx += f"\nTarget: {target}. Further improvements will be generously rewarded."
        else:
            value_ctx = f"\nOptimize the {metric_name} ({'higher' if is_maximize else 'lower'} is better). Target: {target}"
        
        # Show construction length if available
        if state.construction:
            value_ctx += f"\nLength of the construction: {len(state.construction)}"
        
        # Show previous stdout
        if state.observation and state.observation.strip():
            stdout = state.observation.strip()
            if len(stdout) > 500:
                stdout = "\n\n\t\t ...(TRUNCATED)...\n" + stdout[-500:]
            value_ctx += f"\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"
        
        # Build prompt
        last_code = state.code if has_code else "# No previous code available."
        
        if self.problem_type == "ac1":
            return get_ac1_prompt(self.budget_s, last_code, value_ctx)
        else:
            # AC2 prompt - can be added similarly
            raise NotImplementedError("AC2 prompt not yet implemented")
    
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
    print(f"REWARD_ERROR: {{e}}")
    traceback.print_exc()
'''
            f.write(runner_code)
        
        process = None
        try:
            # Run in subprocess with timeout
            env = os.environ.copy()
            t = str(max(1, self.num_cpus))
            env.setdefault("OMP_NUM_THREADS", t)
            env.setdefault("MKL_NUM_THREADS", t)
            env.setdefault("OPENBLAS_NUM_THREADS", t)
            env.setdefault("NUMEXPR_NUM_THREADS", t)
            
            process = subprocess.Popen(
                [sys.executable, runner_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                start_new_session=True,
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
                        except:
                            return None, f"Failed to parse result: {result_str}"
                    elif line.startswith("REWARD_ERROR:"):
                        error_msg = line.split(":", 1)[1].strip()
                        return None, error_msg
                
                # No result found
                if process.returncode != 0:
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
        """Execute the generated code and compute reward."""
        try:
            output, error_msg = self._execute_code(code, state)
            
            if error_msg:
                is_timeout = "timeout" in error_msg.lower()
                return EnvResult(
                    reward=0.0,
                    observation=error_msg,
                    is_valid=False,
                    fail_type="timeout" if is_timeout else "execution_error",
                    metadata={"timeout": is_timeout, "error": error_msg},
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
        """Create new InequalitiesState with parent tracking."""
        construction = None
        if result.metadata and "construction" in result.metadata:
            construction = result.metadata["construction"]
        
        parent_values = []
        parents = []
        if parent_state.value is not None:
            parent_values.append(parent_state.value)
            parents.append({"id": parent_state.id, "timestep": parent_state.timestep})
        if parent_state.parent_values:
            parent_values.extend(parent_state.parent_values)
        if parent_state.parents:
            parents.extend(parent_state.parents)
        
        return InequalitiesState(
            timestep=timestep,
            construction=construction,
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


def create_initial_state_ac1(
    initial_exp_type: str = "best_available",
    budget_s: int = 1000,
    **kwargs
) -> InequalitiesState:
    """Create initial state for AC1."""
    timestep = -1
    
    if initial_exp_type == "best_available":
        # Load SOTA sequence
        try:
            # Try to import from tasks if available
            from tasks.alphaevolve_ac.sota_alphaevolve2 import height_sequence_1
            construction = list(height_sequence_1)
        except ImportError:
            # Fallback: generate random construction
            rng = np.random.default_rng(12345)
            construction = [rng.random()] * rng.integers(1000, 8000)
        
        initial_bound = evaluate_sequence_ac1(construction)
        initial_value = 1.0 / (1e-8 + initial_bound)  # Reciprocal for consistency
        
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
    
    elif initial_exp_type == "random":
        rng = np.random.default_rng(12345)
        construction = [rng.random()] * rng.integers(1000, 8000)
        
        initial_bound = evaluate_sequence_ac1(construction)
        initial_value = 1.0 / (1e-8 + initial_bound)
        
        code = "```python\n" + get_example_program_random_init(budget_s) + "\n```"
        
        return InequalitiesState(
            timestep=timestep,
            construction=construction,
            code=code,
            value=initial_value,
        )
    
    elif initial_exp_type == "random_no_code":
        rng = np.random.default_rng(42)
        construction = list(rng.random(1000))
        
        initial_bound = evaluate_sequence_ac1(construction)
        initial_value = 1.0 / (1e-8 + initial_bound)
        
        return InequalitiesState(
            timestep=timestep,
            construction=construction,
            code="",
            value=initial_value,
        )
    
    else:
        raise ValueError(f"Unknown initial_exp_type: {initial_exp_type}")
