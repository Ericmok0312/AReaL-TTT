"""
Denoising environment for TTT-Discover.

Based on the original TTT-discover codebase - scRNA-seq denoising task.
Requires: openproblems, molecular_cross_validation, graphtools, scprep,
          anndata, scanpy, sklearn
"""

import sys
import os
import tempfile
import subprocess
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from areal.experimental.ttt_discover.envs.env import BaseEnv, EnvResult
from areal.experimental.ttt_discover.state import DenoisingState, State
from areal.utils import logging

import torch
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger("DenoisingEnv")


BASELINES = {
    "pancreas": {
        "baseline_mse": 0.304721,
        "baseline_poisson": 0.257575,
        "perfect_mse": 0.000000,
        "perfect_poisson": 0.031739,
    },
}


# =============================================================================
# Evaluation functions (injected into the sandbox)
# =============================================================================

def evaluate_mse(test_data, denoised):
    import numpy as np
    import scprep
    import anndata
    import scanpy as sc
    import sklearn.metrics

    test_X = scprep.utils.toarray(test_data).copy()
    denoised_X = np.asarray(denoised).copy()

    test_adata = anndata.AnnData(X=test_X)
    denoised_adata = anndata.AnnData(X=denoised_X)

    sc.pp.normalize_total(test_adata, target_sum=10000)
    sc.pp.log1p(test_adata)
    sc.pp.normalize_total(denoised_adata, target_sum=10000)
    sc.pp.log1p(denoised_adata)

    return sklearn.metrics.mean_squared_error(test_adata.X, denoised_adata.X)


def evaluate_poisson(train_data, test_data, denoised):
    import numpy as np
    import scprep
    from molecular_cross_validation.mcv_sweep import poisson_nll_loss

    test_X = scprep.utils.toarray(test_data)
    denoised_X = np.asarray(denoised).copy()

    initial_sum = train_data.sum()
    target_sum = test_X.sum()
    denoised_scaled = denoised_X * target_sum / initial_sum

    return poisson_nll_loss(test_X, denoised_scaled)


def run_denoising_eval(magic_denoise_fn, seed=42):
    import numpy as np
    import scprep
    import openproblems.data

    openproblems.data.no_cleanup()
    from openproblems.data.pancreas import load_pancreas
    from openproblems.tasks.denoising.datasets.utils import split_data

    adata = load_pancreas(test=False, keep_techs=["inDrop1"])
    adata = split_data(adata, seed=seed)

    X_train = scprep.utils.toarray(adata.obsm["train"])
    X_test = scprep.utils.toarray(adata.obsm["test"])

    Y_denoised = magic_denoise_fn(X_train, random_state=seed)

    if not np.isfinite(Y_denoised).all():
        return (np.inf, np.inf)
    if np.any(Y_denoised < 0):
        return (np.inf, np.inf)
    if Y_denoised.max() > X_train.sum():
        return (np.inf, np.inf)

    mse = evaluate_mse(X_test, Y_denoised)
    poisson = evaluate_poisson(X_train, X_test, Y_denoised)

    return (mse, poisson)


def magic_denoise(X, knn=5, t=3, n_pca=100, solver="approximate", decay=1, knn_max=None, random_state=None, n_jobs=1, verbose=False):
    import numpy as np
    import graphtools
    import scprep

    if knn_max is None:
        knn_max = knn * 3

    X_work = scprep.utils.toarray(X).astype(np.float64)
    X_work = np.sqrt(X_work)
    X_work, libsize = scprep.normalize.library_size_normalize(X_work, rescale=1, return_library_size=True)

    graph = graphtools.Graph(
        X_work,
        n_pca=n_pca if X_work.shape[1] > n_pca else None,
        knn=knn,
        knn_max=knn_max,
        decay=decay,
        thresh=1e-4,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=0,
    )

    diff_op = graph.diff_op

    if solver == "approximate":
        data = graph.data_nu
    else:
        data = scprep.utils.to_array_or_spmatrix(graph.data)

    if verbose:
        print(f"    [magic_denoise] data shape: {data.shape}, sum: {data.sum():.6f}")
        print(f"    [magic_denoise] diff_op sum: {diff_op.sum():.6f}")

    data_imputed = scprep.utils.toarray(data)

    if t > 0 and diff_op.shape[1] < data_imputed.shape[1]:
        diff_op_t = np.linalg.matrix_power(scprep.utils.toarray(diff_op), t)
        data_imputed = diff_op_t.dot(data_imputed)
        if verbose:
            print(f"    [magic_denoise] used matrix_power path")
    else:
        for _ in range(t):
            data_imputed = diff_op.dot(data_imputed)
        if verbose:
            print(f"    [magic_denoise] used iteration path")

    if verbose:
        print(f"    [magic_denoise] after diffusion sum: {data_imputed.sum():.6f}")

    if solver == "approximate":
        data_imputed = graph.inverse_transform(data_imputed, columns=None)
        if verbose:
            print(f"    [magic_denoise] after inverse_transform sum: {data_imputed.sum():.6f}")

    data_imputed = np.square(data_imputed)
    data_imputed = scprep.utils.matrix_vector_elementwise_multiply(data_imputed, libsize, axis=0)

    return data_imputed


# =============================================================================
# Prompt template
# =============================================================================

SYSTEM_PROMPT = '''You are an expert in computational biology and single-cell RNA-seq analysis.
Your task is to develop a denoising algorithm for scRNA-seq count data. You are experienced in
compuational biology libraries and tools and are familiar with problems in denoising in the single-cell field.

## Problem

Single-cell RNA-seq data is noisy due to technical dropout and low capture efficiency.
Given noisy count data, predict the true expression levels.

Your prediction is evaluated against held-out molecules using two metrics:
1. **MSE** - Mean Squared Error in log-normalized space
2. **Poisson Loss** - Poisson negative log-likelihood

You need to implement a novel denoising algorithm that outperforms the current state-of-the-art without overfitting.

## Data Format

- Input `X`: numpy array of shape (n_cells, n_genes) - **raw count data**
- Output: numpy array of same shape - your denoised counts

## Evaluation

Your output is evaluated using these exact functions:

```python
<<<EVALUATE_MSE_FUNC>>>
```

```python
<<<EVALUATE_POISSON_FUNC>>>
```

## Scoring

**Poisson is a HARD CONSTRAINT.** Your solution is REJECTED if `poisson_norm < 0.97`.
- `poisson_norm = (0.257575 - poisson) / (0.257575 - 0.031739)`
- MAGIC baseline achieves ≈0.97

**Reward = MSE score only** (after passing Poisson constraint).

## Budget & Resources

- **Time budget**: 400s for your code to run. You should time your code and make sure it runs within the time budget.
- **CPUs**: 2 available

## Function Signature to return

```python
def magic_denoise(X, **kwargs):
    # kwargs may include: budget_s, random_state, knn, t, n_pca, solver, decay, knn_max, n_jobs
    # You can add your own parameters too
    # Your implementation
    return denoised_X  # same shape as X
```

## Rules

- Implement `magic_denoise(X, ...)` that returns denoised data
- Use numpy, scipy, sklearn, graphtools, scprep, scanpy
- Make all helper functions top level, no closures or lambdas
- No filesystem or network IO

## Key Insights from Benchmarks

- NORMALIZATION ORDER MATTERS: Denoise raw/log counts first, then normalize. "Reversed normalization order" achieves Poisson ~0.98 vs ~0.55 for standard order.
- Square root transform is variance-stabilizing for Poisson distributions
- Poisson loss is highly affected by low non-zero values - push values < 1 toward zero
- The original MAGIC with reversed normalization achieves best results
'''


# =============================================================================
# Environment class
# =============================================================================

class DenoisingEnv(BaseEnv):
    """
    Environment for scRNA-seq denoising (TTT-Discover).
    """

    def __init__(
        self,
        eval_timeout: int = 530,
        log_dir: str = "/tmp/ttt_logs",
        num_cpus: int = 2,
        memory_threshold: float = 0.60,
    ):
        self.eval_timeout = eval_timeout
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.num_cpus = num_cpus
        self.memory_threshold = memory_threshold
        self.is_maximize = False
        self.entrypoint = "run_denoising"

    def get_prompt(self, state: DenoisingState) -> str:
        """Generate improvement prompt using state.to_prompt."""
        import inspect

        evaluate_mse_src = inspect.getsource(evaluate_mse)
        evaluate_poisson_src = inspect.getsource(evaluate_poisson)

        prompt = SYSTEM_PROMPT
        prompt = prompt.replace("<<<EVALUATE_MSE_FUNC>>>", evaluate_mse_src)
        prompt = prompt.replace("<<<EVALUATE_POISSON_FUNC>>>", evaluate_poisson_src)

        value_ctx = state.to_prompt(
            target=0.0, metric_name="MSE", maximize=False, language="python"
        )

        # Append current metrics
        if state.mse is not None or state.poisson is not None:
            metrics = []
            if state.mse is not None:
                metrics.append(f"MSE: {state.mse:.6f}")
            if state.poisson is not None:
                metrics.append(f"Poisson: {state.poisson:.6f}")
            value_ctx += f"\nCurrent metrics (lower is better): {', '.join(metrics)}"

        has_code = state.code and state.code.strip()
        if has_code:
            clean_code = state.code.strip()
            if clean_code.startswith("```python"):
                clean_code = clean_code[len("```python"):].strip()
            if clean_code.startswith("```"):
                clean_code = clean_code[3:].strip()
            if clean_code.endswith("```"):
                clean_code = clean_code[:-3].strip()
            code_section = f"""
Here is the current implementation:
```python
{clean_code}
```

You are iteratively improving the denoising algorithm.{value_ctx}

Reason about how you could improve this approach.
"""
        else:
            code_section = f"""
{value_ctx}

Write code to implement a denoising algorithm.
"""

        return f"""{prompt}
{code_section}
Write your improved `magic_denoise` function."""

    def _execute_code(self, code: str, state: DenoisingState) -> tuple[Any, str]:
        """
        Execute the generated code in a sandboxed subprocess.
        Returns (result, error_msg).
        """
        import inspect

        # Preprocess code: inject verifier and wrapper
        evaluate_mse_src = inspect.getsource(evaluate_mse)
        evaluate_poisson_src = inspect.getsource(evaluate_poisson)
        run_denoising_eval_src = inspect.getsource(run_denoising_eval)

        imports = """import numpy as np
import scipy
import scipy.sparse
from scipy import linalg
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
import graphtools
import scprep
import anndata
import scanpy as sc
import sklearn.metrics
import math
import random
from molecular_cross_validation.mcv_sweep import poisson_nll_loss

_SEED = 42
"""

        wrapper = """
def run_denoising():
    return run_denoising_eval(magic_denoise, seed=_SEED)
"""

        full_code = (
            imports
            + "\n\n"
            + evaluate_mse_src
            + "\n\n"
            + evaluate_poisson_src
            + "\n\n"
            + run_denoising_eval_src
            + "\n\n"
            + code
            + "\n\n"
            + wrapper
        )

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
                        except Exception:
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
                except Exception:
                    pass
                # Must wait() to reap zombie process
                try:
                    process.wait(timeout=5)
                except Exception:
                    pass
                return None, f"Timeout after {self.eval_timeout}s"

        finally:
            # Ensure subprocess is always cleaned up
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

            # Cleanup temp files
            try:
                os.unlink(code_path)
                os.unlink(runner_path)
            except Exception:
                pass

    def execute(self, code: str, state: DenoisingState) -> EnvResult:
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

            # Verify output format
            if not isinstance(output, (list, tuple)) or len(output) < 2:
                return EnvResult(
                    reward=0.0,
                    observation=f"Invalid output format: {output}",
                    is_valid=False,
                    fail_type="execution_error",
                    metadata={"error": f"Invalid output format: {output}"},
                )

            mse, poisson = output[0], output[1]

            if not np.isfinite(mse) or not np.isfinite(poisson):
                return EnvResult(
                    reward=0.0,
                    observation="Non-finite metrics returned",
                    is_valid=False,
                    fail_type="execution_error",
                    metadata={"error": "Non-finite metrics"},
                )

            # Hard constraint: poisson_norm >= 0.97
            baseline = BASELINES["pancreas"]
            poisson_range = baseline["baseline_poisson"] - baseline["perfect_poisson"]
            poisson_norm = (baseline["baseline_poisson"] - poisson) / poisson_range if poisson_range > 0 else 0.0
            poisson_norm = max(0.0, min(1.0, poisson_norm))

            if poisson_norm < 0.97 or poisson < baseline["perfect_poisson"]:
                return EnvResult(
                    reward=0.0,
                    observation=f"Poisson constraint not met: poisson={poisson:.6f}, norm={poisson_norm:.4f}",
                    is_valid=False,
                    fail_type="execution_error",
                    metadata={"error": "Poisson constraint not met", "mse": mse, "poisson": poisson, "poisson_norm": poisson_norm},
                )

            # Reward = 1 / mse (higher is better)
            reward = 1.0 / (1e-12 + mse)

            mse_range = baseline["baseline_mse"] - baseline["perfect_mse"]
            mse_normalized = (baseline["baseline_mse"] - mse) / mse_range if mse_range > 0 else 0.0
            mse_normalized = max(0.0, min(1.0, mse_normalized))

            return EnvResult(
                reward=reward,
                observation="Success",
                is_valid=True,
                metadata={
                    "mse": mse,
                    "poisson": poisson,
                    "mse_normalized": mse_normalized,
                    "poisson_normalized": poisson_norm,
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
        parent_state: DenoisingState,
        code: str,
        reward: float,
        result: EnvResult,
        timestep: int,
    ) -> DenoisingState:
        """Create new DenoisingState with parent tracking."""
        mse = None
        poisson = None
        if result.metadata:
            mse = result.metadata.get("mse")
            poisson = result.metadata.get("poisson")

        parent_values = []
        parents = []
        if parent_state.value is not None:
            parent_values.append(parent_state.value)
            parents.append({"id": parent_state.id, "timestep": parent_state.timestep})
        if parent_state.parent_values:
            parent_values.extend(parent_state.parent_values)
        if parent_state.parents:
            parents.extend(parent_state.parents)

        # Display value: higher = better, so we store -mse for minimization
        display_value = -mse if mse is not None else reward

        return DenoisingState(
            timestep=timestep,
            code=code,
            value=display_value,
            mse=mse,
            poisson=poisson,
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


def create_initial_state_denoising(
    initial_exp_type: str = "best_available",
    **kwargs
) -> DenoisingState:
    """Create initial state for denoising."""
    import inspect

    magic_func_src = inspect.getsource(magic_denoise)
    code = "```python\n" + magic_func_src + "\n```"

    # Initial metrics from MAGIC baseline on pancreas
    initial_mse = 0.2316
    initial_poisson = 0.0370
    initial_value = -initial_mse

    if initial_exp_type in ("best_available", "random"):
        return DenoisingState(
            timestep=-1,
            code=code,
            value=initial_value,
            mse=initial_mse,
            poisson=initial_poisson,
        )
    elif initial_exp_type == "none":
        return DenoisingState(
            timestep=-1,
            code=code,
            value=0.0,
            mse=None,
            poisson=None,
        )
    elif initial_exp_type == "random_no_code":
        return DenoisingState(
            timestep=-1,
            code="",
            value=initial_value,
            mse=initial_mse,
            poisson=initial_poisson,
        )
    else:
        raise ValueError(f"Unknown initial_exp_type: {initial_exp_type}")
