import inspect
import numpy as np
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "openproblems"))

from tasks.base_reward_task import BaseRewardTask
from tasks.denoising.verifier import evaluate_mse, evaluate_poisson, run_denoising_eval, BASELINES
from tasks.denoising.magic_function import magic_denoise

MAGIC_FUNC = inspect.getsource(magic_denoise)


class DenoisingTask(BaseRewardTask):

    def __init__(self, config, log_path="", seed=42):
        super().__init__(config, log_path)
        self.seed = seed

    def get_function_name(self) -> str:
        return "run_denoising"

    def preprocess_generation(self, generation, *, step, state=None, **kwargs) -> str:
        imports = f"""import numpy as np
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
import sys
import os
from molecular_cross_validation.mcv_sweep import poisson_nll_loss

sys.path.insert(0, "{ROOT}")
sys.path.insert(0, "{os.path.join(ROOT, 'openproblems')}")

_SEED = {self.seed}
"""

        evaluate_mse_src = inspect.getsource(evaluate_mse)
        evaluate_poisson_src = inspect.getsource(evaluate_poisson)
        run_denoising_eval_src = inspect.getsource(run_denoising_eval)

        wrapper = """
def run_denoising():
    return run_denoising_eval(magic_denoise, seed=_SEED)
"""

        return (
            imports + "\n\n" +
            evaluate_mse_src + "\n\n" +
            evaluate_poisson_src + "\n\n" +
            run_denoising_eval_src + "\n\n" +
            generation + "\n\n" +
            wrapper
        )

    def verify(self, result, *, step, **kwargs) -> bool:
        if not isinstance(result, (list, tuple)) or len(result) < 2:
            return False
        mse, poisson = result[0], result[1]
        if not np.isfinite(mse) or not np.isfinite(poisson):
            return False
        baseline = BASELINES["pancreas"]
        if poisson < baseline["perfect_poisson"]:
            return False
        poisson_range = baseline["baseline_poisson"] - baseline["perfect_poisson"]
        poisson_norm = (baseline["baseline_poisson"] - poisson) / poisson_range if poisson_range > 0 else 0
        if poisson_norm < 0.97:
            return False
        return True

    def get_reward(self, result, dataset="pancreas") -> float:
        baseline = BASELINES[dataset]
        mse = result[0]
        
        mse_range = baseline["baseline_mse"] - baseline["perfect_mse"]
        mse_norm = (baseline["baseline_mse"] - mse) / mse_range if mse_range > 0 else 0
        mse_norm = max(0.0, min(1.0, mse_norm))
        
        return float(mse_norm)


EVALUATE_MSE_FUNC = inspect.getsource(evaluate_mse)
EVALUATE_POISSON_FUNC = inspect.getsource(evaluate_poisson)
