import os
import pickle
import numpy as np
import scprep
import anndata
import scanpy as sc
import sklearn.metrics
from molecular_cross_validation.mcv_sweep import poisson_nll_loss

CACHE_DIR = os.path.expanduser("~/.cache/denoising_datasets")
os.makedirs(CACHE_DIR, exist_ok=True)

_DATASET_CACHE = {}

BASELINES = {
    "pancreas": {
        "baseline_mse": 0.304721,
        "baseline_poisson": 0.257575,
        "perfect_mse": 0.000000,
        "perfect_poisson": 0.031739,
    },
    "pbmc": {
        "baseline_mse": 0.270945,
        "baseline_poisson": 0.300447,
        "perfect_mse": 0.000000,
        "perfect_poisson": 0.043569,
    },
    "tabula": {
        "baseline_mse": 0.261763,
        "baseline_poisson": 0.206542,
        "perfect_mse": 0.000000,
        "perfect_poisson": 0.026961,
    },
}


def preprocess(X, normtype="sqrt", reverse_norm_order=True):
    if normtype == "sqrt":
        norm_fn = np.sqrt
        denorm_fn = np.square
    elif normtype == "log":
        norm_fn = np.log1p
        denorm_fn = np.expm1
    else:
        raise ValueError(f"Unknown normtype: {normtype}")

    if reverse_norm_order:
        X = scprep.utils.matrix_transform(X, norm_fn)
        X, libsize = scprep.normalize.library_size_normalize(X, rescale=1, return_library_size=True)
    else:
        X, libsize = scprep.normalize.library_size_normalize(X, rescale=1, return_library_size=True)
        X = scprep.utils.matrix_transform(X, norm_fn)
    return X, libsize, denorm_fn


def postprocess(Y, libsize, denorm_fn):
    Y = scprep.utils.matrix_transform(Y, denorm_fn)
    Y = scprep.utils.matrix_vector_elementwise_multiply(Y, libsize, axis=0)
    return Y


def evaluate_mse(test_data, denoised):
    """MSE metric - exactly as OpenProblems does it."""
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
    """Poisson loss metric - exactly as OpenProblems does it."""
    test_X = scprep.utils.toarray(test_data)
    denoised_X = np.asarray(denoised).copy()
    
    # Scale denoised to match test sum (OpenProblems scaling)
    initial_sum = train_data.sum()
    target_sum = test_X.sum()
    denoised_scaled = denoised_X * target_sum / initial_sum
    
    return poisson_nll_loss(test_X, denoised_scaled)


def normalize_score(score, worst, best):
    if worst == best:
        return 0.0
    return (worst - score) / (worst - best)


def run_denoising_eval(magic_denoise_fn, seed=42):
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
