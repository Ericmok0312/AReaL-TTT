import inspect
import numpy as np

from tasks.base_reward_task import BaseRewardTask
from tasks.erdos_min_overlap.verifier import evaluate_erdos_solution


class ErdosMinOverlapTask(BaseRewardTask):

    def __init__(self, config, log_path=""):
        super().__init__(config, log_path)

    def get_function_name(self) -> str:
        return "run"

    def preprocess_generation(self, generation, *, step, state=None, **kwargs) -> str:
        numpy_import = "import numpy as np"
        
        verifier_src = inspect.getsource(evaluate_erdos_solution)
        
        from tasks.erdos_min_overlap.verifier import verify_c5_solution
        verify_src = inspect.getsource(verify_c5_solution)
        
        base = (
            numpy_import + "\n\n" + 
            verify_src + "\n\n" +
            verifier_src + "\n\n"
        )
        
        if state is not None and hasattr(state, "h_values") and state.h_values is not None:
            initial_h_values = f"initial_h_values = np.array({list(state.h_values)!r})"
            base += initial_h_values + "\n\n"
        
        return base + generation

    def get_reward(self, result) -> float:
        h_values, c5_bound, n_points = result
        c5_bound = evaluate_erdos_solution(h_values, c5_bound, n_points)
        return float(1.0 / (1e-8 + c5_bound))

    def verify(self, result, *, step, **kwargs) -> bool:
        try:
            h_values, c5_bound, n_points = result
            c5_bound = evaluate_erdos_solution(h_values, c5_bound, n_points)
            if c5_bound <= 0 or np.isnan(c5_bound) or np.isinf(c5_bound):
                return False
        except Exception:
            return False
        return True

    
if __name__ == "__main__":
    config = {
        "ttt_rm": {
            "num_cpus_per_task": 2,
            "rew_type": "linear",
            "fail_score": 0.0,
            "eval_timeout": 330,
            "worst_perf_log": -10000,
            "n_item": 200,
            "sweep_hyperparams": False,
            "max_hyperparam_combos": 1,
        }
    }

    from types import SimpleNamespace
    import ray

    ray.init("auto")

    def dict_to_ns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
        return d

    config_ns = dict_to_ns(config)

    task = ErdosMinOverlapTask(config_ns)

    generation = """```python
import numpy as np

def run():
    n_points = 200
    h_values = np.ones(n_points) * 0.5
    dx = 2.0 / n_points
    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    c5_bound = np.max(correlation)
    return h_values, c5_bound, n_points
```"""
    
    @ray.remote
    def run_score(task, generation):
        return task.compute_score(generation, step=0)

    generations = [generation] * 1

    futures = [run_score.remote(task, g) for g in generations]
    scores = ray.get(futures)
    print([(score['performance'], score['msg']) for score in scores])

