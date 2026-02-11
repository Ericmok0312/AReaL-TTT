import inspect
import numpy as np

from tasks.base_reward_task import BaseRewardTask
from tasks.alphaevolve_ac2.ae_verifier import evaluate_sequence
# from tasks.alphaevolve_ac2.sota_alphaevolve2 import height_sequence_1
from tasks.alphaevolve_ac.best_sequence_utils import get_best_sequence, try_save_best_sequence, get_best_bound_path



class AC2InequalitiesTask(BaseRewardTask):

    def get_function_name(self) -> str:
        return "construct_function"

    def preprocess_generation(self, generation, *, step, state=None, **kwargs) -> str:
        """Preprocess generation by adding verifier and injecting construction from state."""
        verifier_src = inspect.getsource(evaluate_sequence)
        numpy_import = "import numpy as np"
        
        base = numpy_import + "\n\n" + verifier_src + "\n\n"
        
        # State with construction is required - no silent fallback
        if state is None:
            raise ValueError(
                "state is required for preprocess_generation. "
                "Use ExperienceSampler to provide initial state with construction."
            )
        if state.construction is None:
            raise ValueError(
                "state.construction is required but was None. "
                "Ensure the sampled experience has a valid construction."
            )
        
        # Inject the construction as height_sequence_1
        sota_sequence = f"height_sequence_1 = np.array({list(state.construction)!r})"
        base += sota_sequence + "\n\n"
        
        return base + generation 

    def get_reward(self, result) -> float:
        return evaluate_sequence(result)

    def verify(self, result, *, step, **kwargs) -> bool:
        try:
            value = evaluate_sequence(result)
            if value == np.inf:
                return False
        except Exception:
            return False

        return True

    
if __name__ == "__main__":
    config = {
        "ttt_rm": {
            "num_cpus_per_task": 2,
            "rew_type": "scaled_reciprocal_cf",
            "fail_score": 0.0,
            "eval_timeout": 330,
            "worst_perf_log": -10000,
            "n_item": 26,
            "sweep_hyperparams": False,
            "max_hyperparam_combos": 8,
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

    task = AC2InequalitiesTask(config_ns)

    # This part isnt working right now
    # from tasks.alphaevolve_ac.ae_test_gen import TEST_GENERATION
    # generation = TEST_GENERATION

    
    @ray.remote
    def run_score(task, generation):
        return task.compute_score(generation)

    generations = [generation] * 1  # replace with your actual generations

    # Fire off 4 compute_score calls in parallel
    futures = [run_score.remote(task, g) for g in generations]
    # Gather results once all are done
    scores = ray.get(futures)
    # score = task.compute_score(generation)
    # print(scores)
    print([(score['performance'], score['msg']) for score in scores])
