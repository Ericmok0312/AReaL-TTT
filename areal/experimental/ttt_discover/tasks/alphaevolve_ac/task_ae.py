import inspect
import numpy as np

from tasks.base_reward_task import BaseRewardTask
from tasks.alphaevolve_ac.verifier_ae import evaluate_sequence


class ACInequalitiesTaskAE(BaseRewardTask):

    def get_function_name(self) -> str:
        return "propose_candidate"

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
