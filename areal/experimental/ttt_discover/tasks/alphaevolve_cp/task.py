import inspect
import numpy as np

from tasks.base_reward_task import BaseRewardTask
from tasks.alphaevolve_cp.verifier import validate_packing


class CirclePackingTask(BaseRewardTask):

    def get_function_name(self) -> str:
        return "run_packing"

    def preprocess_generation(self, generation, *args, **kwargs) -> str:
        """Inject validate_packing into the code so it's available if needed."""
        verifier_src = inspect.getsource(validate_packing)
        numpy_import = "import numpy as np"
        return numpy_import + "\n\n" + verifier_src + "\n\n" + generation

    def get_reward(self, result) -> float:
        centers, radii, _ = result
        
        if not isinstance(centers, np.ndarray):
            centers = np.array(centers)
        if not isinstance(radii, np.ndarray):
            radii = np.array(radii)

        return np.sum(radii)

    def verify(self, result, *args, **kwargs) -> bool:

        centers, radii, _ = result
        
        if not isinstance(centers, np.ndarray):
            centers = np.array(centers)
        if not isinstance(radii, np.ndarray):
            radii = np.array(radii)

        shape_valid = centers.shape == (self.n_item, 2) and radii.shape == (self.n_item,)
        if not shape_valid:
            return False

        return validate_packing(centers, radii)
