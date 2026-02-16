import inspect
import re
import numpy as np

from tasks.base_reward_task import BaseRewardTask
from tasks.alphaevolve_cp.verifier import validate_packing


class CirclePackingTask(BaseRewardTask):

    def get_function_name(self) -> str:
        return "run_packing"
    
    def _extract_code(self, response: str) -> str | None:
        """Extract Python code with fallback for raw code without markdown."""
        # Try markdown code block first
        m = re.search(r"```python\s+([\s\S]*?)\s*```", response)
        if m is not None:
            return m.group(1).strip()
        
        # Fallback: look for any code block
        m = re.search(r"```\s*([\s\S]*?)\s*```", response)
        if m is not None:
            return m.group(1).strip()
        
        # Fallback: try to find function definitions in raw text
        m = re.search(r"(def\s+\w+\s*\([^)]*\):[\s\S]*)", response)
        if m is not None:
            return m.group(1).strip()
        
        # Last resort: if response looks like code
        if any(kw in response for kw in ["def ", "import ", "return ", "class "]):
            return response.strip()
        
        return None

    def preprocess_generation(self, generation, *args, **kwargs) -> str:
        """Inject validate_packing and common imports into the code so it's available if needed."""
        verifier_src = inspect.getsource(validate_packing)
        numpy_import = "import numpy as np"
        scipy_import = "from scipy.optimize import minimize"
        return numpy_import + "\n" + scipy_import + "\n\n" + verifier_src + "\n\n" + generation

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
