"""
Inequalities (AlphaEvolve AC1) environment for TTT-Discover.

Optimizes height sequences for inequalities problems.
"""

import sys
import os
import tempfile
from pathlib import Path

import numpy as np

# Note: If importing from tasks subdirectories is needed, add DISCOVER_DIR to sys.path:
# DISCOVER_DIR = os.path.dirname(os.path.dirname(__file__))
# if DISCOVER_DIR not in sys.path:
#     sys.path.insert(0, DISCOVER_DIR)

from areal.experimental.ttt_discover.envs.env import BaseEnv, EnvResult
from areal.experimental.ttt_discover.state import InequalitiesState
from areal.utils import logging

logger = logging.getLogger("InequalitiesEnv")


class InequalitiesEnv(BaseEnv):
    """
    Environment for inequalities (AC1) optimization.
    
    Task: Find height sequence that minimizes the inequality bound.
    """
    
    def __init__(self, budget_s: int = 1000, eval_timeout: int = 60, log_dir: str = "/tmp/ttt_logs"):
        """
        Args:
            budget_s: Budget parameter for the task
            eval_timeout: Timeout for code execution (seconds)
            log_dir: Directory for temporary code files
        """
        self.budget_s = budget_s
        self.eval_timeout = eval_timeout
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def get_prompt(self, state: InequalitiesState) -> str:
        """Generate prompt for inequalities task."""
        last_code = state.code if state.code else "# No previous code"
        
        value_context = ""
        if state.parent_values:
            avg_parent = sum(state.parent_values) / len(state.parent_values)
            value_context = f"\nParent average reward: {avg_parent:.4f}"
        
        prompt = f"""You are an expert mathematician specializing in combinatorial optimization and inequalities.

Your task is to find a height sequence that minimizes the inequality bound.

Here is the last code we ran:
```python
{last_code}
```

You are iteratively optimizing the construction.{value_context}

Reason about how you could further improve this:
- Can you find a better height sequence?
- Are there patterns or structures you can exploit?
- Can you use optimization techniques more effectively?

Rules:
- You must define the solve function: def solve() -> list[float]
- Returns a list of heights (the construction)
- You can use numpy, scipy, and other scientific libraries
- Make all helper functions top level and have no closures from function nesting
- No lambda functions
- No filesystem or network IO

Make sure to think step by step, then return the final program between ```python and ```.
"""
        return prompt
    
    def execute(self, code: str, state: InequalitiesState) -> EnvResult:
        """Execute the generated code and compute reward."""
        try:
            # Write to temp file
            code_to_run = f"import numpy as np\n\n{code}"
            
            with tempfile.NamedTemporaryFile(
                suffix=".py",
                delete=False,
                mode="w",
                dir=str(self.log_dir),
            ) as f:
                code_path = f.name
                f.write(code_to_run)
            
            try:
                # Execute in subprocess
                import subprocess
                
                result = subprocess.run(
                    [sys.executable, code_path],
                    capture_output=True,
                    text=True,
                    timeout=self.eval_timeout,
                )
                
                stdout = result.stdout
                stderr = result.stderr
                
                if result.returncode != 0:
                    return EnvResult(
                        reward=0.0,
                        observation=f"Execution failed: {stderr}",
                        is_valid=False,
                    )
                
                # Execute and get return value
                exec_globals = {"np": np}
                exec(code_to_run, exec_globals)
                
                if "solve" not in exec_globals:
                    return EnvResult(
                        reward=0.0,
                        observation="solve function not found",
                        is_valid=False,
                    )
                
                construction = exec_globals["solve"]()
                
                # Compute reward (to be minimized, so we negate)
                # This is a placeholder - actual reward depends on the specific inequality
                reward = self._compute_reward(construction)
                
                return EnvResult(
                    reward=reward,
                    observation=stdout,
                    is_valid=True,
                    metadata={"construction": construction},
                )
                
            finally:
                try:
                    os.unlink(code_path)
                except:
                    pass
                    
        except Exception as e:
            logger.warning(f"Exception in execute: {e}", exc_info=True)
            return EnvResult(
                reward=0.0,
                observation=str(e),
                is_valid=False,
            )
    
    def _compute_reward(self, construction: list) -> float:
        """Compute reward for a construction (placeholder)."""
        # Actual implementation depends on the specific inequality
        # For now, return a dummy reward
        return float(len(construction))
    
    def create_state(
        self,
        parent_state: InequalitiesState,
        code: str,
        reward: float,
        result: EnvResult,
        timestep: int,
    ) -> InequalitiesState:
        """Create new InequalitiesState from execution result."""
        construction = []
        if result.metadata and "construction" in result.metadata:
            construction = result.metadata["construction"]
        
        return InequalitiesState(
            timestep=timestep,
            construction=construction,
            code=code,
            value=reward,
            observation=result.observation,
        )
