"""
Erdos Min Overlap environment for TTT-Discover.

Minimizes overlap in C5 graph constructions.
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
from areal.experimental.ttt_discover.state import ErdosState
from areal.utils import logging

logger = logging.getLogger("ErdosEnv")


class ErdosEnv(BaseEnv):
    """
    Environment for Erdos minimum overlap problem.
    
    Task: Find construction that minimizes C5 overlap bound.
    """
    
    def __init__(self, n: int = 100, eval_timeout: int = 60, log_dir: str = "/tmp/ttt_logs"):
        """
        Args:
            n: Size parameter for the construction
            eval_timeout: Timeout for code execution (seconds)
            log_dir: Directory for temporary code files
        """
        self.n = n
        self.eval_timeout = eval_timeout
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def get_prompt(self, state: ErdosState) -> str:
        """Generate prompt for Erdos problem."""
        last_code = state.code if state.code else "# No previous code"
        
        value_context = ""
        if state.parent_values:
            avg_parent = sum(state.parent_values) / len(state.parent_values)
            value_context = f"\nParent average C5 bound: {avg_parent:.4f}"
        
        prompt = f"""You are an expert mathematician specializing in graph theory and combinatorial optimization.

Your task is to find a construction that minimizes the C5 overlap bound for n={self.n}.

Here is the last code we ran:
```python
{last_code}
```

You are iteratively optimizing the construction.{value_context}

Reason about how you could further improve:
- Can you find a better pattern for the construction?
- Are there symmetries or structures you can exploit?
- Can you use optimization techniques more effectively?

Rules:
- You must define the solve function: def solve() -> tuple[list[float], float]
- Returns (h_values, c5_bound) where h_values is a list of length {self.n}
- You can use numpy, scipy, and other scientific libraries
- Make all helper functions top level and have no closures from function nesting
- No lambda functions
- No filesystem or network IO

Make sure to think step by step, then return the final program between ```python and ```.
"""
        return prompt
    
    def execute(self, code: str, state: ErdosState) -> EnvResult:
        """Execute the generated code and compute reward."""
        try:
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
                
                h_values, c5_bound = exec_globals["solve"]()
                
                # Validate construction length
                if len(h_values) != self.n:
                    return EnvResult(
                        reward=0.0,
                        observation=f"Invalid construction length: expected {self.n}, got {len(h_values)}",
                        is_valid=False,
                    )
                
                # Reward is negative C5 bound (we want to minimize)
                reward = -c5_bound
                
                return EnvResult(
                    reward=reward,
                    observation=stdout,
                    is_valid=True,
                    metadata={
                        "construction": h_values,
                        "c5_bound": c5_bound,
                    },
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
    
    def create_state(
        self,
        parent_state: ErdosState,
        code: str,
        reward: float,
        result: EnvResult,
        timestep: int,
    ) -> ErdosState:
        """Create new ErdosState from execution result."""
        c5_bound = None
        construction = None
        
        if result.metadata:
            c5_bound = result.metadata.get("c5_bound")
            construction = result.metadata.get("construction")
        
        return ErdosState(
            timestep=timestep,
            code=code,
            value=reward,
            c5_bound=c5_bound,
            construction=construction,
            observation=result.observation,
        )
