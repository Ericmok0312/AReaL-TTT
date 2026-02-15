"""
Circle Packing environment for TTT-Discover.

Maximizes sum of radii for packing n circles in a unit square.
Uses tasks/alphaevolve_cp/prompt.py for prompt templates.

This implementation is compatible with AReaL's BaseEnv interface
and follows the same patterns as discover/tinker_cookbook/recipes/ttt/env_cp.py
"""

import sys
import os
import tempfile
from pathlib import Path

import numpy as np

# Add parent directory to sys.path for tasks imports
DISCOVER_DIR = os.path.dirname(os.path.dirname(__file__))
if DISCOVER_DIR not in sys.path:
    sys.path.insert(0, DISCOVER_DIR)

from areal.experimental.ttt_discover.envs.env import BaseEnv, EnvResult
from areal.experimental.ttt_discover.state import CirclePackingState
from areal.utils import logging

logger = logging.getLogger("CirclePackingEnv")


class CirclePackingEnv(BaseEnv):
    """
    Environment for circle packing optimization.
    
    Task: Pack n circles in [0,1]×[0,1] to maximize sum of radii.
    
    Example:
        >>> env = CirclePackingEnv(n_item=26, eval_timeout=60)
        >>> state = create_initial_state_cp(n=26, initial_exp_type="best_available")
        >>> prompt = env.get_prompt(state)
    """
    
    def __init__(
        self,
        n_item: int = 26,
        eval_timeout: int = 60,
        log_dir: str = "/tmp/ttt_logs",
    ):
        """
        Args:
            n_item: Number of circles to pack (26 or 32)
            eval_timeout: Timeout for code execution (seconds)
            log_dir: Directory for temporary code files
        """
        self.n_item = n_item
        self.eval_timeout = eval_timeout
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Import prompt template
        from tasks.alphaevolve_cp.prompt import CP_IMPROVEMENT_TEMPLATE
        self._prompt_template = CP_IMPROVEMENT_TEMPLATE
        
        # Import verifier
        from tasks.alphaevolve_cp.verifier import validate_packing
        self._validate = validate_packing
        
    def get_prompt(self, state: CirclePackingState) -> str:
        """
        Generate improvement prompt for the given state.
        
        This follows the same logic as discover's env_cp.py:_get_improvement_prompt
        """
        import inspect
        
        validator_src = inspect.getsource(self._validate)
        
        # Target values from discover
        target = 2.636 if self.n_item == 26 else 2.940
        
        # Get last code from state
        has_code = state.code and state.code.strip()
        
        # Value context: show before/after if we have parent values
        if state.parent_values and state.value is not None:
            before_sum = state.parent_values[0]
            after_sum = state.value
            value_ctx = (
                f"\nHere are the sum of radii before and after running the code above "
                f"(higher is better): {before_sum:.6f} -> {after_sum:.6f}"
            )
            value_ctx += (
                f"\nTarget: {target}. Current gap: {target - after_sum:.6f}. "
                f"Further improvements will also be generously rewarded."
            )
        elif state.value is not None:
            value_ctx = f"\nCurrent sum of radii (higher is better): {state.value:.6f}"
            value_ctx += (
                f"\nTarget: {target}. Current gap: {target - state.value:.6f}. "
                f"Further improvements will also be generously rewarded."
            )
        else:
            value_ctx = f"\nTarget sum of radii: {target}"
        
        # Show previous stdout if available
        if state.observation and state.observation.strip():
            stdout = state.observation.strip()
            if len(stdout) > 500:
                stdout = "\n\n\t\t ...(TRUNCATED)...\n" + stdout[-500:]
            value_ctx += f"\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"
        
        # Build prompt
        prompt = self._prompt_template
        prompt = prompt.replace("<<<N>>>", str(self.n_item))
        prompt = prompt.replace("<<<VALIDATOR_SRC>>>", validator_src)
        
        if has_code:
            prompt = prompt.replace("<<<LAST_CODE>>>", state.code)
        else:
            prompt = prompt.replace("<<<LAST_CODE>>>", "# No previous code available.")
        
        prompt = prompt.replace("<<<VALUE_CONTEXT>>>", value_ctx)
        
        return prompt
    
    def execute(self, code: str, state: CirclePackingState) -> EnvResult:
        """
        Execute the generated code and validate circle packing.
        
        This uses the same verification logic as discover's base_reward_task.py
        """
        try:
            # Import task for verification
            from tasks.alphaevolve_cp.task import CirclePackingTask
            
            # Create task instance
            class Config:
                pass
            
            config = Config()
            config.ttt_rm = {
                "n_item": self.n_item,
                "eval_timeout": self.eval_timeout,
                "num_cpus_per_task": 1,
            }
            
            task = CirclePackingTask(config, str(self.log_dir))
            
            # Run verification
            out = task.compute_score(code, step=0, state=state)
            
            # Convert result_construction to circles format
            circles = None
            raw_constr = out.get("result_construction")
            if raw_constr and len(raw_constr) >= 2:
                centers, radii = raw_constr[0], raw_constr[1]
                try:
                    circles = [
                        [float(centers[i][0]), float(centers[i][1]), float(radii[i])]
                        for i in range(len(radii))
                    ]
                except Exception:
                    circles = None
            
            return EnvResult(
                reward=out.get("score", 0.0),
                observation=out.get("stdout", ""),
                is_valid=out.get("correctness", 0.0) > 0,
                metadata={
                    "circles": circles,
                    "sum_radii": out.get("score", 0.0),
                    "msg": out.get("msg", ""),
                    "construction": raw_constr,
                },
            )
            
        except Exception as e:
            logger.warning(f"Exception in execute: {e}", exc_info=True)
            return EnvResult(
                reward=0.0,
                observation=str(e),
                is_valid=False,
                metadata={"error": str(e)},
            )
    
    def create_state(
        self,
        parent_state: CirclePackingState,
        code: str,
        reward: float,
        result: EnvResult,
        timestep: int,
    ) -> CirclePackingState:
        """
        Create new CirclePackingState with proper parent tracking.
        
        This follows the same pattern as discover's env_cp.py:_create_next_state
        """
        # Extract construction from metadata
        circles = None
        if result.metadata and "circles" in result.metadata:
            circles = result.metadata["circles"]
        
        # Build parent tracking (discover pattern)
        parent_values = []
        parents = []
        if parent_state.value is not None:
            parent_values.append(parent_state.value)
            parents.append({"id": parent_state.id, "timestep": parent_state.timestep})
        # Add ancestors
        if parent_state.parent_values:
            parent_values.extend(parent_state.parent_values)
        if parent_state.parents:
            parents.extend(parent_state.parents)
        
        return CirclePackingState(
            timestep=timestep,
            construction=circles,
            code=code,
            value=reward,
            parent_values=parent_values,
            parents=parents,
            observation=result.observation,
        )


def create_initial_state_cp(
    n: int = 26,
    initial_exp_type: str = "best_available",
    **kwargs
) -> CirclePackingState:
    """
    Create initial state for circle packing.
    
    This follows the same pattern as discover's sampler.py:create_initial_state
    
    Args:
        n: Number of circles (26 or 32)
        initial_exp_type: One of "best_available", "none", "random"
        **kwargs: Additional arguments
    
    Returns:
        Initial CirclePackingState
    """
    timestep = -1  # Initial states have timestep=-1
    
    if initial_exp_type == "best_available":
        # Load best available code
        if n == 26:
            from tasks.alphaevolve_cp.results.circle_packing_init_program import code as init_code
            from tasks.alphaevolve_cp.results.circle_packing_init_program import sum_radii as init_value
        elif n == 32:
            from tasks.alphaevolve_cp.results.new_our_best_32 import code as init_code
            from tasks.alphaevolve_cp.results.new_our_best_32 import sum_radii as init_value
        else:
            init_code = ""
            init_value = 0.0
        
        return CirclePackingState(
            timestep=timestep,
            construction=None,
            code=init_code,
            value=init_value,
        )
    
    elif initial_exp_type == "none":
        # Empty state
        return CirclePackingState(
            timestep=timestep,
            construction=None,
            code="",
            value=0.0,
        )
    
    elif initial_exp_type == "random":
        # Random initial state (simplified)
        return CirclePackingState(
            timestep=timestep,
            construction=None,
            code="# Random initialization",
            value=0.0,
        )
    
    else:
        raise ValueError(f"Unknown initial_exp_type: {initial_exp_type}")
