"""
GPU Mode environment for TTT-Discover.

Optimizes GPU kernels (MLA decode, TriMul).
Uses tasks/gpu_mode/prompt_*.py for prompt templates.

NOTE: This environment requires Modal (modal.com) for GPU execution.
If Modal is not available, the environment will return mock results.
"""

import sys
import os
from pathlib import Path

import numpy as np

# Add parent directory to sys.path for gpu_mode imports
DISCOVER_DIR = os.path.dirname(os.path.dirname(__file__))
if DISCOVER_DIR not in sys.path:
    sys.path.insert(0, DISCOVER_DIR)

from areal.experimental.ttt_discover.envs.env import BaseEnv, EnvResult
from areal.experimental.ttt_discover.state import GpuModeState
from areal.utils import logging

logger = logging.getLogger("GpuModeEnv")



class GpuModeEnv(BaseEnv):
    """
    Environment for GPU kernel optimization.
    
    Tasks: MLA decode, TriMul
    
    NOTE: Requires Modal (modal.com) for actual GPU execution.
    Without Modal, returns mock results for testing.
    
    Example:
        >>> env = GpuModeEnv(task_name="mla_decode", gpu_type="H200")
        >>> state = create_initial_state_gpu_mode(task_name="mla_decode")
        >>> prompt = env.get_prompt(state)
    """
    
    def __init__(
        self,
        task_name: str = "mla_decode",  # or "trimul"
        gpu_type: str = "H200",
        eval_timeout: int = 300,
        score_scale: float = 3000.0,
        log_dir: str = "/tmp/ttt_logs",
    ):
        """
        Args:
            task_name: "mla_decode" or "trimul"
            gpu_type: GPU type for Modal execution (H100, H200, etc.)
            eval_timeout: Timeout for code execution (seconds)
            score_scale: Score scale for reward computation
            log_dir: Directory for logs
        """
        self.task_name = task_name
        self.gpu_type = gpu_type
        self.eval_timeout = eval_timeout
        self.score_scale = score_scale
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Import prompt templates
        if task_name == "mla_decode":
            from tasks.gpu_mode.prompt_mla_decode import MLA_DECODE_IMPROVEMENT_TEMPLATE_V1
            self._prompt_template = MLA_DECODE_IMPROVEMENT_TEMPLATE_V1
            self.target = 1700  # Human best is 1787.457μs (on MI300X)
        elif task_name == "trimul":
            from tasks.gpu_mode.prompt_trimul import TRIMUL_IMPROVEMENT_TEMPLATE_V0
            self._prompt_template = TRIMUL_IMPROVEMENT_TEMPLATE_V0
            self.target = 1000  # Human best is 1371.057μs
        else:
            raise ValueError(f"Unknown task_name: {task_name}")
        

    def get_prompt(self, state: GpuModeState) -> str:
        """
        Generate improvement prompt for the given state.
        """
        target = self.target
        
        # Build value context
        if state.value is not None and state.value < 1_000_000:
            current_runtime = -state.value  # value is negative latency
            value_ctx = (
                f"\nCurrent runtime (lower is better): {current_runtime:.4f} microseconds"
            )
            value_ctx += (
                f"\nTarget: {target} microseconds. "
                f"Current gap: {current_runtime - target:.4f} microseconds."
            )
        else:
            value_ctx = f"\nTarget runtime: {target} microseconds"
        
        # Build prompt
        prompt = self._prompt_template
        
        if state.code and state.code.strip():
            prompt = prompt.replace(
                "<<<LAST_CODE>>>",
                state.code
            )
        else:
            prompt = prompt.replace(
                "<<<LAST_CODE>>>",
                "# No previous attempt has been made."
            )
        
        prompt = prompt.replace("<<<VALUE_CONTEXT>>>", value_ctx)
        
        return prompt
    
    def execute(self, code: str, state: GpuModeState) -> EnvResult:
        """
        Execute code using Modal (GPU execution) or return mock results.
        """
        
        # Actual GPU execution via Modal
        try:
            import asyncio
            from tasks.gpu_mode.task import run_gpu_mode_task
            
            # Determine app_name based on task
            if self.task_name == "mla_decode":
                app_name = "discord-bot-runner-mla-decode-nvidia"
            elif self.task_name == "trimul":
                app_name = "discord-bot-runner"
            else:
                app_name = "discord-bot-runner"
            
            # Run on Modal
            out = asyncio.run(run_gpu_mode_task(
                submission_code=code,
                gpu_type=self.gpu_type,
                task_name=self.task_name,
                score_scale=self.score_scale,
                app_name=app_name,
            ))
            
            score = out.get("score", 0.0)
            msg = out.get("msg", "")
            correctness = out.get("correctness", 0.0)
            performance = out.get("performance", -1_000_000)
            
            # Combine msg and benchmark details for observation
            obs_parts = [msg]
            if out.get("benchmark_details"):
                obs_parts.append(
                    f"\nPer-benchmark timing:\n{out['benchmark_details']}"
                )
            observation = "\n".join(obs_parts)
            
            return EnvResult(
                reward=score,
                observation=observation,
                is_valid=correctness > 0,
                metadata={
                    "performance": performance,  # negative latency
                    "latency_us": -performance if performance else None,
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
        parent_state: GpuModeState,
        code: str,
        reward: float,
        result: EnvResult,
        timestep: int,
    ) -> GpuModeState:
        """
        Create new GpuModeState with proper parent tracking.
        """
        # Build parent tracking
        parent_values = []
        parents = []
        if parent_state.value is not None:
            parent_values.append(parent_state.value)
            parents.append({"id": parent_state.id, "timestep": parent_state.timestep})
        if parent_state.parent_values:
            parent_values.extend(parent_state.parent_values)
        if parent_state.parents:
            parents.extend(parent_state.parents)
        
        # Extract performance from metadata
        performance = None
        if result.metadata and "performance" in result.metadata:
            performance = result.metadata["performance"]
        
        return GpuModeState(
            timestep=timestep,
            code=code,
            value=performance if performance is not None else reward,
            parent_values=parent_values,
            parents=parents,
            observation=result.observation,
        )


def create_initial_state_gpu_mode(
    task_name: str = "mla_decode",
    initial_exp_type: str = "best_available",
    **kwargs
) -> GpuModeState:
    """
    Create initial state for GPU mode tasks.
    
    Args:
        task_name: "mla_decode" or "trimul"
        initial_exp_type: One of "best_available", "none"
        **kwargs: Additional arguments
    
    Returns:
        Initial GpuModeState
    """
    timestep = -1  # Initial states have timestep=-1
    
    if task_name == "mla_decode":
        if initial_exp_type == "best_available":
            from tasks.gpu_mode.initial_program_mla_decode import INITIAL_CODE, INITIAL_VALUE
            return GpuModeState(
                timestep=timestep,
                code=INITIAL_CODE,
                value=INITIAL_VALUE,
            )
        else:
            return GpuModeState(
                timestep=timestep,
                code="",
                value=-1_000_000,
            )
    
    elif task_name == "trimul":
        # No initial code or value for trimul
        return GpuModeState(
            timestep=timestep,
            code="",
            value=-1_000_000,
        )
    
    else:
        raise ValueError(f"Unknown task_name: {task_name}")
