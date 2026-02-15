"""
TTT-Discover environments for AReaL.

Each environment wraps a TTT-Discover task and implements BaseEnv interface.
Uses prompt templates from tasks/ directory.

This module provides AReaL-compatible versions of the environments from
discover/tinker_cookbook/recipes/ttt/

Usage:
    >>> from areal.experimental.ttt_discover.envs import CirclePackingEnv
    >>> from areal.experimental.ttt_discover.envs import create_initial_state_cp
    >>> 
    >>> env = CirclePackingEnv(n_item=26, eval_timeout=60)
    >>> state = create_initial_state_cp(n=26, initial_exp_type="best_available")
    >>> prompt = env.get_prompt(state)
"""

from areal.experimental.ttt_discover.envs.circle_packing import (
    CirclePackingEnv,
    create_initial_state_cp,
)
from areal.experimental.ttt_discover.envs.gpu_mode import (
    GpuModeEnv,
    create_initial_state_gpu_mode,
)

__all__ = [
    "CirclePackingEnv",
    "create_initial_state_cp",
    "GpuModeEnv",
    "create_initial_state_gpu_mode",
]
