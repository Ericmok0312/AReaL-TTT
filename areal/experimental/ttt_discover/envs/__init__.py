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
    >>> env = CirclePackingEnv(n_item=26, eval_timeout=300)
    >>> state = create_initial_state_cp(n=26, initial_exp_type="best_available")
    >>> prompt = env.get_prompt(state)
    
    >>> # For AC1 (Inequalities)
    >>> from areal.experimental.ttt_discover.envs import InequalitiesEnv
    >>> from areal.experimental.ttt_discover.envs import create_initial_state_ac1
    >>> 
    >>> env = InequalitiesEnv(problem_type="ac1", budget_s=1000, eval_timeout=600)
    >>> state = create_initial_state_ac1(initial_exp_type="best_available")
    >>> prompt = env.get_prompt(state)
"""

from areal.experimental.ttt_discover.envs.circle_packing import (
    CirclePackingEnv,
    create_initial_state_cp,
)

from areal.experimental.ttt_discover.envs.inequalities import (
    InequalitiesEnv,
    create_initial_state_ac1,
    evaluate_sequence_ac1,
    evaluate_sequence_ac2,
)

from areal.experimental.ttt_discover.envs.erdos import (
    ErdosEnv,
    create_initial_state_erdos,
)

from areal.experimental.ttt_discover.envs.denoising import (
    DenoisingEnv,
    create_initial_state_denoising,
)

from areal.experimental.ttt_discover.envs.ale_bench import (
    AleBenchEnv,
    create_initial_state_ale_bench,
)

# GpuModeEnv is optional and requires additional dependencies
try:
    from areal.experimental.ttt_discover.envs.gpu_mode import (
        GpuModeEnv,
        create_initial_state_gpu_mode,
    )
    _GPU_MODE_AVAILABLE = True
except ImportError:
    _GPU_MODE_AVAILABLE = False
    GpuModeEnv = None  # type: ignore
    create_initial_state_gpu_mode = None  # type: ignore

__all__ = [
    # Circle Packing
    "CirclePackingEnv",
    "create_initial_state_cp",
    # AC1/AC2 (Inequalities)
    "InequalitiesEnv",
    "create_initial_state_ac1",
    "evaluate_sequence_ac1",
    "evaluate_sequence_ac2",
    # Erdos
    "ErdosEnv",
    "create_initial_state_erdos",
    # Denoising
    "DenoisingEnv",
    "create_initial_state_denoising",
    # ALE-Bench
    "AleBenchEnv",
    "create_initial_state_ale_bench",
    # GPU Mode (optional)
    "GpuModeEnv",
    "create_initial_state_gpu_mode",
]
