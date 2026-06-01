#!/usr/bin/env python3
"""
Standalone test: Load a teacher hint state and verify its code works on random constructions.

Usage:
    python test_hint_code_standalone.py --config <path_to_config.yaml>

Prerequisites:
    - The AReaL-TTT package must be importable (pip install -e . or PYTHONPATH)
    - Config must have teacher_sampler_checkpoint pointing to valid sampler states
"""

import os
import sys
import time
import numpy as np


def _evaluate_sequence_ac1(construction):
    """Standalone AC1 evaluation."""
    n = len(construction)
    r = np.array(construction, dtype=np.float64)
    i = np.arange(1, n + 1, dtype=np.float64)
    
    # numerator
    num = (
        np.sum((i**2 + 1) * r**2)
        + 2 * np.sum(np.cumsum(r[:-1]) * r[1:] * (np.arange(2, n+1)**2 + 1))
    )
    # denominator
    den = np.sum(r**2 * i * (n + 1 - i))
    return num / den


def _run_code(code: str, construction: list, env, timeout: int = 120):
    """Execute code with given construction and return reward."""
    from areal.experimental.ttt_discover.state import InequalitiesState
    
    state = InequalitiesState(
        timestep=-1,
        construction=construction,
        code=code,
        value=-_evaluate_sequence_ac1(construction),
    )
    
    start = time.time()
    try:
        result = env.execute(code, state)
        elapsed = time.time() - start
        if result.is_valid:
            return result.reward, f"OK ({elapsed:.1f}s)"
        else:
            return 0.0, f"FAIL: {result.fail_type} ({elapsed:.1f}s)"
    except Exception as e:
        elapsed = time.time() - start
        return 0.0, f"ERROR: {str(e)[:200]} ({elapsed:.1f}s)"


def test_hint_code(config_path: str):
    from areal.experimental.ttt_discover.envs.inequalities import (
        evaluate_sequence_ac1,
        InequalitiesEnv,
        create_initial_state_ac1,
    )
    from areal.experimental.ttt_discover.sampler import (
        create_sampler_from_config,
        _find_latest_sampler_step,
    )
    from areal.experimental.ttt_discover.config import TTTDDistillConfig
    from areal.api.cli_args import load_expr_config
    
    # Load config
    config, _ = load_expr_config(["--config", config_path], TTTDDistillConfig)
    
    # Create env
    env = InequalitiesEnv(problem_type='ac1', budget_s=30)
    
    # Create hint sampler
    teacher_sampler_config = config.sampler
    if config.teacher_sampler_checkpoint:
        teacher_sampler_config.checkpoint_dir = config.teacher_sampler_checkpoint
    
    hint_sampler = create_sampler_from_config(
        config=teacher_sampler_config,
        env_type=getattr(config.sampler, 'env_type', 'ac1'),
        max_version_history=3,
    )
    
    if config.teacher_sampler_checkpoint:
        latest_step = _find_latest_sampler_step(
            config.teacher_sampler_checkpoint,
            getattr(config.sampler, 'type', 'puct')
        )
        if latest_step is not None:
            hint_sampler._load(latest_step)
            print(f"[TEST] Loaded hint sampler at step {latest_step}")
            print(f"[TEST] Total states: {len(hint_sampler._states)}")
    
    # Sample hint states
    hint_states = hint_sampler.sample_states(min(5, len(hint_sampler._states)))
    if not hint_states:
        print("[TEST] No hint states available!")
        return
    
    print(f"\n[TEST] Testing {len(hint_states)} hint states")
    
    # Test constructions
    rng = np.random.default_rng(42)
    test_constructions = {
        "const_0.5_1000": [0.5] * 1000,
        "const_0.5_5000": [0.5] * 5000,
        "random_1000": rng.random(1000).tolist(),
        "random_5000": rng.random(5000).tolist(),
    }
    
    for idx, hint_state in enumerate(hint_states):
        print(f"\n{'='*60}")
        print(f"HINT STATE {idx}")
        print(f"{'='*60}")
        print(f"  value: {hint_state.value}")
        if hint_state.construction:
            hint_raw = evaluate_sequence_ac1(hint_state.construction)
            print(f"  own construction raw_score: {hint_raw:.6f}")
        
        # Clean code
        hint_code = hint_state.code or ""
        if hint_code.startswith("```python"):
            hint_code = hint_code[9:]
        if hint_code.endswith("```"):
            hint_code = hint_code[:-3]
        hint_code = hint_code.strip()
        
        if not hint_code:
            print("  [SKIP] No code in hint state")
            continue
        
        print(f"  code length: {len(hint_code)} chars")
        
        # Test on own construction first
        if hint_state.construction:
            r, e = _run_code(hint_code, hint_state.construction, env)
            print(f"  OWN construction:     reward={r:.6f}  ({e})")
        
        # Test on each test construction
        for name, construction in test_constructions.items():
            r, e = _run_code(hint_code, construction, env)
            print(f"  {name:20s}: reward={r:.6f}  ({e})")
    
    # Test default initial state
    print(f"\n{'='*60}")
    print("DEFAULT INITIAL STATE")
    print(f"{'='*60}")
    init_state = create_initial_state_ac1(initial_exp_type='random', budget_s=30)
    print(f"  value: {init_state.value}")
    if init_state.construction:
        init_raw = evaluate_sequence_ac1(init_state.construction)
        print(f"  raw_score: {init_raw:.6f}")
        print(f"  length: {len(init_state.construction)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_hint_code_standalone.py --config <path_to_config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[2] if sys.argv[1] == "--config" else sys.argv[1]
    test_hint_code(config_path)
