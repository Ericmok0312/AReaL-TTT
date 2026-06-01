#!/usr/bin/env python3
"""
Test hint code on random construction to verify if it can work independently.
"""

import os
import sys
import tempfile
import subprocess
import time
import numpy as np

# Setup path
sys.path.insert(0, '/home/eric/AReaL-TTT')

from areal.experimental.ttt_discover.envs.inequalities import (
    evaluate_sequence_ac1,
    InequalitiesEnv,
    create_initial_state_ac1,
)
from areal.experimental.ttt_discover.sampler import create_sampler_from_config, _find_latest_sampler_step
from areal.experimental.ttt_discover.config import TTTDDistillConfig
from areal.api.cli_args import load_expr_config


def test_hint_code(config_path: str):
    """Load hint state and test its code on random construction."""
    
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
    
    # Sample a hint state
    hint_states = hint_sampler.sample_states(1)
    if not hint_states:
        print("[TEST] No hint states available!")
        return
    
    hint_state = hint_states[0]
    print(f"\n[TEST] Hint state value: {hint_state.value}")
    print(f"[TEST] Hint state has code: {bool(hint_state.code)}")
    if hint_state.construction:
        print(f"[TEST] Hint construction length: {len(hint_state.construction)}")
        hint_raw = evaluate_sequence_ac1(hint_state.construction)
        print(f"[TEST] Hint construction raw_score: {hint_raw:.6f} (reward: {1.0/hint_raw:.6f})")
    
    # Extract hint code
    hint_code = hint_state.code
    if not hint_code:
        print("[TEST] Hint state has no code!")
        return
    
    # Clean up code (remove markdown if present)
    if hint_code.startswith("```python"):
        hint_code = hint_code[9:]  # Remove ```python
    if hint_code.endswith("```"):
        hint_code = hint_code[:-3]  # Remove ```
    hint_code = hint_code.strip()
    
    print(f"\n[TEST] Hint code length: {len(hint_code)} chars")
    print("[TEST] First 500 chars of hint code:")
    print(hint_code[:500])
    print("...")
    
    # Test 1: Run on hint state's own construction
    print("\n" + "="*60)
    print("TEST 1: Run on hint state's OWN construction")
    print("="*60)
    if hint_state.construction:
        reward1, error1 = _run_code(hint_code, hint_state.construction, env)
        print(f"Result: reward={reward1:.6f}, error={error1}")
    
    # Test 2: Run on random constant construction
    print("\n" + "="*60)
    print("TEST 2: Run on RANDOM CONSTANT construction [0.5]*1000")
    print("="*60)
    random_const = [0.5] * 1000
    reward2, error2 = _run_code(hint_code, random_const, env)
    print(f"Result: reward={reward2:.6f}, error={error2}")
    
    # Test 3: Run on true random construction
    print("\n" + "="*60)
    print("TEST 3: Run on TRUE RANDOM construction")
    print("="*60)
    rng = np.random.default_rng(42)
    random_seq = rng.random(1000).tolist()
    reward3, error3 = _run_code(hint_code, random_seq, env)
    print(f"Result: reward={reward3:.6f}, error={error3}")
    
    # Test 4: Run on initial state from create_initial_state_ac1
    print("\n" + "="*60)
    print("TEST 4: Run on DEFAULT INITIAL STATE")
    print("="*60)
    initial_state = create_initial_state_ac1(initial_exp_type='random', budget_s=30)
    print(f"Initial state value: {initial_state.value}")
    print(f"Initial construction length: {len(initial_state.construction) if initial_state.construction else 0}")
    if initial_state.construction:
        init_raw = evaluate_sequence_ac1(initial_state.construction)
        print(f"Initial raw_score: {init_raw:.6f} (reward: {1.0/init_raw:.6f})")
        reward4, error4 = _run_code(hint_code, initial_state.construction, env)
        print(f"Result: reward={reward4:.6f}, error={error4}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Hint state own construction:    reward={reward1:.6f if 'reward1' in dir() else 'N/A'}")
    print(f"Random constant [0.5]*1000:     reward={reward2:.6f if 'reward2' in dir() else 'N/A'}")
    print(f"True random sequence:           reward={reward3:.6f if 'reward3' in dir() else 'N/A'}")
    print(f"Default initial state:          reward={reward4:.6f if 'reward4' in dir() else 'N/A'}")
    print(f"Target (SOTA):                  reward=0.6642")


def _run_code(code: str, construction: list, env: InequalitiesEnv, timeout: int = 120):
    """Execute code with given construction and return reward."""
    
    # Create a state with this construction
    from areal.experimental.ttt_discover.state import InequalitiesState
    state = InequalitiesState(
        timestep=-1,
        construction=construction,
        code=code,
        value=-evaluate_sequence_ac1(construction),
    )
    
    # Execute via env
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
        return 0.0, f"ERROR: {str(e)[:100]} ({elapsed:.1f}s)"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_hint_code.py --config <path_to_config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[2] if sys.argv[1] == "--config" else sys.argv[1]
    test_hint_code(config_path)
