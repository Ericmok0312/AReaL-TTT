#!/usr/bin/env python3
"""
Analyze PUCT/teacher sampler tree to find breakthrough transitions.

Usage:
    python analyze_sampler_tree.py --config <path_to_config.yaml>

Outputs:
    - Breakdown of parent-child transitions
    - Distribution of improvements
    - Whether code changed vs construction changed
    - Top breakthrough examples
"""

import sys
import json
import os
import numpy as np
from collections import Counter

sys.path.insert(0, '/home/eric/AReaL-TTT')

from areal.experimental.ttt_discover.sampler import (
    create_sampler_from_config,
    _find_latest_sampler_step,
)
from areal.experimental.ttt_discover.config import TTTDDistillConfig
from areal.api.cli_args import load_expr_config


def analyze_sampler_tree(config_path: str, output_dir: str | None = None):
    config, _ = load_expr_config(["--config", config_path], TTTDDistillConfig)
    
    # Create sampler
    sampler_config = config.sampler
    if config.teacher_sampler_checkpoint:
        sampler_config.checkpoint_dir = config.teacher_sampler_checkpoint
    
    from areal.experimental.ttt_discover.sampler import create_sampler_from_config
    sampler = create_sampler_from_config(
        config=sampler_config,
        env_type=getattr(config.sampler, 'env_type', 'ac1'),
        max_version_history=3,
    )
    
    # Load checkpoint
    if config.teacher_sampler_checkpoint:
        latest_step = _find_latest_sampler_step(
            config.teacher_sampler_checkpoint,
            getattr(config.sampler, 'type', 'puct')
        )
        if latest_step is not None:
            sampler._load(latest_step)
            print(f"Loaded sampler at step {latest_step}")
        else:
            print("No checkpoint found, using fresh sampler")
    
    total_states = len(sampler._states)
    print(f"\n{'='*60}")
    print(f"SAMPLER TREE ANALYSIS")
    print(f"{'='*60}")
    print(f"Total states: {total_states}")
    print(f"Initial states: {len(sampler._initial_states)}")
    print(f"T (total expansions): {sampler._T}")
    
    # Build state lookup by id
    state_by_id = {}
    for s in sampler._states:
        sid = getattr(s, 'id', None)
        if sid is not None:
            state_by_id[str(sid)] = s
    
    # Find all parent-child transitions
    transitions = []
    root_states = []
    
    for child in sampler._states:
        parents_info = getattr(child, 'parents', None)
        if not parents_info:
            # Root/initial state
            root_states.append(child)
            continue
        
        parent_id = parents_info[0]["id"] if isinstance(parents_info[0], dict) else parents_info[0]
        parent = state_by_id.get(str(parent_id))
        if parent is None:
            continue
        
        # For AC1: value = -raw_score, so higher value = better
        parent_val = getattr(parent, 'value', None)
        child_val = getattr(child, 'value', None)
        
        if parent_val is None or child_val is None:
            continue
        
        parent_raw = -parent_val  # raw_score (lower is better)
        child_raw = -child_val
        
        absolute_improvement = parent_raw - child_raw  # positive = better
        relative_improvement = absolute_improvement / parent_raw if parent_raw > 0 else 0
        
        parent_code = getattr(parent, 'code', '') or ''
        child_code = getattr(child, 'code', '') or ''
        code_changed = parent_code != child_code
        
        parent_construction = getattr(parent, 'construction', None)
        child_construction = getattr(child, 'construction', None)
        construction_changed = (
            parent_construction != child_construction
            if (parent_construction is not None and child_construction is not None)
            else True
        )
        
        transitions.append({
            'parent_id': str(parent_id),
            'child_id': str(child.id) if hasattr(child, 'id') else None,
            'parent_raw': parent_raw,
            'child_raw': child_raw,
            'absolute_improvement': absolute_improvement,
            'relative_improvement': relative_improvement,
            'parent_timestep': getattr(parent, 'timestep', None),
            'child_timestep': getattr(child, 'timestep', None),
            'code_changed': code_changed,
            'construction_changed': construction_changed,
            'parent_code_len': len(parent_code),
            'child_code_len': len(child_code),
            'parent_construction_len': len(parent_construction) if parent_construction else 0,
            'child_construction_len': len(child_construction) if child_construction else 0,
        })
    
    print(f"Parent-child transitions: {len(transitions)}")
    print(f"Root/initial states: {len(root_states)}")
    
    if not transitions:
        print("No transitions found!")
        return
    
    # Analysis
    print(f"\n{'='*60}")
    print("BREAKTHROUGH ANALYSIS")
    print(f"{'='*60}")
    
    # Thresholds for breakthrough (using ABSOLUTE raw_score improvement, not relative)
    # In AC1: raw_score goes from ~2.0 (random) to ~1.505 (SOTA)
    # A "significant" single-step improvement might be 0.005-0.01 raw_score
    abs_thresholds = [0.001, 0.005, 0.010, 0.050, 0.100]
    print("ABSOLUTE raw_score improvement (lower is better):")
    for threshold in abs_thresholds:
        count = sum(1 for t in transitions if t['absolute_improvement'] >= threshold)
        pct = count / len(transitions) * 100
        print(f"  Improvement >= {threshold:.3f}: {count:5d} / {len(transitions)} ({pct:5.1f}%)")
    
    print("\nRELATIVE improvement:")
    rel_thresholds = [0.001, 0.005, 0.01, 0.05, 0.10]
    for threshold in rel_thresholds:
        count = sum(1 for t in transitions if t['relative_improvement'] >= threshold)
        pct = count / len(transitions) * 100
        print(f"  >= {threshold*100:5.2f}%: {count:5d} / {len(transitions)} ({pct:5.1f}%)")
    
    # Distribution of parent raw scores
    parent_raws = [t['parent_raw'] for t in transitions]
    print(f"\n{'='*60}")
    print("PARENT RAW SCORE DISTRIBUTION (starting point of transitions)")
    print(f"{'='*60}")
    print(f"  Min: {min(parent_raws):.4f}")
    print(f"  Max: {max(parent_raws):.4f}")
    print(f"  Mean: {np.mean(parent_raws):.4f}")
    print(f"  Median: {np.median(parent_raws):.4f}")
    
    bins = [(0, 1.5), (1.5, 1.7), (1.7, 1.9), (1.9, 2.1), (2.1, 3.0), (3.0, float('inf'))]
    for low, high in bins:
        count = sum(1 for r in parent_raws if low <= r < high)
        pct = count / len(parent_raws) * 100
        print(f"  [{low:.1f}, {high:.1f}): {count:5d} ({pct:5.1f}%)")
    
    # Code vs construction change analysis
    print(f"\n{'='*60}")
    print("CODE VS CONSTRUCTION CHANGE ANALYSIS")
    print(f"{'='*60}")
    
    code_change_construction_change = sum(1 for t in transitions if t['code_changed'] and t['construction_changed'])
    code_change_construction_same = sum(1 for t in transitions if t['code_changed'] and not t['construction_changed'])
    code_same_construction_change = sum(1 for t in transitions if not t['code_changed'] and t['construction_changed'])
    code_same_construction_same = sum(1 for t in transitions if not t['code_changed'] and not t['construction_changed'])
    
    print(f"  Code changed + Construction changed: {code_change_construction_change}")
    print(f"  Code changed + Construction same:    {code_change_construction_same}")
    print(f"  Code same + Construction changed:    {code_same_construction_change}")
    print(f"  Code same + Construction same:       {code_same_construction_same}")
    
    # Top breakthroughs
    print(f"\n{'='*60}")
    print("TOP 20 BREAKTHROUGH TRANSITIONS")
    print(f"{'='*60}")
    
    sorted_transitions = sorted(transitions, key=lambda x: x['relative_improvement'], reverse=True)
    for i, t in enumerate(sorted_transitions[:20]):
        print(f"\nRank {i+1}:")
        print(f"  Parent raw: {t['parent_raw']:.4f} → Child raw: {t['child_raw']:.4f}")
        print(f"  Improvement: {t['absolute_improvement']:.4f} ({t['relative_improvement']*100:.1f}%)")
        print(f"  Code changed: {t['code_changed']}, Construction changed: {t['construction_changed']}")
        print(f"  Parent code len: {t['parent_code_len']}, Child code len: {t['child_code_len']}")
    
    # Breakthrough by parent raw score bin
    print(f"\n{'='*60}")
    print("BREAKTHROUGH RATE BY PARENT RAW SCORE")
    print(f"{'='*60}")
    
    for low, high in bins:
        subset = [t for t in transitions if low <= t['parent_raw'] < high]
        if not subset:
            continue
        breakthroughs_abs = [t for t in subset if t['absolute_improvement'] >= 0.005]
        breakthroughs_rel = [t for t in subset if t['relative_improvement'] >= 0.005]
        print(f"  [{low:.1f}, {high:.1f}): {len(subset):4d} transitions, "
              f"{len(breakthroughs_abs):3d} abs>=0.005 ({len(breakthroughs_abs)/len(subset)*100:4.1f}%), "
              f"{len(breakthroughs_rel):3d} rel>=0.5% ({len(breakthroughs_rel)/len(subset)*100:4.1f}%)")
    
    # Save detailed results
    if output_dir is None:
        output_dir = os.path.join(
            config.saver.fileroot,
            config.experiment_name,
            config.trial_name,
        )
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "sampler_tree_analysis.json")
    with open(output_path, 'w') as f:
        json.dump({
            'total_states': total_states,
            'n_transitions': len(transitions),
            'n_root_states': len(root_states),
            'thresholds': {
                str(threshold): sum(1 for t in transitions if t['relative_improvement'] >= threshold)
                for threshold in thresholds
            },
            'parent_raw_stats': {
                'min': float(min(parent_raws)),
                'max': float(max(parent_raws)),
                'mean': float(np.mean(parent_raws)),
                'median': float(np.median(parent_raws)),
            },
            'top_breakthroughs': sorted_transitions[:50],
            'all_transitions': sorted_transitions,  # May be large
        }, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Full results saved to: {output_path}")
    print(f"{'='*60}")
    
    # Key conclusion
    print(f"\n{'='*60}")
    print("KEY QUESTION FOR BREAKTHROUGH-AWARE OPD")
    print(f"{'='*60}")
    
    # Count transitions by parent quality AND improvement size
    print("\nBreakdown by parent raw_score and absolute improvement >= 0.005:")
    
    random_to_any = sum(1 for t in transitions if t['parent_raw'] >= 1.9)
    random_to_good = sum(1 for t in transitions 
                        if t['parent_raw'] >= 1.9 and t['absolute_improvement'] >= 0.005)
    
    intermediate_to_any = sum(1 for t in transitions if 1.6 <= t['parent_raw'] < 1.9)
    intermediate_to_good = sum(1 for t in transitions 
                               if 1.6 <= t['parent_raw'] < 1.9 and t['absolute_improvement'] >= 0.005)
    
    good_to_any = sum(1 for t in transitions if 1.5 <= t['parent_raw'] < 1.6)
    good_to_better = sum(1 for t in transitions 
                        if 1.5 <= t['parent_raw'] < 1.6 and t['absolute_improvement'] >= 0.005)
    
    print(f"  Random parent (>=1.9):     {random_to_good:4d} / {random_to_any:4d} with >=0.005 improvement")
    print(f"  Intermediate (1.6-1.9):    {intermediate_to_good:4d} / {intermediate_to_any:4d} with >=0.005 improvement")
    print(f"  Good parent (1.5-1.6):     {good_to_better:4d} / {good_to_any:4d} with >=0.005 improvement")
    
    # Also show all transitions (no threshold)
    print(f"\n  ALL transitions: {len(transitions)}")
    print(f"  Mean absolute improvement: {np.mean([t['absolute_improvement'] for t in transitions]):.6f}")
    print(f"  Median absolute improvement: {np.median([t['absolute_improvement'] for t in transitions]):.6f}")
    
    if random_to_good > 50:
        print("\n✅ BREAKTHROUGH-AWARE OPD LOOKS PROMISING")
        print("   Many random→improved transitions available")
    elif intermediate_to_good > 100:
        print("\n⚠️  MIXED RESULTS")
        print("   Most improvements are from intermediate states, not random")
        print("   But still potentially useful for curriculum learning")
    elif good_to_better > 100:
        print("\n⚠️  MOSTLY FINE-TUNING TRANSITIONS")
        print("   Breakthroughs are mostly good→SOTA (small improvements)")
        print("   Similar to current final-state OPD")
    else:
        print("\n❌ FEW USEFUL TRANSITIONS")
        print("   Very few significant improvements found")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_sampler_tree.py --config <path_to_config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[2] if sys.argv[1] == "--config" else sys.argv[1]
    analyze_sampler_tree(config_path)
