#!/usr/bin/env python3
"""
分析 PUCT Selection 质量

检测由于 Async FCFS 更新导致的信息滞后问题：
1. 在 Step X 被选中的 parent 是否是最优选择
2. 识别被低估的 alternatives（selection score 低但最终 Q 高）
3. 量化信息滞后造成的损失

Usage:
    python analyze_selection_quality.py /path/to/training_history.pkl [--step STEP] [--output output.json]
"""

import pickle
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Any


def load_history(history_path: str) -> Dict:
    """Load training history from pickle file."""
    with open(history_path, 'rb') as f:
        data = pickle.load(f)
    return data.get('history', data)  # Handle both formats


def get_parent_episodes_from_snapshot(snapshot: Dict) -> Dict[Tuple[str, int], Dict]:
    """Extract parent_episodes from a step snapshot."""
    puct_data = snapshot.get('puct_analysis', {})
    return puct_data.get('parent_episodes', {})


def find_selections_at_step(step: int, parent_episodes: Dict) -> List[Tuple[str, Dict]]:
    """
    Find all parents that were selected at given step.
    Returns list of (parent_id, episode) tuples.
    """
    selections = []
    for (parent_id, sampled_step), episode in parent_episodes.items():
        if sampled_step == step:
            selections.append((parent_id, episode))
    return selections


def get_active_episodes_at_step(step: int, parent_episodes: Dict) -> List[Tuple[str, int, Dict]]:
    """
    Find all episodes that were active (incomplete) at given step.
    These are the parents that were in the candidate pool.
    
    Returns: list of (parent_id, sampled_step, episode) tuples
    """
    active = []
    for (parent_id, sampled_step), episode in parent_episodes.items():
        # Must have been sampled before this step
        if sampled_step >= step:
            continue
        
        # Must not have been completed before this step
        completed_step = episode.get('completed_step')
        if completed_step is not None and completed_step <= step:
            continue
        
        active.append((parent_id, sampled_step, episode))
    
    return active


def analyze_single_selection(
    step: int,
    selected_pid: str,
    selected_episode: Dict,
    all_episodes: Dict[Tuple[str, int], Dict],
    verbose: bool = False
) -> Dict:
    """
    Analyze quality of a single selection decision.
    
    Returns analysis dict with:
    - selected_parent_info
    - alternatives
    - missed_opportunities
    - loss
    """
    result = {
        'step': step,
        'selected_parent': selected_pid,
        'alternatives': [],
        'missed_opportunities': [],
        'was_optimal': True,
        'total_loss': 0.0,
    }
    
    # Get selected parent info
    sel_info = selected_episode.get('selection_info', {})
    selected_selection_score = sel_info.get('score', 0.0)
    selected_selection_q = sel_info.get('q_value', 0.0)
    
    # Calculate final Q for selected parent
    children = selected_episode.get('children_completed', [])
    selected_final_q = max((c['reward'] for c in children), default=selected_selection_q)
    
    result['selected_parent_info'] = {
        'sampled_step': selected_episode['sampled_step'],
        'selection_score': selected_selection_score,
        'selection_q': selected_selection_q,
        'final_q': selected_final_q,
        'n_children': len(children),
        'completed': selected_episode.get('completed', False),
        'completed_step': selected_episode.get('completed_step'),
    }
    
    if verbose:
        print(f"\nStep {step}: Selected Parent {selected_pid[:8]}...")
        print(f"  Selection: score={selected_selection_score:.3f}, Q={selected_selection_q:.3f}")
        print(f"  Final Q: {selected_final_q:.3f} ({len(children)} children)")
        print(f"  Completed: {selected_episode.get('completed', False)} at step {selected_episode.get('completed_step', 'N/A')}")
    
    # Find all active alternatives at this step
    active_episodes = get_active_episodes_at_step(step, all_episodes)
    
    for alt_pid, alt_sampled_step, alt_episode in active_episodes:
        if alt_pid == selected_pid:
            continue  # Skip selected parent
        
        alt_sel_info = alt_episode.get('selection_info', {})
        alt_selection_score = alt_sel_info.get('score', 0.0)
        alt_selection_q = alt_sel_info.get('q_value', 0.0)
        
        alt_children = alt_episode.get('children_completed', [])
        alt_final_q = max((c['reward'] for c in alt_children), default=alt_selection_q)
        
        alt_info = {
            'parent_id': alt_pid,
            'sampled_step': alt_sampled_step,
            'selection_score': alt_selection_score,
            'selection_q': alt_selection_q,
            'final_q': alt_final_q,
            'n_children': len(alt_children),
            'completed': alt_episode.get('completed', False),
        }
        result['alternatives'].append(alt_info)
        
        # Check if this was a missed opportunity
        # Criteria: selection score lower (not chosen) but final Q higher (better)
        if alt_selection_score < selected_selection_score and alt_final_q > selected_final_q:
            opportunity = {
                'parent_id': alt_pid,
                'selection_score': alt_selection_score,
                'selected_score': selected_selection_score,
                'score_deficit': selected_selection_score - alt_selection_score,
                'final_q': alt_final_q,
                'selected_final_q': selected_final_q,
                'reward_loss': alt_final_q - selected_final_q,
                'reason': (f"Score {alt_selection_score:.3f} < {selected_selection_score:.3f} "
                          f"but Q {alt_final_q:.3f} > {selected_final_q:.3f}"),
            }
            result['missed_opportunities'].append(opportunity)
            result['total_loss'] += opportunity['reward_loss']
            result['was_optimal'] = False
            
            if verbose:
                print(f"  MISSED: Parent {alt_pid[:8]}... (sampled step {alt_sampled_step})")
                print(f"    Selection: score={alt_selection_score:.3f}, Q={alt_selection_q:.3f}")
                print(f"    Final Q: {alt_final_q:.3f}")
                print(f"    Loss: {opportunity['reward_loss']:.3f}")
    
    if verbose:
        if result['missed_opportunities']:
            print(f"  *** MISSED {len(result['missed_opportunities'])} BETTER ALTERNATIVES ***")
            print(f"  Total loss: {result['total_loss']:.3f}")
        else:
            print(f"  ✓ Optimal selection (no better alternatives)")
    
    return result


def analyze_all_steps(history: Dict, verbose: bool = False) -> Dict:
    """Analyze all selection steps in history."""
    all_results = []
    summary = {
        'total_selections': 0,
        'optimal_selections': 0,
        'suboptimal_selections': 0,
        'total_missed_opportunities': 0,
        'total_reward_loss': 0.0,
        'step_losses': [],
    }
    
    # Collect all steps
    steps = sorted(int(k.split('_')[1]) for k in history.keys() if k.startswith('step_'))
    
    for step_num in steps:
        key = f"step_{step_num}"
        if key not in history:
            continue
        
        snapshot = history[key]
        parent_episodes = get_parent_episodes_from_snapshot(snapshot)
        
        if not parent_episodes:
            continue
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Analyzing Step {step_num}")
            print(f"Total episodes: {len(parent_episodes)}")
        
        # Find selections at this step
        selections = find_selections_at_step(step_num, parent_episodes)
        
        for selected_pid, selected_episode in selections:
            result = analyze_single_selection(
                step=step_num,
                selected_pid=selected_pid,
                selected_episode=selected_episode,
                all_episodes=parent_episodes,
                verbose=verbose,
            )
            all_results.append(result)
            
            # Update summary
            summary['total_selections'] += 1
            if result['was_optimal']:
                summary['optimal_selections'] += 1
            else:
                summary['suboptimal_selections'] += 1
                summary['total_missed_opportunities'] += len(result['missed_opportunities'])
                summary['total_reward_loss'] += result['total_loss']
                summary['step_losses'].append({
                    'step': step_num,
                    'loss': result['total_loss'],
                    'selected': selected_pid[:8],
                    'n_missed': len(result['missed_opportunities']),
                })
    
    # Calculate statistics
    if summary['total_selections'] > 0:
        summary['optimal_rate'] = summary['optimal_selections'] / summary['total_selections']
        summary['avg_loss_per_selection'] = summary['total_reward_loss'] / summary['total_selections']
        summary['avg_loss_per_suboptimal'] = (
            summary['total_reward_loss'] / summary['suboptimal_selections']
            if summary['suboptimal_selections'] > 0 else 0.0
        )
    
    return {
        'summary': summary,
        'detailed_results': all_results,
    }


def print_summary(summary: Dict):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total selections analyzed: {summary['total_selections']}")
    print(f"Optimal selections: {summary['optimal_selections']} ({summary.get('optimal_rate', 0)*100:.1f}%)")
    print(f"Suboptimal selections: {summary['suboptimal_selections']}")
    print(f"Total missed opportunities: {summary['total_missed_opportunities']}")
    print(f"Total reward loss: {summary['total_reward_loss']:.3f}")
    print(f"Average loss per selection: {summary.get('avg_loss_per_selection', 0):.3f}")
    print(f"Average loss per suboptimal: {summary.get('avg_loss_per_suboptimal', 0):.3f}")
    
    if summary['step_losses']:
        print("\nTop 5 steps with highest loss:")
        sorted_losses = sorted(summary['step_losses'], key=lambda x: x['loss'], reverse=True)[:5]
        for loss_info in sorted_losses:
            print(f"  Step {loss_info['step']}: loss={loss_info['loss']:.3f}, "
                  f"selected={loss_info['selected']}..., "
                  f"missed={loss_info['n_missed']} alternatives")


def main():
    parser = argparse.ArgumentParser(description='Analyze PUCT selection quality')
    parser.add_argument('history_path', help='Path to training_history.pkl')
    parser.add_argument('--step', type=int, help='Analyze specific step only')
    parser.add_argument('--output', '-o', help='Output JSON file for detailed results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    # Load history
    print(f"Loading history from {args.history_path}...")
    history = load_history(args.history_path)
    print(f"Loaded {len(history)} snapshots")
    
    if args.step is not None:
        # Analyze single step
        key = f"step_{args.step}"
        if key not in history:
            print(f"Step {args.step} not found in history")
            return
        
        snapshot = history[key]
        parent_episodes = get_parent_episodes_from_snapshot(snapshot)
        
        print(f"\nAnalyzing Step {args.step}...")
        print(f"Total episodes: {len(parent_episodes)}")
        
        selections = find_selections_at_step(args.step, parent_episodes)
        print(f"Selections at this step: {len(selections)}")
        
        for selected_pid, selected_episode in selections:
            result = analyze_single_selection(
                step=args.step,
                selected_pid=selected_pid,
                selected_episode=selected_episode,
                all_episodes=parent_episodes,
                verbose=True,
            )
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"\nDetailed results saved to {args.output}")
    else:
        # Analyze all steps
        results = analyze_all_steps(history, verbose=args.verbose)
        print_summary(results['summary'])
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nDetailed results saved to {args.output}")


if __name__ == '__main__':
    main()
