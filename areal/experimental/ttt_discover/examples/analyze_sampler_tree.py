#!/usr/bin/env python3
"""
Analyze PUCT sampler tree to extract unique root-to-leaf paths,
milestones, and diff-based compression statistics.

Usage:
    python areal/experimental/ttt_discover/examples/analyze_sampler_tree.py \
        --sampler_path ./outputs_ac1_async_v2/tttd-async-qwen3-8b-ac1/trial0/sampler/puct_sampler_step_000149.json \
        [--topk 20] \
        [--max_path_length 50] \
        [--output_dir ./tree_analysis] \
        [--milestone_mode full|milestone|diff|diff_summary] \
        [--n_milestones 4] \
        [--min_improvement 0.005]
"""

import argparse
import difflib
import json
import os
import sys
import textwrap
from collections import defaultdict
from pathlib import Path


def load_sampler_checkpoint(path: str) -> dict:
    """Load sampler checkpoint JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def build_parent_child_maps(states: list[dict]) -> tuple[dict, dict, dict]:
    """
    Build lookup maps from raw state dicts.

    Returns:
        id_to_state: dict[str, dict] - state_id -> state dict
        children_map: dict[str, list[str]] - parent_id -> list of child_ids
        parent_map: dict[str, str] - child_id -> parent_id (most recent parent)
    """
    id_to_state = {s["id"]: s for s in states if s.get("id")}
    children_map = defaultdict(list)
    parent_map = {}

    for s in states:
        sid = s.get("id")
        if not sid:
            continue
        parents = s.get("parents", [])
        if parents:
            # Most recent parent is first in the list
            pid = parents[0].get("id")
            if pid:
                children_map[pid].append(sid)
                parent_map[sid] = pid

    return id_to_state, dict(children_map), parent_map


def find_root_states(states: list[dict]) -> list[dict]:
    """Find states with no parents (root/initial states)."""
    roots = []
    for s in states:
        parents = s.get("parents", [])
        if not parents:
            roots.append(s)
    return roots


def find_leaf_states(states: list[dict], children_map: dict) -> list[dict]:
    """Find states with no children (leaf/terminal states)."""
    all_ids = {s["id"] for s in states if s.get("id")}
    parent_ids = set(children_map.keys())
    leaf_ids = all_ids - parent_ids
    return [s for s in states if s.get("id") in leaf_ids]


def extract_all_paths(
    root_ids: list[str],
    children_map: dict,
    id_to_state: dict,
    max_depth: int = 100,
) -> list[list[str]]:
    """
    Extract all unique root-to-leaf paths via DFS.

    Returns:
        List of paths, where each path is a list of state IDs from root to leaf.
    """
    paths = []

    def dfs(node_id: str, path: list[str]):
        if len(path) > max_depth:
            paths.append(path.copy())
            return
        children = children_map.get(node_id, [])
        if not children:
            paths.append(path.copy())
            return
        for child_id in children:
            dfs(child_id, path + [child_id])

    for rid in root_ids:
        dfs(rid, [rid])

    return paths


def extract_milestones(
    path_ids: list[str],
    id_to_state: dict,
    n_milestones: int = 4,
    min_improvement: float = 0.005,
) -> list[dict]:
    """
    Extract key milestone states from a path.

    Strategy:
    1. Always keep root (index 0) and leaf (last)
    2. Among intermediate states, pick those with largest improvement over predecessor
    3. Filter out improvements below min_improvement threshold
    4. If too few, fall back to evenly spaced samples

    Returns:
        List of state dicts (milestones), ordered from root to leaf.
    """
    states = [id_to_state[sid] for sid in path_ids if sid in id_to_state]
    if len(states) <= n_milestones:
        return states

    # Compute improvements
    improvements = []
    for i in range(1, len(states)):
        prev_val = states[i - 1].get("value")
        curr_val = states[i].get("value")
        if prev_val is not None and curr_val is not None:
            imp = curr_val - prev_val
            improvements.append((i, imp))

    # Sort by improvement magnitude (descending)
    improvements.sort(key=lambda x: x[1], reverse=True)

    # Select top breakthroughs above threshold
    selected_indices = [0]  # Always include root
    for idx, imp in improvements:
        if imp >= min_improvement and idx not in selected_indices:
            selected_indices.append(idx)
        if len(selected_indices) >= n_milestones - 1:
            break

    # If we don't have enough, fill with evenly spaced states
    if len(selected_indices) < n_milestones - 1:
        n_needed = n_milestones - 1 - len(selected_indices)
        step = len(states) // (n_needed + 1)
        for i in range(1, n_needed + 1):
            idx = i * step
            if idx not in selected_indices and idx < len(states) - 1:
                selected_indices.append(idx)

    # Always include leaf
    if (len(states) - 1) not in selected_indices:
        selected_indices.append(len(states) - 1)

    selected_indices = sorted(set(selected_indices))
    return [states[i] for i in selected_indices]


def compute_code_diff(prev_code: str, curr_code: str, context_lines: int = 3) -> str:
    """
    Compute unified diff between two code strings.

    Returns:
        Unified diff string, or empty string if no differences.
    """
    prev_lines = prev_code.splitlines(keepends=True)
    curr_lines = curr_code.splitlines(keepends=True)

    # Ensure lines end with newline for clean diff
    prev_lines = [l if l.endswith('\n') else l + '\n' for l in prev_lines]
    curr_lines = [l if l.endswith('\n') else l + '\n' for l in curr_lines]

    diff = difflib.unified_diff(
        prev_lines,
        curr_lines,
        fromfile="phase_prev.py",
        tofile="phase_curr.py",
        n=context_lines,
    )
    return "".join(diff)


def summarize_diff(diff_text: str, max_lines: int = 20) -> str:
    """
    Summarize a diff by extracting key changes.

    Returns a compact description like:
    "+3 lines (added: propose_with_restart, modified: loop condition), -1 line (removed: old_init)"
    """
    added = 0
    removed = 0
    added_funcs = []
    removed_funcs = []

    for line in diff_text.splitlines():
        if line.startswith('+') and not line.startswith('+++'):
            added += 1
            stripped = line[1:].strip()
            if stripped.startswith('def ') and len(stripped) > 4:
                func_name = stripped[4:].split('(')[0].strip()
                if func_name and func_name not in added_funcs:
                    added_funcs.append(func_name)
        elif line.startswith('-') and not line.startswith('---'):
            removed += 1
            stripped = line[1:].strip()
            if stripped.startswith('def ') and len(stripped) > 4:
                func_name = stripped[4:].split('(')[0].strip()
                if func_name and func_name not in removed_funcs:
                    removed_funcs.append(func_name)

    parts = []
    if added > 0:
        parts.append(f"+{added} lines")
    if removed > 0:
        parts.append(f"-{removed} lines")
    if added_funcs:
        parts.append(f"added funcs: {', '.join(added_funcs[:3])}")
    if removed_funcs:
        parts.append(f"removed funcs: {', '.join(removed_funcs[:3])}")

    return "; ".join(parts) if parts else "no structural changes"


def estimate_hint_length(
    path_states: list[dict],
    mode: str = "full",
    n_milestones: int = 4,
    min_improvement: float = 0.005,
) -> dict:
    """
    Estimate the length of a teacher hint in different formats.

    Args:
        path_states: List of state dicts along a path (root to leaf)
        mode: 'full' (all states), 'milestone' (key states only),
              'diff' (first state + diffs), 'diff_summary' (diff summaries only)
        n_milestones: Number of milestones for milestone mode
        min_improvement: Minimum improvement threshold for milestones

    Returns:
        Dict with length estimates (chars, lines, approx_tokens).
    """
    if not path_states:
        return {"chars": 0, "lines": 0, "approx_tokens": 0, "n_phases": 0}

    if mode == "milestone":
        states = extract_milestones(
            [s["id"] for s in path_states],
            {s["id"]: s for s in path_states},
            n_milestones=n_milestones,
            min_improvement=min_improvement,
        )
    else:
        states = path_states

    total_chars = 0
    total_lines = 0
    n_phases = len(states)

    if mode in ("full", "milestone"):
        for s in states:
            code = s.get("code", "")
            total_chars += len(code)
            total_lines += code.count('\n') + 1 if code else 0

    elif mode == "diff":
        # First phase: full code
        first_code = states[0].get("code", "")
        total_chars += len(first_code)
        total_lines += first_code.count('\n') + 1 if first_code else 0

        # Subsequent phases: diff only
        for i in range(1, len(states)):
            prev_code = states[i - 1].get("code", "")
            curr_code = states[i].get("code", "")
            diff = compute_code_diff(prev_code, curr_code)
            total_chars += len(diff)
            total_lines += diff.count('\n') + 1 if diff else 0

    elif mode == "diff_summary":
        # First phase: full code
        first_code = states[0].get("code", "")
        total_chars += len(first_code)
        total_lines += first_code.count('\n') + 1 if first_code else 0

        # Subsequent phases: diff summary only
        for i in range(1, len(states)):
            prev_code = states[i - 1].get("code", "")
            curr_code = states[i].get("code", "")
            diff = compute_code_diff(prev_code, curr_code)
            summary = summarize_diff(diff)
            total_chars += len(summary)
            total_lines += 1

    # Rough token estimate: ~4 chars per token for code
    approx_tokens = total_chars // 4

    return {
        "chars": total_chars,
        "lines": total_lines,
        "approx_tokens": approx_tokens,
        "n_phases": n_phases,
    }


def compute_path_stats(path_ids: list[str], id_to_state: dict) -> dict:
    """Compute statistics for a single path."""
    states = [id_to_state[sid] for sid in path_ids if sid in id_to_state]
    values = [s.get("value") for s in states if s.get("value") is not None]
    timesteps = [s.get("timestep", -1) for s in states]

    # Check height_sequence_1 dependency
    uses_height_sequence = []
    for s in states:
        code = s.get("code", "")
        uses_height_sequence.append("height_sequence_1" in code)

    # Extract code length per state
    code_lengths = [len(s.get("code", "")) for s in states]

    stats = {
        "length": len(states),
        "root_value": values[0] if values else None,
        "leaf_value": values[-1] if values else None,
        "best_value": max(values) if values else None,
        "value_improvement": (values[-1] - values[0]) if len(values) >= 2 else 0.0,
        "timestep_span": (max(timesteps) - min(timesteps)) if timesteps else 0,
        "uses_height_sequence_ratio": sum(uses_height_sequence) / max(len(uses_height_sequence), 1),
        "all_use_height_sequence": all(uses_height_sequence) if uses_height_sequence else False,
        "avg_code_length": sum(code_lengths) / max(len(code_lengths), 1),
        "max_code_length": max(code_lengths) if code_lengths else 0,
    }
    return stats


def print_path_detail(path_ids: list[str], id_to_state: dict, max_states_to_show: int = 10):
    """Print detailed info for a single path."""
    states = [id_to_state[sid] for sid in path_ids if sid in id_to_state]
    print(f"  Path length: {len(states)}")

    # Show first few and last few states
    indices_to_show = list(range(min(max_states_to_show, len(states))))
    if len(states) > max_states_to_show:
        indices_to_show.append(len(states) - 1)

    for idx in indices_to_show:
        s = states[idx]
        raw_score = -s.get("value", 0) if s.get("value") is not None else None
        raw_score_str = f"{raw_score:.6f}" if raw_score is not None else 'N/A'
        code_preview = (s.get("code", "")[:200].replace("\n", " ") + "...") if s.get("code") else "N/A"
        uses_hs = "height_sequence_1" in s.get("code", "")
        print(f"    Step {idx}: score={raw_score_str} "
              f"timestep={s.get('timestep')} "
              f"uses_hs={uses_hs} "
              f"code_len={len(s.get('code', ''))} "
              f"code_preview={code_preview}")


def analyze_tree(sampler_path: str, topk: int, max_path_length: int, output_dir: str,
                 milestone_mode: str, n_milestones: int, min_improvement: float):
    """Main analysis function."""
    print(f"Loading sampler checkpoint: {sampler_path}")
    checkpoint = load_sampler_checkpoint(sampler_path)
    states = checkpoint.get("states", [])
    print(f"Total states in checkpoint: {len(states)}")

    # Build maps
    id_to_state, children_map, parent_map = build_parent_child_maps(states)
    print(f"Unique states: {len(id_to_state)}")
    print(f"Parent-child edges: {sum(len(v) for v in children_map.values())}")

    # Find roots and leaves
    roots = find_root_states(states)
    root_ids = [r["id"] for r in roots]
    print(f"Root states (no parents): {len(roots)}")

    leaves = find_leaf_states(states, children_map)
    leaf_ids = {l["id"] for l in leaves}
    print(f"Leaf states (no children): {len(leaves)}")

    # Extract all unique root-to-leaf paths
    print(f"\nExtracting all root-to-leaf paths (max_depth={max_path_length})...")
    paths = extract_all_paths(root_ids, children_map, id_to_state, max_depth=max_path_length)
    print(f"Total unique root-to-leaf paths: {len(paths)}")

    if not paths:
        print("No paths found. Tree may be very shallow or all states are roots.")
        return

    # Compute stats for all paths
    path_stats = []
    for path_ids in paths:
        stats = compute_path_stats(path_ids, id_to_state)
        stats["path_ids"] = path_ids
        path_stats.append(stats)

    # Sort by leaf value (best first for AC1: value = -raw_score, higher is better)
    path_stats.sort(key=lambda x: (x["leaf_value"] if x["leaf_value"] is not None else float("-inf")), reverse=True)

    # Overall statistics
    lengths = [p["length"] for p in path_stats]
    leaf_values = [p["leaf_value"] for p in path_stats if p["leaf_value"] is not None]
    improvements = [p["value_improvement"] for p in path_stats if p["value_improvement"] is not None]
    hs_ratios = [p["uses_height_sequence_ratio"] for p in path_stats]

    print("\n" + "=" * 70)
    print("OVERALL PATH STATISTICS")
    print("=" * 70)
    print(f"Path length: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.2f}")
    if leaf_values:
        print(f"Leaf value (AC1 raw score = -value): best={-max(leaf_values):.6f}, worst={-min(leaf_values):.6f}, mean={-sum(leaf_values)/len(leaf_values):.6f}")
    if improvements:
        print(f"Value improvement (leaf - root): best={max(improvements):.6f}, worst={min(improvements):.6f}, mean={sum(improvements)/len(improvements):.6f}")
    print(f"Height_sequence_1 dependency ratio: min={min(hs_ratios):.2%}, max={max(hs_ratios):.2%}, mean={sum(hs_ratios)/len(hs_ratios):.2%}")

    # Distribution of path lengths
    length_dist = defaultdict(int)
    for l in lengths:
        length_dist[l] += 1
    print(f"\nPath length distribution:")
    for l in sorted(length_dist.keys()):
        print(f"  Length {l}: {length_dist[l]} paths")

    # ============================================================
    # MILESTONE & COMPRESSION ANALYSIS
    # ============================================================
    print("\n" + "=" * 70)
    print("HINT LENGTH ANALYSIS (Top 10 paths)")
    print("=" * 70)
    print(f"Comparing hint formats: full vs milestone({n_milestones}) vs diff vs diff_summary")
    print(f"{'Rank':<6} {'Full(k)':<10} {'Milestone(k)':<14} {'Diff(k)':<10} {'DiffSum(k)':<12} {'Len':<5} {'Improve':<10} {'Phases':<8}")
    print("-" * 85)

    for rank, ps in enumerate(path_stats[:10], 1):
        path_states = [id_to_state[sid] for sid in ps["path_ids"] if sid in id_to_state]

        full_len = estimate_hint_length(path_states, mode="full")
        milestone_len = estimate_hint_length(path_states, mode="milestone", n_milestones=n_milestones, min_improvement=min_improvement)
        diff_len = estimate_hint_length(path_states, mode="diff", n_milestones=n_milestones, min_improvement=min_improvement)
        diff_sum_len = estimate_hint_length(path_states, mode="diff_summary", n_milestones=n_milestones, min_improvement=min_improvement)

        print(f"{rank:<6} "
              f"{full_len['approx_tokens']/1000:<10.1f} "
              f"{milestone_len['approx_tokens']/1000:<14.1f} "
              f"{diff_len['approx_tokens']/1000:<10.1f} "
              f"{diff_sum_len['approx_tokens']/1000:<12.1f} "
              f"{ps['length']:<5} "
              f"{ps['value_improvement']:<10.4f} "
              f"{milestone_len['n_phases']:<8}")

    # Average compression ratios
    print("\n" + "-" * 85)
    full_total = sum(estimate_hint_length([id_to_state[sid] for sid in ps["path_ids"] if sid in id_to_state], mode="full")["approx_tokens"] for ps in path_stats[:topk])
    ms_total = sum(estimate_hint_length([id_to_state[sid] for sid in ps["path_ids"] if sid in id_to_state], mode="milestone", n_milestones=n_milestones, min_improvement=min_improvement)["approx_tokens"] for ps in path_stats[:topk])
    diff_total = sum(estimate_hint_length([id_to_state[sid] for sid in ps["path_ids"] if sid in id_to_state], mode="diff", n_milestones=n_milestones, min_improvement=min_improvement)["approx_tokens"] for ps in path_stats[:topk])
    ds_total = sum(estimate_hint_length([id_to_state[sid] for sid in ps["path_ids"] if sid in id_to_state], mode="diff_summary", n_milestones=n_milestones, min_improvement=min_improvement)["approx_tokens"] for ps in path_stats[:topk])

    print(f"\nAVERAGE COMPRESSION (top {topk} paths)")
    print(f"  Full code hint:        {full_total/topk/1000:.1f}k tokens")
    print(f"  Milestone hint:        {ms_total/topk/1000:.1f}k tokens  ({ms_total/full_total*100:.1f}% of full)")
    print(f"  Diff hint:             {diff_total/topk/1000:.1f}k tokens  ({diff_total/full_total*100:.1f}% of full)")
    print(f"  Diff summary hint:     {ds_total/topk/1000:.1f}k tokens  ({ds_total/full_total*100:.1f}% of full)")

    # ============================================================
    # BEST PATHS
    # ============================================================
    print("\n" + "=" * 70)
    print(f"TOP {topk} BEST PATHS (by leaf value)")
    print("=" * 70)
    for rank, ps in enumerate(path_stats[:topk], 1):
        root_score_str = f"{-ps['root_value']:.6f}" if ps['root_value'] is not None else 'N/A'
        print(f"\n[Rank {rank}] Leaf score: {-ps['leaf_value']:.6f} | "
              f"Root score: {root_score_str} | "
              f"Improvement: {ps['value_improvement']:.6f} | "
              f"Length: {ps['length']} | "
              f"All use height_sequence_1: {ps['all_use_height_sequence']}")
        print_path_detail(ps["path_ids"], id_to_state, max_states_to_show=5)

    # ============================================================
    # MILESTONE DETAIL FOR BEST PATH
    # ============================================================
    if path_stats:
        best_path = path_stats[0]
        best_states = [id_to_state[sid] for sid in best_path["path_ids"] if sid in id_to_state]
        milestones = extract_milestones(
            best_path["path_ids"],
            id_to_state,
            n_milestones=n_milestones,
            min_improvement=min_improvement,
        )

        print("\n" + "=" * 70)
        print("MILESTONE EXTRACTION FOR BEST PATH")
        print("=" * 70)
        print(f"Original path length: {len(best_states)}")
        print(f"Selected milestones: {len(milestones)}")
        for i, ms in enumerate(milestones):
            raw_score = -ms.get("value", 0) if ms.get("value") is not None else None
            raw_score_str = f"{raw_score:.6f}" if raw_score is not None else 'N/A'
            # Find improvement over previous milestone
            if i > 0:
                prev_val = milestones[i-1].get("value")
                curr_val = ms.get("value")
                if prev_val is not None and curr_val is not None:
                    imp = curr_val - prev_val
                    print(f"  Milestone {i}: score={raw_score_str} (Δ{imp:+.4f}) | "
                          f"timestep={ms.get('timestep')} | code_len={len(ms.get('code', ''))}")
                else:
                    print(f"  Milestone {i}: score={raw_score_str} | "
                          f"timestep={ms.get('timestep')} | code_len={len(ms.get('code', ''))}")
            else:
                print(f"  Milestone {i}: score={raw_score_str} (baseline) | "
                      f"timestep={ms.get('timestep')} | code_len={len(ms.get('code', ''))}")

        # Show diff between consecutive milestones
        print("\n" + "-" * 70)
        print("DIFF SUMMARIES BETWEEN MILESTONES")
        print("-" * 70)
        for i in range(1, len(milestones)):
            prev_code = milestones[i-1].get("code", "")
            curr_code = milestones[i].get("code", "")
            diff = compute_code_diff(prev_code, curr_code)
            summary = summarize_diff(diff)
            diff_tokens = len(diff) // 4
            print(f"  Phase {i-1} → {i}: {summary} (diff: ~{diff_tokens} tokens)")

    # ============================================================
    # PATHS WITH NO HEIGHT_SEQUENCE_1 DEPENDENCY
    # ============================================================
    print("\n" + "=" * 70)
    print("PATHS WITH NO height_sequence_1 DEPENDENCY")
    print("=" * 70)
    no_hs_paths = [p for p in path_stats if p["uses_height_sequence_ratio"] == 0.0]
    print(f"Count: {len(no_hs_paths)} / {len(path_stats)}")
    if no_hs_paths:
        no_hs_paths.sort(key=lambda x: (x["leaf_value"] if x["leaf_value"] is not None else float("-inf")), reverse=True)
        for rank, ps in enumerate(no_hs_paths[:topk], 1):
            leaf_score_str = f"{-ps['leaf_value']:.6f}" if ps['leaf_value'] is not None else 'N/A'
            print(f"\n[Rank {rank}] Leaf score: {leaf_score_str} | "
                  f"Length: {ps['length']}")
            print_path_detail(ps["path_ids"], id_to_state, max_states_to_show=3)
    else:
        print("No fully self-contained paths found.")

    # ============================================================
    # PARTIAL INDEPENDENCE PATHS
    # ============================================================
    print("\n" + "=" * 70)
    print("PARTIALLY SELF-CONTAINED PATHS (hs_ratio <= 50%)")
    print("=" * 70)
    partial_paths = [p for p in path_stats if 0 < p["uses_height_sequence_ratio"] <= 0.5]
    partial_paths.sort(key=lambda x: (x["leaf_value"] if x["leaf_value"] is not None else float("-inf")), reverse=True)
    print(f"Count: {len(partial_paths)} / {len(path_stats)}")
    for rank, ps in enumerate(partial_paths[:topk], 1):
        leaf_score_str = f"{-ps['leaf_value']:.6f}" if ps['leaf_value'] is not None else 'N/A'
        print(f"\n[Rank {rank}] Leaf score: {leaf_score_str} | "
              f"Length: {ps['length']} | hs_ratio: {ps['uses_height_sequence_ratio']:.0%}")
        print_path_detail(ps["path_ids"], id_to_state, max_states_to_show=3)

    # Save detailed results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "tree_analysis.json")

        # Prepare serializable output
        serializable_stats = []
        for ps in path_stats:
            serializable_stats.append({
                "path_length": ps["length"],
                "root_value": ps["root_value"],
                "leaf_value": ps["leaf_value"],
                "best_value": ps["best_value"],
                "value_improvement": ps["value_improvement"],
                "timestep_span": ps["timestep_span"],
                "uses_height_sequence_ratio": ps["uses_height_sequence_ratio"],
                "all_use_height_sequence": ps["all_use_height_sequence"],
                "avg_code_length": ps["avg_code_length"],
                "max_code_length": ps["max_code_length"],
                "path_ids": ps["path_ids"],
            })

        with open(output_path, 'w') as f:
            json.dump({
                "sampler_path": sampler_path,
                "total_states": len(states),
                "root_count": len(roots),
                "leaf_count": len(leaves),
                "total_paths": len(paths),
                "paths": serializable_stats[:topk * 2],  # Save top 2*topk for inspection
            }, f, indent=2)
        print(f"\nSaved detailed analysis to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze PUCT sampler tree unique paths")
    parser.add_argument("--sampler_path", type=str, required=True,
                        help="Path to sampler checkpoint JSON (e.g., puct_sampler_step_000149.json)")
    parser.add_argument("--topk", type=int, default=20,
                        help="Number of top paths to display")
    parser.add_argument("--max_path_length", type=int, default=100,
                        help="Max depth for DFS path extraction")
    parser.add_argument("--output_dir", type=str, default="./tree_analysis",
                        help="Directory to save analysis results")
    parser.add_argument("--milestone_mode", type=str, default="milestone",
                        choices=["full", "milestone", "diff", "diff_summary"],
                        help="Hint format for length estimation")
    parser.add_argument("--n_milestones", type=int, default=4,
                        help="Number of milestones to extract")
    parser.add_argument("--min_improvement", type=float, default=0.005,
                        help="Minimum value improvement to qualify as milestone")
    args = parser.parse_args()

    if not os.path.exists(args.sampler_path):
        print(f"Error: sampler file not found: {args.sampler_path}")
        sys.exit(1)

    analyze_tree(
        sampler_path=args.sampler_path,
        topk=args.topk,
        max_path_length=args.max_path_length,
        output_dir=args.output_dir,
        milestone_mode=args.milestone_mode,
        n_milestones=args.n_milestones,
        min_improvement=args.min_improvement,
    )


if __name__ == "__main__":
    main()
