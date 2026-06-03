#!/usr/bin/env python3
"""
Extract milestone hints from PUCT sampler tree for teacher distillation.

For each top root-to-leaf path, extracts key milestone states (based on value
improvement thresholds) and saves their full code as structured hints.

Usage:
    python areal/experimental/ttt_discover/examples/extract_milestone_hints.py \
        --sampler_path ./outputs_ac1_async_v2/tttd-async-qwen3-8b-ac1/trial0/sampler/puct_sampler_step_000149.json \
        --topk_paths 20 \
        --n_milestones 4 \
        --min_improvement 0.005 \
        --output ./milestone_hints.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict


def load_sampler_checkpoint(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def build_maps(states: list[dict]):
    id_to_state = {s["id"]: s for s in states if s.get("id")}
    children_map = defaultdict(list)
    for s in states:
        sid = s.get("id")
        if not sid:
            continue
        parents = s.get("parents", [])
        if parents:
            pid = parents[0].get("id")
            if pid:
                children_map[pid].append(sid)
    return id_to_state, dict(children_map)


def extract_all_paths(root_ids: list[str], children_map: dict, max_depth: int = 100):
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


def extract_milestones(path_ids: list[str], id_to_state: dict, n_milestones: int, min_improvement: float) -> list[dict]:
    states = [id_to_state[sid] for sid in path_ids if sid in id_to_state]
    if len(states) <= n_milestones:
        return states

    improvements = []
    for i in range(1, len(states)):
        prev_val = states[i - 1].get("value")
        curr_val = states[i].get("value")
        if prev_val is not None and curr_val is not None:
            improvements.append((i, curr_val - prev_val))

    improvements.sort(key=lambda x: x[1], reverse=True)

    selected_indices = [0]
    for idx, imp in improvements:
        if imp >= min_improvement and idx not in selected_indices:
            selected_indices.append(idx)
        if len(selected_indices) >= n_milestones - 1:
            break

    if len(selected_indices) < n_milestones - 1:
        n_needed = n_milestones - 1 - len(selected_indices)
        step = len(states) // (n_needed + 1)
        for i in range(1, n_needed + 1):
            idx = i * step
            if idx not in selected_indices and idx < len(states) - 1:
                selected_indices.append(idx)

    if (len(states) - 1) not in selected_indices:
        selected_indices.append(len(states) - 1)

    selected_indices = sorted(set(selected_indices))
    return [states[i] for i in selected_indices]


def clean_code(code: str) -> str:
    """Strip markdown code block wrappers if present."""
    code = code.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


def format_milestone(state: dict, phase_idx: int, prev_state: dict | None = None) -> dict:
    """Format a single milestone into a structured dict."""
    value = state.get("value")
    raw_score = -value if value is not None else None
    code = clean_code(state.get("code", ""))

    milestone = {
        "phase": phase_idx,
        "state_id": state.get("id"),
        "timestep": state.get("timestep"),
        "raw_score": raw_score,
        "value": value,
        "code": code,
        "code_length": len(code),
        "has_height_sequence_1": "height_sequence_1" in code,
    }

    if prev_state is not None:
        prev_value = prev_state.get("value")
        if prev_value is not None and value is not None:
            milestone["improvement"] = value - prev_value
            milestone["improvement_from_previous"] = value - prev_value

    return milestone


def format_path_hint(path_states: list[dict], path_rank: int) -> dict:
    """Format a complete path into a hint structure."""
    milestones = []
    for i, state in enumerate(path_states):
        prev = path_states[i - 1] if i > 0 else None
        milestones.append(format_milestone(state, i, prev))

    root = path_states[0]
    leaf = path_states[-1]
    root_value = root.get("value")
    leaf_value = leaf.get("value")

    return {
        "path_rank": path_rank,
        "path_length": len(path_states),
        "num_milestones": len(path_states),
        "root_raw_score": -root_value if root_value is not None else None,
        "leaf_raw_score": -leaf_value if leaf_value is not None else None,
        "total_improvement": (leaf_value - root_value) if (leaf_value is not None and root_value is not None) else None,
        "milestones": milestones,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler_path", type=str, required=True)
    parser.add_argument("--topk_paths", type=int, default=20)
    parser.add_argument("--n_milestones", type=int, default=4)
    parser.add_argument("--min_improvement", type=float, default=0.005)
    parser.add_argument("--max_depth", type=int, default=100)
    parser.add_argument("--output", type=str, default="./milestone_hints.json")
    args = parser.parse_args()

    if not os.path.exists(args.sampler_path):
        print(f"Error: file not found: {args.sampler_path}")
        sys.exit(1)

    checkpoint = load_sampler_checkpoint(args.sampler_path)
    states = checkpoint.get("states", [])
    print(f"Loaded {len(states)} states")

    id_to_state, children_map = build_maps(states)
    root_ids = [s["id"] for s in states if s.get("id") and not s.get("parents")]
    print(f"Found {len(root_ids)} roots")

    paths = extract_all_paths(root_ids, children_map, max_depth=args.max_depth)
    print(f"Found {len(paths)} unique root-to-leaf paths")

    # Score paths by leaf value
    def path_leaf_value(path_ids):
        leaf = id_to_state.get(path_ids[-1], {})
        return leaf.get("value") if leaf else float("-inf")

    paths.sort(key=path_leaf_value, reverse=True)

    # Extract milestone hints for top-k paths
    hint_data = {
        "source_sampler": args.sampler_path,
        "total_paths": len(paths),
        "extracted_paths": min(args.topk_paths, len(paths)),
        "n_milestones_config": args.n_milestones,
        "min_improvement_config": args.min_improvement,
        "paths": [],
    }

    total_hint_chars = 0
    total_milestones = 0

    for rank, path_ids in enumerate(paths[:args.topk_paths], 1):
        milestone_states = extract_milestones(
            path_ids, id_to_state,
            n_milestones=args.n_milestones,
            min_improvement=args.min_improvement,
        )
        path_hint = format_path_hint(milestone_states, rank)
        hint_data["paths"].append(path_hint)

        for ms in path_hint["milestones"]:
            total_hint_chars += ms["code_length"]
        total_milestones += len(path_hint["milestones"])

    # Save
    with open(args.output, 'w') as f:
        json.dump(hint_data, f, indent=2)

    print(f"\nSaved {hint_data['extracted_paths']} paths to {args.output}")
    print(f"Total milestones: {total_milestones}")
    print(f"Avg milestones per path: {total_milestones / hint_data['extracted_paths']:.1f}")
    print(f"Avg hint size per path: {total_hint_chars / hint_data['extracted_paths'] / 1000:.1f}k chars "
          f"(~{total_hint_chars / hint_data['extracted_paths'] / 4 / 1000:.1f}k tokens)")

    # Print sample
    if hint_data["paths"]:
        sample = hint_data["paths"][0]
        print(f"\nSample hint (Path #{sample['path_rank']}):")
        print(f"  Root score: {sample['root_raw_score']:.6f}")
        print(f"  Leaf score: {sample['leaf_raw_score']:.6f}")
        print(f"  Milestones: {sample['num_milestones']}")
        for ms in sample["milestones"]:
            imp = ms.get("improvement")
            imp_str = f" (Δ{imp:+.4f})" if imp is not None else ""
            print(f"    Phase {ms['phase']}: score={ms['raw_score']:.6f}{imp_str} | "
                  f"code_len={ms['code_length']} | uses_hs={ms['has_height_sequence_1']}")


if __name__ == "__main__":
    main()
