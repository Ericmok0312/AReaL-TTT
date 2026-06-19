#!/usr/bin/env python3
"""
Adaptive milestone hint extraction from PUCT sampler tree.

Uses a general, problem-agnostic rule to select milestones from any root-to-leaf
path given a context budget. Selection is based purely on reward improvement,
not value density.

Usage:
    python areal/experimental/ttt_discover/examples/extract_milestone_hints_v2.py \
        --sampler_path <path_to_sampler.json> \
        --max_hint_tokens 4000 \
        --min_improvement 0.001 \
        --topk_paths 20 \
        --output ./adaptive_hints.json
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


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def clean_code(code: str) -> str:
    code = code.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


def compute_state_value(state: dict) -> float:
    return state.get("value") if state.get("value") is not None else float("-inf")


def select_milestones_by_improvement(
    path_states: list[dict],
    max_hint_tokens: int,
    min_improvement: float,
    preserve_root_leaf: bool = True,
) -> list[dict]:
    """
    General milestone selection rule: pure improvement-based.

    Strategy:
    1. Compute improvement between every adjacent pair of states.
    2. Filter out improvements below min_improvement.
    3. Sort candidates by improvement (descending).
    4. Greedy selection: pick highest-improvement candidates until token budget
       is exhausted. Always keep root and leaf if preserve_root_leaf=True.
    5. Return selected states in path order.

    Why pure improvement (not improvement/length)?
    - A large code rewrite with big improvement is more informative for teacher
      than a tiny tweak with high "density".
    - Context budget limits total length, but within the budget we want the
      maximum total improvement shown to teacher.
    """
    n = len(path_states)
    if n <= 2:
        return path_states

    # Step 1: Compute improvements
    candidates = []
    for i in range(1, n):
        prev_state = path_states[i - 1]
        curr_state = path_states[i]
        prev_val = compute_state_value(prev_state)
        curr_val = compute_state_value(curr_state)
        improvement = curr_val - prev_val

        if improvement < min_improvement:
            continue

        code = clean_code(curr_state.get("code", ""))
        code_tokens = estimate_tokens(code)

        candidates.append({
            "index": i,
            "state": curr_state,
            "improvement": improvement,
            "code_tokens": code_tokens,
        })

    # Step 2: Determine fixed cost
    root_code = clean_code(path_states[0].get("code", ""))
    leaf_code = clean_code(path_states[-1].get("code", ""))
    root_tokens = estimate_tokens(root_code)
    leaf_tokens = estimate_tokens(leaf_code)

    fixed_cost = root_tokens + leaf_tokens if preserve_root_leaf else 0
    remaining_budget = max_hint_tokens - fixed_cost

    if remaining_budget < 0:
        return [path_states[0], path_states[-1]] if preserve_root_leaf else [path_states[-1]]

    # Step 3: Sort by pure improvement (descending)
    candidates.sort(key=lambda x: x["improvement"], reverse=True)

    selected_indices = set()
    if preserve_root_leaf:
        selected_indices.add(0)
        selected_indices.add(n - 1)

    current_tokens = fixed_cost
    for cand in candidates:
        idx = cand["index"]
        if idx in selected_indices:
            continue
        if current_tokens + cand["code_tokens"] > max_hint_tokens:
            break
        selected_indices.add(idx)
        current_tokens += cand["code_tokens"]

    # Step 4: Return in path order
    selected_indices = sorted(selected_indices)
    return [path_states[i] for i in selected_indices]


def format_milestone(state: dict, phase_idx: int, prev_state: dict | None = None) -> dict:
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
        "code_tokens": estimate_tokens(code),
        "has_height_sequence_1": "height_sequence_1" in code,
        "construction": state.get("construction"),
    }

    if prev_state is not None:
        prev_value = compute_state_value(prev_state)
        curr_value = compute_state_value(state)
        if curr_value > float("-inf") and prev_value > float("-inf"):
            milestone["improvement"] = curr_value - prev_value

    return milestone


def analyze_path(path_states: list[dict], path_rank: int) -> dict:
    root = path_states[0]
    leaf = path_states[-1]
    root_val = compute_state_value(root)
    leaf_val = compute_state_value(leaf)
    values = [compute_state_value(s) for s in path_states]
    best_val = max(values)
    hs_count = sum(1 for s in path_states if "height_sequence_1" in s.get("code", ""))

    return {
        "path_rank": path_rank,
        "path_length": len(path_states),
        "root_raw_score": -root_val if root_val > float("-inf") else None,
        "leaf_raw_score": -leaf_val if leaf_val > float("-inf") else None,
        "best_raw_score": -best_val if best_val > float("-inf") else None,
        "total_improvement": leaf_val - root_val if (leaf_val > float("-inf") and root_val > float("-inf")) else None,
        "hs_dependency_ratio": hs_count / max(len(path_states), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler_path", type=str, required=True)
    parser.add_argument("--topk_paths", type=int, default=20)
    parser.add_argument("--max_hint_tokens", type=int, default=4000)
    parser.add_argument("--min_improvement", type=float, default=0.001)
    parser.add_argument("--max_depth", type=int, default=100)
    parser.add_argument("--output", type=str, default="./adaptive_hints.json")
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

    def path_leaf_value(path_ids):
        leaf = id_to_state.get(path_ids[-1], {})
        return leaf.get("value") if leaf else float("-inf")

    paths.sort(key=path_leaf_value, reverse=True)

    hint_data = {
        "source_sampler": args.sampler_path,
        "total_paths": len(paths),
        "extracted_paths": min(args.topk_paths, len(paths)),
        "config": {
            "max_hint_tokens": args.max_hint_tokens,
            "min_improvement": args.min_improvement,
        },
        "paths": [],
    }

    for rank, path_ids in enumerate(paths[:args.topk_paths], 1):
        path_states = [id_to_state[sid] for sid in path_ids if sid in id_to_state]
        if not path_states:
            continue

        milestones_raw = select_milestones_by_improvement(
            path_states,
            max_hint_tokens=args.max_hint_tokens,
            min_improvement=args.min_improvement,
            preserve_root_leaf=True,
        )

        milestones = []
        for i, ms in enumerate(milestones_raw):
            prev = milestones_raw[i - 1] if i > 0 else None
            milestones.append(format_milestone(ms, i, prev))

        path_info = analyze_path(path_states, rank)
        path_info["num_milestones"] = len(milestones)
        path_info["milestones"] = milestones
        path_info["total_hint_tokens"] = sum(m["code_tokens"] for m in milestones)
        hint_data["paths"].append(path_info)

    with open(args.output, 'w') as f:
        json.dump(hint_data, f, indent=2)

    print(f"\nSaved {hint_data['extracted_paths']} paths to {args.output}")

    nums = [p["num_milestones"] for p in hint_data["paths"]]
    tokens = [p["total_hint_tokens"] for p in hint_data["paths"]]
    print(f"Milestones per path: min={min(nums)}, max={max(nums)}, mean={sum(nums)/len(nums):.1f}")
    print(f"Hint tokens per path: min={min(tokens):.0f}, max={max(tokens):.0f}, mean={sum(tokens)/len(tokens):.0f}")

    if hint_data["paths"]:
        s = hint_data["paths"][0]
        print(f"\nSample (Path #{s['path_rank']}):")
        print(f"  Leaf score: {s['leaf_raw_score']:.6f} | Milestones: {s['num_milestones']} | Tokens: {s['total_hint_tokens']}")
        for ms in s["milestones"]:
            imp = ms.get("improvement")
            imp_str = f" (Δ{imp:+.4f})" if imp is not None else ""
            print(f"    Phase {ms['phase']}: score={ms['raw_score']:.6f}{imp_str} | tokens={ms['code_tokens']} | hs={ms['has_height_sequence_1']}")


if __name__ == "__main__":
    main()
