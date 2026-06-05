#!/usr/bin/env python3
"""
Merge per-rank eval reward files into a single analysis.

Usage:
    python merge_eval_rewards.py <eval_output_dir>

Example:
    python merge_eval_rewards.py ./outputs_ac1_distill_comparisons/tttd-distill-J-fromscratch/trial0
"""

import sys
import os
import json
import glob
import numpy as np


def merge_rewards_for_label(eval_dir: str, label: str):
    """Find and merge all rank reward files for a given model label."""
    pattern = os.path.join(eval_dir, f"eval_rewards_{label}_rank*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None

    all_rewards = []
    for f in files:
        with open(f, 'r') as fp:
            all_rewards.extend(json.load(fp))
    return all_rewards


def analyze_distribution(rewards):
    """Compute statistics and percentile distribution."""
    arr = np.array(rewards)
    total = len(arr)
    nonzero = arr[arr != 0.0]

    stats = {
        "n": total,
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "zero_pct": float(np.sum(arr == 0.0) / total * 100),
    }

    if len(nonzero) > 0:
        stats.update({
            "p25": float(np.percentile(nonzero, 25)),
            "p50": float(np.percentile(nonzero, 50)),
            "p75": float(np.percentile(nonzero, 75)),
            "p90": float(np.percentile(nonzero, 90)),
            "p95": float(np.percentile(nonzero, 95)),
            "p99": float(np.percentile(nonzero, 99)),
        })
    else:
        stats.update({"p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0})

    return stats


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <eval_output_dir>")
        sys.exit(1)

    eval_dir = sys.argv[1]
    if not os.path.isdir(eval_dir):
        print(f"Error: {eval_dir} is not a directory")
        sys.exit(1)

    # Auto-discover labels from filenames
    pattern = os.path.join(eval_dir, "eval_rewards_*_rank*.json")
    files = glob.glob(pattern)
    labels = sorted(set(
        os.path.basename(f).split('_rank')[0].replace('eval_rewards_', '')
        for f in files
    ))

    if not labels:
        print(f"No eval_rewards_*_rank*.json files found in {eval_dir}")
        sys.exit(1)

    print(f"Found labels: {labels}\n")

    results = {}
    for label in labels:
        rewards = merge_rewards_for_label(eval_dir, label)
        if rewards is None:
            continue
        stats = analyze_distribution(rewards)
        results[label] = {"rewards": rewards, "stats": stats}

    # Print comparison table
    print("=" * 100)
    print(f"{'Label':20s} | {'n':>6s} | {'max':>7s} | {'mean':>7s} | {'zero':>6s} | {'p25':>7s} | {'p50':>7s} | {'p75':>7s} | {'p90':>7s}")
    print("=" * 100)
    for label in sorted(results.keys()):
        s = results[label]["stats"]
        print(
            f"{label:20s} | {s['n']:6d} | {s['max']:7.4f} | {s['mean']:7.4f} | "
            f"{s['zero_pct']:5.1f}% | {s['p25']:7.4f} | {s['p50']:7.4f} | {s['p75']:7.4f} | {s['p90']:7.4f}"
        )
    print("=" * 100)

    # Save merged results
    output_path = os.path.join(eval_dir, "eval_rewards_merged.json")
    with open(output_path, 'w') as f:
        json.dump({
            label: {
                "stats": results[label]["stats"],
                "rewards": results[label]["rewards"],
            }
            for label in results
        }, f, indent=2)
    print(f"\nMerged results saved to {output_path}")


if __name__ == "__main__":
    main()
