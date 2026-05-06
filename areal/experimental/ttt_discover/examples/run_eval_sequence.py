#!/usr/bin/env python3
"""
Sequential multi-model evaluation launcher for TTT-Discover.

Reads model paths from the YAML config, runs ``eval_tttd_single_v2.py`` for each
model in order, and finally aggregates all per-model JSONs into a single
``eval_comparison.json`` under the config's output folder.

Usage
-----
::

    python run_eval_sequence.py \
        -c conf/fsdp_lora_vllm_ac1_qwen3_8b_distill.yaml \
        --models teacher,student \
        [any extra args forwarded to eval_tttd_single_v2.py]

Supported model labels (mapped from YAML keys):
  * ``teacher``   → ``teacher_path`` (or ``teacher_lora_path``)
  * ``student``   → ``student_lora_path``
  * ``base``      → ``actor.path`` (base model without LoRA)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

# Path to the single-model eval script (relative to this file)
EVAL_SINGLE_PY = Path(__file__).with_name("eval_tttd_single_v2.py")


def resolve_yaml(yaml_path: str) -> tuple[OmegaConf, str]:
    """Load YAML with OmegaConf interpolation and return (config, output_dir)."""
    conf = OmegaConf.load(yaml_path)
    # Resolve all interpolations so we get concrete strings
    conf = OmegaConf.to_container(conf, resolve=True)
    conf = OmegaConf.create(conf)

    output_dir = conf.get("cluster", {}).get("fileroot", "")
    if not output_dir:
        output_dir = conf.get("saver", {}).get("fileroot", "./outputs")
    output_dir = os.path.expanduser(output_dir)
    return conf, output_dir


def get_model_path(conf: OmegaConf, label: str) -> str | None:
    """Map a user-friendly label to the actual checkpoint path in the config."""
    if label == "teacher":
        # Prefer teacher_lora_path if present, fall back to teacher_path
        path = conf.get("teacher_lora_path", conf.get("teacher_path", ""))
    elif label == "student":
        path = conf.get("student_lora_path", "")
    elif label == "base":
        path = conf.get("actor", {}).get("path", "")
    else:
        # Allow arbitrary dotted keys, e.g. "teacher_path"
        path = conf.get(label, "")

    return str(path) if path else None


def run_single_eval(lora_path: str, config_path: str, extra_argv: list[str]) -> None:
    """Invoke ``eval_tttd_single_v2.py`` through AReaL's local launcher."""
    # We must go through the launcher so that torch.distributed env vars
    # (RANK, WORLD_SIZE, etc.) and AREAL_SPMD_MODE are set properly.
    # NOTE: --lora-path is passed via env var because launcher's parse_cli_args
    # treats unknown --flags as Hydra overrides and crashes.
    cmd = [
        sys.executable,
        "-m", "areal.infra.launcher.local",
        str(EVAL_SINGLE_PY),
        "--config", config_path,
        *extra_argv,
    ]
    env = os.environ.copy()
    env["EVAL_LORA_PATH"] = lora_path
    print("=" * 70)
    print("[run_eval_sequence] Running:\n  ", " ".join(cmd))
    print(f"[run_eval_sequence] EVAL_LORA_PATH={lora_path}")
    print("=" * 70)
    subprocess.run(cmd, check=True, env=env)


def aggregate_results(output_dir: str, model_paths: dict[str, str]) -> dict:
    """Read per-model ``eval_single_*.json`` files and build comparison dict."""
    results = {}
    for label, path in model_paths.items():
        # The JSON name written by eval_tttd_single.py is:
        #   eval_single_{basename(lora_path)}.json
        basename = os.path.basename(path.rstrip("/"))
        pattern = f"eval_single_{basename}.json"
        candidate = os.path.join(output_dir, pattern)

        if os.path.isfile(candidate):
            with open(candidate) as f:
                data = json.load(f)
            results[label] = data
            print(f"[run_eval_sequence] Loaded results for '{label}' from {candidate}")
        else:
            print(f"[run_eval_sequence] WARNING: result file not found: {candidate}")
            results[label] = {
                "model": label,
                "error": f"Result file not found: {candidate}",
            }
    return results


def save_comparison(results: dict, output_dir: str, filename: str = "eval_comparison.json") -> None:
    """Save aggregated comparison JSON."""
    out_path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[run_eval_sequence] Comparison saved to {out_path}")

    # Pretty-print summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPARISON (Initial States)")
    print("=" * 70)
    for label, r in results.items():
        if "error" in r:
            print(f"{label:10s} | ERROR: {r['error']}")
        else:
            print(
                f"{label:10s} | max_reward={r.get('max_reward', 0):.4f} | "
                f"mean_reward={r.get('mean_reward', 0):.4f} | "
                f"rollouts={r.get('global_rollouts', 0)} | "
                f"children={r.get('n_children', 0)} | failed={r.get('n_failed', 0)}"
            )
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential multi-model evaluation for TTT-Discover",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate teacher then student (default)
  python run_eval_sequence.py -c conf/fsdp_lora_vllm_ac1_qwen3_8b_distill.yaml

  # Evaluate only teacher
  python run_eval_sequence.py -c conf/...yaml --models teacher

  # Evaluate teacher + student with extra CLI overrides
  python run_eval_sequence.py -c conf/...yaml --models teacher,student +seed=42
""",
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Path to the YAML config file (e.g. fsdp_lora_vllm_ac1_qwen3_8b_distill.yaml)",
    )
    parser.add_argument(
        "--models",
        default="teacher,student",
        help="Comma-separated list of model labels to evaluate (default: teacher,student)",
    )
    parser.add_argument(
        "--output-name",
        default="eval_comparison.json",
        help="Filename for the aggregated comparison JSON (default: eval_comparison.json)",
    )
    # Everything else is forwarded to eval_tttd_single.py
    return parser.parse_known_args()


def main() -> None:
    known, extra_argv = parse_args()

    conf, output_dir = resolve_yaml(known.config)
    print(f"[run_eval_sequence] Output directory: {output_dir}")

    labels = [lbl.strip() for lbl in known.models.split(",") if lbl.strip()]
    print(f"[run_eval_sequence] Models to evaluate: {labels}")

    model_paths = {}
    for lbl in labels:
        path = get_model_path(conf, lbl)
        if not path:
            raise ValueError(
                f"Model label '{lbl}' could not be resolved to a path from config. "
                f"Available keys: teacher_path, teacher_lora_path, student_lora_path, actor.path"
            )
        resolved = os.path.expanduser(path)
        if not os.path.isdir(resolved) and not os.path.isdir(path):
            # base model (actor.path) is usually a HuggingFace hub name, not a local dir.
            # For base model we warn but do not abort, because eval_tttd_single.py
            # will load it via from_pretrained.
            if lbl == "base":
                print(f"[run_eval_sequence] WARNING: '{lbl}' path '{resolved}' is not a local dir. "
                      "Assuming HuggingFace hub identifier.")
            else:
                raise ValueError(f"Model path for '{lbl}' does not exist: {resolved}")
        model_paths[lbl] = resolved

    # Run evaluation for each model sequentially
    for lbl, path in model_paths.items():
        print(f"\n[run_eval_sequence] >>> Evaluating '{lbl}' from {path}")
        run_single_eval(path, known.config, extra_argv)
        print(f"[run_eval_sequence] <<< Finished '{lbl}'")

    # Aggregate results
    print("\n[run_eval_sequence] Aggregating results...")
    results = aggregate_results(output_dir, model_paths)
    save_comparison(results, output_dir, known.output_name)


if __name__ == "__main__":
    main()
