#!/usr/bin/env python3
"""
Preprocessing script to create initial LoRA adapter for vLLM.

This script must be run BEFORE starting training to ensure vLLM can load
the initial LoRA adapter at startup.

Usage:
    python prepare_lora_init.py --config-path conf/fsdp_lora_vllm.yaml
    
Then run training:
    torchrun --nproc_per_node=8 train_fsdp_lora_vllm.py --config-path conf/fsdp_lora_vllm.yaml
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from areal.api.cli_args import load_expr_config
from areal.experimental.ttt_discover.config import TTTDPPOActorConfig


def create_initial_lora(config: TTTDPPOActorConfig) -> str:
    """Create initial LoRA adapter for vLLM to load at startup."""
    
    # Extract lora_modules path from vllm config
    lora_output_path = "./lora_init"
    lora_modules_str = None
    
    if hasattr(config, 'vllm'):
        import json
        # Try dict first (raw config before __post_init__)
        if isinstance(config.vllm, dict):
            lora_modules_str = config.vllm.get('lora_modules', '')
        # Try dataclass object (after config is processed by __post_init__)
        elif hasattr(config.vllm, 'lora_modules'):
            lora_modules_str = config.vllm.lora_modules
        
        if lora_modules_str:
            try:
                lora_modules = json.loads(lora_modules_str)
                if isinstance(lora_modules, dict):
                    lora_output_path = lora_modules.get('path', lora_output_path)
            except json.JSONDecodeError:
                pass
    
    # Convert to absolute path
    lora_output_path = os.path.abspath(lora_output_path)
    
    # Check if already exists
    adapter_config_path = os.path.join(lora_output_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        print(f"LoRA adapter already exists at {lora_output_path}")
        return lora_output_path
    
    print(f"Creating initial LoRA adapter...")
    print(f"  Base model: {config.path}")
    print(f"  Output path: {lora_output_path}")
    print(f"  LoRA rank: {config.lora_rank}, alpha: {config.lora_alpha}")
    
    # Create parent directory
    parent_dir = os.path.dirname(lora_output_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    
    # Load base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.path,
        torch_dtype="auto",
        device_map="cpu",
    )
    
    # Load tokenizer
    tok_path = config.tokenizer_path or config.path
    print(f"Loading tokenizer from {tok_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    
    # Configure LoRA
    target_modules = config.target_modules if config.target_modules else ["all-linear"]
    target_mods = "all-linear" if target_modules == ["all-linear"] else target_modules
    
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=target_mods,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    print("Applying LoRA...")
    model = get_peft_model(model, lora_config)
    
    # Save
    print(f"Saving LoRA adapter to {lora_output_path}...")
    os.makedirs(lora_output_path, exist_ok=True)
    model.save_pretrained(lora_output_path)
    tokenizer.save_pretrained(lora_output_path)
    
    print(f"✓ LoRA adapter created successfully at {lora_output_path}")
    print(f"  Contents: {os.listdir(lora_output_path)}")
    
    return lora_output_path


def main():
    parser = argparse.ArgumentParser(description="Prepare initial LoRA adapter for vLLM")
    parser.add_argument(
        "--config-path",
        "--config",
        type=str,
        required=True,
        dest="config_path",
        help="Path to the configuration file (e.g., conf/fsdp_lora_vllm.yaml)",
    )
    args = parser.parse_args()
    
    # Load config
    # Try TTTDDistillConfig first (for distillation configs with teacher_path),
    # fall back to TTTDPPOActorConfig for standard training configs.
    try:
        from train_tttd_distill import TTTDDistillConfig
        config, _ = load_expr_config([f"--config={args.config_path}"], TTTDDistillConfig)
    except Exception:
        config, _ = load_expr_config([f"--config={args.config_path}"], TTTDPPOActorConfig)
    
    if not config.use_lora:
        print("LoRA is not enabled in config (use_lora=false). Nothing to do.")
        return
    
    # Create LoRA
    lora_path = create_initial_lora(config)
    
    print(f"\n✓ Preparation complete!")
    print(f"  LoRA adapter ready at: {lora_path}")
    print(f"\nYou can now start training with:")
    print(f"  torchrun --nproc_per_node=8 train_fsdp_lora_vllm.py --config-path {args.config_path}")


if __name__ == "__main__":
    main()
