# TTT-Discover for AReaL (Experimental)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Status](https://img.shields.io/badge/Status-Experimental-orange)](https://github.com/areal-team/areal/tree/main/experimental)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green)](https://www.python.org/)

**Discovery-based Reinforcement Learning Trainer for AReaL Framework**

This is an experimental implementation of [TTT-Discover](https://arxiv.org/abs/xxxx.xxxxx) (Test-Time Training for Discovery) integrated into the AReaL ecosystem. It extends standard PPO with **Entropic Objective** and **PUCT-based State Selection** for discovering high-reward solutions in single-problem optimization scenarios.

⚠️ **Experimental Notice**: This module is under active development and may introduce breaking changes. APIs are not yet stabilized for production use.

---

## 🔬 Core Concepts

### Entropic Objective
Unlike standard RL that maximizes expected reward, TTT-Discover maximizes the best reward via entropic regularization:

$$J_\beta(\theta) = \mathbb{E}[\log \mathbb{E}[e^{\beta \cdot R(s,a)}]]$$

With advantage weights:
$$w_\beta(a) = \frac{e^{\beta \cdot R(s,a)}}{\mathbb{E}[e^{\beta \cdot R(s,a)}]}$$

### PUCT State Selection
Replaces random state reuse with tree-search inspired selection:

$$\text{score}(s) = Q(s) + c \cdot P(s) \cdot \sqrt{\frac{1+T}{1+n(s)}}$$

Where $Q(s)$ uses **max reward** (not mean) to prioritize promising trajectories.

---

## 🚀 Quick Start

### Prerequisites

Ensure AReaL is installed with experimental dependencies:

```bash
# Install AReaL with all dependencies
pip install -e ".[dev]"
```

---

## 📝 Training Workflow

### Step 1: Initialize LoRA Adapter (REQUIRED)

**CRITICAL**: You must initialize the LoRA adapter BEFORE starting training because vLLM loads it at startup before the training script runs.

```bash
# Initialize LoRA for Qwen3-8B on Circle Packing
uv run python areal/experimental/ttt_discover/examples/prepare_lora_init.py \
    --config areal/experimental/ttt_discover/examples/conf/fsdp_lora_vllm_qwen3_8b.yaml
```

This creates the LoRA adapter at the path specified in your config (e.g., `./lora_init_qwen3_8b`).

**Note**: This step only needs to be done once per experiment. The adapter is reused across training restarts.

---

### Step 2: Start Training

```bash
# Launch training with local launcher
uv run python -m areal.infra.launcher.local \
    areal/experimental/ttt_discover/examples/train_fsdp_lora_vllm_v2.py \
    --config areal/experimental/ttt_discover/examples/conf/fsdp_lora_vllm_qwen3_8b.yaml
```

Or using `torchrun` directly:

```bash
torchrun --nproc_per_node=8 \
    areal/experimental/ttt_discover/examples/train_fsdp_lora_vllm_v2.py \
    --config areal/experimental/ttt_discover/examples/conf/fsdp_lora_vllm_qwen3_8b.yaml
```

---

### Available Configurations

| Config | Model | Environment | Description |
|--------|-------|-------------|-------------|
| `fsdp_lora_vllm.yaml` | Qwen2.5-1.5B | Circle Packing (n=26) | Small model for testing |
| `fsdp_lora_vllm_qwen3_8b.yaml` | Qwen3-8B | Circle Packing (n=26) | Main config for H20/H100 |
| `fsdp_lora_vllm_qwen3_32b.yaml` | Qwen3-32B | Circle Packing (n=26) | Large model |

---

## 📊 Training Visualization

### Automatic Data Recording

Training automatically records reward distributions at specified steps for visualization. Configure in your YAML:

```yaml
save_steps: [0, 9, 24, 49]  # Steps to save snapshots
```

Training history is saved to:
```
./outputs/{experiment_name}/{trial_name}/training_history.pkl
```

### Generate Training Dynamics Plot

After training completes, generate the paper-style KDE distribution plot:

```bash
# Basic usage
uv run python areal/experimental/ttt_discover/generate_plot.py \
    --history_path ./outputs/tttd-qwen3-8b-cp/trial0/training_history.pkl \
    --benchmark_value 2.635983

# With all options
uv run python areal/experimental/ttt_discover/generate_plot.py \
    --history_path ./outputs/tttd-qwen3-8b-cp/trial0/training_history.pkl \
    --steps 0 9 24 49 \
    --benchmark_value 2.635983 \
    --benchmark_label "Best Human" \
    --xlabel "Sum of Radii (higher is better →)" \
    --output_path ./training_dynamics.png \
    --show_progression
```

See [VISUALIZATION.md](./VISUALIZATION.md) for detailed documentation.

---

## 🧪 Environments

TTT-Discover supports multiple optimization environments:

| Environment | Type | Reward | Description |
|-------------|------|--------|-------------|
| `cp` | Circle Packing | Sum of radii | Pack n circles in unit circle |
| `ac1` | Inequalities | -budget | Discover inequality bounds |
| `trimul` | GPU Kernel | Runtime μs | Optimize triangular matrix multiplication |
| `erdos` | Graph Theory | Construction size | Erdos construction problems |

Configure in your YAML:
```yaml
sampler:
  env_type: cp  # Environment type
  n_item: 26    # Problem size (e.g., 26 circles)
```

---

## ⚙️ Configuration Guide

### Key Parameters

```yaml
# TTT-Discover specific
max_steps: 50                    # Training steps (not epochs)
adv_estimator: entropic          # Use entropic objective
adv_estimator_beta: 1.0          # Temperature for exploration

# LoRA
use_lora: true
lora_rank: 32
lora_alpha: 16

# PUCT Sampler
sampler:
  batch_size: 8                  # Parents per step
  c_puct: 1.5                    # Exploration constant
  max_states: 10000              # State buffer size

# Generation
gconfig:
  n_samples: 16                  # Rollouts per parent
  max_new_tokens: 16384          # Max code length
  temperature: 1.0               # Sampling temperature
```

### Customizing LoRA Path

Edit your config file to change the LoRA adapter path:

```yaml
vllm:
  lora_modules: '{"name": "tttd_lora_adapter", "path": "./your_custom_lora_path", "base_model_name": "${path}"}'
```

Remember to run `prepare_lora_init.py` with the same config after changing the path.

---

## 🐛 Troubleshooting

### "Initial LoRA adapter not found"

**Cause**: You forgot to run `prepare_lora_init.py` before training.

**Solution**:
```bash
python areal/experimental/ttt_discover/examples/prepare_lora_init.py \
    --config <your_config.yaml>
```

### "Ranks have INCONSISTENT sampler state"

**Cause**: Distributed synchronization issue.

**Solution**: This is usually a warning, not fatal. Check if training continues normally.

### vLLM fails to load LoRA

**Cause**: LoRA adapter created with different model or config.

**Solution**: Delete the old adapter and re-run `prepare_lora_init.py`:
```bash
rm -rf ./lora_init_qwen3_8b
python areal/experimental/ttt_discover/examples/prepare_lora_init.py \
    --config <your_config.yaml>
```

---

## 📚 Additional Documentation

- [VISUALIZATION.md](./VISUALIZATION.md) - Training visualization and plotting
- [areal/README.md](../../areal/README.md) - AReaL framework documentation

---

## 📄 Citation

If you use TTT-Discover in your research, please cite:

```bibtex
@article{tttdiscover2024,
  title={Learning to Discover at Test Time},
  author={[Authors]},
  journal={arXiv preprint},
  year={2024}
}
```

---

## 🤝 Contributing

This is an experimental module. For contributions or issues:

1. Ensure changes work with the `LocalLauncher` workflow
2. Test on multi-GPU setups if modifying distributed code
3. Update this README if adding new features

---

## 📜 License

Apache 2.0 - See [LICENSE](../../../LICENSE) for details.
