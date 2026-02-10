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


### Environments
Enviroments to execute model output and calculate rewards
---

## 🚀 Quick Start

### Installation

```bash
# 1. 确保已安装 AReaL 主框架 (Ensure AReaL is installed)
pip install -e .

```