# TTT-Discover with FSDP + LoRA + Group Rollout

这个目录包含了 TTT-Discover 的优化训练实现，结合了：

1. **TTTDActor**: FSDP-based 训练 + 熵目标函数
2. **LoRA**: 低秩适应，高效微调
3. **Group Rollout**: 8 parents × 64 rollouts = 512 总 rollout
4. **External PUCTSampler**: 外部状态管理，避免重复更新

## 文件结构

```
examples/
├── train_fsdp_lora.py          # SGLang 版本 (推荐，稳定)
├── train_fsdp_lora_vllm.py     # vLLM 版本 (分布式 rollout)
├── conf/
│   ├── fsdp_lora.yaml          # SGLang 配置
│   └── fsdp_lora_vllm.yaml     # vLLM 配置
├── train_optimized.py          # 纯 group rollout 版本 (无 FSDP)
├── train_with_group.py         # 早期版本 (已弃用)
└── train_group_external.py     # 手动 group 版本 (无优化)
```

## 版本对比

| 特性 | SGLang (`train_fsdp_lora.py`) | vLLM (`train_fsdp_lora_vllm.py`) |
|------|------------------------------|----------------------------------|
| **推理引擎** | SGLang | vLLM |
| **Rollout 策略** | 单 rank (rank 0) | 分布式 (每个 DP rank) |
| **Broadcast** | 需要 (`bcast_and_split_from_rank0`) | 不需要 |
| **代码复杂度** | 较复杂 (需处理 broadcast) | 较简单 |
| **性能** | 稳定，但 rank 0 可能成为瓶颈 | 更高吞吐，无单点瓶颈 |
| **已知问题** | 多 rank rollout 有并发问题 | 可能OOM (vLLM内存管理) |
| **推荐场景** | 生产环境，稳定性优先 | 追求极致性能 |

### 选择建议

- **选 SGLang**：如果你追求稳定，或遇到 vLLM 的内存/并发问题
- **选 vLLM**：如果你需要最大化 throughput，且能处理分布式复杂性

## 使用方法

### SGLang 版本

```bash
torchrun --nproc_per_node=8 train_fsdp_lora.py \
    --config-path conf/fsdp_lora.yaml \
    --experiment-name my_experiment \
    --trial-name run1
```

### vLLM 版本

```bash
torchrun --nproc_per_node=8 train_fsdp_lora_vllm.py \
    --config-path conf/fsdp_lora_vllm.yaml \
    --experiment-name my_experiment \
    --trial-name run1
```

## 实现你的环境

### 1. 创建环境类

首先，在 `areal/experimental/ttt_discover/tasks/` 下创建你的任务环境：

```python
# areal/experimental/ttt_discover/tasks/my_task/task.py
from areal.experimental.ttt_discover.env import BaseEnv, EnvResult

class MyTaskEnv(BaseEnv):
    def get_prompt(self, state) -> str:
        # 返回 prompt
        pass
    
    def execute(self, code: str, state) -> EnvResult:
        # 执行代码并返回结果
        pass
    
    def create_state(self, parent_state, code, reward, result, timestep):
        # 创建新状态
        pass
```

### 2. 修改训练脚本

根据选择的版本，修改对应的训练脚本：

**SGLang 版本** (`train_fsdp_lora.py`)：

```python
# 在 main() 函数中替换:
from areal.experimental.ttt_discover.tasks.my_task.task import MyTaskEnv
env = MyTaskEnv()
```

**vLLM 版本** (`train_fsdp_lora_vllm.py`)：

```python
# 在 main() 函数中替换:
from areal.experimental.ttt_discover.tasks.my_task.task import MyTaskEnv
env = MyTaskEnv()
```

```python
# 在 main() 函数中替换:
from areal.experimental.ttt_discover.tasks.my_task.task import MyTaskEnv
env = MyTaskEnv()
```

### 3. 运行训练

根据选择的版本运行（见上文）。

### 配置要点

## 配置差异

### SGLang vs vLLM 配置差异

**SGLang** (`conf/fsdp_lora.yaml`):
```yaml
allocation_mode: sglang:d4p1t1+d4p1t1

rollout:
  train_data_parallel_size: 1  # 单 rank rollout

train_dataset:
  batch_size: 8  # Total parents per step
```

**vLLM** (`conf/fsdp_lora_vllm.yaml`):
```yaml
allocation_mode: fsdp:d8p1t1+d8p1t1

rollout:
  # vLLM handles distributed rollout internally

train_dataset:
  batch_size: 8  # Per DP rank (total = 8 × dp_size)
```

### Group Rollout 配置

```yaml
gconfig:
  n_samples: 64        # 每个 parent 的 rollout 数

sampler:
  batch_size: 8        # 每步采样的 parent 数

train_dataset:
  batch_size: 8        # 与 sampler.batch_size 一致
```

### LoRA 配置

```yaml
actor:
  use_lora: true
  lora_rank: 32
  lora_alpha: 16
  target_modules: [all-linear]
  weight_update_mode: disk  # 或 'xccl'

sglang:
  enable_lora: true
  max_lora_rank: 32
```

### TTT-Discover 配置

```yaml
actor:
  # 熵优势函数 (论文算法)
  adv_estimator: entropic  # 或 'entropic_adaptive_beta'
  adv_estimator_beta: 1.0
  
  # KL 惩罚
  kl_ctl: 0.01
```

## 架构流程

### SGLang 版本（单 Rank Rollout）

```
┌─────────────────────────────────────────────────────────────────┐
│ Step N                                                          │
├─────────────────────────────────────────────────────────────────┤
│ 1. Sample Parents                                               │
│    └── PUCTSampler.sample_states(8) → [state_1, ..., state_8]   │
│                                                                 │
│ 2. Group Rollout (Rank 0 only, async)                           │
│    └── prepare_batch(group_size=64) → [512, seq_len]            │
│        ├── state_1 × 64 rollouts                                │
│        ├── state_2 × 64 rollouts                                │
│        └── ...                                                  │
│                                                                 │
│ 3. Broadcast & Split (All ranks)                                │
│    └── [512] → split → [64] per rank (for DP=8)                 │
│                                                                 │
│ 4. Group Processing (All ranks)                                 │
│    ├── Split [512] → 8 groups of 64                             │
│    ├── Top-2 selection per group                                │
│    ├── PUCTSampler.update_states() (external)                   │
│    └── Best trajectory selection for training                   │
│                                                                 │
│ 5. Training (All ranks, FSDP)                                   │
│    ├── Recompute logprobs                                       │
│    ├── Compute entropic advantages                              │
│    └── PPO update                                               │
│                                                                 │
│ 6. Checkpoint & Evaluate                                        │
│    ├── Update weights                                           │
│    ├── Save LoRA adapter                                        │
│    └── Sampler.flush()                                          │
└─────────────────────────────────────────────────────────────────┘
```

### vLLM 版本（分布式 Rollout）

```
┌─────────────────────────────────────────────────────────────────┐
│ Step N                                                          │
├─────────────────────────────────────────────────────────────────┤
│ 1. Sample Parents (Per DP Rank)                                 │
│    └── PUCTSampler.sample_states(8) per rank                    │
│    Total: 8 × dp_size parents across all ranks                  │
│                                                                 │
│ 2. Group Rollout (Distributed, each rank async)                 │
│    └── actor.prepare_batch(group_size=64)                       │
│        Each rank: [8 × 64, seq_len] = [512, seq_len]            │
│                                                                 │
│ 3. Group Processing (Each rank locally)                         │
│    ├── Split local [512] → 8 groups of 64                       │
│    ├── Top-2 selection per group                                │
│    ├── PUCTSampler.update_states() (local view)                 │
│    └── Best trajectory selection for training                   │
│                                                                 │
│ 4. Sampler Sync (Rank 0 flushes to disk)                        │
│    └── Other ranks reload from file                             │
│                                                                 │
│ 5. Training (All ranks, FSDP)                                   │
│    ├── Recompute logprobs                                       │
│    ├── Compute entropic advantages                              │
│    └── PPO update                                               │
│                                                                 │
│ 6. Checkpoint & Evaluate                                        │
│    ├── Update weights                                           │
│    ├── Save LoRA adapter                                        │
│    └── Sampler.flush() (rank 0 only)                            │
└─────────────────────────────────────────────────────────────────┘
```

## 关键设计

### 1. 外部 Sampler 更新

PUCTSampler 的更新**在 workflow 外部**进行，避免 `GroupedRolloutWorkflow` 多次调用 `arun_episode` 时的重复更新问题。

```python
# workflow.py: 只返回 metadata，不更新 sampler
trajectory["_tttd_metadata"] = {
    "parent_state": state,
    "reward": reward,
    ...
}

# train_fsdp_lora.py: 批量更新 sampler
process_group_results_and_update_sampler(
    batch=batch,           # [512, seq_len]
    sampler=sampler,
    group_size=64,         # 8 groups
    topk_per_parent=2,     # top-2 per group
)
```

### 2. LoRA 权重更新

```python
# Disk mode (推荐，稳定)
weight_update_meta = WeightUpdateMeta.from_disk(
    experiment_name, trial_name, fileroot,
    use_lora=True,
    lora_name="tttd_adapter",
    lora_int_id=1,
)

# XCCL mode (快速，需要 RDMA)
weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(
    allocation_mode,
    use_lora=True,
    lora_name="tttd_adapter",
    lora_int_id=1,
)
```

### 3. Entropic Advantage

TTTDActor 使用熵目标函数计算优势（论文公式）：

```
w_β(a) = exp(β · R(a)) / E[exp(β · R)]
A(a;s) = w_β(a) - 1 - λ · KL(π_θ || π_ref)
```

配置选项：
- `adv_estimator=entropic`: 固定 β
- `adv_estimator=entropic_adaptive_beta`: 自适应 β（基于 KL 约束）

## 性能优化

| 优化 | 说明 |
|------|------|
| **Async Rollout** | SGLang 异步调度，最大化 GPU 利用率 |
| **Group Parallel** | 64 rollouts 并行执行 |
| **FSDP** | 8 GPU 数据并行训练 |
| **LoRA** | 只训练 1-2% 参数，减少内存和通信 |
| **Disk Weight Update** | 稳定的权重同步机制 |

## 调试提示

### 检查 Group Rollout 是否正确

```python
# 添加 debug 打印
print(f"Batch size: {batch['input_ids'].shape[0]}")  # 应为 512
print(f"Num parents: {len(groups)}")                  # 应为 8
print(f"Group size: {groups[0].rewards.shape[0]}")    # 应为 64
```

### 检查 Sampler 更新

```python
# 在 process_group_results_and_update_sampler 后
print(f"Sampler total states: {sampler.T}")
print(f"Sampler top state reward: {max(s.m for s in sampler.states.values())}")
```

### 内存优化

如果 OOM，尝试：

```yaml
actor:
  lora_rank: 16          # 降低 LoRA rank
  gradient_checkpointing: true
  mb_spec:
    max_tokens_per_mb: 5120  # 降低 micro-batch

gconfig:
  max_new_tokens: 512    # 降低生成长度
```

## 文件版本对比

| 文件 | 引擎 | 特点 | 推荐度 |
|------|------|------|--------|
| `train_fsdp_lora.py` | SGLang + FSDP + LoRA | 单 rank rollout，稳定 | ✅ 生产使用 |
| `train_fsdp_lora_vllm.py` | vLLM + FSDP + LoRA | 分布式 rollout，高性能 | ✅ 生产使用 |
| `train_optimized.py` | Pure Inference | 无 FSDP，测试用 | ⚠️ 开发测试 |
| `train_with_group.py` | - | 早期版本，有 bug | ❌ 已弃用 |
| `train_group_external.py` | - | 手动实现，无优化 | ❌ 已弃用 |
