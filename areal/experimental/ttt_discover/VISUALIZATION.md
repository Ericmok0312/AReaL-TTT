# TTT-Discover 训练可视化指南

本文档说明如何使用训练历史记录和可视化功能来监控 TTT-Discover 算法的训练过程。

## 功能概述

本模块提供以下功能：

1. **数据记录** (`ttt_logger.py`): 记录每个 step 的 reward 分布数据
2. **可视化** (`ttt_visualizer.py`): 绘制类似论文 Figure 1 的 KDE 分布图
3. **独立绘图脚本** (`generate_plot.py`): 从保存的历史文件生成图表

## 文件结构

```
areal/experimental/ttt_discover/
├── ttt_logger.py          # 训练历史记录器
├── ttt_visualizer.py      # 可视化模块
├── generate_plot.py       # 独立绘图脚本
└── VISUALIZATION.md       # 本文档
```

## 使用方法

### 1. 训练时自动记录

修改后的 `train_fsdp_lora_vllm_v2.py` 已经集成了数据记录功能。训练时会自动：

- 记录指定 steps 的 reward 分布
- 每个 step 后保存 checkpoint（支持中断恢复）
- 训练结束时保存完整历史

**配置记录的 steps**（在配置中添加）：

```yaml
# 在 YAML 配置文件中
save_steps: [0, 9, 24, 49]  # 默认记录这些 steps
```

或在代码中修改默认值：

```python
save_steps = [0, 9, 24, 49]  # 第29行附近
```

### 2. 生成可视化图表

训练完成后，运行以下命令生成图表：

```bash
# 基本用法（自动检测所有可用的 steps）
python areal/experimental/ttt_discover/generate_plot.py \
    --history_path ./outputs/training_history.pkl

# 限制最多显示的 step 数量（均匀采样）
python areal/experimental/ttt_discover/generate_plot.py \
    --history_path ./outputs/training_history.pkl \
    --max_steps 5

# 指定特定 steps（覆盖自动检测）
python areal/experimental/ttt_discover/generate_plot.py \
    --history_path ./outputs/training_history.pkl \
    --steps 0 9 24 49

# 指定 benchmark 值（不同任务）
python areal/experimental/ttt_discover/generate_plot.py \
    --history_path ./outputs/training_history.pkl \
    --benchmark_value 2.635983

# 生成进展图
python areal/experimental/ttt_discover/generate_plot.py \
    --history_path ./outputs/training_history.pkl \
    --show_progression
```

**Step 选择逻辑**：
- 默认：自动检测历史数据中所有可用的 steps 并全部绘制
- `--max_steps N`：如果历史中有超过 N 个 steps，均匀采样 N 个 steps
- `--steps x y z`：显式指定要绘制的 steps（覆盖自动检测）

### 3. 不同任务的配置示例

#### Circle Packing（越大越好）

```bash
python areal/experimental/ttt_discover/generate_plot.py \
    --history_path ./outputs/cp_n26/training_history.pkl \
    --steps 0 9 24 49 \
    --benchmark_value 2.635983 \
    --benchmark_label "Best Human" \
    --xlabel "Sum of Radii (higher is better →)" \
    --higher_is_better True \
    --output_path ./figures/cp_training_dynamics.png
```

#### TriMul Runtime（越小越好）

```bash
python areal/experimental/ttt_discover/generate_plot.py \
    --history_path ./outputs/trimul/training_history.pkl \
    --steps 0 9 24 49 \
    --benchmark_value 100 \
    --benchmark_label "Target Runtime" \
    --xlabel "Runtime μs (lower is better ←)" \
    --higher_is_better False \
    --output_path ./figures/trimul_training_dynamics.png
```

### 4. 编程方式使用

```python
from areal.experimental.ttt_discover.ttt_logger import TTTTrainingLogger
from areal.experimental.ttt_discover.ttt_visualizer import TTTVisualizer

# 1. 记录训练数据
logger = TTTTrainingLogger(
    save_steps=[0, 9, 24, 49],
    output_dir="./outputs",
    is_dp_head=True,
)

for step in range(50):
    # ... 生成 rollouts ...
    rewards = [r for _, r in rollouts]
    
    # 记录当前 step
    logger.record_step(
        step=step,
        rewards=rewards,
        best_solution=best_code,
    )

# 保存历史
logger.save()

# 2. 可视化
visualizer = TTTVisualizer(higher_is_better=True)
visualizer.plot_training_dynamics(
    history_dict=history_data,
    steps=[0, 9, 24, 49],
    benchmark_value=2.635983,
    output_path='training_dynamics.png',
)
```

## 输出文件

训练完成后，输出目录会包含：

```
outputs/
├── training_history.pkl              # 完整训练历史（pickle 格式）
├── training_history.json             # 摘要信息（JSON 格式，便于查看）
├── training_history_checkpoint.pkl   # 用于中断恢复的 checkpoint
└── training_dynamics.png             # 生成的可视化图表（手动运行脚本）
```

### 文件格式说明

**training_history.pkl** 结构：

```python
{
    "history": {
        "step_0": {
            "step": 0,
            "rewards": [r1, r2, ..., r512],  # 所有 rollouts 的 rewards
            "max_reward": 2.5,
            "mean_reward": 2.1,
            "min_reward": 1.8,
            "std_reward": 0.15,
            "num_rollouts": 512,
            "best_solution": "...",  # 最佳解的代码
            "metrics": {...},  # 额外指标
        },
        "step_9": {...},
        "step_24": {...},
        "step_49": {...},
    },
    "best_of_n": {  # 可选
        "rewards": [...],
        "max_reward": ...,
        "mean_reward": ...,
    },
    "metadata": {
        "save_steps": [0, 9, 24, 49],
        "num_snapshots": 4,
        "overall_best_reward": 2.63,
        "overall_best_step": 49,
    }
}
```

## 图表解读

生成的图表类似于论文 Figure 1，展示了训练过程中 reward 分布的变化：

### 健康训练的特征

1. **分布右移**（越大越好的任务）：蓝色曲线整体向右移动
2. **峰值上升**：分布的峰值逐渐增高
3. **方差减小**：曲线变得更尖锐（策略更确定）
4. **超过基准**：后期的分布峰值超过黑色虚线（Best Human）

### 异常情况的识别

| 现象 | 可能原因 | 建议 |
|------|---------|------|
| 分布左移 | 训练发散/奖励信号问题 | 检查奖励函数，降低学习率 |
| 分布扁平 | 策略随机/探索过度 | 调整 temperature，减少熵正则 |
| 无明显变化 | 梯度消失/学习率过低 | 检查梯度流，增大学习率 |
| 多峰分布 | 策略陷入局部最优 | 增加多样性奖励，调整采样 |

### 与 Best-of-N 对比

- **灰色曲线**（Best-of-N）：冻结策略采样大量结果后的分布
- **蓝色曲线**（TTT-Discover）：测试时训练后的分布
- **预期**：TTT-Discover 的后期分布应显著优于 Best-of-N

## 进阶用法

### 自定义颜色方案

```python
visualizer = TTTVisualizer(higher_is_better=True)
visualizer.plot_training_dynamics(
    history_dict=history,
    steps=[0, 9, 24, 49],
    colors=['#ff7f0e', '#2ca02c', '#d62728', '#1f77b4'],  # 自定义颜色
)
```

### 绘制 Reward 进展曲线

```python
# 绘制 mean/max/std 随 step 变化的曲线
visualizer.plot_reward_progression(
    history_dict=history,
    output_path='reward_progression.png',
)
```

### 添加 Best-of-N 基线

```python
# 记录 Best-of-N 数据
logger.record_best_of_n(
    rewards=bon_rewards,
    best_solution=bon_best,
    metadata={'n_samples': 25600, 'model': 'frozen_policy'},
)
```

## 故障排除

### 问题：历史文件很大

**解决方案**：只保存关键 steps 的数据，reward 列表可以用 numpy 的 float32 存储：

```python
rewards = np.array(rewards, dtype=np.float32).tolist()
```

### 问题：分布式训练下数据不一致

**原因**：不同 rank 的数据没有正确聚合

**检查**：确保 `aggregate_distributed=True` 且调用了 `dist.all_gather_object`

### 问题：KDE 计算失败

**原因**：所有 rewards 相同（方差为 0）

**解决**：已自动处理，添加微小噪声

## 参考

- 论文: "Learning to Discover at Test Time"
- Figure 1: 展示 TTT-Discover 在 Circle Packing 任务上的训练动态
