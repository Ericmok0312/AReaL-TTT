# PUCT Update Mode 配置指南

## 概述

`puct_update_mode` 配置允许你严格控制 Cross-Batch Contamination，从而可以独立研究两个变数：

1. **Policy Training Staleness**: 使用老模型生成的 rollout 训练新模型的影响
2. **PUCT Update Logic Change**: Async 模式下 PUCTSampler 更新逻辑改变的影响

## 两种模式

### Mode 1: `eager` (默认)

**行为**: 所有完成的 children 立即用于更新 PUCTSampler（当前 async 行为）

**特点**:
- Cross-Batch Contamination 存在
- PUCTSampler 会混合不同 batch 的 children 进行更新
- 更接近"纯 async"行为

**适用场景**: 你想要测试完整的 async 系统行为

```python
workflow = TTTDiscoverWorkflowV2(
    ...,
    puct_update_mode="eager",  # 默认
)
```

### Mode 2: `strict`

**行为**: 只使用当前 step 采样的 parent 的 children 更新 PUCTSampler

**特点**:
- **阻止 Cross-Batch Contamination**
- 延迟的 children 会被暂存，等待 parent 再次被采样时才更新 PUCT
- PUCT 更新逻辑与 Sync 模式一致（同一 batch 的 children 一起更新）
- 但 Policy Training 仍然有 staleness（这是你想要测试的）

**适用场景**: 你想**隔离** staleness 对 policy training 的影响，而不改变 PUCT 更新逻辑

```python
workflow = TTTDiscoverWorkflowV2(
    ...,
    puct_update_mode="strict",
)
```

## 技术细节

### Cross-Batch Contamination 场景

```
Step 5:
  - 采样 Parent A
  - Children: A1(快), A2(快), A3(慢), A4(慢)
  - Step 5 结束时: A1, A2 完成
  - get_pending_updates 返回 [A1, A2]
  
  eager mode PUCT update:
    - _m[A] = max(A1, A2)
    - PUCT 现在基于不完整信息
    
  strict mode PUCT update:
    - _m[A] = max(A1, A2)  # 同样更新（因为这是当前 batch）
    - 没有 contamination（因为 A1,A2 属于 Step 5）

Step 6:
  - 采样 Parent B
  - A3, A4 完成（来自 Step 5 的 parent！）
  - B1, B2 完成
  - get_pending_updates 返回 [A3, A4, B1, B2]
  
  eager mode PUCT update:
    - _m[A] = max(之前的_m, A3, A4)  # Cross-batch！Step 6 更新 Step 5 的 parent
    - _m[B] = max(B1, B2)
    - **这就是 Cross-Batch Contamination**
    
  strict mode PUCT update:
    - _m[A] 不更新（A 不是 Step 6 采样的）
    - _m[B] = max(B1, B2)
    - A3, A4 被暂存到 _delayed_puct_children
    - **Cross-Batch Contamination 被阻止**
```

### Strict Mode 的内部机制

```python
# 在 get_pending_updates 中
if puct_update_mode == "strict":
    for child, parent in zip(all_children, all_parents):
        parent_sampled_step = get_parent_sampled_step(parent)
        
        if parent_sampled_step == current_step:
            # 当前 batch 的 parent - 正常更新 PUCT
            filtered_children.append(child)
            filtered_parents.append(parent)
        else:
            # 之前 batch 的 parent - 延迟更新 PUCT
            delayed_children.append(child)
            delayed_parents.append(parent)
    
    # 返回过滤后的列表给 sync_sampler 使用
    # 但所有 children 仍然用于 training（保持 sample efficiency）
```

## 两次试验设计

### 试验 A: 仅测试 Policy Training Staleness

**目标**: 隔离 staleness 对 policy training 的影响

**配置**:
```python
workflow = TTTDiscoverWorkflowV2(
    ...,
    puct_update_mode="strict",  # 阻止 PUCT 的 cross-batch update
)
```

**预期**:
- PUCTSampler 更新逻辑与 Sync 一致
- 只有 Policy Training 受到 staleness 影响
- 可以准确测量 staleness → policy quality 的因果关系

### 试验 B: 测试完整的 Async 影响（包括 PUCT）

**目标**: 测试完整的 async 系统，包括 PUCT 更新逻辑改变

**配置**:
```python
workflow = TTTDiscoverWorkflowV2(
    ...,
    puct_update_mode="eager",  # 允许 cross-batch PUCT update
)
```

**预期**:
- 存在 Policy Training staleness
- 同时存在 PUCT update logic change
- 测量整体 async 系统的性能

### 对比分析

```python
# 试验 A vs 试验 B 的差异
diff = {
    "policy_staleness": "Both have",  # 两者都有
    "puct_update_logic": {
        "A": "Sync-like (strict)",     # 试验 A: 像 sync
        "B": "Async (eager)",          # 试验 B: async
    },
    "cross_batch_contamination": {
        "A": "Prevented",              # 试验 A: 阻止
        "B": "Present",                # 试验 B: 存在
    }
}

# 如果试验 A 性能接近 Sync，但试验 B 性能下降
# → 说明 PUCT update logic change 是主要影响因素

# 如果试验 A 性能就显著低于 Sync
# → 说明 Policy Training staleness 是主要影响因素
```

## 日志识别

### Strict Mode 日志

```
[Step 6] STRICT_PUCT: Prevented cross-batch contamination. This batch: 4, Delayed: 2
```

表示：
- Step 6 有 4 个 children 属于当前 batch 的 parents
- 有 2 个 children 来自之前 batch 的 parents，被延迟更新 PUCT

### 分析 Delayed Children

```python
# cross_batch_info 结构
cross_batch_info = {
    'mode': 'strict',
    'current_step': 6,
    'n_this_batch': 4,       # 当前 batch 的 children 数
    'n_delayed': 2,          # 被延迟的 children 数
    'delayed_children_ids': ['parent_A_id', ...],
}
```

## 注意事项

### 1. Strict Mode 不损害 Sample Efficiency

- 所有 children（包括 delayed）仍然用于 Policy Training
- 只是 PUCT Update 被延迟
- 保持 Async 的 throughput 优势

### 2. Delayed Children 的处理

当前实现将 delayed children 暂存在 `_delayed_puct_children` 中：

```python
# 暂存 delayed children
self._delayed_puct_children.extend(delayed_children)
self._delayed_puct_parents.extend(delayed_parents)
```

**未来扩展**: 可以在 parent 再次被采样时，使用这些 delayed children 更新 PUCT（类似于"catch-up" update）。

### 3. 与 strict_sync_mode 的区别

| 配置 | 作用 | 影响 |
|------|------|------|
| `strict_sync_mode=True` | 严格限制 rollout 数量 | 影响 throughput |
| `puct_update_mode="strict"` | 控制 PUCT update 时机 | 不影响 throughput |

两者可以独立使用：

```python
# 组合 1: 纯 Sync 行为
strict_sync_mode=True, puct_update_mode="strict"  # 或 "eager"（无区别，因为没有 cross-batch）

# 组合 2: Async training + Sync-like PUCT
strict_sync_mode=False, puct_update_mode="strict"

# 组合 3: 纯 Async 行为
strict_sync_mode=False, puct_update_mode="eager"
```

## 推荐实验设计

### 基准实验

1. **Pure Sync** (baseline)
   - `strict_sync_mode=True`
   - 用于获取最佳性能 baseline

2. **Async Training Only** (试验 A)
   - `strict_sync_mode=False, puct_update_mode="strict"`
   - 隔离 staleness 对 policy training 的影响

3. **Full Async** (试验 B)
   - `strict_sync_mode=False, puct_update_mode="eager"`
   - 测量完整 async 系统的性能

### 对比分析

```python
results = {
    "pure_sync": 0.85,           # Baseline
    "async_training_only": 0.82,  # 试验 A: 如果接近 baseline，说明 staleness 影响小
    "full_async": 0.75,           # 试验 B: 如果显著低于 A，说明 PUCT logic 影响大
}

# 解读:
# - A 与 baseline 的差距 = staleness 影响
# - B 与 A 的差距 = PUCT logic change 影响
```
