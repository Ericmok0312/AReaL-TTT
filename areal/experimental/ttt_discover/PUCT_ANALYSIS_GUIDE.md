# PUCT 三个核心指标分析指南

## 概述

本系统记录了计算 PUCTSampler 三个核心指标所需的数据：

1. **Q-value 估计误差 (Q-value Estimation Error)**
2. **信息延迟导致的 Selection 切换 (Selection Switching)**
3. **Q-value 收敛延迟 (Q-value Convergence Delay)**

**关键特性**: 支持同一个 state 被多次采样为 parent，每次采样都是一个独立的 episode。

## 数据记录

数据记录在 `training_history.pkl` 中，每个 step 包含以下结构：

```python
{
    "step_0": {
        "step": 0,
        "rewards": [...],
        "puct_analysis": {
            "parent_episodes": {
                "parent_id_1": [
                    # Episode 0: 第一次被采样 (step 0)
                    {
                        "episode_id": 0,
                        "parent_id": "parent_id_1",
                        "sampled_step": 0,
                        "parent_value": 0.5,
                        "expected_children": 4,
                        "children_completed": [
                            {"reward": 0.9, "complete_step": 1, "exec_time_ms": 150.0}
                        ],
                        "puct_updates": [
                            {"update_step": 1, "n_visits": 1, "m_value": 0.9, "score": 0.9, 
                             "children_rewards": [0.9]}
                        ],
                        "completed": True,
                        "completed_step": 2
                    },
                    # Episode 1: 第二次被采样 (step 5) - 同一个 parent！
                    {
                        "episode_id": 1,
                        "parent_id": "parent_id_1", 
                        "sampled_step": 5,
                        "parent_value": 0.9,  # 可能更新后的值
                        "expected_children": 4,
                        "children_completed": [...],
                        "puct_updates": [...],
                        "completed": False
                    }
                ]
            },
            "puct_updates": [...],  # 全局更新日志
            "selection_events": [...]  # 选择事件（待实现）
        }
    }
}
```

## 关键概念：区分多次采样

### 问题场景

```
Step 0: Parent A 被采样 → 生成 children A1, A2, A3, A4 → 完成
Step 5: Parent A 再次被采样 → 生成 children A5, A6, A7, A8 → 完成

问题：如何区分第一批 children (A1-A4) 和第二批 children (A5-A8)？
```

### 解决方案

使用 `episode_id` 区分：

```python
parent_episodes["parent_A"] = [
    {"episode_id": 0, "sampled_step": 0, "children_completed": [A1, A2, A3, A4], ...},
    {"episode_id": 1, "sampled_step": 5, "children_completed": [A5, A6, A7, A8], ...}
]
```

### 活跃 Episode 机制

- **活跃 Episode**: 最近一次采样且尚未完成（`completed=False`）
- **已完成 Episode**: 所有 children 已完成（`completed=True`）
- 新 children 自动关联到**活跃 Episode**

```python
# 获取活跃 episode（内部方法）
def _get_active_episode(parent_id):
    episodes = self._parent_episodes[parent_id]
    for episode in reversed(episodes):  # 从最近开始找
        if not episode.get('completed', False):
            return episode
    return None  # 没有活跃 episode（该 parent 当前未被采样）
```

## 指标计算方法

### 指标 1: Q-value 估计误差

```python
def calculate_q_value_errors_for_episode(episode):
    """
    计算单个 episode 的 Q-value 估计误差。
    
    对于每个 PUCT update，比较：
    - actual_m: PUCT 使用的 _m 值
    - true_m: 如果知道所有已完成的 children 的最大 reward
    """
    errors = []
    
    for update in episode['puct_updates']:
        if 'm_value' not in update:
            continue
            
        actual_m = update['m_value']
        update_step = update['update_step']
        
        # 关键：只使用这个 episode 的 children！
        completed_rewards = [
            c['reward'] for c in episode['children_completed']
            if c['complete_step'] <= update_step
        ]
        
        if completed_rewards:
            true_m = max(completed_rewards)
            error = abs(actual_m - true_m)
            
            errors.append({
                'episode_id': episode['episode_id'],
                'step': update_step,
                'actual_m': actual_m,
                'true_m': true_m,
                'error': error,
                'n_children_known': len(completed_rewards),
                'n_children_expected': episode['expected_children'],
            })
    
    return errors


def analyze_all_q_value_errors(history):
    """分析所有 episodes 的 Q-value 误差。"""
    all_errors = []
    
    for step_key, snapshot in history.items():
        puct_data = snapshot.get('puct_analysis', {})
        
        # 遍历所有 parents
        for parent_id, episodes in puct_data.get('parent_episodes', {}).items():
            # 遍历该 parent 的所有 episodes（多次采样）
            for episode in episodes:
                errors = calculate_q_value_errors_for_episode(episode)
                all_errors.extend(errors)
    
    return all_errors


# 使用示例：对比不同采样次数的误差
for parent_id, episodes in history['step_10']['puct_analysis']['parent_episodes'].items():
    print(f"\nParent {parent_id[:8]}: 被采样 {len(episodes)} 次")
    for ep in episodes:
        errors = calculate_q_value_errors_for_episode(ep)
        avg_error = sum(e['error'] for e in errors) / len(errors) if errors else 0
        print(f"  Episode {ep['episode_id']} (step {ep['sampled_step']}): "
              f"avg_error={avg_error:.4f}, children={len(ep['children_completed'])}")
```

### 指标 2: Selection Switching（处理多次采样）

```python
def analyze_selection_patterns_with_resampling(history):
    """
    分析多次采样模式。
    
    关键洞察：
    - 如果 parent 被多次采样，说明它在第一次采样后又被 PUCT 选中
    - 对比不同 episodes 的 Q-value 变化
    """
    
    # 收集所有 parents 的多次采样历史
    parent_resampling = {}
    
    for step_key, snapshot in history.items():
        step = snapshot['step']
        puct_data = snapshot.get('puct_analysis', {})
        
        for parent_id, episodes in puct_data.get('parent_episodes', {}).items():
            if parent_id not in parent_resampling:
                parent_resampling[parent_id] = []
            
            # 记录在这个 step 新开始的 episodes
            for ep in episodes:
                if ep['sampled_step'] == step:  # 本 step 新采样的
                    # 获取该 episode 完成后的最终 Q-value
                    final_m = None
                    if ep.get('puct_updates'):
                        final_m = ep['puct_updates'][-1]['m_value']
                    
                    parent_resampling[parent_id].append({
                        'episode_id': ep['episode_id'],
                        'sampled_step': step,
                        'initial_value': ep['parent_value'],
                        'final_m': final_m,
                        'n_children': len(ep['children_completed']),
                    })
    
    # 分析被多次采样的 parents
    multi_sampled = {pid: hist for pid, hist in parent_resampling.items() 
                     if len(hist) > 1}
    
    print(f"共有 {len(multi_sampled)} 个 parent 被多次采样")
    
    for pid, episodes in multi_sampled.items():
        print(f"\nParent {pid[:8]}:")
        for i, ep in enumerate(episodes):
            print(f"  Episode {ep['episode_id']}: step={ep['sampled_step']}, "
                  f"initial={ep['initial_value']:.3f}, final_m={ep['final_m']:.3f}")
            
            if i > 0:
                prev = episodes[i-1]
                improvement = ep['initial_value'] - prev['initial_value']
                print(f"    → 两次采样间隔: {ep['sampled_step'] - prev['sampled_step']} steps")
                print(f"    → initial_value 提升: {improvement:.3f}")
                
                # 关键判断：如果 initial_value 显著提升，
                # 说明上次采样的 children 成功 back up 到了 parent
                if improvement > 0.1:
                    print(f"    → [INSIGHT] Children 成功提升了 parent 的 value！")
    
    return multi_sampled
```

### 指标 3: Q-value 收敛延迟（按 episode 计算）

```python
def calculate_convergence_delay_for_episode(episode):
    """
    计算单个 episode 的 Q-value 收敛延迟。
    """
    if not episode['children_completed']:
        return None
    
    # 第一个 child 完成的时间
    first_child_step = min(
        c['complete_step'] for c in episode['children_completed']
    )
    
    # 第一次真正的 PUCT update（包含 _m 值）
    first_real_update = None
    for update in episode.get('puct_updates', []):
        if 'm_value' in update:
            first_real_update = update['update_step']
            break
    
    if first_real_update is None:
        return None
    
    delay = first_real_update - first_child_step
    
    return {
        'episode_id': episode['episode_id'],
        'parent_id': episode['parent_id'],
        'sampled_step': episode['sampled_step'],
        'first_child_step': first_child_step,
        'first_update_step': first_real_update,
        'delay': delay,
        'n_children_at_update': sum(
            1 for c in episode['children_completed']
            if c['complete_step'] <= first_real_update
        ),
    }


def compare_delay_across_episodes(history):
    """
    对比同一 parent 在不同 episodes 中的延迟。
    """
    for parent_id, episodes in history['step_X']['puct_analysis']['parent_episodes'].items():
        if len(episodes) > 1:
            print(f"\nParent {parent_id[:8]} 的多次采样延迟对比:")
            
            for ep in episodes:
                delay_info = calculate_convergence_delay_for_episode(ep)
                if delay_info:
                    print(f"  Episode {ep['episode_id']}: delay={delay_info['delay']} steps, "
                          f"first_child@{delay_info['first_child_step']}, "
                          f"first_update@{delay_info['first_update_step']}")
```

## 完整分析脚本

```python
import pickle
import numpy as np
from collections import defaultdict

def full_puct_analysis_v2(history_file):
    """完整分析入口（支持多次采样）。"""
    
    with open(history_file, 'rb') as f:
        history = pickle.load(f)
    
    # ========== 指标 1: Q-value 估计误差 ==========
    print("\n" + "="*60)
    print("指标 1: Q-value 估计误差")
    print("="*60)
    
    all_errors = []
    error_by_episode_count = defaultdict(list)  # 按采样次数分组
    
    for step_key, snapshot in history.items():
        puct_data = snapshot.get('puct_analysis', {})
        
        for parent_id, episodes in puct_data.get('parent_episodes', {}).items():
            n_episodes = len(episodes)
            
            for episode in episodes:
                errors = calculate_q_value_errors_for_episode(episode)
                for e in errors:
                    all_errors.append(e['error'])
                    error_by_episode_count[n_episodes].append(e['error'])
    
    if all_errors:
        print(f"  总体平均误差: {np.mean(all_errors):.4f}")
        print(f"  总体 P90 误差: {np.percentile(all_errors, 90):.4f}")
        
        print("\n  按采样次数分组:")
        for n_eps in sorted(error_by_episode_count.keys()):
            errs = error_by_episode_count[n_eps]
            print(f"    采样 {n_eps} 次的 parents: mean={np.mean(errs):.4f}, n={len(errs)}")
    
    # ========== 指标 2: 多次采样分析 ==========
    print("\n" + "="*60)
    print("指标 2: 多次采样与 Selection 模式")
    print("="*60)
    
    multi_sampled = analyze_selection_patterns_with_resampling(history)
    
    if multi_sampled:
        print(f"\n  共有 {len(multi_sampled)} 个 parent 被多次采样")
        
        # 统计重新采样的时间间隔
        resample_gaps = []
        for pid, episodes in multi_sampled.items():
            for i in range(1, len(episodes)):
                gap = episodes[i]['sampled_step'] - episodes[i-1]['sampled_step']
                resample_gaps.append(gap)
        
        if resample_gaps:
            print(f"  重新采样间隔: mean={np.mean(resample_gaps):.1f}, "
                  f"median={np.median(resample_gaps):.1f} steps")
    
    # ========== 指标 3: 收敛延迟（按 episode） ==========
    print("\n" + "="*60)
    print("指标 3: Q-value 收敛延迟")
    print("="*60)
    
    all_delays = []
    delay_by_episode_id = defaultdict(list)
    
    for step_key, snapshot in history.items():
        puct_data = snapshot.get('puct_analysis', {})
        
        for parent_id, episodes in puct_data.get('parent_episodes', {}).items():
            for episode in episodes:
                delay_info = calculate_convergence_delay_for_episode(episode)
                if delay_info:
                    all_delays.append(delay_info['delay'])
                    delay_by_episode_id[episode['episode_id']].append(delay_info['delay'])
    
    if all_delays:
        print(f"  总体平均延迟: {np.mean(all_delays):.2f} steps")
        print(f"  总体 P90 延迟: {np.percentile(all_delays, 90):.2f} steps")
        
        print("\n  按 episode 序号分组:")
        for ep_id in sorted(delay_by_episode_id.keys()):
            delays = delay_by_episode_id[ep_id]
            print(f"    Episode {ep_id}: mean={np.mean(delays):.2f}, n={len(delays)}")
        
        # 检查：后续 episodes 是否有更低延迟？
        if 0 in delay_by_episode_id and 1 in delay_by_episode_id:
            first_avg = np.mean(delay_by_episode_id[0])
            second_avg = np.mean(delay_by_episode_id[1])
            print(f"\n  对比: 第一次采样 avg_delay={first_avg:.2f}, "
                  f"第二次采样 avg_delay={second_avg:.2f}")
    
    return {
        'q_value_errors': all_errors,
        'multi_sampled_parents': len(multi_sampled),
        'convergence_delays': all_delays,
    }


# 使用
if __name__ == '__main__':
    results = full_puct_analysis_v2('training_history.pkl')
```

## 常见问题

### Q: 如何识别"这是新的采样"还是"之前采样的 children 正在完成"？

**A**: 通过 `episode_id` 和 `sampled_step`：

```python
# 新采样的特征：sampled_step == 当前 step
for ep in episodes:
    if ep['sampled_step'] == current_step:
        print(f"这是新的采样 (episode {ep['episode_id']})")
    else:
        print(f"这是之前采样的延续 (episode {ep['episode_id']}, "
              f"sampled at step {ep['sampled_step']})")
```

### Q: 如何追踪特定 parent 的完整生命周期？

**A**: 

```python
parent_id = "some_parent_id"

# 获取该 parent 的所有 episodes
episodes = history['step_X']['puct_analysis']['parent_episodes'][parent_id]

print(f"Parent {parent_id} 的生命周期:")
for ep in episodes:
    print(f"\nEpisode {ep['episode_id']}:")
    print(f"  采样时间: step {ep['sampled_step']}")
    print(f"  采样时 value: {ep['parent_value']}")
    print(f"  Children: {len(ep['children_completed'])}")
    print(f"  PUCT updates: {len(ep['puct_updates'])}")
    if ep.get('completed'):
        print(f"  完成时间: step {ep['completed_step']}")
        final_m = ep['puct_updates'][-1]['m_value'] if ep['puct_updates'] else None
        print(f"  最终 Q-value: {final_m}")
```

### Q: 如果 parent 被采样时还有未完成的 episode 怎么办？

**A**: 当前实现会创建新的 episode，旧的 episode 会被标记为 `completed=False` 但不再接收新 children。这实际上是一种"episode 泄漏"，但在分析时可以通过 `sampled_step` 来区分。

如果需要严格处理这种情况，可以在 `start_parent_episode` 中添加检查：

```python
def start_parent_episode(self, parent_id, ...):
    # 检查是否有未完成的 episode
    active = self._get_active_episode(parent_id)
    if active:
        # 选择 1: 标记为完成
        active['completed'] = True
        # 选择 2: 报错或警告
        logger.warning(f"Parent {parent_id} has active episode {active['episode_id']} "
                      f"but being resampled at step {sampled_step}")
    
    # 创建新 episode...
```
