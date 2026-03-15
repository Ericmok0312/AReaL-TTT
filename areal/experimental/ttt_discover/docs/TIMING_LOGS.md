# TTT-Discover Async Training Timing Logs

Timing 数据现在保存在 `training_history.pkl` 中，和 reward 数据在一起。

## Execute Tail Latency 定义

**Execute Tail Latency** = 所有 LLM inference 完成后，到所有 code execution 完成的时间差

```
Timeline:

Rollout 1:  [GPU Inference]........[Execute]...
Rollout 2:  [GPU Inference].................[Execute]...
Rollout 3:  [GPU Inference]........[Execute]...
            ...
Rollout 512:[GPU Inference]..............................[Execute]...

            ^ last_gpu_done          ^ last_exec_done
            
Execute Tail Latency = last_exec_done - last_gpu_done
```

这个指标反映了：**当 GPU 全部完成后，还需要等待多久才能开始 training**。

## 存储位置

```python
# training_history.pkl 结构
{
  "step_0": {
    "step": 0,
    "rewards": [...],
    "max_reward": 0.8234,
    "metrics": {
      "batch_parents": 8,
      "group_size": 64,
      "timing": {
        "rollout": 45.23,       # Total rollout time (GPU + Execute overlap)
        "execute_tail": 12.45,   # Time from last GPU done to last Execute done
        "training": 15.67,       # Training phase time
        "total": 60.90           # Total step time
      },
      "timing_stats": {          # Detailed timing (optional)
        "n_rollouts": 512,
        "first_gpu_done": 100.0,
        "last_gpu_done": 145.0,   # All GPU inferences done
        "first_exec_done": 110.0,
        "last_exec_done": 157.45, # All executions done
        "gpu_span": 45.0,         # Time from first to last GPU done
        "exec_span": 47.45,       # Time from first to last exec done
        "tail_latency": 12.45     # Same as execute_tail above
      }
    }
  }
}
```

## 访问 Timing 数据

```python
import pickle

with open('training_history.pkl', 'rb') as f:
    history = pickle.load(f)

# Get timing for step 5
step_5 = history['step_5']
timing = step_5['metrics']['timing']

print(f"Rollout time: {timing['rollout']:.2f}s")
print(f"Execute tail: {timing['execute_tail']:.2f}s")
print(f"Training time: {timing['training']:.2f}s")

# Calculate parallelism efficiency
efficiency = (timing['rollout'] + timing['training']) / timing['total']
print(f"Parallelism efficiency: {efficiency:.2f}x")
```

## Console Log Format

```
[TIMING][Step 5] rollout=45.23s | exec_tail=12.45s | training=15.67s | total=60.90s | reward=0.8234 | n=512
```

## Key Metrics Explained

| Metric | Meaning | Ideal Value |
|--------|---------|-------------|
| `rollout` | Total time for prepare_batch to return | Depends on workload |
| `execute_tail` | Time waiting for code execution after GPU done | 0s (all execute during GPU) |
| `training` | Time for sampler sync + PPO update + save | Depends on model size |
| `total` | Total step time | Minimize this |

### Parallelism Efficiency

```
Efficiency = (rollout + training) / total

- Sync (staleness=0):  ~1.0 (sequential)
- Async (staleness=2): >1.0 (overlap)
- Perfect overlap:      2.0 (training fully hidden behind rollout)
```

### Execute Tail Interpretation

| exec_tail | Interpretation |
|-----------|---------------|
| ~0s | Perfect overlap, all execute during GPU inference |
| Small (<5s) | Good overlap, minor queue at end |
| Medium (5-20s) | Some head-of-line blocking, but manageable |
| Large (>20s) | Severe blocking, need more workers or dynamic BS |

## Research Analysis

### Compare Sync vs Async

```python
import pickle
import numpy as np

def analyze_timings(history_path):
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    
    exec_tails = []
    for key, data in history.items():
        if key.startswith('step_') and 'timing' in data.get('metrics', {}):
            exec_tails.append(data['metrics']['timing']['execute_tail'])
    
    return {
        'mean_tail': np.mean(exec_tails),
        'max_tail': np.max(exec_tails),
        'p99_tail': np.percentile(exec_tails, 99),
    }

sync = analyze_timings('sync/training_history.pkl')
async2 = analyze_timings('async2/training_history.pkl')

print(f"Sync exec tail: {sync['mean_tail']:.2f}s")
print(f"Async exec tail: {async2['mean_tail']:.2f}s")
# Async should have smaller tail due to better overlap
```

### Visualize Overlap

```python
import pickle
import matplotlib.pyplot as plt

with open('training_history.pkl', 'rb') as f:
    history = pickle.load(f)

steps = []
rollouts = []
exec_tails = []

for key in sorted(history.keys()):
    if key.startswith('step_'):
        step_num = int(key.split('_')[1])
        timing = history[key]['metrics']['timing']
        steps.append(step_num)
        rollouts.append(timing['rollout'])
        exec_tails.append(timing['execute_tail'])

plt.figure(figsize=(12, 6))
plt.plot(steps, rollouts, label='Rollout time', linewidth=2)
plt.plot(steps, exec_tails, label='Execute tail', linewidth=2)
plt.xlabel('Step')
plt.ylabel('Time (s)')
plt.title('TTT-Discover: Rollout vs Execute Tail Latency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('execute_tail_analysis.png')
```

## Troubleshooting

### High execute_tail

If `execute_tail` is consistently high:

1. **Increase max_code_workers** (currently 64)
   ```python
   # In workflow_v2.py
   max_code_workers = 128  # or higher
   ```

2. **Enable dynamic batch size**
   ```yaml
   dynamic_bs: true  # Skip slow parents
   ```

3. **Check for timeout outliers**
   - Some code may be hitting timeout (1100s for AC1)
   - These create head-of-line blocking

### Negative efficiency

If `efficiency < 1.0`, something is wrong:
- Check if async is actually enabled (`max_head_offpolicyness > 0`)
- Verify rollout and training are running on different GPUs
