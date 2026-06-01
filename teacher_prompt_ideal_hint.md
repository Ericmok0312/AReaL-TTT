# Teacher Prompt — Ideal Hint Mode

## 设计目标

- **Same initial state** as student
- **Hint placed in value_context** (replaces `No previous code available.`)
- **Distinguishes hint score vs current state score**

---

## Student Prompt (对比)

```text
Act as an expert software developer and inequality specialist specializing in creating step functions with certain properties.

Your task is to generate the sequence of non-negative heights of a step function, that minimizes the following evaluation function:

```python
def evaluate_sequence_ac1(sequence):
    '''Evaluates a sequence for the AC1 inequality.
    
    The AC1 inequality asks: what is the minimum possible value of
        raw_score = 2 * n * max(conv(sequence, sequence)) / (sum(sequence)**2)
    where n = len(sequence).
    
    Lower raw_score is better (the theoretical lower bound is 1.5).
    This function returns the raw_score.
    '''
    import numpy as np

    n = len(sequence)
    b_sequence = np.convolve(sequence, sequence)
    max_b = max(b_sequence)
    sum_a = np.sum(sequence)

    # Protect against the case where the sum is too close to zero
    if sum_a < 0.01:
        return np.inf

    return float(2 * n * max_b / (sum_a**2))
```

[Literature context about AC1 inequality...]

Your task is to write a search function that searches for the best sequence of coefficients. Your function will have 30 seconds to run, and after that it has to have returned the best sequence it found. If after 30 seconds it has not returned anything, it will be terminated with negative infinity points. All numbers in your sequence have to be positive or zero. Larger sequences with 1000s of items often have better attack surface, but too large sequences with 100s of thousands of items may be too slow to search.

You may code up any search method you want, and you are allowed to call the evaluate_sequence() function as many times as you want. You have access to it, you don't need to code up the evaluate_sequence() function.

You are iteratively optimizing upper bound.
No previous code available.                                            ← 这里！
Current upper bound (lower is better): 2.000000
Target: 1.5030. Current gap: 0.497000. Further improvements will also be generously rewarded.
Length of the construction: 1000

You may want to start your search from one of the constructions we have found so far, which you can access through the 'height_sequence_1' global variable. 
However, you are encouraged to explore solutions that use other starting points to prevent getting stuck in a local minimum.

Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparemeters, etc. 
Unless you make a meaningful improvement, you will not be rewarded.

Rules:
- You must define the `propose_candidate` function as this is what will be invoked.
- You can use scientific libraries like scipy, numpy, cvxpy[CBC,CVXOPT,GLOP,GLPK,ECOS,SCS,PDLP,SCIP], math.
- You can use up to 2 CPUs.
- Make all helper functions top level and have no closures from function nesting. Don't use any lambda functions.
- No filesystem or network IO.
- Do not import evaluate_sequence yourself. Assume it will already be imported and can be directly invoked.
- **Print statements**: Use `print()` to log progress, intermediate bounds, timing info, etc. Your output will be shown back to you.
- Include a short docstring at the top summarizing your algorithm.

Make sure to think and return the final program between ```python and ```.
```

---

## Teacher Prompt — Ideal Hint (关键差异)

**唯一改动**：把 `No previous code available.` 替换为 Hint 内容。

```text
Act as an expert software developer and inequality specialist specializing in creating step functions with certain properties.

Your task is to generate the sequence of non-negative heights of a step function, that minimizes the following evaluation function:

```python
def evaluate_sequence_ac1(sequence):
    ... [same as student] ...
```

[Literature context...]

Your task is to write a search function...

You are iteratively optimizing upper bound.
Here is a known good approach for this problem:                      ← 替换！
```python
def propose_candidate():
    import numpy as np
    import time
    from scipy import optimize
    
    n = len(height_sequence_1)
    
    # Gradient-based optimization
    def objective(x):
        return evaluate_sequence(x)
    
    x0 = height_sequence_1.copy()
    bounds = [(0, None) for _ in range(n)]
    result = optimize.minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
    
    return result.x
```
This approach achieves a score of 1.506420.
You should adapt this approach to improve the current initial state. The current initial state has a score of 2.000000.
Current upper bound (lower is better): 2.000000                     ← 保留 current state 信息
Target: 1.5030. Current gap: 0.497000. Further improvements will also be generously rewarded.
Length of the construction: 1000

You may want to start your search from one of the constructions we have found so far...
[rest same as student]
```

---

## 和 Continuation 的区别

| 维度 | Continuation | Ideal Hint |
|------|-------------|------------|
| **State** | Privileged state (有历史 code) | Initial state (无历史) |
| `value_context` 开头 | `Here is the last code we ran:` | `Here is a known good approach:` |
| **Code 来源** | State **自己**的历史 | 外部注入的 SOTA |
| **Current value** | 1.5064 (privileged state 自己的) | 2.0000 (initial state 的) |
| **Gap** | 0.0034 | 0.4970 |
| **Task framing** | "继续优化你自己的代码" | "用这个好代码来优化当前 state" |

---

## 为什么这样更好？

### 1. 位置正确
Hint 出现在 **value_context** 位置，是 prompt 的核心信息，不会被 rules 淹没。

### 2. 无冲突
没有 `No previous code available` 和 `[Hint]` 同时存在的矛盾。

### 3. 区分 Score
- `This approach achieves 1.5064` → hint code 的分数
- `Current initial state has a score of 2.0000` → 当前要优化的起点
- 模型清楚知道：**目标是接近 1.5064，起点是 2.0**

### 4. Same Input Distribution
Student 和 Teacher 面对 **相同的 initial state**，只有 value_context 不同。蒸馏时 KL divergence 有意义。
