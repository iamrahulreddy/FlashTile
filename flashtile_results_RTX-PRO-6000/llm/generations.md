# Qwen3.5-4B (FP8) greedy generations: SDPA vs FlashTile

## short (35 prompt tokens)

Greedy outputs diverge at token 19.

### sdpa (prefill 58 ms, decode 19.7 tok/s)

```text
Attention requires a softmax function to convert raw attention scores into a valid probability distribution, ensuring that the weights assigned to different query-key pairs sum to exactly one. This normalization is crucial because it allows the model to focus on the most relevant information while still considering the entire context, preventing any single token from dominating the output arbitrarily. The scaling factor $1/\sqrt{d}$ is applied to the dot product before the softmax to stabilize training by preventing the values from growing too large, which would otherwise cause the softmax function to become numerically unstable and produce gradients that are too small to be useful.
```

### flashtile (prefill 59 ms, decode 16.4 tok/s)

```text
Attention requires a softmax function to convert raw attention scores into a valid probability distribution, ensuring that the sum of weights across all positions equals one while allowing the model to focus on the most relevant information. The scaling factor of $1/\sqrt{d}$ is applied to the dot product inputs to prevent them from growing too large, which would cause the softmax function to become numerically unstable and output values that are too close to zero or one. This scaling keeps the gradients during backpropagation in a reasonable range, thereby maintaining stable and efficient training dynamics.
```

## code (32 prompt tokens)

Greedy outputs diverge at token 5.

### sdpa (prefill 59 ms, decode 20.5 tok/s)

```text
```python
import math
from typing import List

def log_sum_exp(values: List[float]) -> float:
    """
    Compute log(sum(exp(x))) in a numerically stable way.
    
    Formula: log_sum_exp(x) = max(x) + log(sum(exp(x - max(x))))
    
    Args:
        values: List of floats.
        
    Returns:
        The log-sum-exp value.
    """
    if not values:
        return float('-inf')
    
    m = max(values)
    # Subtract max for stability, then sum exp, then add max back
    s = sum(math.exp(v - m) for v in values)
    return m + math.log(s)
```
```

### flashtile (prefill 59 ms, decode 20.3 tok/s)

```text
```python
import math

def log_sum_exp(values):
    """
    Compute log(sum(exp(x))) in a numerically stable way.
    
    Args:
        values (list[float]): List of floating-point numbers.
    
    Returns:
        float: log(sum(exp(x))).
    """
    if not values:
        return float('-inf')
    
    # Find the maximum value to prevent underflow/overflow
    m = max(values)
    
    # Compute sum of exp(x - m)
    s = sum(math.exp(x - m) for x in values)
    
    return m + math.log(s)
```
```

## long_document_qa (39459 prompt tokens)

Greedy outputs diverge at token 39.

### sdpa (prefill 2254 ms, decode 19.7 tok/s)

```text
For a query row where all keys are masked, FlashTile returns an output of zero, a gradient of zero, and an LSE of negative infinity. This behavior is defined by the fully masked row branch, which explicitly sets the probability weights to zero and prevents any division by zero or invalid arithmetic involving $-\infty$.
```

### flashtile (prefill 2412 ms, decode 11.7 tok/s)

```text
For a query row where all keys are masked, FlashTile returns an output of zero, a gradient of zero, and an LSE of negative infinity. This behavior is defined by the fully masked branch, which explicitly sets the probability weights to zero and prevents the evaluation of undefined operations like $-\infty - (-\infty)$.
```

