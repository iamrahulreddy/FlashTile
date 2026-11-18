# Qwen3.5-4B (FP8) greedy generations: SDPA vs FlashTile

## short (35 prompt tokens)

Greedy outputs diverge at token 19.

### sdpa (prefill 134 ms, decode 6.7 tok/s)

```text
Attention requires a softmax function to convert raw attention scores into a valid probability distribution, ensuring that the sum of weights across all positions equals one so that the output is a proper weighted average of the input values. The scaling factor $1/\sqrt{d}$ is applied to the dot product before the softmax to prevent the values from growing too large, which would cause the softmax function to become numerically unstable and output values close to zero or one. This scaling keeps the gradients during backpropagation stable, allowing the model to learn effective attention patterns without vanishing or exploding gradients.
```

### flashtile (prefill 122 ms, decode 6.2 tok/s)

```text
Attention requires a softmax function to convert raw attention scores into a valid probability distribution, ensuring that the model focuses on a weighted combination of input tokens while strictly summing to one. The scaling factor of $1/\sqrt{d}$ is applied to the dot product before the softmax to prevent the values from growing too large, which would cause the softmax function to become numerically unstable and output values close to zero or one. This scaling maintains the gradient flow during backpropagation, allowing the model to learn effective attention weights without vanishing gradients.
```

## code (32 prompt tokens)

Greedy outputs diverge at token 52.

### sdpa (prefill 131 ms, decode 8.6 tok/s)

```text
```python
import math
from typing import List

def log_sum_exp(values: List[float]) -> float:
    """
    Compute log(sum(exp(x))) in a numerically stable way.
    
    Formula: log_sum_exp(x) = m + log(sum(exp(x_i - m)))
    where m = max(x).
    
    Args:
        values: List of floats.
        
    Returns:
        The log-sum-exp value.
    """
    if not values:
        return float('-inf')
    
    m = max(values)
    # Compute sum of exp(x_i - m) to avoid overflow
    s = sum(math.exp(x - m) for x in values)
    return m + math.log(s)
```
```

### flashtile (prefill 127 ms, decode 9.3 tok/s)

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
    # Subtract max for numerical stability
    s = sum(math.exp(v - m) for v in values)
    return m + math.log(s)
```
```

## long_document_qa (39459 prompt tokens)

Greedy outputs diverge at token 37.

### sdpa (prefill 1009 ms, decode 6.9 tok/s)

```text
For a query row where all keys are masked, FlashTile returns an output of zero, a gradient of zero, and an LSE of negative infinity. This behavior is defined by the fully masked row branch, which explicitly sets the probability weights to zero and prevents the evaluation of undefined operations like $-\infty - (-\infty)$.
```

### flashtile (prefill 1204 ms, decode 8.7 tok/s)

```text
For a query row where all keys are masked, FlashTile returns an output of zero, a gradient of zero, and an LSE of negative infinity. This behavior is defined by the "fully masked rows" contract, which explicitly prevents the evaluation of undefined operations like $-\infty - (-\infty)$ or division by zero.
```

