# Qwen3.5-4B (FP8) greedy generations: SDPA vs FlashTile

## short (35 prompt tokens)

Greedy outputs diverge at token 27.

### sdpa (prefill 116 ms, decode 11.5 tok/s)

```text
Attention requires a softmax function to convert raw attention scores into a valid probability distribution, ensuring that the sum of weights across all positions equals one while allowing the model to focus on the most relevant information. The scaling factor $1/\sqrt{d}$ is applied to the dot product of query and key vectors to prevent the values from growing too large, which would cause the softmax function to become numerically unstable and flatten the output probabilities. This stabilization is crucial because without it, the exponential terms in the softmax calculation can overflow or underflow, leading to a loss of gradient information and degraded model performance.
```

### flashtile (prefill 110 ms, decode 11.5 tok/s)

```text
Attention requires a softmax function to convert raw attention scores into a valid probability distribution, ensuring that the sum of weights across all positions equals one and that no single token dominates the context representation. The scaling factor $1/\sqrt{d}$ is applied to the dot product before the softmax to prevent the values from growing too large, which would cause the softmax function to become numerically unstable and output values close to zero or one. This scaling maintains the gradient flow during backpropagation, allowing the model to learn effective attention weights even as the dimensionality of the input vectors increases.
```

## code (32 prompt tokens)

Greedy outputs identical.

### sdpa (prefill 109 ms, decode 11.0 tok/s)

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
    
    # Find the maximum value to shift the exponentials
    m = max(values)
    
    # Compute sum of exp(x - m)
    s = sum(math.exp(x - m) for x in values)
    
    return m + math.log(s)
```
```

### flashtile (prefill 106 ms, decode 12.0 tok/s)

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
    
    # Find the maximum value to shift the exponentials
    m = max(values)
    
    # Compute sum of exp(x - m)
    s = sum(math.exp(x - m) for x in values)
    
    return m + math.log(s)
```
```

## long_document_qa (39459 prompt tokens)

Greedy outputs diverge at token 50.

### sdpa (prefill 1367 ms, decode 11.8 tok/s)

```text
For a query row where all keys are masked, FlashTile returns an output of zero, a gradient of zero, and an LSE of negative infinity. This behavior is defined by the "fully masked rows" contract, which explicitly sets the probability weights to zero and prevents the evaluation of undefined operations like $-\infty - (-\infty)$.
```

### flashtile (prefill 1459 ms, decode 11.6 tok/s)

```text
For a query row where all keys are masked, FlashTile returns an output of zero, a gradient of zero, and an LSE of negative infinity. This behavior is defined by the "fully masked rows" contract, which explicitly sets the probability weights and weighted value sum to zero while assigning the log-sum-exp statistic the $-\infty$ sentinel to prevent undefined arithmetic.
```

