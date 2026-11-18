# Qwen/Qwen3.8-27B greedy answers on NVIDIA H200: SDPA vs FA2 vs cuDNN vs FlashTile

bf16, batch 1, thinking off, up to 256 new tokens. Only the 16 softmax attention layers change between paths.

## short (30 prompt tokens)

Prompt:

> Explain in three sentences what grouped-query attention is and why it saves memory when decoding.

### SDPA (default pick) (prefill 114 ms, decode 11.7 tok/s)

```text
Grouped-Query Attention (GQA) is a hybrid attention mechanism that partitions the query heads into groups, where each group shares a single set of key and value heads rather than having unique ones for every query. This design significantly reduces the number of key-value pairs that must be stored in the KV cache during the autoregressive decoding process. By sharing these representations, GQA drastically lowers memory bandwidth requirements and allows for larger batch sizes or longer context windows without sacrificing much of the model's performance.
```

### FA2 (SDPA flash backend) (prefill 135 ms, decode 11.8 tok/s, first 67 tokens identical to SDPA)

```text
Grouped-Query Attention (GQA) is a hybrid attention mechanism that partitions the query heads into groups, where each group shares a single set of key and value heads rather than having unique ones for every query. This design significantly reduces the number of key-value pairs that must be stored in the KV cache during the autoregressive decoding process, as the shared heads are computed and cached only once per group. Consequently, this reduction in cache size lowers memory bandwidth requirements and allows for larger batch sizes or longer context windows without exceeding hardware memory limits.
```

### cuDNN (SDPA cuDNN backend) (prefill 135 ms, decode 12.0 tok/s, first 100 tokens identical to SDPA)

```text
Grouped-Query Attention (GQA) is a hybrid attention mechanism that partitions the query heads into groups, where each group shares a single set of key and value heads rather than having unique ones for every query. This design significantly reduces the number of key-value pairs that must be stored in the KV cache during the autoregressive decoding process. By sharing these representations, GQA drastically lowers memory bandwidth requirements and allows for larger batch sizes or longer context windows without sacrificing much of the model's performance.
```

### FlashTile (prefill 115 ms, decode 11.4 tok/s, first 67 tokens identical to SDPA)

```text
Grouped-Query Attention (GQA) is a hybrid attention mechanism that partitions the query heads into groups, where each group shares a single set of key and value heads rather than having unique ones for every query. This design significantly reduces the number of key-value pairs that must be stored in the KV cache during the autoregressive decoding process, as the shared heads are computed and cached only once per group. Consequently, this reduction in cache size lowers memory bandwidth requirements and allows for larger batch sizes or longer context windows without exceeding hardware memory limits.
```

## math (62 prompt tokens)

Prompt:

> A tank holds 300 litres and starts with 30 litres. Water flows in at 12 litres per minute and drains at 4.5 litres per minute. How many minutes until it is full? Show the steps briefly.

### SDPA (default pick) (prefill 139 ms, decode 11.8 tok/s)

```text
Here are the steps to solve the problem:

**1. Determine the net rate of water flow**
Calculate how much the water level rises per minute by subtracting the outflow rate from the inflow rate.
$$ \text{Net Rate} = \text{Inflow} - \text{Outflow} $$
$$ \text{Net Rate} = 12 \text{ L/min} - 4.5 \text{ L/min} = 7.5 \text{ L/min} $$

**2. Determine the volume needed to fill the tank**
Calculate the difference between the tank's total capacity and the starting volume.
$$ \text{Volume Needed} = \text{Total Capacity} - \text{Starting Volume} $$
$$ \text{Volume Needed} = 300 \text{ L} - 30 \text{ L} = 270 \text{ L} $$

**3. Calculate the time required**
Divide the volume needed by the net rate to find the time in minutes.
$$ \text{Time} = \frac{\text{Volume Needed}}{\text{Net Rate}} $$
$$ \text{Time} = \frac
```

### FA2 (SDPA flash backend) (prefill 118 ms, decode 11.6 tok/s, first 256 tokens identical to SDPA)

```text
Here are the steps to solve the problem:

**1. Determine the net rate of water flow**
Calculate how much the water level rises per minute by subtracting the outflow rate from the inflow rate.
$$ \text{Net Rate} = \text{Inflow} - \text{Outflow} $$
$$ \text{Net Rate} = 12 \text{ L/min} - 4.5 \text{ L/min} = 7.5 \text{ L/min} $$

**2. Determine the volume needed to fill the tank**
Calculate the difference between the tank's total capacity and the starting volume.
$$ \text{Volume Needed} = \text{Total Capacity} - \text{Starting Volume} $$
$$ \text{Volume Needed} = 300 \text{ L} - 30 \text{ L} = 270 \text{ L} $$

**3. Calculate the time required**
Divide the volume needed by the net rate to find the time in minutes.
$$ \text{Time} = \frac{\text{Volume Needed}}{\text{Net Rate}} $$
$$ \text{Time} = \frac
```

### cuDNN (SDPA cuDNN backend) (prefill 128 ms, decode 11.8 tok/s, first 256 tokens identical to SDPA)

```text
Here are the steps to solve the problem:

**1. Determine the net rate of water flow**
Calculate how much the water level rises per minute by subtracting the outflow rate from the inflow rate.
$$ \text{Net Rate} = \text{Inflow} - \text{Outflow} $$
$$ \text{Net Rate} = 12 \text{ L/min} - 4.5 \text{ L/min} = 7.5 \text{ L/min} $$

**2. Determine the volume needed to fill the tank**
Calculate the difference between the tank's total capacity and the starting volume.
$$ \text{Volume Needed} = \text{Total Capacity} - \text{Starting Volume} $$
$$ \text{Volume Needed} = 300 \text{ L} - 30 \text{ L} = 270 \text{ L} $$

**3. Calculate the time required**
Divide the volume needed by the net rate to find the time in minutes.
$$ \text{Time} = \frac{\text{Volume Needed}}{\text{Net Rate}} $$
$$ \text{Time} = \frac
```

### FlashTile (prefill 118 ms, decode 11.5 tok/s, first 256 tokens identical to SDPA)

```text
Here are the steps to solve the problem:

**1. Determine the net rate of water flow**
Calculate how much the water level rises per minute by subtracting the outflow rate from the inflow rate.
$$ \text{Net Rate} = \text{Inflow} - \text{Outflow} $$
$$ \text{Net Rate} = 12 \text{ L/min} - 4.5 \text{ L/min} = 7.5 \text{ L/min} $$

**2. Determine the volume needed to fill the tank**
Calculate the difference between the tank's total capacity and the starting volume.
$$ \text{Volume Needed} = \text{Total Capacity} - \text{Starting Volume} $$
$$ \text{Volume Needed} = 300 \text{ L} - 30 \text{ L} = 270 \text{ L} $$

**3. Calculate the time required**
Divide the volume needed by the net rate to find the time in minutes.
$$ \text{Time} = \frac{\text{Volume Needed}}{\text{Net Rate}} $$
$$ \text{Time} = \frac
```

## code (39 prompt tokens)

Prompt:

> Write a Python function that merges overlapping intervals in a list of (start, end) pairs, plus a few asserts that test it.

### SDPA (default pick) (prefill 122 ms, decode 11.5 tok/s)

```text
```python
def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Merge overlapping intervals in a list of (start, end) pairs.
    
    Args:
        intervals: A list of tuples, each representing an interval as (start, end).
        
    Returns:
        A list of merged non-overlapping intervals, sorted by start time.
    """
    if not intervals:
        return []
    
    # Sort intervals by start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    
    merged = [sorted_intervals[0]]
    
    for current in sorted_intervals[1:]:
        last = merged[-1]
        # Check if current interval overlaps with the last merged interval
        if current[0] <= last[1]:
            # Merge: update the end of the last interval if current extends further
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            # No overlap, add current interval as a new merged interval
            merged.append(current)
    
    return merged


# Tests
if __
```

### FA2 (SDPA flash backend) (prefill 133 ms, decode 14.0 tok/s, first 79 tokens identical to SDPA)

```text
```python
def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Merge overlapping intervals in a list of (start, end) pairs.
    
    Args:
        intervals: A list of tuples, each representing an interval as (start, end).
        
    Returns:
        A list of merged, non-overlapping intervals sorted by start time.
    """
    if not intervals:
        return []
    
    # Sort intervals by start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    
    merged = [sorted_intervals[0]]
    
    for current in sorted_intervals[1:]:
        last = merged[-1]
        # Check if current interval overlaps with the last merged interval
        if current[0] <= last[1]:
            # Merge: update the end of the last interval if current extends further
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            # No overlap, add current interval as a new merged interval
            merged.append(current)
    
    return merged


# Tests
if __
```

### cuDNN (SDPA cuDNN backend) (prefill 99 ms, decode 13.9 tok/s, first 256 tokens identical to SDPA)

```text
```python
def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Merge overlapping intervals in a list of (start, end) pairs.
    
    Args:
        intervals: A list of tuples, each representing an interval as (start, end).
        
    Returns:
        A list of merged non-overlapping intervals, sorted by start time.
    """
    if not intervals:
        return []
    
    # Sort intervals by start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    
    merged = [sorted_intervals[0]]
    
    for current in sorted_intervals[1:]:
        last = merged[-1]
        # Check if current interval overlaps with the last merged interval
        if current[0] <= last[1]:
            # Merge: update the end of the last interval if current extends further
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            # No overlap, add current interval as a new merged interval
            merged.append(current)
    
    return merged


# Tests
if __
```

### FlashTile (prefill 106 ms, decode 13.8 tok/s, first 79 tokens identical to SDPA)

```text
```python
def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Merge overlapping intervals in a list of (start, end) pairs.
    
    Args:
        intervals: A list of tuples, each representing an interval as (start, end).
        
    Returns:
        A list of merged, non-overlapping intervals sorted by start time.
    """
    if not intervals:
        return []
    
    # Sort intervals by start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    
    merged = [sorted_intervals[0]]
    
    for current in sorted_intervals[1:]:
        last = merged[-1]
        # Check if current interval overlaps with the last merged interval
        if current[0] <= last[1]:
            # Merge: update the end of the last interval if current extends further
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            # No overlap, add current interval as a new merged interval
            merged.append(current)
    
    return merged


# Tests
if __
```

## docs_qa (22263 prompt tokens)

Prompt (repo files, then the question):

> ... Using only the documents above: on which GPUs, and for which cases, is FlashTile slower than another attention backend, and what does the README list as its weak spots? Answer in under 150 words.

### SDPA (default pick) (prefill 2442 ms, decode 14.0 tok/s)

```text
FlashTile is slower than PyTorch’s cuDNN backend on datacenter GPUs (H200 and B200) for most shapes, with B200 showing 1.6–3.5x slower performance. It also lags behind SDPA’s FlashAttention-2 path on the RTX PRO 6000 for backward passes and specific forward cases.

The README lists these weak spots:
1. **cuDNN superiority:** cuDNN is significantly faster on datacenter GPUs.
2. **Head dim 256:** This is the weakest case, particularly for Qwen3.5 shapes, where FlashTile is 1.38x–3.65x slower than competitors.
3. **Backward pass:** It lags the forward pass, reaching only 0.55–0.85x the speed of cuDNN or FA2.
4. **Decoding:** Long-context decoding at batch 1 underutilizes the GPU due to the lack of a split-KV kernel.
5. **Atomic backward:** The single-pass atomic variant did not provide consistent speedups.
6. **Missing features:** No support for variable-length packing, p
```

### FA2 (SDPA flash backend) (prefill 2608 ms, decode 14.0 tok/s, first 256 tokens identical to SDPA)

```text
FlashTile is slower than PyTorch’s cuDNN backend on datacenter GPUs (H200 and B200) for most shapes, with B200 showing 1.6–3.5x slower performance. It also lags behind SDPA’s FlashAttention-2 path on the RTX PRO 6000 for backward passes and specific forward cases.

The README lists these weak spots:
1. **cuDNN superiority:** cuDNN is significantly faster on datacenter GPUs.
2. **Head dim 256:** This is the weakest case, particularly for Qwen3.5 shapes, where FlashTile is 1.38x–3.65x slower than competitors.
3. **Backward pass:** It lags the forward pass, reaching only 0.55–0.85x the speed of cuDNN or FA2.
4. **Decoding:** Long-context decoding at batch 1 underutilizes the GPU due to the lack of a split-KV kernel.
5. **Atomic backward:** The single-pass atomic variant did not provide consistent speedups.
6. **Missing features:** No support for variable-length packing, p
```

### cuDNN (SDPA cuDNN backend) (prefill 2451 ms, decode 14.3 tok/s, first 80 tokens identical to SDPA)

```text
FlashTile is slower than PyTorch’s cuDNN backend on datacenter GPUs (H200 and B200) for most shapes, with B200 showing 1.6–3.5x slower performance. It also lags behind SDPA’s FlashAttention-2 path on the RTX PRO 6000 for backward passes and specific forward cases. Additionally, it is slower than SDPA for long-context decoding on the RTX PRO 6000.

The README lists these weak spots:
1. **cuDNN superiority:** cuDNN is significantly faster on datacenter GPUs.
2. **Head dim 256:** This is the weakest case, particularly for Qwen3.5 shapes.
3. **Backward pass:** It lags behind the forward pass performance.
4. **Decoding:** Long-context decoding at batch 1 underutilizes the GPU due to the lack of a split-KV kernel.
5. **Atomic backward:** The single-pass atomic variant did not provide consistent speedups.
6. **Missing features:** No support for variable-length packing, paged KV cache, sliding window, dropout, or FP8 attention.
7. **J
```

### FlashTile (prefill 2538 ms, decode 14.1 tok/s, first 30 tokens identical to SDPA)

```text
FlashTile is slower than PyTorch’s cuDNN backend on datacenter GPUs (H200 and B200) for most workloads, particularly at long contexts and head dimension 256. On B200, cuDNN is 1.6–3.5x faster. On H200, FlashTile reaches only 55–88% of cuDNN’s speed. On the RTX PRO 6000, FlashTile is slower than PyTorch’s FA2 backend for backward passes (0.60–0.85x) and for head dimension 256 workloads.

The README lists these weak spots:
1.  **cuDNN superiority:** cuDNN is significantly faster on datacenter GPUs.
2.  **Head dim 256:** This is the weakest case, making Qwen3.5-4B attention 1.38–3.65x slower than competitors.
3.  **Backward pass:** It lags the forward pass, being 0.55–0.85x the speed of FA2/cuDNN.
4.  **Decoding:** Long
```

## code_qa (33846 prompt tokens)

Prompt (repo files, then the question):

> ... Using only the code above: how does flashtile/hf.py decide between the causal Triton path, the decode path and the masked tiled path, and what does the Triton forward kernel do differently on diagonal tiles? Answer in under 150 words.

### SDPA (default pick) (prefill 3822 ms, decode 14.0 tok/s)

```text
In `hf.py`, if `attention_mask` is `None`, it checks `is_causal`. If causal and `lq == 1`, it treats it as a decode call (non-causal Triton). If causal and `lq > 1`, it passes `q_pos`/`k_pos` to enable the causal Triton path. If a mask exists, it converts additive masks to boolean and routes to the tiled path.

The Triton forward kernel splits the key loop into two phases: "full" tiles (no masking) and "masked" tiles (diagonal/ragged). On masked tiles, it applies the causal condition `offs_n <= offs_m + DIAG` to set invalid scores to `-inf`. It also updates the running max `m_i` and rescales the accumulator `acc` and sum `l_i` using `alpha = exp(m_i - m_new)` to maintain numerical stability, whereas full tiles skip the mask check and the `-inf` handling.
```

### FA2 (SDPA flash backend) (prefill 4220 ms, decode 13.9 tok/s, first 35 tokens identical to SDPA)

```text
In `hf.py`, if `attention_mask` is `None`, it checks `is_causal`. If causal and `lq == 1`, it treats it as decode (non-causal). If causal and `lq > 1`, it passes `q_pos`/`k_pos` to `attention()`, which routes to the Triton path if positions are contiguous. If `attention_mask` is provided, it converts additive masks to boolean and calls `attention()` with `allow_mask`, forcing the tiled path.

The Triton forward kernel splits the key loop into two phases: "full" tiles (no masking) and "masked" tiles (diagonal/ragged). On masked tiles, it applies causal masking via `offs_n <= offs_m + DIAG` and handles boundary checks (`offs_n < LK`). It also updates the running max (`m_i`) and rescales the accumulator (`acc`) and sum (`l_i`) using `alpha = exp(m_i - m_new)` to maintain numerical stability, whereas full tiles skip the causal mask check.
```

### cuDNN (SDPA cuDNN backend) (prefill 3841 ms, decode 14.1 tok/s, first 35 tokens identical to SDPA)

```text
In `hf.py`, if `attention_mask` is `None`, it checks `is_causal`. If causal and `lq == 1`, it treats it as decode (non-causal). If causal and `lq > 1`, it passes `q_pos`/`k_pos` to `attention()`, which routes to the Triton path if positions are contiguous. If `attention_mask` is provided, it converts additive masks to boolean and calls `attention()` with `allow_mask`, forcing the tiled path.

In the Triton forward kernel, diagonal tiles (where `MASKED` is true) apply causal masking via `offs_n <= offs_m + DIAG` and handle potential `-inf` scores by computing `m_shift` safely to avoid `-inf - (-inf)`. Non-diagonal tiles skip masking and assume no `-inf` scores, simplifying the max/sum updates.
```

### FlashTile (prefill 4048 ms, decode 14.2 tok/s, first 35 tokens identical to SDPA)

```text
In `hf.py`, if `attention_mask` is `None`, it checks `is_causal`. If causal and `lq == 1`, it treats it as decode (non-causal). If causal and `lq > 1`, it passes `q_pos`/`k_pos` to `attention()`, which routes to the Triton path if positions are contiguous. If `attention_mask` is provided, it converts additive masks to boolean and calls `attention()` with `allow_mask`, forcing the tiled path.

The Triton forward kernel splits the key loop into two phases: "full" tiles (no masking) and "masked" tiles (diagonal/ragged). On masked tiles, it applies causal logic (`offs_n <= offs_m + DIAG`) and boundary checks (`offs_n < LK`) to set invalid scores to `-inf`. It then updates the running max (`m_i`) and sum (`l_i`) using these masked scores, ensuring numerical stability by shifting exponents based on the new max, before accumulating the output.
```

