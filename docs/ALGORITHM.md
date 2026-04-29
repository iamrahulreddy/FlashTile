# Flash Attention Algorithm

Technical overview of the tiled attention algorithm used by Flash Attention.

## Summary

| Aspect | Standard Attention | Flash Attention |
|--------|-------------------|-----------------|
| Memory | O(N²) | O(N) |
| Speed | Baseline | Implementation-dependent |
| Accuracy | Exact | Exact (not an approximation!) |
| Max Context (8GB GPU) | ~2K tokens | ~16K+ tokens |

Flash Attention avoids materializing the full $N \times N$ attention matrix. The computation is done in tiles instead.

Whether this is faster depends on the implementation. Optimized fused kernels usually benefit from lower HBM traffic; the Python reference implementations in this repo trade that speed for clarity.

## Part 1: Understanding the Problem

### The Attention Equation

Standard scaled dot-product attention computes:

```math
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
```

Where:
- **Q** (Query): What am I looking for? - shape $Q \in \mathbb{R}^{N \times d}$
- **K** (Key): What do I contain? - shape $K \in \mathbb{R}^{N \times d}$
- **V** (Value): What information do I provide? - shape $V \in \mathbb{R}^{N \times d}$
- **N**: Sequence length (number of tokens)
- **d**: Head dimension (typically 64-128)

### The Quadratic Memory Problem

The bottleneck is computing $QK^\top$, which produces a score matrix
$S \in \mathbb{R}^{N \times N}$:

```math
Q \in \mathbb{R}^{N \times d}, \quad
K^\top \in \mathbb{R}^{d \times N}, \quad
QK^\top \in \mathbb{R}^{N \times N}
```

**Memory scaling reality check:**

| Sequence Length | Attention Matrix | Memory (FP32) | Memory (FP16) |
|-----------------|------------------|---------------|---------------|
| 512 | 262K elements | 1 MB | 0.5 MB |
| 2,048 | 4.2M elements | 16 MB | 8 MB |
| 8,192 | 67M elements | 256 MB | 128 MB |
| 32,768 | 1.07B elements | 4 GB | 2 GB |
| 131,072 | 17.2B elements | 64 GB | 32 GB |

This cost is per attention head, per layer, per batch item.

For a model like GPT-3 ($96$ layers, $96$ heads, batch size $8$) at
32K tokens:

```math
4\ \text{GB} \times 96 \times 96 \times 8 \approx 2.9\ \text{PB}
```

That is not practical.

## Part 2: Tiled Attention

### Avoid Materializing the Full Matrix

Instead of computing the entire $N \times N$ attention matrix at once, Flash Attention:

1. **Tiles** the computation into small blocks (typically 64×64)
2. **Streams** through blocks sequentially
3. **Accumulates** results using online algorithms

```
┌─────────────────────────────────────┐
│     STANDARD ATTENTION              │
│                                     │
│  Compute entire N×N matrix at once  │
│  Store in GPU memory (HBM)          │
│  Then apply softmax                 │
│  Then multiply by V                 │
│                                     │
│  Memory: O(N²)                      │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│     FLASH ATTENTION                 │
│                                     │
│  For each block of Q:               │
│    For each block of K, V:          │
│      Compute small block scores     │
│      Update running statistics      │
│      Accumulate partial output      │
│                                     │
│  Memory: O(N) — only current block! │
└─────────────────────────────────────┘
```

### Two Pieces of the Algorithm

#### Innovation 1: Block-wise Tiling

Split $Q$, $K$, and $V$ into blocks of size $B$ (for example, $64$):

```python
# Instead of:
scores = Q @ K.T  # Creates N×N matrix!

# Do this:
for i in range(0, N, block_size):
    for j in range(0, N, block_size):
        # Only B×B in memory at once
        block_scores = Q[i:i+B] @ K[j:j+B].T
        # Process and accumulate...
```

The working-set reduction is from $O(N^2)$ to $O(B^2)$, where $B \ll N$.

#### Innovation 2: Online Softmax

Softmax normally needs the full row to compute the denominator:

```math
\mathrm{softmax}(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
```

The denominator depends on all $x_j$ values in the row.

**The Solution**: Maintain running statistics that can be updated incrementally.

## Part 3: Online Softmax Deep Dive

### The Mathematical Foundation

Standard softmax with numerical stability can be written as:

```math
\mathrm{softmax}(x_i) =
\frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)},
\qquad
m = \max_j x_j
```

### The Online Algorithm

We maintain three running values:
- **m**: Running maximum (for numerical stability)
- **l**: Running sum of exponentials
- **O**: Running weighted output

When processing a new block of scores $`S_{\text{block}}`$, the online update is:

```math
\begin{aligned}
m_{\text{new}} &= \max\!\left(m_{\text{old}}, \max(S_{\text{block}})\right) \\
\alpha &= \exp(m_{\text{old}} - m_{\text{new}}) \\
l_{\text{new}} &= \alpha l_{\text{old}} + \sum \exp(S_{\text{block}} - m_{\text{new}}) \\
O_{\text{new}} &=
\frac{
\alpha l_{\text{old}} O_{\text{old}} +
\exp(S_{\text{block}} - m_{\text{new}})\,V_{\text{block}}
}{l_{\text{new}}}
\end{aligned}
```

Then update the running state with `m, l, O <- m_new, l_new, O_new`.

### Why This Works: Mathematical Proof

**Claim**: Online softmax produces identical results to standard softmax.

**Proof**:

Let $S = [S_1, S_2, \ldots, S_k]$ be the full score row split into $k$ blocks.

After processing all blocks:
- $m = \max(S)$ is the global maximum
- $l = \sum_i \exp(S_i - m) = \sum \exp(S - m)$ is the correct denominator
- $O = \frac{\sum_i \left[\exp(S_i - m)V_i\right]}{l}$ is the correct weighted sum

The key identity that makes rescaling work is:

```math
\exp(x - m_{\text{old}})\,\exp(m_{\text{old}} - m_{\text{new}})
=
\exp(x - m_{\text{new}})
```

This means we can always adjust previous computations when we discover a new maximum. **QED**

## Part 4: GPU Memory Hierarchy

Flash Attention is **IO-aware** — designed around GPU memory architecture:

```
┌─────────────────────────────────────────────────────┐
│                  GPU MEMORY HIERARCHY               │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────┐                                    │
│  │  REGISTERS  │  ~20 KB/SM   │  19 TB/s  │ 1 cyc  │
│  └──────┬──────┘                                    │
│         │ ~5x slower                                │
│  ┌──────▼──────┐                                    │
│  │    SRAM     │  192 KB/SM   │  19 TB/s  │ 30 cyc │
│  │  (L1/Shared)│  ← Flash Attention lives here!    │
│  └──────┬──────┘                                    │
│         │ ~5x slower                                │
│  ┌──────▼──────┐                                    │
│  │  L2 CACHE   │  50 MB       │  4 TB/s   │ 200 cyc│
│  └──────┬──────┘                                    │
│         │ ~1.2x slower                              │
│  ┌──────▼──────┐                                    │
│  │    HBM      │  80 GB       │  3.35 TB/s│ 400 cyc│
│  │(Global Mem) │  ← Standard attention stores here │
│  └─────────────┘                                    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Why Standard Attention is Slow

Standard attention performs multiple HBM round-trips:

```
1. Load Q, K from HBM → Compute QK^T → Store to HBM
2. Load QK^T from HBM → Compute softmax → Store to HBM  
3. Load softmax, V from HBM → Compute output → Store to HBM
```

```math
\text{HBM reads} = O(N^2 + Nd), \qquad
\text{HBM writes} = O(N^2 + Nd)
```

### Why Flash Attention is Fast

Flash Attention minimizes HBM traffic:

```
1. Load Q, K, V blocks into SRAM (fast!)
2. Compute everything in SRAM — no intermediate writes
3. Write only final output to HBM
```

```math
\text{HBM reads} = O(Nd), \qquad
\text{HBM writes} = O(Nd)
```

**Speedup source**: Not fewer FLOPs, but fewer memory stalls!

## Part 5: Complete Algorithm

### Pseudocode

```python
def flash_attention(Q, K, V, block_size=64):
    """
    Flash Attention Algorithm (V2 Logic)
    
    Args:
        Q, K, V: (batch, heads, seq_len, head_dim)
        block_size: Tile size B (typically 64)
    
    Returns:
        O: Attention output (batch, heads, seq_len, head_dim)
    """
    B, H, N, d = Q.shape
    scale = 1.0 / sqrt(d)
    
    # Initialize output and statistics
    O = zeros(B, H, N, d)
    
    # Process Q in blocks (outer loop)
    for i in range(0, N, block_size):
        # Load Q block
        Q_block = Q[:, :, i:i+block_size, :]  # (B, H, Br, d)
        
        # Initialize block-local statistics
        m_i = full((B, H, block_size), -inf)  # Running max
        l_i = zeros((B, H, block_size))        # Running sum
        O_i = zeros((B, H, block_size, d))     # Running output
        
        # Process K, V in blocks (inner loop)
        for j in range(0, N, block_size):
            # Load K, V blocks
            K_block = K[:, :, j:j+block_size, :]
            V_block = V[:, :, j:j+block_size, :]
            
            # Compute attention scores for this block
            S_ij = (Q_block @ K_block.T) * scale  # (B, H, Br, Bc)
            
            # === Online Softmax Update ===
            
            # New row-wise maximum
            m_new = maximum(m_i, S_ij.max(dim=-1))
            
            # Rescaling factors
            exp_old = exp(m_i - m_new)
            exp_new = exp(S_ij - m_new.unsqueeze(-1))
            
            # Update running sum
            l_new = exp_old * l_i + exp_new.sum(dim=-1)
            
            # Update output with proper scaling
            O_i = (l_i.unsqueeze(-1) * exp_old.unsqueeze(-1) * O_i 
                   + exp_new @ V_block) / l_new.unsqueeze(-1)
            
            # Update statistics
            m_i, l_i = m_new, l_new
        
        # Store block output
        O[:, :, i:i+block_size, :] = O_i
    
    return O
```

### Complexity Analysis

| Metric | Standard Attention | Flash Attention |
|--------|-------------------|-----------------|
| Time Complexity | O(N²d) | O(N²d) |
| Space Complexity | O(N² + Nd) | O(Nd) |
| HBM Reads | O(N² + Nd) | O(N²d/M) |
| HBM Writes | O(N² + Nd) | O(Nd) |

Where $M$ is the effective SRAM tile capacity. In optimized kernels, the reduced
HBM traffic is a major source of speedup.

## Part 6: Causal (Autoregressive) Masking

For decoder models, position $i$ should only attend to positions $j \le i$.

### Implementation

When computing S_ij, apply causal mask:

```python
# For block starting at query position i, key position j
if j > i:
    S_ij = -inf  # Mask out future positions
```

### Optimization: Skip Masked Blocks

For blocks entirely in the "future" (all masked), skip computation entirely:

```python
for j in range(0, N, block_size):
    # Skip blocks that are entirely masked
    if j >= i + block_size:
        continue  # No computation needed!
    
    # ... compute attention for this block
```

This reduces computation by about $50\%$ for causal attention. In block form,
the skip condition is $j \ge i + B$ for block size $B$.

## Part 7: Numerical Stability

### Technique 1: Max Subtraction

Always subtract the maximum before computing $\exp(\cdot)$:

```math
\exp(x)
\quad \longrightarrow \quad
\exp(x - \max(x))
```

This is mathematically equivalent (cancels in softmax) but prevents overflow.

### Technique 2: Log-Space Computation

For extreme values, compute in log space:

```math
l_{\text{new}} =
l_{\text{old}} \exp(m_{\text{old}} - m_{\text{new}})
+ \sum \exp(S - m_{\text{new}})
```

or, equivalently, in log-sum-exp form:

```math
\log l_{\text{new}}
=
\log l_{\text{old}}
+ (m_{\text{old}} - m_{\text{new}})
+ \log\!\left(
1 + \exp\!\left(
\log\!\sum \exp(S - m_{\text{new}})
- \log l_{\text{old}}
- (m_{\text{old}} - m_{\text{new}})
\right)
\right)
```

### Technique 3: Kahan Summation

For very long sequences, use compensated summation to reduce floating-point error accumulation.

## Part 8: Flash Attention V2 Improvements

Flash Attention V2 (2023) introduced several optimizations:

### 1. Swapped Loop Order

V1: Outer loop over K/V blocks, inner loop over Q blocks
V2: Outer loop over Q blocks, inner loop over K/V blocks

This loop order exposes more parallel work and also enables causal block skipping.

### 2. Better Work Partitioning

- Parallelize over sequence length within thread blocks
- Reduce synchronization points
- Better warp utilization

### 3. Practical Impact

The original FlashAttention-2 paper reports roughly 2x speedups over FlashAttention-1 on A100-class GPUs when implemented as optimized kernels. In this repo, V2 is still materially faster than V1, but the Python reference implementation is not a production-speed path.

## References

1. **Dao, T., et al.** "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

2. **Dao, T.** "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." ICLR 2024. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)

3. **Milakov, M., Gimelshein, N.** "Online Normalizer Calculation for Softmax." [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)

4. **Rabe, M.N., Staats, C.** "Self-attention Does Not Need O(n²) Memory." [arXiv:2112.05682](https://arxiv.org/abs/2112.05682)

---

*For implementation details, see [flash_attention_v1.py](../flashtile/attention/flash_attention_v1.py)*
