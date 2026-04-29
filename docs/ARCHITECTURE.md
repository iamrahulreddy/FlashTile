# FlashTile Architecture

Technical overview of the codebase structure and design decisions.

## Project Structure

FlashTile is organized into three layers:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CODE                                        │
│                                                                     │
│   from flashtile import FlashAttentionV2, SlidingWindowAttention    │
│   model = FlashAttentionV2(embed_dim=512, num_heads=8, causal=True) │
│   output, _ = model(x)                                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        flashtile/                                   │
│  ┌─────────────────────────────────────────────────────────────-┐   │
│  │  attention/                        High-Level PyTorch Modules│   │
│  │  ├── base_attention.py             Abstract base class       │   │
│  │  ├── naive_attention.py            O(N²) reference           │   │
│  │  ├── flash_attention_v1.py         O(N) Flash Attention V1   │   │
│  │  ├── flash_attention_v2.py         O(N) Flash Attention V2   │   │
│  │  ├── sliding_window_attention.py   O(N×W) Sliding Window     │   │
│  │  ├── grouped_query_attention.py    GQA and MQA               │   │
│  │  └── masked_attention.py           Custom mask support       │   │
│  └─────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  kernels/                          Low-Level GPU Kernels     │   │
│  │  └── triton_flash_kernel.py        Triton kernel (forward)   │   │
│  └─────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  utils/                            Helpers                   │   │
│  │  ├── memory_profiler.py            Memory/time measurement   │   │
│  │  ├── attention_visualizer.py       Pattern visualization     │   │
│  │  └── kernel_profiler.py            CUDA kernel profiling     │   │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

## Module Overview

### `base_attention.py` - Abstract Base Class

Defines the common interface and shared functionality for all attention implementations.

**Features:**
- Parameter validation and error handling
- QKV projection and causal mask generation
- Abstract methods for `forward()` and `get_memory_usage()`

### `naive_attention.py` - Reference Implementation

Standard $O(N^2)$ attention for correctness testing and visualization.

**Use cases:**
- Ground truth for validating Flash Attention output
- Visualization (returns full attention matrix)
- Education (matches textbook implementation)
- Custom attention masks

### `flash_attention_v1.py` - Flash Attention V1

Memory-efficient $O(N)$ attention with:
- Block-wise tiling
- Online softmax
- **Memory-efficient backward pass** with gradient recomputation

**Key feature:** Custom autograd function (`FlashAttentionV1Function`) that:
- Forward: Computes attention without storing the full $N \times N$ matrix
- Backward: Recomputes attention block-by-block during backprop

### `flash_attention_v2.py` - Flash Attention V2

Improved Flash Attention with:
- **Swapped loop order** (outer Q, inner K/V) for better parallelism
- **Causal block skipping** (~50% compute savings for decoder models)
- Memory-efficient backward pass

**Key optimization:** For causal attention, if a K/V block is entirely in the "future" for a Q block, skip it entirely.

### `grouped_query_attention.py` - GQA and MQA

Attention variants optimized for inference KV cache:
- **GroupedQueryAttention:** Multiple Q heads share K/V heads
- **MultiQueryAttention:** All Q heads share single K/V head

**Memory savings (inference):** Reduces KV cache by a factor of
$`\frac{\text{num\_heads}}{\text{num\_kv\_heads}}`$.

**Training:** Uses standard PyTorch autograd. Backward pass is $O(N^2)$, not $O(N)$.
For memory-efficient training backward passes, use `FlashAttentionV1/V2` instead.

**Note:** GQA's forward pass uses block-wise tiling for $O(N)$ memory, but the
backward pass relies on standard autograd, resulting in $O(N^2)$ memory during
training.

### `sliding_window_attention.py` - Sliding Window Attention

Local attention with $O(NW)$ complexity for very long sequences
(Mistral-style).

### `masked_attention.py` - Custom Mask Support

Fallback implementation supporting arbitrary attention masks with memory-efficient chunked processing.

### `triton_flash_kernel.py` - Custom Triton Kernel

High-performance GPU kernel with:
- Explicit HBM/SRAM memory management
- Fused operations (no intermediate tensors)
- Causal masking with block-level skipping
- **Forward-only** (backward pass not implemented in Triton)

## Data Flow Diagrams

### Flash Attention Forward Flow

```
Input: x (B, N, D)
         │
         ▼
┌─────────────────────┐
│   QKV Projection    │  Linear(D → 3D)
└─────────┬───────────┘
          │
    ┌─────┼─────┐
    ▼     ▼     ▼
   Q      K     V     Each: (B, H, N, d)
    │     │     │
    ▼     ▼     ▼
┌─────────────────────────────────────────────────┐
│              BLOCK-WISE LOOP                    │
│                                                 │
│  Initialize: O=0, m=-∞, l=0                    │
│                                                 │
│  for i in range(0, N, block_size):  ← Q blocks │
│    Q_block = Q[:,:,i:i+B,:]                    │
│                                                 │
│    for j in range(0, N, block_size): ← K/V     │
│      [CAUSAL SKIP if j > i + block_size]       │
│                                                 │
│      K_block = K[:,:,j:j+B,:]                  │
│      V_block = V[:,:,j:j+B,:]                  │
│                                                 │
│      ┌─────────────────────────────┐           │
│      │  S = Q_block @ K_block^T    │ ← Only    │
│      │      / √d                   │   B×B     │
│      └─────────────┬───────────────┘   memory! │
│                    │                           │
│      ┌─────────────▼───────────────┐           │
│      │  Online Softmax Update      │           │
│      │  m_new = max(m, S.max())    │           │
│      │  l_new = rescale + sum      │           │
│      │  O = (rescale*O + S@V)/l    │           │
│      └─────────────────────────────┘           │
│                                                 │
│  Save for backward: Q, K, V, O, LSE            │
│  (NOT the attention matrix!)                   │
└─────────────────────────────────────────────────┘
            │
            ▼
     ┌──────────────┐
     │   Reshape    │  (B, H, N, d) → (B, N, D)
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │  Out Proj    │  Linear(D → D)
     └──────┬───────┘
            │
            ▼
      Output: (B, N, D)
```

### Memory-Efficient Backward Flow

```
Inputs: dO (gradient of output), saved Q, K, V, O, LSE
         │
         ▼
┌─────────────────────────────────────────────────┐
│           RECOMPUTATION BACKWARD                │
│                                                 │
│  D = rowsum(dO * O)  ← For softmax gradient    │
│                                                 │
│  for i in Q blocks:                            │
│    for j in K/V blocks:                        │
│      [CAUSAL SKIP if j > i + block_size]       │
│                                                 │
│      ┌─────────────────────────────┐           │
│      │  RECOMPUTE attention:       │           │
│      │  S = Q_block @ K_block^T    │           │
│      │  P = exp(S - LSE)           │           │
│      └─────────────────────────────┘           │
│                                                 │
│      ┌─────────────────────────────┐           │
│      │  Compute gradients:         │           │
│      │  dV += P^T @ dO             │           │
│      │  dP = dO @ V^T              │           │
│      │  dS = P * (dP - D)          │           │
│      │  dQ += dS @ K               │           │
│      │  dK += dS^T @ Q             │           │
│      └─────────────────────────────┘           │
│                                                 │
│  Memory: O(N) - never stores full attention!   │
└─────────────────────────────────────────────────┘
            │
            ▼
      dQ, dK, dV gradients
```

### GQA Data Flow

```
Input: x (B, N, D)
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│ Q Proj │ │KV Proj │  Q: full size, KV: reduced
│ (D→D)  │ │(D→D/G) │  G = num_heads / num_kv_heads
└───┬────┘ └───┬────┘
    │          │
    ▼          ▼
   Q          K, V
(B,H,N,d)  (B,H/G,N,d)
    │          │
    │    ┌─────┴─────┐
    │    │  EXPAND   │  Repeat KV heads to match Q
    │    │  K,V → H  │
    │    └─────┬─────┘
    │          │
    └────┬─────┘
         │
         ▼
┌─────────────────┐
│ Flash Attention │  Standard flash with expanded KV
└────────┬────────┘
         │
         ▼
    Output (B, N, D)
```

## Memory Comparison

### Theoretical Memory Usage

| Component | Naive | Flash V1/V2 | GQA |
|-----------|-------|-------------|-----|
| Q tensor | O(Nd) | O(Nd) | O(Nd) |
| K tensor | O(Nd) | O(Nd) | O(Nd/G) |
| V tensor | O(Nd) | O(Nd) | O(Nd/G) |
| Attention scores | **O(N^2)** | O(B^2) | O(B^2) |
| Running statistics | -- | O(N) | O(N) |
| Output | O(Nd) | O(Nd) | O(Nd) |
| **Total** | **O(N^2 + Nd)** | **O(Nd)** | **O(Nd)** |

Where:
- $N$ = sequence length
- $d$ = head dimension
- $B$ = block size (constant, typically 64)
- $`G = \frac{\text{num\_heads}}{\text{num\_kv\_heads}}`$ (GQA grouping factor)

### Concrete Example

Configuration: batch size $2$, heads $8$, sequence length $4096$, head dimension $64$

```math
\text{Naive score tensor size}
=
2 \times 8 \times 4096 \times 4096
\approx 2.68 \times 10^8 \text{ elements}
```

```math
\text{Naive memory}
\approx
2.68 \times 10^8 \times 4 \text{ bytes}
\approx 1.07 \text{ GB}
```

```math
\text{Flash block size}
=
2 \times 8 \times 64 \times 64
= 65536 \text{ elements}
```

```math
\text{Flash block memory}
=
65536 \times 4 \text{ bytes}
\approx 0.26 \text{ MB}
```

```math
\text{Reduction}
\approx
\frac{1.07 \text{ GB}}{0.26 \text{ MB}}
\approx 4{,}000\times
```

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Custom autograd function** | Enables memory-efficient backward with recomputation |
| **Store LSE not attention** | LSE (log-sum-exp) is O(N), attention is O(N^2) |
| **Triton for kernels** | Higher-level than CUDA, still efficient, easier to maintain |
| **Block size = 64 default** | Fits in SRAM, good compute efficiency, works on most GPUs |
| **Separate GQA module** | Different projection sizes require different architecture |
| **Causal as constructor arg** | Enables compile-time optimization in Triton |

---

## Extension Points

### Adding New Attention Variants

1. Create new file in `flashtile/attention/`
2. Inherit from `BaseAttention`
3. Implement custom autograd function if needed for memory efficiency
4. Implement `forward(x, attn_mask=None)`
5. Add to `__init__.py` exports
6. Add tests comparing against naive

### Adding New Kernels

1. Create new file in `flashtile/kernels/`
2. Implement kernel with same interface as `TritonFlashAttention`
3. Add validation function comparing against PyTorch reference
4. Add to `__init__.py` with graceful import fallback

---

## File Structure

```
flashtile/
├── __init__.py                    # Public API exports, factory function
├── attention/
│   ├── __init__.py                # Attention module exports
│   ├── base_attention.py          # Abstract base class
│   ├── naive_attention.py         # O(N²) reference implementation
│   ├── flash_attention_v1.py      # O(N) Flash Attention V1 + backward
│   ├── flash_attention_v2.py      # O(N) Flash Attention V2 + causal skip
│   ├── grouped_query_attention.py # GQA and MQA implementations
│   ├── sliding_window_attention.py # O(N×W) sliding window attention
│   └── masked_attention.py        # Custom mask support
├── kernels/
│   ├── __init__.py                # Kernel exports
│   ├── triton_flash_kernel.py     # Triton kernel (forward-only)
│   └── kernel_benchmarks.py       # Kernel performance benchmarking
└── utils/
    ├── __init__.py                # Utility exports
    ├── memory_profiler.py         # Memory/time profiling
    ├── kernel_profiler.py         # CUDA kernel profiling
    ├── attention_visualizer.py    # Attention pattern visualization
    └── visualization.py           # Benchmark visualization
```

## Dependencies

```
Core:
├── torch >= 2.0.0           # PyTorch framework (2.1+ recommended for AMP decorators)
└── numpy >= 1.24.0          # Numerical operations

Optional:
├── triton >= 2.1.0          # GPU kernels (for TritonFlashAttention)
├── matplotlib >= 3.7.0      # Visualization
└── psutil >= 5.9.0          # CPU memory profiling
```

*For algorithm details, see [ALGORITHM.md](ALGORITHM.md). For usage examples, see [USAGE.md](USAGE.md).*
