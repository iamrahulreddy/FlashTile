# FlashTile

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![CI](https://github.com/iamrahulreddy/FlashTile/actions/workflows/ci.yml/badge.svg)](https://github.com/iamrahulreddy/FlashTile/actions)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iamrahulreddy/FlashTile/blob/main/demo/FlashTile_Demo.ipynb)

**FlashTile** is a reference implementation of **Flash Attention** (V1 and V2) focused on clarity. It includes readable PyTorch implementations of IO-aware tiling, online softmax, GQA/MQA variants, and a Triton kernel, with $O(N)$ memory behavior for the V1/V2 forward and backward passes.

### Overview
- **Flash-V1/V2**: Custom autograd implementations with memory-efficient recomputation in the backward pass.
- **GQA/MQA**: Reference implementations for grouped-query and multi-query attention.
- **Triton kernel**: Forward-only fused kernel benchmarked at **11.8×** naive speed at 8K in the committed A100 run.
- **Validation**: Full benchmarks and the 146-test suite were validated on **NVIDIA A100-80GB**. An archived **H100-80GB** cross-check run (145/146 before a since-fixed stress-test threshold) is documented below.
- **Compatibility**: Works with `torch.compile`, mixed precision (`fp16`/`bf16`), and a masked fallback for arbitrary attention patterns.

The Python modules are written to be easy to inspect. They are useful for understanding the algorithm and validating memory behavior, while the Triton kernel is the performance-oriented path in this repo.

## Performance Results

Benchmarked on **NVIDIA A100-SXM4-80GB** (CUDA 12.8, PyTorch 2.8.0) with `batch_size=4, embed_dim=768, num_heads=12`:

### Memory Scaling (Causal Attention)

| Sequence Length | Naive Memory | Flash V2 Memory | Reduction | Triton Kernel Speed vs Naive |
|-----------------|-------------|-----------------|-----------|------------------------------|
| 512 | 282 MB | 54 MB | **5.2×** | 1.9× faster |
| 1,024 | 1,026 MB | 90 MB | **11.4×** | 2.8× faster |
| 2,048 | 3,958 MB | 162 MB | **24.4×** | 3.9× faster |
| 4,096 | 15,586 MB | 306 MB | **50.9×** | **6.4× faster** |
| 8,192 | 61,907 MB | 594 MB | **104.2×** | **11.8× faster** |

At 8K tokens in the committed A100 run, naive attention peaked at **61.9 GB** while Flash V2 used **594 MB**. In the same configuration, the Triton kernel ran **11.8×** faster than naive attention.
> *Note on the 104× figure: The memory reduction scales with sequence length due to O(N²) vs O(N) complexity. At 2K tokens it is ~24×, scaling linearly up to ~104× at 8K in the committed A100 benchmark run.*

The Python **V1/V2** implementations provide true **O(N)** memory for both forward and backward via custom autograd. **GQA/MQA** keep **O(N)** forward memory but rely on standard autograd for backward, which is **O(N²)** during training. Because the Python implementations iterate over blocks in Python, they are slower than naive attention's fused cuBLAS calls. For speed, use the included **Triton kernel** (forward-only) or PyTorch's built-in `F.scaled_dot_product_attention()`.

The memory gap becomes significant at longer sequence lengths. By 4K+ tokens, naive attention often becomes the limiting factor while the tiled implementations remain tractable.

```bash
# Reproduce these results
python benchmark/benchmark.py --max-seq-len 8192 --batch-size 4 --embed-dim 768 --num-heads 12
```

## The Algorithm

Flash Attention gets O(N) memory from two ideas:

### 1. Block-wise Tiling

Instead of computing the full NxN attention matrix, we process in small BxB blocks:

```
Standard: Materialize full NxN        Flash: Process BxB blocks
┌─────────────────────────┐          ┌──┬──┬──┬──┐
│                         │          │▓▓│  │  │  │ → Process
│   16 MILLION elements   │    →     ├──┼──┼──┼──┤    block by
│   (4096 × 4096)         │          │  │▓▓│  │  │    block
│                         │          ├──┼──┼──┼──┤
└─────────────────────────┘          │  │  │▓▓│  │ → Never store
     O(N²) memory                    └──┴──┴──┴──┘    full matrix!
                                         O(N) memory
```

### 2. Online Softmax

Softmax needs the full row to compute the denominator.

Flash Attention handles this by updating running statistics block by block:

```python
# Initialize running statistics
m = -inf  # running max (for numerical stability)
l = 0     # running sum of exp(scores - max)
O = 0     # running weighted output

# For each new block of scores S:
m_new = max(m, S.max())                           # Update max
l_new = l * exp(m - m_new) + sum(exp(S - m_new))  # Rescale and accumulate
O = O * (l * exp(m - m_new)) / l_new + ...        # Rescale output
m, l = m_new, l_new                               # Update state
```

This produces the same output as standard attention while keeping memory O(N).

## Installation

```bash
# Clone the repository
git clone https://github.com/iamrahulreddy/FlashTile.git
cd FlashTile

# Install with pip (recommended)
pip install -e .

# With all optional dependencies
pip install -e ".[all]"  # Includes Triton, benchmarks, demo, dev tools
```

**Requirements**: Python 3.9+, PyTorch 2.0+, CUDA-capable GPU (recommended)

## Quick Start

```python
import torch
from flashtile import FlashAttentionV2, get_attention

# Create Flash Attention V2 with causal masking (for decoder models)
model = FlashAttentionV2(embed_dim=512, num_heads=8, causal=True)

# Input: (batch, seq_len, embed_dim)
x = torch.randn(2, 4096, 512).cuda()
model = model.cuda()

# Forward pass - works with sequences that would OOM with naive attention
output, _ = model(x)
print(f"Output shape: {output.shape}")  # torch.Size([2, 4096, 512])

# Or use the factory function
model = get_attention("flash_v2", embed_dim=512, num_heads=8, causal=True)
```

## Demo

A small Gradio app is included for comparing memory usage across implementations:

```bash
# Run the interactive showcase
python demo/app.py
```

## Implementations

| Implementation | Memory | Key Feature | Use Case |
|----------------|--------|-------------|----------|
| **NaiveAttention** | O(N²) | Returns attention weights | Correctness reference, visualization |
| **FlashAttentionV1** | O(N) | Online softmax, memory-efficient backward | Algorithm study, reference baseline |
| **FlashAttentionV2** | O(N) | Swapped loops + causal skip (~50% faster) | Decoder models (GPT-style) |
| **SlidingWindowAttention** | O(N×W) | Local attention window (Mistral-style) | Long context with local patterns |
| **GroupedQueryAttention** | O(N) forward | Shared KV heads (LLaMA 2 style) | Inference optimization |
| **MultiQueryAttention** | O(N) forward | Single KV head | Maximum KV cache savings |
| **TritonFlashAttention** | O(N) | Low-level fused GPU kernel | Max performance (Inference/Forward-only) |

### Flash Attention V2: Causal Optimization

```python
from flashtile import FlashAttentionV2

# V2 with causal masking skips ~50% of block computations
model = FlashAttentionV2(embed_dim=512, num_heads=8, causal=True)

# Check efficiency stats
efficiency = model.get_causal_efficiency(seq_len=1024)
print(f"Compute savings: {efficiency['compute_savings_percent']:.1f}%")
# Output: Compute savings: 46.9%
```

### Grouped-Query Attention (GQA)

```python
from flashtile import GroupedQueryAttention, MultiQueryAttention

# GQA: 8 query heads share 2 KV heads (4x KV cache savings)
gqa = GroupedQueryAttention(
    embed_dim=512, num_heads=8, num_kv_heads=2, causal=True
)

# MQA: All heads share 1 KV head (8x KV cache savings)
mqa = MultiQueryAttention(embed_dim=512, num_heads=8, causal=True)

# Memory comparison
mem = gqa.get_memory_usage(batch_size=2, seq_len=4096)
print(f"KV cache savings: {mem['kv_savings_ratio']:.1f}x vs MHA")
```

### Using torch.compile (PyTorch 2.0+)

```python
import torch
from flashtile import FlashAttentionV2

model = FlashAttentionV2(embed_dim=512, num_heads=8, causal=True).cuda()
model = torch.compile(model)  # Enables kernel fusion

x = torch.randn(2, 2048, 512).cuda()
out, _ = model(x)  # Compiled execution
```

## Project Structure

```
FlashTile/
├── flashtile/                     # Core package
│   ├── attention/                 # Attention implementations
│   │   ├── base_attention.py      # Abstract base class
│   │   ├── naive_attention.py     # O(N²) baseline
│   │   ├── flash_attention_v1.py  # O(N) Flash Attention V1
│   │   ├── flash_attention_v2.py  # O(N) Flash Attention V2
│   │   ├── sliding_window_attention.py  # O(N×W) Sliding Window
│   │   ├── masked_attention.py    # Custom mask fallback
│   │   ├── amp_compat.py          # AMP compatibility utilities
│   │   └── grouped_query_attention.py  # GQA and MQA
│   ├── kernels/                   # GPU kernels
│   │   ├── triton_flash_kernel.py # Custom Triton kernel
│   │   └── kernel_benchmarks.py   # Performance benchmarking
│   └── utils/                     # Profiling utilities
│       ├── memory_profiler.py     # Memory/time measurement
│       ├── attention_visualizer.py # Attention pattern visualization
│       ├── visualization.py       # Chart generation utilities
│       └── kernel_profiler.py     # CUDA kernel profiling
├── benchmark/                     # Performance benchmarking
├── demo/                          # Interactive Demos
├── docs/                          # Documentation
│   ├── ARCHITECTURE.md            # Technical design
│   ├── ALGORITHM.md               # Algorithm deep-dive
│   ├── USAGE.md                   # API reference
│   └── PERFORMANCE.md             # Benchmarking methodology
└── tests/                         # Comprehensive test suite (146 tests on A100)
```

## Documentation

- **[Architecture](docs/ARCHITECTURE.md)** — Technical design and data flow
- **[Algorithm](docs/ALGORITHM.md)** — Mathematical derivation of online softmax
- **[Usage Guide](docs/USAGE.md)** — Detailed API reference
- **[Performance](docs/PERFORMANCE.md)** — Benchmarking methodology and A100 results

## Comparison with Other Libraries

| Library | Implementation | Memory Efficient? | Backward Pass | Purpose |
|---------|---------------|-------------------|---------------|---------|
| [flash-attn](https://github.com/Dao-AILab/flash-attention) | Optimized CUDA | ✅ Yes | ✅ Optimized | Production inference |
| [xFormers](https://github.com/facebookresearch/xformers) | Optimized CUDA | ✅ Yes | ✅ Optimized | Meta's efficient ops |
| PyTorch `F.scaled_dot_product_attention` | Fused kernel | ✅ Yes | ✅ Optimized | Built-in (PyTorch 2.0+) |
| **FlashTile** (This repo) | Python + Triton | ✅ Yes (V1/V2) | ✅ Recomputation (V1/V2) | **Educational** |

> For production workloads, prefer PyTorch's built-in `F.scaled_dot_product_attention()` or Dao-AI's `flash-attn`.
> FlashTile is meant to be a readable reference implementation with a small Triton path for comparison.

**Implementation Notes**:
- **V1/V2**: True O(N) memory for forward AND backward (custom autograd)
- **GQA/MQA**: O(N) forward, but O(N²) backward (standard autograd—good for inference)
- **Triton**: Forward-only, fastest implementation (11.8× vs naive at 8K seq)

## Key Features

### Memory-Efficient Backward Pass

Unlike simple implementations, FlashTile includes gradient checkpointing in the backward pass:

```python
# Forward: O(N) memory - never stores attention matrix
# Backward: Recomputes attention block-by-block
# Result: True O(N) memory for both forward AND backward

model = FlashAttentionV2(embed_dim=512, num_heads=8)
x = torch.randn(2, 2048, 512, requires_grad=True).cuda()
model = model.cuda()

out, _ = model(x)
loss = out.mean()
loss.backward()  # Memory-efficient backward pass
```

### Triton Kernel with Causal Masking

```python
from flashtile import TritonFlashAttention

# Custom Triton module with causal support
kernel = TritonFlashAttention(embed_dim=512, num_heads=8, causal=True).cuda().half().eval()

x = torch.randn(2, 2048, 512).cuda().half()

with torch.no_grad():
    out, _ = kernel(x)  # Fused GPU execution (forward-only)
```

### Validation

```
A100: 146/146 passed | Archived H100 run: 145/146 passed
  ✓ 25 correctness tests (V1/V2/GQA vs naive reference)
  ✓ 16 precision tests (fp32/fp16/bf16, GradScaler, numerical stability)
  ✓ 18 stress tests (up to 32K sequence length, memory leak detection)
  ✓ 15 torch.compile tests (inductor, eager, gradient compatibility)
  ✓ 5 Triton kernel tests
  ✓ Max numerical error: 1.5e-07 (machine epsilon)
  ✗ 1 H100 stress test: CUDA allocator retains workspace memory (not a code bug)
```

> *H100 artifacts are from a pre-fix run — see [flashtile_results_H100/NOTE.md](flashtile_results_H100/NOTE.md) for details.*
> *CI covers the CPU-compatible unit subset. GPU-heavy suites (performance, stress, Triton, torch.compile) are archived separately under `flashtile_results_A100/` and `flashtile_results_H100/`.*
> *The archived A100/H100 validation JSON and logs were generated on the Feb 27, 2026 validation run. This cleanup updates public-facing docs/tests/artifacts without claiming a fresh rerun.*

## References

1. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**
   Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re
   *NeurIPS 2022* — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

2. **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**
   Tri Dao
   *ICLR 2024* — [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)

3. **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints**
   Joshua Ainslie et al.
   *EMNLP 2023* — [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)

4. **Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations**
   Philippe Tillet, H.T. Kung, David Cox
   *MAPL 2019* — [Triton Documentation](https://triton-lang.org/)

## Author

**Muskula Rahul** — [@iamrahulreddy](https://github.com/iamrahulreddy)

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
