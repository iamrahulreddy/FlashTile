# FlashTile Performance Guide

Benchmarking methodology, optimization tips, and validated results from A100 GPU testing.

## Quick Benchmark

```python
import torch
import time
from flashtile import NaiveAttention, FlashAttentionV1, FlashAttentionV2


def benchmark(model, x, warmup=3, runs=10):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

        if x.is_cuda:
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            _ = model(x)
        if x.is_cuda:
            torch.cuda.synchronize()

    return (time.perf_counter() - start) / runs * 1000  # ms


# Configuration
batch_size = 2
seq_len = 1024
embed_dim = 512
num_heads = 8

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(batch_size, seq_len, embed_dim).to(device)

naive = NaiveAttention(embed_dim, num_heads).to(device)
flash_v1 = FlashAttentionV1(embed_dim, num_heads).to(device)
flash_v2 = FlashAttentionV2(embed_dim, num_heads, causal=True).to(device)

print(f"Naive:     {benchmark(naive, x):.2f} ms")
print(f"Flash V1:  {benchmark(flash_v1, x):.2f} ms")
print(f"Flash V2:  {benchmark(flash_v2, x):.2f} ms")
```

## A100 Benchmark Results

Validated on **NVIDIA A100-SXM4-80GB** (CUDA 12.8, PyTorch 2.8.0, Triton 3.4.0).

### Memory Scaling (Causal, batch=4, embed_dim=768, num_heads=12)

| Sequence Length | Naive Memory | Flash V1 Memory | Flash V2 Memory | GQA Memory | Triton Memory |
|-----------------|-------------|-----------------|-----------------|------------|---------------|
| 256 | 89.3 MB | 36.8 MB | 36.8 MB | 34.1 MB | 41.1 MB |
| 512 | 281.5 MB | 54.1 MB | 54.1 MB | 52.1 MB | 65.1 MB |
| 1,024 | 1026.4 MB | 90.1 MB | 90.1 MB | 87.4 MB | 113.1 MB |
| 2,048 | 3957.6 MB | 162.1 MB | 162.1 MB | 159.4 MB | 209.1 MB |
| 4,096 | 15586.1 MB | 306.1 MB | 306.1 MB | 303.4 MB | 401.1 MB |
| 8,192 | 61907.0 MB | 594.1 MB | 594.1 MB | 591.4 MB | 785.1 MB |

**Memory reduction at 8K vs Naive**:
- Flash V1: **104.2x**
- Flash V2: **104.2x**
- GQA: **104.7x**
- Triton: **78.8x**

### Execution Time (Causal, ms)

| Sequence Length | Naive | Flash V1 | Flash V2 | GQA | Triton |
|-----------------|-------|----------|----------|-----|--------|
| 256 | 0.8 ms | 7.1 ms | 6.0 ms | 5.8 ms | 0.5 ms |
| 512 | 1.8 ms | 24.2 ms | 18.9 ms | 18.4 ms | 1.0 ms |
| 1,024 | 4.9 ms | 89.8 ms | 66.7 ms | 64.2 ms | 1.8 ms |
| 2,048 | 14.3 ms | 349.2 ms | 247.6 ms | 245.2 ms | 3.7 ms |
| 4,096 | 52.7 ms | 1366.5 ms | 956.6 ms | 938.1 ms | 8.3 ms |
| 8,192 | 206.7 ms | 5329.0 ms | 3760.9 ms | 3644.6 ms | 17.6 ms |

### Triton Kernel Speedup vs Naive

| Sequence Length | Speedup |
|-----------------|---------|
| 256 | 1.4x |
| 512 | 1.9x |
| 1,024 | 2.8x |
| 2,048 | 3.9x |
| 4,096 | 6.4x |
| 8,192 | **11.8x** |

### Memory vs Speed Tradeoff

FlashTile's Python implementations are useful for studying the algorithm and
validating the memory behavior. They are not the fastest path on GPU.

| Implementation | Memory | Speed | Use Case |
|---------------|--------|-------|----------|
| Flash V1/V2 | O(N) forward/backward | Slower than naive on GPU | Algorithm study, memory-efficient training reference |
| GQA/MQA | O(N) forward, O(N^2) backward | Slower than naive on GPU | Inference-oriented KV sharing reference |
| Triton Kernel | O(N) forward | Fast (11.8x at 8K) | Forward-only inference |
| PyTorch SDPA | O(N) | Fastest | Production training/inference |
| Naive | O(N^2) | Fast fused baseline | Correctness reference, short sequences |

The Python V1/V2/GQA implementations use explicit Python loops over blocks.
Naive attention and SDPA rely on fused CUDA kernels, so they can still run
faster despite the less favorable memory complexity. The Triton kernel shows the
same tiled approach once the work is fused. In asymptotic terms, the tiled
reference path replaces $O(N^2)$ score storage with an $O(N)$ working set.

## Key Observations

1. **Memory**: Flash V1/V2/GQA all stay at $O(N)$ peak memory; GQA is slightly lower than V1/V2 in the committed A100 run, and Triton uses more workspace to achieve much higher speed
2. **Speed**: Only the Triton kernel is faster than naive; the Python implementations pay loop overhead
3. **Scaling**: Memory reduction grows linearly with sequence length because
   $`\frac{N^2}{N} = N`$
4. **Triton**: Speedup grows with sequence length because the fused kernel amortizes launch overhead
5. **Correctness**: Max numerical error of 1.5e-07 across all implementations (machine epsilon)

## Block Size Tuning

```python
# Default (most GPUs)
model = FlashAttentionV1(512, 8, block_size=64)

# High-end GPUs (A100, H100) with more SRAM
model = FlashAttentionV1(512, 8, block_size=128)

# Limited SRAM or very large head_dim
model = FlashAttentionV1(512, 8, block_size=32)
```

### Choosing Block Size

| Block Size | SRAM Required | Best For |
|------------|---------------|----------|
| 32 | ~32KB per block | Limited SRAM, debugging |
| 64 | ~128KB per block | **Default**, good balance |
| 128 | ~512KB per block | A100/H100, large head_dim |

Formula:

```math
\mathrm{SRAM}_{\mathrm{block}}
\approx
4 \cdot \mathrm{block\_size} \cdot \mathrm{head\_dim} \cdot \mathrm{sizeof(float)}
```

## Memory Measurement

```python
import torch
from flashtile import FlashAttentionV2

model = FlashAttentionV2(512, 8, causal=True).cuda()
x = torch.randn(2, 2048, 512).cuda()

# Reset stats
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

# Run forward
with torch.no_grad():
    output, _ = model(x)

# Measure
torch.cuda.synchronize()
peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
print(f"Peak memory: {peak_mb:.1f} MB")

# Get theoretical estimate
theory = model.get_memory_usage(batch_size=2, seq_len=2048)
print(f"Theoretical: {theory['total_mb']:.1f} MB")
```

## Profiling Tools

### PyTorch Profiler

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    record_shapes=True
) as prof:
    output, _ = model(x)

print(prof.key_averages().table(sort_by="cpu_time_total"))
```

### Memory Timeline (CUDA)

```python
torch.cuda.memory._record_memory_history()
output, _ = model(x)
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
```

### FlashTile Built-in Profiler

```python
from flashtile.utils import MemoryProfiler

with MemoryProfiler() as profiler:
    output, _ = model(x)

stats = profiler.stats
print(f"Peak memory: {stats.peak_allocated_mb:.1f} MB")
print(f"Duration: {stats.execution_time_ms:.2f} ms")
```

## Run Full Benchmark Suite

```bash
# Run comprehensive benchmarks
python benchmark/benchmark.py

# With custom settings (matching our A100 validation)
python benchmark/benchmark.py --max-seq-len 8192 --batch-size 4 --embed-dim 768 --num-heads 12 --theme dark

# Non-causal mode
python benchmark/benchmark.py --max-seq-len 4096 --non-causal

# Skip plots (JSON only)
python benchmark/benchmark.py --no-plots
```

Results saved to the specified `--save-dir` with:
- `benchmark_memory.png` - Memory scaling visualization
- `benchmark_performance.png` - Time comparison
- `benchmark_dashboard.png` - Comprehensive dashboard
- `benchmark_results.json` - Raw data

## Comparing with PyTorch Native

For production use, compare with PyTorch's optimized implementation:

```python
import torch
import torch.nn.functional as F
from flashtile import FlashAttentionV2

# FlashTile
flash_model = FlashAttentionV2(512, 8, causal=True).cuda()

# PyTorch Native (also uses Flash Attention under the hood on supported GPUs)
x = torch.randn(2, 2048, 512).cuda()

# FlashTile
with torch.no_grad():
    flash_out, _ = flash_model(x)

# PyTorch Native (requires manual QKV projection)
qkv_proj = flash_model.qkv_proj
qkv = qkv_proj(x)
q, k, v = qkv.chunk(3, dim=-1)
q = q.view(2, 2048, 8, 64).transpose(1, 2)
k = k.view(2, 2048, 8, 64).transpose(1, 2)
v = v.view(2, 2048, 8, 64).transpose(1, 2)

with torch.no_grad():
    native_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# Compare
print(
    "Output match:",
    torch.allclose(
        flash_out,
        native_out.transpose(1, 2).reshape(2, 2048, 512),
        atol=1e-3,
    ),
)
```

In practice, PyTorch native is much faster than FlashTile's Python implementation. FlashTile is best used as a reference implementation; PyTorch native is the production path.

## Optimization Tips

### 1. Use Causal Masking for Decoders

```python
# Saves ~50% compute
model = FlashAttentionV2(512, 8, causal=True)
```

### 2. Use torch.compile

```python
model = torch.compile(model)
```

### 3. Use FP16/BF16

```python
model = model.half()  # or .bfloat16()
x = x.half()
```

> **Pitfall**: Custom autograd functions upcast to float32 internally, so mixed precision is safe, but always use `torch.amp.autocast` for best results.

### 4. Use GQA for Inference

```python
from flashtile import GroupedQueryAttention
model = GroupedQueryAttention(512, 8, num_kv_heads=2)  # 4x KV cache savings
```

### 5. Use Appropriate Block Size

```python
# Larger blocks = fewer iterations, but more SRAM
model = FlashAttentionV2(512, 8, block_size=128)  # If you have SRAM to spare
```

## Trade-offs

| Aspect | FlashTile Python | PyTorch SDPA | Triton Kernel |
|--------|------------------|--------------|---------------|
| Memory | O(N) | O(N) | O(N) |
| Speed | Slow (Python loops) | Fast (CUDA) | Fast (11.8x at 8K) |
| Backward | Yes (V1/V2) | Yes | No |
| Customizable | High | Low | Limited |
| Educational | High | Low | Limited |
| Production | For learning | Recommended | Forward only |

**Choose FlashTile Python when:**
- Learning how Flash Attention works
- Customizing the algorithm
- Prototyping new attention variants

**Choose PyTorch SDPA when:**
- Maximum speed needed
- Production training/inference
- Backward pass required

**Choose Triton kernel when:**
- Forward-only inference
- Learning GPU kernel development
- Customizing low-level details

*For architecture details, see [ARCHITECTURE.md](ARCHITECTURE.md). For usage examples, see [USAGE.md](USAGE.md).*
