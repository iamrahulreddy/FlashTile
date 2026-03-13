"""
Flash Attention Kernel Benchmarks
=================================

This module provides comprehensive benchmarking utilities for comparing
FlashTile kernel performance against PyTorch baselines.

Metrics Collected
-----------------
- Latency (ms)
- Throughput (TFLOPS)
- Memory bandwidth utilization (TB/s and % of peak)
- Memory usage (peak GPU memory)

Usage
-----
```bash
# Run all benchmarks
python -m flashtile.kernels.kernel_benchmarks

# Run specific benchmark
python -m flashtile.kernels.kernel_benchmarks --benchmark attention

# Export results to CSV
python -m flashtile.kernels.kernel_benchmarks --output results.csv
```

References
----------
1. NVIDIA Performance Analysis: https://developer.nvidia.com/nsight-systems
2. PyTorch Benchmark Utilities: https://pytorch.org/docs/stable/benchmark_utils.html
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
import warnings

import torch

# Hardware specs for common GPUs (used for utilization calculation)
GPU_SPECS = {
    "A100-SXM4-80GB": {
        "peak_tflops_fp16": 312.0,
        "peak_tflops_fp32": 156.0,
        "hbm_bandwidth_tb": 2.0,
        "sram_per_sm_kb": 192,
    },
    "A100-SXM4-40GB": {
        "peak_tflops_fp16": 312.0,
        "peak_tflops_fp32": 156.0,
        "hbm_bandwidth_tb": 1.6,
        "sram_per_sm_kb": 192,
    },
    "H100-SXM5": {
        "peak_tflops_fp16": 989.0,
        "peak_tflops_fp32": 495.0,
        "hbm_bandwidth_tb": 3.35,
        "sram_per_sm_kb": 256,
    },
    "RTX4090": {
        "peak_tflops_fp16": 165.0,
        "peak_tflops_fp32": 82.5,
        "hbm_bandwidth_tb": 1.0,
        "sram_per_sm_kb": 128,
    },
    "RTX3090": {
        "peak_tflops_fp16": 71.0,
        "peak_tflops_fp32": 35.5,
        "hbm_bandwidth_tb": 0.936,
        "sram_per_sm_kb": 128,
    },
    "default": {
        "peak_tflops_fp16": 100.0,
        "peak_tflops_fp32": 50.0,
        "hbm_bandwidth_tb": 1.0,
        "sram_per_sm_kb": 128,
    },
}


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    name: str
    batch_size: int
    seq_len: int
    num_heads: int
    head_dim: int
    causal: bool

    # Timing
    latency_ms: float
    throughput_tflops: float

    # Memory
    memory_accessed_gb: float
    memory_bandwidth_tb: float
    memory_utilization_pct: float

    # Compute
    compute_utilization_pct: float

    # Comparison
    vs_baseline_speedup: Optional[float] = None
    baseline_name: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        return {
            "name": self.name,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "causal": self.causal,
            "latency_ms": f"{self.latency_ms:.3f}",
            "throughput_tflops": f"{self.throughput_tflops:.2f}",
            "memory_bandwidth_tb": f"{self.memory_bandwidth_tb:.3f}",
            "memory_utilization_pct": f"{self.memory_utilization_pct:.1f}",
            "compute_utilization_pct": f"{self.compute_utilization_pct:.1f}",
            "vs_baseline_speedup": (
                f"{self.vs_baseline_speedup:.2f}x"
                if self.vs_baseline_speedup is not None
                else "N/A"
            ),
        }


def get_gpu_specs() -> Dict[str, Any]:
    """Detect GPU and return specs."""
    if not torch.cuda.is_available():
        return GPU_SPECS["default"]

    gpu_name = torch.cuda.get_device_name(0)

    # Match known GPUs
    for key in GPU_SPECS:
        if key.lower() in gpu_name.lower():
            return GPU_SPECS[key]

    # Return default if unknown
    return GPU_SPECS["default"]


def attention_flops(batch_size: int, seq_len: int, num_heads: int, head_dim: int, causal: bool) -> int:
    """
    Calculate FLOPs for attention operation.

    Attention = softmax(QK^T / sqrt(d)) @ V

    FLOPs breakdown:
    - QK^T: 2 * B * H * N * N * d (matmul)
    - Softmax: ~5 * B * H * N * N (exp, sum, div per element)
    - PV: 2 * B * H * N * N * d (matmul)

    For causal, we process ~half the positions on average.
    """
    qkt_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    softmax_flops = 5 * batch_size * num_heads * seq_len * seq_len
    pv_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim

    total = qkt_flops + softmax_flops + pv_flops

    if causal:
        # Causal mask means we process approximately half the attention
        total = total // 2

    return total


def attention_memory_bytes(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
    memory_efficient: bool = True,
) -> int:
    """Calculate approximate memory traffic for attention.

    Parameters
    ----------
    memory_efficient : bool
        If True, use Flash-style O(N) traffic estimate.
        If False, use naive attention O(N²) traffic estimate.
    """
    if memory_efficient:
        # Flash-like fused path:
        # Read Q, K, V once + write O once.
        return 4 * batch_size * num_heads * seq_len * head_dim * dtype_bytes

    # Naive path (materialized scores/probabilities), approximate:
    # 4 * N^2 term from score/prob traffic + 5 * N*d term from QKV/V/O traffic.
    attn_quadratic = 4 * batch_size * num_heads * seq_len * seq_len * dtype_bytes
    linear_terms = 5 * batch_size * num_heads * seq_len * head_dim * dtype_bytes
    return attn_quadratic + linear_terms


def benchmark_function(
    fn: Callable,
    args: tuple,
    warmup: int = 10,
    iters: int = 100,
    sync: bool = True,
) -> float:
    """
    Benchmark a function and return average latency in milliseconds.

    Parameters
    ----------
    fn : Callable
        Function to benchmark.
    args : tuple
        Arguments to pass to function.
    warmup : int
        Number of warmup iterations.
    iters : int
        Number of timed iterations.
    sync : bool
        Whether to synchronize CUDA after each call.

    Returns
    -------
    float
        Average latency in milliseconds.
    """
    # Warmup
    for _ in range(warmup):
        fn(*args)
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()

    # Timed runs
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(iters):
            fn(*args)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event) / iters
    else:
        start = time.perf_counter()
        for _ in range(iters):
            fn(*args)
        elapsed_ms = (time.perf_counter() - start) * 1000 / iters

    return elapsed_ms


def benchmark_attention(
    implementation: str,
    batch_size: int = 2,
    seq_len: int = 2048,
    num_heads: int = 8,
    head_dim: int = 64,
    causal: bool = True,
    warmup: int = 10,
    iters: int = 100,
    baseline_latency: Optional[float] = None,
) -> BenchmarkResult:
    """
    Benchmark an attention implementation.

    Parameters
    ----------
    implementation : str
        One of: 'naive', 'flash_v1', 'flash_v2', 'triton', 'sdpa'
    batch_size : int
        Batch size.
    seq_len : int
        Sequence length.
    num_heads : int
        Number of attention heads.
    head_dim : int
        Dimension per head.
    causal : bool
        Whether to use causal masking.
    warmup : int
        Warmup iterations.
    iters : int
        Benchmark iterations.
    baseline_latency : float, optional
        Latency of baseline for speedup calculation.

    Returns
    -------
    BenchmarkResult
        Benchmark results with timing and utilization metrics.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_specs = get_gpu_specs()
    use_fp16 = device == "cuda"
    input_dtype = torch.float16 if use_fp16 else torch.float32

    # Create input tensors
    embed_dim = num_heads * head_dim
    x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=input_dtype)

    # Import implementation
    if implementation == "naive":
        from flashtile.attention import NaiveAttention
        model = NaiveAttention(embed_dim, num_heads, causal=causal).to(device)
        if use_fp16:
            model = model.half()
        model.eval()
        fn = lambda: model(x)
    elif implementation == "flash_v1":
        from flashtile.attention import FlashAttentionV1
        model = FlashAttentionV1(embed_dim, num_heads, causal=causal).to(device)
        if use_fp16:
            model = model.half()
        model.eval()
        fn = lambda: model(x)
    elif implementation == "flash_v2":
        from flashtile.attention import FlashAttentionV2
        model = FlashAttentionV2(embed_dim, num_heads, causal=causal).to(device)
        if use_fp16:
            model = model.half()
        model.eval()
        fn = lambda: model(x)
    elif implementation == "triton":
        if device != "cuda":
            raise RuntimeError("Triton benchmark requires CUDA")
        try:
            from flashtile.kernels import TritonFlashAttention, HAS_TRITON as HAS_TRITON_KERNEL
            if not HAS_TRITON_KERNEL or TritonFlashAttention is None:
                raise RuntimeError("Triton not available")
            model = TritonFlashAttention(embed_dim, num_heads, causal=causal).to(device).half()
            model.eval()
            fn = lambda: model(x)
        except (RuntimeError, TypeError):
            warnings.warn("Triton not available, skipping benchmark")
            raise
    elif implementation == "sdpa":
        # PyTorch's scaled_dot_product_attention
        from torch.nn.functional import scaled_dot_product_attention

        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=input_dtype)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=input_dtype)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=input_dtype)

        fn = lambda: scaled_dot_product_attention(Q, K, V, is_causal=causal)
    else:
        raise ValueError(f"Unknown implementation: {implementation}")

    # Run benchmark
    def infer_fn():
        with torch.no_grad():
            return fn()

    latency_ms = benchmark_function(infer_fn, (), warmup=warmup, iters=iters)

    # Calculate metrics
    flops = attention_flops(batch_size, seq_len, num_heads, head_dim, causal)
    memory_efficient_impl = implementation in {"flash_v1", "flash_v2", "triton", "sdpa"}
    memory_bytes = attention_memory_bytes(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        dtype_bytes=2 if use_fp16 else 4,
        memory_efficient=memory_efficient_impl,
    )

    throughput_tflops = (flops / 1e12) / (latency_ms / 1000)
    memory_bandwidth_tb = (memory_bytes / 1e12) / (latency_ms / 1000)

    peak_tflops = gpu_specs["peak_tflops_fp16"] if use_fp16 else gpu_specs["peak_tflops_fp32"]
    compute_util = (throughput_tflops / peak_tflops) * 100
    memory_util = (memory_bandwidth_tb / gpu_specs["hbm_bandwidth_tb"]) * 100

    speedup = baseline_latency / latency_ms if baseline_latency else None

    return BenchmarkResult(
        name=implementation,
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        latency_ms=latency_ms,
        throughput_tflops=throughput_tflops,
        memory_accessed_gb=memory_bytes / 1e9,
        memory_bandwidth_tb=memory_bandwidth_tb,
        memory_utilization_pct=min(memory_util, 100),  # Cap at 100%
        compute_utilization_pct=min(compute_util, 100),
        vs_baseline_speedup=speedup,
        baseline_name="naive" if baseline_latency else None,
    )


def run_benchmark_suite(
    seq_lengths: Optional[List[int]] = None,
    output_file: Optional[str] = None,
    implementations: Optional[List[str]] = None,
    warmup: int = 10,
    iters: int = 100,
) -> List[BenchmarkResult]:
    """
    Run full benchmark suite across multiple configurations.

    Parameters
    ----------
    seq_lengths : list of int, optional
        Sequence lengths to test. Default: [512, 1024, 2048, 4096, 8192]
    output_file : str, optional
        Path to save CSV results.
    implementations : list of str, optional
        Implementations to benchmark. Default: ['naive', 'flash_v1', 'flash_v2', 'sdpa']
    warmup : int, optional
        Warmup iterations per benchmark.
    iters : int, optional
        Timed iterations per benchmark.

    Returns
    -------
    list of BenchmarkResult
        All benchmark results.
    """
    if seq_lengths is None:
        seq_lengths = [512, 1024, 2048, 4096, 8192]

    if implementations is None:
        implementations = ["naive", "flash_v1", "flash_v2", "sdpa"]

    results = []

    print("=" * 80)
    print("FlashTile Benchmark Suite")
    print("=" * 80)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
        print(f"PyTorch: {torch.__version__}")
    else:
        print("WARNING: CUDA not available, running on CPU")

    print("=" * 80)

    for seq_len in seq_lengths:
        print(f"\nSequence Length: {seq_len}")
        print("-" * 40)

        # Get baseline (naive) first
        baseline_latency = None
        try:
            baseline = benchmark_attention("naive", seq_len=seq_len, warmup=warmup, iters=iters)
            baseline_latency = baseline.latency_ms
            results.append(baseline)
            print(f"  naive:    {baseline.latency_ms:.3f}ms, {baseline.throughput_tflops:.2f} TFLOPS")
        except Exception as e:
            print(f"  naive:    FAILED ({e})")

        # Benchmark other implementations
        for impl in implementations:
            if impl == "naive":
                continue

            try:
                result = benchmark_attention(
                    impl,
                    seq_len=seq_len,
                    warmup=warmup,
                    iters=iters,
                    baseline_latency=baseline_latency,
                )
                results.append(result)

                speedup_str = (
                    f"{result.vs_baseline_speedup:.2f}x"
                    if result.vs_baseline_speedup is not None
                    else "N/A"
                )
                print(f"  {impl:10s}: {result.latency_ms:.3f}ms, {result.throughput_tflops:.2f} TFLOPS, {speedup_str} speedup")
            except Exception as e:
                print(f"  {impl:10s}: FAILED ({e})")

    # Save results
    if output_file:
        with open(output_file, "w", newline="") as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].to_dict().keys())
                writer.writeheader()
                for r in results:
                    writer.writerow(r.to_dict())
        print(f"\nResults saved to {output_file}")

    return results


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="FlashTile Kernel Benchmarks")
    parser.add_argument(
        "--seq-lengths",
        nargs="+",
        type=int,
        default=[512, 1024, 2048, 4096],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument(
        "--implementations",
        nargs="+",
        default=["naive", "flash_v1", "flash_v2", "sdpa"],
        help="Implementations to benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Benchmark iterations",
    )

    args = parser.parse_args()

    run_benchmark_suite(
        seq_lengths=args.seq_lengths,
        output_file=args.output,
        implementations=args.implementations,
        warmup=args.warmup,
        iters=args.iters,
    )


if __name__ == "__main__":
    main()
