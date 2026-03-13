#!/usr/bin/env python3
"""
FlashTile Comprehensive Benchmark Suite
======================================

Run comprehensive benchmarks comparing all attention implementations:
- Memory usage vs sequence length
- Execution time vs sequence length
- Speedup analysis
- Memory reduction ratios

Usage:
    python benchmark/benchmark.py
    python benchmark/benchmark.py --max-seq-len 8192 --device cuda
    python benchmark/benchmark.py --theme dark --save-dir ./results

Output:
    - benchmark_memory.png: Memory scaling comparison
    - benchmark_performance.png: Performance comparison
    - benchmark_dashboard.png: Multi-panel dashboard
    - benchmark_results.json: Raw benchmark data
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from flashtile import (
    NaiveAttention,
    FlashAttentionV1,
    FlashAttentionV2,
    GroupedQueryAttention,
    TritonFlashAttention,
    HAS_TRITON,
)

# Optional visualization
try:
    from flashtile.utils.visualization import (
        MemoryScalingPlot,
        PerformancePlot,
        BenchmarkDashboard,
        HAS_MATPLOTLIB,
    )
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("Matplotlib not available. Plots will not be generated.")


# =============================================================================
# Benchmark Configuration
# =============================================================================

DEFAULT_CONFIG = {
    "embed_dim": 512,
    "num_heads": 8,
    "batch_size": 2,
    "seq_lengths": [256, 512, 1024, 2048, 4096],
    "num_runs": 5,
    "warmup_runs": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def _select_num_kv_heads(num_heads: int, target_ratio: int = 4) -> int:
    """
    Pick a valid GQA `num_kv_heads` that exactly divides `num_heads`.

    We target roughly `num_heads / target_ratio` (default: 1/4 of heads),
    then choose the closest divisor so model construction never fails.
    """
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")

    target = max(1, num_heads // target_ratio)
    divisors = [d for d in range(1, num_heads + 1) if num_heads % d == 0]

    # Tie-breaker prefers larger divisor (less aggressive KV sharing).
    return min(divisors, key=lambda d: (abs(d - target), -d))


# =============================================================================
# Benchmark Functions
# =============================================================================

def measure_memory(
    model: nn.Module,
    x: torch.Tensor,
    device: str,
) -> Tuple[float, Optional[float]]:
    """
    Measure peak memory usage during forward pass.
    
    Returns:
        (allocated_mb, reserved_mb) - reserved may be None on CPU
    """
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    # Forward pass
    with torch.no_grad():
        _ = model(x)
    
    if device == "cuda":
        torch.cuda.synchronize()
        allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)  # MB
        return allocated, reserved
    else:
        # CPU - can't easily measure
        return 0.0, None


def measure_time(
    model: nn.Module,
    x: torch.Tensor,
    num_runs: int = 5,
    warmup_runs: int = 2,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Measure execution time with warmup.
    
    Returns dict with mean, std, min, max times in milliseconds.
    """
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(x)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    return {
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
    }


def run_benchmark(
    model_class,
    model_kwargs: Dict,
    seq_lengths: List[int],
    batch_size: int,
    embed_dim: int,
    device: str,
    num_runs: int = 5,
    warmup_runs: int = 2,
) -> Dict[str, List]:
    """
    Run benchmark for a single model across sequence lengths.
    
    Returns dict with lists of memory and timing data.
    """
    results = {
        "seq_lengths": [],
        "memory_mb": [],
        "time_ms": [],
        "time_std": [],
    }
    
    for seq_len in seq_lengths:
        print(f"  Testing seq_len={seq_len}...", end=" ", flush=True)
        
        try:
            # Create model
            model = model_class(embed_dim=embed_dim, **model_kwargs).to(device)
            model.eval()
            
            # Create input
            x = torch.randn(batch_size, seq_len, embed_dim, device=device)
            
            # Measure memory
            mem_mb, _ = measure_memory(model, x, device)
            
            # Measure time
            timing = measure_time(model, x, num_runs, warmup_runs, device)
            
            results["seq_lengths"].append(seq_len)
            results["memory_mb"].append(mem_mb)
            results["time_ms"].append(timing["mean"])
            results["time_std"].append(timing["std"])
            
            print(f"OK (mem={mem_mb:.1f}MB, time={timing['mean']:.2f}ms)")
            
            # Cleanup
            del model, x
            if device == "cuda":
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM")
                results["seq_lengths"].append(seq_len)
                results["memory_mb"].append(float('inf'))
                results["time_ms"].append(float('inf'))
                results["time_std"].append(0)
                if device == "cuda":
                    torch.cuda.empty_cache()
            else:
                print(f"ERROR: {e}")
                raise
    
    return results


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def run_all_benchmarks(config: Dict) -> Dict:
    """Run benchmarks for all implementations."""
    
    device = config["device"]
    embed_dim = config["embed_dim"]
    num_heads = config["num_heads"]
    batch_size = config["batch_size"]
    seq_lengths = config["seq_lengths"]
    num_runs = config["num_runs"]
    warmup_runs = config["warmup_runs"]
    causal = config.get("causal", True)
    
    print("=" * 70)
    print("FlashTile Benchmark Suite")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Embed dim: {embed_dim}, Heads: {num_heads}, Batch: {batch_size}")
    print(f"Sequence lengths: {seq_lengths}")
    print(f"Causal: {causal}")
    print(f"Runs per config: {num_runs} (warmup: {warmup_runs})")
    print("=" * 70)
    
    all_results = {}
    
    # Naive Attention (reference)
    print("\n📊 Benchmarking Naive Attention (O(N²))...")
    all_results["Naive"] = run_benchmark(
        NaiveAttention,
        {"num_heads": num_heads, "causal": causal},
        seq_lengths, batch_size, embed_dim, device,
        num_runs, warmup_runs,
    )
    
    # Flash Attention V1
    print("\n📊 Benchmarking Flash Attention V1...")
    all_results["Flash V1"] = run_benchmark(
        FlashAttentionV1,
        {"num_heads": num_heads, "block_size": 64, "causal": causal},
        seq_lengths, batch_size, embed_dim, device,
        num_runs, warmup_runs,
    )
    
    # Flash Attention V2
    print("\n📊 Benchmarking Flash Attention V2...")
    all_results["Flash V2"] = run_benchmark(
        FlashAttentionV2,
        {"num_heads": num_heads, "block_size": 64, "causal": causal},
        seq_lengths, batch_size, embed_dim, device,
        num_runs, warmup_runs,
    )
    
    # GQA
    print("\n📊 Benchmarking Grouped Query Attention (GQA)...")
    num_kv_heads = _select_num_kv_heads(num_heads)
    all_results["GQA"] = run_benchmark(
        GroupedQueryAttention,
        {"num_heads": num_heads, "num_kv_heads": num_kv_heads, "causal": causal},
        seq_lengths, batch_size, embed_dim, device,
        num_runs, warmup_runs,
    )
    
    # Triton (if available)
    if HAS_TRITON and TritonFlashAttention is not None and device == "cuda":
        print("\n📊 Benchmarking Triton Kernel...")
        all_results["Triton"] = run_benchmark(
            TritonFlashAttention,
            {"num_heads": num_heads, "causal": causal},
            seq_lengths, batch_size, embed_dim, device,
            num_runs, warmup_runs,
        )
    
    return all_results


def calculate_speedups(results: Dict) -> Dict:
    """Calculate speedup ratios relative to naive."""
    if "Naive" not in results:
        return {}
    
    naive_times = np.array(results["Naive"]["time_ms"])
    speedups = {}
    
    for name, data in results.items():
        if name == "Naive":
            continue
        times = np.array(data["time_ms"])
        # Avoid division by zero/inf
        speedup = np.where(
            (naive_times == float('inf')) | (times == 0) | (times == float('inf')),
            np.nan,
            naive_times / times
        )
        speedups[name] = speedup.tolist()
    
    return speedups


def calculate_reductions(results: Dict) -> Dict:
    """Calculate memory reduction ratios."""
    if "Naive" not in results:
        return {}
    
    naive_mem = np.array(results["Naive"]["memory_mb"])
    reductions = {}
    
    for name, data in results.items():
        if name == "Naive":
            continue
        mem = np.array(data["memory_mb"])
        # Avoid division by zero/inf
        reduction = np.where(
            (naive_mem == float('inf')) | (mem == 0) | (mem == float('inf')),
            np.nan,
            naive_mem / mem
        )
        reductions[name] = reduction.tolist()
    
    return reductions


# =============================================================================
# Visualization
# =============================================================================

def create_plots(results: Dict, save_dir: Path, theme: str = "light"):
    """Create all benchmark visualizations."""
    if not HAS_MATPLOTLIB:
        print("\n⚠️  Matplotlib not available. Skipping plot generation.")
        return
    
    print("\n📈 Generating visualizations...")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    seq_lengths = results["Naive"]["seq_lengths"]
    
    # 1. Memory scaling plot
    print("  Creating memory scaling plot...")
    plot = MemoryScalingPlot(theme=theme, title="Memory Usage vs Sequence Length")
    
    colors = {
        "Naive": "#E74C3C",
        "Flash V1": "#3498DB",
        "Flash V2": "#2ECC71",
        "GQA": "#F39C12",
        "Triton": "#9B59B6",
    }
    
    for name, data in results.items():
        # Replace inf with None for plotting
        memory = [m if m != float('inf') else None for m in data["memory_mb"]]
        plot.add_series(name, seq_lengths, memory, color=colors.get(name))
    
    if torch.cuda.is_available():
        plot.add_oom_region(gpu_memory_gb=24.0)
    
    plot.finalize(add_complexity=True)
    plot.save(save_dir / "benchmark_memory.png")
    plot.close()
    
    # 2. Performance plot
    print("  Creating performance plot...")
    plot = PerformancePlot(theme=theme, title="Execution Time vs Sequence Length")
    
    for name, data in results.items():
        times = [t if t != float('inf') else None for t in data["time_ms"]]
        plot.add_series(name, seq_lengths, times, color=colors.get(name))
    
    plot.finalize()
    plot.save(save_dir / "benchmark_performance.png")
    plot.close()
    
    # 3. Dashboard
    print("  Creating dashboard...")
    dashboard = BenchmarkDashboard(theme=theme)
    
    # Prepare data for dashboard
    memory_data = {name: data["memory_mb"] for name, data in results.items()}
    time_data = {name: data["time_ms"] for name, data in results.items()}
    
    speedups = calculate_speedups(results)
    reductions = calculate_reductions(results)
    
    dashboard.plot_memory_scaling(
        seq_lengths=seq_lengths,
        memory_data=memory_data,
    )
    dashboard.plot_performance(
        seq_lengths=seq_lengths,
        time_data=time_data,
    )
    dashboard.plot_speedup(
        seq_lengths=seq_lengths,
        speedup_data=speedups,
    )
    dashboard.plot_reduction_ratio(
        seq_lengths=seq_lengths,
        reduction_data=reductions,
    )
    
    dashboard.save(save_dir / "benchmark_dashboard.png")
    dashboard.close()
    
    print(f"  ✓ Plots saved to: {save_dir}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FlashTile Comprehensive Benchmark Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run benchmarks on",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="Maximum sequence length to test",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for benchmarks",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=512,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of runs per configuration",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./benchmark_results",
        help="Directory to save results and plots",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="light",
        choices=["light", "dark"],
        help="Plot color theme",
    )
    parser.add_argument(
        "--non-causal",
        action="store_true",
        help="Disable causal masking for all benchmarked implementations",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available. Falling back to CPU.")
        args.device = "cpu"
    
    # Generate sequence lengths
    seq_lengths = [256, 512, 1024, 2048]
    if args.max_seq_len >= 4096:
        seq_lengths.append(4096)
    if args.max_seq_len >= 8192:
        seq_lengths.append(8192)
    
    # Build config
    config = {
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "batch_size": args.batch_size,
        "seq_lengths": seq_lengths,
        "num_runs": args.num_runs,
        "warmup_runs": 2,
        "device": args.device,
        "causal": not args.non_causal,
    }
    
    # Run benchmarks
    results = run_all_benchmarks(config)
    
    # Calculate derived metrics
    speedups = calculate_speedups(results)
    reductions = calculate_reductions(results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)
    
    for name, data in results.items():
        print(f"\n{name}:")
        print(f"  Memory: {data['memory_mb']}")
        print(f"  Time: {data['time_ms']}")
    
    print("\nSpeedups vs Naive:")
    for name, speedup in speedups.items():
        print(f"  {name}: {speedup}")
    
    print("\nMemory Reductions vs Naive:")
    for name, reduction in reductions.items():
        print(f"  {name}: {reduction}")
    
    # Save results to JSON
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "config": config,
        "results": results,
        "speedups": speedups,
        "reductions": reductions,
    }
    
    json_path = save_dir / "benchmark_results.json"
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n✓ Results saved to: {json_path}")
    
    # Create plots
    if not args.no_plots and HAS_MATPLOTLIB:
        create_plots(results, save_dir, args.theme)
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
