from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import torch


@dataclass
class ProfileResult:
    """Container for profiling results."""

    name: str
    latency_ms: float
    latency_std_ms: float
    num_iterations: int

    # Compute metrics
    flops: Optional[int] = None
    tflops: Optional[float] = None
    compute_utilization_pct: Optional[float] = None

    # Memory metrics
    memory_bytes: Optional[int] = None
    memory_bandwidth_tb: Optional[float] = None
    memory_utilization_pct: Optional[float] = None

    # GPU memory
    peak_memory_mb: Optional[float] = None
    allocated_memory_mb: Optional[float] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        lines = [f"ProfileResult(name='{self.name}')"]
        lines.append(f"  Latency: {self.latency_ms:.3f} ± {self.latency_std_ms:.3f} ms")

        if self.tflops is not None:
            lines.append(f"  Throughput: {self.tflops:.2f} TFLOPS")
        if self.compute_utilization_pct is not None:
            lines.append(f"  Compute Utilization: {self.compute_utilization_pct:.1f}%")
        if self.memory_bandwidth_tb is not None:
            lines.append(f"  Memory Bandwidth: {self.memory_bandwidth_tb:.3f} TB/s")
        if self.memory_utilization_pct is not None:
            lines.append(f"  Memory Utilization: {self.memory_utilization_pct:.1f}%")
        if self.peak_memory_mb is not None:
            lines.append(f"  Peak GPU Memory: {self.peak_memory_mb:.1f} MB")

        return "\n".join(lines)


class CUDATimer:
    def __init__(self) -> None:
        """Initialize CUDA timer."""
        self._start_event: Optional[torch.cuda.Event] = None
        self._end_event: Optional[torch.cuda.Event] = None

    def start(self) -> None:
        """Record start time."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._cpu_start = time.perf_counter()

    def stop(self) -> float:
        if torch.cuda.is_available() and self._end_event is not None:
            self._end_event.record()
            torch.cuda.synchronize()
            return self._start_event.elapsed_time(self._end_event)
        else:
            return (time.perf_counter() - self._cpu_start) * 1000


class KernelProfiler:

    def __init__(
        self,
        warmup: int = 10,
        iterations: int = 100,
        peak_tflops_fp16: float = 312.0,
        peak_bandwidth_tb: float = 2.0,
    ) -> None:
        """Initialize the profiler."""
        self.warmup = warmup
        self.iterations = iterations
        self.peak_tflops_fp16 = peak_tflops_fp16
        self.peak_bandwidth_tb = peak_bandwidth_tb

        self._results: List[ProfileResult] = []
        self._timer = CUDATimer()

    def profile_function(
        self,
        fn: Callable[[], Any],
        name: str,
        flops: Optional[int] = None,
        memory_bytes: Optional[int] = None,
        **kwargs,
    ) -> ProfileResult:
        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Warmup
        for _ in range(self.warmup):
            fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # Timed iterations
        latencies = []
        for _ in range(self.iterations):
            self._timer.start()
            fn()
            latency = self._timer.stop()
            latencies.append(latency)

        # Calculate statistics
        import statistics
        mean_latency = statistics.mean(latencies)
        std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

        # Calculate throughput
        tflops = None
        compute_util = None
        if flops is not None:
            tflops = (flops / 1e12) / (mean_latency / 1000)
            compute_util = (tflops / self.peak_tflops_fp16) * 100

        # Calculate memory bandwidth
        memory_bw = None
        memory_util = None
        if memory_bytes is not None:
            memory_bw = (memory_bytes / 1e12) / (mean_latency / 1000)
            memory_util = (memory_bw / self.peak_bandwidth_tb) * 100

        # Get peak memory
        peak_memory_mb = None
        allocated_mb = None
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6
            allocated_mb = torch.cuda.memory_allocated() / 1e6

        result = ProfileResult(
            name=name,
            latency_ms=mean_latency,
            latency_std_ms=std_latency,
            num_iterations=self.iterations,
            flops=flops,
            tflops=tflops,
            compute_utilization_pct=min(compute_util, 100) if compute_util is not None else None,
            memory_bytes=memory_bytes,
            memory_bandwidth_tb=memory_bw,
            memory_utilization_pct=min(memory_util, 100) if memory_util is not None else None,
            peak_memory_mb=peak_memory_mb,
            allocated_memory_mb=allocated_mb,
            metadata=kwargs,
        )

        self._results.append(result)
        return result

    @contextlib.contextmanager
    def profile(self, name: str):
        result_dict: Dict[str, Any] = {}

        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        self._timer.start()
        try:
            yield result_dict
        finally:
            latency = self._timer.stop()
            result_dict["latency_ms"] = latency

            peak_memory_mb = None
            if torch.cuda.is_available():
                peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6
            result_dict["peak_memory_mb"] = peak_memory_mb

            result = ProfileResult(
                name=name,
                latency_ms=latency,
                latency_std_ms=0.0,
                num_iterations=1,
                peak_memory_mb=peak_memory_mb,
            )
            self._results.append(result)

    def compare_implementations(
        self,
        implementations: Dict[str, Callable[[], Any]],
        flops: Optional[int] = None,
        memory_bytes: Optional[int] = None,
    ) -> Dict[str, ProfileResult]:
        results = {}
        baseline_latency = None

        for name, fn in implementations.items():
            result = self.profile_function(
                fn=fn,
                name=name,
                flops=flops,
                memory_bytes=memory_bytes,
            )
            results[name] = result

            if baseline_latency is None:
                baseline_latency = result.latency_ms

        return results

    def summary(self) -> str:
        if not self._results:
            return "No profiling results recorded."

        lines = ["=" * 60, "Kernel Profiling Summary", "=" * 60]

        for result in self._results:
            lines.append(str(result))
            lines.append("-" * 60)

        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all stored results."""
        self._results.clear()

    @property
    def results(self) -> List[ProfileResult]:
        """Get all stored profiling results."""
        return self._results.copy()


def profile_attention_implementations(
    batch_size: int = 2,
    seq_len: int = 2048,
    num_heads: int = 8,
    head_dim: int = 64,
    causal: bool = True,
) -> Dict[str, ProfileResult]:

    from flashtile.attention import (
        NaiveAttention,
        FlashAttentionV1,
        FlashAttentionV2,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    embed_dim = num_heads * head_dim
    x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

    # Calculate FLOPs
    qkt_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    softmax_flops = 5 * batch_size * num_heads * seq_len * seq_len
    pv_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    total_flops = qkt_flops + softmax_flops + pv_flops
    if causal:
        total_flops //= 2

    # Memory traffic estimates
    dtype_bytes = 2 if dtype == torch.float16 else 4
    flash_memory_bytes = 4 * batch_size * num_heads * seq_len * head_dim * dtype_bytes
    naive_memory_bytes = (
        4 * batch_size * num_heads * seq_len * seq_len * dtype_bytes
        + 5 * batch_size * num_heads * seq_len * head_dim * dtype_bytes
    )

    implementations = {
        "naive": NaiveAttention(embed_dim, num_heads, causal=causal).to(device).eval(),
        "flash_v1": FlashAttentionV1(embed_dim, num_heads, causal=causal).to(device).eval(),
        "flash_v2": FlashAttentionV2(embed_dim, num_heads, causal=causal).to(device).eval(),
    }
    if dtype == torch.float16:
        implementations = {name: model.half() for name, model in implementations.items()}

    profiler = KernelProfiler()
    results = {}

    for name, model in implementations.items():
        try:
            def _inference_call(m=model):
                with torch.no_grad():
                    return m(x)

            result = profiler.profile_function(
                fn=_inference_call,
                name=name,
                flops=total_flops,
                memory_bytes=naive_memory_bytes if name == "naive" else flash_memory_bytes,
            )
            results[name] = result
        except Exception as e:
            print(f"Failed to profile {name}: {e}")

    return results


if __name__ == "__main__":
    # Demo usage
    print("FlashTile Kernel Profiler Demo")
    print("=" * 60)

    if torch.cuda.is_available():
        results = profile_attention_implementations(
            batch_size=2,
            seq_len=1024,
            num_heads=8,
            head_dim=64,
            causal=True,
        )

        print("\nResults:")
        for name, result in results.items():
            print(f"\n{result}")
    else:
        print("CUDA not available. Profiler requires GPU.")
