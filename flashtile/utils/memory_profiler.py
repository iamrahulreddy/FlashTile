from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch


@dataclass
class MemoryStats:
    peak_allocated_mb: float
    peak_reserved_mb: float
    execution_time_ms: float
    device: str


class MemoryProfiler:
    def __init__(self, device: Optional[str] = None) -> None:
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = device

        self.stats: Optional[MemoryStats] = None
        self._start_time: float = 0.0

    def __enter__(self) -> "MemoryProfiler":
        
        if self.device == "cuda":
            # Reset peak memory statistics for isolated measurement
            torch.cuda.reset_peak_memory_stats()
            # Synchronize to ensure all previous operations complete
            torch.cuda.synchronize()

        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.device == "cuda":
            # Synchronize to ensure all operations complete before timing
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0

        if self.device == "cuda":
            peak_allocated_bytes = torch.cuda.max_memory_allocated()
            peak_reserved_bytes = torch.cuda.max_memory_reserved()
        else:
            # CPU memory tracking not directly supported
            import warnings
            warnings.warn(
                "MemoryProfiler: CPU memory tracking not supported. "
                "Peak memory stats will report 0. Use CUDA for accurate profiling.",
                stacklevel=2,
            )
            peak_allocated_bytes = 0
            peak_reserved_bytes = 0

        self.stats = MemoryStats(
            peak_allocated_mb=peak_allocated_bytes / 1e6,
            peak_reserved_mb=peak_reserved_bytes / 1e6,
            execution_time_ms=elapsed_ms,
            device=self.device,
        )


def profile_attention_function(
    fn: Callable,
    *args,
    device: Optional[str] = None,
    num_warmup: int = 5,
    num_iterations: int = 20,
    **kwargs,
) -> Dict[str, Any]:
    # Auto-select valid device for current environment.
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Warmup iterations
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = fn(*args, **kwargs)

    if device == "cuda":
        torch.cuda.synchronize()

    # Timed iterations
    execution_times: List[float] = []
    peak_memories: List[float] = []

    for _ in range(num_iterations):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.no_grad():
            _ = fn(*args, **kwargs)

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        execution_times.append(elapsed_ms)

        if device == "cuda":
            peak_memories.append(torch.cuda.max_memory_allocated() / 1e6)

    # Calculate statistics
    avg_time = sum(execution_times) / len(execution_times)
    min_time = min(execution_times)
    max_time = max(execution_times)

    # Sample standard deviation (N-1 denominator for unbiased estimate)
    n = len(execution_times)
    variance = sum((t - avg_time) ** 2 for t in execution_times) / max(n - 1, 1)
    std_time = variance ** 0.5

    return {
        "avg_execution_time_ms": avg_time,
        "min_execution_time_ms": min_time,
        "max_execution_time_ms": max_time,
        "std_execution_time_ms": std_time,
        "avg_peak_allocated_mb": sum(peak_memories) / len(peak_memories) if peak_memories else 0,
        "max_peak_allocated_mb": max(peak_memories) if peak_memories else 0,
        "num_warmup": num_warmup,
        "num_iterations": num_iterations,
        "device": device,
    }


@contextmanager
def timer(name: str = "Operation"):
    start = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(f"{name}: {elapsed_ms:.2f} ms")
