"""
Memory Profiler for Attention Implementations
==============================================

This module provides utilities for measuring and analyzing memory usage
of attention implementations. It includes both context managers for
interactive profiling and functions for systematic benchmarking.

Key Components
--------------
- MemoryStats: Data class holding profiling results
- MemoryProfiler: Context manager for memory measurement
- profile_attention_function: Benchmark function with warmup and statistics

Usage Examples
--------------
Context manager for quick profiling:

>>> from flashtile.utils import MemoryProfiler
>>> with MemoryProfiler(device="cuda") as profiler:
...     output = model(input_tensor)
>>> print(f"Peak memory: {profiler.stats.peak_allocated_mb:.2f} MB")
>>> print(f"Execution time: {profiler.stats.execution_time_ms:.2f} ms")

Function profiling with statistics:

>>> from flashtile.utils import profile_attention_function
>>> stats = profile_attention_function(
...     model,
...     x,  # input tensor
...     device="cuda",
...     num_warmup=5,
...     num_iterations=20,
... )
>>> print(f"Avg time: {stats['avg_execution_time_ms']:.2f} ms")
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch


@dataclass
class MemoryStats:
    """
    Container for memory profiling statistics.

    This data class holds the results from a memory profiling session,
    including both memory and timing information.

    Attributes
    ----------
    peak_allocated_mb : float
        Peak GPU memory allocated in megabytes. This is the maximum
        memory actually used by tensors (not including cached memory).

    peak_reserved_mb : float
        Peak GPU memory reserved by the allocator in megabytes.
        This includes cached memory that may not be actively used.

    execution_time_ms : float
        Wall-clock execution time in milliseconds.

    device : str
        The device where profiling was performed ("cuda" or "cpu").

    Examples
    --------
    >>> stats = MemoryStats(
    ...     peak_allocated_mb=512.5,
    ...     peak_reserved_mb=1024.0,
    ...     execution_time_ms=15.3,
    ...     device="cuda",
    ... )
    >>> print(f"Memory: {stats.peak_allocated_mb:.1f} MB, Time: {stats.execution_time_ms:.1f} ms")
    """

    peak_allocated_mb: float
    peak_reserved_mb: float
    execution_time_ms: float
    device: str


class MemoryProfiler:
    """
    Context manager for measuring GPU/CPU memory usage.

    This class provides a convenient way to measure memory consumption
    and execution time of code blocks. It automatically handles GPU
    synchronization for accurate timing.

    Parameters
    ----------
    device : str, optional
        Device to profile. If None, automatically detects CUDA availability.
        Valid values are "cuda" or "cpu". Default is None.

    Attributes
    ----------
    device : str
        The device being profiled.

    stats : MemoryStats
        Profiling results after the context exits. None before execution.

    Examples
    --------
    Basic usage:

    >>> model = FlashAttentionV2(embed_dim=512, num_heads=8).cuda()
    >>> x = torch.randn(2, 1024, 512).cuda()
    >>>
    >>> with MemoryProfiler() as profiler:
    ...     output, _ = model(x)
    >>>
    >>> print(f"Peak memory: {profiler.stats.peak_allocated_mb:.2f} MB")
    >>> print(f"Time: {profiler.stats.execution_time_ms:.2f} ms")

    Comparing implementations:

    >>> naive_profiler = MemoryProfiler()
    >>> flash_profiler = MemoryProfiler()
    >>>
    >>> with naive_profiler:
    ...     _ = naive_model(x)
    >>> with flash_profiler:
    ...     _ = flash_model(x)
    >>>
    >>> savings = naive_profiler.stats.peak_allocated_mb / flash_profiler.stats.peak_allocated_mb
    >>> print(f"Flash uses {savings:.1f}x less memory")

    Notes
    -----
    For GPU profiling, the profiler calls torch.cuda.synchronize() before
    and after the code block to ensure accurate timing. Memory statistics
    are reset before entering the context for isolated measurements.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        """
        Initialize the memory profiler.

        Parameters
        ----------
        device : str, optional
            Device to profile ("cuda" or "cpu"). If None, automatically
            uses CUDA if available, otherwise CPU.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = device

        self.stats: Optional[MemoryStats] = None
        self._start_time: float = 0.0

    def __enter__(self) -> "MemoryProfiler":
        """
        Enter the profiling context.

        Resets GPU memory statistics and records start time.
        """
        if self.device == "cuda":
            # Reset peak memory statistics for isolated measurement
            torch.cuda.reset_peak_memory_stats()
            # Synchronize to ensure all previous operations complete
            torch.cuda.synchronize()

        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the profiling context and collect statistics.

        Records execution time and peak memory usage.
        """
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
    """
    Profile a function over multiple iterations with warmup.

    This function provides systematic benchmarking with warmup iterations
    to ensure the GPU is in a steady state, followed by timed iterations
    to collect statistics.

    Parameters
    ----------
    fn : Callable
        The function to profile. Should accept *args and **kwargs.

    *args : tuple
        Positional arguments to pass to the function.

    device : str, optional
        Device to profile on ("cuda" or "cpu"). If None, automatically uses
        CUDA when available, otherwise CPU.

    num_warmup : int, optional
        Number of warmup iterations to run before timing. This ensures
        the GPU is in a steady state and JIT compilation is complete.
        Default is 5.

    num_iterations : int, optional
        Number of timed iterations for collecting statistics.
        Default is 20.

    **kwargs : dict
        Keyword arguments to pass to the function.

    Returns
    -------
    dict
        Dictionary containing profiling statistics:
        - avg_execution_time_ms: Mean execution time
        - min_execution_time_ms: Minimum execution time
        - max_execution_time_ms: Maximum execution time
        - std_execution_time_ms: Standard deviation of execution time
        - avg_peak_allocated_mb: Mean peak memory allocated
        - max_peak_allocated_mb: Maximum peak memory allocated
        - num_warmup: Number of warmup iterations performed
        - num_iterations: Number of timed iterations performed
        - device: Device used for profiling

    Examples
    --------
    >>> model = FlashAttentionV2(embed_dim=512, num_heads=8).cuda()
>>> x = torch.randn(2, 1024, 512).cuda()
>>>
>>> stats = profile_attention_function(
...     model,
...     x,
...     device="cuda",
...     num_warmup=3,
    ...     num_iterations=10,
    ... )
    >>>
    >>> print(f"Execution time: {stats['avg_execution_time_ms']:.2f} ± "
    ...       f"{stats['std_execution_time_ms']:.2f} ms")
    >>> print(f"Peak memory: {stats['avg_peak_allocated_mb']:.2f} MB")

    Notes
    -----
    The function runs in inference mode (torch.no_grad) to avoid
    storing intermediate tensors for gradient computation.
    """
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
    """
    Simple context manager for timing operations and printing results.

    This is a convenience function for quick timing during development
    and debugging.

    Parameters
    ----------
    name : str, optional
        Name to display in the timing output. Default is "Operation".

    Yields
    ------
    None

    Examples
    --------
    >>> with timer("Attention forward"):
    ...     output = model(input_tensor)
    Attention forward: 12.34 ms
    """
    start = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(f"{name}: {elapsed_ms:.2f} ms")
