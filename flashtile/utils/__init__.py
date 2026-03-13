"""
FlashTile Utility Modules
=========================

This subpackage provides utility functions and classes for profiling,
benchmarking, visualizing, and analyzing attention implementations.

Available Utilities
-------------------
- MemoryProfiler: Context manager for memory usage measurement
- profile_attention_function: Benchmark a function over multiple iterations
- AttentionVisualizer: Visualization toolkit for attention patterns
- KernelProfiler: CUDA kernel profiling with TFLOPS/bandwidth metrics
"""

from __future__ import annotations

from flashtile.utils.memory_profiler import (
    MemoryProfiler,
    MemoryStats,
    profile_attention_function,
)

# Optional imports that require matplotlib
try:
    from flashtile.utils.attention_visualizer import AttentionVisualizer
    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False

from flashtile.utils.kernel_profiler import (
    KernelProfiler,
    ProfileResult,
    CUDATimer,
    profile_attention_implementations,
)

__all__ = [
    "MemoryProfiler",
    "MemoryStats",
    "profile_attention_function",
    "KernelProfiler",
    "ProfileResult",
    "CUDATimer",
    "profile_attention_implementations",
]

if HAS_VISUALIZER:
    __all__.append("AttentionVisualizer")

