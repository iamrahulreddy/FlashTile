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

