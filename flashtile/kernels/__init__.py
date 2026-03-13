"""
GPU Kernels for FlashTile
=========================

This subpackage provides optimized GPU kernels using Triton for
maximum performance Flash Attention computation.

Triton is an optional dependency. If not installed, the kernels
will not be available but other FlashTile functionality remains
fully operational.

Available Kernels (requires triton)
-----------------------------------
- TritonFlashAttention: Module wrapper for Triton kernel
- triton_flash_attention: Functional interface to the kernel

Installation
------------
To enable Triton kernels:
    pip install triton

Note: Triton requires NVIDIA GPU with compute capability >= 7.0 (Volta+)
"""

from __future__ import annotations

# Triton is an optional dependency
HAS_TRITON = False
TritonFlashAttention = None
triton_flash_attention = None

try:
    from flashtile.kernels.triton_flash_kernel import (
        TritonFlashAttention,
        triton_flash_attention,
    )
    HAS_TRITON = True
except ImportError:
    pass

__all__ = [
    "HAS_TRITON",
    "TritonFlashAttention",
    "triton_flash_attention",
]
