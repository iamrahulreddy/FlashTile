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
