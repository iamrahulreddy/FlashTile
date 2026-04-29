from __future__ import annotations

__version__ = "0.2.2"
__license__ = "MIT"

# Core attention implementations
from flashtile.attention.naive_attention import NaiveAttention
from flashtile.attention.flash_attention_v1 import FlashAttentionV1
from flashtile.attention.flash_attention_v2 import FlashAttentionV2
from flashtile.attention.sliding_window_attention import SlidingWindowAttention
from flashtile.attention.grouped_query_attention import (
    GroupedQueryAttention,
    MultiQueryAttention,
)
from flashtile.attention.masked_attention import (
    MaskedAttention,
    create_padding_mask,
    create_causal_mask,
)

# Triton kernel (optional - requires triton)
try:
    from flashtile.kernels.triton_flash_kernel import TritonFlashAttention
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    TritonFlashAttention = None  # type: ignore

# Utility exports
from flashtile.utils.memory_profiler import MemoryProfiler, profile_attention_function

# Package-level exports (what appears in `from flashtile import *`)
__all__ = [
    # Version info
    "__version__",
    # Core implementations
    "NaiveAttention",
    "FlashAttentionV1",
    "FlashAttentionV2",
    "SlidingWindowAttention",
    "GroupedQueryAttention",
    "MultiQueryAttention",
    # Masked attention fallback
    "MaskedAttention",
    "create_padding_mask",
    "create_causal_mask",
    # Factory function
    "get_attention",
    # Utilities
    "MemoryProfiler",
    "profile_attention_function",
    # Optional (only if Triton available)
    "TritonFlashAttention",
    "HAS_TRITON",
]

# Registry mapping attention type names to their implementing classes
_ATTENTION_REGISTRY = {
    "naive": NaiveAttention,
    "flash_v1": FlashAttentionV1,
    "flash_v2": FlashAttentionV2,
    "sliding_window": SlidingWindowAttention,
    "gqa": GroupedQueryAttention,
    "grouped_query": GroupedQueryAttention,
    "mqa": MultiQueryAttention,
    "multi_query": MultiQueryAttention,
}

# Add Triton to registry if available
if HAS_TRITON and TritonFlashAttention is not None:
    _ATTENTION_REGISTRY["triton"] = TritonFlashAttention
    _ATTENTION_REGISTRY["triton_flash"] = TritonFlashAttention

# Add MaskedAttention to registry
_ATTENTION_REGISTRY["masked"] = MaskedAttention


def get_attention(
    attention_type: str,
    embed_dim: int,
    num_heads: int,
    **kwargs,
):

    # Normalize the attention type string for flexible matching
    normalized_type = attention_type.lower().replace("-", "_").replace(" ", "_")

    if normalized_type not in _ATTENTION_REGISTRY:
        available_types = list(_ATTENTION_REGISTRY.keys())
        raise ValueError(
            f"Unknown attention type: '{attention_type}'. "
            f"Available types are: {available_types}"
        )

    attention_class = _ATTENTION_REGISTRY[normalized_type]
    return attention_class(embed_dim=embed_dim, num_heads=num_heads, **kwargs)

def check_installation() -> dict:
    import torch

    info = {
        "version": __version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "triton_available": False,
        "triton_version": None,
    }

    try:
        import triton

        info["triton_available"] = True
        info["triton_version"] = triton.__version__
    except ImportError:
        pass

    return info
