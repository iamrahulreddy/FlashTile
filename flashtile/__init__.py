"""
FlashTile: Memory-Efficient Flash Attention Implementation
===========================================================

An educational implementation of Flash Attention algorithms achieving O(N)
memory complexity through block-wise computation and online softmax. This package
provides multiple attention implementations optimized for different use cases.

Core Implementations
--------------------
- NaiveAttention: Standard O(N²) attention for correctness reference
- FlashAttentionV1: Memory-efficient O(N) attention with online softmax
- FlashAttentionV2: Optimized Flash Attention with causal block skipping
- SlidingWindowAttention: Local attention with O(N×W) complexity (Mistral-style)
- GroupedQueryAttention: Reduced KV cache memory (LLaMA 2-style)
- MultiQueryAttention: Maximum KV cache savings (single KV head)

Example Usage
-------------
>>> import torch
>>> from flashtile import FlashAttentionV2, get_attention
>>>
>>> # Direct instantiation
>>> model = FlashAttentionV2(
...     embed_dim=512,
...     num_heads=8,
...     block_size=64,
...     causal=True,
... )
>>> model = model.cuda()
>>>
>>> # Forward pass with long sequences
>>> x = torch.randn(2, 4096, 512, device="cuda")
>>> output, attention_weights = model(x)
>>> # attention_weights is None for Flash implementations (memory efficient)
>>>
>>> # Factory function for flexible instantiation
>>> model = get_attention(
...     attention_type="flash_v2",
...     embed_dim=512,
...     num_heads=8,
...     causal=True,
... )

Performance Characteristics
---------------------------
At sequence length 4096 with batch_size=2, embed_dim=512, num_heads=8:
- NaiveAttention: ~11 GB memory (O(N²) attention matrix)
- FlashAttentionV2: ~1.6 GB memory (O(N) block-wise computation)
- Memory reduction: ~7x

References
----------
1. Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention"
   NeurIPS 2022, arXiv:2205.14135
2. Dao. "FlashAttention-2: Faster Attention with Better Parallelism"
   ICLR 2024, arXiv:2307.08691
3. Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models"
   EMNLP 2023, arXiv:2305.13245

License
-------
MIT License - see LICENSE file for details.
"""

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
    """
    Factory function to create attention module instances.

    This function provides a convenient way to instantiate different attention
    implementations using a string identifier. It supports all attention types
    available in the FlashTile package.

    Parameters
    ----------
    attention_type : str
        The type of attention to instantiate. Valid options are:
        - "naive": Standard O(N²) attention (NaiveAttention)
        - "flash_v1": Flash Attention V1 (FlashAttentionV1)
        - "flash_v2": Flash Attention V2 with optimizations (FlashAttentionV2)
        - "sliding_window": Sliding Window Attention (SlidingWindowAttention)
        - "gqa" or "grouped_query": Grouped-Query Attention (GroupedQueryAttention)
        - "mqa" or "multi_query": Multi-Query Attention (MultiQueryAttention)
        - "masked": Masked Attention fallback (MaskedAttention)
        - "triton" or "triton_flash": Triton kernel (requires triton)

    embed_dim : int
        The embedding dimension (model hidden size). Must be divisible by num_heads.

    num_heads : int
        The number of attention heads for queries. For GQA/MQA, this is the number
        of query heads, not the number of key-value heads.

    **kwargs : dict
        Additional keyword arguments passed to the attention class constructor.
        Common options include:
        - block_size (int): Block size for tiling (Flash implementations, default: 64)
        - causal (bool): Whether to apply causal masking (default: False)
        - dropout (float): Dropout probability (default: 0.0)
        - bias (bool): Whether to use bias in projections (default: True)
        - num_kv_heads (int): Number of KV heads for GQA (default: 1 for MQA)

    Returns
    -------
    nn.Module
        An instance of the requested attention class.

    Raises
    ------
    ValueError
        If attention_type is not recognized.
    ValueError
        If embed_dim is not divisible by num_heads.
    ValueError
        If any of the construction parameters are invalid.

    Examples
    --------
    >>> # Create Flash Attention V2 with causal masking
    >>> model = get_attention(
    ...     attention_type="flash_v2",
    ...     embed_dim=512,
    ...     num_heads=8,
    ...     causal=True,
    ... )

    >>> # Create GQA with 8 query heads and 2 KV heads
    >>> model = get_attention(
    ...     attention_type="gqa",
    ...     embed_dim=512,
    ...     num_heads=8,
    ...     num_kv_heads=2,
    ... )

    >>> # Case-insensitive and flexible naming
    >>> model1 = get_attention("Flash_V2", embed_dim=512, num_heads=8)
    >>> model2 = get_attention("FLASH-V2", embed_dim=512, num_heads=8)
    >>> # Both create FlashAttentionV2 instances
    """
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
    """
    Check the installation status of FlashTile and its dependencies.

    This utility function verifies that all required and optional dependencies
    are installed and returns information about the installation state.

    Returns
    -------
    dict
        A dictionary containing:
        - version: FlashTile version string
        - torch_version: PyTorch version string
        - cuda_available: Whether CUDA is available
        - cuda_version: CUDA version if available, None otherwise
        - triton_available: Whether Triton is installed
        - triton_version: Triton version if available, None otherwise

    Examples
    --------
    >>> info = check_installation()
    >>> print(f"FlashTile {info['version']} with PyTorch {info['torch_version']}")
    >>> if info['cuda_available']:
    ...     print(f"CUDA {info['cuda_version']} available")
    """
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
