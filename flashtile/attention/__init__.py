"""
Attention Module Implementations
================================

This subpackage contains all attention mechanism implementations in FlashTile.
Each implementation provides the same interface but with different memory and
compute characteristics.

Available Implementations
-------------------------
- NaiveAttention: Standard O(N²) scaled dot-product attention
- FlashAttentionV1: Memory-efficient O(N) attention with online softmax
- FlashAttentionV2: Optimized Flash Attention with causal block skipping
- SlidingWindowAttention: Local attention with O(N×W) complexity (Mistral-style)
- GroupedQueryAttention: Reduced KV cache with shared key-value heads
- MultiQueryAttention: Single shared KV head for maximum efficiency

All implementations inherit from BaseAttention which enforces a consistent
interface across the package.
"""

from __future__ import annotations

from flashtile.attention.base_attention import BaseAttention
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

__all__ = [
    "BaseAttention",
    "NaiveAttention",
    "FlashAttentionV1",
    "FlashAttentionV2",
    "SlidingWindowAttention",
    "GroupedQueryAttention",
    "MultiQueryAttention",
    "MaskedAttention",
    "create_padding_mask",
    "create_causal_mask",
]

