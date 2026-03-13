"""
Sliding Window Attention Implementation
=======================================

This module implements Sliding Window Attention (SWA), a local attention variant
that restricts each query to only attend to a fixed window of nearby keys.

Architecture
------------
Used in: Mistral 7B, Mixtral, Phi-3, Longformer (local attention component)

Complexity
----------
- Standard attention: O(N²) memory and compute
- Sliding window: O(N × W) memory and compute, where W = window size

When combined with Flash Attention tiling, we achieve:
- O(N × W / B²) IO complexity where B = block size
- Exact attention within windows (no approximation)

Key Insight
-----------
For many NLP tasks, most relevant context is local. A window size of 4096-8192
captures >90% of attention mass while enabling 128K+ effective context through
stacking layers (each layer "sees" an additional window of context).

References
----------
1. Beltagy et al. (2020). "Longformer: The Long-Document Transformer"
2. Jiang et al. (2023). "Mistral 7B"
3. Child et al. (2019). "Generating Long Sequences with Sparse Transformers"
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from flashtile.attention.base_attention import BaseAttention


class SlidingWindowAttention(BaseAttention):
    """
    Sliding Window Attention with Flash Attention memory efficiency.

    Each query position attends only to keys within a window of size
    `window_size` centered (or left-aligned for causal) around that position.

    Parameters
    ----------
    embed_dim : int
        Total embedding dimension. Must be divisible by num_heads.

    num_heads : int
        Number of attention heads.

    window_size : int, optional
        Size of the attention window. Default is 4096 (Mistral-style).
        Position i attends to positions max(0, i - window_size + 1) to i (causal)
        or i - window_size//2 to i + window_size//2 (bidirectional).

    block_size : int, optional
        Block size for tiled computation. Default is 64.

    dropout : float, optional
        Dropout probability. Default is 0.0.

    bias : bool, optional
        Use bias in projections. Default is True.

    causal : bool, optional
        Apply causal masking within window. Default is True (decoder-style).

    Examples
    --------
    >>> model = SlidingWindowAttention(
    ...     embed_dim=512,
    ...     num_heads=8,
    ...     window_size=4096,
    ...     causal=True
    ... ).cuda()
    >>> x = torch.randn(2, 32768, 512).cuda()  # 32K sequence
    >>> output, _ = model(x)
    >>> # Only O(N × W) compute, not O(N²)!
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int = 4096,
        block_size: int = 64,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = True,
    ) -> None:
        """Initialize Sliding Window Attention."""
        super().__init__(embed_dim, num_heads, dropout, bias, causal)

        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.window_size = window_size
        self.block_size = block_size

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        """
        Compute Sliding Window Attention forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch_size, sequence_length, embed_dim).

        attn_mask : torch.Tensor, optional
            Not supported. Use window_size and causal for masking.

        Returns
        -------
        output : torch.Tensor
            Attention output (batch_size, sequence_length, embed_dim).

        attention_weights : None
            Always None (memory efficient, no weights materialized).
        """
        if attn_mask is not None:
            raise NotImplementedError(
                "Custom attention masks not supported in Sliding Window Attention. "
                "Use window_size and causal parameters."
            )

        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q, K, V = self._project_qkv(x)

        # Compute sliding window attention with Flash-style tiling
        output = self._sliding_window_flash(Q, K, V, seq_len)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)
        output = self.dropout(output)

        return output, None

    def _sliding_window_flash(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Compute sliding window attention with Flash-style tiling.

        Key optimization: For each Q block, we only process K/V blocks
        that fall within the attention window, skipping distant blocks.
        """
        batch_size, num_heads, _, head_dim = Q.shape
        input_dtype = Q.dtype
        scale = self.scale

        # Upcast to float32 for numerical stability in online softmax
        Q, K, V = Q.float(), K.float(), V.float()

        # Output tensor
        O = torch.zeros_like(Q)

        # Running statistics for online softmax
        L = torch.zeros(
            batch_size, num_heads, seq_len,
            device=Q.device, dtype=torch.float32,
        )
        M = torch.full(
            (batch_size, num_heads, seq_len),
            float("-inf"), device=Q.device, dtype=torch.float32,
        )

        num_blocks = math.ceil(seq_len / self.block_size)

        # Process each Q block
        for q_block_idx in range(num_blocks):
            q_start = q_block_idx * self.block_size
            q_end = min(q_start + self.block_size, seq_len)

            Q_block = Q[:, :, q_start:q_end, :]

            # Initialize block-local accumulators
            M_block = M[:, :, q_start:q_end].clone()
            L_block = L[:, :, q_start:q_end].clone()
            O_block = O[:, :, q_start:q_end, :].clone()

            # Determine which K/V blocks are within the window
            # For causal: position i attends to [max(0, i - window + 1), i]
            # For bidirectional: position i attends to [i - window//2, i + window//2]
            if self.causal:
                # Last position in Q block determines the latest K we need
                window_end = q_end
                # First position in Q block minus window determines earliest K
                window_start = max(0, q_start - self.window_size + 1)
            else:
                # Bidirectional window: cover the full range needed by all
                # positions in the Q block [q_start, q_end)
                window_start = max(0, q_start - self.window_size // 2)
                window_end = min(seq_len, (q_end - 1) + self.window_size // 2 + 1)

            # Convert to block indices
            kv_start_block = window_start // self.block_size
            kv_end_block = min(math.ceil(window_end / self.block_size), num_blocks)

            # Only process K/V blocks within the window
            for kv_block_idx in range(kv_start_block, kv_end_block):
                kv_start = kv_block_idx * self.block_size
                kv_end = min(kv_start + self.block_size, seq_len)

                # Skip if this K/V block is entirely outside the window
                if self.causal and kv_start > q_end - 1:
                    continue

                K_block = K[:, :, kv_start:kv_end, :]
                V_block = V[:, :, kv_start:kv_end, :]

                # Compute attention scores
                S_block = torch.matmul(Q_block, K_block.transpose(-2, -1)) * scale

                # Apply window mask: mask positions outside the window
                q_positions = torch.arange(q_start, q_end, device=Q.device)
                kv_positions = torch.arange(kv_start, kv_end, device=Q.device)

                # Window mask: position i can only attend to positions in window
                if self.causal:
                    # Causal + window: attend to [max(0, i - window + 1), i]
                    # For each query position q, it can attend to keys in [q - window + 1, q]
                    window_starts = (q_positions - self.window_size + 1).clamp(min=0)
                    window_mask = (
                        (q_positions[:, None] < kv_positions[None, :]) |  # Causal (no future)
                        (kv_positions[None, :] < window_starts[:, None])   # Window boundary
                    )
                else:
                    # Bidirectional window: attend to [i - window//2, i + window//2]
                    half_window = self.window_size // 2
                    window_mask = (
                        (kv_positions[None, :] < q_positions[:, None] - half_window) |
                        (kv_positions[None, :] > q_positions[:, None] + half_window)
                    )

                S_block = S_block.masked_fill(window_mask, float("-inf"))

                # Online softmax update (same as Flash Attention V2)
                # with fully-masked row handling for numerical stability
                S_block_max = S_block.max(dim=-1).values
                
                # Handle fully masked rows (where S_block_max is -inf)
                # In such cases, keep the old statistics unchanged to avoid NaN
                valid_mask = ~torch.isinf(S_block_max)
                S_block_max = torch.where(valid_mask, S_block_max, M_block)
                
                M_new = torch.maximum(M_block, S_block_max)
                exp_diff = torch.exp(M_block - M_new)
                exp_S = torch.exp(S_block - M_new.unsqueeze(-1))

                # Guard against NaN from exp(-inf - (-inf)) when a row is
                # fully masked in this block AND no prior block had valid data.
                exp_diff = torch.nan_to_num(exp_diff, nan=0.0)
                exp_S = torch.nan_to_num(exp_S, nan=0.0)

                L_new = L_block * exp_diff + exp_S.sum(dim=-1)

                P_block = exp_S

                O_block = O_block * (L_block * exp_diff).unsqueeze(-1) / L_new.unsqueeze(-1).clamp(min=1e-8)
                O_block = O_block + torch.matmul(P_block, V_block) / L_new.unsqueeze(-1).clamp(min=1e-8)

                M_block = M_new
                L_block = L_new

            # Write back results
            O[:, :, q_start:q_end, :] = O_block
            M[:, :, q_start:q_end] = M_block
            L[:, :, q_start:q_end] = L_block

        return O.to(input_dtype)

    def get_memory_usage(self, batch_size: int, seq_len: int) -> Dict[str, Any]:
        """
        Calculate theoretical memory usage.

        Sliding window attention is O(N × W) instead of O(N²), providing
        significant savings when window_size << seq_len.
        """
        dtype_bytes = 4

        # Blocks within window per Q block
        blocks_per_window = math.ceil(self.window_size / self.block_size)

        block_attention_bytes = (
            batch_size * self.num_heads * self.block_size * self.block_size * dtype_bytes
        )
        qkv_bytes = 3 * batch_size * self.num_heads * seq_len * self.head_dim * dtype_bytes
        statistics_bytes = 2 * batch_size * self.num_heads * seq_len * dtype_bytes
        output_bytes = batch_size * seq_len * self.embed_dim * dtype_bytes

        total_bytes = block_attention_bytes + qkv_bytes + statistics_bytes + output_bytes

        # Approximate logical windowed attention matrix size (if materialized)
        effective_window = min(self.window_size, seq_len)
        window_bytes = (
            batch_size * self.num_heads * seq_len * effective_window * dtype_bytes
        )

        # Compare to full attention
        full_attention_bytes = batch_size * self.num_heads * seq_len * seq_len * dtype_bytes
        naive_total = full_attention_bytes + qkv_bytes + output_bytes

        return {
            "block_attention_mb": block_attention_bytes / 1e6,
            "window_mb": window_bytes / 1e6,
            "qkv_mb": qkv_bytes / 1e6,
            "statistics_mb": statistics_bytes / 1e6,
            "output_mb": output_bytes / 1e6,
            "total_mb": total_bytes / 1e6,
            "total_gb": total_bytes / 1e9,
            "complexity": f"O(N × {self.window_size})",
            "window_size": self.window_size,
            "block_size": self.block_size,
            "blocks_per_window": blocks_per_window,
            "vs_full_reduction": full_attention_bytes / total_bytes if total_bytes > 0 else 0,
            "effective_context": f"{seq_len} tokens with {self.window_size}-token windows",
            "comparison_with_naive": {
                "naive_mb": naive_total / 1e6,
                "sliding_window_mb": total_bytes / 1e6,
                "reduction_ratio": naive_total / total_bytes if total_bytes > 0 else 0,
            },
        }

    def extra_repr(self) -> str:
        """Return string representation."""
        return (
            f"{super().extra_repr()}, window_size={self.window_size}, "
            f"block_size={self.block_size}"
        )
