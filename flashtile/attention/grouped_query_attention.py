"""
Grouped-Query Attention (GQA) and Multi-Query Attention (MQA)
=============================================================

This module implements attention variants that reduce KV-cache size during
inference by sharing key-value heads across multiple query heads.

In standard multi-head attention (MHA), each query head has its own key and
value projections, so the KV cache grows with `num_heads`. GQA reduces that to
`num_kv_heads`, where multiple query heads share the same KV head. MQA is the
special case where `num_kv_heads = 1`.

For example, with 32 query heads and 8 KV heads, the KV cache is reduced by
4x relative to MHA. This is useful for autoregressive inference, where KV-cache
size is often the bottleneck.

Implementation note
-------------------
This implementation uses standard PyTorch autograd for the backward pass.
Forward memory remains linear in sequence length, but backward is still
O(N²) during training because intermediate activations are retained by
autograd. A custom autograd path would be required for O(N) memory in both
forward and backward.

"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from flashtile.attention.base_attention import BaseAttention


class GroupedQueryAttention(BaseAttention):
    """
    Grouped-Query Attention with block-wise computation.

    GQA uses fewer key-value heads than query heads, reducing the KV cache
    size during inference while maintaining quality close to full MHA.

    .. note::
        This implementation uses standard PyTorch autograd. The forward pass
        uses block-wise computation, but the backward pass stores intermediate
        activations (O(N²) memory). For memory-efficient training, consider
        using FlashAttentionV1/V2 with standard attention, or implement a
        custom autograd Function for GQA.

    The architecture has:
    - H query heads (same as standard MHA)
    - G key-value heads (G < H, typically H/4 or H/8)
    - Each KV head is shared by H/G query heads

    Parameters
    ----------
    embed_dim : int
        Total embedding dimension. Must be divisible by num_heads.

    num_heads : int
        Number of query heads. This determines the query parallelism.

    num_kv_heads : int, optional
        Number of key-value heads. Must divide num_heads evenly.
        Default is 1 (equivalent to multi-query attention).

    block_size : int, optional
        Block size for Flash Attention. Default is 64.

    dropout : float, optional
        Dropout probability. Default is 0.0.

    bias : bool, optional
        Use bias in projections. Default is True.

    causal : bool, optional
        Apply causal masking. Default is False.

    Attributes
    ----------
    num_kv_heads : int
        Number of key-value heads.

    num_groups : int
        Number of query heads per KV head group (num_heads / num_kv_heads).

    q_proj : nn.Linear
        Query projection (embed_dim -> embed_dim).

    k_proj : nn.Linear
        Key projection (embed_dim -> num_kv_heads * head_dim).

    v_proj : nn.Linear
        Value projection (embed_dim -> num_kv_heads * head_dim).

    Examples
    --------
    >>> # LLaMA 2 70B style: 64 query heads, 8 KV heads
    >>> gqa = GroupedQueryAttention(
    ...     embed_dim=8192,
    ...     num_heads=64,
    ...     num_kv_heads=8,
    ...     causal=True,
    ... )
    >>>
    >>> # Check KV cache savings
    >>> mem = gqa.get_memory_usage(batch_size=1, seq_len=4096)
    >>> print(f"KV savings: {mem['kv_savings_ratio']:.1f}x")  # 8x

    Notes
    -----
    During attention computation, the K and V tensors are expanded to match
    the number of query heads using repeat_interleave. This expansion happens
    in fast GPU memory and doesn't affect the KV cache size during inference.

    See Also
    --------
    MultiQueryAttention : Extreme case with single KV head
    FlashAttentionV2 : Standard attention without KV sharing
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int = 1,
        block_size: int = 64,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = False,
    ) -> None:
        """Initialize Grouped-Query Attention."""
        # Don't call BaseAttention.__init__ for qkv_proj (we use separate projections)
        nn.Module.__init__(self)

        # Validation
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads}). "
                f"This ensures each KV head group has equal query heads."
            )
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        # Store configuration
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.block_size = block_size
        self.dropout_p = dropout
        self.causal = causal

        # Number of query heads per KV head group
        self.num_groups = num_heads // num_kv_heads

        # Separate projections for Q, K, V
        # Q: full num_heads projections
        # K, V: reduced num_kv_heads projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        """
        Compute Grouped-Query Attention forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch_size, sequence_length, embed_dim).

        attn_mask : torch.Tensor, optional
            Not supported. Use causal=True for causal masking.

        Returns
        -------
        output : torch.Tensor
            Attention output (batch_size, sequence_length, embed_dim).

        attention_weights : None
            Always None (memory efficient).
        """
        if self.training:
            warnings.warn(
                "GQA backward pass uses standard autograd (O(N²) memory). "
                "For memory-efficient training, use FlashAttentionV1 or V2 instead.",
                stacklevel=2,
            )

        if attn_mask is not None:
            raise NotImplementedError(
                "Custom attention masks not supported. Use causal=True for causal masking."
            )

        batch_size, seq_len, _ = x.shape

        # Project Q, K, V separately with different output dimensions
        Q = self.q_proj(x)  # (batch, seq_len, embed_dim)
        K = self.k_proj(x)  # (batch, seq_len, num_kv_heads * head_dim)
        V = self.v_proj(x)

        # Reshape Q to (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)

        # Reshape K, V to (batch, num_kv_heads, seq_len, head_dim)
        K = K.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        K = K.transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        V = V.transpose(1, 2)

        # Expand K, V to match Q's num_heads by repeating
        # (batch, num_kv_heads, seq, head_dim) -> (batch, num_heads, seq, head_dim)
        K = K.repeat_interleave(self.num_groups, dim=1)
        V = V.repeat_interleave(self.num_groups, dim=1)

        # Apply Flash Attention computation
        output = self._flash_attention_forward(Q, K, V)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)
        output = self.dropout(output)

        return output, None

    def _flash_attention_forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        """
        Flash Attention computation with online softmax.

        Uses V2-style loop order (Q outer, K/V inner) with causal block skipping.
        """
        batch_size, num_heads, seq_len, head_dim = Q.shape
        input_dtype = Q.dtype

        # Upcast to float32 for numerical stability in online softmax
        Q, K, V = Q.float(), K.float(), V.float()

        O = torch.zeros_like(Q)
        L = torch.zeros(batch_size, num_heads, seq_len, device=Q.device, dtype=torch.float32)
        M = torch.full((batch_size, num_heads, seq_len), float("-inf"), device=Q.device, dtype=torch.float32)

        num_blocks = math.ceil(seq_len / self.block_size)

        for q_idx in range(num_blocks):
            q_start = q_idx * self.block_size
            q_end = min(q_start + self.block_size, seq_len)

            Q_block = Q[:, :, q_start:q_end, :]
            M_block = M[:, :, q_start:q_end].clone()
            L_block = L[:, :, q_start:q_end].clone()
            O_block = O[:, :, q_start:q_end, :].clone()

            kv_end_idx = q_idx + 1 if self.causal else num_blocks

            for kv_idx in range(kv_end_idx):
                kv_start = kv_idx * self.block_size
                kv_end = min(kv_start + self.block_size, seq_len)

                K_block = K[:, :, kv_start:kv_end, :]
                V_block = V[:, :, kv_start:kv_end, :]

                S_block = torch.matmul(Q_block, K_block.transpose(-2, -1)) * self.scale


                if self.causal and kv_idx == q_idx:
                    q_positions = torch.arange(q_start, q_end, device=Q.device)
                    kv_positions = torch.arange(kv_start, kv_end, device=Q.device)
                    mask = q_positions[:, None] < kv_positions[None, :]
                    S_block = S_block.masked_fill(mask, float("-inf"))

                # Online softmax update with fully-masked row handling
                S_block_max = S_block.max(dim=-1).values
                
                # Handle fully masked rows (where S_block_max is -inf)
                # In such cases, keep the old statistics unchanged to avoid NaN
                valid_mask = ~torch.isinf(S_block_max)
                S_block_max = torch.where(valid_mask, S_block_max, M_block)
                
                M_new = torch.maximum(M_block, S_block_max)
                exp_diff = torch.exp(M_block - M_new)
                exp_S = torch.exp(S_block - M_new.unsqueeze(-1))
                L_new = L_block * exp_diff + exp_S.sum(dim=-1)

                P_block = exp_S
                O_block = O_block * (L_block * exp_diff).unsqueeze(-1) / L_new.unsqueeze(-1).clamp(min=1e-8)
                O_block = O_block + torch.matmul(P_block, V_block) / L_new.unsqueeze(-1).clamp(min=1e-8)

                M_block = M_new
                L_block = L_new

            O[:, :, q_start:q_end, :] = O_block

        return O.to(input_dtype)

    def get_memory_usage(self, batch_size: int, seq_len: int) -> Dict[str, Any]:
        """
        Calculate memory usage and KV cache savings (forward pass only).

        .. note::
            These estimates apply to the forward pass. The backward pass
            uses standard PyTorch autograd and requires O(N²) additional
            memory for intermediate activations.

        Returns
        -------
        dict
            Memory breakdown including:
            - q_mb: Memory for Q tensor
            - kv_mb: Memory for K, V tensors (reduced)
            - kv_savings_ratio: Reduction factor vs full MHA
            - total_mb/total_gb: Total memory (forward only)
        """
        dtype_bytes = 4

        # Q uses full num_heads
        q_bytes = batch_size * seq_len * self.embed_dim * dtype_bytes

        # K, V use reduced num_kv_heads
        kv_bytes = 2 * batch_size * seq_len * self.num_kv_heads * self.head_dim * dtype_bytes
        # During forward computation, K/V are expanded to query-head count.
        expanded_kv_bytes = 2 * batch_size * seq_len * self.num_heads * self.head_dim * dtype_bytes

        # What full MHA would use
        full_kv_bytes = 2 * batch_size * seq_len * self.num_heads * self.head_dim * dtype_bytes

        output_bytes = batch_size * seq_len * self.embed_dim * dtype_bytes
        total_bytes = q_bytes + kv_bytes + output_bytes
        effective_forward_bytes = q_bytes + expanded_kv_bytes + output_bytes

        # Calculate comparison with full MHA
        full_mha_bytes = q_bytes + full_kv_bytes + output_bytes
        
        return {
            "q_mb": q_bytes / 1e6,
            "kv_mb": kv_bytes / 1e6,
            "expanded_kv_mb": expanded_kv_bytes / 1e6,
            "full_mha_kv_mb": full_kv_bytes / 1e6,
            "kv_savings_ratio": self.num_heads / self.num_kv_heads,
            "output_mb": output_bytes / 1e6,
            "total_mb": total_bytes / 1e6,
            "effective_forward_mb": effective_forward_bytes / 1e6,
            "total_gb": total_bytes / 1e9,
            "complexity": "O(N) forward / O(N²) backward",
            "num_kv_heads": self.num_kv_heads,
            "num_query_heads": self.num_heads,
            "comparison_with_naive": {
                "naive_mb": full_mha_bytes / 1e6,
                "gqa_mb": total_bytes / 1e6,
                "reduction_ratio": full_mha_bytes / total_bytes if total_bytes > 0 else 0,
            },
            "note": (
                "KV savings reflect inference cache/projection memory. "
                "Forward attention compute expands KV to query-head count."
            ),
        }

    def extra_repr(self) -> str:
        """Return string representation."""
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, head_dim={self.head_dim}, "
            f"kv_savings={self.num_heads // self.num_kv_heads}x, "
            f"causal={self.causal}, block_size={self.block_size}"
        )


class MultiQueryAttention(GroupedQueryAttention):
    """
    Multi-Query Attention - extreme KV cache savings.

    MQA uses a single shared key-value head for all query heads,
    providing maximum KV cache reduction (num_heads × savings).

    .. note::
        Inherits GQA's implementation. Forward pass is memory-efficient,
        but backward pass uses standard autograd (O(N²) memory).

    This is a convenience wrapper around GroupedQueryAttention with
    num_kv_heads=1.

    Parameters
    ----------
    embed_dim : int
        Total embedding dimension.

    num_heads : int
        Number of query heads.

    block_size : int, optional
        Block size for Flash Attention. Default is 64.

    dropout : float, optional
        Dropout probability. Default is 0.0.

    bias : bool, optional
        Use bias in projections. Default is True.

    causal : bool, optional
        Apply causal masking. Default is False.

    Examples
    --------
    >>> mqa = MultiQueryAttention(embed_dim=512, num_heads=8)
    >>> mem = mqa.get_memory_usage(batch_size=1, seq_len=4096)
    >>> print(f"KV savings: {mem['kv_savings_ratio']:.0f}x")  # 8x

    See Also
    --------
    GroupedQueryAttention : General case with configurable KV heads
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        block_size: int = 64,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = False,
    ) -> None:
        """Initialize Multi-Query Attention with single KV head."""
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=1,  # MQA uses exactly 1 KV head
            block_size=block_size,
            dropout=dropout,
            bias=bias,
            causal=causal,
        )

