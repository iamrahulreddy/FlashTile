"""
Flash Attention V2 Implementation
=================================

This module implements Flash Attention V2 as described in the paper
"FlashAttention-2: Faster Attention with Better Parallelism and Work
Partitioning" by Tri Dao (ICLR 2024).

V2 Improvements Over V1
-----------------------
1. Swapped Loop Order: Outer loop iterates over Q blocks instead of K/V blocks.
   This enables better GPU parallelism since each Q block can be processed
   independently once K/V are loaded.

2. Causal Block Skipping: For causal attention, entire K/V blocks that would
   be completely masked are skipped. This saves approximately 50% of compute
   for decoder-style autoregressive models.

3. Better Work Partitioning: The algorithm is designed to better utilize
   GPU thread blocks and warp-level parallelism.

.. note::
    **Implementation**: Pure Python with PyTorch. Memory-efficient (O(N))
    but has Python loop overhead. For production speed, use PyTorch's native
    `F.scaled_dot_product_attention()` or the Triton kernel (forward-only).

Causal Block Skipping Explained
-------------------------------
In causal attention, position i can only attend to positions 0..i.
This means the attention matrix is lower triangular.

For blocks, if we're computing Q block [q_start, q_end) with K/V block
[kv_start, kv_end), the block is partially or fully masked when:
- Fully masked (skip): kv_start > q_end - 1
  (All K/V positions are after all Q positions)
- Partially masked: kv_start <= q_end - 1 and kv_end > q_start
  (Block overlaps the diagonal, apply per-element masking)

For a 4096-length sequence with block_size=64:
- Without causal: 64 × 64 = 4096 block pairs computed
- With causal: ~2048 block pairs (lower triangle)
- Savings: approximately 50%

References
----------
1. Dao. "FlashAttention-2: Faster Attention with Better Parallelism and
   Work Partitioning" ICLR 2024, arXiv:2307.08691

See Also
--------
FlashAttentionV1 : Original Flash Attention implementation
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch

from flashtile.attention.amp_compat import custom_fwd, custom_bwd

from flashtile.attention.base_attention import BaseAttention


class FlashAttentionV2Function(torch.autograd.Function):
    """
    Custom autograd function for Flash Attention V2.

    Key difference from V1: Outer loop is over Q blocks, enabling
    better GPU parallelism. Causal block skipping is implemented
    at the inner loop level.
    """

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        scale: float,
        block_size: int,
        causal: bool,
    ) -> torch.Tensor:
        """
        Compute Flash Attention V2 forward pass.

        V2 loop order: Q blocks (outer), K/V blocks (inner).
        This allows each Q block to be processed independently.
        """
        batch_size, num_heads, seq_len, head_dim = Q.shape
        input_dtype = Q.dtype

        # Upcast to float32 for numerical stability in online softmax
        # This ensures exp/matmul operations don't hit dtype mismatches
        # when inputs are fp16/bf16. Output is cast back at the end.
        Q, K, V = Q.float(), K.float(), V.float()

        O = torch.zeros_like(Q)
        L = torch.zeros(
            batch_size, num_heads, seq_len,
            device=Q.device, dtype=torch.float32,
        )
        M = torch.full(
            (batch_size, num_heads, seq_len),
            float("-inf"), device=Q.device, dtype=torch.float32,
        )

        num_q_blocks = math.ceil(seq_len / block_size)
        num_kv_blocks = math.ceil(seq_len / block_size)

        # V2: Outer loop over Q blocks (parallelizable across GPU SMs)
        for q_block_idx in range(num_q_blocks):
            q_start = q_block_idx * block_size
            q_end = min(q_start + block_size, seq_len)

            Q_block = Q[:, :, q_start:q_end, :]

            # Initialize block-local accumulators
            M_block = M[:, :, q_start:q_end].clone()
            L_block = L[:, :, q_start:q_end].clone()
            O_block = O[:, :, q_start:q_end, :].clone()

            # V2 Causal Optimization: Determine the last K/V block to process
            # For causal attention, Q at position i can only attend to K at j <= i
            # So for Q block ending at q_end-1, we only need K blocks up to q_end-1
            if causal:
                # Only process K/V blocks [0, q_block_idx] (inclusive)
                kv_end_block = q_block_idx + 1
            else:
                kv_end_block = num_kv_blocks

            # Inner loop over K/V blocks (all or up to current Q block for causal)
            for kv_block_idx in range(kv_end_block):
                kv_start = kv_block_idx * block_size
                kv_end = min(kv_start + block_size, seq_len)

                K_block = K[:, :, kv_start:kv_end, :]
                V_block = V[:, :, kv_start:kv_end, :]

                # Compute attention scores: S = Q @ K^T / sqrt(d)
                # Shape: [batch, heads, block_size, kv_block_size]
                S_block = torch.matmul(Q_block, K_block.transpose(-2, -1)) * scale

                # Apply causal mask for the diagonal block
                # Only needed when Q and K/V blocks overlap on the diagonal
                if causal and kv_block_idx == q_block_idx:
                    q_positions = torch.arange(q_start, q_end, device=Q.device)
                    kv_positions = torch.arange(kv_start, kv_end, device=Q.device)
                    mask = q_positions[:, None] < kv_positions[None, :]
                    S_block = S_block.masked_fill(mask, float("-inf"))

                # ═══════════════════════════════════════════════════════════════
                # ONLINE SOFTMAX UPDATE - The heart of Flash Attention
                # ═══════════════════════════════════════════════════════════════
                # We maintain running statistics to compute exact softmax without
                # storing the full N×N attention matrix. The key identity is:
                #
                #   exp(x - m_old) × exp(m_old - m_new) = exp(x - m_new)
                #
                # This allows us to rescale previous accumulations when we
                # discover a new maximum, keeping all terms relative to m_new.
                # ═══════════════════════════════════════════════════════════════

                # Step 1: Find maximum of current block (for numerical stability)
                S_block_max = S_block.max(dim=-1).values
                
                # Handle fully masked rows (where S_block_max is -inf)
                valid_mask = ~torch.isinf(S_block_max)
                S_block_max = torch.where(valid_mask, S_block_max, M_block)

                # Step 2: Update running maximum
                # M_new is the global max seen so far (across all processed blocks)
                M_new = torch.maximum(M_block, S_block_max)

                # Step 3: Compute rescaling factor α = exp(m_old - m_new)
                # This is ≤ 1 because M_new ≥ M_block
                # It rescales all previous contributions to use M_new as baseline
                exp_diff = torch.exp(M_block - M_new)

                # Step 4: Compute unnormalized attention weights for current block
                # exp(S - M_new) ensures all values are in (0, 1] range
                exp_S = torch.exp(S_block - M_new.unsqueeze(-1))

                # Step 5: Update running sum (denominator of softmax)
                # L_new = α × L_old + Σ exp(S_block - M_new)
                # Previous sum is rescaled, then we add new block's contribution
                L_new = L_block * exp_diff + exp_S.sum(dim=-1)

                # Attention probabilities for this block (unnormalized)
                P_block = exp_S

                # Step 6: Update running output
                # The output update is: O_new = (α × L_old × O_old + P @ V) / L_new
                # Split into two operations for numerical stability:
                #   a) Rescale old output: O_old × (α × L_old / L_new)
                #   b) Add new contribution: (P @ V) / L_new
                # clamp(min=1e-8) prevents division by zero for fully-masked rows
                O_block = O_block * (L_block * exp_diff).unsqueeze(-1) / L_new.unsqueeze(-1).clamp(min=1e-8)
                O_block = O_block + torch.matmul(P_block, V_block) / L_new.unsqueeze(-1).clamp(min=1e-8)

                # Update running statistics for next block iteration
                M_block = M_new
                L_block = L_new

            # Write back accumulated results
            O[:, :, q_start:q_end, :] = O_block
            M[:, :, q_start:q_end] = M_block
            L[:, :, q_start:q_end] = L_block

        ctx.save_for_backward(Q, K, V, O, L, M)
        ctx.scale = scale
        ctx.block_size = block_size
        ctx.causal = causal

        return O.to(input_dtype)

    @staticmethod
    @custom_bwd
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        """Compute gradients with same block structure as forward."""
        Q, K, V, O, L, M = ctx.saved_tensors
        scale = ctx.scale
        block_size = ctx.block_size
        causal = ctx.causal

        batch_size, num_heads, seq_len, head_dim = Q.shape
        input_dtype = Q.dtype

        # Upcast to float32 for backward (same as forward)
        Q, K, V, O = Q.float(), K.float(), V.float(), O.float()
        grad_output = grad_output.float()

        grad_Q = torch.zeros_like(Q)
        grad_K = torch.zeros_like(K)
        grad_V = torch.zeros_like(V)

        num_blocks = math.ceil(seq_len / block_size)
        # Row-global softmax backward term (must include contributions from all KV blocks).
        D = (grad_output * O).sum(dim=-1, keepdim=True)

        for q_block_idx in range(num_blocks):
            q_start = q_block_idx * block_size
            q_end = min(q_start + block_size, seq_len)

            Q_block = Q[:, :, q_start:q_end, :]
            grad_O_block = grad_output[:, :, q_start:q_end, :]
            M_block = M[:, :, q_start:q_end]
            L_block = L[:, :, q_start:q_end]
            D_block = D[:, :, q_start:q_end, :]

            kv_end_block = q_block_idx + 1 if causal else num_blocks

            for kv_block_idx in range(kv_end_block):
                kv_start = kv_block_idx * block_size
                kv_end = min(kv_start + block_size, seq_len)

                # Skip future blocks entirely for causal attention (optimization)
                if causal and kv_start > q_end - 1:
                    continue

                K_block = K[:, :, kv_start:kv_end, :]
                V_block = V[:, :, kv_start:kv_end, :]

                # Recompute attention
                S_block = torch.matmul(Q_block, K_block.transpose(-2, -1)) * scale

                if causal and kv_block_idx == q_block_idx:
                    q_positions = torch.arange(q_start, q_end, device=Q.device)
                    kv_positions = torch.arange(kv_start, kv_end, device=Q.device)
                    mask = q_positions[:, None] < kv_positions[None, :]
                    S_block = S_block.masked_fill(mask, float("-inf"))

                # Recompute attention probabilities.
                # Fully-masked rows keep M=-inf and L=0 from forward; force P=0 there.
                row_valid = torch.isfinite(M_block) & (L_block > 0)
                safe_logits = torch.where(
                    row_valid.unsqueeze(-1),
                    S_block - M_block.unsqueeze(-1),
                    torch.full_like(S_block, float("-inf")),
                )
                P_block = torch.exp(safe_logits) / L_block.unsqueeze(-1).clamp(min=1e-8)
                P_block = torch.where(row_valid.unsqueeze(-1), P_block, torch.zeros_like(P_block))

                grad_V[:, :, kv_start:kv_end, :] += torch.matmul(
                    P_block.transpose(-2, -1), grad_O_block
                )

                grad_P_block = torch.matmul(grad_O_block, V_block.transpose(-2, -1))
                # Softmax backward with row-global D term.
                grad_S_block = P_block * (grad_P_block - D_block)

                grad_Q[:, :, q_start:q_end, :] += torch.matmul(grad_S_block, K_block) * scale
                grad_K[:, :, kv_start:kv_end, :] += torch.matmul(
                    grad_S_block.transpose(-2, -1), Q_block
                ) * scale

        return grad_Q.to(input_dtype), grad_K.to(input_dtype), grad_V.to(input_dtype), None, None, None


class FlashAttentionV2(BaseAttention):
    """
    Flash Attention V2 with optimized loop order and causal block skipping.

    This implementation improves upon V1 with:
    1. Swapped loop order for better GPU parallelism
    2. Causal block skipping for ~50% compute savings on decoder models
    3. More efficient memory access patterns

    .. note::
        **Python Implementation**: Memory-efficient (O(N) forward and backward)
        but slower than fused CUDA kernels due to Python loop overhead.
        For production speed, use PyTorch's native `F.scaled_dot_product_attention()`.

    .. note::
        **Dropout behavior**: Dropout is applied to the **output projection**, not
        to the attention weights. Standard transformers apply dropout to the N×N
        attention matrix, but this would require materializing it (defeating O(N)
        memory) or storing per-block random seeds for backward (as in the original
        Flash Attention paper). This simplification keeps the implementation clear
        while preserving the core tiling/online-softmax logic. For attention-weight
        dropout, use ``NaiveAttention``.

    Parameters
    ----------
    embed_dim : int
        Total embedding dimension. Must be divisible by num_heads.

    num_heads : int
        Number of attention heads.

    block_size : int, optional
        Block size for tiled computation. Default is 64.

    dropout : float, optional
        Dropout probability. Default is 0.0.

    bias : bool, optional
        Use bias in projections. Default is True.

    causal : bool, optional
        Apply causal masking. Default is False.

    Examples
    --------
    >>> model = FlashAttentionV2(embed_dim=512, num_heads=8, causal=True).cuda()
    >>> x = torch.randn(2, 8192, 512).cuda()
    >>> output, _ = model(x)
    >>>
    >>> # Check causal efficiency
    >>> efficiency = model.get_causal_efficiency(seq_len=8192)
    >>> print(f"Compute savings: {efficiency['compute_savings_percent']:.1f}%")
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
        """Initialize Flash Attention V2."""
        super().__init__(embed_dim, num_heads, dropout, bias, causal)

        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.block_size = block_size

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        """
        Compute Flash Attention V2 forward pass.

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
            Always None (memory efficient, no weights materialized).
        """
        if attn_mask is not None:
            raise NotImplementedError(
                "Custom attention masks not supported in Flash Attention. "
                "Use causal=True for causal masking, or NaiveAttention for arbitrary masks."
            )

        batch_size, seq_len, _ = x.shape

        Q, K, V = self._project_qkv(x)

        output = FlashAttentionV2Function.apply(
            Q, K, V, self.scale, self.block_size, self.causal
        )

        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)
        output = self.dropout(output)

        return output, None

    def get_memory_usage(self, batch_size: int, seq_len: int) -> Dict[str, Any]:
        """Calculate theoretical memory usage."""
        dtype_bytes = 4

        block_attention_bytes = (
            batch_size * self.num_heads * self.block_size * self.block_size * dtype_bytes
        )
        qkv_bytes = 3 * batch_size * self.num_heads * seq_len * self.head_dim * dtype_bytes
        statistics_bytes = 2 * batch_size * self.num_heads * seq_len * dtype_bytes
        output_bytes = batch_size * seq_len * self.embed_dim * dtype_bytes

        total_bytes = block_attention_bytes + qkv_bytes + statistics_bytes + output_bytes

        naive_attention_bytes = batch_size * self.num_heads * seq_len * seq_len * dtype_bytes
        naive_total = naive_attention_bytes + qkv_bytes + output_bytes

        return {
            "block_attention_mb": block_attention_bytes / 1e6,
            "qkv_mb": qkv_bytes / 1e6,
            "statistics_mb": statistics_bytes / 1e6,
            "output_mb": output_bytes / 1e6,
            "total_mb": total_bytes / 1e6,
            "total_gb": total_bytes / 1e9,
            "complexity": "O(N)",
            "block_size": self.block_size,
            "comparison_with_naive": {
                "naive_mb": naive_total / 1e6,
                "flash_mb": total_bytes / 1e6,
                "reduction_ratio": naive_total / total_bytes if total_bytes > 0 else 0,
            },
        }

    def get_causal_efficiency(self, seq_len: int) -> Dict[str, Any]:
        """
        Calculate compute savings from causal block skipping.

        For causal attention, approximately half the blocks are skipped
        because they would be entirely masked.

        Parameters
        ----------
        seq_len : int
            Sequence length to analyze.

        Returns
        -------
        dict
            Dictionary with:
            - total_blocks: Total possible block pairs
            - computed_blocks: Blocks actually computed
            - skipped_blocks: Blocks skipped due to causal masking
            - compute_savings_percent: Percentage of compute saved
        """
        num_blocks = math.ceil(seq_len / self.block_size)

        total_blocks = num_blocks * num_blocks  # Full attention matrix in blocks
        computed_blocks = num_blocks * (num_blocks + 1) // 2  # Lower triangular
        skipped_blocks = total_blocks - computed_blocks

        return {
            "total_blocks": total_blocks,
            "computed_blocks": computed_blocks,
            "skipped_blocks": skipped_blocks,
            "compute_savings_percent": 100.0 * skipped_blocks / total_blocks if total_blocks > 0 else 0,
            "sequence_length": seq_len,
            "block_size": self.block_size,
        }

    def extra_repr(self) -> str:
        """Return string representation."""
        return f"{super().extra_repr()}, block_size={self.block_size}, memory_complexity=O(N)"
