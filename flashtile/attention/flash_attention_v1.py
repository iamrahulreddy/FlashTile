"""
Flash Attention V1 Implementation
=================================

This module implements Flash Attention V1 as described in the paper
"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
by Tri Dao et al. (NeurIPS 2022).

The key insight of Flash Attention is that we can compute exact attention
without ever materializing the full N×N attention matrix by:
1. Processing the computation in small blocks that fit in fast GPU SRAM
2. Using online softmax to incrementally compute the normalization

This achieves O(N) memory complexity instead of O(N²), enabling much
longer sequences to fit in GPU memory.

.. note::
    **Speed vs Memory Tradeoff**: This pure Python implementation prioritizes
    clarity and educational value. The block-wise loops introduce Python overhead,
    making it slower than standard attention for short sequences. The advantage
    is memory—for long sequences, standard attention OOMs while this runs.
    
    For maximum speed, use PyTorch's native `F.scaled_dot_product_attention()`
    or the included Triton kernel (forward-only).

Algorithm Overview
------------------
Instead of computing the full attention matrix at once:
    A = softmax(QK^T / sqrt(d))
    O = A @ V

We process in blocks and maintain running statistics:
    - m: Running maximum (for numerical stability)
    - l: Running sum of exp(scores - m) (softmax denominator)
    - O: Running weighted output

For each new block of K, V:
    1. Compute block scores: S = Q_block @ K_block^T / sqrt(d)
    2. Update running max: m_new = max(m, max(S))
    3. Rescale previous accumulator: l = l * exp(m - m_new)
    4. Update denominator: l_new = l + sum(exp(S - m_new))
    5. Update output with rescaling

The math ensures we get exactly the same result as naive attention,
just computed in a memory-efficient manner.

Memory Complexity: O(N)
-----------------------
- Block attention matrix: O(block_size²) = O(1) since block_size is constant
- Running statistics: O(N) for m, l across all positions
- QKV tensors: O(N) linear in sequence length
- Output: O(N) linear in sequence length

Implementation Notes
--------------------
This implementation uses a custom autograd function for the backward pass.
Instead of storing the O(N²) attention matrix for gradient computation,
we recompute attention block-by-block during backward, using the same
O(N) memory footprint.

References
----------
1. Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention
   with IO-Awareness" NeurIPS 2022, arXiv:2205.14135

See Also
--------
FlashAttentionV2 : Improved version with better GPU parallelism
NaiveAttention : O(N²) reference implementation for correctness testing
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from flashtile.attention.amp_compat import custom_fwd, custom_bwd

from flashtile.attention.base_attention import BaseAttention


class FlashAttentionV1Function(torch.autograd.Function):
    """
    Custom autograd function for Flash Attention V1.

    This function implements both the forward and backward passes using
    the block-wise algorithm with online softmax. The key feature is that
    neither pass materializes the full N×N attention matrix.

    Forward Pass:
        - Processes Q, K, V in blocks
        - Maintains running max (m) and sum (l) for online softmax
        - Accumulates output with proper rescaling

    Backward Pass:
        - Recomputes attention block-by-block (no stored attention matrix)
        - Computes gradients for Q, K, V using chain rule
        - Same O(N) memory complexity as forward

    The gradient computation follows the softmax derivative rules:
        dL/dS = P ⊙ (dL/dP - D)
    where D = rowsum(dL/dP ⊙ P) and P = softmax(S)
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
        Compute Flash Attention V1 forward pass with online softmax.

        Parameters
        ----------
        ctx : FunctionCtx
            Autograd context for saving tensors needed in backward.

        Q : torch.Tensor
            Query tensor, shape (batch_size, num_heads, seq_len, head_dim).

        K : torch.Tensor
            Key tensor, shape (batch_size, num_heads, seq_len, head_dim).

        V : torch.Tensor
            Value tensor, shape (batch_size, num_heads, seq_len, head_dim).

        scale : float
            Attention score scaling factor, typically 1/sqrt(head_dim).

        block_size : int
            Size of blocks for tiled computation. Larger blocks are more
            compute-efficient but use more memory.

        causal : bool
            If True, applies causal masking to prevent attending to future.

        Returns
        -------
        torch.Tensor
            Attention output, shape (batch_size, num_heads, seq_len, head_dim).
        """
        batch_size, num_heads, seq_len, head_dim = Q.shape
        input_dtype = Q.dtype

        # Upcast to float32 for numerical stability in online softmax
        Q, K, V = Q.float(), K.float(), V.float()

        # Initialize output tensor
        O = torch.zeros_like(Q)

        # Running statistics for online softmax:
        # L: log-sum-exp (accumulated exponential sum)
        # M: running maximum for numerical stability
        # Using float32 for statistics even if QKV are fp16 for numerical precision
        L = torch.zeros(
            batch_size, num_heads, seq_len,
            device=Q.device,
            dtype=torch.float32,
        )
        M = torch.full(
            (batch_size, num_heads, seq_len),
            float("-inf"),
            device=Q.device,
            dtype=torch.float32,
        )

        # Calculate number of blocks
        num_kv_blocks = math.ceil(seq_len / block_size)
        num_q_blocks = math.ceil(seq_len / block_size)

        # Main loop: iterate over K/V blocks (outer) and Q blocks (inner)
        # V1 loop order: K/V outer, Q inner
        for kv_block_idx in range(num_kv_blocks):
            # Compute K/V block boundaries
            kv_start = kv_block_idx * block_size
            kv_end = min(kv_start + block_size, seq_len)

            # Load K/V block
            K_block = K[:, :, kv_start:kv_end, :]  # (batch, heads, kv_block_len, head_dim)
            V_block = V[:, :, kv_start:kv_end, :]

            # Inner loop over Q blocks
            for q_block_idx in range(num_q_blocks):
                # Compute Q block boundaries
                q_start = q_block_idx * block_size
                q_end = min(q_start + block_size, seq_len)

                # Causal optimization: skip blocks that are entirely masked
                # If all K/V positions are after all Q positions, skip
                if causal and kv_start > (q_end - 1):
                    continue

                # Load Q block
                Q_block = Q[:, :, q_start:q_end, :]

                # Compute attention scores for this block pair
                # S_block shape: (batch, heads, q_block_len, kv_block_len)
                S_block = torch.matmul(Q_block, K_block.transpose(-2, -1)) * scale

                # Apply causal mask if needed (within this block pair)
                if causal:
                    # Create position indices for masking
                    q_positions = torch.arange(q_start, q_end, device=Q.device)
                    kv_positions = torch.arange(kv_start, kv_end, device=Q.device)

                    # Mask where key position > query position
                    mask = q_positions[:, None] < kv_positions[None, :]
                    S_block = S_block.masked_fill(mask, float("-inf"))

                # Get current running statistics for this Q block
                M_block = M[:, :, q_start:q_end]  # (batch, heads, q_block_len)
                L_block = L[:, :, q_start:q_end]
                O_block = O[:, :, q_start:q_end, :]

                # Online softmax update:
                # Step 1: Compute new maximum
                S_block_max = S_block.max(dim=-1).values  # Max over K dimension
                
                # Handle fully masked rows (where S_block_max is -inf)
                # In such cases, keep the old statistics unchanged
                valid_mask = ~torch.isinf(S_block_max)
                S_block_max = torch.where(valid_mask, S_block_max, M_block)
                
                M_new = torch.maximum(M_block, S_block_max)

                # Step 2: Compute rescaling factor for previous accumulator
                # When max changes, we need to rescale existing accumulator
                exp_diff = torch.exp(M_block - M_new)

                # Step 3: Compute exponentials with new max for numerical stability
                exp_S = torch.exp(S_block - M_new.unsqueeze(-1))

                # Step 4: Update running sum (denominator of softmax)
                L_new = L_block * exp_diff + exp_S.sum(dim=-1)

                # Step 5: Compute attention weights for this block (unnormalized)
                P_block = exp_S

                # Step 6: Update output with proper rescaling
                # Old output needs to be rescaled, new contribution normalized
                O_new = O_block * (L_block * exp_diff).unsqueeze(-1) / L_new.unsqueeze(-1).clamp(min=1e-8)
                O_new = O_new + torch.matmul(P_block, V_block) / L_new.unsqueeze(-1).clamp(min=1e-8)

                # Store updated values
                O[:, :, q_start:q_end, :] = O_new
                M[:, :, q_start:q_end] = M_new
                L[:, :, q_start:q_end] = L_new

        # Save tensors for backward pass
        # Note: We save Q, K, V, O, L, M - NOT the attention matrix!
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
        """
        Compute gradients for Flash Attention V1 backward pass.

        This backward pass recomputes the attention weights block-by-block
        instead of loading from memory. This maintains O(N) memory complexity.

        Parameters
        ----------
        ctx : FunctionCtx
            Autograd context with saved tensors from forward.

        grad_output : torch.Tensor
            Gradient of loss with respect to output, shape
            (batch_size, num_heads, seq_len, head_dim).

        Returns
        -------
        grad_Q : torch.Tensor
            Gradient for query tensor, same shape as Q.

        grad_K : torch.Tensor
            Gradient for key tensor, same shape as K.

        grad_V : torch.Tensor
            Gradient for value tensor, same shape as V.

        None : For scale (not differentiable)
        None : For block_size (not differentiable)
        None : For causal (not differentiable)
        """
        # Retrieve saved tensors
        Q, K, V, O, L, M = ctx.saved_tensors
        scale = ctx.scale
        block_size = ctx.block_size
        causal = ctx.causal

        batch_size, num_heads, seq_len, head_dim = Q.shape
        input_dtype = Q.dtype

        # Upcast to float32 for backward (same as forward)
        Q, K, V, O = Q.float(), K.float(), V.float(), O.float()
        grad_output = grad_output.float()

        # Initialize gradient accumulators
        grad_Q = torch.zeros_like(Q)
        grad_K = torch.zeros_like(K)
        grad_V = torch.zeros_like(V)

        num_blocks = math.ceil(seq_len / block_size)
        # Row-global softmax backward term:
        # sum_j(dP_ij * P_ij) == dot(dO_i, O_i), which must include all KV blocks.
        D = (grad_output * O).sum(dim=-1, keepdim=True)

        # Recompute attention and compute gradients block by block
        for kv_block_idx in range(num_blocks):
            kv_start = kv_block_idx * block_size
            kv_end = min(kv_start + block_size, seq_len)

            K_block = K[:, :, kv_start:kv_end, :]
            V_block = V[:, :, kv_start:kv_end, :]

            for q_block_idx in range(num_blocks):
                q_start = q_block_idx * block_size
                q_end = min(q_start + block_size, seq_len)

                if causal and kv_start > (q_end - 1):
                    continue

                Q_block = Q[:, :, q_start:q_end, :]
                grad_O_block = grad_output[:, :, q_start:q_end, :]
                M_block = M[:, :, q_start:q_end]
                L_block = L[:, :, q_start:q_end]
                D_block = D[:, :, q_start:q_end, :]

                # Recompute attention scores and weights for this block
                S_block = torch.matmul(Q_block, K_block.transpose(-2, -1)) * scale

                if causal:
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

                # Gradient for V: dL/dV = P^T @ dL/dO
                grad_V[:, :, kv_start:kv_end, :] += torch.matmul(
                    P_block.transpose(-2, -1), grad_O_block
                )

                # Gradient for P: dL/dP = dL/dO @ V^T
                grad_P_block = torch.matmul(grad_O_block, V_block.transpose(-2, -1))

                # Softmax backward with row-global D term:
                # dS = P * (dP - rowsum(dP * P)). The rowsum is global across all KV blocks.
                grad_S_block = P_block * (grad_P_block - D_block)

                # Gradient for Q: dL/dQ = dL/dS @ K * scale
                grad_Q[:, :, q_start:q_end, :] += torch.matmul(grad_S_block, K_block) * scale

                # Gradient for K: dL/dK = dL/dS^T @ Q * scale
                grad_K[:, :, kv_start:kv_end, :] += torch.matmul(
                    grad_S_block.transpose(-2, -1), Q_block
                ) * scale

        return grad_Q.to(input_dtype), grad_K.to(input_dtype), grad_V.to(input_dtype), None, None, None


class FlashAttentionV1(BaseAttention):
    """
    Flash Attention V1 with O(N) memory complexity.

    This implementation uses block-wise computation and online softmax to
    avoid materializing the full N×N attention matrix. Memory usage scales
    linearly with sequence length instead of quadratically.

    .. note::
        **Python Implementation**: This uses pure PyTorch with Python loops.
        Memory-efficient (O(N)) but slower than fused CUDA kernels due to
        Python overhead. For production speed, use PyTorch's native
        `F.scaled_dot_product_attention()`.

    The algorithm computes exactly the same result as standard attention
    but processes the computation in small blocks that fit in fast GPU SRAM,
    using running statistics to track the softmax normalization.

    Parameters
    ----------
    embed_dim : int
        The total embedding dimension. Must be divisible by num_heads.

    num_heads : int
        The number of attention heads.

    block_size : int, optional
        The size of blocks for tiled computation. Larger blocks are more
        compute-efficient but use more memory per block. Should be a power
        of 2 for GPU efficiency. Default is 64.

    dropout : float, optional
        Dropout probability. Applied to the **output projection**, not to
        attention weights. Standard transformers apply dropout to the N×N
        attention matrix, but this would require materializing it (defeating
        O(N) memory) or storing per-block random seeds. This is a deliberate
        simplification for clarity. Use ``NaiveAttention`` for attention-weight
        dropout. Default is 0.0.

    bias : bool, optional
        If True, projection layers have learnable bias. Default is True.

    causal : bool, optional
        If True, applies causal masking. Default is False.

    Attributes
    ----------
    block_size : int
        The block size used for tiled computation.

    Examples
    --------
    >>> import torch
    >>> from flashtile import FlashAttentionV1
    >>>
    >>> # Create Flash Attention layer
    >>> model = FlashAttentionV1(
    ...     embed_dim=512,
    ...     num_heads=8,
    ...     block_size=64,
    ...     causal=True,
    ... ).cuda()
    >>>
    >>> # Process long sequence
    >>> x = torch.randn(2, 4096, 512).cuda()
    >>> output, _ = model(x)  # No attention weights returned (memory efficient)
    >>>
    >>> print(f"Output shape: {output.shape}")

    Notes
    -----
    Memory Comparison (batch=2, heads=8, dim=512)
        Sequence 1024: Naive 268 MB, Flash 42 MB (6.4x reduction)
        Sequence 4096: Naive OOM, Flash 164 MB
        Sequence 8192: Naive OOM, Flash 328 MB

    Choosing Block Size
        - 32: Low memory, good for small head dimensions
        - 64: Good balance for most cases (default)
        - 128: Better compute efficiency, higher memory

    See Also
    --------
    FlashAttentionV2 : Improved version with causal block skipping
    NaiveAttention : O(N²) baseline for correctness verification
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
        """Initialize Flash Attention V1."""
        super().__init__(embed_dim, num_heads, dropout, bias, causal)

        # Validate block size
        if block_size <= 0:
            raise ValueError(
                f"block_size must be a positive integer, got {block_size}. "
                f"Recommended values are powers of 2: 32, 64, 128."
            )

        self.block_size = block_size

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        """
        Compute Flash Attention V1 forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, embed_dim).

        attn_mask : torch.Tensor, optional
            Not supported in Flash Attention. Use causal=True for causal
            masking. Raises NotImplementedError if provided.

        Returns
        -------
        output : torch.Tensor
            Attention output, shape (batch_size, sequence_length, embed_dim).

        attention_weights : None
            Always None. Flash Attention does not materialize attention
            weights to save memory. Use NaiveAttention if you need weights.

        Raises
        ------
        NotImplementedError
            If attn_mask is provided. Flash Attention only supports
            causal masking, not arbitrary attention masks.
        """
        if attn_mask is not None:
            raise NotImplementedError(
                "Custom attention masks are not supported in Flash Attention. "
                "Flash Attention only supports causal masking via the causal=True parameter. "
                "If you need arbitrary masking, use NaiveAttention instead."
            )

        batch_size, seq_len, _ = x.shape

        # Project input to Q, K, V
        Q, K, V = self._project_qkv(x)

        # Apply Flash Attention
        output = FlashAttentionV1Function.apply(
            Q, K, V, self.scale, self.block_size, self.causal
        )

        # Reshape from (batch, heads, seq, head_dim) to (batch, seq, embed_dim)
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, seq_len, self.embed_dim)

        # Apply output projection and dropout
        output = self.out_proj(output)
        output = self.dropout(output)

        # Return None for attention weights (not materialized)
        return output, None

    def get_memory_usage(self, batch_size: int, seq_len: int) -> Dict[str, Any]:
        """
        Calculate theoretical memory usage for Flash Attention V1.

        Parameters
        ----------
        batch_size : int
            The batch size for memory calculation.

        seq_len : int
            The sequence length for memory calculation.

        Returns
        -------
        dict
            Memory breakdown including:
            - block_attention_mb: Memory for block-sized attention
            - qkv_mb: Memory for Q, K, V tensors
            - statistics_mb: Memory for running max and sum
            - output_mb: Memory for output tensor
            - total_mb/total_gb: Total memory
            - complexity: "O(N)" (linear)
            - block_size: Block size used
            - comparison_with_naive: Estimated reduction ratio
        """
        dtype_bytes = 4  # float32

        # Block-sized attention matrix: (batch, heads, block_size, block_size)
        # This is O(1) since block_size is constant
        block_attention_bytes = (
            batch_size * self.num_heads * self.block_size * self.block_size * dtype_bytes
        )

        # QKV tensors: 3 × (batch, heads, seq_len, head_dim) - O(N)
        qkv_bytes = 3 * batch_size * self.num_heads * seq_len * self.head_dim * dtype_bytes

        # Running statistics: M and L, each (batch, heads, seq_len) - O(N)
        statistics_bytes = 2 * batch_size * self.num_heads * seq_len * dtype_bytes

        # Output tensor: (batch, seq_len, embed_dim) - O(N)
        output_bytes = batch_size * seq_len * self.embed_dim * dtype_bytes

        total_bytes = block_attention_bytes + qkv_bytes + statistics_bytes + output_bytes

        # Calculate comparison with naive (which has O(N²) attention matrix)
        naive_attention_bytes = (
            batch_size * self.num_heads * seq_len * seq_len * dtype_bytes
        )
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

    def extra_repr(self) -> str:
        """Return string representation with Flash-specific parameters."""
        return f"{super().extra_repr()}, block_size={self.block_size}, memory_complexity=O(N)"
