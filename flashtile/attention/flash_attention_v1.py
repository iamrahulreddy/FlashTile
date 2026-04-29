from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from flashtile.attention.amp_compat import custom_fwd, custom_bwd

from flashtile.attention.base_attention import BaseAttention


class FlashAttentionV1Function(torch.autograd.Function):
    # Custom autograd function for Flash Attention V1.
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
