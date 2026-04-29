from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch

from flashtile.attention.amp_compat import custom_fwd, custom_bwd
from flashtile.attention.base_attention import BaseAttention


class FlashAttentionV2Function(torch.autograd.Function):
    """Custom autograd function for Flash Attention V2."""

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

        for q_block_idx in range(num_q_blocks):
            q_start = q_block_idx * block_size
            q_end = min(q_start + block_size, seq_len)

            Q_block = Q[:, :, q_start:q_end, :]

            M_block = M[:, :, q_start:q_end].clone()
            L_block = L[:, :, q_start:q_end].clone()
            O_block = O[:, :, q_start:q_end, :].clone()

            if causal:
                kv_end_block = q_block_idx + 1
            else:
                kv_end_block = num_kv_blocks

            for kv_block_idx in range(kv_end_block):
                kv_start = kv_block_idx * block_size
                kv_end = min(kv_start + block_size, seq_len)

                K_block = K[:, :, kv_start:kv_end, :]
                V_block = V[:, :, kv_start:kv_end, :]

                S_block = torch.matmul(Q_block, K_block.transpose(-2, -1)) * scale

                if causal and kv_block_idx == q_block_idx:
                    q_positions = torch.arange(q_start, q_end, device=Q.device)
                    kv_positions = torch.arange(kv_start, kv_end, device=Q.device)
                    mask = q_positions[:, None] < kv_positions[None, :]
                    S_block = S_block.masked_fill(mask, float("-inf"))

                S_block_max = S_block.max(dim=-1).values
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
        Q, K, V, O, L, M = ctx.saved_tensors
        scale = ctx.scale
        block_size = ctx.block_size
        causal = ctx.causal

        batch_size, num_heads, seq_len, head_dim = Q.shape
        input_dtype = Q.dtype

        Q, K, V, O = Q.float(), K.float(), V.float(), O.float()
        grad_output = grad_output.float()

        grad_Q = torch.zeros_like(Q)
        grad_K = torch.zeros_like(K)
        grad_V = torch.zeros_like(V)

        num_blocks = math.ceil(seq_len / block_size)
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

                if causal and kv_start > q_end - 1:
                    continue

                K_block = K[:, :, kv_start:kv_end, :]
                V_block = V[:, :, kv_start:kv_end, :]

                S_block = torch.matmul(Q_block, K_block.transpose(-2, -1)) * scale

                if causal and kv_block_idx == q_block_idx:
                    q_positions = torch.arange(q_start, q_end, device=Q.device)
                    kv_positions = torch.arange(kv_start, kv_end, device=Q.device)
                    mask = q_positions[:, None] < kv_positions[None, :]
                    S_block = S_block.masked_fill(mask, float("-inf"))

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
                grad_S_block = P_block * (grad_P_block - D_block)

                grad_Q[:, :, q_start:q_end, :] += torch.matmul(grad_S_block, K_block) * scale
                grad_K[:, :, kv_start:kv_end, :] += torch.matmul(
                    grad_S_block.transpose(-2, -1), Q_block
                ) * scale

        return grad_Q.to(input_dtype), grad_K.to(input_dtype), grad_V.to(input_dtype), None, None, None


class FlashAttentionV2(BaseAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        block_size: int = 64,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = False,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout, bias, causal)

        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.block_size = block_size

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        """Compute Flash Attention V2 forward pass."""
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
        """Calculate compute savings from causal block skipping."""
        num_blocks = math.ceil(seq_len / self.block_size)

        total_blocks = num_blocks * num_blocks
        computed_blocks = num_blocks * (num_blocks + 1) // 2
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
        return f"{super().extra_repr()}, block_size={self.block_size}, memory_complexity=O(N)"