"""Sliding Window Attention Implementation."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from flashtile.attention.base_attention import BaseAttention


class SlidingWindowAttention(BaseAttention):
    """Sliding Window Attention with Flash Attention memory efficiency."""

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
        if attn_mask is not None:
            raise NotImplementedError(
                "Custom attention masks not supported in Sliding Window Attention. "
                "Use window_size and causal parameters."
            )

        batch_size, seq_len, _ = x.shape

        Q, K, V = self._project_qkv(x)

        output = self._sliding_window_flash(Q, K, V, seq_len)

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
        batch_size, num_heads, _, head_dim = Q.shape
        input_dtype = Q.dtype
        scale = self.scale

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

        num_blocks = math.ceil(seq_len / self.block_size)

        for q_block_idx in range(num_blocks):
            q_start = q_block_idx * self.block_size
            q_end = min(q_start + self.block_size, seq_len)

            Q_block = Q[:, :, q_start:q_end, :]

            M_block = M[:, :, q_start:q_end].clone()
            L_block = L[:, :, q_start:q_end].clone()
            O_block = O[:, :, q_start:q_end, :].clone()

            if self.causal:
                window_end = q_end
                window_start = max(0, q_start - self.window_size + 1)
            else:
                window_start = max(0, q_start - self.window_size // 2)
                window_end = min(seq_len, (q_end - 1) + self.window_size // 2 + 1)

            kv_start_block = window_start // self.block_size
            kv_end_block = min(math.ceil(window_end / self.block_size), num_blocks)

            for kv_block_idx in range(kv_start_block, kv_end_block):
                kv_start = kv_block_idx * self.block_size
                kv_end = min(kv_start + self.block_size, seq_len)

                if self.causal and kv_start > q_end - 1:
                    continue

                K_block = K[:, :, kv_start:kv_end, :]
                V_block = V[:, :, kv_start:kv_end, :]

                S_block = torch.matmul(Q_block, K_block.transpose(-2, -1)) * scale

                q_positions = torch.arange(q_start, q_end, device=Q.device)
                kv_positions = torch.arange(kv_start, kv_end, device=Q.device)

                if self.causal:
                    window_starts = (q_positions - self.window_size + 1).clamp(min=0)
                    window_mask = (
                        (q_positions[:, None] < kv_positions[None, :]) | 
                        (kv_positions[None, :] < window_starts[:, None])
                    )
                else:
                    half_window = self.window_size // 2
                    window_mask = (
                        (kv_positions[None, :] < q_positions[:, None] - half_window) |
                        (kv_positions[None, :] > q_positions[:, None] + half_window)
                    )

                S_block = S_block.masked_fill(window_mask, float("-inf"))

                S_block_max = S_block.max(dim=-1).values
                
                valid_mask = ~torch.isinf(S_block_max)
                S_block_max = torch.where(valid_mask, S_block_max, M_block)
                
                M_new = torch.maximum(M_block, S_block_max)
                exp_diff = torch.exp(M_block - M_new)
                exp_S = torch.exp(S_block - M_new.unsqueeze(-1))

                exp_diff = torch.nan_to_num(exp_diff, nan=0.0)
                exp_S = torch.nan_to_num(exp_S, nan=0.0)

                L_new = L_block * exp_diff + exp_S.sum(dim=-1)

                P_block = exp_S

                O_block = O_block * (L_block * exp_diff).unsqueeze(-1) / L_new.unsqueeze(-1).clamp(min=1e-8)
                O_block = O_block + torch.matmul(P_block, V_block) / L_new.unsqueeze(-1).clamp(min=1e-8)

                M_block = M_new
                L_block = L_new

            O[:, :, q_start:q_end, :] = O_block
            M[:, :, q_start:q_end] = M_block
            L[:, :, q_start:q_end] = L_block

        return O.to(input_dtype)

    def get_memory_usage(self, batch_size: int, seq_len: int) -> Dict[str, Any]:
        dtype_bytes = 4

        blocks_per_window = math.ceil(self.window_size / self.block_size)

        block_attention_bytes = (
            batch_size * self.num_heads * self.block_size * self.block_size * dtype_bytes
        )
        qkv_bytes = 3 * batch_size * self.num_heads * seq_len * self.head_dim * dtype_bytes
        statistics_bytes = 2 * batch_size * self.num_heads * seq_len * dtype_bytes
        output_bytes = batch_size * seq_len * self.embed_dim * dtype_bytes

        total_bytes = block_attention_bytes + qkv_bytes + statistics_bytes + output_bytes

        effective_window = min(self.window_size, seq_len)
        window_bytes = (
            batch_size * self.num_heads * seq_len * effective_window * dtype_bytes
        )

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
        return (
            f"{super().extra_repr()}, window_size={self.window_size}, "
            f"block_size={self.block_size}"
        )