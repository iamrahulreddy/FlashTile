"""Naive Attention Implementation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from flashtile.attention.base_attention import BaseAttention


class NaiveAttention(BaseAttention):
    """Standard scaled dot-product attention with O(N²) memory complexity."""

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        Q, K, V = self._project_qkv(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if self.causal:
            causal_mask = self._create_causal_mask(seq_len, x.device)
            attention_scores = attention_scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0),
                float("-inf"),
            )

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            elif attn_mask.dim() != 4:
                raise ValueError(
                    "attn_mask must be 2D, 3D, or 4D with shape "
                    "(seq, seq), (batch, seq, seq), or (batch, heads, seq, seq)."
                )

            attn_mask = attn_mask.to(device=attention_scores.device)
            if attn_mask.dtype == torch.bool:
                attention_scores = attention_scores.masked_fill(attn_mask, float("-inf"))
            else:
                additive_mask = attn_mask.to(dtype=attention_scores.dtype)
                additive_mask = torch.where(
                    torch.isnan(additive_mask), torch.zeros_like(additive_mask), additive_mask
                )
                attention_scores = attention_scores + additive_mask

        row_max = attention_scores.max(dim=-1, keepdim=True).values
        fully_masked = torch.isneginf(row_max)
        safe_scores = torch.where(fully_masked, torch.zeros_like(attention_scores), attention_scores)
        attention_weights = F.softmax(safe_scores, dim=-1)
        attention_weights = torch.where(fully_masked, torch.zeros_like(attention_weights), attention_weights)

        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, V)

        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)

        return output, attention_weights

    def get_memory_usage(self, batch_size: int, seq_len: int) -> Dict[str, Any]:
        dtype_bytes = 4

        attention_matrix_elements = batch_size * self.num_heads * seq_len * seq_len
        attention_matrix_bytes = attention_matrix_elements * dtype_bytes

        qkv_elements = 3 * batch_size * self.num_heads * seq_len * self.head_dim
        qkv_bytes = qkv_elements * dtype_bytes

        output_elements = batch_size * seq_len * self.embed_dim
        output_bytes = output_elements * dtype_bytes

        total_bytes = attention_matrix_bytes + qkv_bytes + output_bytes

        result = {
            "attention_matrix_mb": attention_matrix_bytes / 1e6,
            "qkv_mb": qkv_bytes / 1e6,
            "output_mb": output_bytes / 1e6,
            "total_mb": total_bytes / 1e6,
            "total_gb": total_bytes / 1e9,
            "complexity": "O(N²)",
            "bottleneck": "attention_matrix",
            "attention_matrix_percentage": 100 * attention_matrix_bytes / total_bytes,
        }

        if total_bytes > 8e9:
            result["warning"] = (
                f"High memory usage ({result['total_gb']:.1f} GB). "
                f"Consider using FlashAttentionV2 for O(N) memory."
            )

        return result

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, memory_complexity=O(N²)"