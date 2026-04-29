from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from flashtile.attention.base_attention import BaseAttention


class GroupedQueryAttention(BaseAttention):
    """Grouped-Query Attention with block-wise computation."""

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
        nn.Module.__init__(self)

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

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.block_size = block_size
        self.dropout_p = dropout
        self.causal = causal

        self.num_groups = num_heads // num_kv_heads

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

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        K = K.repeat_interleave(self.num_groups, dim=1)
        V = V.repeat_interleave(self.num_groups, dim=1)

        output = self._flash_attention_forward(Q, K, V)

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
        batch_size, num_heads, seq_len, head_dim = Q.shape
        input_dtype = Q.dtype

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

        return O.to(input_dtype)

    def get_memory_usage(self, batch_size: int, seq_len: int) -> Dict[str, Any]:
        dtype_bytes = 4

        q_bytes = batch_size * seq_len * self.embed_dim * dtype_bytes
        kv_bytes = 2 * batch_size * seq_len * self.num_kv_heads * self.head_dim * dtype_bytes
        expanded_kv_bytes = 2 * batch_size * seq_len * self.num_heads * self.head_dim * dtype_bytes
        full_kv_bytes = 2 * batch_size * seq_len * self.num_heads * self.head_dim * dtype_bytes

        output_bytes = batch_size * seq_len * self.embed_dim * dtype_bytes
        total_bytes = q_bytes + kv_bytes + output_bytes
        effective_forward_bytes = q_bytes + expanded_kv_bytes + output_bytes

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
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, head_dim={self.head_dim}, "
            f"kv_savings={self.num_heads // self.num_kv_heads}x, "
            f"causal={self.causal}, block_size={self.block_size}"
        )


class MultiQueryAttention(GroupedQueryAttention):
    """Multi-Query Attention (MQA) wrapper with a single KV head."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        block_size: int = 64,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = False,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=1,
            block_size=block_size,
            dropout=dropout,
            bias=bias,
            causal=causal,
        )