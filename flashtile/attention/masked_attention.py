"""Masked Attention Fallback implementation."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashtile.attention.base_attention import BaseAttention


def masked_attention_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    scale: float,
    chunk_size: int = 1024,
) -> torch.Tensor:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    batch_size, num_heads, seq_len, head_dim = Q.shape
    
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            bool_mask = attn_mask.to(device=Q.device)
            attn_mask = torch.zeros_like(bool_mask, dtype=Q.dtype).masked_fill(
                bool_mask, float("-inf")
            )
        else:
            attn_mask = attn_mask.to(dtype=Q.dtype, device=Q.device)
            attn_mask = torch.where(torch.isnan(attn_mask), torch.zeros_like(attn_mask), attn_mask)
            
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        elif attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)
        elif attn_mask.dim() != 4:
            raise ValueError(
                "attn_mask must be 2D, 3D, or 4D with shape "
                "(seq, seq), (batch, seq, seq), or (batch, heads, seq, seq)."
            )
        attn_mask = attn_mask.expand(batch_size, num_heads, seq_len, seq_len)
    
    num_chunks = math.ceil(seq_len / chunk_size)
    outputs = []
    
    for i in range(num_chunks):
        q_start = i * chunk_size
        q_end = min(q_start + chunk_size, seq_len)
        
        Q_chunk = Q[:, :, q_start:q_end, :]
        
        scores = torch.matmul(Q_chunk, K.transpose(-2, -1)) * scale
        
        if attn_mask is not None:
            mask_chunk = attn_mask[:, :, q_start:q_end, :]
            scores = scores + mask_chunk
        
        row_max = scores.max(dim=-1, keepdim=True).values
        fully_masked = torch.isneginf(row_max)
        safe_scores = torch.where(fully_masked, torch.zeros_like(scores), scores)
        attn_weights = F.softmax(safe_scores, dim=-1)
        attn_weights = torch.where(fully_masked, torch.zeros_like(attn_weights), attn_weights)
        output_chunk = torch.matmul(attn_weights, V)
        
        outputs.append(output_chunk)
    
    output = torch.cat(outputs, dim=2)
    
    return output


class MaskedAttention(BaseAttention):
    """Attention layer supporting custom masks with chunked computation."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = False,
        chunk_size: int = 1024,
    ):
        super().__init__(embed_dim, num_heads, dropout, bias, causal=causal)

        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        
        self.chunk_size = chunk_size
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        batch_size, seq_len, _ = x.shape

        if self.causal:
            causal_mask = create_causal_mask(seq_len, device=x.device)
            if attn_mask is None:
                attn_mask = causal_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.to(device=x.device) | causal_mask
            else:
                attn_mask = attn_mask.to(device=x.device)
                causal_additive = torch.zeros(
                    seq_len, seq_len, dtype=attn_mask.dtype, device=x.device
                ).masked_fill(causal_mask, float("-inf"))
                attn_mask = attn_mask + causal_additive
        
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        output = masked_attention_forward(
            Q, K, V, attn_mask, self.scale, self.chunk_size
        )
        
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)
        output = self.dropout(output)
        
        return output, None
    
    def get_memory_usage(self, batch_size: int, seq_len: int) -> Dict[str, Any]:
        dtype_bytes = 4
        
        qkv_bytes = 3 * batch_size * self.num_heads * seq_len * self.head_dim * dtype_bytes
        mask_bytes = batch_size * self.num_heads * seq_len * seq_len * dtype_bytes
        chunk_bytes = batch_size * self.num_heads * self.chunk_size * seq_len * dtype_bytes
        output_bytes = batch_size * seq_len * self.embed_dim * dtype_bytes
        
        total_bytes = qkv_bytes + mask_bytes + chunk_bytes + output_bytes
        
        naive_attention_bytes = batch_size * self.num_heads * seq_len * seq_len * dtype_bytes
        naive_total = naive_attention_bytes + qkv_bytes + output_bytes
        
        return {
            "qkv_mb": qkv_bytes / 1e6,
            "mask_mb": mask_bytes / 1e6,
            "chunk_processing_mb": chunk_bytes / 1e6,
            "output_mb": output_bytes / 1e6,
            "total_mb": total_bytes / 1e6,
            "total_gb": total_bytes / 1e9,
            "complexity": "O(N²) with O(N * chunk_size) working memory",
            "chunk_size": self.chunk_size,
            "comparison_with_naive": {
                "naive_mb": naive_total / 1e6,
                "masked_mb": total_bytes / 1e6,
                "reduction_ratio": naive_total / total_bytes if total_bytes > 0 else 0,
            },
        }
    
    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"chunk_size={self.chunk_size}"
        )


def create_padding_mask(
    lengths: torch.Tensor,
    max_len: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if device is None:
        device = lengths.device
    lengths = lengths.to(device=device)
    
    mask = torch.arange(max_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
    
    return mask.unsqueeze(1).unsqueeze(1)


def create_causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    return torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )


__all__ = [
    "masked_attention_forward",
    "MaskedAttention",
    "create_padding_mask",
    "create_causal_mask",
]