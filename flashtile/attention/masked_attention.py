"""
Masked Attention Fallback
=========================

Provides attention computation with support for custom attention masks.
This is a fallback implementation for when Flash Attention's optimized
kernels cannot be used with arbitrary masks.

This implementation uses chunked computation to reduce peak memory
during the forward pass, but remains O(N²) overall due to the mask
and the Q·Kᵀ product spanning all positions.

Use Cases
---------
- Padding masks for variable-length sequences
- Prefix LM masks (causal within prefix, bidirectional after)
- Structured sparsity patterns
- Custom attention biases

Performance Note
----------------
The attention mask itself is O(N²). Chunking reduces the peak working
set by processing Q in blocks, but each chunk still computes scores
against all K positions. Memory savings are a constant factor, not
asymptotic.
"""

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
    """
    Compute attention with custom mask using chunked processing.
    
    This implementation processes attention in chunks to limit peak memory,
    even though the mask itself is O(N²).
    
    Parameters
    ----------
    Q : torch.Tensor
        Query tensor, shape (batch_size, num_heads, seq_len, head_dim).
    K : torch.Tensor
        Key tensor, shape (batch_size, num_heads, seq_len, head_dim).
    V : torch.Tensor
        Value tensor, shape (batch_size, num_heads, seq_len, head_dim).
    attn_mask : torch.Tensor, optional
        Attention mask. Can be:
        - (seq_len, seq_len): 2D mask applied to all batches/heads
        - (batch_size, seq_len, seq_len): 3D mask per batch
        - (batch_size, num_heads, seq_len, seq_len): 4D mask per head
        Values of True or -inf indicate positions to mask.
    scale : float
        Scaling factor for attention scores (typically 1/sqrt(head_dim)).
    chunk_size : int, optional
        Size of chunks for processing. Larger chunks use more memory
        but are more efficient. Default is 1024.
    
    Returns
    -------
    torch.Tensor
        Attention output, shape (batch_size, num_heads, seq_len, head_dim).
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    batch_size, num_heads, seq_len, head_dim = Q.shape
    
    # Normalize mask to (batch, heads, seq, seq) float additive format
    if attn_mask is not None:
        # Convert boolean masks to float additive masks
        # Boolean True means "mask this position" -> set to -inf
        if attn_mask.dtype == torch.bool:
            bool_mask = attn_mask.to(device=Q.device)
            attn_mask = torch.zeros_like(bool_mask, dtype=Q.dtype).masked_fill(
                bool_mask, float("-inf")
            )
        else:
            attn_mask = attn_mask.to(dtype=Q.dtype, device=Q.device)
            # Preserve +/-inf values (valid additive masking) and only sanitize NaN.
            attn_mask = torch.where(torch.isnan(attn_mask), torch.zeros_like(attn_mask), attn_mask)
        if attn_mask.dim() == 2:
            # (seq, seq) -> (1, 1, seq, seq)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        elif attn_mask.dim() == 3:
            # (batch, seq, seq) -> (batch, 1, seq, seq)
            attn_mask = attn_mask.unsqueeze(1)
        elif attn_mask.dim() != 4:
            raise ValueError(
                "attn_mask must be 2D, 3D, or 4D with shape "
                "(seq, seq), (batch, seq, seq), or (batch, heads, seq, seq)."
            )
        # Expand to (batch, heads, seq, seq)
        attn_mask = attn_mask.expand(batch_size, num_heads, seq_len, seq_len)
    
    # Process in chunks to limit memory
    num_chunks = math.ceil(seq_len / chunk_size)
    outputs = []
    
    for i in range(num_chunks):
        q_start = i * chunk_size
        q_end = min(q_start + chunk_size, seq_len)
        
        Q_chunk = Q[:, :, q_start:q_end, :]
        
        # Compute scores for this chunk against all K
        # (batch, heads, chunk_size, seq_len)
        scores = torch.matmul(Q_chunk, K.transpose(-2, -1)) * scale
        
        # Apply mask for this chunk
        if attn_mask is not None:
            mask_chunk = attn_mask[:, :, q_start:q_end, :]
            scores = scores + mask_chunk
        
        # Softmax and apply to V.
        # Handle fully masked rows explicitly to avoid NaN from softmax(-inf, -inf, ...).
        row_max = scores.max(dim=-1, keepdim=True).values
        fully_masked = torch.isneginf(row_max)
        safe_scores = torch.where(fully_masked, torch.zeros_like(scores), scores)
        attn_weights = F.softmax(safe_scores, dim=-1)
        attn_weights = torch.where(fully_masked, torch.zeros_like(attn_weights), attn_weights)
        output_chunk = torch.matmul(attn_weights, V)
        
        outputs.append(output_chunk)
    
    # Concatenate chunks
    output = torch.cat(outputs, dim=2)
    
    return output


class MaskedAttention(BaseAttention):
    """
    Attention layer supporting custom masks with chunked computation.
    
    This is useful when you need:
    - Padding masks for variable-length sequences
    - Custom attention patterns
    - Compatibility with Flash Attention API
    
    Parameters
    ----------
    embed_dim : int
        Total embedding dimension. Must be divisible by num_heads.
    num_heads : int
        Number of attention heads.
    dropout : float, optional
        Dropout probability. Default is 0.0.
    bias : bool, optional
        Use bias in projections. Default is True.
    causal : bool, optional
        If True, apply causal masking in addition to any provided attn_mask.
        Default is False.
    chunk_size : int, optional
        Chunk size for memory-efficient processing. Default is 1024.
    
    Examples
    --------
    >>> # Padding mask example
    >>> model = MaskedAttention(embed_dim=512, num_heads=8)
    >>> x = torch.randn(2, 100, 512)
    >>> 
    >>> # Create padding mask (True = mask)
    >>> lengths = torch.tensor([80, 100])  # Actual lengths
    >>> mask = torch.arange(100).unsqueeze(0) >= lengths.unsqueeze(1)
    >>> mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, seq)
    >>> 
    >>> output, _ = model(x, attn_mask=mask)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = False,
        chunk_size: int = 1024,
    ):
        # Initialize BaseAttention for projection layers
        super().__init__(embed_dim, num_heads, dropout, bias, causal=causal)

        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        
        self.chunk_size = chunk_size
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        """
        Compute masked attention.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, seq_len, embed_dim).
        attn_mask : torch.Tensor, optional
            Attention mask. Values of True or -inf indicate masked positions.
        
        Returns
        -------
        output : torch.Tensor
            Attention output, shape (batch_size, seq_len, embed_dim).
        attention_weights : None
            Always None (for API consistency with Flash Attention).
        """
        batch_size, seq_len, _ = x.shape

        # Apply built-in causal mask when requested; combine with custom masks if provided.
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
        
        # Project QKV
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Apply masked attention
        output = masked_attention_forward(
            Q, K, V, attn_mask, self.scale, self.chunk_size
        )
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)
        output = self.dropout(output)
        
        return output, None
    
    def get_memory_usage(self, batch_size: int, seq_len: int) -> Dict[str, Any]:
        """Calculate theoretical memory usage."""
        dtype_bytes = 4
        
        # QKV: O(N)
        qkv_bytes = 3 * batch_size * self.num_heads * seq_len * self.head_dim * dtype_bytes
        
        # Mask: O(N²) - this is the bottleneck
        mask_bytes = batch_size * self.num_heads * seq_len * seq_len * dtype_bytes
        
        # Chunked scores: O(N * chunk_size)
        chunk_bytes = batch_size * self.num_heads * self.chunk_size * seq_len * dtype_bytes
        
        # Output: O(N)
        output_bytes = batch_size * seq_len * self.embed_dim * dtype_bytes
        
        total_bytes = qkv_bytes + mask_bytes + chunk_bytes + output_bytes
        
        # Calculate comparison with naive (which has O(N²) attention matrix)
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


# =============================================================================
# Utility Functions for Common Mask Types
# =============================================================================

def create_padding_mask(
    lengths: torch.Tensor,
    max_len: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create padding mask from sequence lengths.
    
    Parameters
    ----------
    lengths : torch.Tensor
        Actual lengths of each sequence, shape (batch_size,).
    max_len : int
        Maximum sequence length (padded).
    device : torch.device, optional
        Device for the mask. Inferred from lengths if not provided.
    
    Returns
    -------
    torch.Tensor
        Padding mask of shape (batch_size, 1, 1, max_len) where
        True indicates positions beyond the actual length (should be masked).
    """
    if device is None:
        device = lengths.device
    lengths = lengths.to(device=device)
    
    # (batch, 1) >= (1, max_len) -> (batch, max_len)
    mask = torch.arange(max_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
    
    # Expand to (batch, 1, 1, max_len) for broadcasting with attention scores
    return mask.unsqueeze(1).unsqueeze(1)


def create_causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Create causal (autoregressive) mask.
    
    Parameters
    ----------
    seq_len : int
        Sequence length.
    device : torch.device, optional
        Device for the mask.
    
    Returns
    -------
    torch.Tensor
        Causal mask of shape (seq_len, seq_len) where
        True indicates positions that should be masked.
    """
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
