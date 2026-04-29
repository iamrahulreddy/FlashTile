from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class BaseAttention(nn.Module, ABC):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = False,
    ) -> None:
        super().__init__()

        if embed_dim <= 0:
            raise ValueError(
                f"embed_dim must be a positive integer, but got embed_dim={embed_dim}. "
                f"The embedding dimension represents the size of input feature vectors."
            )

        if num_heads <= 0:
            raise ValueError(
                f"num_heads must be a positive integer, but got num_heads={num_heads}. "
                f"The number of heads determines how the embedding is split for multi-head attention."
            )

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be evenly divisible by num_heads ({num_heads}). "
                f"Each attention head operates on a slice of size head_dim = embed_dim // num_heads. "
                f"Current configuration would result in head_dim = {embed_dim / num_heads:.2f}, "
                f"which is not an integer. Consider using embed_dim={num_heads * (embed_dim // num_heads)} "
                f"or num_heads that divides {embed_dim} evenly."
            )

        if not (0.0 <= dropout < 1.0):
            raise ValueError(
                f"dropout must be in the range [0.0, 1.0), but got dropout={dropout}. "
                f"A value of 0.0 means no dropout, values approaching 1.0 drop most activations."
            )

        # Store configuration
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout_p = dropout
        self.causal = causal

        # Create projection layers
        # QKV projection: projects input to queries, keys, and values simultaneously
        # Output dimension is 3 * embed_dim (Q, K, V concatenated)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)

        # Output projection: projects concatenated head outputs back to embed_dim
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Dropout layer (use Identity for zero dropout to avoid unnecessary operations)
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pass

    @abstractmethod
    def get_memory_usage(self, batch_size: int, seq_len: int) -> Dict[str, Any]:
        pass

    def _project_qkv(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        # Single projection for Q, K, V (more efficient than three separate projections)
        # Output shape: (batch_size, seq_len, 3 * embed_dim)
        qkv = self.qkv_proj(x)

        # Reshape to separate Q, K, V and split for multi-head attention
        # Shape: (batch_size, seq_len, 3, num_heads, head_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)

        # Permute to (3, batch_size, num_heads, seq_len, head_dim)
        # This puts the Q/K/V dimension first for easy unpacking
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # Unpack Q, K, V
        # Each has shape: (batch_size, num_heads, seq_len, head_dim)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        return Q, K, V

    def _create_causal_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        # Create upper triangular mask (True above diagonal)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        return mask

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, "
            f"causal={self.causal}, "
            f"dropout={self.dropout_p}"
        )
