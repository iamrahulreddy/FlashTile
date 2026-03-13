"""
Base Attention Module
=====================

This module defines the abstract base class for all attention implementations
in FlashTile. The BaseAttention class enforces a consistent interface and
provides shared functionality for parameter validation, projection layers,
and causal mask generation.

Design Philosophy
-----------------
All attention implementations in FlashTile inherit from BaseAttention to ensure:
1. Consistent constructor signatures for interchangeable usage
2. Unified input/output tensor shapes across implementations
3. Shared validation logic to catch configuration errors early
4. Common utility methods (QKV projection, causal masking)

The abstract methods `forward()` and `get_memory_usage()` must be implemented
by all subclasses to provide the actual attention computation and memory
estimation functionality.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class BaseAttention(nn.Module, ABC):
    """
    Abstract base class for all attention mechanism implementations.

    This class provides a common interface and shared functionality for
    attention layers. All attention implementations in FlashTile must
    inherit from this class.

    The class handles:
    - Parameter validation during construction
    - QKV and output projection layer initialization
    - Shared projection and reshaping logic
    - Causal mask generation

    Subclasses must implement:
    - forward(): The actual attention computation
    - get_memory_usage(): Theoretical memory usage calculation

    Parameters
    ----------
    embed_dim : int
        The total embedding dimension (also known as d_model or hidden_size).
        Must be positive and divisible by num_heads.

    num_heads : int
        The number of attention heads. Must be positive and divide embed_dim evenly.
        Each head operates on a subspace of dimension head_dim = embed_dim // num_heads.

    dropout : float, optional
        Dropout probability applied to attention outputs. Must be in range [0, 1).
        Default is 0.0 (no dropout).

    bias : bool, optional
        If True, linear projection layers include a learnable bias term.
        Default is True.

    causal : bool, optional
        If True, applies causal (autoregressive) masking where each position
        can only attend to previous positions. Required for decoder-style models.
        Default is False.

    Attributes
    ----------
    embed_dim : int
        The embedding dimension.

    num_heads : int
        The number of attention heads.

    head_dim : int
        The dimension of each attention head (embed_dim // num_heads).

    scale : float
        The scaling factor for attention scores (1 / sqrt(head_dim)).

    dropout_p : float
        The dropout probability.

    causal : bool
        Whether causal masking is applied.

    qkv_proj : nn.Linear
        Linear layer projecting input to queries, keys, and values.
        Input: (*, embed_dim), Output: (*, 3 * embed_dim)

    out_proj : nn.Linear
        Linear layer projecting attention output back to embed_dim.
        Input: (*, embed_dim), Output: (*, embed_dim)

    dropout : nn.Module
        Dropout layer (nn.Dropout if dropout > 0, else nn.Identity).

    Raises
    ------
    ValueError
        If embed_dim is not positive.
    ValueError
        If num_heads is not positive.
    ValueError
        If embed_dim is not divisible by num_heads.
    ValueError
        If dropout is not in range [0, 1).

    Examples
    --------
    Subclasses should call the parent constructor and then implement
    the abstract methods:

    >>> class MyAttention(BaseAttention):
    ...     def forward(self, x, attn_mask=None):
    ...         # Implementation here
    ...         pass
    ...
    ...     def get_memory_usage(self, batch_size, seq_len):
    ...         # Memory calculation here
    ...         return {"total_mb": 0.0}
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = False,
    ) -> None:
        """
        Initialize the base attention module.

        This constructor validates all parameters and creates the shared
        projection layers used by all attention implementations.
        """
        super().__init__()

        # Comprehensive parameter validation with descriptive error messages
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
        """
        Compute attention output for the input tensor.

        This method must be implemented by all subclasses to perform the
        actual attention computation. The implementation determines the
        memory complexity and performance characteristics.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, embed_dim).
            All values should be finite (no NaN or Inf).

        attn_mask : torch.Tensor, optional
            Attention mask tensor. The exact shape and semantics depend on
            the implementation:
            - For NaiveAttention: (seq_len, seq_len) or (batch, seq_len, seq_len)
            - For Flash implementations: Not supported (use causal=True instead)
            Values of True indicate positions that should be masked (not attended to).
            Default is None (no masking).

        Returns
        -------
        output : torch.Tensor
            Attention output of shape (batch_size, sequence_length, embed_dim).
            This is the weighted sum of values based on attention weights.

        attention_weights : torch.Tensor or None
            For NaiveAttention: Attention weight matrix of shape
                (batch_size, num_heads, sequence_length, sequence_length).
            For memory-efficient implementations (Flash, GQA, MQA): None.
            The attention matrix is not materialized to save memory.

        Examples
        --------
        >>> model = SomeAttention(embed_dim=512, num_heads=8)
        >>> x = torch.randn(2, 1024, 512)
        >>> output, weights = model(x)
        >>> output.shape
        torch.Size([2, 1024, 512])
        """
        pass

    @abstractmethod
    def get_memory_usage(self, batch_size: int, seq_len: int) -> Dict[str, Any]:
        """
        Calculate the theoretical memory usage for given input dimensions.

        This method provides insight into the memory characteristics of each
        attention implementation, helping users understand the trade-offs
        between different implementations.

        Parameters
        ----------
        batch_size : int
            The batch size for memory calculation.

        seq_len : int
            The sequence length for memory calculation.

        Returns
        -------
        dict
            Dictionary containing memory usage information with at least:
            - "total_mb" (float): Total theoretical memory in megabytes
            - "total_gb" (float): Total theoretical memory in gigabytes
            - "complexity" (str): Memory complexity class ("O(N)" or "O(N²)")

            Additional implementation-specific keys may include:
            - "attention_matrix_mb": Memory for NxN attention matrix
            - "qkv_mb": Memory for Q, K, V tensors
            - "block_attention_mb": Memory for block-sized attention (Flash)
            - "kv_savings_ratio": KV cache reduction (GQA/MQA)

        Examples
        --------
        >>> model = FlashAttentionV2(embed_dim=512, num_heads=8)
        >>> mem = model.get_memory_usage(batch_size=2, seq_len=4096)
        >>> print(f"Memory: {mem['total_mb']:.1f} MB ({mem['complexity']})")
        Memory: 1624.0 MB (O(N))
        """
        pass

    def _project_qkv(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project input tensor to queries, keys, and values for multi-head attention.

        This shared utility method performs the QKV projection and reshapes
        the output for multi-head attention computation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, embed_dim).

        Returns
        -------
        Q : torch.Tensor
            Query tensor of shape (batch_size, num_heads, sequence_length, head_dim).

        K : torch.Tensor
            Key tensor of shape (batch_size, num_heads, sequence_length, head_dim).

        V : torch.Tensor
            Value tensor of shape (batch_size, num_heads, sequence_length, head_dim).

        Notes
        -----
        The projection is performed efficiently by computing Q, K, V in a single
        matrix multiplication and then splitting the result. This is more efficient
        than three separate projections due to better memory access patterns.
        """
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
        """
        Create a causal (autoregressive) attention mask.

        The mask ensures that position i can only attend to positions j where j <= i.
        This is essential for decoder-style models like GPT.

        Parameters
        ----------
        seq_len : int
            The sequence length for which to create the mask.

        device : torch.device
            The device on which to create the mask tensor.

        Returns
        -------
        torch.Tensor
            Boolean mask of shape (seq_len, seq_len) where True indicates
            positions that should be masked (cannot be attended to).
            The upper triangular portion (excluding diagonal) is True.

        Examples
        --------
        >>> mask = self._create_causal_mask(4, torch.device("cpu"))
        >>> mask
        tensor([[False,  True,  True,  True],
                [False, False,  True,  True],
                [False, False, False,  True],
                [False, False, False, False]])
        """
        # Create upper triangular mask (True above diagonal)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        return mask

    def extra_repr(self) -> str:
        """
        Return a string representation of the module's configuration.

        This method is called by PyTorch's __repr__ to provide additional
        information about the module's configuration.

        Returns
        -------
        str
            A formatted string showing key configuration parameters.
        """
        return (
            f"embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, "
            f"causal={self.causal}, "
            f"dropout={self.dropout_p}"
        )
