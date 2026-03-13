"""
Naive Attention Implementation
==============================

This module implements standard scaled dot-product attention with O(N²) memory
complexity. It serves as the correctness reference for validating memory-efficient
implementations and provides attention weight visualization capabilities.

Algorithm Overview
------------------
Standard attention computes:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

This requires materializing the full N×N attention matrix, resulting in
O(N²) memory usage where N is the sequence length. For a sequence of length
4096 with batch_size=2, num_heads=8, this means storing 2 × 8 × 4096 × 4096 × 4
= 1.07 GB just for the attention weights.

Use Cases
---------
1. Correctness Testing: Validate that Flash Attention produces identical outputs
2. Attention Visualization: Inspect attention patterns for interpretability
3. Short Sequences: For seq_len < 512, the overhead may be acceptable
4. Debugging: Full attention matrix aids in understanding model behavior

Limitations
-----------
- Memory grows quadratically with sequence length
- Will OOM for long sequences (typically > 4096 tokens on 16GB GPU)
- May be faster than the repo's Python Flash Attention references on GPU because
  it uses fused GEMMs, but it is far less memory-efficient
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from flashtile.attention.base_attention import BaseAttention


class NaiveAttention(BaseAttention):
    """
    Standard scaled dot-product attention with O(N²) memory complexity.

    This implementation follows the original Transformer attention mechanism
    as described in "Attention Is All You Need" (Vaswani et al., 2017). It
    materializes the full attention matrix, making it suitable for:
    - Correctness verification of memory-efficient implementations
    - Attention weight visualization and interpretability analysis
    - Short sequence tasks where memory is not a constraint

    The attention computation follows these steps:
    1. Project input to queries (Q), keys (K), and values (V)
    2. Compute attention scores: scores = Q @ K^T / sqrt(head_dim)
    3. Apply causal mask if specified
    4. Apply softmax to get attention weights
    5. Compute weighted sum: output = attention_weights @ V
    6. Project output back to embed_dim

    Parameters
    ----------
    embed_dim : int
        The total embedding dimension. Must be divisible by num_heads.

    num_heads : int
        The number of attention heads. Each head operates on a subspace
        of dimension head_dim = embed_dim // num_heads.

    dropout : float, optional
        Dropout probability applied to attention weights before computing
        the weighted sum. Default is 0.0 (no dropout).

    bias : bool, optional
        If True, linear projection layers include learnable bias terms.
        Default is True.

    causal : bool, optional
        If True, applies causal (autoregressive) masking where each position
        can only attend to positions at or before it. This is essential for
        decoder-style language models. Default is False.

    Attributes
    ----------
    All attributes from BaseAttention, plus no additional attributes.

    Examples
    --------
    >>> import torch
    >>> from flashtile import NaiveAttention
    >>>
    >>> # Create attention layer
    >>> attention = NaiveAttention(
    ...     embed_dim=512,
    ...     num_heads=8,
    ...     causal=True,
    ... )
    >>>
    >>> # Forward pass
    >>> x = torch.randn(2, 256, 512)  # (batch, seq_len, embed_dim)
    >>> output, weights = attention(x)
    >>>
    >>> print(f"Output shape: {output.shape}")  # (2, 256, 512)
    >>> print(f"Weights shape: {weights.shape}")  # (2, 8, 256, 256)
    >>>
    >>> # Visualize attention for first head
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(weights[0, 0].detach().cpu())
    >>> plt.title("Attention Weights (Head 0)")

    Notes
    -----
    Memory Usage
        For batch_size=B, seq_len=N, num_heads=H, head_dim=D:
        - Attention matrix: B × H × N × N × 4 bytes (float32)
        - QKV tensors: 3 × B × H × N × D × 4 bytes
        - Total scales as O(N²) due to attention matrix

    Numerical Stability
        The implementation uses PyTorch's F.softmax which handles numerical
        stability internally by subtracting the maximum value before exp().

    See Also
    --------
    FlashAttentionV1 : Memory-efficient O(N) attention
    FlashAttentionV2 : Optimized Flash Attention with causal optimization
    """

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute standard scaled dot-product attention.

        This method performs the full attention computation, materializing
        the N×N attention matrix. Unlike Flash Attention implementations,
        it returns the attention weights for visualization purposes.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, embed_dim).
            The tensor should contain finite values (no NaN or Inf).

        attn_mask : torch.Tensor, optional
            Attention mask of shape (sequence_length, sequence_length),
            (batch_size, sequence_length, sequence_length), or
            (batch_size, num_heads, sequence_length, sequence_length).
            Supported formats:
            - Boolean mask: True means masked (set to -inf before softmax)
            - Additive mask: values added directly to attention scores
              (e.g., 0 for keep, -inf for mask)
            This mask is applied **in addition to** any causal masking
            (i.e., both causal and explicit masks are OR-combined). If you
            pass a causal mask explicitly while ``causal=True``, positions
            will be double-masked; this is redundant but not incorrect.
            Default is None.

        Returns
        -------
        output : torch.Tensor
            Attention output of shape (batch_size, sequence_length, embed_dim).
            This is the result of the weighted sum of values.

        attention_weights : torch.Tensor
            Attention weight matrix of shape
            (batch_size, num_heads, sequence_length, sequence_length).
            Each entry [b, h, i, j] represents how much position i attends
            to position j in batch b, head h. Rows sum to 1 after softmax.

        Raises
        ------
        RuntimeError
            If the input tensor has incorrect dimensions.
            If CUDA runs out of memory for long sequences.

        Examples
        --------
        >>> model = NaiveAttention(embed_dim=512, num_heads=8, causal=True)
        >>> x = torch.randn(2, 1024, 512)
        >>> output, weights = model(x)
        >>>
        >>> # Verify causal masking
        >>> # Upper triangle should be zero (no attending to future)
        >>> upper_triangle = torch.triu(weights[0, 0], diagonal=1)
        >>> assert upper_triangle.sum() == 0
        """
        batch_size, seq_len, _ = x.shape

        # Step 1: Project input to Q, K, V
        # Each has shape: (batch_size, num_heads, seq_len, head_dim)
        Q, K, V = self._project_qkv(x)

        # Step 2: Compute attention scores
        # scores shape: (batch_size, num_heads, seq_len, seq_len)
        # This is the O(N²) operation that creates the memory bottleneck
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Step 3: Apply causal mask if specified
        # Causal masking ensures position i only attends to positions <= i
        if self.causal:
            causal_mask = self._create_causal_mask(seq_len, x.device)
            # Expand mask to match attention_scores shape for broadcasting
            # True positions are set to -inf so softmax gives them weight 0
            attention_scores = attention_scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0),  # Add batch and head dims
                float("-inf"),
            )

        # Step 4: Apply optional additional attention mask
        if attn_mask is not None:
            # Handle different mask shapes
            if attn_mask.dim() == 2:
                # (seq_len, seq_len) -> broadcast over batch and heads
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                # (batch, seq_len, seq_len) -> broadcast over heads
                attn_mask = attn_mask.unsqueeze(1)
            elif attn_mask.dim() != 4:
                raise ValueError(
                    "attn_mask must be 2D, 3D, or 4D with shape "
                    "(seq, seq), (batch, seq, seq), or (batch, heads, seq, seq)."
                )

            # Normalize mask device/dtype for reliable broadcasting.
            attn_mask = attn_mask.to(device=attention_scores.device)
            if attn_mask.dtype == torch.bool:
                attention_scores = attention_scores.masked_fill(attn_mask, float("-inf"))
            else:
                additive_mask = attn_mask.to(dtype=attention_scores.dtype)
                # Preserve +/-inf values (valid additive masking) and only sanitize NaN.
                additive_mask = torch.where(
                    torch.isnan(additive_mask), torch.zeros_like(additive_mask), additive_mask
                )
                attention_scores = attention_scores + additive_mask

        # Step 5: Apply softmax to get attention weights
        # Each row sums to 1 (probability distribution over keys).
        # Handle fully masked rows explicitly to avoid NaN from softmax(-inf, -inf, ...).
        row_max = attention_scores.max(dim=-1, keepdim=True).values
        fully_masked = torch.isneginf(row_max)
        safe_scores = torch.where(fully_masked, torch.zeros_like(attention_scores), attention_scores)
        attention_weights = F.softmax(safe_scores, dim=-1)
        attention_weights = torch.where(fully_masked, torch.zeros_like(attention_weights), attention_weights)

        # Step 6: Apply dropout to attention weights (if configured)
        attention_weights = self.dropout(attention_weights)

        # Step 7: Compute weighted sum of values
        # output shape: (batch_size, num_heads, seq_len, head_dim)
        output = torch.matmul(attention_weights, V)

        # Step 8: Reshape and project output
        # Transpose: (batch_size, seq_len, num_heads, head_dim)
        output = output.transpose(1, 2).contiguous()

        # Combine heads: (batch_size, seq_len, embed_dim)
        output = output.reshape(batch_size, seq_len, self.embed_dim)

        # Final output projection
        output = self.out_proj(output)

        return output, attention_weights

    def get_memory_usage(self, batch_size: int, seq_len: int) -> Dict[str, Any]:
        """
        Calculate theoretical memory usage for naive attention.

        This method provides a detailed breakdown of memory consumption,
        highlighting the O(N²) attention matrix as the primary bottleneck.

        Parameters
        ----------
        batch_size : int
            The batch size for memory calculation.

        seq_len : int
            The sequence length for memory calculation.

        Returns
        -------
        dict
            Dictionary containing:
            - attention_matrix_mb: Memory for NxN attention weights (float)
            - qkv_mb: Memory for Q, K, V tensors (float)
            - output_mb: Memory for output tensor (float)
            - total_mb: Total memory in megabytes (float)
            - total_gb: Total memory in gigabytes (float)
            - complexity: Memory complexity class, always "O(N²)" (str)
            - bottleneck: The component causing O(N²) complexity (str)
            - warning: Warning message if memory is high (str, optional)

        Examples
        --------
        >>> model = NaiveAttention(embed_dim=512, num_heads=8)
        >>> mem = model.get_memory_usage(batch_size=2, seq_len=4096)
        >>> print(f"Attention matrix: {mem['attention_matrix_mb']:.1f} MB")
        >>> print(f"Total: {mem['total_gb']:.2f} GB")
        >>> if 'warning' in mem:
        ...     print(f"Warning: {mem['warning']}")
        """
        # Assuming float32 (4 bytes per element)
        dtype_bytes = 4

        # Attention matrix: (batch, heads, seq_len, seq_len)
        # This is the O(N²) bottleneck
        attention_matrix_elements = batch_size * self.num_heads * seq_len * seq_len
        attention_matrix_bytes = attention_matrix_elements * dtype_bytes

        # QKV tensors: 3 × (batch, heads, seq_len, head_dim)
        qkv_elements = 3 * batch_size * self.num_heads * seq_len * self.head_dim
        qkv_bytes = qkv_elements * dtype_bytes

        # Output tensor: (batch, seq_len, embed_dim)
        output_elements = batch_size * seq_len * self.embed_dim
        output_bytes = output_elements * dtype_bytes

        # Total memory
        total_bytes = attention_matrix_bytes + qkv_bytes + output_bytes

        # Build result dictionary
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

        # Add warning for high memory usage
        if total_bytes > 8e9:  # > 8 GB
            result["warning"] = (
                f"High memory usage ({result['total_gb']:.1f} GB). "
                f"Consider using FlashAttentionV2 for O(N) memory."
            )

        return result

    def extra_repr(self) -> str:
        """Return string representation with naive attention specific info."""
        return f"{super().extra_repr()}, memory_complexity=O(N²)"
