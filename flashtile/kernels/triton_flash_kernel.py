from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _flash_attention_forward_kernel(
        # Pointers to matrices
        Q_ptr, K_ptr, V_ptr, Out_ptr,
        # Strides for Q (batch, head, seq, dim)
        stride_qb, stride_qh, stride_qm, stride_qk,
        # Strides for K
        stride_kb, stride_kh, stride_kn, stride_kk,
        # Strides for V
        stride_vb, stride_vh, stride_vn, stride_vk,
        # Strides for output
        stride_ob, stride_oh, stride_om, stride_ok,
        # Dimensions
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        # Scale factor
        scale,
        # Block sizes (compile-time constants)
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        # Causal masking flag
        IS_CAUSAL: tl.constexpr,
    ):
        # Get program IDs
        pid_m = tl.program_id(0)  # Which Q block
        pid_bh = tl.program_id(1)  # Which batch-head combo

        # Compute batch and head indices from combined index
        # Use explicit num_heads parameter for robustness
        pid_b = pid_bh // num_heads
        pid_h = pid_bh % num_heads

        # Compute offsets for this Q block
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # Q positions
        offs_k = tl.arange(0, BLOCK_K)  # Head dimension

        # Compute Q block pointer
        q_ptrs = (
            Q_ptr
            + pid_b * stride_qb
            + pid_h * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_k[None, :] * stride_qk
        )

        # Load Q block with masking for out-of-bounds
        q_mask = offs_m[:, None] < seq_len
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

        # Initialize running accumulators for online softmax
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        o_i = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

        # Determine how many K/V blocks to process
        if IS_CAUSAL:
            # For causal, only attend to positions up to current Q block
            kv_bound = (pid_m + 1) * BLOCK_M
            kv_bound = tl.minimum(kv_bound, seq_len)
            num_kv_blocks = tl.cdiv(kv_bound, BLOCK_N)
        else:
            num_kv_blocks = tl.cdiv(seq_len, BLOCK_N)

        # Iterate over K/V blocks
        for kv_block_idx in range(0, num_kv_blocks):
            # Compute K/V block offsets
            offs_n = kv_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)

            # Compute K and V pointers
            k_ptrs = (
                K_ptr
                + pid_b * stride_kb
                + pid_h * stride_kh
                + offs_n[:, None] * stride_kn
                + offs_k[None, :] * stride_kk
            )
            v_ptrs = (
                V_ptr
                + pid_b * stride_vb
                + pid_h * stride_vh
                + offs_n[:, None] * stride_vn
                + offs_k[None, :] * stride_vk
            )

            # Load K and V blocks
            # [MEMORY HIERARCHY] Load K/V blocks from HBM to SRAM
            # We load these once per Q-loop iteration.
            k_mask = offs_n[:, None] < seq_len
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)
            v = tl.load(v_ptrs, mask=k_mask, other=0.0)

            # Cast to float32 for accumulation (needed when inputs are fp16/bf16
            # because softmax math runs in fp32 and tl.dot requires matching dtypes)
            k = k.to(tl.float32)
            v = v.to(tl.float32)

            # Compute attention scores: S = Q @ K^T * scale
            # s shape: [BLOCK_M, BLOCK_N]
            q_f32 = q.to(tl.float32)
            s = tl.dot(q_f32, tl.trans(k)) * scale

            # Apply causal mask if needed
            if IS_CAUSAL:
                # Create causal mask: position i can only attend to j <= i
                causal_mask = offs_m[:, None] < offs_n[None, :]
                s = tl.where(causal_mask, float("-inf"), s)

            # Apply boundary mask for positions beyond sequence length
            boundary_mask = offs_n[None, :] >= seq_len
            s = tl.where(boundary_mask, float("-inf"), s)

            # Online softmax update
            # 1. Find new maximum
            # [ONLINE SOFTMAX] Step 1: Update running max (m)
            # Corresponds to: m_new = max(m_old, max(S_ij))
            # Critical for numerical stability (prevents exp overflow)
            m_ij = tl.max(s, axis=1)
            m_new = tl.maximum(m_i, m_ij)

            # 2. Compute scaling factor for old accumulator
            # [ONLINE SOFTMAX] Step 2: Compute rescaling factor (alpha)
            # Identity: exp(x - m_new) = exp(x - m_old) * exp(m_old - m_new)
            # alpha = exp(m_old - m_new)
            # This factor rescales the OLD accumulator to be compatible with the NEW max.
            alpha = tl.exp(m_i - m_new)

            # 3. Compute exp(s - m_new) for current block
            p = tl.exp(s - m_new[:, None])

            # 4. Update running sum
            l_new = l_i * alpha + tl.sum(p, axis=1)

            # 5. Update output with rescaling
            # o_new = (l_i * alpha * o_i + p @ v) / l_new
            # [ONLINE SOFTMAX] Step 5: Update weighted input (O)
            # O_new = (O_old * l_old * alpha + P_ij * V_j) / l_new
            # We implement this as in-place updates:
            # 1. Rescale old approximation: o_i *= (l_old / l_new) * alpha
            # 2. Add new contribution: o_i += (P * V) / l_new
            o_i = o_i * (l_i * alpha)[:, None] / l_new[:, None]
            o_i = o_i + tl.dot(p, v) / l_new[:, None]

            # Update running statistics
            m_i = m_new
            l_i = l_new

        # Store final output
        out_ptrs = (
            Out_ptr
            + pid_b * stride_ob
            + pid_h * stride_oh
            + offs_m[:, None] * stride_om
            + offs_k[None, :] * stride_ok
        )
        out_mask = offs_m[:, None] < seq_len
        tl.store(out_ptrs, o_i, mask=out_mask)


def triton_flash_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError(
            "Triton is required for triton_flash_attention. "
            "Install with: pip install triton"
        )

    # Validate inputs
    if Q.dim() != 4 or K.dim() != 4 or V.dim() != 4:
        raise ValueError("Q, K, and V must be 4D tensors: (batch, heads, seq_len, head_dim)")
    if Q.shape != K.shape or Q.shape != V.shape:
        raise ValueError(
            f"Q, K, and V must have identical shapes, got "
            f"Q={tuple(Q.shape)}, K={tuple(K.shape)}, V={tuple(V.shape)}"
        )
    if not (Q.is_cuda and K.is_cuda and V.is_cuda):
        raise ValueError("Q, K, and V must all be on CUDA device")
    if Q.device != K.device or Q.device != V.device:
        raise ValueError(
            f"Q, K, and V must be on the same CUDA device, got "
            f"Q={Q.device}, K={K.device}, V={V.device}"
        )
    if Q.dtype != K.dtype or Q.dtype != V.dtype:
        raise ValueError(
            f"Q, K, and V must have the same dtype, got "
            f"Q={Q.dtype}, K={K.dtype}, V={V.dtype}"
        )

    batch_size, num_heads, seq_len, head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)

    # Ensure contiguous
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    # Allocate output
    Out = torch.empty_like(Q)

    # Block sizes (tuned for common GPUs)
    BLOCK_M = 64
    BLOCK_N = 64
    # Triton requires block dimensions to be powers of 2 for tl.dot
    if head_dim & (head_dim - 1) != 0:
        raise ValueError(
            f"Triton kernel requires head_dim to be a power of 2, got {head_dim}. "
            f"Common valid values: 32, 64, 128. Consider adjusting embed_dim and num_heads "
            f"so that embed_dim // num_heads is a power of 2."
        )
    BLOCK_K = head_dim  # Process full head dimension

    # Grid: one program per Q block per (batch, head)
    num_q_blocks = math.ceil(seq_len / BLOCK_M)
    grid = (num_q_blocks, batch_size * num_heads)

    # Launch kernel
    _flash_attention_forward_kernel[grid](
        Q, K, V, Out,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        IS_CAUSAL=causal,
    )

    return Out


# Import BaseAttention for inheritance
from flashtile.attention.base_attention import BaseAttention


class TritonFlashAttention(BaseAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = False,
        block_size: int = 64,  # Ignored but kept for API consistency
    ) -> None:
        """Initialize Triton Flash Attention."""
        if not HAS_TRITON:
            raise RuntimeError(
                "Triton is required for TritonFlashAttention. "
                "Install with: pip install triton"
            )
        
        # Call BaseAttention init for standard projections
        super().__init__(embed_dim, num_heads, dropout, bias, causal)
        
        # Store block_size for API consistency (not used in kernel)
        self.block_size = block_size

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        if attn_mask is not None:
            raise NotImplementedError(
                "Custom attention masks are not supported in TritonFlashAttention. "
                "Use causal=True for causal masking."
            )
        params_require_grad = any(param.requires_grad for param in self.parameters())
        if torch.is_grad_enabled() and (self.training or x.requires_grad or params_require_grad):
            raise RuntimeError(
                "TritonFlashAttention is forward-only and does not implement backward. "
                "Run it in inference mode with torch.no_grad() (and typically model.eval()). "
                "Use FlashAttentionV1/V2 for training."
            )

        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V using BaseAttention's projection
        Q, K, V = self._project_qkv(x)

        # Apply Triton kernel
        output = triton_flash_attention(Q, K, V, causal=self.causal)

        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)
        output = self.dropout(output)

        # Return None for attention weights (consistent with other Flash implementations)
        return output, None

    def get_memory_usage(self, batch_size: int, seq_len: int) -> Dict[str, Any]:
        dtype_bytes = 4  # float32

        # QKV tensors: 3 × (batch, heads, seq_len, head_dim) - O(N)
        qkv_bytes = 3 * batch_size * self.num_heads * seq_len * self.head_dim * dtype_bytes

        # Output tensor: (batch, seq_len, embed_dim) - O(N)
        output_bytes = batch_size * seq_len * self.embed_dim * dtype_bytes

        # Block-sized attention matrix: O(1) since block_size is constant
        block_attention_bytes = (
            batch_size * self.num_heads * 64 * 64 * dtype_bytes
        )

        total_bytes = qkv_bytes + output_bytes + block_attention_bytes

        # Calculate comparison with naive (which has O(N²) attention matrix)
        naive_attention_bytes = (
            batch_size * self.num_heads * seq_len * seq_len * dtype_bytes
        )
        naive_total = naive_attention_bytes + qkv_bytes + output_bytes

        return {
            "qkv_mb": qkv_bytes / 1e6,
            "output_mb": output_bytes / 1e6,
            "block_attention_mb": block_attention_bytes / 1e6,
            "total_mb": total_bytes / 1e6,
            "total_gb": total_bytes / 1e9,
            "complexity": "O(N)",
            "kernel": "Triton",
            "comparison_with_naive": {
                "naive_mb": naive_total / 1e6,
                "flash_mb": total_bytes / 1e6,
                "reduction_ratio": naive_total / total_bytes if total_bytes > 0 else 0,
            },
        }

    def extra_repr(self) -> str:
        """Return string representation."""
        return (
            f"{super().extra_repr()}, kernel=Triton, memory_complexity=O(N)"
        )
