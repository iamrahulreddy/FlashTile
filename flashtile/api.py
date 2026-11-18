from __future__ import annotations
from typing import Optional, Tuple

import torch

from flashtile.contract import validate
from flashtile.tiled import tiled_attention

try:
    import triton  # noqa: F401

    from flashtile.triton_kernels import TritonAttention

    HAS_TRITON = True
except Exception:  # ImportError, or a broken install that fails at import
    HAS_TRITON = False

_TRITON_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MAX_HEAD_DIM = 256

def _contiguous_start(pos: Optional[torch.Tensor]) -> Optional[int]:
    """pos[0] if pos is a contiguous range, else None. Syncs only if pos lives on the GPU."""
    if pos is None:
        return 0
    start = int(pos[0])
    expected = torch.arange(start, start + pos.shape[0], device=pos.device, dtype=pos.dtype)
    return start if torch.equal(pos, expected) else None

def _plan(q, k, v, causal, allow_mask, q_pos, k_pos) -> Tuple[str, str, Optional[int]]:
    # returns (backend, reason, diag) with diag = q_pos[0] - k_pos[0]
    if not HAS_TRITON:
        return "tiled", "triton is not installed", None
    if q.device.type != "cuda":
        return "tiled", f"tensors are on {q.device.type}, triton needs CUDA", None
    if q.dtype not in _TRITON_DTYPES:
        return "tiled", f"dtype {q.dtype} is not handled by the triton kernels", None
    if allow_mask is not None:
        return "tiled", "arbitrary Boolean masks are only handled by the tiled path", None
    if q.shape[-1] > _MAX_HEAD_DIM or v.shape[-1] > _MAX_HEAD_DIM:
        return "tiled", f"head dim above {_MAX_HEAD_DIM}", None
    if not causal:  # positions only matter for the causal rule
        return "triton", "supported", 0
    q_start, k_start = _contiguous_start(q_pos), _contiguous_start(k_pos)
    if q_start is None or k_start is None:
        return "tiled", "causal positions are not contiguous ranges", None
    return "triton", "supported", q_start - k_start

def select_backend(q, k, v, *, causal=False, allow_mask=None, q_pos=None, k_pos=None) -> Tuple[str, str]:
    """(backend, reason) that "auto" would pick for this call."""
    return _plan(q, k, v, causal, allow_mask, q_pos, k_pos)[:2]

def attention(
    q,
    k,
    v,
    *,
    scale=None,
    causal: bool = False,
    allow_mask=None,
    q_pos=None,
    k_pos=None,
    backend: str = "auto",
    return_lse: bool = False,
    block_q: int = 64,
    block_k: int = 64,
):
    """Exact attention. q [B,Hq,Lq,d], k [B,Hkv,Lk,d], v [B,Hkv,Lk,dv] -> [B,Hq,Lq,dv].

    allow_mask is bool, True = attend. Causal with Lq != Lk needs q_pos/k_pos (CPU tensors
    avoid a GPU sync). Fully masked rows give 0 and lse -inf.
    """
    if backend not in ("auto", "tiled", "triton"):
        raise ValueError(f"unknown backend {backend!r}")
    if backend == "tiled":
        return tiled_attention(
            q, k, v, scale=scale, causal=causal, allow_mask=allow_mask, q_pos=q_pos,
            k_pos=k_pos, block_q=block_q, block_k=block_k, return_lse=return_lse,
        )

    problem, _, _, _ = validate(
        q, k, v, scale=scale, causal=causal, allow_mask=allow_mask, q_pos=q_pos, k_pos=k_pos,
        materialize_positions=False,
    )
    chosen, reason, diag = _plan(q, k, v, causal, allow_mask, q_pos, k_pos)
    if chosen == "tiled":
        if backend == "triton":
            raise ValueError(f"triton backend cannot run this call: {reason}")
        return tiled_attention(
            q, k, v, scale=scale, causal=causal, allow_mask=allow_mask, q_pos=q_pos,
            k_pos=k_pos, block_q=block_q, block_k=block_k, return_lse=return_lse,
        )

    B, Hq, Lq, _ = q.shape
    G = problem.group
    if Lq == 1 and G > 1 and (not causal or diag >= problem.k_len - 1):
        # decode with GQA: pack the group's heads into one query tile so K/V is read once
        qp = q.reshape(B, problem.kv_heads, G, q.shape[-1])
        out, lse = TritonAttention.apply(qp, k, v, problem.scale, False, 0)
        out, lse = out.reshape(B, Hq, 1, out.shape[-1]), lse.reshape(B, Hq, 1)
    else:
        out, lse = TritonAttention.apply(q, k, v, problem.scale, bool(causal), int(diag))
    return (out, lse) if return_lse else out
