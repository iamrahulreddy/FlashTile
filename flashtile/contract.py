from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

@dataclass(frozen=True)
class Problem:
    batch: int
    q_heads: int
    kv_heads: int
    q_len: int
    k_len: int
    head_dim: int
    value_dim: int
    scale: float

    @property
    def group(self) -> int:
        return self.q_heads // self.kv_heads

def accum_dtype(dtype: torch.dtype) -> torch.dtype:
    """fp64 stays fp64 (for gradcheck), everything else accumulates in fp32."""
    return torch.float64 if dtype == torch.float64 else torch.float32

def _positions(pos: Optional[torch.Tensor], length: int, name: str, device, materialize: bool):
    if pos is None:
        return torch.arange(length, device=device) if materialize else None
    if not isinstance(pos, torch.Tensor) or pos.dim() != 1 or pos.shape[0] != length:
        raise ValueError(f"{name} must be a 1-D tensor of length {length}")
    if pos.dtype.is_floating_point or pos.dtype == torch.bool:
        raise TypeError(f"{name} must be an integer tensor, got {pos.dtype}")
    return pos.to(device=device, dtype=torch.long) if materialize else pos

def validate(q, k, v, *, scale, causal, allow_mask, q_pos, k_pos, materialize_positions=True):
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if t.dim() != 4:
            raise ValueError(f"{name} must be 4-D [B, H, L, D], got shape {tuple(t.shape)}")
        if not t.is_floating_point():
            raise TypeError(f"{name} must be floating point, got {t.dtype}")
        if 0 in t.shape:
            raise ValueError(f"{name} has an empty dimension: {tuple(t.shape)}")
    if not (q.dtype == k.dtype == v.dtype):
        raise TypeError(f"q, k, v must share one dtype, got {q.dtype}, {k.dtype}, {v.dtype}")
    if not (q.device == k.device == v.device):
        raise ValueError("q, k, v must be on the same device")

    batch, q_heads, q_len, head_dim = q.shape
    kb, kv_heads, k_len, k_dim = k.shape
    vb, v_heads, v_len, value_dim = v.shape
    if not (batch == kb == vb):
        raise ValueError(f"batch sizes differ: q {batch}, k {kb}, v {vb}")
    if kv_heads != v_heads or k_len != v_len:
        raise ValueError("k and v must agree on heads and length")
    if head_dim != k_dim:
        raise ValueError(f"q and k feature widths differ: {head_dim} vs {k_dim}")
    if q_heads % kv_heads:
        raise ValueError(f"q heads ({q_heads}) must be a multiple of kv heads ({kv_heads})")

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    scale = float(scale)
    if not (math.isfinite(scale) and scale > 0):
        raise ValueError(f"scale must be positive and finite, got {scale}")

    if causal and q_len != k_len and (q_pos is None or k_pos is None):
        raise ValueError(
            "causal attention with q_len != k_len needs explicit q_pos and k_pos; "
            "alignment is never guessed"
        )
    q_pos = _positions(q_pos, q_len, "q_pos", q.device, materialize_positions)
    k_pos = _positions(k_pos, k_len, "k_pos", q.device, materialize_positions)

    if allow_mask is not None:
        if not isinstance(allow_mask, torch.Tensor) or allow_mask.dtype != torch.bool:
            raise TypeError("allow_mask must be a bool tensor (True = may attend)")
        if allow_mask.dim() > 4:
            raise ValueError("allow_mask must broadcast to [B, Hq, Lq, Lk]")
        try:
            allow_mask = allow_mask.to(q.device).expand(batch, q_heads, q_len, k_len)
        except RuntimeError as err:
            raise ValueError(
                f"allow_mask of shape {tuple(allow_mask.shape)} does not broadcast to "
                f"{(batch, q_heads, q_len, k_len)}"
            ) from err

    problem = Problem(batch, q_heads, kv_heads, q_len, k_len, head_dim, value_dim, scale)
    return problem, allow_mask, q_pos, k_pos
