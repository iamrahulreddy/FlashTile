# Dense reference implementation, the oracle every test compares against.
# Builds the full score table and repeats K/V for GQA on purpose. It's the judge, not
# something you'd run for speed.

from __future__ import annotations

import torch

from flashtile.contract import accum_dtype, validate

def attention_reference(q, k, v, *, scale=None, causal=False, allow_mask=None, q_pos=None, k_pos=None):
    """Return (output, lse). Fully masked rows give output 0 and lse -inf."""
    p, mask, qp, kp = validate(
        q, k, v, scale=scale, causal=causal, allow_mask=allow_mask, q_pos=q_pos, k_pos=k_pos
    )
    acc = accum_dtype(q.dtype)
    kx = k.to(acc).repeat_interleave(p.group, dim=1)
    vx = v.to(acc).repeat_interleave(p.group, dim=1)

    scores = torch.matmul(q.to(acc), kx.transpose(-1, -2)) * p.scale

    allowed = torch.ones(p.q_len, p.k_len, dtype=torch.bool, device=q.device)
    if causal:
        allowed = kp[None, :] <= qp[:, None]
    allowed = allowed.expand(p.batch, p.q_heads, p.q_len, p.k_len)
    if mask is not None:
        allowed = allowed & mask
    scores = scores.masked_fill(~allowed, float("-inf"))

    # the max cancels out anyway; detaching keeps amax tie-breaking out of the gradient
    row_max = scores.amax(dim=-1, keepdim=True).detach()
    empty = torch.isneginf(row_max)
    row_max = torch.where(empty, torch.zeros_like(row_max), row_max)

    weights = torch.exp(scores - row_max)
    denom = weights.sum(dim=-1, keepdim=True)
    denom_safe = torch.where(empty, torch.ones_like(denom), denom)
    probs = weights / denom_safe

    out = torch.matmul(probs, vx)
    lse = torch.where(empty, torch.full_like(denom, float("-inf")), row_max + torch.log(denom_safe))
    return out.to(q.dtype), lse.squeeze(-1)
