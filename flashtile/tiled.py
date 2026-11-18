# Tiled attention in plain PyTorch, with a backward that recomputes each tile.
# Readable version of what the Triton kernels do: running max/sum per query tile, one divide
# at the end, and the backward rebuilds P from Q, K and the saved lse. Never stores [Lq, Lk].

from __future__ import annotations

from typing import List, Tuple

import torch

from flashtile.contract import Problem, accum_dtype, validate

NEG_INF = float("-inf")

def _blocks(pos_cpu: torch.Tensor, block: int) -> List[Tuple[int, int, int, int]]:
    # (start, end, min pos, max pos) per tile
    out = []
    for start in range(0, pos_cpu.shape[0], block):
        end = min(start + block, pos_cpu.shape[0])
        chunk = pos_cpu[start:end]
        out.append((start, end, int(chunk.min()), int(chunk.max())))
    return out

def _tile_kind(causal: bool, q_blk, k_blk) -> str:
    # skip: all keys in the future, full: no mask needed, partial: needs a mask
    if not causal:
        return "full"
    _, _, q_min, q_max = q_blk
    _, _, k_min, k_max = k_blk
    if k_min > q_max:
        return "skip"
    if k_max <= q_min:
        return "full"
    return "partial"

def _tile_allowed(kind, mask5, q_pos, k_pos, q_blk, k_blk):
    # bool mask for one tile, None if everything is allowed
    qs, qe = q_blk[0], q_blk[1]
    ks, ke = k_blk[0], k_blk[1]
    allowed = None
    if kind == "partial":
        allowed = k_pos[ks:ke][None, :] <= q_pos[qs:qe][:, None]
    if mask5 is not None:
        tile = mask5[..., qs:qe, ks:ke]
        allowed = tile if allowed is None else allowed & tile
    return allowed

def tiled_forward(q, k, v, p: Problem, mask, q_pos, k_pos, causal, block_q, block_k):
    # returns (out, lse, number of tiles computed)
    acc = accum_dtype(q.dtype)
    G = p.group
    qg = q.unflatten(1, (p.kv_heads, G))  # [B, Hkv, G, Lq, d]  (view)
    kg = k.unsqueeze(2)  # [B, Hkv, 1, Lk, d]  (view, broadcast over G)
    vg = v.unsqueeze(2)
    mask5 = None if mask is None else mask.unflatten(1, (p.kv_heads, G))

    out = torch.empty(p.batch, p.kv_heads, G, p.q_len, p.value_dim, dtype=acc, device=q.device)
    lse = torch.empty(p.batch, p.kv_heads, G, p.q_len, dtype=acc, device=q.device)
    q_blocks = _blocks(q_pos.cpu(), block_q)
    k_blocks = _blocks(k_pos.cpu(), block_k)
    tiles = 0

    for q_blk in q_blocks:
        qs, qe = q_blk[0], q_blk[1]
        qb = qg[..., qs:qe, :].to(acc)
        m = torch.full(qb.shape[:-1], NEG_INF, dtype=acc, device=q.device)
        l = torch.zeros_like(m)
        o = torch.zeros(*qb.shape[:-1], p.value_dim, dtype=acc, device=q.device)

        for k_blk in k_blocks:
            kind = _tile_kind(causal, q_blk, k_blk)
            if kind == "skip":
                continue
            tiles += 1
            ks, ke = k_blk[0], k_blk[1]
            s = torch.matmul(qb, kg[..., ks:ke, :].to(acc).transpose(-1, -2)) * p.scale
            allowed = _tile_allowed(kind, mask5, q_pos, k_pos, q_blk, k_blk)
            if allowed is not None:
                s = s.masked_fill(~allowed, NEG_INF)

            m_new = torch.maximum(m, s.amax(dim=-1))
            # rows with nothing allowed yet keep m = -inf; shift by 0 so -inf - (-inf) never happens
            m_shift = torch.where(torch.isneginf(m_new), torch.zeros_like(m_new), m_new)
            alpha = torch.exp(m - m_shift)
            pt = torch.exp(s - m_shift.unsqueeze(-1))
            l = l * alpha + pt.sum(dim=-1)
            o = o * alpha.unsqueeze(-1) + torch.matmul(pt, vg[..., ks:ke, :].to(acc))
            m = m_new

        empty = l == 0  # fully masked row
        l_safe = torch.where(empty, torch.ones_like(l), l)
        out[..., qs:qe, :] = torch.where(
            empty.unsqueeze(-1), torch.zeros_like(o), o / l_safe.unsqueeze(-1)
        )
        m_safe = torch.where(empty, torch.zeros_like(m), m)
        lse[..., qs:qe] = torch.where(empty, torch.full_like(m, NEG_INF), m_safe + torch.log(l_safe))

    return out.flatten(1, 2).to(q.dtype), lse.flatten(1, 2), tiles

def tiled_backward(do, q, k, v, out, lse, p: Problem, mask, q_pos, k_pos, causal, block_q, block_k, dlse=None):
    # dK/dV sum over the query heads of each group. dlse is the gradient of the returned lse;
    # since d lse / d s = P it just shifts delta: dS = P * (dP - (delta - dlse))
    acc = accum_dtype(q.dtype)
    G = p.group
    qg = q.unflatten(1, (p.kv_heads, G))
    dog = do.unflatten(1, (p.kv_heads, G))
    og = out.unflatten(1, (p.kv_heads, G))
    lseg = lse.unflatten(1, (p.kv_heads, G))
    kg = k.unsqueeze(2)
    vg = v.unsqueeze(2)
    mask5 = None if mask is None else mask.unflatten(1, (p.kv_heads, G))

    # delta_i = <dO_i, O_i>, needs the whole row so compute it once up front
    delta = (dog.to(acc) * og.to(acc)).sum(dim=-1)
    if dlse is not None:
        delta = delta - dlse.to(acc).unflatten(1, (p.kv_heads, G))
    # empty rows have lse = -inf, shifting by 0 makes their P = 0
    lse_shift = torch.where(torch.isneginf(lseg), torch.zeros_like(lseg), lseg)

    dq = torch.zeros(qg.shape, dtype=acc, device=q.device)
    dk = torch.zeros(k.shape, dtype=acc, device=q.device)
    dv = torch.zeros(v.shape, dtype=acc, device=q.device)
    q_blocks = _blocks(q_pos.cpu(), block_q)
    k_blocks = _blocks(k_pos.cpu(), block_k)

    for k_blk in k_blocks:
        ks, ke = k_blk[0], k_blk[1]
        kb = kg[..., ks:ke, :].to(acc)
        vb = vg[..., ks:ke, :].to(acc)
        dk_tile = torch.zeros(p.batch, p.kv_heads, ke - ks, p.head_dim, dtype=acc, device=q.device)
        dv_tile = torch.zeros(p.batch, p.kv_heads, ke - ks, p.value_dim, dtype=acc, device=q.device)

        for q_blk in q_blocks:
            kind = _tile_kind(causal, q_blk, k_blk)
            if kind == "skip":
                continue
            qs, qe = q_blk[0], q_blk[1]
            qb = qg[..., qs:qe, :].to(acc)
            dob = dog[..., qs:qe, :].to(acc)

            s = torch.matmul(qb, kb.transpose(-1, -2)) * p.scale
            allowed = _tile_allowed(kind, mask5, q_pos, k_pos, q_blk, k_blk)
            if allowed is not None:
                s = s.masked_fill(~allowed, NEG_INF)
            pt = torch.exp(s - lse_shift[..., qs:qe].unsqueeze(-1))

            dv_tile += torch.matmul(pt.transpose(-1, -2), dob).sum(dim=2)
            dp = torch.matmul(dob, vb.transpose(-1, -2))
            ds = pt * (dp - delta[..., qs:qe].unsqueeze(-1))
            dq[..., qs:qe, :] += torch.matmul(ds, kb) * p.scale
            dk_tile += torch.matmul(ds.transpose(-1, -2), qb).sum(dim=2) * p.scale

        dk[..., ks:ke, :] = dk_tile
        dv[..., ks:ke, :] = dv_tile

    return dq.flatten(1, 2).to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)

class _TiledAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, q_pos, k_pos, problem, causal, block_q, block_k):
        out, lse, _ = tiled_forward(q, k, v, problem, mask, q_pos, k_pos, causal, block_q, block_k)
        ctx.save_for_backward(q, k, v, out, lse, mask, q_pos, k_pos)
        ctx.cfg = (problem, causal, block_q, block_k)
        ctx.set_materialize_grads(False)  # unused grads come in as None
        return out, lse

    @staticmethod
    def backward(ctx, d_out, d_lse):
        q, k, v, out, lse, mask, q_pos, k_pos = ctx.saved_tensors
        problem, causal, block_q, block_k = ctx.cfg
        if d_out is None:
            d_out = torch.zeros_like(out)
        dq, dk, dv = tiled_backward(
            d_out, q, k, v, out, lse, problem, mask, q_pos, k_pos, causal, block_q, block_k, d_lse
        )
        return dq, dk, dv, None, None, None, None, None, None, None

def tiled_attention(
    q,
    k,
    v,
    *,
    scale=None,
    causal=False,
    allow_mask=None,
    q_pos=None,
    k_pos=None,
    block_q: int = 64,
    block_k: int = 64,
    return_lse: bool = False,
):
    """Tiled attention, forward and backward. Same arguments as flashtile.attention."""
    if block_q <= 0 or block_k <= 0:
        raise ValueError("block sizes must be positive")
    problem, mask, qp, kp = validate(
        q, k, v, scale=scale, causal=causal, allow_mask=allow_mask, q_pos=q_pos, k_pos=k_pos
    )
    out, lse = _TiledAttention.apply(q, k, v, mask, qp, kp, problem, causal, block_q, block_k)
    return (out, lse) if return_lse else out
