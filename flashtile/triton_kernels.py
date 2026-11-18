"""Triton forward/backward kernels (FlashAttention-2 style).

Causal masking uses one offset, DIAG = q_pos[0] - k_pos[0]: key j is visible to query i when
j <= i + DIAG. That covers prefill, decode and chunked prefill. Calls with a bool mask never
get here, api.py sends them to the tiled path.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
# kernels can only read constexpr globals
_LOG2E = tl.constexpr(LOG2E)
_LN2 = tl.constexpr(LN2)

_FWD_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=1),
]
# dK/dV: one program per key tile, loops over query tiles
_DKDV_CONFIGS = [
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=1),
]
# dQ: one program per query tile, loops over key tiles
_DQ_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=1),
]
_KEY = ["LQ_BUCKET", "LK_BUCKET", "GROUP", "DK", "DV", "CAUSAL"]

# two-kernel is the default (deterministic). atomic does dQ with fp32 atomics in the dK/dV kernel:
# fewer matmuls, but it was slower on most shapes in the benchmarks.
BACKWARD_MODE = os.environ.get("FLASHTILE_BACKWARD", "two-kernel")
if BACKWARD_MODE not in ("two-kernel", "atomic"):
    raise ValueError(f"FLASHTILE_BACKWARD must be 'two-kernel' or 'atomic', got {BACKWARD_MODE!r}")

@triton.jit
def _fwd_inner(
    acc, l_i, m_i, q, k_base, v_base, skn, skd, svn, svd,
    offs_m, offs_dk, offs_dv, start, end, LK, DIAG, qk_scale,
    MASKED: tl.constexpr, CAUSAL: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
    DOT_PREC: tl.constexpr, BLOCK_N: tl.constexpr,
):
    for start_n in range(start, end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        if MASKED:
            n_ok = offs_n < LK
            k_t = tl.load(
                k_base + offs_n[None, :] * skn + offs_dk[:, None] * skd,
                mask=n_ok[None, :] & (offs_dk[:, None] < DK), other=0.0,
            )
        else:
            k_t = tl.load(k_base + offs_n[None, :] * skn + offs_dk[:, None] * skd,
                          mask=offs_dk[:, None] < DK, other=0.0)
        s = tl.dot(q, k_t, input_precision=DOT_PREC) * qk_scale  # log2 units
        if MASKED:
            valid = n_ok[None, :]
            if CAUSAL:
                valid = valid & (offs_n[None, :] <= offs_m[:, None] + DIAG)
            s = tl.where(valid, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_shift = tl.where(m_new == float("-inf"), 0.0, m_new)  # never evaluate -inf - (-inf)
        else:  # no -inf scores in an unmasked tile
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_shift = m_new
        alpha = tl.math.exp2(m_i - m_shift)
        p = tl.math.exp2(s - m_shift[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if MASKED:
            v = tl.load(v_base + offs_n[:, None] * svn + offs_dv[None, :] * svd,
                        mask=n_ok[:, None] & (offs_dv[None, :] < DV), other=0.0)
        else:
            v = tl.load(v_base + offs_n[:, None] * svn + offs_dv[None, :] * svd,
                        mask=offs_dv[None, :] < DV, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision=DOT_PREC)
        m_i = m_new
    return acc, l_i, m_i

@triton.autotune(configs=_FWD_CONFIGS, key=_KEY)
@triton.jit
def _fwd_kernel(
    Q, K, V, O, LSE,
    sqb, sqh, sqm, sqd,
    skb, skh, skn, skd,
    svb, svh, svn, svd,
    sob, soh, som, sod,
    HQ, GROUP, LQ, LK, DIAG, qk_scale, LQ_BUCKET, LK_BUCKET,  # *_BUCKET only feed the autotune key
    CAUSAL: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
    BLOCK_DK: tl.constexpr, BLOCK_DV: tl.constexpr,
    DOT_PREC: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # heads on grid axis 0 so the same tile index of every head launches together
    pid_m = tl.program_id(1)
    if CAUSAL:  # last query tiles have the most keys, run them first
        pid_m = tl.num_programs(1) - 1 - pid_m
    pid_bh = tl.program_id(0)
    b = pid_bh // HQ
    h = pid_bh % HQ
    hk = h // GROUP

    m0 = pid_m * BLOCK_M
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_dk = tl.arange(0, BLOCK_DK)
    offs_dv = tl.arange(0, BLOCK_DV)
    m_ok = offs_m < LQ

    q = tl.load(
        Q + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_dk[None, :] * sqd,
        mask=m_ok[:, None] & (offs_dk[None, :] < DK), other=0.0,
    )
    k_base = K + b * skb + hk * skh
    v_base = V + b * svb + hk * svh

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], tl.float32)

    if CAUSAL:
        hi = tl.maximum(tl.minimum(LK, m0 + BLOCK_M + DIAG), 0)
        # key tiles fully below the diagonal for every row here don't need a mask
        full = tl.minimum(tl.maximum(m0 + DIAG + 1, 0), LK)
    else:
        hi = LK
        full = LK
    full = (full // BLOCK_N) * BLOCK_N  # also excludes the ragged last key tile

    acc, l_i, m_i = _fwd_inner(
        acc, l_i, m_i, q, k_base, v_base, skn, skd, svn, svd, offs_m, offs_dk, offs_dv,
        0, full, LK, DIAG, qk_scale, False, CAUSAL, DK, DV, DOT_PREC, BLOCK_N,
    )
    acc, l_i, m_i = _fwd_inner(
        acc, l_i, m_i, q, k_base, v_base, skn, skd, svn, svd, offs_m, offs_dk, offs_dv,
        full, hi, LK, DIAG, qk_scale, True, CAUSAL, DK, DV, DOT_PREC, BLOCK_N,
    )

    empty = l_i == 0.0  # fully masked row
    l_safe = tl.where(empty, 1.0, l_i)
    out = acc / l_safe[:, None]
    lse = tl.where(empty, float("-inf"), (tl.where(empty, 0.0, m_i) + tl.math.log2(l_safe)) * _LN2)
    tl.store(
        O + b * sob + h * soh + offs_m[:, None] * som + offs_dv[None, :] * sod,
        out.to(O.dtype.element_ty), mask=m_ok[:, None] & (offs_dv[None, :] < DV),
    )
    tl.store(LSE + pid_bh * LQ + offs_m, lse, mask=m_ok)

@triton.jit
def _bwd_pre_kernel(
    O, DO, DLSE, DELTA,
    sob, soh, som, sod,
    sdob, sdoh, sdom, sdod,
    HQ, LQ,
    HAS_DLSE: tl.constexpr, DV: tl.constexpr, BLOCK_DV: tl.constexpr, BLOCK_M: tl.constexpr,
):
    # delta = rowsum(dO * O) - dlse, fp32
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // HQ
    h = pid_bh % HQ
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DV)
    ok = (offs_m[:, None] < LQ) & (offs_d[None, :] < DV)
    o = tl.load(O + b * sob + h * soh + offs_m[:, None] * som + offs_d[None, :] * sod, mask=ok, other=0.0)
    do = tl.load(DO + b * sdob + h * sdoh + offs_m[:, None] * sdom + offs_d[None, :] * sdod, mask=ok, other=0.0)
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), 1)
    if HAS_DLSE:
        delta -= tl.load(DLSE + pid_bh * LQ + offs_m, mask=offs_m < LQ, other=0.0)
    tl.store(DELTA + pid_bh * LQ + offs_m, delta, mask=offs_m < LQ)

@triton.jit
def _dkdv_inner(
    dk, dv, k, v, Q, DO, LSE, DELTA, DQACC, b, hk, offs_n, n_ok, offs_dk, offs_dv,
    sqb, sqh, sqm, sqd, sdob, sdoh, sdom, sdod,
    first_m, n_tiles, HQ, GROUP, LQ, DIAG, scale, qk_scale,
    MASKED: tl.constexpr, ATOMIC_DQ: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
    DOT_PREC: tl.constexpr, BLOCK_M: tl.constexpr,
):
    # single flat loop over (head in group, query tile). nesting a loop over the group crashed
    # Triton 3.4's compiler on B200 (TritonGPUHoistTMEMAlloc)
    for it in range(0, GROUP * n_tiles):
        h = hk * GROUP + it // n_tiles
        start_m = first_m + (it % n_tiles) * BLOCK_M
        row_base = (b * HQ + h) * LQ
        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_ok = offs_m < LQ
        q = tl.load(
            Q + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_dk[None, :] * sqd,
            mask=m_ok[:, None] & (offs_dk[None, :] < DK), other=0.0,
        )
        do = tl.load(
            DO + b * sdob + h * sdoh + offs_m[:, None] * sdom + offs_dv[None, :] * sdod,
            mask=m_ok[:, None] & (offs_dv[None, :] < DV), other=0.0,
        )
        # rows past LQ load zeros and add nothing; rows for keys past LK are never stored
        lse2 = tl.load(LSE + row_base + offs_m, mask=m_ok, other=0.0) * _LOG2E
        lse2 = tl.where(lse2 == float("-inf"), 0.0, lse2)
        delta = tl.load(DELTA + row_base + offs_m, mask=m_ok, other=0.0)

        s = tl.dot(q, tl.trans(k), input_precision=DOT_PREC) * qk_scale
        p = tl.math.exp2(s - lse2[:, None])
        if MASKED:
            p = tl.where(offs_n[None, :] <= offs_m[:, None] + DIAG, p, 0.0)
        if ATOMIC_DQ:  # dQ sums over keys, zero the ones past LK
            p = tl.where(n_ok[None, :], p, 0.0)

        dv += tl.dot(tl.trans(p).to(do.dtype), do, input_precision=DOT_PREC)
        dp = tl.dot(do, tl.trans(v), input_precision=DOT_PREC)
        ds = p * (dp - delta[:, None])
        dk += tl.dot(tl.trans(ds).to(q.dtype), q, input_precision=DOT_PREC)
        if ATOMIC_DQ:
            dq = tl.dot(ds.to(k.dtype), k, input_precision=DOT_PREC) * scale
            tl.atomic_add(
                DQACC + (row_base + offs_m[:, None]) * DK + offs_dk[None, :], dq,
                mask=m_ok[:, None] & (offs_dk[None, :] < DK), sem="relaxed",
            )
    return dk, dv

@triton.autotune(configs=_DKDV_CONFIGS, key=_KEY + ["ATOMIC_DQ"], reset_to_zero=["DQACC"])
@triton.jit
def _bwd_dkdv_kernel(
    Q, K, V, DO, LSE, DELTA, DKO, DVO, DQACC,
    sqb, sqh, sqm, sqd,
    skb, skh, skn, skd,
    svb, svh, svn, svd,
    sdob, sdoh, sdom, sdod,
    sdkb, sdkh, sdkn, sdkd,
    sdvb, sdvh, sdvn, sdvd,
    HQ, HKV, GROUP, LQ, LK, DIAG, scale, qk_scale, LQ_BUCKET, LK_BUCKET,  # *_BUCKET only feed the autotune key
    CAUSAL: tl.constexpr, ATOMIC_DQ: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
    BLOCK_DK: tl.constexpr, BLOCK_DV: tl.constexpr,
    DOT_PREC: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # with causal masking the first key tiles have the most work, so they go first
    pid_n = tl.program_id(1)
    pid_bh = tl.program_id(0)
    b = pid_bh // HKV
    hk = pid_bh % HKV

    n0 = pid_n * BLOCK_N
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_dk = tl.arange(0, BLOCK_DK)
    offs_dv = tl.arange(0, BLOCK_DV)
    n_ok = offs_n < LK

    k = tl.load(
        K + b * skb + hk * skh + offs_n[:, None] * skn + offs_dk[None, :] * skd,
        mask=n_ok[:, None] & (offs_dk[None, :] < DK), other=0.0,
    )
    v = tl.load(
        V + b * svb + hk * svh + offs_n[:, None] * svn + offs_dv[None, :] * svd,
        mask=n_ok[:, None] & (offs_dv[None, :] < DV), other=0.0,
    )
    dk = tl.zeros([BLOCK_N, BLOCK_DK], tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DV], tl.float32)

    if CAUSAL:
        # first query tile that can see this key tile
        lo = (tl.maximum(n0 - DIAG, 0) // BLOCK_M) * BLOCK_M
        # from here on every query sees every key in the tile (no mask)
        full = tl.cdiv(tl.maximum(n0 + BLOCK_N - 1 - DIAG, 0), BLOCK_M) * BLOCK_M
        full = tl.maximum(full, lo)
        n_masked = tl.cdiv(tl.maximum(tl.minimum(full, LQ) - lo, 0), BLOCK_M)
        dk, dv = _dkdv_inner(
            dk, dv, k, v, Q, DO, LSE, DELTA, DQACC, b, hk, offs_n, n_ok, offs_dk, offs_dv,
            sqb, sqh, sqm, sqd, sdob, sdoh, sdom, sdod,
            lo, n_masked, HQ, GROUP, LQ, DIAG, scale, qk_scale, True, ATOMIC_DQ, DK, DV, DOT_PREC, BLOCK_M,
        )
        n_full = tl.cdiv(tl.maximum(LQ - full, 0), BLOCK_M)
    else:
        full = 0
        n_full = tl.cdiv(LQ, BLOCK_M)
    dk, dv = _dkdv_inner(
        dk, dv, k, v, Q, DO, LSE, DELTA, DQACC, b, hk, offs_n, n_ok, offs_dk, offs_dv,
        sqb, sqh, sqm, sqd, sdob, sdoh, sdom, sdod,
        full, n_full, HQ, GROUP, LQ, DIAG, scale, qk_scale, False, ATOMIC_DQ, DK, DV, DOT_PREC, BLOCK_M,
    )

    dk = dk * scale
    tl.store(
        DKO + b * sdkb + hk * sdkh + offs_n[:, None] * sdkn + offs_dk[None, :] * sdkd,
        dk.to(DKO.dtype.element_ty), mask=n_ok[:, None] & (offs_dk[None, :] < DK),
    )
    tl.store(
        DVO + b * sdvb + hk * sdvh + offs_n[:, None] * sdvn + offs_dv[None, :] * sdvd,
        dv.to(DVO.dtype.element_ty), mask=n_ok[:, None] & (offs_dv[None, :] < DV),
    )

@triton.jit
def _dq_inner(
    dq, q, do, lse2, delta, k_base, v_base, skn, skd, svn, svd,
    offs_m, offs_dk, offs_dv, start, end, LK, DIAG, qk_scale,
    MASKED: tl.constexpr, CAUSAL: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
    DOT_PREC: tl.constexpr, BLOCK_N: tl.constexpr,
):
    for start_n in range(start, end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        if MASKED:
            n_ok = offs_n < LK
            k = tl.load(k_base + offs_n[:, None] * skn + offs_dk[None, :] * skd,
                        mask=n_ok[:, None] & (offs_dk[None, :] < DK), other=0.0)
            v = tl.load(v_base + offs_n[:, None] * svn + offs_dv[None, :] * svd,
                        mask=n_ok[:, None] & (offs_dv[None, :] < DV), other=0.0)
        else:
            k = tl.load(k_base + offs_n[:, None] * skn + offs_dk[None, :] * skd,
                        mask=offs_dk[None, :] < DK, other=0.0)
            v = tl.load(v_base + offs_n[:, None] * svn + offs_dv[None, :] * svd,
                        mask=offs_dv[None, :] < DV, other=0.0)
        s = tl.dot(q, tl.trans(k), input_precision=DOT_PREC) * qk_scale
        p = tl.math.exp2(s - lse2[:, None])
        if MASKED:  # dQ sums over keys, zero the ones past LK
            valid = n_ok[None, :]
            if CAUSAL:
                valid = valid & (offs_n[None, :] <= offs_m[:, None] + DIAG)
            p = tl.where(valid, p, 0.0)
        dp = tl.dot(do, tl.trans(v), input_precision=DOT_PREC)
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k, input_precision=DOT_PREC)
    return dq

@triton.autotune(configs=_DQ_CONFIGS, key=_KEY)
@triton.jit
def _bwd_dq_kernel(
    Q, K, V, DO, LSE, DELTA, DQO,
    sqb, sqh, sqm, sqd,
    skb, skh, skn, skd,
    svb, svh, svn, svd,
    sdob, sdoh, sdom, sdod,
    sdqb, sdqh, sdqm, sdqd,
    HQ, GROUP, LQ, LK, DIAG, scale, qk_scale, LQ_BUCKET, LK_BUCKET,  # *_BUCKET only feed the autotune key
    CAUSAL: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
    BLOCK_DK: tl.constexpr, BLOCK_DV: tl.constexpr,
    DOT_PREC: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # heads on grid axis 0 so the same tile index of every head launches together
    pid_m = tl.program_id(1)
    if CAUSAL:  # last query tiles have the most keys, run them first
        pid_m = tl.num_programs(1) - 1 - pid_m
    pid_bh = tl.program_id(0)
    b = pid_bh // HQ
    h = pid_bh % HQ
    hk = h // GROUP

    m0 = pid_m * BLOCK_M
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_dk = tl.arange(0, BLOCK_DK)
    offs_dv = tl.arange(0, BLOCK_DV)
    m_ok = offs_m < LQ

    q = tl.load(
        Q + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_dk[None, :] * sqd,
        mask=m_ok[:, None] & (offs_dk[None, :] < DK), other=0.0,
    )
    do = tl.load(
        DO + b * sdob + h * sdoh + offs_m[:, None] * sdom + offs_dv[None, :] * sdod,
        mask=m_ok[:, None] & (offs_dv[None, :] < DV), other=0.0,
    )
    lse2 = tl.load(LSE + pid_bh * LQ + offs_m, mask=m_ok, other=0.0) * _LOG2E
    lse2 = tl.where(lse2 == float("-inf"), 0.0, lse2)
    delta = tl.load(DELTA + pid_bh * LQ + offs_m, mask=m_ok, other=0.0)
    k_base = K + b * skb + hk * skh
    v_base = V + b * svb + hk * svh
    dq = tl.zeros([BLOCK_M, BLOCK_DK], tl.float32)

    if CAUSAL:
        hi = tl.maximum(tl.minimum(LK, m0 + BLOCK_M + DIAG), 0)
        full = tl.minimum(tl.maximum(m0 + DIAG + 1, 0), LK)
    else:
        hi = LK
        full = LK
    full = (full // BLOCK_N) * BLOCK_N

    dq = _dq_inner(dq, q, do, lse2, delta, k_base, v_base, skn, skd, svn, svd, offs_m, offs_dk, offs_dv,
                   0, full, LK, DIAG, qk_scale, False, CAUSAL, DK, DV, DOT_PREC, BLOCK_N)
    dq = _dq_inner(dq, q, do, lse2, delta, k_base, v_base, skn, skd, svn, svd, offs_m, offs_dk, offs_dv,
                   full, hi, LK, DIAG, qk_scale, True, CAUSAL, DK, DV, DOT_PREC, BLOCK_N)

    dq = dq * scale
    tl.store(
        DQO + b * sdqb + h * sdqh + offs_m[:, None] * sdqm + offs_dk[None, :] * sdqd,
        dq.to(DQO.dtype.element_ty), mask=m_ok[:, None] & (offs_dk[None, :] < DK),
    )

def _common(q, k, v):
    B, HQ, LQ, DK = q.shape
    HKV, LK, DV = k.shape[1], k.shape[2], v.shape[3]
    consts = dict(
        DK=DK,
        DV=DV,
        BLOCK_DK=max(16, triton.next_power_of_2(DK)),
        BLOCK_DV=max(16, triton.next_power_of_2(DV)),
        DOT_PREC="ieee" if q.dtype == torch.float32 else "tf32",  # no tf32 for fp32 inputs
    )
    buckets = (triton.next_power_of_2(LQ), triton.next_power_of_2(LK))
    return B, HQ, HKV, LQ, LK, DK, DV, consts, buckets

def triton_forward(q, k, v, scale: float, causal: bool, diag: int):
    # returns (out in q.dtype, lse fp32 in natural log)
    B, HQ, HKV, LQ, LK, DK, DV, consts, buckets = _common(q, k, v)
    o = torch.empty(B, HQ, LQ, DV, dtype=q.dtype, device=q.device)
    lse = torch.empty(B, HQ, LQ, dtype=torch.float32, device=q.device)
    grid = lambda meta: (B * HQ, triton.cdiv(LQ, meta["BLOCK_M"]))
    _fwd_kernel[grid](
        q, k, v, o, lse, *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        HQ, HQ // HKV, LQ, LK, diag, scale * LOG2E, *buckets, CAUSAL=causal, **consts,
    )
    return o, lse

def triton_backward(do, q, k, v, o, lse, scale: float, causal: bool, diag: int, dlse=None, atomic_dq=None):
    # atomic_dq=None -> use BACKWARD_MODE
    atomic = (BACKWARD_MODE == "atomic") if atomic_dq is None else bool(atomic_dq)
    B, HQ, HKV, LQ, LK, DK, DV, consts, buckets = _common(q, k, v)
    delta = torch.empty(B, HQ, LQ, dtype=torch.float32, device=q.device)
    has_dlse = dlse is not None
    dlse_arg = dlse.float().contiguous() if has_dlse else delta  # unused when has_dlse is False
    _bwd_pre_kernel[(triton.cdiv(LQ, 64), B * HQ)](
        o, do, dlse_arg, delta, *o.stride(), *do.stride(), HQ, LQ,
        HAS_DLSE=has_dlse, DV=DV, BLOCK_DV=consts["BLOCK_DV"], BLOCK_M=64, num_warps=4,
    )
    dk = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    dv = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    # fp32 dQ buffer for the atomic path. otherwise a dummy, the autotuner zeroes it between trials
    dq_acc = torch.zeros(B, HQ, LQ, DK, dtype=torch.float32, device=q.device) if atomic else \
        torch.zeros(1, dtype=torch.float32, device=q.device)
    ins = (*q.stride(), *k.stride(), *v.stride(), *do.stride())
    grid_kv = lambda meta: (B * HKV, triton.cdiv(LK, meta["BLOCK_N"]))
    _bwd_dkdv_kernel[grid_kv](
        q, k, v, do, lse, delta, dk, dv, dq_acc, *ins, *dk.stride(), *dv.stride(),
        HQ, HKV, HQ // HKV, LQ, LK, diag, scale, scale * LOG2E, *buckets, CAUSAL=causal,
        ATOMIC_DQ=atomic, **consts,
    )
    if atomic:
        return dq_acc.to(q.dtype), dk, dv
    dq = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    grid_q = lambda meta: (B * HQ, triton.cdiv(LQ, meta["BLOCK_M"]))
    _bwd_dq_kernel[grid_q](
        q, k, v, do, lse, delta, dq, *ins, *dq.stride(),
        HQ, HQ // HKV, LQ, LK, diag, scale, scale * LOG2E, *buckets, CAUSAL=causal, **consts,
    )
    return dq, dk, dv

def tuned_configs() -> dict:
    """Autotuner picks so far, per kernel and key."""
    out = {}
    for name, kern in (("fwd", _fwd_kernel), ("bwd_dkdv", _bwd_dkdv_kernel), ("bwd_dq", _bwd_dq_kernel)):
        out[name] = {str(key): str(cfg) for key, cfg in getattr(kern, "cache", {}).items()}
    return out

def last_configs() -> dict:
    """Config each kernel used on its last launch."""
    return {
        name: (str(kern.best_config) if getattr(kern, "best_config", None) is not None else None)
        for name, kern in (("fwd", _fwd_kernel), ("bwd_dkdv", _bwd_dkdv_kernel), ("bwd_dq", _bwd_dq_kernel))
    }

class TritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, causal, diag):
        o, lse = triton_forward(q, k, v, scale, causal, diag)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.cfg = (scale, causal, diag)
        ctx.atomic_dq = BACKWARD_MODE == "atomic"  # fixed when the graph is built
        ctx.set_materialize_grads(False)  # unused lse grad comes in as None
        return o, lse

    @staticmethod
    def backward(ctx, do, dlse):
        q, k, v, o, lse = ctx.saved_tensors
        if do is None:
            do = torch.zeros_like(o)
        dq, dk, dv = triton_backward(do, q, k, v, o, lse, *ctx.cfg, dlse=dlse, atomic_dq=ctx.atomic_dq)
        return dq, dk, dv, None, None, None
