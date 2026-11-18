# Triton kernels vs the fp64 oracle: output, lse and all three gradients.
# Lengths around tile edges, odd head dims, GQA, decode / chunked prefill / empty rows.

import pytest
import torch

from flashtile import HAS_TRITON, attention, attention_reference, select_backend

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not HAS_TRITON, reason="needs triton"),
]

TOL = {torch.float32: 1e-4, torch.bfloat16: 3e-2, torch.float16: 5e-3}

def make(B, Hq, Hkv, Lq, Lk, d, dv, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    mk = lambda *s: torch.randn(*s, device="cuda", generator=g).to(dtype)
    return mk(B, Hq, Lq, d), mk(B, Hkv, Lk, d), mk(B, Hkv, Lk, dv)

def _finite(x):
    return torch.where(torch.isfinite(x), x, torch.zeros_like(x))

def check(q, k, v, dtype, lse_grad=False, **kw):
    # Run triton and the fp64 oracle on the same inputs and compare everything.
    # lse_grad=True also puts the lse in the loss, to test its gradient.

    torch.backends.cuda.matmul.allow_tf32 = False
    tol = TOL[dtype]
    upstream = torch.randn(q.shape[:-1] + (v.shape[-1],), device="cuda", dtype=torch.float32)
    up_lse = torch.randn(q.shape[:-1], device="cuda", dtype=torch.float32) if lse_grad else None

    leaves = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    out, lse = attention(*leaves, backend="triton", return_lse=True, **kw)
    loss = (out.float() * upstream).sum()
    if lse_grad:
        loss = loss + (_finite(lse) * up_lse).sum()
    loss.backward()

    ref_leaves = [t.detach().double().requires_grad_() for t in (q, k, v)]
    ref, ref_lse = attention_reference(*ref_leaves, **kw)
    ref_loss = (ref * upstream.double()).sum()
    if lse_grad:
        ref_loss = ref_loss + (_finite(ref_lse) * up_lse.double()).sum()
    ref_loss.backward()

    assert out.dtype == dtype
    torch.testing.assert_close(out.double(), ref, rtol=0, atol=tol)
    finite = torch.isfinite(ref_lse)
    assert torch.equal(torch.isfinite(lse), finite), "empty-row pattern differs"
    torch.testing.assert_close(lse.double()[finite], ref_lse[finite], rtol=0, atol=tol)
    for name, got, want in zip("qkv", leaves, ref_leaves):
        assert got.grad.dtype == dtype and torch.isfinite(got.grad).all(), f"d{name} not finite"
        scale = max(want.grad.abs().max().item(), 1.0)
        err = (got.grad.double() - want.grad).abs().max().item()
        assert err <= 10 * tol * scale, f"d{name}: max err {err:.3g} (grad scale {scale:.3g})"

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("L", [1, 15, 16, 17, 63, 64, 65, 127, 129, 300])
def test_square_lengths_around_tile_edges(L, causal, dtype):
    check(*make(2, 4, 4, L, L, 64, 64, dtype), dtype, causal=causal)

@pytest.mark.parametrize("d, dv", [(16, 16), (32, 32), (80, 80), (128, 128), (256, 256), (64, 32), (24, 72)])
def test_head_dims(d, dv):
    check(*make(1, 4, 2, 133, 133, d, dv, torch.bfloat16, seed=d), torch.bfloat16, lse_grad=True, causal=True)

@pytest.mark.parametrize("Hq, Hkv", [(8, 8), (8, 4), (8, 1), (32, 8), (12, 3), (14, 2)])
def test_grouped_heads(Hq, Hkv):
    check(*make(2, Hq, Hkv, 150, 150, 64, 64, torch.bfloat16, seed=Hq + Hkv), torch.bfloat16,
          lse_grad=True, causal=True)

@pytest.mark.parametrize(
    "Lq, Lk, q_start, k_start",
    [
        (1, 1000, 999, 0),     # decode: one new token over a long prefix
        (7, 777, 770, 0),      # speculative decode block
        (64, 300, 236, 0),     # chunked prefill
        (100, 100, 0, 30),     # keys start later: first 30 queries see nothing
        (50, 200, 0, 0),       # queries are the first 50 positions of 200 keys
    ],
)

@pytest.mark.parametrize("pos_device", ["cuda", "cpu"])
def test_causal_with_positions(Lq, Lk, q_start, k_start, pos_device):
    q, k, v = make(1, 8, 2, Lq, Lk, 64, 64, torch.bfloat16, seed=Lq)
    kw = dict(
        causal=True,
        q_pos=torch.arange(q_start, q_start + Lq, device=pos_device),
        k_pos=torch.arange(k_start, k_start + Lk, device=pos_device),
    )
    check(q, k, v, torch.bfloat16, lse_grad=True, **kw)

ATOMIC_CASES = {
    # name: (B, Hq, Hkv, Lq, Lk, d, dv, dtype, causal, q_start, k_start)
    "square_causal_ragged": (2, 4, 4, 129, 129, 64, 64, torch.bfloat16, True, None, None),
    "noncausal_rect_gqa": (2, 6, 2, 133, 333, 64, 64, torch.float16, False, None, None),
    "decode": (1, 8, 2, 1, 1000, 64, 64, torch.bfloat16, True, 999, 0),
    "keys_start_later_empty_rows": (1, 8, 2, 100, 100, 64, 64, torch.bfloat16, True, 0, 30),
    "head_dims_80": (1, 4, 2, 133, 133, 80, 80, torch.bfloat16, True, None, None),
    "qwen3.5_shape_d256": (1, 16, 4, 300, 300, 256, 256, torch.bfloat16, True, None, None),
    "fp32_ieee": (1, 4, 2, 200, 200, 64, 64, torch.float32, True, None, None),
}

@pytest.mark.parametrize("case", list(ATOMIC_CASES))
def test_atomic_dq_backward(case, monkeypatch):
    """The single-pass backward (fp32 atomic dQ) against the same oracle as the default path."""
    import flashtile.triton_kernels as tk

    monkeypatch.setattr(tk, "BACKWARD_MODE", "atomic")
    B, Hq, Hkv, Lq, Lk, d, dv, dtype, causal, qs, ks = ATOMIC_CASES[case]
    q, k, v = make(B, Hq, Hkv, Lq, Lk, d, dv, dtype, seed=Lq + d)
    kw = dict(causal=causal)
    if qs is not None:
        kw.update(q_pos=torch.arange(qs, qs + Lq), k_pos=torch.arange(ks, ks + Lk))
    check(q, k, v, dtype, lse_grad=True, **kw)

@pytest.mark.parametrize("Hq, Hkv, d", [(16, 4, 256), (14, 2, 64), (8, 1, 128)])
def test_decode_packs_grouped_heads(Hq, Hkv, d):
    """One query per head (HF decode step, no mask): the group's heads are packed into one query tile."""
    q, k, v = make(2, Hq, Hkv, 1, 777, d, d, torch.bfloat16, seed=Hq + d)
    check(q, k, v, torch.bfloat16, lse_grad=True)

def test_noncausal_ignores_gapped_positions():
    """Regression: positions with gaps on a non-causal call used to crash the triton dispatch."""
    q, k, v = make(1, 4, 2, 40, 40, 32, 32, torch.float16)
    gaps = torch.arange(0, 80, 2, device="cuda")
    assert select_backend(q, k, v, q_pos=gaps, k_pos=gaps)[0] == "triton"
    check(q, k, v, torch.float16, q_pos=gaps, k_pos=gaps)

def test_autotune_reuses_config_within_length_bucket():
    from flashtile.triton_kernels import _fwd_kernel

    q, k, v = make(1, 4, 4, 300, 300, 64, 64, torch.bfloat16)
    attention(q, k, v, causal=True, backend="triton")
    tuned = len(_fwd_kernel.cache)
    q, k, v = make(1, 4, 4, 310, 310, 64, 64, torch.bfloat16)  # same power-of-two bucket (512)
    attention(q, k, v, causal=True, backend="triton")
    assert len(_fwd_kernel.cache) == tuned

@pytest.mark.parametrize("causal", [False, True])
def test_rectangular_noncausal_and_causal(causal):
    kw = dict(causal=causal)
    if causal:
        kw.update(q_pos=torch.arange(200, 333, device="cuda"), k_pos=torch.arange(333, device="cuda"))
    check(*make(2, 6, 2, 133, 333, 64, 64, torch.float16), torch.float16, **kw)

def test_noncontiguous_inputs():
    g = torch.Generator(device="cuda").manual_seed(4)
    q = torch.randn(2, 129, 8, 64, device="cuda", generator=g).to(torch.bfloat16).transpose(1, 2)
    k = torch.randn(2, 129, 2, 64, device="cuda", generator=g).to(torch.bfloat16).transpose(1, 2)
    v = torch.randn(2, 129, 2, 64, device="cuda", generator=g).to(torch.bfloat16).transpose(1, 2)
    check(q, k, v, torch.bfloat16, causal=True)

def test_large_scores_stay_finite():
    q, k, v = make(1, 4, 4, 257, 257, 64, 64, torch.float32)
    check(q * 8, k * 8, v, torch.float32, causal=True)

def test_matches_tiled_backend_bitwise_close():
    q, k, v = make(2, 8, 2, 200, 200, 64, 64, torch.float32)
    a = attention(q, k, v, causal=True, backend="triton")
    b = attention(q, k, v, causal=True, backend="tiled")
    torch.testing.assert_close(a, b, rtol=0, atol=1e-5)

def test_matches_pytorch_sdpa():
    q, k, v = make(2, 8, 2, 512, 512, 128, 128, torch.bfloat16)
    ours = attention(q, k, v, causal=True, backend="triton")
    theirs = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    torch.testing.assert_close(ours.float(), theirs.float(), rtol=0, atol=2e-2)

def test_backend_selection_is_explicit():
    q, k, v = make(1, 2, 2, 32, 32, 16, 16, torch.bfloat16)
    assert select_backend(q, k, v)[0] == "triton"
    mask = torch.ones(32, 32, dtype=torch.bool, device="cuda")
    assert select_backend(q, k, v, allow_mask=mask) == (
        "tiled", "arbitrary Boolean masks are only handled by the tiled path"
    )
    gaps = torch.arange(0, 64, 2, device="cuda")
    assert select_backend(q, k, v, causal=True, q_pos=gaps, k_pos=gaps)[0] == "tiled"
    with pytest.raises(ValueError, match="triton backend cannot run"):
        attention(q, k, v, allow_mask=mask, backend="triton")
    # auto quietly uses tiled for the mask case, and gives the oracle's answer
    ref, _ = attention_reference(q.double(), k.double(), v.double(), allow_mask=mask)
    torch.testing.assert_close(attention(q, k, v, allow_mask=mask).double(), ref, rtol=0, atol=3e-2)

def test_triton_training_step_memory():
    B, H, L, d = 1, 8, 8192, 64
    q, k, v = make(B, H, H, L, L, d, d, torch.bfloat16)
    upstream = torch.randn(B, H, L, d, device="cuda", dtype=torch.bfloat16)
    table = B * H * L * L * 2 / 2**20

    def step():
        leaves = [t.clone().requires_grad_() for t in (q, k, v)]
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        attention(*leaves, causal=True, backend="triton").backward(upstream)
        torch.cuda.synchronize()
        return (torch.cuda.max_memory_allocated() - base) / 2**20

    # first call autotunes, and Triton's benchmarker grabs a 256 MB buffer while it does
    first = step()
    steady = step()
    print(
        f"\n[memory] triton L={L} bf16 fwd+bwd peak above inputs: steady state {steady:.1f} MiB, "
        f"first call incl. autotuning {first:.1f} MiB (one LxL bf16 table {table:.0f} MiB)"
    )
    assert steady < 0.1 * table
