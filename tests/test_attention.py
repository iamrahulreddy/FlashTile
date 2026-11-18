# CPU tests in float64. Every parity case spans at least three tiles with a ragged tail.

import math
import pytest
import torch

from flashtile import attention, attention_reference
from flashtile.contract import validate
from flashtile.tiled import tiled_forward

F64 = torch.float64

def make_inputs(B, Hq, Hkv, Lq, Lk, d, dv, *, seed=0, dtype=F64, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, Hq, Lq, d, generator=g, dtype=F64) * 1.5
    k = torch.randn(B, Hkv, Lk, d, generator=g, dtype=F64) * 1.5
    v = torch.randn(B, Hkv, Lk, dv, generator=g, dtype=F64)
    return [t.to(device=device, dtype=dtype) for t in (q, k, v)]

def random_mask(B, Hq, Lq, Lk, *, seed=1, empty_rows=(), device="cpu"):
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand(B, Hq, Lq, Lk, generator=g) > 0.3
    for b, h, i in empty_rows:
        mask[b, h, i, :] = False
    return mask.to(device)

# (B, Hq, Hkv, Lq, Lk, d, dv, block_q, block_k, causal, masked, positions)
CASES = {
    "mha": (2, 4, 4, 37, 37, 8, 8, 16, 16, False, False, None),
    "mha_causal": (2, 4, 4, 37, 37, 8, 8, 16, 16, True, False, None),
    "gqa4_causal_mask_dv": (1, 8, 2, 50, 50, 16, 12, 8, 13, True, True, None),
    "mqa_rect_mask": (2, 6, 1, 33, 47, 8, 8, 10, 7, False, True, None),
    "decode_like": (1, 4, 2, 5, 41, 8, 8, 2, 6, True, False, "suffix"),
    "chunked_prefill": (1, 2, 1, 19, 40, 4, 4, 4, 5, True, True, "suffix"),
    "block_bigger_than_seq": (1, 2, 2, 9, 9, 4, 4, 64, 64, True, False, None),
    "block_size_one": (1, 2, 1, 6, 7, 3, 2, 1, 1, False, True, None),
}

def build(case, seed=0, dtype=F64, device="cpu"):
    B, Hq, Hkv, Lq, Lk, d, dv, bq, bk, causal, masked, positions = CASES[case]
    q, k, v = make_inputs(B, Hq, Hkv, Lq, Lk, d, dv, seed=seed, dtype=dtype, device=device)
    kwargs = {"causal": causal}
    if masked:
        kwargs["allow_mask"] = random_mask(B, Hq, Lq, Lk, seed=seed + 1, empty_rows=[(0, 0, 2)], device=device)
    if positions == "suffix":  # queries are the last Lq positions of an Lk-long sequence
        kwargs["q_pos"] = torch.arange(Lk - Lq, Lk, device=device)
        kwargs["k_pos"] = torch.arange(Lk, device=device)
    return q, k, v, kwargs, bq, bk

def test_reference_hand_example():
    # worked by hand: q = 1, keys [0, log 2], values [2, 8], scale 1 -> O = 6, LSE = log 3
    q = torch.tensor([[[[1.0]]]], dtype=F64)
    k = torch.tensor([[[[0.0], [math.log(2.0)]]]], dtype=F64)
    v = torch.tensor([[[[2.0], [8.0]]]], dtype=F64)
    out, lse = attention_reference(q, k, v, scale=1.0)
    assert out.item() == pytest.approx(6.0, abs=1e-12)
    assert lse.item() == pytest.approx(math.log(3.0), abs=1e-12)
    out_t, lse_t = attention(q, k, v, scale=1.0, return_lse=True)
    assert out_t.item() == pytest.approx(6.0, abs=1e-12)
    assert lse_t.item() == pytest.approx(math.log(3.0), abs=1e-12)

def test_reference_matches_framework_softmax():
    q, k, v = make_inputs(2, 3, 3, 9, 11, 5, 4)
    mask = random_mask(2, 3, 9, 11)
    mask[..., 0] = True  # keep every row nonempty so torch.softmax is defined
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(5)
    expected = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1) @ v
    out, _ = attention_reference(q, k, v, allow_mask=mask)
    torch.testing.assert_close(out, expected, rtol=0, atol=1e-12)

@pytest.mark.parametrize("case", list(CASES))
def test_forward_matches_reference(case):
    q, k, v, kwargs, bq, bk = build(case)
    ref_out, ref_lse = attention_reference(q, k, v, **kwargs)
    out, lse = attention(q, k, v, block_q=bq, block_k=bk, return_lse=True, **kwargs)
    torch.testing.assert_close(out, ref_out, rtol=0, atol=1e-12)
    torch.testing.assert_close(lse, ref_lse, rtol=0, atol=1e-12)

@pytest.mark.parametrize("blocks", [(1, 1), (3, 5), (7, 2), (16, 16), (100, 100)])
def test_result_does_not_depend_on_tiling(blocks):
    q, k, v, kwargs, _, _ = build("gqa4_causal_mask_dv")
    base = attention(q, k, v, block_q=50, block_k=50, **kwargs)
    out = attention(q, k, v, block_q=blocks[0], block_k=blocks[1], **kwargs)
    torch.testing.assert_close(out, base, rtol=0, atol=1e-12)

def test_large_score_shift_is_stable():
    # scores near 1e3 overflow exp() unless every tile is rescaled against the running max
    q, k, v = make_inputs(1, 2, 2, 20, 20, 4, 4)
    q = q * 40
    ref, _ = attention_reference(q, k, v, causal=True)
    out = attention(q, k, v, causal=True, block_q=3, block_k=3)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, rtol=0, atol=1e-10)

def test_noncontiguous_inputs():
    g = torch.Generator().manual_seed(3)
    q = torch.randn(2, 30, 4, 8, generator=g, dtype=F64).transpose(1, 2)  # [B, H, L, d] view
    k = torch.randn(2, 30, 2, 8, generator=g, dtype=F64).transpose(1, 2)
    v = torch.randn(2, 30, 2, 6, generator=g, dtype=F64).transpose(1, 2)
    assert not q.is_contiguous()
    ref, _ = attention_reference(q, k, v, causal=True)
    torch.testing.assert_close(attention(q, k, v, causal=True, block_q=7, block_k=9), ref, rtol=0, atol=1e-12)

def test_causal_tiles_are_skipped():
    q, k, v = make_inputs(1, 1, 1, 64, 64, 4, 4)
    p, mask, qp, kp = validate(q, k, v, scale=None, causal=True, allow_mask=None, q_pos=None, k_pos=None)
    _, _, tiles = tiled_forward(q, k, v, p, mask, qp, kp, True, 16, 16)
    assert tiles == 4 * 5 // 2  # lower triangle of a 4 x 4 tile grid
    _, _, tiles = tiled_forward(q, k, v, p, mask, qp, kp, False, 16, 16)
    assert tiles == 16

def test_fully_masked_row_contract():
    q, k, v = make_inputs(1, 4, 2, 12, 12, 4, 3)
    q, k, v = (t.requires_grad_() for t in (q, k, v))
    mask = torch.ones(1, 4, 12, 12, dtype=torch.bool)
    mask[0, 1, 5, :] = False
    out, lse = attention(q, k, v, allow_mask=mask, block_q=4, block_k=5, return_lse=True)
    assert torch.all(out[0, 1, 5] == 0)
    assert torch.isneginf(lse[0, 1, 5])
    assert torch.isfinite(lse[0, 1, :5]).all() and torch.isfinite(lse[0, 1, 6:]).all()
    (out * torch.randn_like(out)).sum().backward()
    assert torch.all(q.grad[0, 1, 5] == 0)
    assert all(torch.isfinite(t.grad).all() for t in (q, k, v))

def test_every_row_masked():
    q, k, v = make_inputs(1, 2, 2, 7, 7, 4, 4)
    q.requires_grad_()
    mask = torch.zeros(7, 7, dtype=torch.bool)
    out, lse = attention(q, k, v, allow_mask=mask, block_q=3, block_k=3, return_lse=True)
    assert torch.all(out == 0) and torch.isneginf(lse).all()
    out.sum().backward()
    assert torch.all(q.grad == 0)

@pytest.mark.parametrize("case", list(CASES))
def test_gradients_match_reference(case):
    q, k, v, kwargs, bq, bk = build(case, seed=5)
    upstream = torch.randn(q.shape[:-1] + (v.shape[-1],), dtype=F64, generator=torch.Generator().manual_seed(9))

    def grads(fn):
        leaves = [t.detach().clone().requires_grad_() for t in (q, k, v)]
        out = fn(*leaves)
        (out * upstream).sum().backward()  # nonuniform loss: out.sum() hides errors
        return [t.grad for t in leaves]

    ref = grads(lambda a, b, c: attention_reference(a, b, c, **kwargs)[0])
    got = grads(lambda a, b, c: attention(a, b, c, block_q=bq, block_k=bk, **kwargs))
    for name, r, g in zip("qkv", ref, got):
        torch.testing.assert_close(g, r, rtol=0, atol=1e-10, msg=lambda m: f"d{name}: {m}")

@pytest.mark.parametrize("causal", [False, True])
def test_gradcheck_finite_differences(causal):
    q, k, v = make_inputs(1, 4, 2, 10, 10, 3, 2, seed=11)
    mask = random_mask(1, 4, 10, 10, seed=12, empty_rows=[(0, 3, 4)])
    inputs = tuple(t.requires_grad_() for t in (q, k, v))
    fn = lambda a, b, c: attention(a, b, c, causal=causal, allow_mask=mask, block_q=3, block_k=4)
    assert torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-7, rtol=1e-5)

@pytest.mark.parametrize("case", list(CASES))
def test_lse_gradients_match_reference(case):
    """A loss that uses both the output and the lse (as in cross-entropy over attention logits)."""
    q, k, v, kwargs, bq, bk = build(case, seed=6)
    g = torch.Generator().manual_seed(10)
    up_out = torch.randn(q.shape[:-1] + (v.shape[-1],), dtype=F64, generator=g)
    up_lse = torch.randn(q.shape[:-1], dtype=F64, generator=g)

    def grads(fn):
        leaves = [t.detach().clone().requires_grad_() for t in (q, k, v)]
        out, lse = fn(*leaves)
        finite_lse = torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))  # empty rows: lse = -inf
        ((out * up_out).sum() + (finite_lse * up_lse).sum()).backward()
        return [t.grad for t in leaves]

    ref = grads(lambda a, b, c: attention_reference(a, b, c, **kwargs))
    got = grads(lambda a, b, c: attention(a, b, c, block_q=bq, block_k=bk, return_lse=True, **kwargs))
    for name, r, gr in zip("qkv", ref, got):
        torch.testing.assert_close(gr, r, rtol=0, atol=1e-10, msg=lambda m: f"d{name}: {m}")

def test_lse_only_loss_gradcheck():
    q, k, v = make_inputs(1, 4, 2, 9, 9, 3, 2, seed=13)
    mask = random_mask(1, 4, 9, 9, seed=14, empty_rows=[(0, 2, 3)])
    inputs = tuple(t.requires_grad_() for t in (q, k, v))

    def fn(a, b, c):
        _, lse = attention(a, b, c, causal=True, allow_mask=mask, block_q=4, block_k=3, return_lse=True)
        return torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))

    assert torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-7, rtol=1e-5)

@pytest.mark.parametrize(
    "shapes, kwargs, err",
    [
        (((1, 3, 4, 2), (1, 2, 4, 2), (1, 2, 4, 2)), {}, ValueError),  # Hq not a multiple of Hkv
        (((1, 2, 4, 2), (1, 2, 5, 2), (1, 2, 5, 2)), {"causal": True}, ValueError),  # no positions
        (((1, 2, 4, 2), (1, 2, 4, 3), (1, 2, 4, 2)), {}, ValueError),  # q/k width mismatch
        (((1, 2, 4, 2), (1, 2, 4, 2), (1, 2, 4, 2)), {"scale": 0.0}, ValueError),
        (((1, 2, 4, 2), (1, 2, 4, 2), (1, 2, 4, 2)), {"allow_mask": torch.ones(4, 4)}, TypeError),
        (((1, 2, 4, 2), (1, 2, 4, 2), (1, 2, 4, 2)), {"allow_mask": torch.ones(3, 4, dtype=torch.bool)}, ValueError),
        (((1, 2, 0, 2), (1, 2, 4, 2), (1, 2, 4, 2)), {}, ValueError),  # empty sequence
    ],
)

def test_contract_errors(shapes, kwargs, err):
    q, k, v = (torch.randn(*s, dtype=F64) for s in shapes)
    with pytest.raises(err):
        attention(q, k, v, **kwargs)
    with pytest.raises(err):
        attention_reference(q, k, v, **kwargs)
