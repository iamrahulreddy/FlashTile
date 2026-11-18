# FlashTile inside a small random Qwen2 (GQA + RoPE), no download needed.
# Covers prefill, cached decode, padding masks and gradients. The real-model run is
# benchmark/llm_eval.py.

import pytest
import torch

transformers = pytest.importorskip("transformers")

from flashtile import HAS_TRITON  # noqa: E402
import flashtile.hf as fhf  # noqa: E402

FT = fhf.NAME  # the name FlashTile is registered under in transformers

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not HAS_TRITON, reason="needs triton"),
]

@pytest.fixture(scope="module")
def model():
    torch.backends.cuda.matmul.allow_tf32 = False  # fp32 projections stay fp32 in both models
    fhf.register()
    cfg = transformers.Qwen2Config(
        vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
        num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=1024,
    )
    torch.manual_seed(0)
    m = transformers.Qwen2ForCausalLM(cfg).to("cuda", torch.float32).eval()
    return m

def _logits(model, impl, ids, **kw):
    model.set_attn_implementation(impl)
    with torch.no_grad():
        return model(ids, **kw).logits

def test_prefill_logits_match_eager(model):
    ids = torch.randint(0, 512, (2, 300), device="cuda", generator=torch.Generator("cuda").manual_seed(1))
    fhf.stats.clear()
    ours = _logits(model, FT, ids)
    ref = _logits(model, "eager", ids)
    assert fhf.stats["causal"] == 2 and fhf.stats["masked"] == 0  # one call per layer, fused path
    torch.testing.assert_close(ours, ref, rtol=0, atol=2e-4)

def test_cached_greedy_decode_matches_eager(model):
    ids = torch.randint(0, 512, (1, 40), device="cuda", generator=torch.Generator("cuda").manual_seed(2))
    outs = {}
    for impl in (FT, "eager"):
        model.set_attn_implementation(impl)
        fhf.stats.clear()
        outs[impl] = model.generate(ids, max_new_tokens=30, do_sample=False, min_new_tokens=30, pad_token_id=0)
        if impl == FT:
            assert fhf.stats["decode"] == 2 * 29  # every decode step after prefill, per layer
    assert torch.equal(outs[FT], outs["eager"])

def test_padding_mask_uses_tiled_path_and_matches(model):
    ids = torch.randint(0, 512, (2, 64), device="cuda", generator=torch.Generator("cuda").manual_seed(3))
    am = torch.ones(2, 64, dtype=torch.long, device="cuda")
    am[1, :10] = 0  # left padding on the second sequence
    fhf.stats.clear()
    ours = _logits(model, FT, ids, attention_mask=am)
    assert fhf.stats["masked"] == 2
    ref = _logits(model, "eager", ids, attention_mask=am)
    keep = am.bool()
    torch.testing.assert_close(ours[keep], ref[keep], rtol=0, atol=2e-4)

def test_training_gradients_match_eager(model):
    ids = torch.randint(0, 512, (2, 200), device="cuda", generator=torch.Generator("cuda").manual_seed(4))
    grads = {}
    for impl in (FT, "eager"):
        model.set_attn_implementation(impl)
        model.zero_grad(set_to_none=True)
        model(ids, labels=ids).loss.backward()
        grads[impl] = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    for name, g in grads["eager"].items():
        scale = max(g.abs().max().item(), 1e-6)
        err = (grads[FT][name] - g).abs().max().item()
        assert err <= 1e-3 * scale, f"{name}: {err:.3g} vs grad scale {scale:.3g}"
