# GPU tests for the tiled path: low-precision accuracy and peak memory.

import pytest
import torch

from flashtile import attention, attention_reference

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]

def inputs(B, Hq, Hkv, L, d, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    mk = lambda *s: torch.randn(*s, device="cuda", generator=g, dtype=torch.float32).to(dtype)
    return mk(B, Hq, L, d), mk(B, Hkv, L, d), mk(B, Hkv, L, d)

@pytest.mark.parametrize("dtype, atol", [(torch.float32, 2e-5), (torch.bfloat16, 2e-2), (torch.float16, 4e-3)])
def test_low_precision_against_fp64_oracle(dtype, atol):
    torch.backends.cuda.matmul.allow_tf32 = False
    q, k, v = inputs(2, 8, 2, 300, 64, dtype)
    upstream = torch.randn(2, 8, 300, 64, device="cuda")

    leaves = [t.clone().requires_grad_() for t in (q, k, v)]
    out = attention(*leaves, causal=True, block_q=64, block_k=64, backend="tiled")
    (out.float() * upstream).sum().backward()

    ref_leaves = [t.double().requires_grad_() for t in (q, k, v)]
    ref, _ = attention_reference(*ref_leaves, causal=True)
    (ref * upstream.double()).sum().backward()

    assert out.dtype == dtype and all(t.grad.dtype == dtype for t in leaves)
    torch.testing.assert_close(out.double(), ref, rtol=0, atol=atol)
    for name, got, want in zip("qkv", leaves, ref_leaves):
        scale = want.grad.abs().max().item()
        err = (got.grad.double() - want.grad).abs().max().item()
        assert err <= 10 * atol * max(scale, 1.0), f"d{name}: max err {err:.3g} vs grad scale {scale:.3g}"

def _peak_mib(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20

def test_training_step_memory_is_not_quadratic():
    """Forward + backward at L = 4096: the tiled path must never hold an L x L tensor."""
    B, H, L, d = 1, 8, 4096, 64
    score_table_mib = B * H * L * L * 4 / 2**20  # one fp32 [B, H, L, L] tensor = 512 MiB
    q, k, v = inputs(B, H, H, L, d, torch.float32)
    upstream = torch.randn(B, H, L, d, device="cuda")

    def step(fn):
        leaves = [t.clone().requires_grad_() for t in (q, k, v)]
        out = fn(*leaves)
        (out * upstream).sum().backward()

    tiled = _peak_mib(lambda: step(lambda a, b, c: attention(a, b, c, causal=True, block_q=128, block_k=128, backend="tiled")))
    dense = _peak_mib(lambda: step(lambda a, b, c: attention_reference(a, b, c, causal=True)[0]))
    print(
        f"\n[memory] L={L} fwd+bwd peak above inputs: tiled {tiled:.1f} MiB, dense {dense:.1f} MiB "
        f"(one LxL fp32 table = {score_table_mib:.0f} MiB)"
    )
    assert tiled < 0.25 * score_table_mib
    assert dense > 2 * score_table_mib  # sanity check: the oracle really is quadratic

def test_device_and_dtype_reported():
    props = torch.cuda.get_device_properties(0)
    print(
        f"\n[device] {props.name}, sm_{props.major}{props.minor}, {props.total_memory / 2**30:.0f} GiB, "
        f"{props.multi_processor_count} SMs, torch {torch.__version__}, CUDA {torch.version.cuda}"
    )
    # cheap check that this torch build actually has kernels for the GPU
    a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close((a @ a).float(), a.float() @ a.float(), rtol=2e-2, atol=1.0)
