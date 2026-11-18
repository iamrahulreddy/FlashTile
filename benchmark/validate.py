"""Quick sanity run in v1's validation format (backend choice, correctness, memory).

    python benchmark/validate.py results/local/validation
"""

from __future__ import annotations

import json
import pathlib
import platform
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flashtile import HAS_TRITON, attention, attention_reference, select_backend  # noqa: E402


def _peak(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def run(out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    lines, res = [], {}

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    import triton

    props = torch.cuda.get_device_properties(0)
    res["device"] = "cuda"
    res["gpu_name"] = props.name
    res["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
    res["installation"] = {
        "package": "flashtile", "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda, "triton_available": HAS_TRITON, "triton_version": triton.__version__,
        "python": platform.python_version(),
    }
    log(f"Date: {time.strftime('%a %b %d %H:%M:%S UTC %Y', time.gmtime())}")
    log(f"GPU: {props.name}  sm_{props.major}{props.minor}  {props.total_memory / 2**30:.0f} GiB")
    log(f"Python {platform.python_version()}  torch {torch.__version__}  CUDA {torch.version.cuda}  triton {triton.__version__}")
    log("")

    # backend selection
    g = torch.Generator(device="cuda").manual_seed(0)
    mk = lambda *s, dt=torch.bfloat16: torch.randn(*s, device="cuda", generator=g).to(dt)
    q, k, v = mk(2, 8, 512, 64), mk(2, 2, 512, 64), mk(2, 2, 512, 64)
    calls = [
        ("bf16 causal prefill", dict(causal=True), (q, k, v)),
        ("bf16 decode, CPU positions", dict(causal=True, q_pos=torch.arange(511, 512), k_pos=torch.arange(512)),
         (q[:, :, :1], k, v)),
        ("bf16 Boolean mask", dict(allow_mask=torch.ones(512, 512, dtype=torch.bool, device="cuda")), (q, k, v)),
        ("bf16 causal, positions with gaps", dict(causal=True, q_pos=torch.arange(0, 1024, 2), k_pos=torch.arange(0, 1024, 2)), (q, k, v)),
        ("fp64 inputs", dict(causal=True), (q.double(), k.double(), v.double())),
        ("CPU tensors", dict(causal=True), (q.float().cpu(), k.float().cpu(), v.float().cpu())),
    ]
    log("Backend selection (auto):")
    res["backend_selection"] = {}
    for name, kw, (a, b, c) in calls:
        be, why = select_backend(a, b, c, **kw)
        res["backend_selection"][name] = {"backend": be, "reason": why}
        log(f"  {name:<34} -> {be:<7} ({why})")
    log("")

    # forward
    log("Forward pass (B=2, Hq=8, Hkv=2, L=512, d=64, bf16, causal):")
    res["forward_pass"] = {}
    outs = {}
    for name, fn in (
        ("FlashTile triton", lambda: attention(q, k, v, causal=True, backend="triton")),
        ("FlashTile tiled", lambda: attention(q, k, v, causal=True, backend="tiled")),
        ("Reference (fp64 oracle)", lambda: attention_reference(q.double(), k.double(), v.double(), causal=True)[0]),
        ("PyTorch SDPA", lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)),
    ):
        o = fn()
        outs[name] = o
        ok = bool(torch.isfinite(o).all())
        res["forward_pass"][name] = {"status": "OK" if ok else "NONFINITE", "shape": list(o.shape), "finite": ok}
        log(f"  {name:<26} {'OK' if ok else 'FAIL'}   shape={list(o.shape)}")
    log("")

    log("Correctness (vs fp64 oracle, same bf16 inputs):")
    ref = outs["Reference (fp64 oracle)"]
    res["correctness"] = {}
    for name in ("FlashTile triton", "FlashTile tiled", "PyTorch SDPA"):
        d = (outs[name].double() - ref).abs()
        verdict = "PASS" if d.max().item() < 2e-2 else "FAIL"
        res["correctness"][name] = {"max_diff": d.max().item(), "mean_diff": d.mean().item(), "verdict": verdict}
        log(f"  {name:<26} max_diff={d.max().item():.8f}  mean_diff={d.mean().item():.2e}  {verdict}")
    log("")

    log("Backward pass (loss = sum(out * random), gradients vs fp64 oracle):")
    up = torch.randn(2, 8, 512, 64, device="cuda", generator=g)
    leaves_r = [t.double().requires_grad_() for t in (q, k, v)]
    (attention_reference(*leaves_r, causal=True)[0] * up.double()).sum().backward()
    res["backward"] = {}
    for name, be in (("FlashTile triton", "triton"), ("FlashTile tiled", "tiled")):
        leaves = [t.clone().requires_grad_() for t in (q, k, v)]
        (attention(*leaves, causal=True, backend=be).float() * up).sum().backward()
        rel = {n: ((a.grad.double() - b.grad).abs().max() / b.grad.abs().max()).item()
               for n, a, b in zip(("dq", "dk", "dv"), leaves, leaves_r)}
        verdict = "PASS" if max(rel.values()) < 5e-2 else "FAIL"
        res["backward"][name] = {"max_rel_err": rel, "grad_shape": [list(t.grad.shape) for t in leaves], "verdict": verdict}
        log(f"  {name:<26} {verdict}  " + "  ".join(f"{n} rel err={e:.2e}" for n, e in rel.items()))
    log("")

    log("Memory, measured (fwd+bwd, B=1, H=8, d=64, bf16, causal; peak above inputs):")
    res["memory_mb"] = {}
    for L in (4096, 8192):
        qq, kk, vv = mk(1, 8, L, 64), mk(1, 8, L, 64), mk(1, 8, L, 64)
        upL = torch.randn(1, 8, L, 64, device="cuda", generator=g, dtype=torch.bfloat16)

        def step(fn):
            leaves = [t.clone().requires_grad_() for t in (qq, kk, vv)]
            fn(*leaves).backward(upL)

        def naive(a, b, c):
            with sdpa_kernel(SDPBackend.MATH):
                return F.scaled_dot_product_attention(a, b, c, is_causal=True)

        table = 8 * L * L * 2 / 2**20
        entries = {
            "Naive (PyTorch math)": (lambda: step(naive), "O(N²)"),
            "FlashTile triton": (lambda: step(lambda a, b, c: attention(a, b, c, causal=True, backend="triton")), "O(N)"),
            "FlashTile tiled": (lambda: step(lambda a, b, c: attention(a, b, c, causal=True, backend="tiled", block_q=256, block_k=256)), "O(N)"),
        }
        step(lambda a, b, c: attention(a, b, c, causal=True, backend="triton"))  # autotune outside the measurement
        res["memory_mb"][str(L)] = {"one_LxL_bf16_table_mb": table}
        log(f"  seq={L} (one L x L bf16 score table for 8 heads = {table:.0f} MB)")
        for name, (fn, order) in entries.items():
            try:
                mb = _peak(fn)
                res["memory_mb"][str(L)][name] = mb
                log(f"    {name:<26} {mb:10.1f} MB   {order}")
            except torch.OutOfMemoryError:
                res["memory_mb"][str(L)][name] = None
                log(f"    {name:<26}        OOM   {order}")
            torch.cuda.empty_cache()
    log("")

    from flashtile.triton_kernels import tuned_configs

    res["autotune"] = tuned_configs()
    log("Autotuner choices in this process (key = LQ bucket, LK bucket, group, d, dv, causal, dtypes):")
    for kern, table in res["autotune"].items():
        for key, cfg in table.items():
            log(f"  {kern:<9} {key}  ->  {cfg}")
    log("")
    log(f"Saved to {out_dir / 'validation_results.json'}")
    (out_dir / "validation_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "validation_results.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    run(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results/local/validation"))
