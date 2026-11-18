"""Kernel benchmark: FlashTile vs PyTorch SDPA backends (flash, cuDNN, mem-efficient) and naive.

Every row is checked against the fp32 oracle before it's timed (gradients too up to L=4096).
Timing is CUDA events around back-to-back calls, so Python dispatch overlaps with GPU work
the way it would in a model. Same rule for every backend.

    python benchmark/bench_attention.py --out results/local
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
import sys
import time
import traceback

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flashtile import attention, attention_reference  # noqa: E402
import flashtile.triton_kernels as tk  # noqa: E402
from flashtile.triton_kernels import last_configs  # noqa: E402

TOKENS = 16384  # batch * length is held constant so every row does comparable work per token
LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]
WORKLOADS = [
    # name, Hq, Hkv, d, causal settings
    ("mha-d128", 16, 16, 128, (True, False)),
    ("mha-d64", 32, 32, 64, (True,)),
    ("gqa4-d128", 32, 8, 128, (True,)),
    ("qwen3.5-d256", 16, 4, 256, (True,)),  # Qwen3.5-4B full-attention layer shape
]
MODES = ("fwd", "fwd+bwd")
DTYPE = torch.bfloat16
FWD_TOL, GRAD_TOL = 2e-2, 1e-1  # bf16 output vs fp32 oracle; grads relative to their max magnitude
GRAD_CHECK_MAX_L = 4096
MIN_SAMPLE_MS = 2.0


def backends(gqa: bool):
    def flashtile_mode(mode):
        def run(q, k, v, causal):
            tk.BACKWARD_MODE = mode  # read when the autograd node is created
            return attention(q, k, v, causal=causal, backend="triton")
        return run

    def sdpa(kind):
        def run(q, k, v, causal):
            with sdpa_kernel(kind):
                return F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=gqa)
        return run

    return {
        "flashtile-triton": flashtile_mode("two-kernel"),
        "flashtile-atomic": flashtile_mode("atomic"),  # fwd+bwd only: the forward is the same kernel
        "sdpa-flash": sdpa(SDPBackend.FLASH_ATTENTION),
        "sdpa-cudnn": sdpa(SDPBackend.CUDNN_ATTENTION),
        "sdpa-efficient": sdpa(SDPBackend.EFFICIENT_ATTENTION),
        "naive": sdpa(SDPBackend.MATH),  # PyTorch's reference math: materialises the L x L table
    }


def flops(B, Hq, L, d, causal, mode):
    fwd = 4 * B * Hq * L * L * d  # QK^T and PV, 2 flops per multiply-add
    if causal:
        fwd = fwd * (L + 1) / (2 * L)
    return fwd * (1.0 if mode == "fwd" else 3.5)  # backward ~2.5x forward


def time_cuda(fn, min_samples=20, min_ms=400.0, max_samples=500, warmup=3):
    """Return (per-call ms samples, reps per sample)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    single = (time.perf_counter() - t0) * 1e3
    reps = max(1, math.ceil(MIN_SAMPLE_MS / max(single, 1e-3)))
    samples, total = [], 0.0
    while (len(samples) < min_samples or total < min_ms) and len(samples) < max_samples:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            fn()
        end.record()
        end.synchronize()
        ms = start.elapsed_time(end)
        samples.append(ms / reps)
        total += ms
    return samples, reps


def peak_mib(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def quantile(xs, q):
    xs = sorted(xs)
    pos = q * (len(xs) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def parity(fn, q, k, v, causal, group, check_grads, dout):
    """Max errors of backend output (and grads) vs the fp32 oracle on batch 0, KV head 0."""
    leaves = [t.detach().clone().requires_grad_(check_grads) for t in (q, k, v)]
    out = fn(*leaves, causal)
    sl_q, sl_kv = (slice(0, 1), slice(0, group)), (slice(0, 1), slice(0, 1))
    ref_in = [leaves[0][sl_q].detach().float(), leaves[1][sl_kv].detach().float(), leaves[2][sl_kv].detach().float()]
    for t in ref_in:
        t.requires_grad_(check_grads)
    ref, _ = attention_reference(*ref_in, causal=causal)
    fwd_err = (out[sl_q].float() - ref).abs().max().item()
    grad_err = None
    if check_grads:
        grads = torch.autograd.grad(out, leaves, dout)
        ref_grads = torch.autograd.grad(ref, ref_in, dout[sl_q].float())
        sls = (sl_q, sl_kv, sl_kv)
        grad_err = max(
            ((g[s].float() - r).abs().max() / r.abs().max().clamp_min(1e-6)).item()
            for g, r, s in zip(grads, ref_grads, sls)
        )
    del out, ref, leaves, ref_in
    return fwd_err, grad_err


def run(out_dir: pathlib.Path, lengths=LENGTHS, workloads=WORKLOADS):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "bench_log.txt", "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    torch.backends.cuda.matmul.allow_tf32 = False
    props = torch.cuda.get_device_properties(0)
    gpu = props.name
    import triton

    log("=" * 110)
    log(f"FlashTile attention benchmark   {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    log(f"GPU {gpu}  sm_{props.major}{props.minor}  {props.total_memory / 2**30:.0f} GiB  {props.multi_processor_count} SMs")
    log(f"torch {torch.__version__}  CUDA {torch.version.cuda}  cuDNN {torch.backends.cudnn.version()}  triton {triton.__version__}")
    log(f"dtype {DTYPE}  tokens per row {TOKENS} (batch = {TOKENS} / L)")
    log(f"Timing: CUDA events around back-to-back calls, each sample >= {MIN_SAMPLE_MS} ms, >= 20 samples and "
        ">= 400 ms total, 3 warmup calls. Reported per call: median, p10, p90.")
    log("Parity: fp32 dense oracle on batch 0 / KV head 0 group; forward always, gradients for L <= "
        f"{GRAD_CHECK_MAX_L}. Tolerance fwd {FWD_TOL} abs, grads {GRAD_TOL} relative to max |grad|.")
    log("naive = PyTorch SDPA MATH backend (materialises the full L x L score table).")
    log("=" * 110)

    rows = []
    for name, Hq, Hkv, d, causal_opts in workloads:
        group = Hq // Hkv
        for causal in causal_opts:
            for mode in MODES:
                log(f"\n--- workload {name}  Hq={Hq} Hkv={Hkv} d={d}  causal={causal}  mode={mode} ---")
                log(f"{'L':>6} {'B':>3}  {'backend':<17} {'status':<12} {'median ms':>10} {'p10':>9} {'p90':>9} "
                    f"{'n':>4} {'reps':>4} {'TFLOP/s':>8} {'peak MiB':>10} {'fwd err':>9} {'grad err':>9}")
                for L in lengths:
                    B = max(1, TOKENS // L)
                    g = torch.Generator(device="cuda").manual_seed(L + d)
                    mk = lambda *s: torch.randn(*s, device="cuda", generator=g, dtype=DTYPE)
                    q, k, v = mk(B, Hq, L, d), mk(B, Hkv, L, d), mk(B, Hkv, L, d)
                    dout = mk(B, Hq, L, d)
                    for backend, fn in backends(group > 1).items():
                        if backend == "flashtile-atomic" and mode == "fwd":
                            continue
                        row = dict(gpu=gpu, workload=name, Hq=Hq, Hkv=Hkv, d=d, causal=causal, mode=mode,
                                   L=L, B=B, backend=backend, status="ok", median_ms=None, p10_ms=None,
                                   p90_ms=None, n=0, reps=0, tflops=None, peak_mib=None, fwd_err=None,
                                   grad_err=None, triton_config=None, note="")
                        try:
                            check_grads = mode == "fwd+bwd" and L <= GRAD_CHECK_MAX_L
                            fe, ge = parity(fn, q, k, v, causal, group, check_grads, dout)
                            row["fwd_err"], row["grad_err"] = fe, ge
                            if fe > FWD_TOL or (ge is not None and ge > GRAD_TOL) or not math.isfinite(fe):
                                row["status"] = "parity-fail"
                            else:
                                if mode == "fwd":
                                    def step():
                                        with torch.no_grad():
                                            fn(q, k, v, causal)
                                else:
                                    leaves = [t.detach().requires_grad_() for t in (q, k, v)]

                                    def step():
                                        out = fn(*leaves, causal)
                                        torch.autograd.grad(out, leaves, dout)
                                samples, reps = time_cuda(step)
                                med = statistics.median(samples)
                                row.update(median_ms=med, p10_ms=quantile(samples, 0.1), p90_ms=quantile(samples, 0.9),
                                           n=len(samples), reps=reps,
                                           tflops=flops(B, Hq, L, d, causal, mode) / (med * 1e-3) / 1e12,
                                           peak_mib=peak_mib(step))
                                if backend.startswith("flashtile"):
                                    cfgs = last_configs()
                                    row["triton_config"] = cfgs if mode == "fwd+bwd" else {"fwd": cfgs["fwd"]}
                        except torch.OutOfMemoryError as err:
                            row["status"], row["note"] = "oom", str(err).splitlines()[0][:200]
                        except RuntimeError as err:
                            msg = str(err).strip().splitlines()[0] if str(err).strip() else repr(err)
                            row["status"] = "unsupported" if "No available kernel" in str(err) else "error"
                            row["note"] = msg[:300]
                            if row["status"] == "error":
                                log(traceback.format_exc())
                        finally:
                            torch.cuda.empty_cache()
                        rows.append(row)
                        fmt = lambda x, f: "-" if x is None else format(x, f)
                        log(f"{L:>6} {B:>3}  {backend:<17} {row['status']:<12} {fmt(row['median_ms'], '10.3f')} "
                            f"{fmt(row['p10_ms'], '9.3f')} {fmt(row['p90_ms'], '9.3f')} {row['n']:>4} {row['reps']:>4} "
                            f"{fmt(row['tflops'], '8.1f')} {fmt(row['peak_mib'], '10.1f')} "
                            f"{fmt(row['fwd_err'], '9.2e')} {fmt(row['grad_err'], '9.2e')}"
                            + (f"   ({row['note'][:90]})" if row["note"] else ""))
                        if row["triton_config"]:
                            log(f"{'':>12}autotune: " + "; ".join(f"{k}: {v}" for k, v in row["triton_config"].items()))
                    del q, k, v, dout

    (out_dir / "bench_results.json").write_text(json.dumps(rows, indent=2))
    with open(out_dir / "bench_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({**r, "triton_config": json.dumps(r["triton_config"]) if r["triton_config"] else ""} for r in rows)
    log(f"\n{len(rows)} rows written: " + ", ".join(
        f"{s}={sum(r['status'] == s for r in rows)}" for s in ("ok", "parity-fail", "unsupported", "oom", "error")))
    log_file.close()
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/local")
    ap.add_argument("--quick", action="store_true", help="two lengths, for smoke testing")
    args = ap.parse_args()
    run(pathlib.Path(args.out), lengths=[512, 2048] if args.quick else LENGTHS)
