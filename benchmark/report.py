"""Turns bench_results.json into the v1-style dashboards, logs and REPORT.md / SUMMARY.md.

    python benchmark/report.py results/<run>
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import style  # noqa: E402

BACKEND_ORDER = ["flashtile-triton", "flashtile-atomic", "sdpa-flash", "sdpa-cudnn", "sdpa-efficient", "naive"]
GPU_COLORS = ["#9B59B6", "#3498DB", "#2ECC71", "#F39C12", "#1ABC9C", "#E74C3C"]


def _groups(rows):
    out = defaultdict(list)
    for r in rows:
        out[(r["workload"], r["causal"], r["mode"])].append(r)
    return out


def _slug(mode):
    return mode.replace("+", "_")


def _series(grp, backend, field):
    """(lengths, values) for ok rows of one backend."""
    pts = sorted((r["L"], r[field]) for r in grp if r["backend"] == backend and r["status"] == "ok")
    return [p[0] for p in pts], [p[1] for p in pts]


def _ratio_series(grp, backend, field, base="naive", invert=False):
    """value(base) / value(backend) per length where both ran (speedup or memory reduction)."""
    xs, ys = [], []
    for L in sorted({r["L"] for r in grp}):
        a = next((r for r in grp if r["L"] == L and r["backend"] == backend and r["status"] == "ok"), None)
        b = next((r for r in grp if r["L"] == L and r["backend"] == base and r["status"] == "ok"), None)
        if a and b and a[field] and b[field]:
            xs.append(L)
            ys.append(a[field] / b[field] if invert else b[field] / a[field])
    return xs, ys


def _len_axis(ax, lengths):
    ax.set_xscale("log", base=2)
    ax.set_xticks(lengths)
    ax.set_xticklabels([str(L) for L in lengths])
    ax.set_xlabel("Sequence Length")


def _mark_failures(ax, grp, backend, color):
    for r in grp:
        if r["backend"] == backend and r["status"] not in ("ok",):
            ax.annotate(f"{r['status'].upper()}", (r["L"], ax.get_ylim()[0]), color=color, fontsize=8,
                        rotation=90, va="bottom", ha="center", fontweight="bold")


def config_figures(grp, out_dir: pathlib.Path, gpu: str):
    plt = style.plt()
    wl, causal, mode = grp[0]["workload"], grp[0]["causal"], grp[0]["mode"]
    lengths = sorted({r["L"] for r in grp})
    head = f"{gpu}  |  {wl}  |  {'causal' if causal else 'non-causal'}  |  {mode}  |  bf16"

    # 1. dashboard, v1 panel order
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    style.style_axes(fig, axes.flat)
    (ax_mem, ax_time), (ax_speed, ax_red) = axes
    for b in BACKEND_ORDER:
        c, lab = style.COLORS[b], style.LABELS[b]
        xs, ys = _series(grp, b, "peak_mib")
        if xs:
            ax_mem.plot(xs, ys, marker="o", color=c, linewidth=2, label=lab)
        xs, ys = _series(grp, b, "median_ms")
        if xs:
            ax_time.plot(xs, ys, marker="s", color=c, linewidth=2, label=lab)
        if b != "naive":
            xs, ys = _ratio_series(grp, b, "median_ms")
            if xs:
                ax_speed.plot(xs, ys, marker="^", color=c, linewidth=2, label=lab)
            xs, ys = _ratio_series(grp, b, "peak_mib")
            if xs:
                ax_red.plot(xs, ys, marker="d", color=c, linewidth=2, label=lab)
    style.title(ax_mem, "Memory Scaling")
    ax_mem.set_ylabel("Peak memory above inputs (MB)")
    ax_mem.set_yscale("log")
    style.title(ax_time, "Execution Time")
    ax_time.set_ylabel("Time per call (ms)")
    ax_time.set_yscale("log")
    style.title(ax_speed, "Speedup vs Naive")
    ax_speed.set_ylabel("Speedup (×)")
    ax_speed.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    style.title(ax_red, "Memory Reduction vs Naive")
    ax_red.set_ylabel("Reduction Ratio (×)")
    for ax in axes.flat:
        _len_axis(ax, lengths)
        style.legend(ax)
    _mark_failures(ax_time, grp, "naive", style.COLORS["naive"])
    style.suptitle(fig, head)
    style.save(fig, out_dir / "benchmark_dashboard.png")
    plt.close(fig)

    # 2. memory plot
    fig, ax = plt.subplots(figsize=(12, 8))
    style.style_axes(fig, [ax])
    for b in BACKEND_ORDER:
        xs, ys = _series(grp, b, "peak_mib")
        if xs:
            ax.plot(xs, ys, marker="o", color=style.COLORS[b], linewidth=2, markersize=8, label=style.LABELS[b])
    _mark_failures(ax, grp, "naive", style.COLORS["naive"])
    ax.set_yscale("log")
    ax.set_ylabel("Peak memory above inputs (MB)")
    _len_axis(ax, lengths)
    ax.set_title(f"Memory Usage vs Sequence Length\n{head}", fontsize=16, fontweight="bold", pad=20, color=style.TEXT)
    style.legend(ax, loc="upper left")
    style.save(fig, out_dir / "benchmark_memory.png")
    plt.close(fig)

    # 3. throughput plot
    fig, ax = plt.subplots(figsize=(12, 8))
    style.style_axes(fig, [ax])
    for b in BACKEND_ORDER:
        xs, ys = _series(grp, b, "tflops")
        if xs:
            ax.plot(xs, ys, marker="s", color=style.COLORS[b], linewidth=2, markersize=8, label=style.LABELS[b])
    ax.set_ylabel("TFLOP/s (higher is better)")
    ax.set_ylim(bottom=0)
    _len_axis(ax, lengths)
    ax.set_title(f"Throughput vs Sequence Length\n{head}", fontsize=16, fontweight="bold", pad=20, color=style.TEXT)
    style.legend(ax, loc="upper left")
    style.save(fig, out_dir / "benchmark_performance.png")
    plt.close(fig)

    # 4. numbers behind the figures
    data = {
        "config": {k: grp[0][k] for k in ("gpu", "workload", "Hq", "Hkv", "d", "causal", "mode")}
        | {"seq_lengths": lengths, "batch": {str(L): next(r["B"] for r in grp if r["L"] == L) for L in lengths},
           "dtype": "bfloat16"},
        "results": {
            b: {
                "seq_lengths": lengths,
                "status": [next((r["status"] for r in grp if r["backend"] == b and r["L"] == L), None) for L in lengths],
                "time_ms": [next((r["median_ms"] for r in grp if r["backend"] == b and r["L"] == L), None) for L in lengths],
                "time_p10_ms": [next((r["p10_ms"] for r in grp if r["backend"] == b and r["L"] == L), None) for L in lengths],
                "time_p90_ms": [next((r["p90_ms"] for r in grp if r["backend"] == b and r["L"] == L), None) for L in lengths],
                "memory_mb": [next((r["peak_mib"] for r in grp if r["backend"] == b and r["L"] == L), None) for L in lengths],
                "tflops": [next((r["tflops"] for r in grp if r["backend"] == b and r["L"] == L), None) for L in lengths],
            }
            for b in BACKEND_ORDER if any(r["backend"] == b for r in grp)
        },
    }
    (out_dir / "benchmark_results.json").write_text(json.dumps(data, indent=2))


def v1_log(rows, causal: bool, env: dict | None) -> str:
    """The run log in v1's format, one block per workload and mode."""
    gpu = rows[0]["gpu"]
    lines = []
    for (wl, c, mode), grp in _groups(rows).items():
        if c != causal:
            continue
        lengths = sorted({r["L"] for r in grp})
        r0 = grp[0]
        lines += ["=" * 70, "FlashTile Benchmark Suite", "=" * 70,
                  f"Device: {gpu}" + (f" ({env['compute_capability']}, {env['sm_count']} SMs)" if env else ""),
                  f"Workload: {wl}  (Hq={r0['Hq']}, Hkv={r0['Hkv']}, head dim={r0['d']})  Mode: {mode}",
                  f"Sequence lengths: {lengths}",
                  f"Batch: {[next(r['B'] for r in grp if r['L'] == L) for L in lengths]} (batch x length = 16384 tokens)",
                  f"Causal: {causal}", "dtype: bfloat16",
                  "Timing: median of >= 20 samples, each >= 2 ms of back-to-back calls (CUDA events)",
                  "=" * 70, ""]
        for b in BACKEND_ORDER:
            mine = sorted((r for r in grp if r["backend"] == b), key=lambda r: r["L"])
            if not mine:
                continue
            lines.append(f"📊 Benchmarking {style.LABELS[b]}...")
            for r in mine:
                if r["status"] == "ok":
                    lines.append(f"  Testing seq_len={r['L']}... OK (mem={r['peak_mib']:.1f}MB, time={r['median_ms']:.3f}ms, "
                                 f"{r['tflops']:.1f} TFLOP/s, max err={r['fwd_err']:.2e})")
                else:
                    lines.append(f"  Testing seq_len={r['L']}... {r['status'].upper()}" + (f" ({r['note'][:80]})" if r["note"] else ""))
            lines.append("")
        lines += ["=" * 70, "Benchmark Summary", "=" * 70, ""]
        for b in BACKEND_ORDER:
            mine = sorted((r for r in grp if r["backend"] == b), key=lambda r: r["L"])
            if not mine:
                continue
            lines += [f"{style.LABELS[b]}:",
                      f"  Memory (MB): {[round(r['peak_mib'], 1) if r['peak_mib'] is not None else None for r in mine]}",
                      f"  Time (ms):   {[round(r['median_ms'], 3) if r['median_ms'] is not None else None for r in mine]}",
                      f"  TFLOP/s:     {[round(r['tflops'], 1) if r['tflops'] is not None else None for r in mine]}"]
            if b != "naive":
                _, sp = _ratio_series(grp, b, "median_ms")
                _, red = _ratio_series(grp, b, "peak_mib")
                if sp:
                    lines.append(f"  Speedup vs naive: {[round(x, 2) for x in sp]}")
                    lines.append(f"  Memory reduction vs naive: {[round(x, 1) for x in red]}")
            lines.append("")
        ours = {r["L"]: r for r in grp if r["backend"] == "flashtile-triton" and r["status"] == "ok"}
        sdpa = [r for r in grp if r["backend"].startswith("sdpa") and r["status"] == "ok"]
        lines.append("FlashTile time / fastest PyTorch SDPA time (below 1.00 = FlashTile faster):")
        for L in lengths:
            best = min((r for r in sdpa if r["L"] == L), key=lambda r: r["median_ms"], default=None)
            if best and L in ours:
                lines.append(f"  seq_len={L}: {ours[L]['median_ms'] / best['median_ms']:.2f}  (fastest: {best['backend']})")
        lines += ["", ""]
    return "\n".join(lines)


def _table(grp, field, fmt):
    lengths = sorted({r["L"] for r in grp})
    lines = ["| backend | " + " | ".join(f"L={L}" for L in lengths) + " |",
             "|---|" + "---:|" * len(lengths)]
    for backend in BACKEND_ORDER:
        if not any(x["backend"] == backend for x in grp):
            continue
        cells = []
        for L in lengths:
            r = next((x for x in grp if x["backend"] == backend and x["L"] == L), None)
            if r is None:
                cells.append("")
            elif r["status"] != "ok":
                cells.append(r["status"])
            else:
                cells.append(format(r[field], fmt))
        lines.append(f"| {backend} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def gpu_report(rows, out_path: pathlib.Path, env: dict | None):
    gpu = rows[0]["gpu"]
    md = [f"# FlashTile benchmark: {gpu}", ""]
    if env:
        md += [f"- {k}: {v}" for k, v in env.items()] + [""]
    md += [
        "All numbers are measured on this GPU, bf16 inputs, batch x length = 16384 tokens.",
        "Each row passed a parity check against the fp32 dense oracle before it was timed.",
        "Times are per-call medians of >= 20 samples; each sample is >= 2 ms of back-to-back calls between CUDA events.",
        "TFLOP/s uses 4*B*H*L^2*d for the forward (halved, plus the diagonal, for causal) and 3.5x that for fwd+bwd.",
        "`naive` is PyTorch's SDPA MATH backend, which materialises the L x L score table.",
        "",
    ]
    for (wl, causal, mode), grp in _groups(rows).items():
        folder = f"{'causal' if causal else 'non_causal'}/{wl}_{_slug(mode)}"
        md += [f"## {wl}, causal={causal}, {mode}", "", f"![dashboard]({folder}/benchmark_dashboard.png)", "",
               "TFLOP/s:", "", _table(grp, "tflops", ".1f"), "",
               "Median ms per call:", "", _table(grp, "median_ms", ".3f"), "",
               "Peak memory above inputs, MiB:", "", _table(grp, "peak_mib", ".1f"), ""]
        cfg = [r for r in grp if r["backend"] == "flashtile-triton" and r.get("triton_config")]
        if cfg:
            md += ["Autotuner choice (FlashTile):", ""] + [
                f"- L={r['L']}: " + "; ".join(f"{k}: `{v}`" for k, v in r["triton_config"].items()) for r in cfg] + [""]
    notes = [r for r in rows if r["note"]]
    if notes:
        md += ["## Rows that did not run", "", "| workload | causal | mode | L | backend | status | note |",
               "|---|---|---|---:|---|---|---|"]
        md += [f"| {r['workload']} | {r['causal']} | {r['mode']} | {r['L']} | {r['backend']} | {r['status']} | "
               f"{r['note'][:160].replace('|', '/')} |" for r in notes]
    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")


def gpu_outputs(rows, bench_dir: pathlib.Path, env: dict | None):
    """Everything per GPU: figures and JSON per configuration, v1 logs, REPORT.md."""
    gpu = rows[0]["gpu"]
    for (wl, causal, mode), grp in _groups(rows).items():
        config_figures(grp, bench_dir / ("causal" if causal else "non_causal") / f"{wl}_{_slug(mode)}", gpu)
    for causal in (True, False):
        if any(r["causal"] == causal for r in rows):
            name = "causal_log.txt" if causal else "non_causal_log.txt"
            (bench_dir / name).write_text(v1_log(rows, causal, env), encoding="utf-8")
    gpu_report(rows, bench_dir / "REPORT.md", env)


def _best_sdpa(at):
    theirs = [r for r in at if r["backend"].startswith("sdpa")]
    return min(theirs, key=lambda r: r["median_ms"]) if theirs else None


def summary(run_dir: pathlib.Path):
    """Cross-GPU summary: FlashTile vs the fastest SDPA backend that ran, per configuration."""
    per_gpu = {}
    for p in sorted(run_dir.glob("*/benchmarks/bench_results.json")):
        per_gpu[p.parts[-3]] = json.loads(p.read_text())
    if not per_gpu:
        return
    md = ["# FlashTile results across GPUs", "",
          "Ratio = FlashTile Triton time / fastest PyTorch SDPA backend time for the same inputs "
          "(below 1.0 means FlashTile was faster). Both measured in the same process.", "",
          "![cross-GPU dashboard](cross_gpu_dashboard.png)", ""]
    for rows in per_gpu.values():
        md += [f"## {rows[0]['gpu']}", "",
               "| workload | causal | mode | L | FlashTile TFLOP/s | best SDPA | SDPA TFLOP/s | ratio | vs sdpa-flash "
               "| atomic-dQ TFLOP/s | atomic-dQ ratio |",
               "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|"]
        for (wl, causal, mode), grp in _groups(rows).items():
            for L in sorted({r["L"] for r in grp}):
                at = [r for r in grp if r["L"] == L and r["status"] == "ok"]
                ours = next((r for r in at if r["backend"] == "flashtile-triton"), None)
                atom = next((r for r in at if r["backend"] == "flashtile-atomic"), None)
                best = _best_sdpa(at)
                fl = next((r for r in at if r["backend"] == "sdpa-flash"), None)
                f = lambda r, k, fmt: format(r[k], fmt) if r else "-"
                md.append(
                    f"| {wl} | {causal} | {mode} | {L} | {f(ours, 'tflops', '.1f')} | "
                    f"{best['backend'] if best else '-'} | {f(best, 'tflops', '.1f')} | "
                    f"{format(ours['median_ms'] / best['median_ms'], '.2f') if ours and best else '-'} | "
                    f"{format(ours['median_ms'] / fl['median_ms'], '.2f') if ours and fl else '-'} | "
                    f"{f(atom, 'tflops', '.1f')} | "
                    f"{format(atom['median_ms'] / best['median_ms'], '.2f') if atom and best else '-'} |"
                )
        md.append("")
    (run_dir / "SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    cross_gpu_dashboard(per_gpu, run_dir / "cross_gpu_dashboard.png")


def cross_gpu_dashboard(per_gpu: dict, path: pathlib.Path, workload="mha-d128"):
    plt = style.plt()
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    style.style_axes(fig, axes.flat)
    lengths = set()
    for j, mode in enumerate(("fwd", "fwd+bwd")):
        ax_t, ax_r = axes[0][j], axes[1][j]
        for i, rows in enumerate(per_gpu.values()):
            grp = [r for r in rows if r["workload"] == workload and r["causal"] and r["mode"] == mode]
            if not grp:
                continue
            c = GPU_COLORS[i % len(GPU_COLORS)]
            name = rows[0]["gpu"].replace("NVIDIA ", "")
            xs, ys = _series(grp, "flashtile-triton", "tflops")
            lengths.update(xs)
            ax_t.plot(xs, ys, marker="o", color=c, linewidth=2, label=f"FlashTile, {name}")
            bx, by = [], []
            rx, ry = [], []
            for L in sorted({r["L"] for r in grp}):
                at = [r for r in grp if r["L"] == L and r["status"] == "ok"]
                best = _best_sdpa(at)
                ours = next((r for r in at if r["backend"] == "flashtile-triton"), None)
                if best:
                    bx.append(L)
                    by.append(best["tflops"])
                if best and ours:
                    rx.append(L)
                    ry.append(best["median_ms"] / ours["median_ms"])
            ax_t.plot(bx, by, marker="s", color=c, linewidth=1.5, linestyle="--", alpha=0.7, label=f"best SDPA, {name}")
            ax_r.plot(rx, ry, marker="^", color=c, linewidth=2, label=name)
        style.title(ax_t, f"Throughput, {workload} causal, {mode}")
        ax_t.set_ylabel("TFLOP/s")
        ax_t.set_ylim(bottom=0)
        style.title(ax_r, f"FlashTile speed relative to fastest SDPA, {mode}")
        ax_r.set_ylabel("SDPA time / FlashTile time (×, 1 = parity)")
        ax_r.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
        ax_r.set_ylim(bottom=0)
    for ax in axes.flat:
        if lengths:
            _len_axis(ax, sorted(lengths))
        style.legend(ax, fontsize=9)
    style.suptitle(fig, "FlashTile across GPUs (bf16, batch x length = 16384 tokens)")
    style.save(fig, path)
    plt.close(fig)


def build(run_dir: pathlib.Path):
    for gpu_dir in sorted(p for p in run_dir.iterdir() if (p / "benchmarks" / "bench_results.json").exists()):
        rows = json.loads((gpu_dir / "benchmarks" / "bench_results.json").read_text())
        env_path = gpu_dir / "environment.json"
        env = json.loads(env_path.read_text()) if env_path.exists() else None
        gpu_outputs(rows, gpu_dir / "benchmarks", env)
    summary(run_dir)


if __name__ == "__main__":
    build(pathlib.Path(sys.argv[1]))
