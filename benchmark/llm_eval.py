"""FlashTile inside Qwen3.5-4B (FP8 checkpoint for inference, bf16 original for training).

Qwen3.5-4B mixes 24 Gated-DeltaNet layers with 8 softmax attention layers (16 q heads, 4 kv
heads, d=256). FlashTile replaces those 8 through flashtile.hf, nothing else changes.

What it measures: per-layer precision on real activations and gradients vs an fp64 oracle,
long-context prefill (eager / SDPA / FlashTile), greedy generation, and fine-tuning the 8
attention blocks with SDPA and with FlashTile from the same weights. No perplexity.

The text comes from --corpus (any long English .md files, ~66k+ tokens). The published runs
used the project's own study notes.

    python benchmark/llm_eval.py --out results/local/llm --corpus corpus
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import pathlib
import statistics
import sys
import time
import traceback

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import style  # noqa: E402

import flashtile.hf as fhf  # noqa: E402
from flashtile import attention, attention_reference  # noqa: E402

MODEL_8BIT = "RedHatAI/Qwen3.5-4B-FP8-dynamic"
MODEL_BF16 = "Qwen/Qwen3.5-4B"
CORPUS_FILES = ["TECHNICAL-REFERENCE.md", "BLUEPRINT.md", "INTERVIEW-PRACTICE.md", "STUDY-GUIDE.md"]
IMPLS = ["sdpa", "flashtile", "eager"]  # sdpa first: the others are compared with its logits
PRECISION_L = 4096
PREFILL_LENGTHS = [2048, 4096, 8192, 16384, 32768, 65536]
PREFILL_REPEATS = 3
KEEP_LOGITS = 256
GEN_TOKENS = 200
TRAIN_L = 4096
TRAIN_STEPS = 30
TRAIN_LR = 2e-5
TRAIN_SWEEP = [2048, 4096, 8192, 16384]  # every layer's activations are kept for backward


class Timer:
    """CUDA events around every attention call while `on`; total_ms() is the time spent inside attention."""

    on = False
    events: list = []

    @classmethod
    def wrap(cls, fn):
        def timed(*args, **kwargs):
            if not cls.on:
                return fn(*args, **kwargs)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            out = fn(*args, **kwargs)
            e.record()
            cls.events.append((s, e))
            return out

        return timed

    @classmethod
    def start(cls):
        cls.events, cls.on = [], True

    @classmethod
    def stop_ms(cls):
        cls.on = False
        torch.cuda.synchronize()
        return sum(s.elapsed_time(e) for s, e in cls.events)


CAPTURED: list = []


def _register():
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.masking_utils import eager_mask, sdpa_mask
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    fhf.register()

    def capture(module, query, key, value, attention_mask, **kw):
        rec = {"layer": module.layer_idx, "q": query.detach().clone(), "k": key.detach().clone(),
               "v": value.detach().clone()}
        out, w = sdpa_attention_forward(module, query, key, value, attention_mask, **kw)
        if out.requires_grad:  # out is [B, L, H, D]; keep dO as [B, H, L, D]
            out.register_hook(lambda g: rec.__setitem__("do", g.detach().transpose(1, 2).contiguous()))
        CAPTURED.append(rec)
        return out, w

    for name, fn, mask in (
        ("ftile_t", fhf.flashtile_attention_forward, sdpa_mask),
        ("sdpa_t", sdpa_attention_forward, sdpa_mask),
        ("eager_t", mq.eager_attention_forward, eager_mask),
    ):
        AttentionInterface.register(name, Timer.wrap(fn))
        AttentionMaskInterface.register(name, mask)
    AttentionInterface.register("capture", capture)
    AttentionMaskInterface.register("capture", sdpa_mask)


TIMED = {"eager": "eager_t", "sdpa": "sdpa_t", "flashtile": "ftile_t"}


def _load(name):
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda",
                                                        attn_implementation="sdpa")
    model.eval()
    return model


def _set(model, impl):
    """impl is a display name ("flashtile", "sdpa", "eager") or a registered name (e.g. "ftile_t")."""
    model.set_attn_implementation(fhf.NAME if impl == "flashtile" else impl)


def _free():
    gc.collect()
    torch.cuda.empty_cache()


def _peak_reset():
    torch.cuda.synchronize()
    _free()
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated()


def _peak_mib(base):
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def _attn_layers(model):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

    return [m for m in model.modules() if isinstance(m, Qwen3_5Attention)]



def _row_stats(q, k, lse_ref, scale):
    """Peakedness of the real attention rows: max |score|, mean max-probability, share of rows above 0.9."""
    hq, hk = q.shape[1], k.shape[1]
    L = q.shape[2]
    causal = torch.ones(L, L, dtype=torch.bool, device=q.device).tril()
    max_s, maxp = 0.0, []
    for h in range(hq):
        s = (q[0, h].float() @ k[0, h // (hq // hk)].float().T) * scale
        s = s.masked_fill(~causal, float("-inf"))
        rmax = s.amax(-1)
        max_s = max(max_s, rmax.abs().max().item())
        maxp.append(torch.exp(rmax.double() - lse_ref[0, h]).float())
        del s
    maxp = torch.cat(maxp)
    return {"max_abs_score": max_s, "mean_max_prob": maxp.mean().item(), "rows_max_prob_gt_0.9": (maxp > 0.9).float().mean().item()}


def layer_precision(rec, scale, with_grad):
    q, k, v = rec["q"], rec["k"], rec["v"]
    res = {"layer": rec["layer"], "max_abs_q": q.abs().max().item(), "max_abs_k": k.abs().max().item()}
    ref_in = [t.double().requires_grad_(with_grad) for t in (q, k, v)]
    ref_out, ref_lse = attention_reference(*ref_in, scale=scale, causal=True)
    ref_grads = torch.autograd.grad(ref_out, ref_in, rec["do"].double()) if with_grad else None
    ref_out = ref_out.detach()
    res.update(_row_stats(q, k, ref_lse.detach(), scale))
    ref_lse = ref_lse.detach()
    denom = ref_out.abs().max().item()

    def run(name, fn):
        leaves = [t.clone().requires_grad_(with_grad) for t in (q, k, v)]
        out, lse = fn(*leaves)
        d = (out.double() - ref_out).abs()
        r = {"out_max_abs": d.max().item(), "out_mean_abs": d.mean().item(), "out_rel": d.max().item() / denom}
        if lse is not None:
            r["lse_max_abs"] = (lse.double() - ref_lse).abs().max().item()
        if with_grad:
            grads = torch.autograd.grad(out, leaves, rec["do"])
            for n, g, rg in zip(("dq", "dk", "dv"), grads, ref_grads):
                r[f"{n}_rel"] = ((g.double() - rg).abs().max() / rg.abs().max()).item()
        res[name] = r

    run("flashtile", lambda a, b, c: attention(a, b, c, scale=scale, causal=True, backend="triton", return_lse=True))
    run("sdpa", lambda a, b, c: (F.scaled_dot_product_attention(a, b, c, is_causal=True, scale=scale, enable_gqa=True), None))
    del ref_in, ref_out, ref_grads
    _free()
    return res


def precision(model, ids, with_grad, log):
    CAPTURED.clear()
    _set(model, "capture")
    if with_grad:
        model.train()
        loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        loss.backward()
        model.zero_grad(set_to_none=True)
        model.eval()
    else:
        with torch.no_grad():
            model(input_ids=ids, use_cache=False, logits_to_keep=1)
    recs = list(CAPTURED)
    CAPTURED.clear()
    scale = _attn_layers(model)[0].scaling
    out = []
    for rec in recs:
        r = layer_precision(rec, scale, with_grad)
        out.append(r)
        ft, sd = r["flashtile"], r["sdpa"]
        line = (f"  layer {r['layer']:>2}: FlashTile out max|err|={ft['out_max_abs']:.2e} (rel {ft['out_rel']:.1e}, "
                f"lse {ft['lse_max_abs']:.1e})  SDPA out max|err|={sd['out_max_abs']:.2e} (rel {sd['out_rel']:.1e})"
                f"  | max|s|={r['max_abs_score']:.1f}  mean max-prob={r['mean_max_prob']:.3f}")
        if with_grad:
            line += ("\n            grads rel err  FlashTile dq/dk/dv=" + "/".join(f"{ft[n]:.1e}" for n in ("dq_rel", "dk_rel", "dv_rel"))
                     + "   SDPA dq/dk/dv=" + "/".join(f"{sd[n]:.1e}" for n in ("dq_rel", "dk_rel", "dv_rel")))
        log(line)
        del rec
    _free()
    return out



def prefill(model, ids_all, log):
    rows, logits = [], {}
    for L in PREFILL_LENGTHS:
        ids = ids_all[:, :L]
        logits[L] = {}
        for impl in IMPLS:
            row = {"L": L, "impl": impl, "status": "ok"}
            try:
                _set(model, TIMED[impl])
                with torch.no_grad():
                    model(input_ids=ids, use_cache=False, logits_to_keep=KEEP_LOGITS)  # warmup / autotune
                    base = _peak_reset()
                    times, attn = [], []
                    for _ in range(PREFILL_REPEATS):
                        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        Timer.start()
                        s.record()
                        out = model(input_ids=ids, use_cache=False, logits_to_keep=KEEP_LOGITS)
                        e.record()
                        attn.append(Timer.stop_ms())
                        times.append(s.elapsed_time(e))
                    row["peak_mib"] = _peak_mib(base)
                row.update(ms=statistics.median(times), attn_ms=statistics.median(attn),
                           tok_per_s=L / (statistics.median(times) / 1e3))
                logits[L][impl] = out.logits[0].float()
                del out
            except torch.OutOfMemoryError as err:
                row.update(status="oom", note=str(err).splitlines()[0][:160])
            except Exception as err:  # noqa: BLE001 - recorded, not hidden
                row.update(status="error", note=f"{type(err).__name__}: {str(err)[:200]}")
                log(traceback.format_exc())
            Timer.on = False
            _free()
            ref = logits[L].get("sdpa")
            mine = logits[L].get(impl)
            if ref is not None and mine is not None and impl != "sdpa":
                row["max_abs_logit_diff_vs_sdpa"] = (mine - ref).abs().max().item()
                row["mean_abs_logit_diff_vs_sdpa"] = (mine - ref).abs().mean().item()
                row["top1_agreement_vs_sdpa"] = (mine.argmax(-1) == ref.argmax(-1)).float().mean().item()
            rows.append(row)
            if row["status"] == "ok":
                extra = ""
                if "top1_agreement_vs_sdpa" in row:
                    extra = (f", vs SDPA: max|Δlogit|={row['max_abs_logit_diff_vs_sdpa']:.3f}, "
                             f"top-1 agree={100 * row['top1_agreement_vs_sdpa']:.1f}%")
                log(f"  seq_len={L:>6} {impl:<9} OK (time={row['ms']:.1f}ms, attention={row['attn_ms']:.1f}ms "
                    f"[{100 * row['attn_ms'] / row['ms']:.0f}%], mem={row['peak_mib']:.0f}MB, "
                    f"{row['tok_per_s']:.0f} tok/s{extra})")
            else:
                log(f"  seq_len={L:>6} {impl:<9} {row['status'].upper()} ({row.get('note', '')[:100]})")
        logits.pop(L)
        _free()
    return rows



def generation(model, tok, corpus_dir, log):
    doc = (corpus_dir / "TECHNICAL-REFERENCE.md").read_text(encoding="utf-8")
    prompts = {
        "short": "Explain in three sentences why attention needs a softmax and what the scale 1/sqrt(d) is for.",
        "code": "Write a short, numerically stable Python function that computes log-sum-exp of a list of floats.",
        "long_document_qa": doc + "\n\n---\nUsing only the document above: what does FlashTile return for a query row "
                            "whose keys are all masked (output, gradient and LSE)? Answer in two sentences.",
    }
    res = {}
    for name, prompt in prompts.items():
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, return_tensors="pt").input_ids.cuda()
        res[name] = {"prompt_tokens": ids.shape[1]}
        log(f"  [{name}] prompt tokens: {ids.shape[1]}")
        for impl in ("sdpa", "flashtile"):
            _set(model, impl)
            fhf.stats.clear()
            with torch.no_grad():
                model(input_ids=ids, logits_to_keep=1)  # warmup / autotune
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(input_ids=ids, logits_to_keep=1)
                torch.cuda.synchronize()
                prefill_s = time.perf_counter() - t0
                # untimed run first so decode shapes get autotuned outside the timing
                model.generate(ids, max_new_tokens=GEN_TOKENS, do_sample=False, temperature=None, top_p=None, top_k=None)
                fhf.stats.clear()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = model.generate(ids, max_new_tokens=GEN_TOKENS, do_sample=False, temperature=None, top_p=None,
                                     top_k=None)
                torch.cuda.synchronize()
                total_s = time.perf_counter() - t0
            new = out[0, ids.shape[1]:].tolist()
            decode_tps = (len(new) - 1) / max(total_s - prefill_s, 1e-6)
            res[name][impl] = {"tokens": new, "text": tok.decode(new, skip_special_tokens=True),
                               "prefill_ms": prefill_s * 1e3, "decode_tok_per_s": decode_tps,
                               "attention_calls": dict(fhf.stats) if impl == "flashtile" else None}
            log(f"    {impl:<9} prefill={prefill_s * 1e3:.0f}ms  decode={decode_tps:.1f} tok/s  new tokens={len(new)}"
                + (f"  FlashTile calls={dict(fhf.stats)}" if impl == "flashtile" else ""))
        a, b = res[name]["sdpa"]["tokens"], res[name]["flashtile"]["tokens"]
        same = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        res[name]["identical_prefix_tokens"] = same
        res[name]["identical"] = a == b
        log(f"    greedy outputs {'IDENTICAL' if a == b else f'diverge at token {same}'} "
            f"({len(a)} vs {len(b)} tokens)")
        log("    FlashTile answer: " + res[name]["flashtile"]["text"].strip().replace("\n", "\n      ")[:1500])
        _free()
    return res



def _lm_loss(model, ids, chunk=2048):
    """Next-token loss with the vocabulary projection done per chunk (and recomputed in backward)."""
    h = model.model(input_ids=ids, use_cache=False).last_hidden_state
    head = model.lm_head
    total, n = 0.0, ids.shape[1] - 1

    def piece(hc, yc):
        return F.cross_entropy(head(hc).float().flatten(0, 1), yc.flatten(), reduction="sum")

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        total = total + checkpoint(piece, h[:, s:e], ids[:, s + 1:e + 1], use_reentrant=False)
    return total / n


def _trainable(model):
    params = []
    for p in model.parameters():
        p.requires_grad_(False)
    for m in _attn_layers(model):
        for p in m.parameters():
            p.requires_grad_(True)
            params.append(p)
    return params


def train_curves(model, ids_all, log):
    params = _trainable(model)
    n_params = sum(p.numel() for p in params)
    log(f"  trainable: the 8 softmax-attention blocks, {n_params / 1e6:.1f}M parameters "
        f"(fp32 master copies, AdamW lr={TRAIN_LR}); every other layer frozen but back-propagated through")
    init = [p.detach().clone() for p in params]
    chunks = ids_all[0, : (ids_all.shape[1] // TRAIN_L) * TRAIN_L].view(-1, TRAIN_L)
    curves = {}
    for impl in ("sdpa", "flashtile"):
        with torch.no_grad():
            for p, p0 in zip(params, init):
                p.copy_(p0)
        master = [p.detach().float().clone().requires_grad_(True) for p in params]
        opt = torch.optim.AdamW(master, lr=TRAIN_LR, weight_decay=0.0)
        _set(model, impl)
        model.train()
        losses, times = [], []
        for step in range(TRAIN_STEPS):
            ids = chunks[step % chunks.shape[0]].unsqueeze(0)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss = _lm_loss(model, ids)
            loss.backward()
            for m, p in zip(master, params):
                m.grad = p.grad.float()
                p.grad = None
            opt.step()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                for m, p in zip(master, params):
                    p.copy_(m)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            losses.append(loss.item())
        model.eval()
        curves[impl] = {"loss": losses, "step_s": times}
        log(f"  {impl:<9} loss {losses[0]:.4f} -> {losses[-1]:.4f}   median step {1e3 * statistics.median(times[2:]):.0f} ms")
        del master, opt
        _free()
    d = [abs(a - b) for a, b in zip(curves["sdpa"]["loss"], curves["flashtile"]["loss"])]
    curves["max_abs_loss_diff"] = max(d)
    curves["first_step_abs_loss_diff"] = d[0]
    log(f"  |loss(SDPA) - loss(FlashTile)|: step 0 {d[0]:.2e}, max over {TRAIN_STEPS} steps {max(d):.2e}")
    with torch.no_grad():
        for p, p0 in zip(params, init):
            p.copy_(p0)
    return curves


def train_sweep(model, ids_all, log):
    _trainable(model)
    rows = []
    for L in TRAIN_SWEEP:
        ids = ids_all[:, :L]
        for impl in IMPLS:
            row = {"L": L, "impl": impl, "status": "ok"}
            try:
                _set(model, impl)
                model.train()
                _lm_loss(model, ids).backward()  # warmup / autotune
                model.zero_grad(set_to_none=True)
                base = _peak_reset()
                times = []
                for _ in range(2):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    _lm_loss(model, ids).backward()
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - t0)
                    model.zero_grad(set_to_none=True)
                row.update(ms=1e3 * min(times), peak_mib=_peak_mib(base), tok_per_s=L / min(times))
            except torch.OutOfMemoryError as err:
                row.update(status="oom", note=str(err).splitlines()[0][:160])
            except Exception as err:  # noqa: BLE001
                row.update(status="error", note=f"{type(err).__name__}: {str(err)[:200]}")
                log(traceback.format_exc())
            model.zero_grad(set_to_none=True)
            model.eval()
            _free()
            rows.append(row)
            if row["status"] == "ok":
                log(f"  seq_len={L:>6} {impl:<9} OK (fwd+bwd={row['ms']:.0f}ms, mem={row['peak_mib']:.0f}MB, "
                    f"{row['tok_per_s']:.0f} tok/s)")
            else:
                log(f"  seq_len={L:>6} {impl:<9} {row['status'].upper()} ({row.get('note', '')[:100]})")
    return rows



def figures(res, out_dir, gpu):
    plt = style.plt()
    lab = {"eager": style.LABELS["eager"], "sdpa": style.LABELS["sdpa"], "flashtile": style.LABELS["flashtile"]}

    def kw(impl, i):
        """SDPA dashed under FlashTile, so both lines show up when they overlap."""
        if impl == "sdpa":
            return dict(marker=style.MARKERS[i], markersize=11, markerfacecolor="none", markeredgewidth=2,
                        color=style.COLORS[impl], linewidth=2.5, linestyle="--", zorder=2, label=lab[impl])
        return dict(marker=style.MARKERS[i], markersize=6, color=style.COLORS[impl], linewidth=2, zorder=3,
                    label=lab[impl])

    def sweep(ax, rows, field, ylabel, ttl, logy=True):
        lengths = sorted({r["L"] for r in rows})
        for i, impl in enumerate(IMPLS):
            pts = [(r["L"], r[field]) for r in rows if r["impl"] == impl and r["status"] == "ok"]
            if pts:
                ax.plot(*zip(*pts), **kw(impl, i))
            for r in rows:
                if r["impl"] == impl and r["status"] != "ok":
                    ax.axvline(r["L"], color=style.COLORS[impl], linestyle=":", alpha=0.6)
                    ax.annotate(f"{lab[impl].split(' (')[0]} {r['status'].upper()}", (r["L"], 0.5),
                                xycoords=("data", "axes fraction"), color=style.COLORS[impl], rotation=90,
                                fontsize=8, fontweight="bold", ha="right")
                    break
        ax.set_xscale("log", base=2)
        ax.set_xticks(lengths)
        ax.set_xticklabels([str(L) for L in lengths])
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Sequence Length (tokens)")
        ax.set_ylabel(ylabel)
        style.title(ax, ttl)
        style.legend(ax)

    head = f"Qwen3.5-4B (FP8 8-bit) with FlashTile  |  {gpu}"
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    style.style_axes(fig, axes.flat)
    pre = res.get("prefill", [])
    if pre:
        sweep(axes[0][0], pre, "ms", "Whole-model prefill time (ms)", "Prefill Latency")
        sweep(axes[0][1], pre, "peak_mib", "Peak memory above weights (MB)", "Prefill Memory")
        sweep(axes[1][0], pre, "attn_ms", "Time inside the 8 attention layers (ms)", "Attention Time")
    ax = axes[1][1]
    curves = res.get("training", {}).get("curves", {})
    for i, impl in enumerate(("sdpa", "flashtile")):
        if impl not in curves:
            continue
        c = curves[impl]
        ax.plot(range(1, len(c["loss"]) + 1), c["loss"], **kw(impl, i))
    if "max_abs_loss_diff" in curves:
        ax.text(0.98, 0.95, f"max |Δloss| = {curves['max_abs_loss_diff']:.1e}", transform=ax.transAxes, ha="right",
                va="top", color=style.TEXT, fontsize=11, fontweight="bold")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlabel("Training step (bf16 model, 8 attention blocks trained)")
    ax.set_ylabel("Next-token loss")
    style.title(ax, "Fine-tuning Loss")
    style.legend(ax)
    style.suptitle(fig, head)
    style.save(fig, out_dir / "llm_dashboard.png")
    plt.close(fig)

    sw = res.get("training", {}).get("sweep", [])
    if sw:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))
        style.style_axes(fig, axes)
        sweep(axes[0], sw, "ms", "fwd+bwd time per step (ms)", "Training Step Time")
        sweep(axes[1], sw, "peak_mib", "Peak memory above weights (MB)", "Training Step Memory")
        style.suptitle(fig, f"Qwen3.5-4B (bf16) training step  |  {gpu}")
        style.save(fig, out_dir / "llm_training.png")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))
    style.style_axes(fig, axes)
    for ax, key, ttl in ((axes[0], "fp8_forward", "Forward error, FP8 model activations"),
                         (axes[1], "bf16_train", "Gradient error, bf16 model + real loss gradient")):
        rows = res["precision"].get(key, [])
        if not rows:
            continue
        layers = [r["layer"] for r in rows]
        x = range(len(layers))
        if key == "fp8_forward":
            for i, impl in enumerate(("sdpa", "flashtile")):
                ax.bar([j + (i - 0.5) * 0.38 for j in x], [r[impl]["out_max_abs"] for r in rows], width=0.38,
                       color=style.COLORS[impl], label=lab[impl])
            ax.set_ylabel("max |output - fp64 oracle|")
        else:
            for impl in ("sdpa", "flashtile"):
                for n, mk in (("dq_rel", "o"), ("dk_rel", "s"), ("dv_rel", "^")):
                    ax.plot(list(x), [r[impl][n] for r in rows], marker=mk, color=style.COLORS[impl], linewidth=2,
                            linestyle="-" if impl == "flashtile" else "--", label=f"{lab[impl]} {n[:2]}")
            ax.set_ylabel("max |grad - fp64 oracle| / max |oracle grad|")
            ax.set_yscale("log")
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(L) for L in layers])
        ax.set_xlabel("Model layer (full-attention layers only)")
        style.title(ax, ttl)
        style.legend(ax, fontsize=8, ncol=2)
    style.suptitle(fig, f"Qwen3.5-4B real-activation precision (bf16 kernels vs fp64)  |  {gpu}")
    style.save(fig, out_dir / "llm_precision.png")
    plt.close(fig)



def run(out_dir: pathlib.Path, corpus_dir: pathlib.Path, quick: bool = False):
    from transformers import AutoTokenizer
    import transformers

    global PRECISION_L, PREFILL_LENGTHS, GEN_TOKENS, TRAIN_L, TRAIN_STEPS, TRAIN_SWEEP
    if quick:  # smoke test on a small GPU: same code paths, small sizes
        PRECISION_L, PREFILL_LENGTHS, GEN_TOKENS = 1024, [1024, 4096], 32
        TRAIN_L, TRAIN_STEPS, TRAIN_SWEEP = 1024, 4, [1024, 2048]

    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "llm_log.txt", "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    torch.backends.cuda.matmul.allow_tf32 = False
    _register()
    props = torch.cuda.get_device_properties(0)
    gpu = props.name
    res = {"gpu": gpu, "model_8bit": MODEL_8BIT, "model_bf16": MODEL_BF16, "transformers": transformers.__version__,
           "torch": torch.__version__, "precision": {}, "errors": {}}
    log("=" * 78)
    log("FlashTile in a real LLM: Qwen3.5-4B")
    log("=" * 78)
    log(f"GPU: {gpu}  sm_{props.major}{props.minor}  {props.total_memory / 2**30:.0f} GiB")
    log(f"torch {torch.__version__}  transformers {transformers.__version__}")
    log(f"8-bit model: {MODEL_8BIT} (FP8 E4M3 weights per channel, FP8 activations per token, compressed-tensors)")
    log(f"bf16 model:  {MODEL_BF16} (training section)")

    tok = AutoTokenizer.from_pretrained(MODEL_BF16)
    corpus = "\n\n".join((corpus_dir / f).read_text(encoding="utf-8") for f in CORPUS_FILES)
    ids_all = tok(corpus, return_tensors="pt").input_ids.cuda()
    need = max(PREFILL_LENGTHS + TRAIN_SWEEP)
    if ids_all.shape[1] < need:
        ids_all = ids_all.repeat(1, math.ceil(need / ids_all.shape[1]))
    log(f"Text: FlashTile docs ({', '.join(CORPUS_FILES)}), {len(tok(corpus).input_ids)} tokens")

    t0 = time.perf_counter()
    model = _load(MODEL_8BIT)
    layers = _attn_layers(model)
    cfg = model.config.text_config
    res["model"] = {
        "load_s": time.perf_counter() - t0, "weights_mib": torch.cuda.memory_allocated() / 2**20,
        "layers": cfg.num_hidden_layers, "full_attention_layers": [m.layer_idx for m in layers],
        "q_heads": cfg.num_attention_heads, "kv_heads": cfg.num_key_value_heads, "head_dim": cfg.head_dim,
    }
    log("Note (measured with torch.profiler on H200): transformers runs this compressed-tensors FP8 checkpoint by "
        "quantize-dequantize in PyTorch and the same bf16 GEMM kernels as the bf16 model, so 8-bit saves weight "
        "memory, not time; at batch 1 below ~8k tokens the whole model is limited by CPU kernel launches, so "
        "attention time is reported separately (CUDA events around each attention call).")
    log(f"Loaded in {res['model']['load_s']:.0f}s, {res['model']['weights_mib']:.0f} MB on GPU. "
        f"{cfg.num_hidden_layers} layers; softmax attention in layers {res['model']['full_attention_layers']} "
        f"(Hq={cfg.num_attention_heads}, Hkv={cfg.num_key_value_heads}, d={cfg.head_dim}); the rest are Gated DeltaNet.")
    log("=" * 78)

    sections = [
        ("precision_fp8", "📊 Precision on real activations (FP8 model, forward, L=%d)" % PRECISION_L,
         lambda: res["precision"].__setitem__("fp8_forward", precision(model, ids_all[:, :PRECISION_L], False, log))),
        ("prefill", "📊 Long-context prefill (FP8 model, batch 1)",
         lambda: res.__setitem__("prefill", prefill(model, ids_all, log))),
        ("generation", "📊 Greedy generation (FP8 model, chat template, thinking off, %d new tokens)" % GEN_TOKENS,
         lambda: res.__setitem__("generation", generation(model, tok, corpus_dir, log))),
    ]
    for key, ttl, fn in sections:
        log(f"\n{ttl}...")
        try:
            fn()
        except Exception as err:  # noqa: BLE001 - one failing section must not hide the others
            res["errors"][key] = f"{type(err).__name__}: {err}"
            log(f"  SECTION FAILED: {type(err).__name__}: {err}\n{traceback.format_exc()}")
        _free()
    del model
    _free()

    model = _load(MODEL_BF16)
    res["model"]["bf16_weights_mib"] = torch.cuda.memory_allocated() / 2**20
    log(f"\nLoaded {MODEL_BF16}: {res['model']['bf16_weights_mib']:.0f} MB on GPU "
        f"(8-bit checkpoint: {res['model']['weights_mib']:.0f} MB)")
    res["training"] = {}
    sections = [
        ("precision_bf16", "📊 Precision with real gradients (bf16 model, training loss, L=%d)" % PRECISION_L,
         lambda: res["precision"].__setitem__("bf16_train", precision(model, ids_all[:, :PRECISION_L], True, log))),
        ("training", "📊 Fine-tuning, SDPA vs FlashTile from the same weights (L=%d, %d steps)" % (TRAIN_L, TRAIN_STEPS),
         lambda: res["training"].__setitem__("curves", train_curves(model, ids_all, log))),
        ("train_sweep", "📊 Training step (fwd+bwd) vs sequence length (bf16 model, batch 1)",
         lambda: res["training"].__setitem__("sweep", train_sweep(model, ids_all, log))),
    ]
    for key, ttl, fn in sections:
        log(f"\n{ttl}...")
        try:
            fn()
        except Exception as err:  # noqa: BLE001
            res["errors"][key] = f"{type(err).__name__}: {err}"
            log(f"  SECTION FAILED: {type(err).__name__}: {err}\n{traceback.format_exc()}")
        _free()
    del model
    _free()

    log("\n" + "=" * 78)
    log("Summary")
    log("=" * 78)
    _summary(res, log)
    (out_dir / "llm_results.json").write_text(json.dumps(res, indent=2, default=str))
    _generations_md(res, out_dir / "generations.md")
    try:
        figures(res, out_dir, gpu)
        log("\nFigures: llm_dashboard.png, llm_training.png, llm_precision.png")
    except Exception as err:  # noqa: BLE001
        log(f"Figure generation failed: {err}\n{traceback.format_exc()}")
    log_file.close()
    return res


def _summary(res, log):
    for key, name in (("fp8_forward", "FP8 forward"), ("bf16_train", "bf16 fwd+bwd")):
        rows = res["precision"].get(key)
        if rows:
            ft = max(r["flashtile"]["out_max_abs"] for r in rows)
            sd = max(r["sdpa"]["out_max_abs"] for r in rows)
            log(f"  {name} precision, worst layer: FlashTile out err {ft:.2e} vs SDPA {sd:.2e}")
            if key == "bf16_train":
                for n in ("dq_rel", "dk_rel", "dv_rel"):
                    log(f"    {n[:2]}: FlashTile {max(r['flashtile'][n] for r in rows):.2e} vs SDPA {max(r['sdpa'][n] for r in rows):.2e}")
    for r in res.get("prefill", []):
        if r["impl"] == "flashtile" and r["status"] == "ok":
            sd = next((x for x in res["prefill"] if x["L"] == r["L"] and x["impl"] == "sdpa" and x["status"] == "ok"), None)
            if sd:
                log(f"  prefill L={r['L']:>6}: FlashTile {r['ms']:.0f} ms vs SDPA {sd['ms']:.0f} ms "
                    f"(attention {r['attn_ms']:.0f} vs {sd['attn_ms']:.0f} ms), top-1 agree "
                    f"{100 * r.get('top1_agreement_vs_sdpa', float('nan')):.1f}%")
    for name, g in res.get("generation", {}).items():
        log(f"  generation [{name}]: {'identical' if g['identical'] else 'diverged at token %d' % g['identical_prefix_tokens']}")
    c = res.get("training", {}).get("curves")
    if c:
        log(f"  training: max |loss(SDPA) - loss(FlashTile)| over {TRAIN_STEPS} steps = {c['max_abs_loss_diff']:.2e}")
    if res["errors"]:
        log(f"  sections that failed: {list(res['errors'])}")


def _generations_md(res, path):
    md = ["# Qwen3.5-4B (FP8) greedy generations: SDPA vs FlashTile", ""]
    for name, g in res.get("generation", {}).items():
        md += [f"## {name} ({g['prompt_tokens']} prompt tokens)", "",
               f"Greedy outputs {'identical' if g['identical'] else 'diverge at token %d' % g['identical_prefix_tokens']}.", ""]
        for impl in ("sdpa", "flashtile"):
            r = g[impl]
            md += [f"### {impl} (prefill {r['prefill_ms']:.0f} ms, decode {r['decode_tok_per_s']:.1f} tok/s)", "",
                   "```text", r["text"].strip(), "```", ""]
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/local/llm")
    ap.add_argument("--corpus", default=str(HERE.parent / "corpus"),
                    help=f"folder holding {', '.join(CORPUS_FILES)} (the text the model reads)")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run(pathlib.Path(args.out), pathlib.Path(args.corpus), quick=args.quick)
