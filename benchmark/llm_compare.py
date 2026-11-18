"""Qwen3.8-27B with four attention paths, same five prompts: SDPA (PyTorch's own pick), FA2
(SDPA forced to its built-in FlashAttention-2 kernel), cuDNN (SDPA forced to cuDNN) and FlashTile.

Qwen3.8-27B has 64 layers; every 4th one is softmax attention (24 q heads, 4 kv heads, d=256),
the other 48 are Gated DeltaNet. Only those 16 attention layers change between paths. bf16
weights, batch 1, greedy decoding, thinking off.

Per prompt and path: prefill time, time inside attention, decode speed, peak memory, the answer,
and how close logits and greedy tokens are to SDPA's. The two long prompts are this repo's own
docs and source, so run it from the repo root (or pass --root).

    python benchmark/llm_compare.py --out results/local/qwen27b
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import statistics
import sys
import time
import traceback
from collections import Counter

import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import style  # noqa: E402
from llm_eval import Timer, _free, _peak_mib, _peak_reset  # noqa: E402

import flashtile.hf as fhf  # noqa: E402

MODEL = "Qwen/Qwen3.8-27B"
PATHS = ["sdpa", "fa2", "cudnn", "flashtile"]  # sdpa first: the others are compared with it
REGISTERED = {"sdpa": "sdpa_t", "fa2": "fa2_t", "cudnn": "cudnn_t", "flashtile": "ftile_t"}
LABELS = {"sdpa": "SDPA (default pick)", "fa2": "FA2 (SDPA flash backend)", "cudnn": "cuDNN (SDPA cuDNN backend)",
          "flashtile": "FlashTile"}
COLORS = {"sdpa": "#BDC3C7", "fa2": style.COLORS["sdpa-flash"], "cudnn": style.COLORS["sdpa-cudnn"],
          "flashtile": style.COLORS["flashtile"]}
PREFILL_REPEATS = 3
KEEP_LOGITS = 64  # last positions whose logits get compared between paths
GEN_TOKENS = 256
FALLBACKS: Counter = Counter()  # calls a forced backend refused and plain SDPA ran instead

DOCS = ["README.md", "docs/USAGE.md", "docs/PERFORMANCE.md", "CHANGELOG.md", "CONTRIBUTING.md"]
SOURCE = ["flashtile/contract.py", "flashtile/reference.py", "flashtile/tiled.py", "flashtile/triton_kernels.py",
          "flashtile/api.py", "flashtile/hf.py", "benchmark/bench_attention.py", "benchmark/llm_eval.py",
          "benchmark/report.py"]


def queries(root: pathlib.Path) -> dict[str, str]:
    def bundle(files):
        return "\n\n".join(f"===== {f} =====\n{(root / f).read_text(encoding='utf-8')}" for f in files)

    return {
        "short": "Explain in three sentences what grouped-query attention is and why it saves memory when decoding.",
        "math": "A tank holds 300 litres and starts with 30 litres. Water flows in at 12 litres per minute and drains "
                "at 4.5 litres per minute. How many minutes until it is full? Show the steps briefly.",
        "code": "Write a Python function that merges overlapping intervals in a list of (start, end) pairs, "
                "plus a few asserts that test it.",
        "docs_qa": bundle(DOCS) + "\n\n---\nUsing only the documents above: on which GPUs, and for which cases, is "
                   "FlashTile slower than another attention backend, and what does the README list as its weak spots? "
                   "Answer in under 150 words.",
        "code_qa": bundle(SOURCE) + "\n\n---\nUsing only the code above: how does flashtile/hf.py decide between "
                   "the causal Triton path, the decode path and the masked tiled path, and what does the Triton "
                   "forward kernel do differently on diagonal tiles? Answer in under 150 words.",
    }


def _register():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.masking_utils import sdpa_mask

    def forced(backend):
        def fn(module, query, key, value, attention_mask, **kw):
            try:
                with sdpa_kernel(backend):
                    return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)
            except RuntimeError as err:
                if "No available kernel" not in str(err):
                    raise
                FALLBACKS[backend.name] += 1
                return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)

        return fn

    fns = {"sdpa": sdpa_attention_forward, "fa2": forced(SDPBackend.FLASH_ATTENTION),
           "cudnn": forced(SDPBackend.CUDNN_ATTENTION), "flashtile": fhf.flashtile_attention_forward}
    for path, fn in fns.items():  # none of these names may contain "flash", see flashtile/hf.py
        AttentionInterface.register(REGISTERED[path], Timer.wrap(fn))
        AttentionMaskInterface.register(REGISTERED[path], sdpa_mask)


def _kernels(model, ids) -> set[str]:
    """CUDA kernel names launched by one prefill."""
    from torch.profiler import ProfilerActivity, profile

    with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA]) as prof:
        model(input_ids=ids, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()
    # cudaGetErrorString & co. are runtime API calls the profiler lists too, not kernels
    return {e.key for e in prof.key_averages() if not e.key.startswith("cuda")}


def _prefill(model, ids):
    with torch.no_grad():
        model(input_ids=ids, use_cache=False, logits_to_keep=KEEP_LOGITS)  # warmup / autotune
        base = _peak_reset()
        ms, attn = [], []
        for _ in range(PREFILL_REPEATS):
            torch.cuda.synchronize()
            Timer.start()
            t0 = time.perf_counter()
            out = model(input_ids=ids, use_cache=False, logits_to_keep=KEEP_LOGITS)
            attn.append(Timer.stop_ms())  # syncs
            ms.append((time.perf_counter() - t0) * 1e3)
        peak = _peak_mib(base)
    return {"ms": statistics.median(ms), "attn_ms": statistics.median(attn), "all_ms": ms,
            "peak_mib": peak}, out.logits[0].float().cpu()


def _generate(model, ids):
    kw = dict(max_new_tokens=GEN_TOKENS, do_sample=False, temperature=None, top_p=None, top_k=None)
    with torch.no_grad():
        warm = model.generate(ids, **kw)  # decode shapes get autotuned outside the timing
        base = _peak_reset()
        fhf.stats.clear()
        FALLBACKS.clear()
        Timer.start()
        t0 = time.perf_counter()
        out = model.generate(ids, **kw)
        attn_ms = Timer.stop_ms()
        total_ms = (time.perf_counter() - t0) * 1e3
        peak = _peak_mib(base)
    n = ids.shape[1]
    return {"tokens": out[0, n:].tolist(), "repeatable": out[0, n:].tolist() == warm[0, n:].tolist(),
            "total_ms": total_ms, "attn_ms": attn_ms, "peak_mib": peak,
            "flashtile_calls": dict(fhf.stats), "fallbacks": dict(FALLBACKS)}


def _same_prefix(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def _logit_stats(ref, x):
    lp_ref, lp = F.log_softmax(ref, -1), F.log_softmax(x, -1)
    return {"max_abs": (ref - x).abs().max().item(), "mean_abs": (ref - x).abs().mean().item(),
            "top1_agree": (ref.argmax(-1) == x.argmax(-1)).float().mean().item(),
            "kl": (lp_ref.exp() * (lp_ref - lp)).sum(-1).mean().item()}


def run(out_dir: pathlib.Path, root: pathlib.Path, model_name: str = MODEL, save=lambda: None):
    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "compare_log.txt", "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    torch.backends.cuda.matmul.allow_tf32 = False
    _register()
    props = torch.cuda.get_device_properties(0)
    res = {"gpu": props.name, "model": model_name, "torch": torch.__version__, "cudnn": torch.backends.cudnn.version(),
           "transformers": transformers.__version__, "paths": LABELS, "gen_tokens": GEN_TOKENS,
           "prefill_repeats": PREFILL_REPEATS, "keep_logits": KEEP_LOGITS, "queries": {}, "errors": {}}
    log("=" * 78)
    log(f"{model_name}: SDPA vs FA2 vs cuDNN vs FlashTile, same five prompts")
    log("=" * 78)
    log(f"GPU: {props.name}  sm_{props.major}{props.minor}  {props.total_memory / 2**30:.0f} GiB")
    log(f"torch {torch.__version__}  cuDNN {res['cudnn']}  transformers {transformers.__version__}")
    log("FA2 = PyTorch's built-in FlashAttention-2 kernel (SDPBackend.FLASH_ATTENTION). The flash-attn package "
        "has no wheels for torch 2.15 / CUDA 13, so attn_implementation='flash_attention_2' isn't an option here.")

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda",
                                                        attn_implementation="sdpa")
    model.eval()
    cfg = model.config.text_config
    attn_layers = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    res["load"] = {"s": time.perf_counter() - t0, "weights_mib": torch.cuda.memory_allocated() / 2**20,
                   "layers": cfg.num_hidden_layers, "attention_layers": attn_layers,
                   "q_heads": cfg.num_attention_heads, "kv_heads": cfg.num_key_value_heads, "head_dim": cfg.head_dim}
    log(f"Loaded in {res['load']['s']:.0f}s (download included on the first run), "
        f"{res['load']['weights_mib'] / 1024:.1f} GiB on GPU")
    log(f"{cfg.num_hidden_layers} layers, softmax attention in {len(attn_layers)} of them {attn_layers} "
        f"(Hq={cfg.num_attention_heads}, Hkv={cfg.num_key_value_heads}, d={cfg.head_dim}), the rest Gated DeltaNet")
    save()

    prompts = queries(root)
    ids = {}
    for name, prompt in prompts.items():
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        ids[name] = tok(text, return_tensors="pt").input_ids.cuda()
    log("Prompts: " + ", ".join(f"{n} ({x.shape[1]} tokens)" for n, x in ids.items()))
    log(f"Each path: 1 warmup + {PREFILL_REPEATS} timed prefills (median), then 1 warmup + 1 timed greedy "
        f"generation of up to {GEN_TOKENS} tokens. Attention time = CUDA events around every attention call.")

    log("\n📊 Which attention kernels actually ran (torch.profiler, one prefill of the 'short' prompt)...")
    try:
        seen = {}
        for path in PATHS:
            model.set_attn_implementation(REGISTERED[path])
            with torch.no_grad():
                model(input_ids=ids["short"], use_cache=False, logits_to_keep=1)  # warmup / autotune
            seen[path] = _kernels(model, ids["short"])
        common = set.intersection(*seen.values())
        res["kernels"] = {p: sorted(s - common) for p, s in seen.items()}
        for path in PATHS:
            names = res["kernels"][path]
            log(f"  {LABELS[path]}:")
            for k in names[:8] or ["(nothing that the other paths don't also launch)"]:
                log(f"      {k[:150]}")
    except Exception as err:  # noqa: BLE001 - nice to have, not worth losing the run over
        res["errors"]["kernels"] = f"{type(err).__name__}: {err}"
        log(f"  profiler failed: {type(err).__name__}: {err}")
    _free()
    save()

    for name, x in ids.items():
        log(f"\n📊 [{name}] {x.shape[1]} prompt tokens")
        q = res["queries"][name] = {"prompt_tokens": x.shape[1], "prompt_tail": prompts[name][-400:]}
        logits = {}
        for path in PATHS:
            model.set_attn_implementation(REGISTERED[path])
            try:
                pre, logits[path] = _prefill(model, x)
                gen = _generate(model, x)
            except Exception as err:  # noqa: BLE001 - one failing path must not hide the others
                q[path] = {"error": f"{type(err).__name__}: {err}"}
                res["errors"][f"{name}/{path}"] = q[path]["error"]
                log(f"  {path:<9} FAILED: {type(err).__name__}: {err}\n{traceback.format_exc()}")
                _free()
                continue
            n_new = len(gen["tokens"])
            decode_ms = max(gen["total_ms"] - pre["ms"], 1e-6)
            q[path] = {"prefill": pre, "generate": gen, "new_tokens": n_new,
                       "decode_tok_per_s": (n_new - 1) / decode_ms * 1e3,
                       "decode_attn_ms_per_token": (gen["attn_ms"] - pre["attn_ms"]) / max(n_new - 1, 1),
                       "text": tok.decode(gen["tokens"], skip_special_tokens=True)}
            extra = ""
            if gen["flashtile_calls"]:
                extra += f"  FlashTile calls={gen['flashtile_calls']}"
            if gen["fallbacks"]:
                extra += f"  FELL BACK to default SDPA: {gen['fallbacks']}"
            log(f"  {path:<9} prefill {pre['ms']:7.0f} ms (attention {pre['attn_ms']:6.1f} ms)  "
                f"decode {q[path]['decode_tok_per_s']:5.1f} tok/s (attention {q[path]['decode_attn_ms_per_token']:.2f} "
                f"ms/token)  peak +{pre['peak_mib']:.0f} MiB prefill / +{gen['peak_mib']:.0f} MiB generate  "
                f"{n_new} tokens{'' if gen['repeatable'] else '  (NOT repeatable run to run)'}{extra}")
            _free()
        if "sdpa" in logits:
            for path in PATHS[1:]:
                if path in logits:
                    q[path]["logits_vs_sdpa"] = s = _logit_stats(logits["sdpa"], logits[path])
                    log(f"  {path:<9} vs SDPA, last {KEEP_LOGITS} positions: max |dlogit| {s['max_abs']:.3f}  "
                        f"mean {s['mean_abs']:.4f}  top-1 agree {100 * s['top1_agree']:.1f}%  KL {s['kl']:.2e}")
        ok = [p for p in PATHS if "generate" in q.get(p, {})]
        q["same_prefix"] = {f"{a}|{b}": _same_prefix(q[a]["generate"]["tokens"], q[b]["generate"]["tokens"])
                            for a, b in itertools.combinations(ok, 2)}
        for a, b in itertools.combinations(ok, 2):
            ta, tb = q[a]["generate"]["tokens"], q[b]["generate"]["tokens"]
            same = q["same_prefix"][f"{a}|{b}"]
            log(f"  greedy {a} vs {b}: " + ("IDENTICAL" if ta == tb else f"diverge at token {same}")
                + f" ({len(ta)} vs {len(tb)} tokens)")
        if "flashtile" in ok:
            log("  FlashTile answer: " + q["flashtile"]["text"].strip().replace("\n", "\n    ")[:1200])
        del logits
        _free()
        (out_dir / "compare_results.json").write_text(json.dumps(res, indent=2, default=str))
        save()

    log("\n" + "=" * 78)
    log("Summary")
    log("=" * 78)
    _summary(res, log)
    (out_dir / "compare_results.json").write_text(json.dumps(res, indent=2, default=str))
    _generations_md(res, out_dir / "generations.md")
    try:
        figure(res, out_dir / "compare_dashboard.png")
        log("\nFigure: compare_dashboard.png")
    except Exception as err:  # noqa: BLE001
        log(f"Figure failed: {err}\n{traceback.format_exc()}")
    log_file.close()
    save()
    return res


def _ok(q, path):
    return "prefill" in q.get(path, {})


def _summary(res, log):
    log(f"{'prompt':<9}{'tokens':>7} | " + " | ".join(f"{p:^21}" for p in PATHS))
    log(f"{'':<9}{'':>7} | " + " | ".join(f"{'prefill/attn ms, tok/s':^21}" for _ in PATHS))
    for name, q in res["queries"].items():
        cells = []
        for p in PATHS:
            if _ok(q, p):
                cells.append(f"{q[p]['prefill']['ms']:6.0f}/{q[p]['prefill']['attn_ms']:5.1f} {q[p]['decode_tok_per_s']:5.1f}")
            else:
                cells.append("failed")
        log(f"{name:<9}{q['prompt_tokens']:>7} | " + " | ".join(f"{c:^21}" for c in cells))
    log("")
    for name, q in res["queries"].items():
        if _ok(q, "flashtile"):
            parts = []
            for p in ("sdpa", "fa2", "cudnn"):
                if _ok(q, p):
                    parts.append(f"{p} {q[p]['prefill']['attn_ms'] / q['flashtile']['prefill']['attn_ms']:.2f}x")
            log(f"  [{name}] prefill attention time / FlashTile's: " + ", ".join(parts))
    for name, q in res["queries"].items():
        same = q.get("same_prefix", {})
        log(f"  [{name}] greedy tokens matching SDPA: "
            + ", ".join(f"{p} {same.get('sdpa|' + p, '-')}/{q[p]['new_tokens'] if _ok(q, p) else '-'}"
                        for p in PATHS[1:]))
    if res["errors"]:
        log(f"  failures: {list(res['errors'])}")


def _generations_md(res, path):
    md = [f"# {res['model']} greedy answers on {res['gpu']}: SDPA vs FA2 vs cuDNN vs FlashTile", "",
          f"bf16, batch 1, thinking off, up to {res['gen_tokens']} new tokens. Only the {len(res['load']['attention_layers'])} "
          "softmax attention layers change between paths.", ""]
    for name, q in res["queries"].items():
        md += [f"## {name} ({q['prompt_tokens']} prompt tokens)", ""]
        tail = q["prompt_tail"] if q["prompt_tokens"] < 300 else "... " + q["prompt_tail"].split("---")[-1].strip()
        md += ["Prompt" + (" (repo files, then the question)" if q["prompt_tokens"] >= 300 else "") + ":", "",
               "> " + tail.replace("\n", " "), ""]
        same = q.get("same_prefix", {})
        for p in PATHS:
            r = q.get(p, {})
            if not _ok(q, p):
                md += [f"### {LABELS[p]}", "", f"Failed: {r.get('error')}", ""]
                continue
            agree = "" if p == "sdpa" else f", first {same.get('sdpa|' + p, '-')} tokens identical to SDPA"
            md += [f"### {LABELS[p]} (prefill {r['prefill']['ms']:.0f} ms, decode {r['decode_tok_per_s']:.1f} tok/s{agree})",
                   "", "```text", r["text"].strip(), "```", ""]
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


def figure(res, path):
    plt = style.plt()
    names = list(res["queries"])
    xlabels = [f"{n}\n{res['queries'][n]['prompt_tokens']} tok" for n in names]
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    style.style_axes(fig, axes.flat)
    width = 0.8 / len(PATHS)

    def bars(ax, value, paths=PATHS, log_y=False):
        for i, p in enumerate(paths):
            ys = [value(res["queries"][n], p) if _ok(res["queries"][n], p) else 0 for n in names]
            xs = [j + (i - (len(paths) - 1) / 2) * width for j in range(len(names))]
            ax.bar(xs, ys, width, color=COLORS[p], label=LABELS[p])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(xlabels)
        if log_y:
            ax.set_yscale("log")

    bars(axes[0, 0], lambda q, p: q[p]["prefill"]["ms"], log_y=True)
    axes[0, 0].set_ylabel("ms (log)")
    style.title(axes[0, 0], "Prefill, whole model")
    bars(axes[0, 1], lambda q, p: q[p]["prefill"]["attn_ms"], log_y=True)
    axes[0, 1].set_ylabel("ms (log)")
    style.title(axes[0, 1], f"Prefill, time inside the {len(res['load']['attention_layers'])} attention layers")
    bars(axes[1, 0], lambda q, p: q[p]["decode_tok_per_s"])
    axes[1, 0].set_ylabel("tokens / s")
    style.title(axes[1, 0], "Greedy decode speed")
    bars(axes[1, 1], lambda q, p: q[p].get("logits_vs_sdpa", {}).get("max_abs", 0), paths=PATHS[1:], log_y=True)
    axes[1, 1].set_ylabel("max |logit difference| (log)")
    style.title(axes[1, 1], f"Logits vs SDPA, last {res['keep_logits']} positions")
    for ax in axes.flat:
        style.legend(ax, fontsize=9)
    style.suptitle(fig, f"{res['model']} (bf16): SDPA vs FA2 vs cuDNN vs FlashTile  |  {res['gpu']}")
    style.save(fig, path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/local/qwen27b")
    ap.add_argument("--root", default=str(HERE.parent), help="repo root, the long prompts read its docs and source")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    run(pathlib.Path(args.out), pathlib.Path(args.root), args.model)
