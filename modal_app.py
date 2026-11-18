# Runs the tests, validation, kernel benchmark and Qwen3.5 evaluation on Modal GPUs.
#    modal run modal_app.py                                       # H200, B200, RTX PRO 6000
#    modal run modal_app.py --gpus B200 --skip-bench --skip-llm   # just tests + validation
#    modal run modal_app.py --gpus H200 --quick                   # small sizes, every stage
#    modal run modal_app.py --qwen27b                             # Qwen3.8-27B: SDPA/FA2/cuDNN/FlashTile, H200/H100/B200

from __future__ import annotations

import datetime
import json
import os
import pathlib
import subprocess
import sys
import traceback

import modal

ROOT = pathlib.Path(__file__).parent
REMOTE = "/root/flashtile"
# Text for the Qwen3.5 eval. Not in the repo: drop long .md files with these names into corpus/
# (or point FLASHTILE_CORPUS somewhere else), or run with --skip-llm.
CORPUS_DIR = pathlib.Path(os.environ.get("FLASHTILE_CORPUS", ROOT / "corpus"))
CORPUS = ["TECHNICAL-REFERENCE.md", "BLUEPRINT.md", "INTERVIEW-PRACTICE.md", "STUDY-GUIDE.md"]

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.15.0",  # CUDA 13 wheels; pulls triton 3.9.0
    "transformers==5.23.0",
    "compressed-tensors==0.20.1",  # loads the FP8 (W8A8) Qwen3.5 checkpoint
    "flash-linear-attention==0.6.0",  # fast Gated-DeltaNet kernels for Qwen3.5's linear-attention layers
    "accelerate==1.16.0",
    "pytest==8.4.2",
    "numpy",
    "matplotlib==3.10.6",
)
image = (
    image.add_local_dir(ROOT / "flashtile", f"{REMOTE}/flashtile", ignore=["__pycache__"])
    .add_local_dir(ROOT / "tests", f"{REMOTE}/tests", ignore=["__pycache__"])
    .add_local_dir(ROOT / "benchmark", f"{REMOTE}/benchmark", ignore=["__pycache__"])
    .add_local_file(ROOT / "pyproject.toml", f"{REMOTE}/pyproject.toml")
)
for name in CORPUS:
    if (CORPUS_DIR / name).exists():  # missing files only break the LLM stage, which reports it
        image = image.add_local_file(CORPUS_DIR / name, f"{REMOTE}/corpus/{name}")

app = modal.App("flashtile", image=image)
hf_cache = modal.Volume.from_name("flashtile-hf-cache", create_if_missing=True)  # model weights, downloaded once
results_vol = modal.Volume.from_name("flashtile-results", create_if_missing=True)  # stage-by-stage copies
VOLUMES = {"/root/.cache/huggingface": hf_cache, "/results": results_vol}

def _sh(cmd: str) -> str:
    run = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return f"$ {cmd}\n{run.stdout}{run.stderr}\n"

def _per_file_logs(junit: pathlib.Path, out: pathlib.Path, header: str) -> tuple[list[str], list[str]]:
    """Write tests/test_<file>.txt in pytest -v layout from one junit.xml (the suite runs once)."""
    import xml.etree.ElementTree as ET

    cases = ET.parse(junit).getroot().iter("testcase")
    by_file: dict[str, list] = {}
    for c in cases:
        mod = c.get("classname", "").split(".")[-1]
        outcome, detail = "PASSED", ""
        for tag, word in (("failure", "FAILED"), ("error", "ERROR"), ("skipped", "SKIPPED")):
            el = c.find(tag)
            if el is not None:
                outcome, detail = word, (el.get("message") or "") + "\n" + (el.text or "")
        by_file.setdefault(mod, []).append((c.get("name"), outcome, float(c.get("time", 0)), detail))
    summary, failures = [], []
    for mod, items in sorted(by_file.items()):
        lines = [header, f"collected {len(items)} items", ""]
        for i, (name, outcome, _, _) in enumerate(items, 1):
            lines.append(f"tests/{mod}.py::{name} {outcome}".ljust(78) + f"[{100 * i // len(items):3d}%]")
        bad = [(n, o, d) for n, o, _, d in items if o in ("FAILED", "ERROR")]
        for n, o, d in bad:
            lines += ["", f"{'_' * 20} {n} {'_' * 20}", d.strip()]
            failures.append(f"{o} tests/{mod}.py::{n}")
        counts = {o: sum(1 for _, x, _, _ in items if x == o) for o in ("PASSED", "FAILED", "ERROR", "SKIPPED")}
        secs = sum(t for _, _, t, _ in items)
        result = ", ".join(f"{v} {k.lower()}" for k, v in counts.items() if v)
        result_line = f" {result} in {secs:.2f}s "
        lines += ["", result_line.center(78, "=")]
        (out / f"{mod}.txt").write_text("\n".join(lines) + "\n")
        summary.append(f"  {mod}: {result_line.center(78, '=')}")
    return summary, failures

def full_run(expect: str, skip_tests=False, skip_bench=False, skip_llm=False, quick=False,
             pytest_args: str = "", run_id: str = "", folder: str = "") -> dict[str, bytes]:
    """Runs inside a GPU container; each per-GPU Modal function below calls it."""
    import torch
    import triton

    out = pathlib.Path("/tmp/out")
    out.mkdir(parents=True)
    props = torch.cuda.get_device_properties(0)
    if expect.lower() not in props.name.lower():  # never record one card under another's name
        (out / "WRONG_DEVICE.txt").write_text(f"requested {expect!r}, got {props.name!r}\n")
        return {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*") if p.is_file()}

    started = datetime.datetime.now(datetime.timezone.utc)
    env = {
        "date_utc": started.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": props.name,
        "compute_capability": f"sm_{props.major}{props.minor}",
        "gpu_memory_gib": round(props.total_memory / 2**30, 1),
        "sm_count": props.multi_processor_count,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "triton": triton.__version__,
        "provider": "Modal",
        "quick": quick,
    }
    (out / "environment.json").write_text(json.dumps(env, indent=2))
    (out / "system_info.txt").write_text(
        f"FlashTile run on {props.name}\n{json.dumps(env, indent=2)}\n\n"
        + _sh("nvidia-smi")
        + _sh("nvidia-smi --query-gpu=name,pci.bus_id,driver_version,memory.total,clocks.max.sm,"
              "clocks.max.mem,power.limit,temperature.gpu --format=csv")
        + _sh("nvidia-smi -q -d CLOCK,PERFORMANCE,POWER | head -80")
        + _sh("lscpu | head -20")
        + _sh("free -g")
        + _sh("uname -a")
    )
    (out / "pip_freeze.txt").write_text(_sh(f"{sys.executable} -m pip freeze"))
    sys.path[:0] = [f"{REMOTE}/benchmark", REMOTE]

    def save():  # copy everything finished so far to the results volume, so a timeout loses nothing
        if run_id and folder:
            import shutil

            shutil.copytree(out, f"/results/{run_id}/{folder}", dirs_exist_ok=True)
            results_vol.commit()

    save()

    def stage(name, fn):
        try:
            fn()
        except Exception:  # one failing stage must not lose the others' artifacts
            (out / f"{name.upper()}_CRASH.txt").write_text(traceback.format_exc())
        save()

    def tests():
        (out / "tests").mkdir()
        import shlex

        run = subprocess.run([sys.executable, "-m", "pytest", "-v", "-rA", "--durations=30",
                              "--junitxml=/tmp/out/tests/junit.xml", *shlex.split(pytest_args)],
                             cwd=REMOTE, capture_output=True, text=True)
        (out / "tests" / "full_suite.txt").write_text(run.stdout + run.stderr)
        header = f"platform linux -- Python {env['python']}, torch {env['torch']}, triton {env['triton']}, GPU {props.name}"
        per_file, failures = _per_file_logs(out / "tests" / "junit.xml", out / "tests", header)
        summary = [f"Test Summary — {props.name} — {env['date_utc']} UTC", f"Full suite exit code: {run.returncode}", ""]
        summary += per_file + ["", "Failures:" if failures else "Failures: none"] + [f"  {x}" for x in failures]
        (out / "tests" / "summary.txt").write_text("\n".join(summary) + "\n")

    def validation():
        import validate

        validate.run(out / "validation")

    def bench():
        import bench_attention
        import report

        rows = bench_attention.run(out / "benchmarks", lengths=[512, 2048] if quick else bench_attention.LENGTHS)
        report.gpu_outputs(rows, out / "benchmarks", env)

    def llm():
        import llm_eval

        llm_eval.run(out / "llm", pathlib.Path(f"{REMOTE}/corpus"), quick=quick)
        hf_cache.commit()

    if not skip_tests:
        stage("tests", tests)
    stage("validation", validation)
    if not skip_bench:
        stage("benchmarks", bench)
    if not skip_llm:
        stage("llm", llm)

    env["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    (out / "environment.json").write_text(json.dumps(env, indent=2))
    return {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*") if p.is_file()}

# optional Modal secret "huggingface" with HF_TOKEN, for faster model downloads; runs fine without it
HF_SECRET = []
if modal.is_local():
    try:
        modal.Secret.from_name("huggingface").hydrate()
        HF_SECRET = [modal.Secret.from_name("huggingface")]
    except modal.exception.NotFoundError:
        pass

# one Modal function per GPU so the dashboard shows the right card
KW = dict(timeout=4 * 3600, volumes=VOLUMES, secrets=HF_SECRET)

@app.function(gpu="H200", **KW)
def run_h200(skip_tests: bool, skip_bench: bool, skip_llm: bool, quick: bool, pytest_args: str, run_id: str = "",
         folder: str = "") -> dict[str, bytes]:
    return full_run("H200", skip_tests, skip_bench, skip_llm, quick, pytest_args, run_id, folder)

@app.function(gpu="B200", **KW)
def run_b200(skip_tests: bool, skip_bench: bool, skip_llm: bool, quick: bool, pytest_args: str, run_id: str = "",
         folder: str = "") -> dict[str, bytes]:
    return full_run("B200", skip_tests, skip_bench, skip_llm, quick, pytest_args, run_id, folder)

@app.function(gpu="RTX-PRO-6000", **KW)
def run_rtx_pro_6000(skip_tests: bool, skip_bench: bool, skip_llm: bool, quick: bool, pytest_args: str, run_id: str = "",
         folder: str = "") -> dict[str, bytes]:
    return full_run("RTX PRO 6000", skip_tests, skip_bench, skip_llm, quick, pytest_args, run_id, folder)

# Qwen3.8-27B, SDPA vs FA2 vs cuDNN vs FlashTile; the long prompts read the repo's docs and source
compare_image = image.add_local_dir(ROOT / "docs", f"{REMOTE}/docs", ignore=["*.png"])
for name in ("README.md", "CHANGELOG.md", "CONTRIBUTING.md"):
    compare_image = compare_image.add_local_file(ROOT / name, f"{REMOTE}/{name}")

@app.function(gpu=["H200", "H100", "B200"], image=compare_image, **KW)  # first one that's free
def compare_27b(run_id: str) -> tuple[str, dict[str, bytes]]:
    import shutil

    import torch

    out = pathlib.Path("/tmp/out")
    gpu = torch.cuda.get_device_name(0)
    folder = "qwen3.8-27b_" + next((g for g in ("H200", "H100", "B200") if g in gpu), gpu.replace(" ", "-"))
    sys.path[:0] = [f"{REMOTE}/benchmark", REMOTE]
    import llm_compare

    def save():
        shutil.copytree(out, f"/results/{run_id}/{folder}", dirs_exist_ok=True)
        results_vol.commit()

    try:
        llm_compare.run(out, pathlib.Path(REMOTE), save=save)
    except Exception:
        out.mkdir(parents=True, exist_ok=True)
        (out / "COMPARE_CRASH.txt").write_text(traceback.format_exc())
    hf_cache.commit()  # keep the 55 GB of weights for next time
    (out / "pip_freeze.txt").write_text(_sh(f"{sys.executable} -m pip freeze"))
    (out / "system_info.txt").write_text(_sh("nvidia-smi"))
    return folder, {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*") if p.is_file()}

# --gpus value -> (results folder, Modal function)
RUNNERS = {
    "H200": ("flashtile_results_H200", run_h200),
    "B200": ("flashtile_results_B200", run_b200),
    "RTX-PRO-6000": ("flashtile_results_RTX-PRO-6000", run_rtx_pro_6000),
}

@app.function(timeout=600)
def summarize(bench_jsons: dict[str, bytes]) -> dict[str, bytes]:
    """SUMMARY.md and cross_gpu_dashboard.png on a CPU container (matplotlib is not needed locally)."""
    sys.path.insert(0, f"{REMOTE}/benchmark")
    import report

    run_dir = pathlib.Path("/tmp/run")
    for folder, data in bench_jsons.items():
        p = run_dir / folder / "benchmarks" / "bench_results.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    report.summary(run_dir)
    return {n: (run_dir / n).read_bytes() for n in ("SUMMARY.md", "cross_gpu_dashboard.png") if (run_dir / n).exists()}

@app.local_entrypoint()
def main(gpus: str = "H200,B200,RTX-PRO-6000", skip_tests: bool = False, skip_bench: bool = False,
         skip_llm: bool = False, quick: bool = False, pytest_args: str = "", qwen27b: bool = False):
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "results" / stamp
    if qwen27b:  # only the Qwen3.8-27B comparison, on H200 (else H100, else B200)
        try:
            folder, files = compare_27b.remote(stamp)
        except Exception as err:
            folder, files = "qwen3.8-27b", {"RUN_FAILED.txt": f"{type(err).__name__}: {err}\n".encode()}
            subprocess.run(["modal", "volume", "get", "--force", "flashtile-results", stamp, str(run_dir)],
                           capture_output=True, text=True)
        for rel, data in files.items():
            path = run_dir / folder / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        print(f"\nSaved to {run_dir / folder}")
        return
    calls = {}
    for g in gpus.split(","):
        if g not in RUNNERS:
            raise SystemExit(f"unknown GPU {g!r}; choose from {', '.join(RUNNERS)}")
        folder, fn = RUNNERS[g]
        calls[folder] = fn.spawn(skip_tests, skip_bench, skip_llm, quick, pytest_args, stamp, folder)  # all GPUs run in parallel
    for folder, call in calls.items():
        try:
            files = call.get()
        except Exception as err:  # keep the other GPUs' results even if one container fails
            files = {"RUN_FAILED.txt": f"{type(err).__name__}: {err}\n".encode()}
            # recover every stage that finished before the failure from the results volume
            subprocess.run(["modal", "volume", "get", "--force", "flashtile-results", f"{stamp}/{folder}",
                            str(run_dir)], capture_output=True, text=True)
        for rel, data in files.items():
            path = run_dir / folder / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        summary = run_dir / folder / "tests" / "summary.txt"
        print(f"\n===== {folder} =====")
        print(summary.read_text() if summary.exists() else "\n".join(sorted(files)))
        for crash in sorted((run_dir / folder).glob("*_CRASH.txt")):
            print(f"--- {crash.name} ---\n{crash.read_text()[-3000:]}")
    jsons = {p.parts[-3]: p.read_bytes() for p in run_dir.glob("*/benchmarks/bench_results.json")}
    if jsons:
        try:
            for name, data in summarize.remote(jsons).items():
                (run_dir / name).write_bytes(data)
        except Exception as err:
            print(f"cross-GPU summary failed: {type(err).__name__}: {err}")
    flags = "".join(f" --{n.replace('_', '-')}" for n, v in
                    (("skip_tests", skip_tests), ("skip_bench", skip_bench), ("skip_llm", skip_llm), ("quick", quick)) if v)
    (run_dir / "RUN_INFO.md").write_text(
        f"# Run {stamp}\n\nCommand: `modal run modal_app.py --gpus {gpus}{flags}`\n\n"
        "Each flashtile_results_<gpu>/ folder holds the full test, validation, benchmark and LLM logs. "
        "SUMMARY.md compares FlashTile with the fastest PyTorch SDPA backend per configuration.\n"
    )
    print(f"\nSaved to {run_dir}")
