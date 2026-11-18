# FlashTile Usage Guide

How to call FlashTile directly, and how to use it as the attention of a Hugging Face model.

## Installation

```bash
git clone https://github.com/iamrahulreddy/FlashTile.git
cd FlashTile
pip install -e .            # core, needs torch >= 2.4
pip install -e ".[hf]"      # + Hugging Face transformers integration

python -c "import flashtile; print(flashtile.__version__, flashtile.HAS_TRITON)"
```

The fused Triton path needs an NVIDIA GPU (Linux, where torch ships Triton). Without one,
everything still runs on the tiled PyTorch path, just slower. Tested with torch 2.15, Triton 3.9
and transformers 5.23.

## Quick Start

```python
import torch
from flashtile import attention

q = torch.randn(2, 32, 4096, 128, device="cuda", dtype=torch.bfloat16)   # [B, Hq, L, d]
k = torch.randn(2, 8, 4096, 128, device="cuda", dtype=torch.bfloat16)    # [B, Hkv, L, d]
v = torch.randn(2, 8, 4096, 128, device="cuda", dtype=torch.bfloat16)    # [B, Hkv, L, dv]

out = attention(q, k, v, causal=True)    # [B, Hq, L, dv]
```

Note the layout: heads come before the sequence (`[B, H, L, d]`, same as
`F.scaled_dot_product_attention`). v1's modules took `(batch, seq, embed)` and did the projections
themselves; v2 is just the attention op.

## API Reference

### attention

```python
def attention(
    q, k, v,                # [B, Hq, Lq, d], [B, Hkv, Lk, d], [B, Hkv, Lk, dv]
    *,
    scale=None,             # default 1/sqrt(d)
    causal=False,           # key j visible to query i when k_pos[j] <= q_pos[i]
    allow_mask=None,        # bool, True = attend, broadcastable to [B, Hq, Lq, Lk]
    q_pos=None, k_pos=None, # integer positions, default 0..L-1; required for causal with Lq != Lk
    backend="auto",         # "auto", "triton" or "tiled"
    return_lse=False,       # also return the log-sum-exp, [B, Hq, Lq]
    block_q=64, block_k=64, # tile sizes for the tiled backend only (triton autotunes)
): ...                      # returns out, or (out, lse) with return_lse=True
```

- Hq must be a multiple of Hkv. Query head `h` reads KV head `h // (Hq // Hkv)`, so MHA, GQA
  and MQA all go through the same call, and K/V are never copied.
- q, k and v must share one dtype and device. fp16, bf16, fp32 and fp64 are accepted.
  Accumulation is fp32 (fp64 stays fp64), and the output comes back in the input dtype.
- A query row with nothing to attend to (fully masked) returns 0, with lse = -inf and zero
  gradient. No NaNs.
- With `return_lse=True` it returns `(out, lse)`, and the lse is differentiable.

`backend="auto"` uses Triton when it can and the tiled path otherwise. `backend="triton"` raises
instead of quietly falling back, which is handy when benchmarking.

### select_backend

```python
flashtile.select_backend(q, k, v, causal=False, allow_mask=None, q_pos=None, k_pos=None)
# -> ("triton", "supported")  or  ("tiled", "<reason>")
```

Shows what `"auto"` would pick and why, without running anything. Triton needs all of these:
- CUDA tensors in fp16/bf16/fp32
- no `allow_mask`
- head dims <= 256
- for causal calls, positions that are contiguous ranges

### attention_reference

```python
out, lse = flashtile.attention_reference(q, k, v, scale=None, causal=False, allow_mask=None,
                                         q_pos=None, k_pos=None)
```

Dense implementation that builds the full score table. It's what every test compares against
(run it in fp64 for that). Don't use it for long sequences: memory grows as L².

### tiled_attention

```python
out = flashtile.tiled_attention(q, k, v, causal=False, allow_mask=None, block_q=64, block_k=64)
```

Plain PyTorch version of the same algorithm, forward and backward. Runs on any device and dtype
and supports masks. It's slow (Python loops over tiles) but easy to read and step through.

### flashtile.hf

```python
import flashtile.hf

flashtile.hf.register()       # once per process, before loading or switching a model
flashtile.hf.NAME             # "ftile", the name to pass as attn_implementation
flashtile.hf.stats            # Counter of calls per path: causal, decode, full, masked
```

## Using FlashTile with Hugging Face models

FlashTile plugs into transformers' attention registry, so a model can use it without any code
changes. It should work with any model whose attention goes through that registry (Llama- and
Qwen-style models). The ones actually tested are Qwen3.5-4B (bf16 and FP8), Qwen3.8-27B (bf16)
and a small random Qwen2 in `tests/test_hf.py`.

### 1. Load a model with FlashTile

```python
import torch
import flashtile.hf
from transformers import AutoModelForCausalLM, AutoTokenizer

flashtile.hf.register()

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation=flashtile.hf.NAME,
)
```

Use `flashtile.hf.NAME` (`"ftile"`), not `"flashtile"`. transformers treats any name containing
"flash" as a FlashAttention kernel and tries to download one.

### 2. Generate

```python
messages = [{"role": "user", "content": "Explain online softmax in two sentences."}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to("cuda")

out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Prefill runs on the Triton causal path. Every decode step runs on the Triton decode path, which
packs a GQA group's query heads into one tile so each K/V tile is read once.

### 3. Switch an already-loaded model, or compare with SDPA

```python
model.set_attn_implementation(flashtile.hf.NAME)   # FlashTile
model.set_attn_implementation("sdpa")              # back to PyTorch SDPA
```

Handy for checking outputs side by side. Expect tiny differences (bf16 rounding). Greedy
generations can drift apart after a few dozen tokens, especially with FP8 checkpoints, and HF's
own eager attention drifts from SDPA just as much.

`"sdpa"` lets PyTorch pick a kernel (cuDNN on H200). To compare with one specific kernel, such
as PyTorch's built-in FlashAttention-2, `benchmark/llm_compare.py` registers SDPA wrappers that
force a backend.

### 4. Check that FlashTile is actually running

```python
flashtile.hf.stats.clear()
model.generate(**inputs, max_new_tokens=20)
print(flashtile.hf.stats)    # e.g. Counter({'decode': 532, 'causal': 28})
```

- `causal`, `decode`, `full`: the fast Triton path.
- `masked`: the call had a padding mask and went to the tiled PyTorch path (correct, but slow).

### 5. Fine-tune with it

```python
model.set_attn_implementation(flashtile.hf.NAME)
model.train()

batch = tok(["some training text ..."], return_tensors="pt").to("cuda")
loss = model(**batch, labels=batch["input_ids"]).loss
loss.backward()
```

Gradients go through FlashTile's backward. Attention dropout has to be 0, which is the default for
Llama/Qwen-style configs (`attention_dropout=0.0`).

### 6. Qwen3.5 and Qwen3.8 (the models used in the results)

Qwen3.5 checkpoints are multimodal, so they load through `AutoModelForImageTextToText`. FlashTile
replaces the 8 softmax-attention layers. The other 24 (Gated DeltaNet) aren't attention in this
sense and stay as they are. Qwen3.8-27B uses the same architecture and loads the same way. There
FlashTile replaces 16 of 64 layers, and the bf16 weights take about 51 GB of GPU memory.

```python
from transformers import AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3.5-4B",                # or "RedHatAI/Qwen3.5-4B-FP8-dynamic" (needs compressed-tensors)
    dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation=flashtile.hf.NAME,
)
```

- `pip install flash-linear-attention` gives the fast DeltaNet kernels. Without it,
  transformers falls back to slower PyTorch code for those layers. `pip install causal-conv1d`
  does the same for their short convolution (the published runs didn't have it).
- Only text inputs were tested.
- The FP8 checkpoint saves memory (6.1 vs 9.0 GB), but transformers runs it as
  quantize-dequantize with bf16 matmuls, so it isn't faster.

`benchmark/llm_eval.py` is the full example: precision, prefill, generation and fine-tuning.
`benchmark/llm_compare.py` runs Qwen3.8-27B with SDPA, FA2, cuDNN and FlashTile on the same prompts.

### What the integration does with each call

| Situation | What happens |
|---|---|
| No padding, prefill (Lq = Lk) | Triton, causal |
| No padding, decode (Lq = 1) | Triton, GQA heads packed into one tile |
| Padding mask in the batch | tiled PyTorch path, exact but slow |
| Sliding-window layer, sequence longer than the window | `NotImplementedError` |
| Attention dropout > 0 in training mode | `NotImplementedError` |
| `output_attentions=True` | no attention weights are returned (`None`) |

## Common Use Cases

### 1. Causal self-attention (training or prefill)

```python
out = attention(q, k, v, causal=True)
out.float().sum().backward()        # dq, dk, dv through the Triton backward
```

### 2. GQA / MQA

```python
q = torch.randn(1, 32, 2048, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 8, 2048, 128, device="cuda", dtype=torch.bfloat16)   # 4 query heads per KV head
v = torch.randn(1, 8, 2048, 128, device="cuda", dtype=torch.bfloat16)
out = attention(q, k, v, causal=True)                                     # K/V not repeated
```

For MQA use one KV head (`k.shape[1] == 1`).

### 3. Decode with a KV cache

The new token sits at position `L` and attends to `L + 1` cached keys:

```python
L = 4096
q1 = torch.randn(1, 32, 1, 128, device="cuda", dtype=torch.bfloat16)
k_cache = torch.randn(1, 8, L + 1, 128, device="cuda", dtype=torch.bfloat16)
v_cache = torch.randn(1, 8, L + 1, 128, device="cuda", dtype=torch.bfloat16)

o = attention(q1, k_cache, v_cache, causal=True,
              q_pos=torch.arange(L, L + 1), k_pos=torch.arange(L + 1))
```

Pass positions as CPU tensors. FlashTile reads the first value to work out the causal offset,
and on a GPU tensor that read waits for the GPU.

### 4. Chunked prefill

Processing a long prompt in pieces: queries are the current chunk, keys are everything so far.

```python
chunk, done = 512, 3072
qc = torch.randn(1, 32, chunk, 128, device="cuda", dtype=torch.bfloat16)
kc = torch.randn(1, 8, done + chunk, 128, device="cuda", dtype=torch.bfloat16)
vc = torch.randn(1, 8, done + chunk, 128, device="cuda", dtype=torch.bfloat16)
o = attention(qc, kc, vc, causal=True,
              q_pos=torch.arange(done, done + chunk), k_pos=torch.arange(done + chunk))
```

### 5. Padding or custom masks

```python
B, L = 2, 1024
allow = torch.ones(B, 1, L, L, dtype=torch.bool, device="cuda")   # True = attend
allow[1, :, :, :100] = False                                      # sequence 1: first 100 keys are padding
out = attention(q[:B, :, :L], k[:B, :, :L], v[:B, :, :L], causal=True, allow_mask=allow)
```

Masks run on the tiled path (`select_backend` will say so). Mask polarity changed from v1:
**True now means "may attend"** (v1 used True for masked out).

### 6. Using the log-sum-exp

```python
out, lse = attention(q, k, v, causal=True, return_lse=True)
```

- `lse` is in natural log, fp32, shape `[B, Hq, Lq]`, and differentiable.
- Useful for merging attention computed over separate key blocks (ring/context parallel style),
  or anything that needs the softmax normaliser.
- Fully masked rows have `lse = -inf`.

## Configuration Guide

### Backward variant

The default backward uses two kernels (dK/dV, then dQ) with no atomics, so a given config gives
the same result every run. A single-pass variant accumulates dQ with fp32 atomics:

```bash
FLASHTILE_BACKWARD=atomic python train.py
```

It was slower on most shapes in the benchmarks. It only won on RTX PRO 6000 at head dim 256 with
long sequences, and on d=64 on B200. It also makes the last bits of dQ vary between runs. Set it
before importing flashtile.

### Tile sizes

The Triton kernels autotune their tiles per GPU and per length bucket (powers of two), so there's
nothing to set. `block_q` / `block_k` only matter for `backend="tiled"`.

## Troubleshooting

### "Could not find the currently requested flash attention implementation at `flashtile`"

The name passed to `attn_implementation` contains "flash". Use `flashtile.hf.NAME` (`"ftile"`)
and call `flashtile.hf.register()` first.

### `ValueError: triton backend cannot run this call: ...`

`backend="triton"` was forced on a call Triton can't handle; the message says why (mask, dtype,
CPU tensors, head dim, gapped positions). Use `backend="auto"` to fall back to the tiled path.

### `ValueError: causal attention with q_len != k_len needs explicit q_pos and k_pos`

With different query and key lengths, FlashTile won't guess how they line up. Pass positions,
see Common Use Cases 3 and 4.

### The first call is slow

Each new shape (length bucket, head dim, dtype, causal or not) is compiled and autotuned on first
use. That takes seconds to tens of seconds. Do a warm-up call before timing anything.

### Batched generation is slow

Padding masks send every call to the tiled PyTorch path (`flashtile.hf.stats` shows `masked`).
Generate one sequence at a time, or batch prompts of equal length so there's no padding.

### Triton not available

`flashtile.HAS_TRITON` is False on CPU-only installs, on Windows, and with a broken Triton.
Everything still works on the tiled path.

## Performance Tips

- Warm up once per shape before measuring (see above).
- Keep q/k/v in bf16 or fp16. fp32 runs with full IEEE precision (no TF32), which is slower.
- Pass positions as CPU tensors to avoid a GPU sync per call.
- Avoid padding masks in hot paths: they leave the Triton path.
- Head dims 64 and 128 do best. d=256 works, but it's the slowest case relative to PyTorch's
  kernels (see the README).
- On datacenter GPUs, PyTorch's cuDNN backend is still faster (README, "Where it falls short").
  FlashTile is the readable, checked alternative, not the fastest option there.
- Long-context decode at batch 1 is the slowest part. With a 22-34K cache, Qwen3.8-27B spent
  9-13 ms per token in FlashTile's attention vs 2.2-2.4 ms with FA2 or cuDNN. HF generate's
  own overhead hid it there, but a faster serving loop wouldn't.

## Important Notes

### Mask polarity

`allow_mask=True` means "may attend". v1's masks used True for "masked out", so flip old masks
with `~mask`.

### Fully masked rows

They return zeros, lse = -inf and zero gradient, not NaN. This matches the reference
implementation, not every other library.

### What isn't there

Dropout, sliding window, ALiBi, softcapping, variable-length packing (cu_seqlens), paged KV cache
and FP8 attention aren't implemented. Arbitrary masks work, but only on the slow tiled path.
