# Changelog

This document summarizes the notable public milestones for FlashTile. It is
intended as a high-level release log for major features, fixes, and project
updates.

## [2.0.0]

Rewrite. One set of rules, three implementations that have to agree: a dense fp64 reference, a
tiled PyTorch version and fused Triton kernels. The v1 module classes (`FlashAttentionV1`,
`FlashAttentionV2`, `NaiveAttention`, ...) are gone; the API is now `flashtile.attention(q, k, v)`.

### Added

- Triton backward pass (v1's kernel was forward only).
- GQA/MQA inside the kernels, no K/V copies, O(N) backward.
- Causal attention with position offsets, so decode and chunked prefill work.
- Any head dim up to 256, not just powers of two.
- Gradients through the returned LSE, and fully masked rows that return 0 with zero gradient.
- Faster kernels: masks only on diagonal tiles, exp2, a small delta kernel, no q/k/v copies in
  the backward, heaviest tiles scheduled first, GQA packing for decode.
- An optional single-pass backward with atomic dQ (`FLASHTILE_BACKWARD=atomic`). Off by default,
  it's slower on most shapes.
- `flashtile.hf` to use FlashTile inside Hugging Face transformers (`attn_implementation="ftile"`).
- `docs/USAGE.md`: API reference, using FlashTile with Hugging Face models, troubleshooting.
- Qwen3.5-4B (FP8) evaluation: precision, prefill to 65K tokens, generation, fine-tuning.
- Qwen3.8-27B (bf16) on H200: SDPA, PyTorch's FA2 kernel, cuDNN and FlashTile on the same five
  prompts (`benchmark/llm_compare.py`, `modal run modal_app.py --qwen27b`).
- `modal_app.py` to run tests, benchmarks and the LLM eval on Modal GPUs.
- Results for H200, B200 and RTX PRO 6000.

### Fixed

- The grouped-head backward no longer crashes Triton's compiler on B200.
- Calls that pass positions without `causal=True` no longer error out, and plain calls don't
  sync the GPU anymore.

### Removed

- Sliding-window and masked modules (masks still work, through the tiled path), the
  `torch.compile` test and the Colab/gradio demo.

### Project Notes

- Version `2.0.0`, Python `>=3.10`, tested on torch 2.15 + Triton 3.9.
- 158/158 tests on H200, B200 and RTX PRO 6000.
- Known gaps are listed in the README (cuDNN is faster on datacenter GPUs, head dim 256 and the
  backward are the weak spots, no split-KV decode yet). In Qwen3.8-27B, long-context prefill
  attention is 1.3x faster than FA2 and 1.7x slower than cuDNN.

## [0.2.2]

### Changed

- Raised the minimum supported Python version to `3.9`.
- Consolidated AMP compatibility helpers into `flashtile.attention.amp_compat`.
- Updated public documentation to align implementation notes, benchmark
  summaries, and archived validation references with the current repository
  state.

### Fixed

- Added a training-time warning for GQA to make its `O(N^2)` backward-memory
  behavior explicit.
- Corrected masking, backward-pass, and performance notes that previously
  overstated support or efficiency in some public-facing docs.
- Tightened test and benchmark descriptions to better reflect actual coverage
  and expected behavior.

### Project Notes

- Current package version: `0.2.2`
- Current Python requirement: `>=3.9`
- Main focus of this milestone: documentation accuracy, presentation cleanup,
  and consistency across the public repo

## [0.2.1]

### Added

- Colab notebook and demo assets for easier inspection of the project.
- Published benchmark artifacts covering memory and runtime behavior across
  multiple sequence lengths.
- Archived GPU validation outputs for the main A100 run, with an H100
  cross-check retained as a reference artifact.

### Fixed

- Dtype handling in forward and backward paths for mixed-precision inputs.
- Stability issues around fp16 training examples and loss scaling.
- Edge cases in sliding-window attention and allocator-related test thresholds.

### Changed

- Replaced older theoretical benchmark claims with measured benchmark data.
- Expanded the public test and validation footprint reflected in the repo.

## [0.2.0]

### Added

- Memory-efficient backward passes for Flash Attention V1 and V2 using custom
  autograd and recomputation.
- Grouped-query attention and multi-query attention implementations.
- Causal optimizations and a forward-only Triton path for performance
  comparison.
- Broader test coverage and supporting documentation for the attention modules.

## [0.1.0]

### Added

- Initial project baseline with naive attention, Flash Attention reference
  implementations, benchmark scripts, and core documentation.
