# Changelog

This document summarizes the notable public milestones for FlashTile. It is
intended as a high-level release log for major features, fixes, and project
updates.

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
