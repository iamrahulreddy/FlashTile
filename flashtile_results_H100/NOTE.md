## Note on H100 Results

These artifacts are the initial H100 validation run from Feb 27, 2026. They record
145 out of 146 tests passing.

The single failure, `test_memory_cleanup_after_oom_recovery`, turned out to be a
test-threshold issue rather than an attention bug. The original test required
post-cleanup memory to stay below `model_params * 50` (~26 MB), but the H100 CUDA
runtime keeps roughly ~68 MB of workspace memory (cuBLAS pools and allocator state)
even after `empty_cache()`. A100 retained much less memory, so the same threshold
happened to pass there.

The source test has since been updated to use a flat `< 150 MB` threshold instead
of tying the limit to parameter size. The attention implementations themselves were
otherwise healthy on H100: the correctness, precision, and kernel-focused suites
matched the A100 run.

These H100 files are kept as historical artifacts from that pre-fix run. They have
not been regenerated after the threshold fix, so treat them as archived evidence
rather than current post-fix status.
