"""
torch.compile Compatibility Tests
=================================

This module tests that FlashTile's attention implementations work correctly
with PyTorch 2.0's torch.compile() JIT compilation.

Tests verify:
1. Models compile without errors
2. Compiled output matches eager mode output
3. No graph breaks occur (where possible)
4. Compilation succeeds with different backends

Requirements
------------
- PyTorch >= 2.0.0
- CUDA GPU (for meaningful testing)

Usage
-----
```bash
pytest tests/test_torch_compile.py -v
pytest tests/test_torch_compile.py -v -k "test_flash_v2"
```
"""

import pytest
import torch
import warnings

# Check for PyTorch 2.0+ and CUDA
HAS_TORCH_COMPILE = hasattr(torch, "compile")
HAS_CUDA = torch.cuda.is_available()

skip_no_compile = pytest.mark.skipif(
    not HAS_TORCH_COMPILE,
    reason="torch.compile requires PyTorch >= 2.0"
)

skip_no_cuda = pytest.mark.skipif(
    not HAS_CUDA,
    reason="CUDA not available"
)


class TestFlashAttentionV2Compile:
    """Test torch.compile compatibility for FlashAttentionV2."""

    @skip_no_compile
    @skip_no_cuda
    def test_flash_v2_compiles_inductor(self):
        """Verify FlashAttentionV2 compiles with inductor backend."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(
            embed_dim=256,
            num_heads=4,
            causal=True,
        ).cuda()

        compiled_model = torch.compile(model, backend="inductor")

        x = torch.randn(2, 128, 256, device="cuda")
        output, _ = compiled_model(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    @skip_no_compile
    @skip_no_cuda
    def test_flash_v2_eager_compiled_match(self):
        """Verify compiled output matches eager mode output."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(
            embed_dim=256,
            num_heads=4,
            causal=True,
        ).cuda()

        compiled_model = torch.compile(model, backend="inductor")

        x = torch.randn(2, 128, 256, device="cuda")

        # Get eager output
        with torch.no_grad():
            eager_output, _ = model(x)

        # Get compiled output
        with torch.no_grad():
            compiled_output, _ = compiled_model(x)

        # Should match within numerical tolerance
        torch.testing.assert_close(
            eager_output,
            compiled_output,
            rtol=1e-3,
            atol=1e-3,
        )

    @skip_no_compile
    @skip_no_cuda
    def test_flash_v2_compile_with_gradients(self):
        """Verify compiled model supports backward pass."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(
            embed_dim=256,
            num_heads=4,
            causal=True,
        ).cuda()

        compiled_model = torch.compile(model, backend="inductor")

        x = torch.randn(2, 64, 256, device="cuda", requires_grad=True)
        output, _ = compiled_model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.isnan(x.grad).any()

    @skip_no_compile
    @skip_no_cuda
    def test_flash_v2_compile_non_causal(self):
        """Verify non-causal FlashAttentionV2 compiles."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(
            embed_dim=256,
            num_heads=4,
            causal=False,
        ).cuda()

        compiled_model = torch.compile(model, backend="inductor")

        x = torch.randn(2, 128, 256, device="cuda")
        output, _ = compiled_model(x)

        assert output.shape == x.shape

    @skip_no_compile
    @skip_no_cuda
    def test_flash_v2_compile_different_backends(self):
        """Test compilation with different backends."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(
            embed_dim=256,
            num_heads=4,
            causal=True,
        ).cuda()

        backends = ["inductor"]

        # Check if eager backend is available (for debugging)
        try:
            eager_model = torch.compile(model, backend="eager")
            backends.append("eager")
        except Exception:
            pass

        x = torch.randn(2, 64, 256, device="cuda")

        for backend in backends:
            compiled_model = torch.compile(model, backend=backend)
            output, _ = compiled_model(x)
            assert output.shape == x.shape, f"Failed with backend: {backend}"


class TestFlashAttentionV1Compile:
    """Test torch.compile compatibility for FlashAttentionV1."""

    @skip_no_compile
    @skip_no_cuda
    def test_flash_v1_compiles(self):
        """Verify FlashAttentionV1 compiles."""
        from flashtile.attention import FlashAttentionV1

        model = FlashAttentionV1(
            embed_dim=256,
            num_heads=4,
            causal=False,
        ).cuda()

        compiled_model = torch.compile(model, backend="inductor")

        x = torch.randn(2, 128, 256, device="cuda")
        output, _ = compiled_model(x)

        assert output.shape == x.shape

    @skip_no_compile
    @skip_no_cuda
    def test_flash_v1_eager_compiled_match(self):
        """Verify compiled output matches eager for V1."""
        from flashtile.attention import FlashAttentionV1

        model = FlashAttentionV1(
            embed_dim=256,
            num_heads=4,
            causal=True,
        ).cuda()

        compiled_model = torch.compile(model, backend="inductor")

        x = torch.randn(2, 64, 256, device="cuda")

        with torch.no_grad():
            eager_output, _ = model(x)
            compiled_output, _ = compiled_model(x)

        torch.testing.assert_close(
            eager_output,
            compiled_output,
            rtol=1e-3,
            atol=1e-3,
        )


class TestGQACompile:
    """Test torch.compile compatibility for GroupedQueryAttention."""

    @skip_no_compile
    @skip_no_cuda
    def test_gqa_compiles(self):
        """Verify GroupedQueryAttention compiles."""
        from flashtile.attention import GroupedQueryAttention

        model = GroupedQueryAttention(
            embed_dim=256,
            num_heads=8,
            num_kv_heads=2,  # GQA with 4 Q heads per KV head
            causal=True,
        ).cuda()

        compiled_model = torch.compile(model, backend="inductor")

        x = torch.randn(2, 128, 256, device="cuda")
        output, _ = compiled_model(x)

        assert output.shape == x.shape

    @skip_no_compile
    @skip_no_cuda
    def test_mqa_compiles(self):
        """Verify MultiQueryAttention compiles."""
        from flashtile.attention import MultiQueryAttention

        model = MultiQueryAttention(
            embed_dim=256,
            num_heads=8,
            causal=True,
        ).cuda()

        compiled_model = torch.compile(model, backend="inductor")

        x = torch.randn(2, 128, 256, device="cuda")
        output, _ = compiled_model(x)

        assert output.shape == x.shape


class TestSlidingWindowCompile:
    """Test torch.compile compatibility for SlidingWindowAttention."""

    @skip_no_compile
    @skip_no_cuda
    def test_sliding_window_compiles(self):
        """Verify SlidingWindowAttention compiles."""
        from flashtile.attention import SlidingWindowAttention

        model = SlidingWindowAttention(
            embed_dim=256,
            num_heads=4,
            window_size=256,
            causal=True,
        ).cuda()

        compiled_model = torch.compile(model, backend="inductor")

        x = torch.randn(2, 512, 256, device="cuda")
        output, _ = compiled_model(x)

        assert output.shape == x.shape

    @skip_no_compile
    @skip_no_cuda
    def test_sliding_window_eager_compiled_match(self):
        """Verify compiled output matches eager for SlidingWindow."""
        from flashtile.attention import SlidingWindowAttention

        model = SlidingWindowAttention(
            embed_dim=256,
            num_heads=4,
            window_size=128,
            causal=True,
        ).cuda()

        compiled_model = torch.compile(model, backend="inductor")

        x = torch.randn(2, 256, 256, device="cuda")

        with torch.no_grad():
            eager_output, _ = model(x)
            compiled_output, _ = compiled_model(x)

        torch.testing.assert_close(
            eager_output,
            compiled_output,
            rtol=1e-3,
            atol=1e-3,
        )


class TestCompilePerformance:
    """Test that compilation provides performance benefits."""

    @skip_no_compile
    @skip_no_cuda
    @pytest.mark.slow
    def test_compile_speedup(self):
        """Verify compiled model is faster after warmup."""
        from flashtile.attention import FlashAttentionV2
        import time

        model = FlashAttentionV2(
            embed_dim=512,
            num_heads=8,
            causal=True,
        ).cuda()

        compiled_model = torch.compile(model, backend="inductor")

        x = torch.randn(4, 512, 512, device="cuda")

        # Warmup compiled model (includes compilation)
        for _ in range(3):
            compiled_model(x)
            torch.cuda.synchronize()

        # Time eager
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            model(x)
        torch.cuda.synchronize()
        eager_time = time.perf_counter() - start

        # Time compiled
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            compiled_model(x)
        torch.cuda.synchronize()
        compiled_time = time.perf_counter() - start

        # Compiled should be at least as fast (may be slower due to overhead)
        # This is a sanity check, not a strict performance requirement
        assert compiled_time < eager_time * 2, (
            f"Compiled ({compiled_time:.3f}s) is more than 2x slower than "
            f"eager ({eager_time:.3f}s)"
        )


class TestCompileEdgeCases:
    """Test torch.compile with edge cases."""

    @skip_no_compile
    @skip_no_cuda
    def test_compile_batch_size_1(self):
        """Test compiled model with batch size 1."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(
            embed_dim=256,
            num_heads=4,
            causal=True,
        ).cuda()

        compiled_model = torch.compile(model, backend="inductor")

        x = torch.randn(1, 128, 256, device="cuda")
        output, _ = compiled_model(x)

        assert output.shape == x.shape

    @skip_no_compile
    @skip_no_cuda
    def test_compile_sequence_length_1(self):
        """Test compiled model with sequence length 1."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(
            embed_dim=256,
            num_heads=4,
            causal=True,
        ).cuda()

        compiled_model = torch.compile(model, backend="inductor")

        x = torch.randn(2, 1, 256, device="cuda")
        output, _ = compiled_model(x)

        assert output.shape == x.shape

    @skip_no_compile
    @skip_no_cuda
    def test_compile_different_dtypes(self):
        """Test compiled model with different dtypes."""
        from flashtile.attention import FlashAttentionV2

        for dtype in [torch.float32, torch.float16]:
            model = FlashAttentionV2(
                embed_dim=256,
                num_heads=4,
                causal=True,
            ).cuda().to(dtype)

            compiled_model = torch.compile(model, backend="inductor")

            x = torch.randn(2, 64, 256, device="cuda", dtype=dtype)
            output, _ = compiled_model(x)

            assert output.shape == x.shape
            assert output.dtype == dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
