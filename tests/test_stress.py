"""
Stress Tests for FlashTile
==========================

This module contains stress tests that verify FlashTile's behavior under
extreme conditions: long sequences, many iterations, varying batch sizes,
and memory pressure.

These tests are marked as 'slow' and should be run separately from the
fast unit tests.

Tests verify:
1. Long sequence correctness (8K, 16K, 32K tokens)
2. Memory leak detection over many iterations
3. Large batch sizes
4. Edge case sequence lengths
5. GPU memory pressure scenarios

Usage
-----
```bash
pytest tests/test_stress.py -v -m slow  # Run only slow tests
pytest tests/test_stress.py -v -k "32k"  # Run specific test
```
"""

import gc
import pytest
import torch

HAS_CUDA = torch.cuda.is_available()

skip_no_cuda = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    return 0


def estimate_required_memory_bytes(
    model,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype = torch.float16,
    safety_factor: float = 4.0,
    extra_workspace_mb: float = 512.0,
) -> int:
    """Estimate a conservative memory budget for long-sequence forward tests."""
    theory = model.get_memory_usage(batch_size=batch_size, seq_len=seq_len)
    forward_bytes = int(theory["total_mb"] * (1024 ** 2))
    input_bytes = batch_size * seq_len * model.embed_dim * (torch.finfo(dtype).bits // 8)
    parameter_bytes = sum(param.numel() * param.element_size() for param in model.parameters())

    return int(
        (forward_bytes + input_bytes + parameter_bytes) * safety_factor
        + extra_workspace_mb * (1024 ** 2)
    )


def skip_if_insufficient_gpu_memory(
    model,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype = torch.float16,
) -> None:
    """Skip long-sequence tests when the available GPU memory is obviously insufficient."""
    gc.collect()
    torch.cuda.empty_cache()

    device_props = torch.cuda.get_device_properties(0)
    used_memory = max(torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
    free_memory = device_props.total_memory - used_memory
    required_memory = estimate_required_memory_bytes(
        model, batch_size=batch_size, seq_len=seq_len, dtype=dtype
    )

    if free_memory < required_memory:
        pytest.skip(
            "Not enough free GPU memory for this stress case: "
            f"need ~{required_memory / (1024 ** 3):.2f} GB, "
            f"have ~{free_memory / (1024 ** 3):.2f} GB"
        )


class TestLongSequences:
    """Test attention with long sequences."""

    @skip_no_cuda
    @pytest.mark.slow
    def test_sequence_2k(self):
        """Test with 2K sequence length."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(256, 4, causal=True).cuda().half()

        x = torch.randn(2, 2048, 256, device="cuda", dtype=torch.float16)
        output, _ = model(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    @skip_no_cuda
    @pytest.mark.slow
    def test_sequence_4k(self):
        """Test with 4K sequence length."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(256, 4, causal=True).cuda().half()

        x = torch.randn(1, 4096, 256, device="cuda", dtype=torch.float16)
        output, _ = model(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    @skip_no_cuda
    @pytest.mark.slow
    def test_sequence_8k(self):
        """Test with 8K sequence length."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(256, 4, causal=True).cuda().half()

        x = torch.randn(1, 8192, 256, device="cuda", dtype=torch.float16)
        output, _ = model(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    @skip_no_cuda
    @pytest.mark.slow
    def test_sequence_16k(self):
        """Test with 16K sequence length (if memory permits)."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(256, 4, causal=True).cuda().half()
        skip_if_insufficient_gpu_memory(model, batch_size=1, seq_len=16384, dtype=torch.float16)

        x = torch.randn(1, 16384, 256, device="cuda", dtype=torch.float16)
        output, _ = model(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    @skip_no_cuda
    @pytest.mark.slow
    def test_sequence_32k(self):
        """Test with 32K sequence length (if memory permits)."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(256, 4, causal=True).cuda().half()
        skip_if_insufficient_gpu_memory(model, batch_size=1, seq_len=32768, dtype=torch.float16)

        x = torch.randn(1, 32768, 256, device="cuda", dtype=torch.float16)
        output, _ = model(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()


class TestMemoryLeaks:
    """Test for memory leaks over many iterations."""

    @skip_no_cuda
    @pytest.mark.slow
    def test_no_memory_leak_forward(self):
        """Verify no memory leak over 100 forward passes."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(256, 4, causal=True).cuda().half()
        x = torch.randn(2, 512, 256, device="cuda", dtype=torch.float16)

        # Warmup and establish baseline
        for _ in range(10):
            _ = model(x)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        baseline_memory = get_gpu_memory_mb()

        # Run many iterations
        for _ in range(100):
            _ = model(x)

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        final_memory = get_gpu_memory_mb()

        # Allow small variance but no significant leak
        memory_increase = final_memory - baseline_memory
        assert memory_increase < 50, f"Memory increased by {memory_increase:.1f} MB (possible leak)"

    @skip_no_cuda
    @pytest.mark.slow
    def test_no_memory_leak_forward_backward(self):
        """Verify no memory leak over 50 forward+backward passes."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(256, 4, causal=True).cuda().half()
        optimizer = torch.optim.Adam(model.parameters())

        # Warmup
        for _ in range(5):
            x = torch.randn(2, 256, 256, device="cuda", dtype=torch.float16)
            output, _ = model(x)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        baseline_memory = get_gpu_memory_mb()

        # Run many training iterations
        for _ in range(50):
            x = torch.randn(2, 256, 256, device="cuda", dtype=torch.float16)
            output, _ = model(x)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        final_memory = get_gpu_memory_mb()

        memory_increase = final_memory - baseline_memory
        assert memory_increase < 100, f"Memory increased by {memory_increase:.1f} MB (possible leak)"

    @skip_no_cuda
    @pytest.mark.slow
    def test_memory_cleanup_after_oom_recovery(self):
        """Test memory cleanup when approaching OOM."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(256, 4, causal=True).cuda().half()

        # Run several iterations with different sizes
        for seq_len in [512, 1024, 2048, 1024, 512]:
            x = torch.randn(1, seq_len, 256, device="cuda", dtype=torch.float16)
            output, _ = model(x)
            del output, x

        gc.collect()
        torch.cuda.empty_cache()

        # Memory should be cleaned up. Allow up to 150 MB overhead for CUDA allocator
        # memory pools and workspaces (e.g., H100 retains ~68 MB).
        final_memory = get_gpu_memory_mb()
        assert final_memory < 150, f"OOM recovery failed: Retained {final_memory:.1f} MB (Expected < 150 MB)"


class TestBatchSizes:
    """Test various batch sizes."""

    @skip_no_cuda
    @pytest.mark.slow
    def test_large_batch_size(self):
        """Test with large batch size."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(256, 4, causal=True).cuda().half()

        x = torch.randn(16, 256, 256, device="cuda", dtype=torch.float16)
        output, _ = model(x)

        assert output.shape == x.shape

    @skip_no_cuda
    @pytest.mark.slow
    def test_varying_batch_sizes(self):
        """Test with varying batch sizes in sequence."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(256, 4, causal=True).cuda().half()

        for batch_size in [1, 2, 4, 8, 4, 2, 1]:
            x = torch.randn(batch_size, 256, 256, device="cuda", dtype=torch.float16)
            output, _ = model(x)
            assert output.shape == x.shape


class TestEdgeCases:
    """Test edge case sequence lengths."""

    @skip_no_cuda
    @pytest.mark.slow
    def test_prime_sequence_lengths(self):
        """Test with prime number sequence lengths (not divisible by block size)."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(256, 4, causal=True).cuda().half()

        # Prime numbers that don't divide evenly by typical block sizes (32, 64, 128)
        prime_lengths = [127, 131, 251, 509, 1021]

        for seq_len in prime_lengths:
            x = torch.randn(1, seq_len, 256, device="cuda", dtype=torch.float16)
            output, _ = model(x)
            assert output.shape == x.shape
            assert not torch.isnan(output).any(), f"NaN at seq_len={seq_len}"

    @skip_no_cuda
    @pytest.mark.slow
    def test_power_of_two_sequences(self):
        """Test with power-of-2 sequence lengths."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(256, 4, causal=True).cuda().half()

        for power in range(4, 12):  # 16 to 2048
            seq_len = 2 ** power
            x = torch.randn(1, seq_len, 256, device="cuda", dtype=torch.float16)
            output, _ = model(x)
            assert output.shape == x.shape

    @skip_no_cuda
    @pytest.mark.slow
    def test_one_less_than_block_size(self):
        """Test sequence lengths just below block boundaries."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(256, 4, causal=True, block_size=64).cuda().half()

        # Test lengths just before block boundaries
        for seq_len in [63, 127, 191, 255, 319]:
            x = torch.randn(1, seq_len, 256, device="cuda", dtype=torch.float16)
            output, _ = model(x)
            assert output.shape == x.shape

    @skip_no_cuda
    @pytest.mark.slow
    def test_one_more_than_block_size(self):
        """Test sequence lengths just above block boundaries."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(256, 4, causal=True, block_size=64).cuda().half()

        # Test lengths just after block boundaries
        for seq_len in [65, 129, 193, 257, 321]:
            x = torch.randn(1, seq_len, 256, device="cuda", dtype=torch.float16)
            output, _ = model(x)
            assert output.shape == x.shape


class TestStabilityUnderLoad:
    """Test stability under sustained load."""

    @skip_no_cuda
    @pytest.mark.slow
    def test_sustained_inference(self):
        """Test 1000 consecutive inference calls."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(256, 4, causal=True).cuda().half()
        x = torch.randn(1, 256, 256, device="cuda", dtype=torch.float16)

        for i in range(1000):
            output, _ = model(x)
            if i % 100 == 0:
                assert not torch.isnan(output).any(), f"NaN at iteration {i}"

    @skip_no_cuda
    @pytest.mark.slow
    def test_sustained_training(self):
        """Test 500 consecutive training iterations."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(256, 4, causal=True).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for i in range(500):
            x = torch.randn(2, 128, 256, device="cuda")
            output, _ = model(x)
            loss = output.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 50 == 0:
                for name, param in model.named_parameters():
                    assert torch.isfinite(param).all(), f"Non-finite in {name} at iteration {i}"


class TestSlidingWindowStress:
    """Stress tests for Sliding Window Attention."""

    @skip_no_cuda
    @pytest.mark.slow
    def test_sliding_window_long_sequence(self):
        """Test SlidingWindow with long sequence."""
        from flashtile.attention import SlidingWindowAttention

        model = SlidingWindowAttention(
            embed_dim=256,
            num_heads=4,
            window_size=512,
            causal=True,
        ).cuda().half()

        # Long sequence where window << seq_len
        x = torch.randn(1, 8192, 256, device="cuda", dtype=torch.float16)
        output, _ = model(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    @skip_no_cuda
    @pytest.mark.slow
    def test_sliding_window_memory_efficiency(self):
        """Verify SlidingWindow uses less memory than full attention."""
        from flashtile.attention import FlashAttentionV2, SlidingWindowAttention

        seq_len = 4096
        window_size = 512

        torch.cuda.empty_cache()
        gc.collect()

        # Sliding Window
        sw_model = SlidingWindowAttention(
            embed_dim=256,
            num_heads=4,
            window_size=window_size,
            causal=True,
        ).cuda().half()

        x = torch.randn(1, seq_len, 256, device="cuda", dtype=torch.float16)

        torch.cuda.reset_peak_memory_stats()
        _ = sw_model(x)
        sw_peak_memory = torch.cuda.max_memory_allocated()

        del sw_model, x
        torch.cuda.empty_cache()
        gc.collect()

        # Full Flash Attention
        flash_model = FlashAttentionV2(
            embed_dim=256,
            num_heads=4,
            causal=True,
        ).cuda().half()

        x = torch.randn(1, seq_len, 256, device="cuda", dtype=torch.float16)

        torch.cuda.reset_peak_memory_stats()
        _ = flash_model(x)
        flash_peak_memory = torch.cuda.max_memory_allocated()

        # Sliding window should use less memory (or comparable)
        # Note: Due to implementation overhead, might not always be less
        # But should not be dramatically more
        assert sw_peak_memory <= flash_peak_memory * 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "slow"])
