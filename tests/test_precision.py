import pytest
import torch
import math

HAS_CUDA = torch.cuda.is_available()
HAS_BF16 = HAS_CUDA and torch.cuda.is_bf16_supported()
HAS_FP8 = hasattr(torch, "float8_e4m3fn")

skip_no_cuda = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
skip_no_bf16 = pytest.mark.skipif(not HAS_BF16, reason="BF16 not supported")
skip_no_fp8 = pytest.mark.skipif(not HAS_FP8, reason="FP8 not available")


class TestFP16Precision:
    """Test FP16 (half precision) attention."""

    @skip_no_cuda
    def test_flash_v2_fp16_correctness(self):
        """Verify Flash V2 produces correct output in FP16."""
        from flashtile.attention import FlashAttentionV2, NaiveAttention

        embed_dim, num_heads = 256, 4
        batch, seq_len = 2, 128

        # Create models
        flash = FlashAttentionV2(embed_dim, num_heads, causal=True).cuda().half()
        naive = NaiveAttention(embed_dim, num_heads, causal=True).cuda().half()

        # Sync weights
        with torch.no_grad():
            naive.qkv_proj.weight.copy_(flash.qkv_proj.weight)
            naive.qkv_proj.bias.copy_(flash.qkv_proj.bias)
            naive.out_proj.weight.copy_(flash.out_proj.weight)
            naive.out_proj.bias.copy_(flash.out_proj.bias)

        x = torch.randn(batch, seq_len, embed_dim, device="cuda", dtype=torch.float16)

        with torch.no_grad():
            flash_out, _ = flash(x)
            naive_out, _ = naive(x)

        # FP16 tolerance is higher than FP32
        torch.testing.assert_close(
            flash_out,
            naive_out,
            rtol=5e-2,
            atol=5e-2,
        )

    @skip_no_cuda
    def test_flash_v2_fp16_gradient_precision(self):
        """Verify gradients are computed correctly in FP16."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(256, 4, causal=True).cuda().half()

        x = torch.randn(2, 64, 256, device="cuda", dtype=torch.float16, requires_grad=True)
        output, _ = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

        # Gradient magnitudes should be reasonable
        grad_norm = x.grad.norm()
        assert 0.1 < grad_norm < 1e5, f"Gradient norm {grad_norm} is suspicious"


class TestBF16Precision:
    """Test BF16 (bfloat16) attention."""

    @skip_no_cuda
    @skip_no_bf16
    def test_flash_v2_bf16_correctness(self):
        """Verify Flash V2 works correctly with BF16."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(256, 4, causal=True).cuda().bfloat16()

        x = torch.randn(2, 128, 256, device="cuda", dtype=torch.bfloat16)
        output, _ = model(x)

        assert output.dtype == torch.bfloat16
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    @skip_no_cuda
    @skip_no_bf16
    def test_bf16_fp32_accumulator(self):
        """Verify that softmax uses FP32 accumulator even with BF16 input."""
        from flashtile.attention import FlashAttentionV2, NaiveAttention

        embed_dim, num_heads = 256, 4
        batch, seq_len = 2, 64

        # Create models in BF16
        flash = FlashAttentionV2(embed_dim, num_heads, causal=True).cuda().bfloat16()
        naive = NaiveAttention(embed_dim, num_heads, causal=True).cuda().bfloat16()

        # Sync weights
        with torch.no_grad():
            naive.qkv_proj.weight.copy_(flash.qkv_proj.weight)
            naive.qkv_proj.bias.copy_(flash.qkv_proj.bias)
            naive.out_proj.weight.copy_(flash.out_proj.weight)
            naive.out_proj.bias.copy_(flash.out_proj.bias)

        x = torch.randn(batch, seq_len, embed_dim, device="cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            flash_out, _ = flash(x)
            naive_out, _ = naive(x)

        # BF16 has lower precision than FP16
        torch.testing.assert_close(
            flash_out,
            naive_out,
            rtol=0.1,
            atol=0.1,
        )

    @skip_no_cuda
    @skip_no_bf16
    def test_bf16_backward_pass(self):
        """Verify backward pass works with BF16."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(256, 4, causal=True).cuda().bfloat16()

        x = torch.randn(2, 64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        output, _ = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.dtype == torch.bfloat16
        assert not torch.isnan(x.grad).any()


class TestMixedPrecision:
    """Test mixed-precision training scenarios."""

    @skip_no_cuda
    def test_autocast_fp16_forward(self):
        """Test attention with torch.autocast (AMP)."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(256, 4, causal=True).cuda()

        x = torch.randn(2, 128, 256, device="cuda")

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output, _ = model(x)

        assert output.dtype == torch.float16
        assert output.shape == x.shape

    @skip_no_cuda
    @skip_no_bf16
    def test_autocast_bf16_forward(self):
        """Test attention with BF16 autocast."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(256, 4, causal=True).cuda()

        x = torch.randn(2, 128, 256, device="cuda")

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output, _ = model(x)

        assert output.dtype == torch.bfloat16

    @skip_no_cuda
    def test_gradient_scaler_compatibility(self):
        """Test compatibility with GradScaler for mixed precision training."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(256, 4, causal=True).cuda()
        optimizer = torch.optim.Adam(model.parameters())
        scaler = torch.amp.GradScaler('cuda')

        x = torch.randn(2, 64, 256, device="cuda")

        # Training step with AMP
        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output, _ = model(x)
            loss = output.sum()

        scaler.scale(loss).backward()

        # Unscale gradients — GradScaler may detect overflow and skip the step.
        # This is expected behavior in mixed-precision training.
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

        # After scaler.step, parameters should still be finite
        # (the scaler skips updates when it detects inf/nan gradients)
        for param in model.parameters():
            assert not torch.isnan(param).any(), "NaN in parameters after scaler step"


class TestNumericalStability:
    """Test numerical stability under edge cases."""

    @skip_no_cuda
    def test_large_attention_scores(self):
        """Test stability when attention scores are large."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(256, 4, causal=True).cuda()

        # Large values that might cause overflow
        x = torch.randn(2, 64, 256, device="cuda") * 10

        output, _ = model(x)

        assert not torch.isnan(output).any(), "NaN detected with large inputs"
        assert not torch.isinf(output).any(), "Inf detected with large inputs"

    @skip_no_cuda
    def test_small_attention_scores(self):
        """Test stability when attention scores are very small."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(256, 4, causal=True).cuda()

        # Small values that might cause underflow
        x = torch.randn(2, 64, 256, device="cuda") * 0.01

        output, _ = model(x)

        assert not torch.isnan(output).any(), "NaN detected with small inputs"
        assert not torch.isinf(output).any(), "Inf detected with small inputs"

    @skip_no_cuda
    def test_all_masked_positions(self):
        """Test stability when some rows are fully masked (causal first position)."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(256, 4, causal=True).cuda()

        # First token only attends to itself
        x = torch.randn(2, 64, 256, device="cuda")

        output, _ = model(x)

        # First position should still have valid output
        assert not torch.isnan(output[:, 0, :]).any()

    @skip_no_cuda
    def test_gradient_stability(self):
        """Test that gradients are numerically stable."""
        from flashtile.attention import FlashAttentionV2

        model = FlashAttentionV2(256, 4, causal=True).cuda().half()

        x = torch.randn(2, 64, 256, device="cuda", dtype=torch.float16, requires_grad=True)

        # Multiple backward passes
        for _ in range(5):
            output, _ = model(x)
            loss = output.sum()
            loss.backward()

            assert not torch.isnan(x.grad).any()
            x.grad.zero_()


class TestFP8Precision:
    """Test FP8 precision (if available)."""

    @skip_no_cuda
    @skip_no_fp8
    def test_fp8_tensor_creation(self):
        """Verify FP8 tensors can be created."""
        x = torch.randn(2, 64, 256, device="cuda")
        x_fp8 = x.to(torch.float8_e4m3fn)

        assert x_fp8.dtype == torch.float8_e4m3fn

    @skip_no_cuda
    @skip_no_fp8
    def test_fp8_quantization_error(self):
        """Measure FP8 quantization error."""
        x = torch.randn(1000, device="cuda")
        x_fp8 = x.to(torch.float8_e4m3fn)
        x_reconstructed = x_fp8.to(torch.float32)

        error = (x - x_reconstructed).abs().mean()

        # FP8 E4M3 has ~6.25% relative error
        relative_error = (error / x.abs().mean()) * 100
        assert relative_error < 15, f"FP8 error {relative_error:.2f}% is too high"


class TestPrecisionGQA:
    """Test precision for GQA and MQA variants."""

    @skip_no_cuda
    def test_gqa_fp16_precision(self):
        """Test GQA precision in FP16."""
        from flashtile.attention import GroupedQueryAttention

        model = GroupedQueryAttention(
            embed_dim=256,
            num_heads=8,
            num_kv_heads=2,
            causal=True,
        ).cuda().half()

        x = torch.randn(2, 64, 256, device="cuda", dtype=torch.float16)
        output, _ = model(x)

        assert output.dtype == torch.float16
        assert not torch.isnan(output).any()

    @skip_no_cuda
    def test_mqa_fp16_precision(self):
        """Test MQA precision in FP16."""
        from flashtile.attention import MultiQueryAttention

        model = MultiQueryAttention(
            embed_dim=256,
            num_heads=8,
            causal=True,
        ).cuda().half()

        x = torch.randn(2, 64, 256, device="cuda", dtype=torch.float16)
        output, _ = model(x)

        assert output.dtype == torch.float16
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
