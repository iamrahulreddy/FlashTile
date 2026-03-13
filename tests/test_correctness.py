"""
Comprehensive Correctness Tests for FlashTile Attention Implementations.

Tests verify that all Flash Attention variants produce identical outputs to
naive attention within floating-point tolerance.

Coverage:
- FlashAttentionV1 vs Naive
- FlashAttentionV2 vs Naive
- GQA/MQA vs expanded MHA
- Triton kernel vs PyTorch reference
- Backward pass gradients
- Edge cases and numerical stability

Run with: pytest tests/test_correctness.py -v
"""

import pytest
import torch
import math
from typing import Tuple

# NOTE: For proper testing, install the package first: pip install -e .
# This sys.path workaround is for development convenience only.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from flashtile import (
    NaiveAttention, 
    FlashAttentionV1, 
    FlashAttentionV2,
    GroupedQueryAttention,
    MultiQueryAttention,
)


# =============================================================================
# Test Configuration
# =============================================================================

TOLERANCE_FP32 = 1e-3
TOLERANCE_FP16 = 1e-2

# Test configurations: (batch, seq_len, embed_dim, num_heads)
TEST_CONFIGS = [
    (1, 64, 256, 4),
    (2, 128, 512, 8),
    (1, 256, 512, 8),
    (4, 64, 256, 4),
    (1, 512, 512, 8),
]

BLOCK_SIZES = [32, 64, 128]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def seed():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    return 42


def sync_weights(flash_model, naive_model):
    """Copy weights from naive to flash model."""
    flash_model.qkv_proj.weight.data = naive_model.qkv_proj.weight.data.clone()
    flash_model.qkv_proj.bias.data = naive_model.qkv_proj.bias.data.clone()
    flash_model.out_proj.weight.data = naive_model.out_proj.weight.data.clone()
    flash_model.out_proj.bias.data = naive_model.out_proj.bias.data.clone()


# =============================================================================
# Flash V1 Tests
# =============================================================================

class TestFlashV1Correctness:
    """Tests for FlashAttentionV1 correctness."""
    
    @pytest.mark.parametrize("batch,seq_len,embed_dim,num_heads", TEST_CONFIGS)
    def test_output_matches_naive(self, batch, seq_len, embed_dim, num_heads, device, seed):
        """Flash V1 output should match naive attention."""
        naive = NaiveAttention(embed_dim, num_heads).to(device).eval()
        flash = FlashAttentionV1(embed_dim, num_heads, block_size=64).to(device).eval()
        sync_weights(flash, naive)
        
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        with torch.inference_mode():
            naive_out, _ = naive(x)
            flash_out, _ = flash(x)
        
        max_diff = (naive_out - flash_out).abs().max().item()
        assert max_diff < TOLERANCE_FP32, f"Max diff {max_diff:.6f} exceeds tolerance"
    
    @pytest.mark.parametrize("block_size", BLOCK_SIZES)
    def test_different_block_sizes(self, block_size, device, seed):
        """Flash V1 should work with different block sizes."""
        batch, seq_len, embed_dim, num_heads = 2, 128, 512, 8
        
        naive = NaiveAttention(embed_dim, num_heads).to(device).eval()
        flash = FlashAttentionV1(embed_dim, num_heads, block_size=block_size).to(device).eval()
        sync_weights(flash, naive)
        
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        with torch.inference_mode():
            naive_out, _ = naive(x)
            flash_out, _ = flash(x)
        
        max_diff = (naive_out - flash_out).abs().max().item()
        assert max_diff < TOLERANCE_FP32
    
    def test_causal_masking(self, device, seed):
        """Flash V1 with causal flag should match naive with causal mask."""
        batch, seq_len, embed_dim, num_heads = 2, 64, 256, 4
        
        naive = NaiveAttention(embed_dim, num_heads).to(device).eval()
        flash = FlashAttentionV1(embed_dim, num_heads, causal=True).to(device).eval()
        sync_weights(flash, naive)
        
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        # Create causal mask for naive (boolean: True = mask out)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        
        with torch.inference_mode():
            naive_out, _ = naive(x, attn_mask=causal_mask)
            flash_out, _ = flash(x)
        
        max_diff = (naive_out - flash_out).abs().max().item()
        assert max_diff < TOLERANCE_FP32


# =============================================================================
# Flash V2 Tests
# =============================================================================

class TestFlashV2Correctness:
    """Tests for FlashAttentionV2 correctness."""
    
    def test_v2_matches_v1(self, device, seed):
        """V2 output should match V1."""
        batch, seq_len, embed_dim, num_heads = 2, 128, 512, 8
        
        v1 = FlashAttentionV1(embed_dim, num_heads).to(device).eval()
        v2 = FlashAttentionV2(embed_dim, num_heads).to(device).eval()
        
        # Sync weights
        v2.qkv_proj.weight.data = v1.qkv_proj.weight.data.clone()
        v2.qkv_proj.bias.data = v1.qkv_proj.bias.data.clone()
        v2.out_proj.weight.data = v1.out_proj.weight.data.clone()
        v2.out_proj.bias.data = v1.out_proj.bias.data.clone()
        
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        with torch.inference_mode():
            v1_out, _ = v1(x)
            v2_out, _ = v2(x)
        
        max_diff = (v1_out - v2_out).abs().max().item()
        assert max_diff < TOLERANCE_FP32
    
    def test_v2_causal_matches_naive(self, device, seed):
        """V2 causal should match naive with explicit mask."""
        batch, seq_len, embed_dim, num_heads = 2, 64, 256, 4
        
        naive = NaiveAttention(embed_dim, num_heads).to(device).eval()
        v2 = FlashAttentionV2(embed_dim, num_heads, causal=True).to(device).eval()
        
        v2.qkv_proj.weight.data = naive.qkv_proj.weight.data.clone()
        v2.qkv_proj.bias.data = naive.qkv_proj.bias.data.clone()
        v2.out_proj.weight.data = naive.out_proj.weight.data.clone()
        v2.out_proj.bias.data = naive.out_proj.bias.data.clone()
        
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        
        with torch.inference_mode():
            naive_out, _ = naive(x, attn_mask=causal_mask)
            v2_out, _ = v2(x)
        
        max_diff = (naive_out - v2_out).abs().max().item()
        assert max_diff < TOLERANCE_FP32
    
    def test_causal_property(self, device, seed):
        """Verify causal masking: position i should only depend on j <= i."""
        batch, seq_len, embed_dim, num_heads = 1, 64, 256, 4
        
        model = FlashAttentionV2(embed_dim, num_heads, causal=True).to(device).eval()
        
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        with torch.inference_mode():
            out1, _ = model(x)
            
            # Modify a future token
            x_modified = x.clone()
            x_modified[0, 32, :] = torch.randn(embed_dim, device=device)
            out2, _ = model(x_modified)
        
        # Positions 0-31 should be unchanged
        max_diff_before = (out1[0, :32, :] - out2[0, :32, :]).abs().max().item()
        assert max_diff_before < 1e-5, "Causal violation: earlier positions changed"
        
        # Positions 32+ should be different
        max_diff_after = (out1[0, 32:, :] - out2[0, 32:, :]).abs().max().item()
        assert max_diff_after > 0.01, "Causal masking not working"


# =============================================================================
# GQA/MQA Tests
# =============================================================================

class TestGQACorrectness:
    """Tests for Grouped-Query Attention correctness."""
    
    def test_gqa_with_full_heads_matches_mha(self, device, seed):
        """GQA with num_kv_heads=num_heads should match standard attention."""
        batch, seq_len, embed_dim, num_heads = 2, 64, 256, 4
        
        # GQA with all heads (equivalent to MHA)
        gqa = GroupedQueryAttention(
            embed_dim, num_heads, num_kv_heads=num_heads
        ).to(device).eval()
        
        naive = NaiveAttention(embed_dim, num_heads).to(device).eval()
        
        # Sync Q projection
        naive.qkv_proj.weight.data[:embed_dim] = gqa.q_proj.weight.data.clone()
        naive.qkv_proj.bias.data[:embed_dim] = gqa.q_proj.bias.data.clone()
        
        # Sync K projection
        naive.qkv_proj.weight.data[embed_dim:2*embed_dim] = gqa.k_proj.weight.data.clone()
        naive.qkv_proj.bias.data[embed_dim:2*embed_dim] = gqa.k_proj.bias.data.clone()
        
        # Sync V projection
        naive.qkv_proj.weight.data[2*embed_dim:] = gqa.v_proj.weight.data.clone()
        naive.qkv_proj.bias.data[2*embed_dim:] = gqa.v_proj.bias.data.clone()
        
        # Sync output projection
        naive.out_proj.weight.data = gqa.out_proj.weight.data.clone()
        naive.out_proj.bias.data = gqa.out_proj.bias.data.clone()
        
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        with torch.inference_mode():
            naive_out, _ = naive(x)
            gqa_out, _ = gqa(x)
        
        max_diff = (naive_out - gqa_out).abs().max().item()
        assert max_diff < TOLERANCE_FP32
    
    @pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
    def test_gqa_output_shape(self, num_kv_heads, device, seed):
        """GQA should produce correct output shape."""
        batch, seq_len, embed_dim, num_heads = 2, 64, 256, 8
        
        gqa = GroupedQueryAttention(
            embed_dim, num_heads, num_kv_heads=num_kv_heads
        ).to(device).eval()
        
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        with torch.inference_mode():
            out, _ = gqa(x)
        
        assert out.shape == (batch, seq_len, embed_dim)
    
    def test_mqa_is_gqa_with_one_head(self, device, seed):
        """MQA should be equivalent to GQA with num_kv_heads=1."""
        batch, seq_len, embed_dim, num_heads = 2, 64, 256, 8
        
        mqa = MultiQueryAttention(embed_dim, num_heads).to(device).eval()
        gqa = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads=1).to(device).eval()
        
        # Sync weights
        gqa.q_proj.weight.data = mqa.q_proj.weight.data.clone()
        gqa.q_proj.bias.data = mqa.q_proj.bias.data.clone()
        gqa.k_proj.weight.data = mqa.k_proj.weight.data.clone()
        gqa.k_proj.bias.data = mqa.k_proj.bias.data.clone()
        gqa.v_proj.weight.data = mqa.v_proj.weight.data.clone()
        gqa.v_proj.bias.data = mqa.v_proj.bias.data.clone()
        gqa.out_proj.weight.data = mqa.out_proj.weight.data.clone()
        gqa.out_proj.bias.data = mqa.out_proj.bias.data.clone()
        
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        with torch.inference_mode():
            mqa_out, _ = mqa(x)
            gqa_out, _ = gqa(x)
        
        max_diff = (mqa_out - gqa_out).abs().max().item()
        assert max_diff < 1e-6


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests for robustness."""
    
    def test_sequence_not_divisible_by_block(self, device, seed):
        """Handle sequences not evenly divisible by block size."""
        batch, seq_len, embed_dim, num_heads = 2, 100, 256, 4
        
        naive = NaiveAttention(embed_dim, num_heads).to(device).eval()
        flash = FlashAttentionV1(embed_dim, num_heads, block_size=64).to(device).eval()
        sync_weights(flash, naive)
        
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        with torch.inference_mode():
            naive_out, _ = naive(x)
            flash_out, _ = flash(x)
        
        max_diff = (naive_out - flash_out).abs().max().item()
        assert max_diff < TOLERANCE_FP32
    
    def test_single_token_sequence(self, device, seed):
        """Handle single-token sequences."""
        batch, seq_len, embed_dim, num_heads = 2, 1, 256, 4
        
        naive = NaiveAttention(embed_dim, num_heads).to(device).eval()
        flash = FlashAttentionV1(embed_dim, num_heads).to(device).eval()
        sync_weights(flash, naive)
        
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        with torch.inference_mode():
            naive_out, _ = naive(x)
            flash_out, _ = flash(x)
        
        max_diff = (naive_out - flash_out).abs().max().item()
        assert max_diff < TOLERANCE_FP32
    
    def test_very_small_values(self, device, seed):
        """Handle very small input values."""
        batch, seq_len, embed_dim, num_heads = 2, 64, 256, 4
        
        naive = NaiveAttention(embed_dim, num_heads).to(device).eval()
        flash = FlashAttentionV1(embed_dim, num_heads).to(device).eval()
        sync_weights(flash, naive)
        
        x = torch.randn(batch, seq_len, embed_dim, device=device) * 1e-5
        
        with torch.inference_mode():
            naive_out, _ = naive(x)
            flash_out, _ = flash(x)
        
        max_diff = (naive_out - flash_out).abs().max().item()
        assert max_diff < TOLERANCE_FP32 * 10
    
    def test_large_values(self, device, seed):
        """Handle large input values without overflow."""
        batch, seq_len, embed_dim, num_heads = 2, 64, 256, 4
        
        flash = FlashAttentionV1(embed_dim, num_heads).to(device).eval()
        
        x = torch.randn(batch, seq_len, embed_dim, device=device) * 100
        
        with torch.inference_mode():
            out, _ = flash(x)
        
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"


# =============================================================================
# Gradient Tests
# =============================================================================

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestGradients:
    
    def test_v1_backward_runs(self, device, seed):
        """V1 backward pass should complete without error."""
        batch, seq_len, embed_dim, num_heads = 2, 64, 256, 4
        
        flash = FlashAttentionV1(embed_dim, num_heads).to(device)
        x = torch.randn(batch, seq_len, embed_dim, device=device, requires_grad=True)
        
        out, _ = flash(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_v2_backward_runs(self, device, seed):
        """V2 backward pass should complete without error."""
        batch, seq_len, embed_dim, num_heads = 2, 64, 256, 4
        
        flash = FlashAttentionV2(embed_dim, num_heads, causal=True).to(device)
        x = torch.randn(batch, seq_len, embed_dim, device=device, requires_grad=True)
        
        out, _ = flash(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_gradients_match_naive(self, device, seed):
        """Flash gradients should approximately match naive gradients."""
        batch, seq_len, embed_dim, num_heads = 2, 32, 128, 4
        
        naive = NaiveAttention(embed_dim, num_heads).to(device)
        flash = FlashAttentionV1(embed_dim, num_heads).to(device)
        sync_weights(flash, naive)
        
        x_naive = torch.randn(batch, seq_len, embed_dim, device=device, requires_grad=True)
        x_flash = x_naive.detach().clone().requires_grad_(True)
        
        # Forward + backward
        naive_out, _ = naive(x_naive)
        flash_out, _ = flash(x_flash)
        
        naive_out.sum().backward()
        flash_out.sum().backward()
        
        # Gradients should be close (not exact due to recomputation)
        grad_diff = (x_naive.grad - x_flash.grad).abs().max().item()
        assert grad_diff < 0.1, f"Gradient diff {grad_diff} too large"


# =============================================================================
# Memory Tests
# =============================================================================

@pytest.mark.gpu
class TestMemoryUsage:
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_flash_uses_less_memory(self, seed):
        """Flash attention should use less memory than naive."""
        batch, seq_len, embed_dim, num_heads = 2, 512, 512, 8
        device = "cuda"
        
        naive = NaiveAttention(embed_dim, num_heads).to(device).eval()
        flash = FlashAttentionV1(embed_dim, num_heads).to(device).eval()
        
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        # Measure naive
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = naive(x)
        naive_memory = torch.cuda.max_memory_allocated()
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Measure flash
        with torch.no_grad():
            _ = flash(x)
        flash_memory = torch.cuda.max_memory_allocated()
        
        assert flash_memory < naive_memory


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
