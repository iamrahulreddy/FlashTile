"""
Tests for Grouped-Query Attention (GQA) and Multi-Query Attention (MQA).

Validates:
- GQA produces correct output shapes
- GQA with num_kv_heads=num_heads matches MHA
- MQA is equivalent to GQA with num_kv_heads=1
- Causal masking works correctly
- Memory savings are as expected

Run with: pytest tests/test_gqa.py -v
"""

import pytest
import torch
import math

# NOTE: For proper testing, install the package first: pip install -e .
# This sys.path workaround is for development convenience only.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from flashtile import GroupedQueryAttention, MultiQueryAttention, NaiveAttention


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def seed():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    return 42


def sync_gqa_to_naive(gqa: GroupedQueryAttention, naive: NaiveAttention) -> None:
    """Copy equivalent GQA weights into a standard MHA module."""
    embed_dim = gqa.embed_dim

    with torch.no_grad():
        naive.qkv_proj.weight[:embed_dim].copy_(gqa.q_proj.weight)
        naive.qkv_proj.bias[:embed_dim].copy_(gqa.q_proj.bias)

        naive.qkv_proj.weight[embed_dim:2 * embed_dim].copy_(gqa.k_proj.weight)
        naive.qkv_proj.bias[embed_dim:2 * embed_dim].copy_(gqa.k_proj.bias)

        naive.qkv_proj.weight[2 * embed_dim:].copy_(gqa.v_proj.weight)
        naive.qkv_proj.bias[2 * embed_dim:].copy_(gqa.v_proj.bias)

        naive.out_proj.weight.copy_(gqa.out_proj.weight)
        naive.out_proj.bias.copy_(gqa.out_proj.bias)


class TestGQABasic:
    """Basic GQA functionality tests."""
    
    @pytest.mark.parametrize("num_kv_heads", [1, 2, 4, 8])
    def test_output_shape(self, num_kv_heads, device, seed):
        """GQA should produce correct output shape."""
        batch, seq_len, embed_dim, num_heads = 2, 128, 512, 8
        
        model = GroupedQueryAttention(
            embed_dim, num_heads, num_kv_heads=num_kv_heads
        ).to(device).eval()
        
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        with torch.inference_mode():
            out, _ = model(x)
        
        assert out.shape == (batch, seq_len, embed_dim)
    
    def test_no_nan_or_inf(self, device, seed):
        """GQA output should not contain NaN or Inf."""
        batch, seq_len, embed_dim, num_heads = 2, 128, 512, 8
        
        model = GroupedQueryAttention(
            embed_dim, num_heads, num_kv_heads=2
        ).to(device).eval()
        
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        with torch.inference_mode():
            out, _ = model(x)
        
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"
    
    def test_deterministic(self, device, seed):
        """GQA should produce deterministic output."""
        batch, seq_len, embed_dim, num_heads = 2, 64, 256, 8
        
        model = GroupedQueryAttention(
            embed_dim, num_heads, num_kv_heads=2
        ).to(device).eval()
        
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        with torch.inference_mode():
            out1, _ = model(x)
            out2, _ = model(x)
        
        assert torch.allclose(out1, out2)


class TestGQACorrectness:
    """GQA correctness tests."""
    
    def test_full_heads_matches_naive(self, device, seed):
        """GQA with num_kv_heads=num_heads should behave like standard attention."""
        batch, seq_len, embed_dim, num_heads = 2, 64, 256, 4
        
        # GQA with all heads
        gqa = GroupedQueryAttention(
            embed_dim, num_heads, num_kv_heads=num_heads
        ).to(device).eval()

        naive = NaiveAttention(embed_dim, num_heads).to(device).eval()
        sync_gqa_to_naive(gqa, naive)
        
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        with torch.inference_mode():
            naive_out, _ = naive(x)
            gqa_out, _ = gqa(x)
        
        max_diff = (naive_out - gqa_out).abs().max().item()
        assert max_diff < 1e-3, f"GQA with full KV heads should match MHA, diff={max_diff:.6f}"
    
    def test_causal_property(self, device, seed):
        """Verify causal masking in GQA."""
        batch, seq_len, embed_dim, num_heads = 1, 64, 256, 8
        
        model = GroupedQueryAttention(
            embed_dim, num_heads, num_kv_heads=2, causal=True
        ).to(device).eval()
        
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        with torch.inference_mode():
            out1, _ = model(x)
            
            # Modify a future token
            x_modified = x.clone()
            x_modified[0, 32, :] = torch.randn(embed_dim, device=device)
            out2, _ = model(x_modified)
        
        # Positions 0-31 should be unchanged
        max_diff = (out1[0, :32, :] - out2[0, :32, :]).abs().max().item()
        assert max_diff < 1e-5, f"Causal violation: diff={max_diff}"


class TestMQA:
    """Multi-Query Attention tests."""
    
    def test_mqa_output_shape(self, device, seed):
        """MQA should produce correct output shape."""
        batch, seq_len, embed_dim, num_heads = 2, 128, 512, 8
        
        model = MultiQueryAttention(embed_dim, num_heads).to(device).eval()
        
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        with torch.inference_mode():
            out, _ = model(x)
        
        assert out.shape == (batch, seq_len, embed_dim)
    
    def test_mqa_equals_gqa_one_head(self, device, seed):
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
        
        assert torch.allclose(mqa_out, gqa_out, atol=1e-6)
    
    def test_mqa_causal(self, device, seed):
        """MQA with causal masking."""
        batch, seq_len, embed_dim, num_heads = 2, 64, 256, 8
        
        model = MultiQueryAttention(embed_dim, num_heads, causal=True).to(device).eval()
        
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        with torch.inference_mode():
            out, _ = model(x)
        
        assert out.shape == (batch, seq_len, embed_dim)
        assert not torch.isnan(out).any()


class TestGQAMemory:
    """Memory usage tests for GQA."""
    
    def test_memory_savings_reported(self, device, seed):
        """GQA should report correct memory savings."""
        embed_dim, num_heads = 512, 8
        
        for num_kv_heads in [1, 2, 4, 8]:
            model = GroupedQueryAttention(
                embed_dim, num_heads, num_kv_heads=num_kv_heads
            )
            
            mem = model.get_memory_usage(batch_size=2, seq_len=1024)
            
            expected_ratio = num_heads / num_kv_heads
            assert abs(mem['kv_savings_ratio'] - expected_ratio) < 0.01, \
                f"Expected {expected_ratio}x savings, got {mem['kv_savings_ratio']}x"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_actual_memory_reduction(self, seed):
        """GQA should use less memory than MHA."""
        device = "cuda"
        batch, seq_len, embed_dim, num_heads = 2, 2048, 512, 8
        
        x = torch.randn(batch, seq_len, embed_dim, device=device)

        def measure_peak_memory(model: GroupedQueryAttention) -> int:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                _ = model(x)
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated()

        mha = GroupedQueryAttention(
            embed_dim, num_heads, num_kv_heads=num_heads
        ).to(device).eval()
        mha_memory = measure_peak_memory(mha)
        del mha
        torch.cuda.empty_cache()

        gqa = GroupedQueryAttention(
            embed_dim, num_heads, num_kv_heads=2
        ).to(device).eval()
        gqa_memory = measure_peak_memory(gqa)

        assert gqa_memory < mha_memory, (
            f"Expected GQA to reduce peak memory: "
            f"MHA={mha_memory / 1e6:.1f} MB, GQA={gqa_memory / 1e6:.1f} MB"
        )


class TestGQAGradients:
    """Gradient tests for GQA."""
    
    def test_backward_runs(self, device, seed):
        """GQA backward pass should complete."""
        batch, seq_len, embed_dim, num_heads = 2, 64, 256, 8
        
        model = GroupedQueryAttention(
            embed_dim, num_heads, num_kv_heads=2
        ).to(device)
        
        x = torch.randn(batch, seq_len, embed_dim, device=device, requires_grad=True)
        
        out, _ = model(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_mqa_backward_runs(self, device, seed):
        """MQA backward pass should complete."""
        batch, seq_len, embed_dim, num_heads = 2, 64, 256, 8
        
        model = MultiQueryAttention(embed_dim, num_heads).to(device)
        
        x = torch.randn(batch, seq_len, embed_dim, device=device, requires_grad=True)
        
        out, _ = model(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestGQAEdgeCases:
    """Edge case tests for GQA."""
    
    def test_single_token(self, device, seed):
        """GQA should handle single-token sequences."""
        batch, seq_len, embed_dim, num_heads = 2, 1, 256, 8
        
        model = GroupedQueryAttention(
            embed_dim, num_heads, num_kv_heads=2
        ).to(device).eval()
        
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        with torch.inference_mode():
            out, _ = model(x)
        
        assert out.shape == (batch, seq_len, embed_dim)
    
    def test_non_divisible_sequence(self, device, seed):
        """GQA should handle sequences not divisible by block size."""
        batch, seq_len, embed_dim, num_heads = 2, 100, 256, 8
        
        model = GroupedQueryAttention(
            embed_dim, num_heads, num_kv_heads=2, block_size=64
        ).to(device).eval()
        
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        with torch.inference_mode():
            out, _ = model(x)
        
        assert out.shape == (batch, seq_len, embed_dim)
    
    def test_large_values(self, device, seed):
        """GQA should handle large input values."""
        batch, seq_len, embed_dim, num_heads = 2, 64, 256, 8
        
        model = GroupedQueryAttention(
            embed_dim, num_heads, num_kv_heads=2
        ).to(device).eval()
        
        x = torch.randn(batch, seq_len, embed_dim, device=device) * 100
        
        with torch.inference_mode():
            out, _ = model(x)
        
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
