import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn.functional as F

from flashtile import SlidingWindowAttention, NaiveAttention


class TestSlidingWindowCorrectness:
    """Tests for SlidingWindowAttention correctness."""
    
    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def test_sliding_window_matches_naive_with_window_mask(self, device):
        """Sliding window should match naive attention with explicit window mask."""
        batch, seq_len, embed_dim, num_heads = 2, 64, 256, 4
        window_size = 16
        
        # Create models
        naive = NaiveAttention(embed_dim, num_heads).to(device).eval()
        sliding = SlidingWindowAttention(
            embed_dim, num_heads, window_size=window_size, causal=False
        ).to(device).eval()
        
        # Sync weights
        sliding.qkv_proj.weight.data = naive.qkv_proj.weight.data.clone()
        sliding.qkv_proj.bias.data = naive.qkv_proj.bias.data.clone()
        sliding.out_proj.weight.data = naive.out_proj.weight.data.clone()
        sliding.out_proj.bias.data = naive.out_proj.bias.data.clone()
        
        # Create input
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        # Create window mask for naive attention
        # Bidirectional window: attend to [i - window//2, i + window//2]
        half_window = window_size // 2
        positions = torch.arange(seq_len, device=device)
        mask = (
            (positions[None, :] < positions[:, None] - half_window) |
            (positions[None, :] > positions[:, None] + half_window)
        )
        
        # Forward pass
        with torch.no_grad():
            naive_out, _ = naive(x, attn_mask=mask)
            sliding_out, _ = sliding(x)
        
        # Check outputs match
        max_diff = (naive_out - sliding_out).abs().max().item()
        assert max_diff < 1e-4, f"Max diff {max_diff} exceeds tolerance"
    
    def test_causal_sliding_window_matches_naive(self, device):
        """Causal sliding window should match naive with causal + window mask."""
        batch, seq_len, embed_dim, num_heads = 2, 64, 256, 4
        window_size = 16
        
        # Create models
        naive = NaiveAttention(embed_dim, num_heads).to(device).eval()
        sliding = SlidingWindowAttention(
            embed_dim, num_heads, window_size=window_size, causal=True
        ).to(device).eval()
        
        # Sync weights
        sliding.qkv_proj.weight.data = naive.qkv_proj.weight.data.clone()
        sliding.qkv_proj.bias.data = naive.qkv_proj.bias.data.clone()
        sliding.out_proj.weight.data = naive.out_proj.weight.data.clone()
        sliding.out_proj.bias.data = naive.out_proj.bias.data.clone()
        
        # Create input
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        # Create causal + window mask for naive attention
        # Attend to [max(0, i - window + 1), i]
        positions = torch.arange(seq_len, device=device)
        causal_mask = positions[None, :] > positions[:, None]  # Future positions
        window_start = (positions - window_size + 1).clamp(min=0)
        window_mask = positions[None, :] < window_start[:, None]  # Beyond window
        mask = (causal_mask | window_mask)
        
        # Forward pass
        with torch.no_grad():
            naive_out, _ = naive(x, attn_mask=mask)
            sliding_out, _ = sliding(x)
        
        # Check outputs match
        max_diff = (naive_out - sliding_out).abs().max().item()
        assert max_diff < 1e-4, f"Max diff {max_diff} exceeds tolerance"
    
    def test_window_boundary_conditions(self, device):
        """Test that window boundaries are handled correctly."""
        batch, seq_len, embed_dim, num_heads = 1, 32, 128, 4
        window_size = 8
        
        model = SlidingWindowAttention(
            embed_dim, num_heads, window_size=window_size, causal=True
        ).to(device).eval()
        
        # Create input
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        with torch.no_grad():
            out1, _ = model(x)
        
        x_modified = x.clone()
        x_modified[0, 10, :] = torch.randn(embed_dim, device=device)
        
        with torch.no_grad():
            out2, _ = model(x_modified)
        
        # Position 0 should be unchanged (position 10 is outside its window)
        diff_at_0 = (out1[0, 0, :] - out2[0, 0, :]).abs().max().item()
        assert diff_at_0 < 1e-6, f"Position 0 changed when modifying position 10: diff={diff_at_0}"
    
    def test_sliding_window_backward(self, device):
        """Test that backward pass works."""
        batch, seq_len, embed_dim, num_heads = 2, 32, 128, 4
        
        model = SlidingWindowAttention(embed_dim, num_heads, window_size=16).to(device)
        x = torch.randn(batch, seq_len, embed_dim, device=device, requires_grad=True)
        
        # Forward + backward
        output, _ = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
    
    def test_memory_usage_report(self):
        """Test memory usage calculation."""
        model = SlidingWindowAttention(512, 8, window_size=128)
        
        mem_info = model.get_memory_usage(batch_size=2, seq_len=1024)
        
        assert "window_mb" in mem_info
        assert "blocks_per_window" in mem_info
        assert "complexity" in mem_info
        assert "comparison_with_naive" in mem_info
        
        # Window should reduce memory vs full attention
        assert mem_info["comparison_with_naive"]["reduction_ratio"] > 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
