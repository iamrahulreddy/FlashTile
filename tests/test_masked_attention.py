"""
Tests for Masked Attention Fallback
===================================

Tests for the custom mask support with memory-efficient chunked processing.

Run with: pytest tests/test_masked_attention.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn.functional as F

from flashtile import (
    NaiveAttention,
    MaskedAttention,
    create_padding_mask,
    create_causal_mask,
)


class TestMaskedAttention:
    """Tests for MaskedAttention implementation."""
    
    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def test_masked_attention_matches_naive(self, device):
        """Masked attention output should match naive attention with same mask."""
        batch, seq_len, embed_dim, num_heads = 2, 64, 256, 4
        
        # Create models
        naive = NaiveAttention(embed_dim, num_heads).to(device).eval()
        masked = MaskedAttention(embed_dim, num_heads).to(device).eval()
        
        # Sync weights
        masked.qkv_proj.weight.data = naive.qkv_proj.weight.data.clone()
        masked.qkv_proj.bias.data = naive.qkv_proj.bias.data.clone()
        masked.out_proj.weight.data = naive.out_proj.weight.data.clone()
        masked.out_proj.bias.data = naive.out_proj.bias.data.clone()
        
        # Create input
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        # Create random mask
        mask = torch.randn(batch, seq_len, seq_len, device=device) > 0.5
        
        # Forward pass
        with torch.no_grad():
            naive_out, _ = naive(x, attn_mask=mask)
            masked_out, _ = masked(x, attn_mask=mask)
        
        # Check outputs match
        max_diff = (naive_out - masked_out).abs().max().item()
        assert max_diff < 1e-4, f"Max diff {max_diff} exceeds tolerance"
    
    def test_padding_mask(self, device):
        """Test padding mask functionality."""
        batch, max_len, embed_dim, num_heads = 3, 100, 256, 4
        
        # Variable lengths
        lengths = torch.tensor([80, 100, 60])
        
        # Create padding mask (boolean) — MaskedAttention handles bool masks directly
        pad_mask = create_padding_mask(lengths, max_len, device=device)
        
        # Create model
        model = MaskedAttention(embed_dim, num_heads).to(device).eval()
        
        # Create input
        x = torch.randn(batch, max_len, embed_dim, device=device)
        
        # Forward pass
        with torch.no_grad():
            output, _ = model(x, attn_mask=pad_mask)
        
        # Check output shape
        assert output.shape == (batch, max_len, embed_dim)
        
        # Check that padded positions produce finite outputs
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_causal_mask(self, device):
        """Test causal mask functionality."""
        seq_len = 64
        
        # Create causal mask (boolean) — MaskedAttention handles bool masks directly
        mask = create_causal_mask(seq_len, device=device)
        
        # Create model
        model = MaskedAttention(256, 4).to(device).eval()
        
        # Create input
        x = torch.randn(2, seq_len, 256, device=device)
        
        # Forward pass
        with torch.no_grad():
            output, _ = model(x, attn_mask=mask)
        
        # Check output shape
        assert output.shape == (2, seq_len, 256)
        
        # Verify causal property: position i should not depend on j > i
        x_modified = x.clone()
        x_modified[0, 50:, :] = torch.randn_like(x_modified[0, 50:, :])
        
        with torch.no_grad():
            out_original = model(x, attn_mask=mask)[0]
            out_modified = model(x_modified, attn_mask=mask)[0]
        
        # Positions 0-49 should be identical
        assert torch.allclose(out_original[0, :50], out_modified[0, :50], atol=1e-6)
    
    def test_no_mask_matches_no_mask(self, device):
        """Masked attention without mask should match naive attention without mask."""
        batch, seq_len, embed_dim, num_heads = 2, 64, 256, 4
        
        # Create models
        naive = NaiveAttention(embed_dim, num_heads).to(device).eval()
        masked = MaskedAttention(embed_dim, num_heads).to(device).eval()
        
        # Sync weights
        masked.qkv_proj.weight.data = naive.qkv_proj.weight.data.clone()
        masked.qkv_proj.bias.data = naive.qkv_proj.bias.data.clone()
        masked.out_proj.weight.data = naive.out_proj.weight.data.clone()
        masked.out_proj.bias.data = naive.out_proj.bias.data.clone()
        
        # Create input
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        # Forward pass without mask
        with torch.no_grad():
            naive_out, _ = naive(x)
            masked_out, _ = masked(x)
        
        # Check outputs match
        max_diff = (naive_out - masked_out).abs().max().item()
        assert max_diff < 1e-4, f"Max diff {max_diff} exceeds tolerance"
    
    def test_backward_pass(self, device):
        """Test that backward pass works with custom mask."""
        batch, seq_len, embed_dim, num_heads = 2, 32, 128, 4
        
        # Create model
        model = MaskedAttention(embed_dim, num_heads).to(device)
        
        # Create input with requires_grad
        x = torch.randn(batch, seq_len, embed_dim, device=device, requires_grad=True)
        
        # Create random mask
        mask = torch.randn(batch, seq_len, seq_len, device=device) > 0.5
        
        # Forward + backward
        output, _ = model(x, attn_mask=mask)
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
    
    def test_memory_usage_report(self):
        """Test that memory usage calculation works."""
        model = MaskedAttention(512, 8, chunk_size=512)
        
        mem_info = model.get_memory_usage(batch_size=2, seq_len=1024)
        
        assert "qkv_mb" in mem_info
        assert "mask_mb" in mem_info
        assert "total_mb" in mem_info
        assert "chunk_size" in mem_info
        
        # Mask should be O(N²)
        assert mem_info["mask_mb"] > mem_info["qkv_mb"]
    
    def test_chunked_processing(self, device):
        """Test that chunked processing produces correct results."""
        batch, seq_len, embed_dim, num_heads = 2, 100, 256, 4
        
        # Create model with small chunk size
        model_small_chunk = MaskedAttention(embed_dim, num_heads, chunk_size=32).to(device).eval()
        model_large_chunk = MaskedAttention(embed_dim, num_heads, chunk_size=512).to(device).eval()
        
        # Sync weights
        model_small_chunk.load_state_dict(model_large_chunk.state_dict())
        
        # Create input
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        # Create random mask
        mask = torch.randn(batch, seq_len, seq_len, device=device) > 0.5
        
        # Forward pass
        with torch.no_grad():
            out_small, _ = model_small_chunk(x, attn_mask=mask)
            out_large, _ = model_large_chunk(x, attn_mask=mask)
        
        # Both should produce same results
        max_diff = (out_small - out_large).abs().max().item()
        assert max_diff < 1e-5, f"Chunk size shouldn't affect output: diff={max_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
