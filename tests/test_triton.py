import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch

from flashtile import NaiveAttention, TritonFlashAttention, HAS_TRITON


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestTritonCorrectness:
    """Tests for TritonFlashAttention correctness."""
    
    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def test_triton_matches_naive(self, device):
        """Triton kernel output should match naive attention."""
        if device != "cuda":
            pytest.skip("Triton requires CUDA")
        
        batch, seq_len, embed_dim, num_heads = 2, 64, 256, 4
        
        # Create models
        naive = NaiveAttention(embed_dim, num_heads).to(device).eval()
        triton_model = TritonFlashAttention(embed_dim, num_heads).to(device).eval()
        
        # Sync weights
        triton_model.qkv_proj.weight.data = naive.qkv_proj.weight.data.clone()
        triton_model.qkv_proj.bias.data = naive.qkv_proj.bias.data.clone()
        triton_model.out_proj.weight.data = naive.out_proj.weight.data.clone()
        triton_model.out_proj.bias.data = naive.out_proj.bias.data.clone()
        
        # Create input
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        # Forward pass
        with torch.no_grad():
            naive_out, _ = naive(x)
            triton_out, _ = triton_model(x)
        
        # Check outputs match (Triton may have slightly different numerics)
        max_diff = (naive_out - triton_out).abs().max().item()
        assert max_diff < 1e-3, f"Max diff {max_diff} exceeds tolerance"
    
    def test_triton_causal_matches_naive(self, device):
        """Triton causal should match naive with causal mask."""
        if device != "cuda":
            pytest.skip("Triton requires CUDA")
        
        batch, seq_len, embed_dim, num_heads = 2, 64, 256, 4
        
        # Create models
        naive = NaiveAttention(embed_dim, num_heads).to(device).eval()
        triton_model = TritonFlashAttention(embed_dim, num_heads, causal=True).to(device).eval()
        
        # Sync weights
        triton_model.qkv_proj.weight.data = naive.qkv_proj.weight.data.clone()
        triton_model.qkv_proj.bias.data = naive.qkv_proj.bias.data.clone()
        triton_model.out_proj.weight.data = naive.out_proj.weight.data.clone()
        triton_model.out_proj.bias.data = naive.out_proj.bias.data.clone()
        
        # Create input
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        # Create causal mask for naive (boolean: True = mask out)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        
        # Forward pass
        with torch.no_grad():
            naive_out, _ = naive(x, attn_mask=causal_mask)
            triton_out, _ = triton_model(x)
        
        # Check outputs match
        max_diff = (naive_out - triton_out).abs().max().item()
        assert max_diff < 1e-3, f"Max diff {max_diff} exceeds tolerance"
    
    def test_triton_various_sequence_lengths(self, device):
        """Test Triton kernel with different sequence lengths."""
        if device != "cuda":
            pytest.skip("Triton requires CUDA")
        
        batch, embed_dim, num_heads = 2, 256, 4
        
        for seq_len in [64, 128, 256, 512]:
            model = TritonFlashAttention(embed_dim, num_heads).to(device).eval()
            x = torch.randn(batch, seq_len, embed_dim, device=device)
            
            with torch.no_grad():
                out, _ = model(x)
            
            assert out.shape == (batch, seq_len, embed_dim)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()
    
    def test_triton_output_shape(self, device):
        """Test Triton kernel produces correct output shape."""
        if device != "cuda":
            pytest.skip("Triton requires CUDA")
        
        batch, seq_len, embed_dim, num_heads = 4, 128, 512, 8
        
        model = TritonFlashAttention(embed_dim, num_heads).to(device).eval()
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        with torch.no_grad():
            out, attn_weights = model(x)
        
        assert out.shape == (batch, seq_len, embed_dim)
        assert attn_weights is None  # Memory efficient
    
    def test_triton_memory_usage_report(self, device):
        """Test memory usage calculation."""
        if device != "cuda":
            pytest.skip("Triton requires CUDA")
        
        model = TritonFlashAttention(512, 8).to(device)
        
        mem_info = model.get_memory_usage(batch_size=2, seq_len=1024)
        
        assert "qkv_mb" in mem_info
        assert "total_mb" in mem_info
        assert "complexity" in mem_info
        assert "kernel" in mem_info
        assert mem_info["kernel"] == "Triton"
        assert "comparison_with_naive" in mem_info


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
