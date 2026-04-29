import pytest
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from flashtile import NaiveAttention, FlashAttentionV1, FlashAttentionV2

@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture
def seed():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    return 42

class TestFlashV2Correctness:
    """V2 should produce identical output to V1 and Naive."""
    
    def test_v2_matches_v1(self, device, seed):
        """V2 output should match V1 within tolerance."""
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
        assert max_diff < 1e-3, f"V2 differs from V1: max_diff={max_diff}"
    
    def test_v2_matches_naive(self, device, seed):
        """V2 output should match naive attention within tolerance."""
        batch, seq_len, embed_dim, num_heads = 2, 64, 256, 4
        
        naive = NaiveAttention(embed_dim, num_heads).to(device).eval()
        v2 = FlashAttentionV2(embed_dim, num_heads).to(device).eval()
        
        # Sync weights
        v2.qkv_proj.weight.data = naive.qkv_proj.weight.data.clone()
        v2.qkv_proj.bias.data = naive.qkv_proj.bias.data.clone()
        v2.out_proj.weight.data = naive.out_proj.weight.data.clone()
        v2.out_proj.bias.data = naive.out_proj.bias.data.clone()
        
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        with torch.inference_mode():
            naive_out, _ = naive(x)
            v2_out, _ = v2(x)
        
        max_diff = (naive_out - v2_out).abs().max().item()
        assert max_diff < 1e-3, f"V2 differs from naive: max_diff={max_diff}"


class TestCausalMasking:
    """Tests for causal (autoregressive) masking in V2."""
    
    def test_causal_output_is_causal(self, device, seed):
        """Verify causal masking: position i should only depend on positions j <= i."""
        batch, seq_len, embed_dim, num_heads = 1, 64, 256, 4
        
        model = FlashAttentionV2(embed_dim, num_heads, causal=True).to(device).eval()
        
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        with torch.inference_mode():
            out1, _ = model(x)
            
            x_modified = x.clone()
            x_modified[0, 32, :] = torch.randn(embed_dim, device=device)
            out2, _ = model(x_modified)
        
        max_diff_before = (out1[0, :32, :] - out2[0, :32, :]).abs().max().item()
        assert max_diff_before < 1e-5, f"Causal violation: earlier positions changed: {max_diff_before}"
        
        max_diff_after = (out1[0, 32:, :] - out2[0, 32:, :]).abs().max().item()
        assert max_diff_after > 0.01, f"Causal masking not working: later positions unchanged"
    
    def test_causal_matches_naive_with_mask(self, device, seed):
        """V2 causal should match naive with explicit causal mask."""
        batch, seq_len, embed_dim, num_heads = 2, 64, 256, 4
        
        naive = NaiveAttention(embed_dim, num_heads).to(device).eval()
        v2_causal = FlashAttentionV2(embed_dim, num_heads, causal=True).to(device).eval()
        
        # Sync weights
        v2_causal.qkv_proj.weight.data = naive.qkv_proj.weight.data.clone()
        v2_causal.qkv_proj.bias.data = naive.qkv_proj.bias.data.clone()
        v2_causal.out_proj.weight.data = naive.out_proj.weight.data.clone()
        v2_causal.out_proj.bias.data = naive.out_proj.bias.data.clone()
        
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        
        with torch.inference_mode():
            naive_out, _ = naive(x, attn_mask=causal_mask)
            v2_out, _ = v2_causal(x)
        
        max_diff = (naive_out - v2_out).abs().max().item()
        assert max_diff < 1e-3, f"V2 causal differs from naive+mask: max_diff={max_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
