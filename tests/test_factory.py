"""
Tests for Factory Function
==========================

Tests verifying get_attention() factory function works correctly.

Run with: pytest tests/test_factory.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch

from flashtile import (
    get_attention,
    NaiveAttention,
    FlashAttentionV1,
    FlashAttentionV2,
    GroupedQueryAttention,
    MultiQueryAttention,
    SlidingWindowAttention,
    MaskedAttention,
    TritonFlashAttention,
    HAS_TRITON,
)


class TestFactoryFunction:
    """Tests for get_attention() factory function."""
    
    @pytest.mark.parametrize("attention_type,expected_class", [
        ("naive", NaiveAttention),
        ("flash_v1", FlashAttentionV1),
        ("flash_v2", FlashAttentionV2),
        ("sliding_window", SlidingWindowAttention),
        ("gqa", GroupedQueryAttention),
        ("grouped_query", GroupedQueryAttention),
        ("mqa", MultiQueryAttention),
        ("multi_query", MultiQueryAttention),
        ("masked", MaskedAttention),
    ])
    def test_factory_creates_correct_type(self, attention_type, expected_class):
        """Factory should create correct attention type."""
        model = get_attention(attention_type, embed_dim=256, num_heads=4)
        assert isinstance(model, expected_class)
    
    @pytest.mark.parametrize("attention_type", [
        "Naive",
        "FLASH_V1",
        "Flash-V2",
        "GQA",
        "gqa",
    ])
    def test_factory_case_insensitive(self, attention_type):
        """Factory should be case-insensitive and handle variations."""
        model = get_attention(attention_type, embed_dim=256, num_heads=4)
        assert model is not None
    
    def test_factory_unknown_type_raises_error(self):
        """Factory should raise error for unknown attention type."""
        with pytest.raises(ValueError, match="Unknown attention type"):
            get_attention("unknown_type", embed_dim=256, num_heads=4)
    
    def test_factory_forwards_kwargs(self):
        """Factory should forward kwargs to constructor."""
        # Test causal parameter
        model = get_attention("flash_v2", embed_dim=256, num_heads=4, causal=True)
        assert model.causal is True
        
        # Test block_size parameter
        model = get_attention("flash_v1", embed_dim=256, num_heads=4, block_size=32)
        assert model.block_size == 32
        
        # Test GQA num_kv_heads
        model = get_attention("gqa", embed_dim=256, num_heads=8, num_kv_heads=2)
        assert model.num_kv_heads == 2

        # Test masked attention causal flag
        model = get_attention("masked", embed_dim=256, num_heads=4, causal=True)
        assert model.causal is True
    
    def test_factory_outputs_work_correctly(self):
        """Models created by factory should work correctly."""
        batch, seq_len, embed_dim, num_heads = 2, 64, 256, 4
        
        for attn_type in ["naive", "flash_v1", "flash_v2"]:
            model = get_attention(attn_type, embed_dim=embed_dim, num_heads=num_heads)
            x = torch.randn(batch, seq_len, embed_dim)
            
            out, attn_weights = model(x)
            
            assert out.shape == (batch, seq_len, embed_dim)
            assert not torch.isnan(out).any()
    
    @pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
    def test_factory_triton(self):
        """Factory should create TritonFlashAttention when available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = get_attention("triton", embed_dim=256, num_heads=4)
        assert isinstance(model, TritonFlashAttention)
    
    def test_factory_all_types_in_error_message(self):
        """Error message should list all available types."""
        with pytest.raises(ValueError) as exc_info:
            get_attention("invalid", embed_dim=256, num_heads=4)
        
        error_msg = str(exc_info.value)
        # Check that common types are mentioned
        assert "naive" in error_msg
        assert "flash_v1" in error_msg
        assert "flash_v2" in error_msg


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
