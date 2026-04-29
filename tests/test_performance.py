import pytest
import torch
import time
from typing import Dict, List, Tuple
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from flashtile import NaiveAttention, FlashAttentionV1, FlashAttentionV2

# Benchmark Utilities
def benchmark_forward(
    model: torch.nn.Module,
    x: torch.Tensor,
    warmup: int = 3,
    runs: int = 10
) -> Dict[str, float]:
    
    device = x.device
    
    for _ in range(warmup):
        with torch.inference_mode():
            _ = model(x)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Timed runs
    start = time.perf_counter()
    for _ in range(runs):
        with torch.inference_mode():
            _ = model(x)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = (time.perf_counter() - start) / runs * 1000  # ms
    
    memory_mb = 0
    if device.type == "cuda":
        memory_mb = torch.cuda.max_memory_allocated() / 1e6
    
    return {"time_ms": elapsed, "memory_mb": memory_mb}

class TestPerformanceScaling:
    """Tests for performance scaling characteristics."""
    
    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for memory tests")
    def test_memory_scaling(self, device):
        """Verify Flash memory grows slower than naive."""
        embed_dim, num_heads = 512, 8
        batch = 2
        
        seq_lengths = [256, 512, 1024]
        naive_memories = []
        flash_memories = []
        
        for seq_len in seq_lengths:
            x = torch.randn(batch, seq_len, embed_dim, device=device)
            
            # Naive memory
            naive = NaiveAttention(embed_dim, num_heads).to(device).eval()
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                _ = naive(x)
            naive_memories.append(torch.cuda.max_memory_allocated() / 1e6)
            del naive
            torch.cuda.empty_cache()
            
            # Flash memory
            flash = FlashAttentionV1(embed_dim, num_heads, block_size=64).to(device).eval()
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                _ = flash(x)
            flash_memories.append(torch.cuda.max_memory_allocated() / 1e6)
            del flash
            torch.cuda.empty_cache()
        
        # Calculate scaling ratios
        # Naive should scale quadratically: 4x sequence -> 16x memory
        # Flash should scale linearly: 4x sequence -> 4x memory
        
        naive_ratio = naive_memories[-1] / naive_memories[0]
        flash_ratio = flash_memories[-1] / flash_memories[0]
        seq_ratio = seq_lengths[-1] / seq_lengths[0]
        
        print(f"\nMemory Scaling Analysis:")
        print(f"  Sequence ratio: {seq_ratio}x")
        print(f"  Naive memory ratio: {naive_ratio:.1f}x (expected ~{seq_ratio**2}x for O(N^2))")
        print(f"  Flash memory ratio: {flash_ratio:.1f}x (expected ~{seq_ratio}x for O(N))")
        
        # Flash should scale better than quadratic
        assert flash_ratio < naive_ratio, \
            f"Flash ({flash_ratio:.1f}x) should scale better than naive ({naive_ratio:.1f}x)"
    
    @pytest.mark.parametrize("seq_len", [512, 1024, 2048])
    def test_flash_timing_characteristics(self, seq_len, device):
        """Log Flash vs Naive timing characteristics at various sequence lengths."""
        if device == "cpu" and seq_len > 1024:
            pytest.skip("Skip very long sequences on CPU")
        
        embed_dim, num_heads = 512, 8
        batch = 2
        
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        naive = NaiveAttention(embed_dim, num_heads).to(device).eval()
        flash = FlashAttentionV1(embed_dim, num_heads, block_size=64).to(device).eval()
        
        naive_bench = benchmark_forward(naive, x, warmup=2, runs=5)
        flash_bench = benchmark_forward(flash, x, warmup=2, runs=5)
        
        print(f"\nSeq {seq_len}: Naive={naive_bench['time_ms']:.2f}ms, Flash={flash_bench['time_ms']:.2f}ms")
        
        # Python-loop implementation is inherently slower than PyTorch's fused
        # CUDA attention. The purpose of this test is to log the ratio for reference.
        # Memory efficiency (tested in test_memory_scaling) is the real metric.
        ratio = flash_bench['time_ms'] / max(naive_bench['time_ms'], 1e-6)
        print(f"  Ratio (flash/naive): {ratio:.1f}x — expected for Python loops")


class TestTheoreticalMemory:
    """Tests for theoretical memory calculations."""
    
    def test_memory_calculation_consistency(self):
        """Verify memory calculation methods are consistent."""
        embed_dim, num_heads = 512, 8
        batch, seq_len = 2, 1024
        
        naive = NaiveAttention(embed_dim, num_heads)
        flash = FlashAttentionV1(embed_dim, num_heads, block_size=64)
        
        naive_mem = naive.get_memory_usage(batch, seq_len)
        flash_mem = flash.get_memory_usage(batch, seq_len)
        
        # Flash should report less memory
        assert flash_mem['total_gb'] < naive_mem['total_gb'], \
            "Flash should report less theoretical memory"
        
        # Print breakdown
        print(f"\nTheoretical Memory @ seq_len={seq_len}:")
        print(f"  Naive: {naive_mem['total_gb']*1000:.2f} MB")
        print(f"  Flash: {flash_mem['total_gb']*1000:.2f} MB")
        print(f"  Ratio: {naive_mem['total_gb']/flash_mem['total_gb']:.1f}x")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
