from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from flashtile import (
    NaiveAttention,
    FlashAttentionV1,
    FlashAttentionV2,
    GroupedQueryAttention,
)

@pytest.fixture
def device():
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def small_config():
    """Small model config for fast testing."""
    return {
        "embed_dim": 128,
        "num_heads": 4,
        "seq_len": 64,
        "batch_size": 4,
    }

class TestTrainingConvergence:
    """Test that models actually train and converge."""
    
    def _create_synthetic_task(self, config, device):
        """Create a simple synthetic task: copy input to output."""
        num_samples = 100
        
        X = torch.randn(num_samples, config["seq_len"], config["embed_dim"], device=device)
        y = X.clone()  # Target = input (identity mapping)
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
        
        return dataloader
    
    def _train_epoch(self, model, dataloader, optimizer, criterion, device):
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            output, _ = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def test_flash_v1_converges(self, device, small_config):
        """Test that Flash V1 trains and loss decreases."""
        model = FlashAttentionV1(
            small_config["embed_dim"],
            small_config["num_heads"],
        ).to(device)
        
        dataloader = self._create_synthetic_task(small_config, device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # Train for a few epochs
        losses = []
        for epoch in range(5):
            loss = self._train_epoch(model, dataloader, optimizer, criterion, device)
            losses.append(loss)
        
        # Verify loss decreased
        assert losses[-1] < losses[0], f"Loss should decrease: {losses}"
        print(f"Flash V1 losses: {losses}")
    
    def test_flash_v2_converges(self, device, small_config):
        """Test that Flash V2 trains and loss decreases."""
        model = FlashAttentionV2(
            small_config["embed_dim"],
            small_config["num_heads"],
            causal=True,
        ).to(device)
        
        dataloader = self._create_synthetic_task(small_config, device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # Train for a few epochs
        losses = []
        for epoch in range(5):
            loss = self._train_epoch(model, dataloader, optimizer, criterion, device)
            losses.append(loss)
        
        # Verify loss decreased
        assert losses[-1] < losses[0], f"Loss should decrease: {losses}"
        print(f"Flash V2 losses: {losses}")
    
    def test_gqa_converges(self, device, small_config):
        """Test that GQA trains and loss decreases."""
        model = GroupedQueryAttention(
            small_config["embed_dim"],
            small_config["num_heads"],
            num_kv_heads=2,
            causal=True,
        ).to(device)
        
        dataloader = self._create_synthetic_task(small_config, device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # Train for a few epochs
        losses = []
        for epoch in range(5):
            loss = self._train_epoch(model, dataloader, optimizer, criterion, device)
            losses.append(loss)
        
        # Verify loss decreased
        assert losses[-1] < losses[0], f"Loss should decrease: {losses}"
        print(f"GQA losses: {losses}")
    
    def test_flash_matches_naive_training(self, device, small_config):
        """Test that Flash and Naive produce similar losses and gradients."""
        # Create identical models
        torch.manual_seed(42)
        naive_model = NaiveAttention(
            small_config["embed_dim"],
            small_config["num_heads"],
        ).to(device)
        
        torch.manual_seed(42)
        flash_model = FlashAttentionV1(
            small_config["embed_dim"],
            small_config["num_heads"],
        ).to(device)
        
        # Same input
        x = torch.randn(2, small_config["seq_len"], small_config["embed_dim"], device=device)
        x_naive = x.clone().detach().requires_grad_(True)
        x_flash = x.clone().detach().requires_grad_(True)
        
        # Forward + backward
        naive_out, _ = naive_model(x_naive)
        flash_out, _ = flash_model(x_flash)
        
        loss_naive = naive_out.mean()
        loss_flash = flash_out.mean()
        
        loss_naive.backward()
        loss_flash.backward()
        
        torch.testing.assert_close(loss_flash.detach(), loss_naive.detach(), rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(x_flash.grad, x_naive.grad, rtol=1e-3, atol=1e-3)
        
        for (naive_name, naive_param), (flash_name, flash_param) in zip(
            naive_model.named_parameters(), flash_model.named_parameters()
        ):
            assert naive_name == flash_name
            assert naive_param.grad is not None, f"{naive_name} is missing gradients"
            assert flash_param.grad is not None, f"{flash_name} is missing gradients"

            assert not torch.isnan(naive_param.grad).any(), f"{naive_name} has NaN gradients"
            assert not torch.isnan(flash_param.grad).any(), f"{flash_name} has NaN gradients"
            assert not torch.isinf(naive_param.grad).any(), f"{naive_name} has Inf gradients"
            assert not torch.isinf(flash_param.grad).any(), f"{flash_name} has Inf gradients"

            torch.testing.assert_close(
                flash_param.grad,
                naive_param.grad,
                rtol=1e-2,
                atol=1e-3,
            )

class SimpleTransformerLayer(nn.Module):
    """Simple transformer layer for testing."""
    
    def __init__(self, embed_dim, num_heads, attention_class, **attn_kwargs):
        super().__init__()
        self.attention = attention_class(embed_dim, num_heads, **attn_kwargs)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # Attention with residual
        attn_out, _ = self.attention(self.norm1(x))
        x = x + attn_out
        
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        
        return x


class TestMultiLayerModels:
    """Test Flash Attention in multi-layer architectures."""
    
    def test_two_layer_model_with_flash_v2(self, device, small_config):
        """Test a 2-layer transformer with Flash V2."""
        model = nn.Sequential(
            SimpleTransformerLayer(
                small_config["embed_dim"],
                small_config["num_heads"],
                FlashAttentionV2,
                causal=True,
            ),
            SimpleTransformerLayer(
                small_config["embed_dim"],
                small_config["num_heads"],
                FlashAttentionV2,
                causal=True,
            ),
        ).to(device)
        
        x = torch.randn(2, small_config["seq_len"], small_config["embed_dim"], device=device)
        
        # Forward pass
        output = model(x)
        assert output.shape == x.shape
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any()
    
    def test_mixed_attention_model(self, device, small_config):
        """Test model with different attention types in different layers."""
        class MixedModel(nn.Module):
            def __init__(self, embed_dim, num_heads):
                super().__init__()
                self.layer1 = FlashAttentionV2(embed_dim, num_heads, causal=True)
                self.layer2 = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads=2, causal=True)
            
            def forward(self, x):
                x, _ = self.layer1(x)
                x, _ = self.layer2(x)
                return x
        
        model = MixedModel(small_config["embed_dim"], small_config["num_heads"]).to(device)
        
        x = torch.randn(2, small_config["seq_len"], small_config["embed_dim"], device=device)
        
        # Forward
        output = model(x)
        assert output.shape == x.shape
        
        # Backward
        loss = output.sum()
        loss.backward()
        
        print("✓ Mixed attention model works!")

class TestEdgeCases:
    """Test edge cases and robustness."""
    
    def test_gradient_accumulation(self, device, small_config):
        """Test gradient accumulation (multiple forwards before backward)."""
        model = FlashAttentionV2(
            small_config["embed_dim"],
            small_config["num_heads"],
        ).to(device)
        
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        # Accumulate gradients over 3 batches
        accumulation_steps = 3
        for i in range(accumulation_steps):
            x = torch.randn(1, small_config["seq_len"], small_config["embed_dim"], device=device)
            output, _ = model(x)
            loss = output.sum() / accumulation_steps
            loss.backward()
        
        # Check gradients accumulated
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "Gradients should accumulate"
    
    def test_mixed_precision_context(self, device, small_config):
        """Test that Flash Attention works in autocast context."""
        if device != "cuda":
            pytest.skip("AMP only relevant for CUDA")
        
        model = FlashAttentionV2(
            small_config["embed_dim"],
            small_config["num_heads"],
        ).to(device)
        
        x = torch.randn(2, small_config["seq_len"], small_config["embed_dim"], device=device)
        
        # Forward in autocast
        with torch.autocast(device_type='cuda'):
            output, _ = model(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
    def test_memory_efficiency_at_long_sequences(self, device, small_config):
        """Verify Flash Attention uses less memory than naive at long sequences."""
        long_seq_len = 2048  # Long sequence
        
        x = torch.randn(1, long_seq_len, small_config["embed_dim"], device=device)
        
        # Measure Flash memory
        torch.cuda.reset_peak_memory_stats()
        flash_model = FlashAttentionV2(
            small_config["embed_dim"],
            small_config["num_heads"],
        ).to(device)
        
        with torch.no_grad():
            _ = flash_model(x)
        
        flash_memory = torch.cuda.max_memory_allocated()
        
        # Measure Naive memory
        torch.cuda.reset_peak_memory_stats()
        naive_model = NaiveAttention(
            small_config["embed_dim"],
            small_config["num_heads"],
        ).to(device)
        
        with torch.no_grad():
            _ = naive_model(x)
        
        naive_memory = torch.cuda.max_memory_allocated()
        
        # Flash should use less memory
        assert flash_memory < naive_memory, \
            f"Flash should use less memory: Flash={flash_memory/1e6:.1f}MB, Naive={naive_memory/1e6:.1f}MB"
        
        reduction = naive_memory / flash_memory
        print(f"Memory reduction: {reduction:.1f}x")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
