# FlashTile Usage Guide

Comprehensive API reference and usage examples.

## Installation

```bash
# Clone and install
git clone https://github.com/iamrahulreddy/FlashTile.git
cd FlashTile
pip install -e .

# With all optional dependencies
pip install -e ".[all]"

# Verify installation
python -c "from flashtile import FlashAttentionV2; print('OK')"
```

## Quick Start

```python
import torch
from flashtile import FlashAttentionV2

# Create model with causal masking (for decoder models like GPT)
model = FlashAttentionV2(embed_dim=512, num_heads=8, causal=True)

# Input: (batch, sequence_length, embedding_dimension)
x = torch.randn(2, 2048, 512)

# Forward pass
output, _ = model(x)
print(output.shape)  # torch.Size([2, 2048, 512])
```

## API Reference

### FlashAttentionV1

Memory-efficient attention with $O(N)$ memory complexity and a
memory-efficient backward pass.

```python
FlashAttentionV1(
    embed_dim: int,           # Embedding dimension (must be divisible by num_heads)
    num_heads: int = 8,       # Number of attention heads
    block_size: int = 64,     # Tile size for blocking (power of 2 recommended)
    dropout: float = 0.0,     # Dropout probability (applied to output, not attention weights)
    bias: bool = True,        # Use bias in projections
    causal: bool = False,     # Enable causal masking
)
```

**Features:**
- $O(N)$ memory forward pass
- Memory-efficient backward pass with recomputation ($O(N)$ memory)
- Causal masking support
- torch.compile compatible

**Tradeoff**: Memory-efficient but slower than fused kernels due to Python overhead.

**Methods:**

```python
# Forward pass
output, _ = model(
    x,                 # Input (batch, seq_len, embed_dim)
    attn_mask=None,    # Not supported (use causal=True instead)
)
# Note: Second return value is always None for Flash implementations

# Memory estimation
memory_dict = model.get_memory_usage(batch_size=2, seq_len=1024)
```

### FlashAttentionV2

Flash Attention V2 with swapped loop order and causal block-skipping optimization.

```python
FlashAttentionV2(
    embed_dim: int,           # Embedding dimension
    num_heads: int = 8,       # Number of attention heads
    block_size: int = 64,     # Tile size for blocking
    dropout: float = 0.0,     # Dropout probability (applied to output)
    bias: bool = True,        # Use bias in projections
    causal: bool = False,     # Enable causal masking (~50% compute savings)
)
```

**Key Feature:** When `causal=True`, V2 skips ~50% of block computations for decoder models.

```python
from flashtile import FlashAttentionV2

# Causal attention for decoder models (GPT-style)
model = FlashAttentionV2(embed_dim=512, num_heads=8, causal=True)
x = torch.randn(2, 1024, 512)
output, _ = model(x)

# Check efficiency stats
efficiency = model.get_causal_efficiency(seq_len=1024)
print(f"Compute savings: {efficiency['compute_savings_percent']:.1f}%")
```

### GroupedQueryAttention

GQA reduces KV-cache memory by sharing K/V heads across multiple Q heads.

```python
GroupedQueryAttention(
    embed_dim: int,           # Embedding dimension
    num_heads: int,           # Number of query heads
    num_kv_heads: int,        # Number of key-value heads (must divide num_heads)
    block_size: int = 64,     # Tile size for blocking
    dropout: float = 0.0,     # Dropout probability
    bias: bool = True,        # Use bias in projections
    causal: bool = False,     # Enable causal masking
)
```

**Memory Savings (Inference):**
- MHA ($`\text{num\_kv\_heads} = \text{num\_heads}`$): baseline
- GQA ($`\text{num\_kv\_heads} < \text{num\_heads}`$): savings of
  $`\frac{\text{num\_heads}}{\text{num\_kv\_heads}}`$
- MQA ($`\text{num\_kv\_heads} = 1`$): maximum savings

**Training Note**: GQA uses standard PyTorch autograd. The forward pass is
memory-efficient, but the backward pass stores intermediate activations
($O(N^2)$ memory). For memory-efficient training, use `FlashAttentionV1/V2`.

```python
from flashtile import GroupedQueryAttention

# GQA with 8 query heads, 2 KV heads (4x KV cache savings)
model = GroupedQueryAttention(
    embed_dim=512, num_heads=8, num_kv_heads=2, causal=True
)

x = torch.randn(2, 1024, 512)
output, _ = model(x)

# Check memory savings
mem = model.get_memory_usage(batch_size=2, seq_len=1024)
print(f"KV savings: {mem['kv_savings_ratio']:.1f}x vs MHA")
```

### MultiQueryAttention

MQA is a special case of GQA where all query heads share a single KV head.

```python
MultiQueryAttention(
    embed_dim: int,           # Embedding dimension
    num_heads: int,           # Number of query heads
    block_size: int = 64,     # Tile size for blocking
    dropout: float = 0.0,     # Dropout probability
    bias: bool = True,        # Use bias in projections
    causal: bool = False,     # Enable causal masking
)
```

```python
from flashtile import MultiQueryAttention

# All 8 heads share 1 KV head (8x KV cache savings)
mqa = MultiQueryAttention(embed_dim=512, num_heads=8, causal=True)
```

### NaiveAttention

Standard $O(N^2)$ attention for baseline comparison and visualization.

```python
NaiveAttention(
    embed_dim: int,           # Embedding dimension
    num_heads: int = 8,       # Number of attention heads
    dropout: float = 0.0,     # Dropout probability
    bias: bool = True,        # Use bias in projections
)
```

**Methods:**

```python
# Forward pass (can return attention weights)
output, attn_weights = model(
    x: Tensor,
    attn_mask: Optional[Tensor] = None,
)

# When attn_mask is provided, applies the mask
# attn_weights returned only when internal implementation stores them
```

**Note:** Only `NaiveAttention` returns attention weights. Flash implementations return `None` for the second value to maintain API consistency.

### SlidingWindowAttention

Local attention with $O(NW)$ complexity for very long sequences.

```python
from flashtile import SlidingWindowAttention

SlidingWindowAttention(
    embed_dim: int,           # Embedding dimension
    num_heads: int,           # Number of attention heads
    window_size: int = 512,   # Attention window size
    dropout: float = 0.0,     # Dropout probability
    bias: bool = True,        # Use bias in projections
    causal: bool = False,     # Enable causal masking
)
```

### MaskedAttention

Fallback implementation supporting arbitrary attention masks.

```python
from flashtile import MaskedAttention, create_padding_mask

MaskedAttention(
    embed_dim: int,           # Embedding dimension
    num_heads: int,           # Number of attention heads
    dropout: float = 0.0,     # Dropout probability
    bias: bool = True,        # Use bias in projections
    causal: bool = False,     # Apply causal mask in addition to attn_mask
    chunk_size: int = 1024,   # Query chunk size for memory-efficient processing
)

# Usage with padding mask
model = MaskedAttention(embed_dim=512, num_heads=8)
x = torch.randn(2, 128, 512)
lengths = torch.tensor([96, 128])
padding_mask = create_padding_mask(lengths, max_len=128, device=x.device)
output, _ = model(x, attn_mask=padding_mask)
```

### TritonFlashAttention (Forward-Only)

High-performance Triton kernel wrapped in an `nn.Module` with QKV projections.
**Cannot be used for training** - forward pass only.

```python
from flashtile import TritonFlashAttention, HAS_TRITON

TritonFlashAttention(
    embed_dim: int,           # Embedding dimension
    num_heads: int = 8,       # Number of attention heads
    dropout: float = 0.0,     # Dropout probability (not implemented in kernel)
    bias: bool = True,        # Use bias in projections
    causal: bool = False,     # Enable causal masking
)
```

> **Note for Windows Users**: Triton is primarily supported on Linux. On Windows, this module may not be available or require specific WSL configurations. FlashTile will gracefully handle missing Triton.

```python
from flashtile import TritonFlashAttention, HAS_TRITON

if HAS_TRITON:
    model = TritonFlashAttention(embed_dim=512, num_heads=8, causal=True).cuda().eval()
    x = torch.randn(2, 1024, 512).cuda()
    with torch.no_grad():
        output, _ = model(x)
else:
    print("Triton not available, use FlashAttentionV2 instead")
```

### triton_flash_attention (functional, Forward-Only)

Low-level functional interface for the Triton kernel. **Forward pass only**.

```python
from flashtile.kernels import triton_flash_attention, HAS_TRITON

# Input: (batch, num_heads, seq_len, head_dim)
q = torch.randn(2, 8, 1024, 64).cuda()
k = torch.randn(2, 8, 1024, 64).cuda()
v = torch.randn(2, 8, 1024, 64).cuda()

if HAS_TRITON:
    output = triton_flash_attention(q, k, v, causal=True)
```

**Note:** This is the raw kernel without projection layers. Returns tensor directly (not tuple).

### Factory Function

```python
from flashtile import get_attention

# Create attention by name
model = get_attention(
    attention_type="flash_v2",   # "naive", "flash_v1", "flash_v2", "gqa", "mqa", "triton", "masked"
    embed_dim=512,
    num_heads=8,
    causal=True,          # V1/V2/GQA/MQA
    num_kv_heads=2,       # GQA only
)
```

## Common Use Cases

### 1. Decoder Model (GPT-style)

```python
from flashtile import FlashAttentionV2

# Causal attention for autoregressive generation
model = FlashAttentionV2(embed_dim=512, num_heads=8, causal=True).cuda()
x = torch.randn(2, 2048, 512).cuda()
output, _ = model(x)
```

### 2. Encoder Model (BERT-style)

```python
from flashtile import FlashAttentionV1

# Bidirectional attention (no causal mask)
model = FlashAttentionV1(embed_dim=512, num_heads=8, causal=False).cuda()
x = torch.randn(2, 512, 512).cuda()
output, _ = model(x)
```

### 3. Inference Optimization with GQA

```python
from flashtile import GroupedQueryAttention

# LLaMA 2 style: 32 query heads, 8 KV heads
model = GroupedQueryAttention(
    embed_dim=4096, num_heads=32, num_kv_heads=8, causal=True
).cuda()

# 4x smaller KV cache during inference!
```

### 4. Training with Memory-Efficient Backward

```python
from flashtile import FlashAttentionV2

model = FlashAttentionV2(embed_dim=512, num_heads=8, causal=True).cuda()
x = torch.randn(2, 2048, 512, requires_grad=True).cuda()

# Forward pass: O(N) memory
output, _ = model(x)

# Backward pass: recomputes attention (O(N) memory)
loss = output.mean()
loss.backward()

# Gradients computed with minimal memory!
print(f"Input gradient shape: {x.grad.shape}")
```

### 5. Using torch.compile

```python
import torch
from flashtile import FlashAttentionV2

model = FlashAttentionV2(embed_dim=512, num_heads=8, causal=True).cuda()
model = torch.compile(model)  # Enable kernel fusion

x = torch.randn(2, 2048, 512).cuda()
output, _ = model(x)  # Compiled execution
```

### 6. Transformer Block Integration

```python
import torch
import torch.nn as nn
from flashtile import FlashAttentionV2

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, causal=True):
        super().__init__()
        self.attention = FlashAttentionV2(embed_dim, num_heads, causal=causal)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # Pre-norm architecture
        attn_out, _ = self.attention(self.norm1(x))
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

# Build a small transformer
blocks = nn.Sequential(*[
    TransformerBlock(512, 8, causal=True) for _ in range(6)
]).cuda()

x = torch.randn(2, 1024, 512).cuda()
output = blocks(x)
```

### 7. Custom Attention Mask

```python
from flashtile import MaskedAttention, create_causal_mask

# For custom masking needs
model = MaskedAttention(embed_dim=512, num_heads=8).cuda()
x = torch.randn(2, 128, 512).cuda()

# Create custom mask (True means masked)
mask = torch.zeros(128, 128, dtype=torch.bool, device=x.device)
mask[:, 50:] = True  # Don't attend to positions 50+

output, _ = model(x, attn_mask=mask)
```

## Configuration Guide

### Block Size Selection

| Block Size | Best For | Trade-off |
|------------|----------|-----------|
| 32 | Limited SRAM, older GPUs | More kernel launches |
| 64 | **Default**, good balance | Works on most GPUs |
| 128 | High-end GPUs (A100, H100) | Better efficiency, more SRAM |

### When to Use Each Implementation

| Scenario | Recommendation |
|----------|----------------|
| seq_len < 512 | Either works, Naive may be faster |
| seq_len 512-2048 | Flash V1/V2 recommended |
| seq_len > 2048 | Flash required (Naive OOMs) |
| Decoder models | Flash V2 with causal=True |
| Need attention weights | Must use Naive |
| Maximum performance | Triton kernel (forward only) |
| Inference optimization | GQA or MQA |
| Custom masks | MaskedAttention or Naive |

### GQA Configuration Guide

| Config | KV Heads | Memory Savings | Quality Impact |
|--------|----------|----------------|----------------|
| MHA | num_heads | 1x (baseline) | Best quality |
| GQA-4 | num_heads/4 | 4x | Minimal impact |
| GQA-8 | num_heads/8 | 8x | Small impact |
| MQA | 1 | num_heads x | Noticeable impact |

## Troubleshooting

### CUDA Out of Memory

```python
# Problem: OOM with naive attention
model = NaiveAttention(512, 8)  # OOMs at long sequences

# Solution: Switch to Flash
model = FlashAttentionV2(512, 8, causal=True)  # 8-15x less memory
```

### Numerical Differences

Small differences (<0.1%) between Naive and Flash are expected due to:
- Different computation order
- Floating-point precision
- Recomputation in backward pass

This is normal and production-safe.

### Triton Not Available

```python
from flashtile import HAS_TRITON, TritonFlashAttention

if HAS_TRITON:
    model = TritonFlashAttention(embed_dim=512, num_heads=8)
else:
    print("Triton not installed or not available on this platform")
    from flashtile import FlashAttentionV2
    model = FlashAttentionV2(512, 8, causal=True)
```

### Gradient Issues

If gradients seem incorrect:
1. Ensure you're using the same random seed for comparison
2. Use smaller sequences for debugging
3. Check that weights are properly synced between models
4. Verify you're comparing against `NaiveAttention`, not PyTorch's SDPA

## Performance Tips

1. **Use causal=True for decoder models** - Saves ~50% compute
2. **Use torch.compile** - Enables kernel fusion
3. **Use GQA for inference** - Reduces KV cache memory
4. **Use appropriate block_size** - 64 is good default, 128 for A100/H100
5. **Use FP16/BF16** - Faster and uses less memory

## Important Notes

### Dropout Location

Flash Attention V1/V2 applies dropout **after the output projection**, not to attention weights (P matrix) like standard attention. This is a limitation of the block-wise computation. For most use cases, this difference is negligible. If you need exact dropout behavior, use `NaiveAttention`.

### Triton Kernel Limitations

The Triton kernel is **forward-only**. For training with backward pass, use `FlashAttentionV1` or `FlashAttentionV2` which have custom autograd functions.

### Return Value Convention

All attention modules return a tuple `(output, attention_weights)`:
- `output`: The attention output tensor
- `attention_weights`: 
  - `NaiveAttention`: May return actual attention weights
  - `MaskedAttention`: Returns `None`
  - Flash implementations: Always return `None` (memory efficient)

This consistent API makes it easy to swap implementations.

*For algorithm details, see [ALGORITHM.md](ALGORITHM.md). For architecture overview, see [ARCHITECTURE.md](ARCHITECTURE.md).*
