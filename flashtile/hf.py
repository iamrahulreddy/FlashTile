"""Hugging Face transformers integration.

    import flashtile.hf
    flashtile.hf.register()
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation=flashtile.hf.NAME)

transformers passes q after RoPE and the un-repeated k/v from the cache, so GQA models run
without copying K/V. No padding -> plain causal on the triton path. Padding masks go to the
tiled path (exact, but slow). `stats` counts which path each call took.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import torch

from flashtile.api import attention

# transformers treats any name containing "flash" as a FlashAttention kernel and tries to load it
# from the hub, hence "ftile"
NAME = "ftile"
stats: Counter = Counter()  # calls per path: "causal", "decode", "full", "masked"

def flashtile_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
):
    if dropout and module.training:
        raise NotImplementedError("FlashTile does not implement attention dropout")
    lq, lk = query.shape[2], key.shape[2]
    if sliding_window is not None and lk > sliding_window:
        raise NotImplementedError("FlashTile does not implement sliding-window attention")

    if attention_mask is None:
        causal = getattr(module, "is_causal", True) if is_causal is None else is_causal
        if causal and lq == 1:
            causal = False  # a single last query sees every cached key anyway
        if causal:
            # queries are the last lq positions; CPU tensors so there's no GPU sync
            q_pos = torch.arange(lk - lq, lk) if lq != lk else None
            k_pos = torch.arange(lk) if lq != lk else None
            out = attention(query, key, value, scale=scaling, causal=True, q_pos=q_pos, k_pos=k_pos)
            stats["causal"] += 1
        else:
            out = attention(query, key, value, scale=scaling)
            stats["decode" if lq == 1 else "full"] += 1
    else:
        mask = attention_mask[:, :, :, :lk]
        if mask.dtype != torch.bool:  # additive mask: 0 means attend
            mask = mask == 0
        out = attention(query, key, value, scale=scaling, allow_mask=mask)
        stats["masked"] += 1
    return out.transpose(1, 2).contiguous(), None

def register() -> None:
    """Register FlashTile with transformers under NAME."""
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

    AttentionInterface.register(NAME, flashtile_attention_forward)
    AttentionMaskInterface.register(NAME, sdpa_mask)
