# FlashTile: exact attention, from a dense oracle to tiled PyTorch to fused Triton kernels.
from flashtile.api import HAS_TRITON, attention, select_backend
from flashtile.reference import attention_reference
from flashtile.tiled import tiled_attention

__version__ = "2.0.0"

__all__ = ["HAS_TRITON", "__version__", "attention", "attention_reference", "select_backend", "tiled_attention"]
