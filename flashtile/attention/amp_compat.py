"""
AMP (Automatic Mixed Precision) compatibility utilities.

Handles the deprecation of torch.cuda.amp.custom_fwd/custom_bwd
in favor of torch.amp.custom_fwd/custom_bwd (PyTorch 2.4+).

This module provides a single source of truth for the custom_fwd and
custom_bwd decorators, avoiding duplication across attention modules.
"""

import functools
import torch

try:
    # PyTorch 2.4+ uses torch.amp with explicit device_type
    from torch.amp import custom_fwd as _custom_fwd, custom_bwd as _custom_bwd

    @functools.wraps(_custom_fwd)
    def custom_fwd(func=None, *, device_type="cuda"):
        """Wrap torch.amp.custom_fwd with device_type default."""
        return _custom_fwd(func, device_type=device_type) if func else _custom_fwd(device_type=device_type)

    @functools.wraps(_custom_bwd)
    def custom_bwd(func=None, *, device_type="cuda"):
        """Wrap torch.amp.custom_bwd with device_type default."""
        return _custom_bwd(func, device_type=device_type) if func else _custom_bwd(device_type=device_type)

except (ImportError, AttributeError):
    # Fallback for PyTorch < 2.4
    try:
        from torch.cuda.amp import custom_fwd, custom_bwd  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        # Neither API available — provide no-op decorators
        def custom_fwd(func=None, **kwargs):
            """No-op fallback when AMP decorators are unavailable."""
            return func if func else lambda f: f

        def custom_bwd(func=None, **kwargs):
            """No-op fallback when AMP decorators are unavailable."""
            return func if func else lambda f: f
