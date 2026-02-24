"""
Meta-device initialization for zero-memory model construction.

last updated: 2026-02-24

Context manager that intercepts torch module construction to use meta tensors
(zero-memory placeholders) instead of real CUDA/CPU tensors. Combined with
`load_state_dict(assign=True)`, this eliminates the 2x memory spike that
occurs when constructing a model before loading weights.

Without meta init:
    1. Construct model -> allocate N bytes of random parameters
    2. Load state_dict -> allocate N bytes of loaded weights
    3. copy weights into model -> peak = 2N bytes
    4. Free state_dict -> back to N bytes

With meta init:
    1. Construct model on meta device -> allocate 0 bytes
    2. Load state_dict -> allocate N bytes
    3. assign=True replaces meta params with real ones -> peak = N bytes

Source: DiffSynth-Studio `diffsynth/core/vram/initialization.py`

Usage:
    with meta_init():
        model = LTX2Transformer(...)
    model.load_state_dict(state_dict, assign=True)
"""

from contextlib import contextmanager

import torch


@contextmanager
def meta_init():
    """Context manager for meta-device model construction.

    All nn.Module parameters created inside this context will be allocated
    on the meta device (zero memory). After construction, load real weights
    with `load_state_dict(state_dict, assign=True)`.

    IMPORTANT: `assign=True` is required when loading state dict after meta
    init. Without it, PyTorch tries to copy tensors into meta placeholders,
    which fails.
    """
    original_register_parameter = torch.nn.Module.register_parameter

    def meta_register_parameter(self, name, param):
        if param is not None:
            param = torch.nn.Parameter(
                torch.empty_like(param, device="meta"),
                requires_grad=param.requires_grad,
            )
        original_register_parameter(self, name, param)

    try:
        torch.nn.Module.register_parameter = meta_register_parameter
        yield
    finally:
        torch.nn.Module.register_parameter = original_register_parameter
