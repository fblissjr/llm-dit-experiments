"""Native PyTorch fp8 layerwise casting for nn.Linear modules.

Stores weights in float8_e4m3fn (~50% memory savings vs bf16) and casts to
bfloat16 on-the-fly during forward pass via hooks. No torchao or diffusers
dependency.

The pre-forward hook upcasts weight to bf16 before matmul; the post-forward
hook downcasts it back to fp8 immediately after. Only one layer's weight is
in bf16 at any given time, so peak VRAM stays close to the fp8 footprint.

Usage:
    apply_fp8_layerwise_casting(model)  # converts in-place, installs hooks
    model(input)  # hooks handle bf16 compute transparently
"""

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _pre_forward_hook(module: nn.Linear, _args: tuple) -> None:
    """Cast fp8 weight to bf16 before matmul."""
    module.weight.data = module.weight.data.to(torch.bfloat16)


def _post_forward_hook(module: nn.Linear, _args: tuple, output: torch.Tensor) -> torch.Tensor:
    """Cast weight back to fp8 after matmul."""
    module.weight.data = module.weight.data.to(torch.float8_e4m3fn)
    return output


def apply_fp8_layerwise_casting(
    model: nn.Module,
    skip_patterns: tuple[str, ...] = ("norm", "embed", "lm_head"),
) -> int:
    """Apply fp8 layerwise casting to all nn.Linear modules in-place.

    Norms and embeddings are skipped by default -- they're numerically
    sensitive and tiny relative to the linear layers (keeping them in bf16
    costs negligible memory).

    Args:
        model: The model to convert.
        skip_patterns: Module name substrings to skip (case-insensitive).

    Returns:
        Number of modules converted.
    """
    converted = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(pat in name.lower() for pat in skip_patterns):
            continue
        # Cast weight to fp8 in-place (bias stays bf16 -- tiny, not worth fp8)
        module.weight.data = module.weight.data.to(torch.float8_e4m3fn)
        module.register_forward_pre_hook(_pre_forward_hook)
        module.register_forward_hook(_post_forward_hook)
        converted += 1
    return converted
