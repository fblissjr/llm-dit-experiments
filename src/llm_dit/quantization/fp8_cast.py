"""FP8-cast quantization for LTX-2.3 transformer.

Last Updated: 2026-03-06

Aligned with the official Lightricks approach (ltx-core/quantization/fp8_cast.py):
stores weights as float8_e4m3fn, patches nn.Linear.forward to upcast to the
input dtype per-forward. Original fp8 weights are never mutated.

Memory footprint: ~12GB for 22B model (vs ~26GB bf16, ~42GB dequant+requant).

Usage:
    # Patch all nn.Linear layers in a model with fp8 weights
    count = amend_forward_with_upcast(model)
"""

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _replace_fwd_with_upcast(layer: nn.Linear) -> None:
    """Replace linear.forward with a version that upcasts fp8 weight per-forward.

    The original forward is stashed as `layer.original_forward` for introspection.
    Weight data is NOT mutated -- upcast creates a temporary bf16 copy.
    """
    layer.original_forward = layer.forward  # type: ignore[attr-defined]

    def new_forward(*args, **_kwargs) -> torch.Tensor:
        x = args[0]
        w = layer.weight.to(x.dtype)
        b = layer.bias.to(x.dtype) if layer.bias is not None else None
        return torch.nn.functional.linear(x, w, b)

    layer.forward = new_forward  # type: ignore[assignment]


def amend_forward_with_upcast(
    model: nn.Module,
    skip_patterns: tuple[str, ...] = ("norm", "embed", "lm_head"),
) -> int:
    """Patch all nn.Linear layers in model to upcast fp8 weights per-forward.

    Norms and embeddings are skipped by default -- they're numerically sensitive
    and tiny compared to linear layers.

    Args:
        model: Model with fp8 (float8_e4m3fn) weights on its nn.Linear layers.
        skip_patterns: Module name substrings to skip (case-insensitive).

    Returns:
        Number of linear layers patched.
    """
    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        name_lower = name.lower()
        if any(p in name_lower for p in skip_patterns):
            continue
        _replace_fwd_with_upcast(module)
        count += 1

    logger.info(f"fp8-cast: patched {count} nn.Linear layers for per-forward upcast")
    return count
