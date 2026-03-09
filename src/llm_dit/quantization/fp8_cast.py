"""FP8-cast quantization for LTX-2.3 transformer.

Last Updated: 2026-03-09

Supports two FP8 checkpoint formats:

1. **Scaled FP8** (official release, e.g. ltx-2.3-transformer-fp8.safetensors):
   Weights quantized with per-tensor scaling. FP8 values span the full [-448, 448]
   range. Each layer has a `weight_scale` factor. Dequantization:
     original_weight = fp8_weight * weight_scale
   Without applying scale, weights are ~1000x too large.

2. **Naive FP8** (from bf16 truncation):
   Weights simply cast from bf16 to fp8 with `.to(float8_e4m3fn)`. Values preserve
   original magnitude. No scale needed for upcast. This is what the official
   ltx-core fp8-cast path assumes (it starts from bf16 and downcasts at runtime).

The loader detects which format is in use by checking for weight_scale keys in
the state dict. When scales are present, they're attached to each nn.Linear as
`_weight_scale` and applied during the per-forward upcast.

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

    If the layer has a `_weight_scale` attribute (set by the loader for scaled FP8
    checkpoints), the upcast weight is multiplied by the scale to recover the
    original magnitude. Without scale, this is a simple dtype cast.

    The original forward is stashed as `layer.original_forward` for introspection.
    Weight data is NOT mutated -- upcast creates a temporary bf16 copy.
    """
    layer.original_forward = layer.forward  # type: ignore[attr-defined]

    def new_forward(*args, **_kwargs) -> torch.Tensor:
        x = args[0]
        w = layer.weight.to(x.dtype)
        # Apply per-tensor weight scale if present (scaled FP8 checkpoint)
        scale = getattr(layer, "_weight_scale", None)
        if scale is not None:
            w = w * scale.to(x.dtype)
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
    scaled = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        name_lower = name.lower()
        if any(p in name_lower for p in skip_patterns):
            continue
        _replace_fwd_with_upcast(module)
        count += 1
        if hasattr(module, "_weight_scale"):
            scaled += 1

    if scaled > 0:
        logger.info(
            f"fp8-cast: patched {count} nn.Linear layers "
            f"({scaled} with weight_scale, {count - scaled} without)"
        )
    else:
        logger.info(f"fp8-cast: patched {count} nn.Linear layers for per-forward upcast")
    return count
