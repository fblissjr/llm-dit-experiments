"""
Native FP8 quantization matching official LTX-2 approach.

Last Updated: 2026-01-19

Uses torch.float8_e4m3fn for weight storage with upcasting during forward.
No frozen buffers, no memory leaks.

Reference: coderef/LTX-2/packages/ltx-core/src/ltx_core/model/transformer/model_configurator.py

Key Design Decisions:
- Weights stored as FP8 (8-bit) for ~50% memory reduction vs bf16
- Forward pass upcasts to input dtype (typically bf16) for numerical stability
- No scale factors or frozen buffers = no memory accumulation
- Skip patterns exclude sensitive layers (norms, projections) that need precision
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Default layers to skip for FP8 quantization
# These layers are sensitive to quantization and should remain in full precision
DEFAULT_SKIP_PATTERNS = [
    "norm",           # All normalization layers (RMSNorm, LayerNorm)
    "adaln",          # Adaptive layer norm (timestep conditioning)
    "proj_out",       # Final output projection
    "patchify",       # Initial patchification projection
    "caption_projection",  # Text projection layer
]


def apply_fp8_native(
    model: nn.Module,
    skip_patterns: Optional[list[str]] = None,
    verbose: bool = True,
) -> tuple[nn.Module, dict]:
    """
    Apply native FP8 quantization with upcast-on-forward.

    Matches official LTX-2 approach:
    - Weights stored as torch.float8_e4m3fn
    - Forward upcasts to input dtype before F.linear()
    - No frozen scale buffers = no memory leak

    Args:
        model: Model to quantize
        skip_patterns: Layer name patterns to skip (e.g., ["norm", "adaln"])
                       Defaults to DEFAULT_SKIP_PATTERNS
        verbose: Log progress

    Returns:
        (quantized_model, stats_dict)
    """
    if skip_patterns is None:
        skip_patterns = DEFAULT_SKIP_PATTERNS.copy()

    quantized_count = 0
    skipped_count = 0
    skipped_by_pattern: dict[str, int] = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check skip patterns
            skip_reason = None
            for pattern in skip_patterns:
                if pattern in name.lower():
                    skip_reason = pattern
                    break

            if skip_reason:
                skipped_count += 1
                skipped_by_pattern[skip_reason] = skipped_by_pattern.get(skip_reason, 0) + 1
                continue

            # Convert weights to FP8
            _convert_linear_to_fp8(module)
            quantized_count += 1

            if verbose and quantized_count % 50 == 0:
                logger.info(f"  Quantized {quantized_count} layers...")

    stats = {
        "quantized": quantized_count,
        "skipped": skipped_count,
        "skipped_by_pattern": skipped_by_pattern,
    }

    if verbose:
        logger.info(f"Native FP8: {quantized_count} quantized, {skipped_count} skipped")
        if skipped_by_pattern:
            logger.info(f"  Skipped by pattern: {skipped_by_pattern}")

    return model, stats


def _convert_linear_to_fp8(layer: nn.Linear) -> None:
    """
    Convert a Linear layer to FP8 storage with upcast-on-forward.

    The weight tensor is stored as torch.float8_e4m3fn (8-bit float).
    During forward pass, weights are upcast to the input dtype.
    """
    original_dtype = layer.weight.dtype

    # Store weight as FP8
    layer.weight.data = layer.weight.data.to(torch.float8_e4m3fn)
    if layer.bias is not None:
        layer.bias.data = layer.bias.data.to(torch.float8_e4m3fn)

    # Store original dtype for upcasting reference
    layer._fp8_original_dtype = original_dtype

    # Replace forward with upcasting version
    _replace_linear_forward_with_upcast(layer)


def _replace_linear_forward_with_upcast(layer: nn.Linear) -> None:
    """
    Replace forward to upcast FP8 weights to input dtype.

    This is the key to avoiding memory leaks - we don't create persistent
    scale buffers like quanto does. Instead, we upcast on-the-fly.
    """
    original_forward = layer.forward

    def new_forward(x: torch.Tensor) -> torch.Tensor:
        # Upcast weights to input dtype (typically bf16)
        w = layer.weight.to(x.dtype)
        b = layer.bias.to(x.dtype) if layer.bias is not None else None
        return F.linear(x, w, b)

    layer.forward = new_forward
    layer._original_forward = original_forward


def estimate_memory_savings(model: nn.Module, skip_patterns: Optional[list[str]] = None) -> dict:
    """
    Estimate memory savings from FP8 quantization without applying it.

    Args:
        model: Model to analyze
        skip_patterns: Layer patterns to skip

    Returns:
        Dict with original_gb, quantized_gb, savings_gb, savings_percent
    """
    if skip_patterns is None:
        skip_patterns = DEFAULT_SKIP_PATTERNS.copy()

    original_bytes = 0
    quantized_bytes = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight_numel = module.weight.numel()
            bias_numel = module.bias.numel() if module.bias is not None else 0
            total_numel = weight_numel + bias_numel

            # Original size (assume bf16 = 2 bytes)
            original_bytes += total_numel * 2

            # Check if this layer would be skipped
            should_skip = any(pattern in name.lower() for pattern in skip_patterns)
            if should_skip:
                # Skipped layers stay in bf16
                quantized_bytes += total_numel * 2
            else:
                # FP8 = 1 byte
                quantized_bytes += total_numel * 1

    original_gb = original_bytes / (1024**3)
    quantized_gb = quantized_bytes / (1024**3)
    savings_gb = original_gb - quantized_gb

    return {
        "original_gb": round(original_gb, 2),
        "quantized_gb": round(quantized_gb, 2),
        "savings_gb": round(savings_gb, 2),
        "savings_percent": round((savings_gb / original_gb) * 100, 1) if original_gb > 0 else 0,
    }


def verify_fp8_applied(model: nn.Module) -> dict:
    """
    Verify FP8 quantization was applied correctly.

    Args:
        model: Model to verify

    Returns:
        Dict with fp8_layers count, bf16_layers count, and any issues
    """
    fp8_layers = 0
    bf16_layers = 0
    issues = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if module.weight.dtype == torch.float8_e4m3fn:
                fp8_layers += 1
                # Verify forward replacement
                if not hasattr(module, "_original_forward"):
                    issues.append(f"{name}: FP8 but missing forward replacement")
            else:
                bf16_layers += 1

    return {
        "fp8_layers": fp8_layers,
        "bf16_layers": bf16_layers,
        "total_linear": fp8_layers + bf16_layers,
        "issues": issues,
    }
