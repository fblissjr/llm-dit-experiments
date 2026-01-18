"""
Block-by-block quantization for memory-efficient model loading.

Last Updated: 2026-01-18

Ported from LTX-2 official implementation with optimum-quanto integration.
This approach enables loading large models (13B+ params) on GPUs with limited VRAM
by quantizing one transformer block at a time.

Original source: https://github.com/Lightricks/LTX-2
License: LTX-2 Community License
Copyright (c) 2025 Lightricks Ltd.

Usage:
    from llm_dit.utils.quantization import quantize_model, QuantizationPrecision

    # Quantize a model block-by-block
    model = quantize_model(model, precision="fp8-quanto")

    # After quantization, model can be moved to GPU
    model = model.to("cuda")
"""

from __future__ import annotations

import gc
import logging
from typing import Literal

import torch

logger = logging.getLogger(__name__)

QuantizationPrecision = Literal[
    "int8-quanto",
    "int4-quanto",
    "int2-quanto",
    "fp8-quanto",
    "fp8uz-quanto",
]

# Modules to exclude from quantization.
# These are glob patterns passed to quanto's `exclude` parameter.
EXCLUDE_PATTERNS = [
    # Input/output projection layers
    "patchify_proj",
    "audio_patchify_proj",
    "proj_out",
    "audio_proj_out",
    # Timestep embedding layers - require stable precision
    "*adaln*",
    "time_proj",
    "timestep_embedder*",
    # Caption/text projection layers
    "caption_projection*",
    "audio_caption_projection*",
    # Normalization layers (usually excluded from quantization)
    "*norm*",
]

# Top-level modules to skip entirely during block-by-block quantization.
SKIP_ROOT_MODULES = {
    "patchify_proj",
    "audio_patchify_proj",
    "proj_out",
    "audio_proj_out",
    "audio_caption_projection",
}


def cleanup_memory() -> None:
    """Free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def quantize_model(
    model: torch.nn.Module,
    precision: QuantizationPrecision = "fp8-quanto",
    quantize_activations: bool = False,
    device: torch.device | str | None = None,
    verbose: bool = True,
) -> torch.nn.Module:
    """
    Quantize a model using optimum-quanto with block-by-block strategy.

    For large models with transformer_blocks, this function quantizes block-by-block
    on GPU then moves back to CPU, which is much faster than quantizing on CPU and
    uses less peak VRAM than loading the entire model to GPU at once.

    Args:
        model: The model to quantize (should have transformer_blocks attribute)
        precision: Quantization precision (fp8-quanto recommended for RTX 4090)
        quantize_activations: Whether to quantize activations (default False)
        device: Device to use for quantization. If None, uses CUDA if available.
        verbose: Whether to print progress

    Returns:
        The quantized model (on CPU, ready to be moved to GPU)

    Example:
        >>> model = load_ltx2_transformer("models/LTX-2/transformer", device="cpu")
        >>> model = quantize_model(model, precision="fp8-quanto")
        >>> model = model.to("cuda")  # Now fits in 24GB VRAM
    """
    try:
        from optimum.quanto import freeze, quantize
    except ImportError:
        raise ImportError(
            "optimum-quanto is required for quantization. "
            "Install with: pip install optimum-quanto"
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    weight_quant = _get_quanto_dtype(precision)
    activations_quant = weight_quant if quantize_activations else None

    # Remember original device to restore after quantization
    original_device = next(model.parameters()).device

    # Check if model has transformer_blocks for block-by-block quantization
    if hasattr(model, "transformer_blocks"):
        if verbose:
            logger.info(
                f"Quantizing model using block-by-block approach ({precision})"
            )
        _quantize_blockwise(
            model,
            weight_quant=weight_quant,
            activations_quant=activations_quant,
            device=device,
            verbose=verbose,
        )
    else:
        # Fallback: quantize entire model at once
        if verbose:
            logger.info(f"Quantizing model at once ({precision})")
        model.to(device)
        quantize(model, weights=weight_quant, activations=activations_quant, exclude=EXCLUDE_PATTERNS)
        freeze(model)

    # Restore model to original device
    model.to(original_device)
    cleanup_memory()

    return model


def _quantize_blockwise(
    model: torch.nn.Module,
    weight_quant: torch.dtype,
    activations_quant: torch.dtype | None,
    device: torch.device,
    verbose: bool = True,
) -> None:
    """
    Quantize a model block-by-block using optimum-quanto.

    This approach:
    1. Moves each transformer block to GPU
    2. Quantizes on GPU (fast!)
    3. Freezes the quantized weights
    4. Moves back to CPU

    This is much faster than quantizing on CPU and uses less peak VRAM
    than loading the entire model to GPU.
    """
    from optimum.quanto import freeze, quantize

    original_dtype = next(model.parameters()).dtype
    transformer_blocks = list(model.transformer_blocks)
    num_blocks = len(transformer_blocks)

    if verbose:
        logger.info(f"Quantizing {num_blocks} transformer blocks...")

    for i, block in enumerate(transformer_blocks):
        if verbose and (i % 10 == 0 or i == num_blocks - 1):
            logger.info(f"  Block {i + 1}/{num_blocks}")

        # Move block to GPU
        block.to(device, dtype=original_dtype, non_blocking=True)

        # Quantize on GPU
        quantize(block, weights=weight_quant, activations=activations_quant, exclude=EXCLUDE_PATTERNS)
        freeze(block)

        # Move back to CPU to free up VRAM for next block
        block.to("cpu", non_blocking=True)

        # Sync and cleanup every few blocks to prevent memory buildup
        if (i + 1) % 8 == 0:
            torch.cuda.synchronize()
            cleanup_memory()

    # Final sync
    torch.cuda.synchronize()
    cleanup_memory()

    # Quantize remaining non-transformer-block modules
    if verbose:
        logger.info("Quantizing remaining model components...")

    for name, module in model.named_children():
        if name == "transformer_blocks":
            continue  # Already quantized

        if name in SKIP_ROOT_MODULES:
            if verbose:
                logger.debug(f"  Skipping: {name}")
            continue

        # Move to device, quantize, freeze, move back
        module.to(device, dtype=original_dtype, non_blocking=True)
        quantize(module, weights=weight_quant, activations=activations_quant, exclude=EXCLUDE_PATTERNS)
        freeze(module)
        module.to("cpu", non_blocking=True)

    torch.cuda.synchronize()
    cleanup_memory()

    if verbose:
        logger.info("Quantization complete")


def _get_quanto_dtype(precision: QuantizationPrecision) -> torch.dtype:
    """Map precision string to quanto dtype."""
    from optimum.quanto import (
        qfloat8,
        qfloat8_e4m3fnuz,
        qint2,
        qint4,
        qint8,
    )

    dtype_map = {
        "int2-quanto": qint2,
        "int4-quanto": qint4,
        "int8-quanto": qint8,
        "fp8-quanto": qfloat8,
        "fp8uz-quanto": qfloat8_e4m3fnuz,
    }

    if precision not in dtype_map:
        raise ValueError(
            f"Invalid quantization precision: {precision}. "
            f"Valid options: {list(dtype_map.keys())}"
        )

    if precision.startswith("fp8") and torch.backends.mps.is_available():
        raise ValueError(
            "FP8 quantization is not supported on MPS devices. "
            "Use int2, int4, or int8 instead."
        )

    return dtype_map[precision]


def estimate_quantized_size(
    num_params: int,
    precision: QuantizationPrecision,
) -> float:
    """
    Estimate the memory size of a quantized model in GB.

    Args:
        num_params: Number of model parameters
        precision: Quantization precision

    Returns:
        Estimated size in GB
    """
    bits_per_param = {
        "int2-quanto": 2,
        "int4-quanto": 4,
        "int8-quanto": 8,
        "fp8-quanto": 8,
        "fp8uz-quanto": 8,
    }

    bits = bits_per_param.get(precision, 16)  # Default to bf16
    bytes_size = num_params * bits / 8
    gb_size = bytes_size / (1024 ** 3)

    return gb_size
