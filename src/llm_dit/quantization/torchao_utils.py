"""
TorchAO quantization utilities.

Provides functions for applying TorchAO quantization to models
and checking TorchAO availability.

last updated: 2025-12-27
"""

import logging
from typing import Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_TORCHAO_AVAILABLE = None


def is_torchao_available() -> bool:
    """Check if TorchAO is available."""
    global _TORCHAO_AVAILABLE
    if _TORCHAO_AVAILABLE is None:
        try:
            import torchao
            _TORCHAO_AVAILABLE = True
        except ImportError:
            _TORCHAO_AVAILABLE = False
    return _TORCHAO_AVAILABLE


def get_torchao_version() -> Optional[str]:
    """Get TorchAO version if available."""
    if is_torchao_available():
        import torchao
        return getattr(torchao, "__version__", "unknown")
    return None


def quantize_model_torchao(
    model: nn.Module,
    method: str,
    in_place: bool = True,
) -> nn.Module:
    """
    Apply TorchAO quantization to a PyTorch model.

    This is for standalone models (not diffusers pipelines).
    For diffusers pipelines, use get_quantization_config() instead.

    Args:
        model: PyTorch model to quantize
        method: Quantization method ("fp8" or "int8")
        in_place: If True, modify model in-place. If False, return new model.

    Returns:
        Quantized model

    Example:
        from llm_dit.quantization import quantize_model_torchao
        quantize_model_torchao(model, "fp8")
    """
    if not is_torchao_available():
        raise ImportError(
            "TorchAO is not available. Install with: uv add torchao"
        )

    import torchao.quantization as tao_quant

    if not in_place:
        import copy
        model = copy.deepcopy(model)

    method = method.lower().strip()

    if method == "fp8":
        # FP8 dynamic quantization
        # Quantizes both weights and activations to FP8
        # Best for RTX 4090+ (compute capability 8.9+)
        logger.info("Applying TorchAO FP8 dynamic quantization...")
        tao_quant.quantize_(
            model,
            tao_quant.float8_dynamic_activation_float8_weight(),
        )
        logger.info("FP8 quantization applied successfully")

    elif method == "int8":
        # INT8 weight-only quantization
        # Only quantizes weights, activations stay in original dtype
        # Works on any GPU
        logger.info("Applying TorchAO INT8 weight-only quantization...")
        tao_quant.quantize_(
            model,
            tao_quant.int8_weight_only(),
        )
        logger.info("INT8 quantization applied successfully")

    elif method == "int4":
        # INT4 weight-only quantization
        # Maximum compression, some quality loss
        logger.info("Applying TorchAO INT4 weight-only quantization...")
        tao_quant.quantize_(
            model,
            tao_quant.int4_weight_only(),
        )
        logger.info("INT4 quantization applied successfully")

    else:
        raise ValueError(
            f"Unknown TorchAO method: {method}. "
            "Supported: fp8, int8, int4"
        )

    return model


def check_fp8_support() -> bool:
    """
    Check if FP8 is supported on current GPU.

    FP8 requires compute capability 8.9+ (Ada Lovelace / RTX 4090, H100)
    or compute capability 9.0+ (Hopper).

    Returns:
        True if FP8 is supported, False otherwise.
    """
    if not torch.cuda.is_available():
        return False

    # Get compute capability
    major, minor = torch.cuda.get_device_capability()
    compute_cap = major + minor / 10

    # FP8 requires 8.9+ (Ada Lovelace) or 9.0+ (Hopper)
    return compute_cap >= 8.9


def get_recommended_method() -> str:
    """
    Get recommended quantization method for current hardware.

    Returns:
        Recommended method string ("fp8", "int8", or "8bit")
    """
    if not torch.cuda.is_available():
        return "8bit"  # BitsAndBytes for CPU-only

    if check_fp8_support():
        return "fp8"  # TorchAO FP8 for RTX 4090+ / H100

    return "int8"  # TorchAO INT8 for older GPUs
