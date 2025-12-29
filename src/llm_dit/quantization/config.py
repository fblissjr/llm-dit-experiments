"""
Quantization configuration and method definitions.

last updated: 2025-12-29
"""

import logging
from enum import Enum
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


class QuantizationMethod(Enum):
    """Supported quantization methods."""

    # No quantization
    NONE = "none"

    # BitsAndBytes (works with bfloat16)
    BNB_4BIT = "4bit"  # NF4 quantization, ~75% VRAM reduction
    BNB_8BIT = "8bit"  # INT8 quantization, ~50% VRAM reduction

    # TorchAO (recommended for RTX 4090+)
    TORCHAO_FP8 = "fp8"  # FP8 dynamic, ~50% VRAM reduction, minimal quality loss
    TORCHAO_FP8_FILTERED = "fp8-filtered"  # FP8 with auto-skip of non-16-aligned layers
    TORCHAO_INT8 = "int8"  # INT8 weight-only, ~50% VRAM reduction

    # Deprecated
    INT8_DYNAMIC = "int8_dynamic"  # PyTorch native, requires float32

    @classmethod
    def from_string(cls, value: str) -> "QuantizationMethod":
        """Parse quantization method from string."""
        value = value.lower().strip()

        # Handle aliases
        aliases = {
            "fp8": cls.TORCHAO_FP8,
            "fp8-filtered": cls.TORCHAO_FP8_FILTERED,
            "fp8_filtered": cls.TORCHAO_FP8_FILTERED,  # Allow underscore variant
            "int8": cls.TORCHAO_INT8,
            "4bit": cls.BNB_4BIT,
            "8bit": cls.BNB_8BIT,
            "none": cls.NONE,
            "int8_dynamic": cls.INT8_DYNAMIC,
        }

        if value in aliases:
            return aliases[value]

        raise ValueError(
            f"Unknown quantization method: {value}. "
            f"Supported: {list(aliases.keys())}"
        )

    def is_torchao(self) -> bool:
        """Check if this method uses TorchAO."""
        return self in (self.TORCHAO_FP8, self.TORCHAO_FP8_FILTERED, self.TORCHAO_INT8)

    def is_fp8(self) -> bool:
        """Check if this method uses FP8 quantization."""
        return self in (self.TORCHAO_FP8, self.TORCHAO_FP8_FILTERED)

    def is_bitsandbytes(self) -> bool:
        """Check if this method uses BitsAndBytes."""
        return self in (self.BNB_4BIT, self.BNB_8BIT)

    def is_deprecated(self) -> bool:
        """Check if this method is deprecated."""
        return self == self.INT8_DYNAMIC


def get_quantization_config(
    method: Union[str, QuantizationMethod],
    dtype: torch.dtype = torch.bfloat16,
    for_diffusers: bool = True,
):
    """
    Get quantization config for the specified method.

    Args:
        method: Quantization method (string or enum)
        dtype: Compute dtype (used for TorchAO)
        for_diffusers: If True, return diffusers-compatible config

    Returns:
        Quantization config object (TorchAoConfig or BitsAndBytesConfig)
        or None if no quantization.

    Example:
        config = get_quantization_config("fp8")
        pipe = Pipeline.from_pretrained(path, quantization_config=config)
    """
    if isinstance(method, str):
        method = QuantizationMethod.from_string(method)

    if method == QuantizationMethod.NONE:
        return None

    # Handle deprecated int8_dynamic
    if method == QuantizationMethod.INT8_DYNAMIC:
        logger.warning(
            "quantization='int8_dynamic' is deprecated and incompatible with bfloat16. "
            "Auto-migrating to 'int8' (TorchAO INT8 weight-only). "
            "Update your config to use 'int8' or '8bit' instead."
        )
        method = QuantizationMethod.TORCHAO_INT8

    # TorchAO methods
    if method.is_torchao():
        try:
            from diffusers import TorchAoConfig
        except ImportError:
            raise ImportError(
                "TorchAoConfig requires diffusers >= 0.32.0. "
                "Install with: uv add diffusers>=0.32.0"
            )

        if method in (QuantizationMethod.TORCHAO_FP8, QuantizationMethod.TORCHAO_FP8_FILTERED):
            # FP8 dynamic quantization - best for RTX 4090+ (compute capability 8.9+)
            # Note: fp8-filtered uses the same config, but quantize_model_torchao_filtered()
            # should be used for post-load quantization to skip incompatible layers
            if method == QuantizationMethod.TORCHAO_FP8_FILTERED:
                logger.info(
                    "Using TorchAO FP8 (filtered) - incompatible layers will be skipped. "
                    "Use quantize_model_torchao_filtered() for post-load quantization."
                )
            else:
                logger.info("Using TorchAO FP8 dynamic quantization (~50% VRAM reduction)")
            return TorchAoConfig("float8dq")
        elif method == QuantizationMethod.TORCHAO_INT8:
            # INT8 weight-only - works with any GPU
            logger.info("Using TorchAO INT8 weight-only quantization (~50% VRAM reduction)")
            return TorchAoConfig("int8wo")

    # BitsAndBytes methods
    if method.is_bitsandbytes():
        try:
            if for_diffusers:
                from diffusers import BitsAndBytesConfig
            else:
                from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "BitsAndBytesConfig requires bitsandbytes. "
                "Install with: uv add bitsandbytes"
            )

        if method == QuantizationMethod.BNB_4BIT:
            logger.info("Using BitsAndBytes NF4 4-bit quantization (~75% VRAM reduction)")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
        elif method == QuantizationMethod.BNB_8BIT:
            logger.info("Using BitsAndBytes INT8 quantization (~50% VRAM reduction)")
            return BitsAndBytesConfig(load_in_8bit=True)

    raise ValueError(f"Unsupported quantization method: {method}")


def validate_quantization_dtype(
    method: Union[str, QuantizationMethod],
    dtype: torch.dtype,
) -> Optional[str]:
    """
    Validate that quantization method is compatible with dtype.

    Args:
        method: Quantization method
        dtype: Model dtype

    Returns:
        Warning message if incompatible, None if OK.
    """
    if isinstance(method, str):
        method = QuantizationMethod.from_string(method)

    # int8_dynamic requires float32
    if method == QuantizationMethod.INT8_DYNAMIC and dtype != torch.float32:
        return (
            f"quantization='int8_dynamic' requires float32 dtype, but got {dtype}. "
            "Use 'int8' (TorchAO) or '8bit' (BitsAndBytes) for bfloat16 support."
        )

    return None
