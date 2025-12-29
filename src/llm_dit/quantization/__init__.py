"""
Quantization utilities for LLM-DiT models.

Supports multiple quantization backends:
- TorchAO: FP8, INT8 weight-only (recommended for RTX 4090+)
- BitsAndBytes: 4-bit NF4, 8-bit INT8 (works with bfloat16)

Example:
    from llm_dit.quantization import get_quantization_config, QuantizationMethod

    # For diffusers pipelines
    config = get_quantization_config(QuantizationMethod.FP8)
    pipe = Pipeline.from_pretrained(path, quantization_config=config)

    # For standalone models with FP8 filtering (skips incompatible layers)
    from llm_dit.quantization import quantize_model_torchao_filtered
    model, stats = quantize_model_torchao_filtered(model, "fp8")
    print(f"Quantized {stats['quantized_layers']}/{stats['total_linear_layers']} layers")
"""

from .config import QuantizationMethod, get_quantization_config, validate_quantization_dtype
from .torchao_utils import (
    quantize_model_torchao,
    quantize_model_torchao_filtered,
    is_torchao_available,
    check_fp8_support,
    get_recommended_method,
    get_torchao_version,
    is_fp8_compatible_layer,
    create_fp8_filter_fn,
    analyze_fp8_compatibility,
)

__all__ = [
    "QuantizationMethod",
    "get_quantization_config",
    "validate_quantization_dtype",
    "quantize_model_torchao",
    "quantize_model_torchao_filtered",
    "is_torchao_available",
    "check_fp8_support",
    "get_recommended_method",
    "get_torchao_version",
    "is_fp8_compatible_layer",
    "create_fp8_filter_fn",
    "analyze_fp8_compatibility",
]
