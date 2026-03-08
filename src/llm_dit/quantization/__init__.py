"""
Quantization utilities for LLM-DiT models.

Unified torchao-based quantization system. All model components (encoders,
transformers, VAEs) are quantized through a single entry point: quantize_component().

Supported methods (SM89 / RTX 4090):
- none: BF16 (no quantization)
- fp8-dynamic: FP8 weights + FP8 activations (scaled_mm)
- fp8-weight-only: FP8 weights, BF16 activations
- int8: INT8 weight-only
- int4: INT4 weight-only (max compression)

Example:
    from llm_dit.quantization import quantize_component, VALID_METHODS

    model, stats = quantize_component(model, method="fp8-weight-only", component_type="transformer")
    print(f"Quantized {stats['quantized_layers']}/{stats['total_layers']} layers")
"""

from .torchao_utils import (
    quantize_component,
    get_quant_compile_warnings,
    VALID_METHODS,
    is_torchao_available,
    check_fp8_support,
    get_recommended_method,
    get_torchao_version,
)
from .vae_utils import (
    quantize_vae,
    estimate_vae_vram,
    get_vae_quant_info,
)
# Canonical alias map: shorthand -> full torchao method name.
# Imported by config.py and pipelines/generate.py. Single source of truth.
QUANT_ALIASES: dict[str, str] = {"fp8": "fp8-dynamic"}

from .layerwise_fp8 import apply_fp8_layerwise_casting

__all__ = [
    # Unified API
    "quantize_component",
    "get_quant_compile_warnings",
    "VALID_METHODS",
    # Utilities
    "is_torchao_available",
    "check_fp8_support",
    "get_recommended_method",
    "get_torchao_version",
    # VAE quantization
    "quantize_vae",
    "estimate_vae_vram",
    "get_vae_quant_info",
    # Layerwise fp8 casting (no torchao dependency)
    "apply_fp8_layerwise_casting",
    # Alias map
    "QUANT_ALIASES",
]
