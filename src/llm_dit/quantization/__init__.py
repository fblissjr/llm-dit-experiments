"""
Quantization utilities for LLM-DiT models.

Supports multiple quantization backends:
- TorchAO: FP8, INT8 weight-only (recommended for RTX 4090+)
- BitsAndBytes: 4-bit NF4, 8-bit INT8 (works with bfloat16)
- DiffSynth-style FP8: Runtime F.linear patching with torch._scaled_mm

Example:
    from llm_dit.quantization import get_quantization_config, QuantizationMethod

    # For diffusers pipelines
    config = get_quantization_config(QuantizationMethod.FP8)
    pipe = Pipeline.from_pretrained(path, quantization_config=config)

    # For standalone models with FP8 filtering (skips incompatible layers)
    from llm_dit.quantization import quantize_model_torchao_filtered
    model, stats = quantize_model_torchao_filtered(model, "fp8")
    print(f"Quantized {stats['quantized_layers']}/{stats['total_linear_layers']} layers")

    # DiffSynth-style FP8 inference (runtime patching)
    from llm_dit.quantization import fp8_inference, enable_fp8_weights
    enable_fp8_weights(model)  # Optional: pre-convert weights for memory savings
    with fp8_inference():
        output = model(input)
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
from .fp8_inference import (
    fp8_inference,
    enable_fp8_weights,
    enable_fp8_autocast,
    check_fp8_available,
    get_fp8_info,
    get_fp8_dtype,
    get_fp8_max,
)
from .vae_utils import (
    quantize_vae,
    estimate_vae_vram,
    get_vae_quant_info,
)
from .fp8_native import (
    apply_fp8_native,
    estimate_memory_savings,
    verify_fp8_applied,
    ALLOWLIST_SUFFIXES,
)

__all__ = [
    # Config
    "QuantizationMethod",
    "get_quantization_config",
    "validate_quantization_dtype",
    # TorchAO
    "quantize_model_torchao",
    "quantize_model_torchao_filtered",
    "is_torchao_available",
    "check_fp8_support",
    "get_recommended_method",
    "get_torchao_version",
    "is_fp8_compatible_layer",
    "create_fp8_filter_fn",
    "analyze_fp8_compatibility",
    # DiffSynth-style FP8 inference
    "fp8_inference",
    "enable_fp8_weights",
    "enable_fp8_autocast",
    "check_fp8_available",
    "get_fp8_info",
    "get_fp8_dtype",
    "get_fp8_max",
    # VAE quantization
    "quantize_vae",
    "estimate_vae_vram",
    "get_vae_quant_info",
    # Native FP8 (official LTX-2 approach)
    "apply_fp8_native",
    "estimate_memory_savings",
    "verify_fp8_applied",
    "ALLOWLIST_SUFFIXES",
]
