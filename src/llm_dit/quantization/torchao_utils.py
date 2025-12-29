"""
TorchAO quantization utilities.

Provides functions for applying TorchAO quantization to models
and checking TorchAO availability.

Includes filtered quantization that skips layers with dimensions
not compatible with FP8 _scaled_mm (requires multiples of 16).

last updated: 2025-12-29
"""

import logging
from typing import Callable, Optional, Union

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
        # Use new config API to avoid deprecation warning
        try:
            from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerTensor
            config = Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor())
        except ImportError:
            # Fall back to old API if new one not available
            config = tao_quant.float8_dynamic_activation_float8_weight()
        tao_quant.quantize_(model, config)
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


def is_fp8_compatible_layer(module: nn.Module) -> bool:
    """
    Check if a Linear layer has FP8-compatible dimensions.

    FP8 _scaled_mm requires both in_features and out_features
    to be multiples of 16 for tensor core operations.

    Args:
        module: PyTorch module to check

    Returns:
        True if the layer is FP8-compatible, False otherwise
    """
    if not isinstance(module, nn.Linear):
        return True  # Non-linear layers don't have this constraint

    return module.in_features % 16 == 0 and module.out_features % 16 == 0


def create_fp8_filter_fn(
    skip_incompatible: bool = True,
    verbose: bool = True,
) -> Callable[[nn.Module, str], bool]:
    """
    Create a filter function for TorchAO FP8 quantization.

    The filter function determines which layers to quantize.
    Layers with dimensions not divisible by 16 are skipped
    because FP8 _scaled_mm requires 16-aligned dimensions.

    Args:
        skip_incompatible: If True, skip layers with non-16-aligned dimensions
        verbose: If True, log skipped layers

    Returns:
        Filter function for use with torchao.quantization.quantize_()

    Example:
        filter_fn = create_fp8_filter_fn(skip_incompatible=True)
        quantize_(model, float8_dynamic_activation_float8_weight(), filter_fn=filter_fn)
    """
    skipped_count = 0
    quantized_count = 0

    def filter_fn(module: nn.Module, fqn: str) -> bool:
        nonlocal skipped_count, quantized_count

        # CRITICAL: Return False for non-Linear modules to recurse into children
        # True = "quantize this module", False = "recurse into children"
        if not isinstance(module, nn.Linear):
            return False

        # Now we know it's a Linear layer - apply FP8 compatibility checks
        if skip_incompatible and not is_fp8_compatible_layer(module):
            if verbose:
                logger.info(
                    f"Skipping FP8 for {fqn}: shape [{module.out_features}, {module.in_features}] "
                    f"not 16-aligned (in={module.in_features % 16}, out={module.out_features % 16})"
                )
            skipped_count += 1
            return False

        quantized_count += 1
        return True

    return filter_fn


def quantize_model_torchao_filtered(
    model: nn.Module,
    method: str = "fp8",
    skip_incompatible: bool = True,
    in_place: bool = True,
    verbose: bool = True,
) -> tuple[nn.Module, dict]:
    """
    Apply TorchAO quantization with automatic skipping of incompatible layers.

    For FP8 quantization, layers with dimensions not divisible by 16 are
    automatically skipped because CUDA's _scaled_mm requires 16-aligned dims.

    Args:
        model: PyTorch model to quantize
        method: Quantization method ("fp8", "fp8-filtered", "int8", "int4")
        skip_incompatible: If True, skip FP8-incompatible layers (default: True)
        in_place: If True, modify model in-place (default: True)
        verbose: If True, log skipped layers (default: True)

    Returns:
        Tuple of (quantized_model, stats_dict) where stats_dict contains:
        - total_linear_layers: Total number of Linear layers
        - quantized_layers: Number of layers that were quantized
        - skipped_layers: Number of layers that were skipped
        - skipped_layer_names: List of skipped layer names

    Example:
        model, stats = quantize_model_torchao_filtered(model, "fp8")
        print(f"Quantized {stats['quantized_layers']}/{stats['total_linear_layers']} layers")
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
    if method == "fp8-filtered":
        method = "fp8"  # Normalize the method name

    # Count layers before quantization
    total_layers = sum(1 for m in model.modules() if isinstance(m, nn.Linear))

    # Track skipped layers
    skipped_layers = []

    quantized_layers = []
    layer_count = [0]  # Use list to allow mutation in nested function

    def tracking_filter_fn(module: nn.Module, fqn: str) -> bool:
        # CRITICAL: Return False for non-Linear modules to recurse into children
        # True = "quantize this module", False = "recurse into children"
        if not isinstance(module, nn.Linear):
            return False

        layer_count[0] += 1

        # Now we know it's a Linear layer - apply FP8 compatibility checks
        if method == "fp8" and skip_incompatible:
            if not is_fp8_compatible_layer(module):
                if verbose:
                    logger.info(
                        f"  [SKIP] {fqn}: [{module.out_features}, {module.in_features}] "
                        f"(in%16={module.in_features % 16}, out%16={module.out_features % 16})"
                    )
                skipped_layers.append(fqn)
                return False

        quantized_layers.append(fqn)

        # Show progress every 50 layers (or every layer if verbose)
        if verbose:
            logger.info(f"  [QUANT] {fqn}: [{module.out_features}, {module.in_features}]")
        elif layer_count[0] % 50 == 0:
            logger.info(f"  Progress: {layer_count[0]}/{total_layers} layers processed...")

        return True

    if method == "fp8":
        logger.info("Applying TorchAO FP8 dynamic quantization (with filtering)...")
        # Use new config API to avoid deprecation warning
        try:
            from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerTensor
            config = Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor())
        except ImportError:
            # Fall back to old API if new one not available
            config = tao_quant.float8_dynamic_activation_float8_weight()
        tao_quant.quantize_(
            model,
            config,
            filter_fn=tracking_filter_fn,
        )

    elif method == "int8":
        logger.info("Applying TorchAO INT8 weight-only quantization...")
        tao_quant.quantize_(
            model,
            tao_quant.int8_weight_only(),
            filter_fn=tracking_filter_fn if skip_incompatible else None,
        )

    elif method == "int4":
        logger.info("Applying TorchAO INT4 weight-only quantization...")
        tao_quant.quantize_(
            model,
            tao_quant.int4_weight_only(),
            filter_fn=tracking_filter_fn if skip_incompatible else None,
        )

    else:
        raise ValueError(
            f"Unknown TorchAO method: {method}. "
            "Supported: fp8, fp8-filtered, int8, int4"
        )

    quantized_count = total_layers - len(skipped_layers)
    logger.info(
        f"Quantization complete: {quantized_count}/{total_layers} layers quantized, "
        f"{len(skipped_layers)} skipped"
    )

    stats = {
        "total_linear_layers": total_layers,
        "quantized_layers": quantized_count,
        "skipped_layers": len(skipped_layers),
        "skipped_layer_names": skipped_layers,
    }

    return model, stats


def analyze_fp8_compatibility(model: nn.Module) -> dict:
    """
    Analyze a model's FP8 compatibility without quantizing.

    Args:
        model: PyTorch model to analyze

    Returns:
        Dict with compatibility analysis:
        - total_linear_layers: Total Linear layers
        - compatible_layers: FP8-compatible layers
        - incompatible_layers: Layers that would be skipped
        - compatibility_rate: Percentage of compatible layers
        - incompatible_layer_info: List of dicts with layer details
    """
    total = 0
    compatible = 0
    incompatible_info = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total += 1
            if is_fp8_compatible_layer(module):
                compatible += 1
            else:
                incompatible_info.append({
                    "name": name,
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "in_remainder": module.in_features % 16,
                    "out_remainder": module.out_features % 16,
                })

    return {
        "total_linear_layers": total,
        "compatible_layers": compatible,
        "incompatible_layers": total - compatible,
        "compatibility_rate": (compatible / total * 100) if total > 0 else 0,
        "incompatible_layer_info": incompatible_info,
    }
