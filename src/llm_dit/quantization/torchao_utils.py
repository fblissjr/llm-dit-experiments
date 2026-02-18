"""
TorchAO quantization utilities.

Provides quantize_component() as the unified entry point for quantizing
any model component (encoder, transformer, VAE).

last updated: 2026-02-06
"""

import logging
import re
from typing import Callable, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical method strings (matches config.VALID_QUANT_METHODS)
# ---------------------------------------------------------------------------
VALID_METHODS = ("none", "fp8-dynamic", "fp8-weight-only", "int8", "int4")

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

    major, minor = torch.cuda.get_device_capability()
    compute_cap = major + minor / 10

    return compute_cap >= 8.9


def get_recommended_method() -> str:
    """
    Get recommended quantization method for current hardware.

    Returns:
        Recommended method string using unified method names.
    """
    if not torch.cuda.is_available():
        return "int8"

    if check_fp8_support():
        return "fp8-dynamic"

    return "int8"


def is_fp8_compatible_layer(module: nn.Module) -> bool:
    """
    Check if a Linear layer has FP8-compatible dimensions.

    FP8 _scaled_mm requires both in_features and out_features
    to be multiples of 16 for tensor core operations.
    """
    if not isinstance(module, nn.Linear):
        return True

    return module.in_features % 16 == 0 and module.out_features % 16 == 0


# ---------------------------------------------------------------------------
# Unified quantization entry point
# ---------------------------------------------------------------------------

def _build_method_map() -> dict:
    """Build METHOD_MAP lazily (imports torchao at call time, not import time)."""
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        Float8WeightOnlyConfig,
        Int4WeightOnlyConfig,
        Int8WeightOnlyConfig,
        PerRow,
        PerTensor,
    )

    return {
        "fp8-dynamic": lambda g: Float8DynamicActivationFloat8WeightConfig(
            granularity=PerRow() if g == "per-row" else PerTensor()
        ),
        "fp8-weight-only": lambda _g: Float8WeightOnlyConfig(),
        "int8": lambda _g: Int8WeightOnlyConfig(),
        "int4": lambda _g: Int4WeightOnlyConfig(),
    }


# Encoder layers that should NOT be quantized (embeddings, norms, LM head)
_ENCODER_SKIP_PATTERNS = [
    r".*embed_tokens.*",
    r".*lm_head.*",
    r".*norm.*",
    r".*rotary_emb.*",
]

# Transformer layers that should NOT be quantized (norms, projections)
_TRANSFORMER_SKIP_PATTERNS = [
    r".*norm.*",
]


def _build_fqn_filter(
    skip_patterns: list[str],
    method: str,
    verbose: bool = True,
) -> Callable[[nn.Module, str], bool]:
    """Build a filter function that skips modules matching regex patterns.

    For FP8 methods, also skips Linear layers with non-16-aligned dims.
    """
    compiled = [re.compile(p) for p in skip_patterns]
    is_fp8 = method in ("fp8-dynamic", "fp8-weight-only")
    stats: dict[str, list[str]] = {"quantized": [], "skipped": []}

    def _filter(module: nn.Module, fqn: str) -> bool:
        if not isinstance(module, nn.Linear):
            return False  # recurse into children

        # Check skip patterns
        for pattern in compiled:
            if pattern.match(fqn):
                stats["skipped"].append(fqn)
                if verbose:
                    logger.debug(f"  [SKIP] {fqn} (pattern match)")
                return False

        # FP8 dim-16 check
        if is_fp8 and not is_fp8_compatible_layer(module):
            stats["skipped"].append(fqn)
            if verbose:
                logger.debug(
                    f"  [SKIP] {fqn}: [{module.out_features}, {module.in_features}] "
                    f"not 16-aligned"
                )
            return False

        stats["quantized"].append(fqn)
        return True

    # Attach stats dict so caller can read it after quantize_() completes
    _filter._stats = stats  # type: ignore[attr-defined]
    return _filter


def quantize_component(
    model: nn.Module,
    method: str,
    component_type: str = "transformer",
    granularity: str = "per-tensor",
    verbose: bool = True,
) -> tuple[nn.Module, dict]:
    """Unified quantization entry point for any model component.

    Applies torchao quantization with component-specific filter logic:
    - encoder: skips embed_tokens, norms, lm_head, rotary_emb
    - transformer: skips norms + dim-16 check for FP8
    - vae: delegates to vae_utils.quantize_vae() for Conv2d int8

    Args:
        model: Model to quantize (modified in place)
        method: One of VALID_METHODS ("none", "fp8-dynamic", "fp8-weight-only",
                "int8", "int4")
        component_type: "encoder", "transformer", or "vae"
        granularity: "per-tensor" or "per-row" (FP8 only)
        verbose: Log progress

    Returns:
        (model, stats_dict) where stats_dict has:
        quantized_layers, skipped_layers, total_layers, method, component_type
    """
    if method == "none":
        return model, {
            "quantized_layers": 0,
            "skipped_layers": 0,
            "total_layers": 0,
            "method": "none",
            "component_type": component_type,
        }

    # Check if model already has quantized weights (avoid redundant re-quantization)
    for p in model.parameters():
        ptype = type(p).__name__
        if ptype in ("Float8Tensor", "AffineQuantizedTensor"):
            logger.info(
                f"Model already has {ptype} weights, skipping {method} quantization"
            )
            return model, {
                "quantized_layers": 0,
                "skipped_layers": 0,
                "total_layers": 0,
                "method": "already_quantized",
                "component_type": component_type,
            }
        if hasattr(p, "dtype") and p.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            logger.info(
                f"Model already has native {p.dtype} weights, skipping {method} quantization"
            )
            return model, {
                "quantized_layers": 0,
                "skipped_layers": 0,
                "total_layers": 0,
                "method": "already_quantized",
                "component_type": component_type,
            }
        break  # Only check first parameter

    if method not in VALID_METHODS:
        raise ValueError(
            f"Unknown quantization method: '{method}'. "
            f"Valid: {', '.join(VALID_METHODS)}"
        )

    if not is_torchao_available():
        raise ImportError("TorchAO is not available. Install with: uv add torchao")

    # VAE special case: Conv2d needs different handling
    if component_type == "vae":
        from .vae_utils import quantize_vae

        vae_method = "int8" if method in ("int8", "int4") else "none"
        if method in ("fp8-dynamic", "fp8-weight-only"):
            logger.warning(
                f"FP8 not recommended for VAE Conv2d layers. "
                f"Falling back to int8 for VAE."
            )
            vae_method = "int8"
        model = quantize_vae(model, method=vae_method)
        return model, {
            "quantized_layers": -1,  # vae_utils doesn't track per-layer
            "skipped_layers": 0,
            "total_layers": -1,
            "method": vae_method,
            "component_type": "vae",
        }

    import torchao.quantization as tao_quant

    method_map = _build_method_map()
    config = method_map[method](granularity)

    # Select filter based on component type
    if component_type == "encoder":
        skip_patterns = _ENCODER_SKIP_PATTERNS
    elif component_type == "transformer":
        skip_patterns = _TRANSFORMER_SKIP_PATTERNS
    else:
        skip_patterns = _TRANSFORMER_SKIP_PATTERNS  # default

    filter_fn = _build_fqn_filter(skip_patterns, method, verbose=verbose)

    total_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    logger.info(
        f"Quantizing {component_type} with {method} "
        f"(granularity={granularity}, {total_linear} linear layers)..."
    )

    tao_quant.quantize_(model, config, filter_fn=filter_fn)

    # Read stats from filter function
    filter_stats = filter_fn._stats  # type: ignore[attr-defined]
    quantized_count = len(filter_stats["quantized"])
    skipped_count = len(filter_stats["skipped"])

    logger.info(
        f"Quantization complete: {quantized_count}/{total_linear} layers quantized, "
        f"{skipped_count} skipped"
    )

    return model, {
        "quantized_layers": quantized_count,
        "skipped_layers": skipped_count,
        "total_layers": total_linear,
        "method": method,
        "component_type": component_type,
    }


def get_quant_compile_warnings(method: str, compile_mode: str) -> list[str]:
    """Return warnings for dangerous quant + compile combinations.

    Args:
        method: Quantization method string
        compile_mode: torch.compile mode string

    Returns:
        List of warning strings (empty if no issues)
    """
    warnings = []
    if method == "fp8-dynamic" and "autotune" in compile_mode:
        warnings.append(
            f"fp8-dynamic + compile_mode={compile_mode} causes 5+ minute autotune "
            f"warmup on first request. Consider fp8-weight-only instead."
        )
    return warnings
