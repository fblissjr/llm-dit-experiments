"""FP8-cast quantization utilities.

Last Updated: 2026-03-26

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

Two forward strategies are available:

- **scaled_mm** (preferred on CUDA): Uses `torch._scaled_mm` to run matmul on FP8
  tensor cores. Activations are cast to fp8 in-flight. ~2x faster than upcast on
  RTX 4090. Requires dims divisible by 16.

- **upcast** (fallback): Upcasts fp8 weight to bf16, multiplies by weight_scale,
  then runs F.linear in bf16. Works everywhere, no dim constraints.

Memory footprint: ~12GB for 22B model (vs ~26GB bf16, ~42GB dequant+requant).

Usage:
    # Patch all nn.Linear layers in a model with fp8 weights
    count = amend_forward_with_upcast(model)
"""

import logging
import os
from functools import lru_cache

import torch
from torch import nn

logger = logging.getLogger(__name__)

# Diagnostic escape hatch: force upcast path even when scaled_mm is available.
# Set LLM_DIT_FORCE_FP8_UPCAST=1 to bypass scaled_mm for debugging.
_FORCE_UPCAST = os.environ.get("LLM_DIT_FORCE_FP8_UPCAST", "").lower() in ("1", "true")


@lru_cache(maxsize=1)
def is_scaled_mm_available() -> bool:
    """Check if torch._scaled_mm is available and functional on current hardware.

    Tests with a small matmul to verify the CUDA kernel actually works
    (not all GPU architectures support all scale configurations).
    """
    if not hasattr(torch, "_scaled_mm"):
        return False
    if not torch.cuda.is_available():
        return False
    try:
        # Minimal test: 16x16 matmul (minimum dims for _scaled_mm)
        a = torch.zeros(16, 16, device="cuda", dtype=torch.float8_e4m3fn)
        b = torch.zeros(16, 16, device="cuda", dtype=torch.float8_e4m3fn)
        s = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        torch._scaled_mm(a, b.T, scale_a=s, scale_b=s, out_dtype=torch.bfloat16)
        return True
    except (RuntimeError, AttributeError):
        return False


def quantize_to_fp8_per_tensor(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight tensor to FP8 (float8_e4m3fn) with per-tensor scaling.

    Matches the official LTX-2 `quantize_weight_to_fp8_per_tensor` pattern:
    compute global max abs, derive scale, clamp and cast.

    Note: reference transposes for cuBLAS layout. We do NOT transpose since
    we use standard PyTorch [out, in] layout with F.linear.

    Args:
        weight: Any-dtype weight tensor.

    Returns:
        (quantized_fp8_weight, weight_scale) where
        weight_scale = max_abs / fp8_max (reciprocal of quantization scale).
        Dequantize: real_weight = fp8_weight * weight_scale.
    """
    w = weight.to(torch.float32)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    max_abs = w.abs().amax().clamp(min=1e-12)
    scale = fp8_max / max_abs
    quantized = (w * scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    weight_scale = scale.reciprocal()
    return quantized, weight_scale


def _replace_fwd_with_scaled_mm(layer: nn.Linear) -> None:
    """Replace linear.forward with torch._scaled_mm for native FP8 tensor core matmul.

    Activations are cast to fp8 in-flight (simple truncation, no dynamic scaling).
    Weight stays fp8, and weight_scale (if present) is passed as scale_b.

    ~2x faster than the upcast path on RTX 4090 for typical DiT layer sizes.

    Requirements:
    - CUDA device with fp8 tensor core support (SM89+)
    - Input trailing dimension divisible by 16
    """
    layer.original_forward = layer.forward  # type: ignore[attr-defined]

    # Mutable cache for scale tensors stored as module attribute (not closure var)
    # so it can be cleared externally via invalidate_scaled_mm_caches() after LoRA
    # fusion updates _weight_scale. Populated lazily on first forward call because
    # models are often patched on CPU then moved to CUDA via model.to().
    layer._smm_cache = {}  # type: ignore[attr-defined]

    def new_forward(*args, **_kwargs) -> torch.Tensor:
        x = args[0]

        # Guard: weight may not be fp8 (e.g. after LoRA fusion converts fp8->bf16,
        # or non-quantized layers in mixed models). Fall back to standard linear.
        if layer.weight.dtype != torch.float8_e4m3fn:
            w = layer.weight.to(x.dtype)
            ws = getattr(layer, "_weight_scale", None)
            if ws is not None:
                w = w * ws.to(x.dtype)
            b = layer.bias.to(x.dtype) if layer.bias is not None else None
            return torch.nn.functional.linear(x, w, b)

        # Lazy-init scales on first call (now on the correct device).
        # Reads _weight_scale fresh (not captured at patch time) so that
        # cache invalidation after LoRA fusion picks up the new scale.
        cache = layer._smm_cache
        if not cache:
            dev = x.device
            cache["one"] = torch.tensor(1.0, device=dev, dtype=torch.float32)
            cache["fp8_max"] = torch.tensor(
                torch.finfo(torch.float8_e4m3fn).max, device=dev, dtype=torch.float32
            )
            ws = getattr(layer, "_weight_scale", None)
            if ws is not None:
                cache["b"] = ws.to(device=dev, dtype=torch.float32)
            else:
                cache["b"] = cache["one"]

        x_2d = x.reshape(-1, x.shape[-1])

        # Dynamic per-tensor activation scaling: scale x to fill fp8 range
        # without clamping. Without this, values > fp8_max (~448) are clipped,
        # producing NaN/garbage after multi-layer residual accumulation.
        x_abs_max = x_2d.abs().amax().clamp(min=1e-12)
        x_scale = cache["fp8_max"] / x_abs_max
        x_fp8 = (x_2d * x_scale).to(torch.float8_e4m3fn)
        scale_a = x_scale.reciprocal()

        # weight.T is a view with stride(0)==1 (column-major) -- no copy
        result = torch._scaled_mm(
            x_fp8, layer.weight.T,
            scale_a=scale_a, scale_b=cache["b"],
            out_dtype=x.dtype,
        )

        if x.ndim > 2:
            result = result.reshape(x.shape[:-1] + (result.shape[-1],))

        if layer.bias is not None:
            result = result + layer.bias.to(x.dtype)

        return result

    layer.forward = new_forward  # type: ignore[assignment]


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
    """Patch all nn.Linear layers in model for fp8 inference.

    Prefers native FP8 tensor core matmul via torch._scaled_mm when available
    (CUDA, SM89+). Falls back to bf16 upcast otherwise.

    Norms and embeddings are skipped by default -- they're numerically sensitive
    and tiny compared to linear layers.

    Args:
        model: Model with fp8 (float8_e4m3fn) weights on its nn.Linear layers.
        skip_patterns: Module name substrings to skip (case-insensitive).

    Returns:
        Number of linear layers patched.
    """
    use_smm = is_scaled_mm_available() and not _FORCE_UPCAST
    patch_fn = _replace_fwd_with_scaled_mm if use_smm else _replace_fwd_with_upcast
    method = "scaled_mm" if use_smm else "upcast"
    if _FORCE_UPCAST and is_scaled_mm_available():
        logger.info("fp8-cast: LLM_DIT_FORCE_FP8_UPCAST=1 -- forcing upcast path")

    count = 0
    scaled = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        name_lower = name.lower()
        if any(p in name_lower for p in skip_patterns):
            continue
        patch_fn(module)
        count += 1
        if hasattr(module, "_weight_scale"):
            scaled += 1

    scale_info = f" ({scaled} with weight_scale, {count - scaled} without)" if scaled else ""
    logger.info(f"fp8-cast: patched {count} nn.Linear layers via {method}{scale_info}")
    return count


def _attach_weight_scales(
    model: nn.Module,
    weight_scales: dict[str, torch.Tensor],
) -> int:
    """Attach per-tensor weight scales to nn.Linear modules as plain attributes.

    Scales are stored as plain attributes (not buffers/parameters) so they don't
    appear in state_dict(). This keeps the cache path clean -- weight_scales are
    stored separately in the cache dict and re-attached during reconstruction.

    Args:
        model: Model with nn.Linear layers.
        weight_scales: Dict mapping weight param names (e.g.
            "transformer_blocks.0.attn1.to_q.weight") to scale tensors.

    Returns:
        Number of scales attached.
    """
    if not weight_scales:
        return 0

    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        weight_key = f"{name}.weight"
        if weight_key in weight_scales:
            module._weight_scale = weight_scales[weight_key]  # type: ignore[attr-defined]
            count += 1

    if count != len(weight_scales):
        logger.warning(
            f"Weight scale mismatch: {len(weight_scales)} scales provided, "
            f"{count} attached to nn.Linear modules"
        )

    return count


def invalidate_scaled_mm_caches(model: nn.Module) -> int:
    """Clear cached scale tensors in scaled_mm forward closures.

    Must be called after modifying _weight_scale attributes (e.g., after LoRA
    fusion re-quantizes weights) so the next forward call re-initializes the
    cache with updated scales.

    Args:
        model: Model whose nn.Linear layers may have _smm_cache dicts.

    Returns:
        Number of caches cleared.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Linear) and hasattr(module, "_smm_cache"):
            module._smm_cache.clear()  # type: ignore[attr-defined]
            count += 1
    if count:
        logger.debug(f"Invalidated {count} scaled_mm caches")
    return count
