"""
DiffSynth-style FP8 inference utilities.

Provides runtime FP8 inference via F.linear patching and torch._scaled_mm.
This is complementary to TorchAO's permanent quantization approach.

Key differences from TorchAO approach:
- TorchAO: Permanently quantizes weights via quantize_() - saves memory at rest
- DiffSynth: Runtime patching with context manager - easy enable/disable, AMD support

Usage:
    from llm_dit.quantization.fp8_inference import fp8_inference, enable_fp8_weights

    # Option 1: Runtime FP8 via context manager (weights stay in original dtype)
    with fp8_inference():
        output = model(input)

    # Option 2: Pre-convert weights to FP8 for memory savings
    enable_fp8_weights(model)
    with fp8_inference():
        output = model(input)

Based on: DiffSynth-Engine/diffsynth_engine/utils/fp8_linear.py

last updated: 2026-01-03
"""

import logging
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# Detect FP8 dtype based on hardware
def _get_fp8_dtype() -> torch.dtype:
    """
    Get the appropriate FP8 dtype for current hardware.

    AMD GPUs (gfx94x series) require float8_e4m3fnuz.
    NVIDIA GPUs use float8_e4m3fn.

    Returns:
        torch.float8_e4m3fn or torch.float8_e4m3fnuz
    """
    if not torch.cuda.is_available():
        return torch.float8_e4m3fn

    # Check for AMD ROCm
    if hasattr(torch.version, 'hip') and torch.version.hip:
        try:
            gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
            if "gfx94" in gcn_arch:
                # AMD MI300 series uses float8_e4m3fnuz
                return torch.float8_e4m3fnuz
        except Exception:
            pass

    return torch.float8_e4m3fn


# Module-level dtype (cached on first use)
_FP8_DTYPE: Optional[torch.dtype] = None


def get_fp8_dtype() -> torch.dtype:
    """Get the FP8 dtype for current hardware (cached)."""
    global _FP8_DTYPE
    if _FP8_DTYPE is None:
        _FP8_DTYPE = _get_fp8_dtype()
    return _FP8_DTYPE


def get_fp8_max() -> float:
    """
    Get the maximum representable value for current FP8 dtype.

    float8_e4m3fn: 448.0
    float8_e4m3fnuz: 224.0 (half of e4m3fn)

    Returns:
        Maximum value for FP8 dtype
    """
    dtype = get_fp8_dtype()
    if dtype == torch.float8_e4m3fnuz:
        return 224.0
    return 448.0


def enable_fp8_weights(module: nn.Module) -> None:
    """
    Convert all Linear layer weights to FP8 in-place.

    This saves memory but weights cannot be modified after conversion.
    Use in combination with fp8_inference() context manager.

    Args:
        module: PyTorch module to convert

    Example:
        enable_fp8_weights(model)  # Convert weights once
        with fp8_inference():
            for batch in dataloader:
                output = model(batch)  # All inference uses FP8
    """
    dtype = get_fp8_dtype()
    converted_count = 0

    for name, submodule in module.named_modules():
        if isinstance(submodule, nn.Linear):
            # Skip if already FP8 or not floating point (e.g., GGUF int)
            if submodule.weight.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
                continue
            if not torch.is_floating_point(submodule.weight):
                continue

            submodule.weight.data = submodule.weight.data.to(dtype)
            converted_count += 1

    logger.info(f"Converted {converted_count} Linear layers to FP8 ({dtype})")
    setattr(module, "_fp8_weights_enabled", True)


def _fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    FP8 linear operation using torch._scaled_mm.

    This function replaces F.linear during fp8_inference context.
    It performs dynamic per-tensor scaling for both input and weights.

    Args:
        input: Input tensor (any dtype, will be converted to FP8)
        weight: Weight tensor (any dtype, will be converted to FP8)
        bias: Optional bias tensor

    Returns:
        Output tensor in original input dtype
    """
    device = input.device
    origin_dtype = input.dtype
    origin_shape = input.shape

    # Flatten to 2D for matmul
    input = input.reshape(-1, origin_shape[-1])

    # Get FP8 parameters
    fp8_dtype = get_fp8_dtype()
    fp8_max = get_fp8_max()

    # Compute dynamic scale for input based on max value
    # Scale = max(1.0, |x|_max / fp8_max)
    x_max = torch.max(torch.abs(input), dim=-1, keepdim=True).values
    scale_a = torch.clamp(x_max / fp8_max, min=1.0).float().to(device=device)

    # Weight scale is 1.0 (assuming weights are already normalized or will be scaled)
    scale_b = torch.ones((weight.shape[0], 1), dtype=torch.float32, device=device)

    # Scale and convert to FP8
    input_fp8 = (input / scale_a).to(fp8_dtype)
    weight_fp8 = weight.to(fp8_dtype)

    # Perform scaled matrix multiplication
    # result = (input_fp8 @ weight_fp8.T) * scale_a * scale_b
    result = torch._scaled_mm(
        input_fp8,
        weight_fp8.T,
        scale_a=scale_a,
        scale_b=scale_b.T,
        bias=bias,
        out_dtype=origin_dtype,
    )

    # Restore original shape
    new_shape = origin_shape[:-1] + (result.shape[-1],)
    result = result.reshape(new_shape)

    return result


@contextmanager
def fp8_inference(enabled: bool = True):
    """
    Context manager for FP8 inference via F.linear patching.

    When enabled, replaces F.linear with an FP8 implementation that uses
    torch._scaled_mm for matrix multiplication. This provides:
    - Dynamic per-tensor scaling for numerical stability
    - Automatic dtype conversion to/from FP8
    - AMD ROCm compatibility (uses float8_e4m3fnuz on MI300)

    Args:
        enabled: If False, context manager is a no-op

    Example:
        # Basic usage
        with fp8_inference():
            output = model(input)

        # Conditional usage
        with fp8_inference(enabled=use_fp8):
            output = model(input)

        # Pre-convert weights for memory savings
        enable_fp8_weights(model)
        with fp8_inference():
            output = model(input)

    Note:
        This patches F.linear globally, so all Linear layers in all models
        will use FP8 during the context. Nested contexts are safe.
    """
    if not enabled:
        yield
        return

    # Store original F.linear
    origin_linear = F.linear

    # Patch F.linear with FP8 version
    F.linear = _fp8_linear

    try:
        yield
    finally:
        # Restore original F.linear
        F.linear = origin_linear


def check_fp8_available() -> bool:
    """
    Check if FP8 inference is available on current hardware.

    Requires:
    - CUDA device with compute capability 8.9+ (RTX 4090, H100)
    - Or AMD MI300 with ROCm

    Returns:
        True if FP8 is available, False otherwise
    """
    if not torch.cuda.is_available():
        return False

    # Check compute capability for NVIDIA
    if not (hasattr(torch.version, 'hip') and torch.version.hip):
        major, minor = torch.cuda.get_device_capability()
        # FP8 requires 8.9+ (Ada Lovelace) or 9.0+ (Hopper)
        return (major * 10 + minor) >= 89

    # For AMD, check architecture
    try:
        gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
        # MI300 series (gfx94x) supports FP8
        return "gfx94" in gcn_arch
    except Exception:
        return False


def get_fp8_info() -> dict:
    """
    Get information about FP8 support on current hardware.

    Returns:
        Dict with FP8 support information:
        - available: Whether FP8 is available
        - dtype: The FP8 dtype that will be used
        - max_value: Maximum representable value
        - platform: "nvidia", "amd", or "cpu"
        - device_name: GPU name if available
    """
    info = {
        "available": check_fp8_available(),
        "dtype": str(get_fp8_dtype()),
        "max_value": get_fp8_max(),
        "platform": "cpu",
        "device_name": None,
    }

    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        if hasattr(torch.version, 'hip') and torch.version.hip:
            info["platform"] = "amd"
        else:
            info["platform"] = "nvidia"

    return info


# Autocast-style hooks for models with pre-converted FP8 weights
def _fp8_autocast_pre_hook(module: nn.Module, inputs):
    """Pre-hook to convert FP8 weights to compute dtype before forward."""
    compute_dtype = getattr(module, "_fp8_compute_dtype", torch.bfloat16)

    for name, param in module.named_parameters(recurse=False):
        if param.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            param.data = param.data.to(compute_dtype)

    # Convert inputs
    new_inputs = []
    for x in inputs:
        if isinstance(x, torch.Tensor) and x.dtype in (
            torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float16, torch.bfloat16
        ):
            new_inputs.append(x.to(compute_dtype))
        else:
            new_inputs.append(x)

    return tuple(new_inputs)


def _fp8_autocast_post_hook(module: nn.Module, inputs, outputs):
    """Post-hook to convert weights back to FP8 after forward."""
    fp8_dtype = get_fp8_dtype()

    for name, param in module.named_parameters(recurse=False):
        compute_dtype = getattr(module, "_fp8_compute_dtype", torch.bfloat16)
        if param.dtype == compute_dtype:
            param.data = param.data.to(fp8_dtype)


def enable_fp8_autocast(
    module: nn.Module,
    compute_dtype: torch.dtype = torch.bfloat16,
    skip_linear_with_fp8_inference: bool = True,
) -> None:
    """
    Enable FP8 autocast hooks on a module.

    This is for models that have already been converted to FP8 weights
    via enable_fp8_weights(). The hooks automatically:
    1. Convert FP8 weights to compute dtype before forward
    2. Convert weights back to FP8 after forward

    This saves memory (weights stored in FP8) while computing in higher precision.

    Args:
        module: Module to add hooks to
        compute_dtype: Dtype to use for computation (default: bfloat16)
        skip_linear_with_fp8_inference: If True, skip Linear layers when
            fp8_inference() context will handle them

    Example:
        enable_fp8_weights(model)
        enable_fp8_autocast(model)
        output = model(input)  # Weights auto-convert for forward, back to FP8 after
    """
    def _add_hooks_recursive(mod: nn.Module):
        # Check if already hooked
        if getattr(mod, "_fp8_autocast_enabled", False):
            return

        # Skip Linear layers if fp8_inference will handle them
        if skip_linear_with_fp8_inference and isinstance(mod, nn.Linear):
            return

        has_params = len(list(mod.parameters(recurse=False))) > 0

        if has_params:
            mod._fp8_compute_dtype = compute_dtype
            mod.register_forward_pre_hook(_fp8_autocast_pre_hook)
            mod.register_forward_hook(_fp8_autocast_post_hook)
            mod._fp8_autocast_enabled = True

        for child in mod.children():
            _add_hooks_recursive(child)

    _add_hooks_recursive(module)
    logger.info(f"Enabled FP8 autocast with compute_dtype={compute_dtype}")
