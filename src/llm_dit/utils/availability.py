"""
Centralized dependency availability checks.

Last Updated: 2026-01-29

Provides cached availability checks for optional dependencies to avoid
duplicated try/except blocks scattered throughout the codebase.

Usage:
    from llm_dit.utils.availability import (
        is_torchao_available,
        is_flash_attn_available,
        check_diffusers_version,
        check_fp8_support,
    )

    if is_torchao_available():
        from torchao.quantization import int8_weight_only, quantize_
        quantize_(model, int8_weight_only())
    else:
        logger.warning("torchao not available, using bf16")
"""

import logging
from functools import lru_cache
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def is_torchao_available() -> bool:
    """
    Check if torchao is available for quantization.

    torchao provides:
    - int4_weight_only() for ~4-bit quantization
    - int8_weight_only() for ~8-bit quantization
    - quantize_() for in-place model quantization

    Returns:
        True if torchao is importable, False otherwise.
    """
    try:
        from torchao.quantization import int8_weight_only, quantize_

        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def is_flash_attn_available() -> bool:
    """
    Check if flash_attn is available.

    Flash Attention 2 provides memory-efficient attention for Ampere+ GPUs.

    Returns:
        True if flash_attn is importable, False otherwise.
    """
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def get_flash_attn_version() -> Optional[str]:
    """
    Get the installed flash_attn version.

    Returns:
        Version string if installed, None otherwise.
    """
    try:
        import flash_attn

        return getattr(flash_attn, "__version__", "unknown")
    except ImportError:
        return None


@lru_cache(maxsize=1)
def is_diffusers_available() -> bool:
    """
    Check if diffusers is available.

    Returns:
        True if diffusers is importable, False otherwise.
    """
    try:
        import diffusers  # noqa: F401

        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def get_diffusers_version() -> Optional[str]:
    """
    Get the installed diffusers version.

    Returns:
        Version string if installed, None otherwise.
    """
    try:
        import diffusers

        return diffusers.__version__
    except ImportError:
        return None


def check_diffusers_version(min_version: str) -> bool:
    """
    Check if installed diffusers version meets minimum requirement.

    Args:
        min_version: Minimum version string (e.g., "0.32.0")

    Returns:
        True if diffusers >= min_version, False otherwise.

    Example:
        if check_diffusers_version("0.32.0"):
            from diffusers import LTX2Transformer3DModel
    """
    version = get_diffusers_version()
    if version is None:
        return False

    try:
        from packaging import version as pkg_version

        return pkg_version.parse(version) >= pkg_version.parse(min_version)
    except ImportError:
        # Fallback: simple string comparison (works for semver)
        return version >= min_version


@lru_cache(maxsize=1)
def check_fp8_support() -> bool:
    """
    Check if the current GPU supports FP8 computation.

    FP8 (8-bit floating point) requires:
    - CUDA compute capability >= 8.9 (Ada Lovelace / Hopper)
    - torch >= 2.1 for FP8 dtypes

    Returns:
        True if FP8 is supported, False otherwise.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False

        # Check compute capability (need SM89+ for FP8)
        capability = torch.cuda.get_device_capability()
        if capability < (8, 9):
            return False

        # Check if FP8 dtypes exist
        return hasattr(torch, "float8_e4m3fn") and hasattr(torch, "float8_e5m2")

    except Exception:
        return False


@lru_cache(maxsize=1)
def get_cuda_capability() -> Tuple[int, int]:
    """
    Get CUDA compute capability of the current device.

    Returns:
        Tuple of (major, minor) version, e.g. (8, 9) for RTX 4090.
        Returns (0, 0) if no CUDA device available.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return (0, 0)

        device = torch.cuda.current_device()
        return torch.cuda.get_device_capability(device)
    except Exception:
        return (0, 0)


@lru_cache(maxsize=1)
def is_xformers_available() -> bool:
    """
    Check if xformers is available.

    xformers provides memory-efficient attention for various GPU architectures.

    Returns:
        True if xformers is importable, False otherwise.
    """
    try:
        import xformers  # noqa: F401

        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def is_sage_attn_available() -> bool:
    """
    Check if SageAttention is available.

    SageAttention provides optimized INT8/FP8 attention kernels.

    Returns:
        True if sageattention is importable, False otherwise.
    """
    try:
        import sageattention  # noqa: F401

        return True
    except ImportError:
        return False


def log_availability_status() -> None:
    """Log the availability status of all optional dependencies."""
    status = {
        "torchao": is_torchao_available(),
        "flash_attn": is_flash_attn_available(),
        "diffusers": is_diffusers_available(),
        "xformers": is_xformers_available(),
        "sage_attn": is_sage_attn_available(),
        "fp8_support": check_fp8_support(),
    }

    logger.info("Dependency availability:")
    for name, available in status.items():
        status_str = "✓" if available else "✗"
        logger.info(f"  {status_str} {name}")

    if is_diffusers_available():
        logger.info(f"  diffusers version: {get_diffusers_version()}")

    if is_flash_attn_available():
        logger.info(f"  flash_attn version: {get_flash_attn_version()}")

    capability = get_cuda_capability()
    logger.info(f"  CUDA capability: SM{capability[0]}.{capability[1]}")
