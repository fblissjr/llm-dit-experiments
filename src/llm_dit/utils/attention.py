"""
Priority-based attention backend selection.

Provides unified attention interface that automatically selects the best
available backend for the current hardware. Supports:
- Flash Attention 3 (Hopper GPUs - H100, etc.)
- Flash Attention 2 (Ampere+ - RTX 3090, 4090, A100, etc.)
- Sage Attention (multiple variants optimized for different GPUs)
  - sage_int8_fp16: INT8 Q/K, FP16 P/V (SM80+, works on RTX 4090)
  - sage_int8_fp8_sm90: INT8 Q/K, FP8 P/V (SM90+, H100 only)
  - sage: Basic fallback
- xFormers
- PyTorch SDPA (fallback, always available)

Based on DiffSynth-Studio implementation (Apache 2.0 license).

Usage:
    from llm_dit.utils.attention import attention_forward, get_attention_backend

    # Check what's available
    print(get_available_backends())  # ['flash_attn_2', 'sage_int8_fp16', 'sdpa']

    # Use unified interface
    out = attention_forward(q, k, v)  # Uses best available

    # Force specific backend
    set_attention_backend("sage_int8_fp16")
    out = attention_forward(q, k, v)

    # Or via environment variable
    # LLM_DIT_ATTENTION=sage_int8_fp16 python script.py
"""

import logging
import os
from functools import lru_cache
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Type for attention backend names
# Priority order: FA3 > FA2 > sage_int8_fp8_sm90 > sage_int8_fp16 > sage > xformers > sdpa
AttentionBackend = Literal[
    "flash_attn_3",
    "flash_attn_2",
    "sage_int8_fp8_sm90",  # SM90+ (H100), INT8 Q/K + FP8 P/V
    "sage_int8_fp16",      # SM80+ (RTX 4090), INT8 Q/K + FP16 P/V
    "sage",                # Basic sage attention
    "xformers",
    "sdpa"
]

# Global state
_ATTENTION_IMPL: Optional[AttentionBackend] = None
_AVAILABLE_BACKENDS: Optional[list[AttentionBackend]] = None


@lru_cache(maxsize=1)
def get_cuda_capability() -> Tuple[int, int]:
    """
    Get CUDA compute capability of the current device.

    Returns:
        Tuple of (major, minor) version, e.g. (8, 9) for RTX 4090.
        Returns (0, 0) if no CUDA device available.
    """
    if not torch.cuda.is_available():
        return (0, 0)

    device = torch.cuda.current_device()
    return torch.cuda.get_device_capability(device)


def is_sm_at_least(major: int, minor: int = 0) -> bool:
    """Check if current GPU is at least SM{major}.{minor}."""
    cap = get_cuda_capability()
    return cap >= (major, minor)


def get_available_backends() -> list[AttentionBackend]:
    """
    Detect available attention backends in priority order.

    Returns:
        List of available backend names, best first.

    Priority order:
    1. Flash Attention 3 (H100/Hopper)
    2. Flash Attention 2 (Ampere+)
    3. Sage INT8/FP8 SM90 (H100 only)
    4. Sage INT8/FP16 (RTX 4090 / SM80+) - best for memory efficiency on consumer GPUs
    5. Sage basic
    6. xFormers
    7. SDPA (always available)
    """
    global _AVAILABLE_BACKENDS

    if _AVAILABLE_BACKENDS is not None:
        return _AVAILABLE_BACKENDS

    available: list[AttentionBackend] = []

    # Flash Attention 3 (Hopper GPUs - H100, etc.)
    try:
        from flash_attn_interface import flash_attn_func  # noqa: F401

        available.append("flash_attn_3")
        logger.debug("Flash Attention 3 available")
    except ImportError:
        pass

    # Flash Attention 2 (Ampere+ - RTX 3090, 4090, A100, etc.)
    try:
        from flash_attn import flash_attn_func  # noqa: F401

        available.append("flash_attn_2")
        logger.debug("Flash Attention 2 available")
    except ImportError:
        pass

    # Sage Attention variants - check in priority order
    try:
        # Check for advanced Sage variants
        from sageattention import sageattn  # noqa: F401

        # Try SM90+ FP8 variant (H100)
        if is_sm_at_least(9, 0):
            try:
                from sageattention import sageattn_qk_int8_pv_fp8_cuda_sm90  # noqa: F401

                available.append("sage_int8_fp8_sm90")
                logger.debug("Sage Attention INT8/FP8 SM90 available")
            except ImportError:
                pass

        # Try SM80+ INT8/FP16 variant (RTX 4090, A100, etc.)
        if is_sm_at_least(8, 0):
            try:
                from sageattention import sageattn_qk_int8_pv_fp16_cuda  # noqa: F401

                available.append("sage_int8_fp16")
                logger.debug("Sage Attention INT8/FP16 available (best for RTX 4090)")
            except ImportError:
                pass

        # Basic sage attention (fallback)
        available.append("sage")
        logger.debug("Sage Attention (basic) available")

    except ImportError:
        pass

    # xFormers
    try:
        from xformers.ops import memory_efficient_attention  # noqa: F401

        available.append("xformers")
        logger.debug("xFormers available")
    except ImportError:
        pass

    # PyTorch SDPA (always available in PyTorch 2.0+)
    available.append("sdpa")

    _AVAILABLE_BACKENDS = available
    return available


def get_attention_backend() -> AttentionBackend:
    """
    Get the current attention backend.

    Priority:
    1. Explicit set via set_attention_backend()
    2. Environment variable LLM_DIT_ATTENTION
    3. Best available backend

    Returns:
        Name of the attention backend to use.
    """
    global _ATTENTION_IMPL

    if _ATTENTION_IMPL is not None:
        return _ATTENTION_IMPL

    # Check environment override
    env_impl = os.environ.get("LLM_DIT_ATTENTION")
    if env_impl:
        available = get_available_backends()
        if env_impl in available:
            _ATTENTION_IMPL = env_impl  # type: ignore
            logger.info(f"Using attention backend from environment: {env_impl}")
            return _ATTENTION_IMPL
        else:
            logger.warning(
                f"Requested backend '{env_impl}' not available. "
                f"Available: {available}. Using best available."
            )

    # Use best available
    available = get_available_backends()
    _ATTENTION_IMPL = available[0]
    logger.info(f"Using attention backend: {_ATTENTION_IMPL}")
    return _ATTENTION_IMPL


def set_attention_backend(backend: AttentionBackend) -> None:
    """
    Set the attention backend explicitly.

    Args:
        backend: Backend name to use.

    Raises:
        ValueError: If backend is not available.
    """
    global _ATTENTION_IMPL

    available = get_available_backends()
    if backend not in available:
        raise ValueError(
            f"Backend '{backend}' not available. "
            f"Available backends: {available}"
        )

    _ATTENTION_IMPL = backend
    logger.info(f"Attention backend set to: {backend}")


def reset_attention_backend() -> None:
    """Reset attention backend to auto-detect."""
    global _ATTENTION_IMPL
    _ATTENTION_IMPL = None


def attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Unified attention forward using the selected backend.

    Args:
        q: Query tensor, shape (B, H, S, D) or (B, S, H, D)
        k: Key tensor, same shape as q
        v: Value tensor, same shape as q
        mask: Optional attention mask
        dropout_p: Dropout probability (training only)
        scale: Optional scale factor (default: 1/sqrt(D))
        is_causal: Whether to apply causal masking

    Returns:
        Output tensor, same shape as input.

    Note:
        - If mask is provided, falls back to SDPA (other backends may not support)
        - Input shape is auto-detected and output matches input shape
    """
    backend = get_attention_backend()

    # Fall back to SDPA if mask provided (other backends may not support)
    if mask is not None and backend not in ("sdpa", "xformers"):
        logger.debug(f"Mask provided, falling back to SDPA (was {backend})")
        backend = "sdpa"

    # Detect input format: (B, H, S, D) vs (B, S, H, D)
    # Flash Attention and xFormers expect (B, S, H, D)
    # SDPA and Sage expect (B, H, S, D)
    # We assume input is (B, H, S, D) and transpose as needed

    if backend == "flash_attn_3":
        return _flash_attn_3_forward(q, k, v, dropout_p, scale, is_causal)
    elif backend == "flash_attn_2":
        return _flash_attn_2_forward(q, k, v, dropout_p, scale, is_causal)
    elif backend == "sage_int8_fp8_sm90":
        return _sage_int8_fp8_sm90_forward(q, k, v, scale, is_causal)
    elif backend == "sage_int8_fp16":
        return _sage_int8_fp16_forward(q, k, v, scale, is_causal)
    elif backend == "sage":
        return _sage_forward(q, k, v, scale, is_causal)
    elif backend == "xformers":
        return _xformers_forward(q, k, v, mask, dropout_p, scale)
    else:  # sdpa
        return _sdpa_forward(q, k, v, mask, dropout_p, scale, is_causal)


def _flash_attn_2_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Flash Attention 2 forward pass."""
    from flash_attn import flash_attn_func

    # FA2 expects (B, S, H, D), we have (B, H, S, D)
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()

    # FA2 uses softmax_scale instead of scale
    softmax_scale = scale if scale is not None else (q.shape[-1] ** -0.5)

    out = flash_attn_func(
        q, k, v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=is_causal,
    )

    # Back to (B, H, S, D)
    return out.transpose(1, 2)


def _flash_attn_3_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Flash Attention 3 forward pass."""
    from flash_attn_interface import flash_attn_func

    # FA3 expects (B, S, H, D)
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()

    softmax_scale = scale if scale is not None else (q.shape[-1] ** -0.5)

    out = flash_attn_func(
        q, k, v,
        softmax_scale=softmax_scale,
        causal=is_causal,
    )

    return out.transpose(1, 2)


def _sage_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Sage Attention forward pass (basic variant)."""
    from sageattention import sageattn

    # Sage expects (B, H, S, D) - same as our format
    out = sageattn(
        q, k, v,
        is_causal=is_causal,
        sm_scale=scale,
    )

    return out


def _sage_int8_fp16_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Sage Attention INT8/FP16 forward pass.

    Uses INT8 quantization for Q/K and FP16 for P/V.
    Requires SM80+ (RTX 4090, A100, etc.) - best for memory efficiency on consumer GPUs.
    """
    from sageattention import sageattn_qk_int8_pv_fp16_cuda

    # Sage expects (B, H, S, D) - same as our format
    out = sageattn_qk_int8_pv_fp16_cuda(
        q, k, v,
        is_causal=is_causal,
        sm_scale=scale,
    )

    return out


def _sage_int8_fp8_sm90_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Sage Attention INT8/FP8 SM90 forward pass.

    Uses INT8 quantization for Q/K and FP8 for P/V.
    Requires SM90+ (H100, etc.) - maximum memory efficiency on Hopper GPUs.
    """
    from sageattention import sageattn_qk_int8_pv_fp8_cuda_sm90

    # Sage expects (B, H, S, D) - same as our format
    out = sageattn_qk_int8_pv_fp8_cuda_sm90(
        q, k, v,
        is_causal=is_causal,
        sm_scale=scale,
    )

    return out


def _xformers_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """xFormers memory efficient attention forward pass."""
    from xformers.ops import memory_efficient_attention

    # xFormers expects (B, S, H, D)
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()

    out = memory_efficient_attention(
        q, k, v,
        attn_bias=mask,
        p=dropout_p,
        scale=scale,
    )

    return out.transpose(1, 2)


def _sdpa_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """PyTorch scaled dot product attention forward pass."""
    # SDPA expects (B, H, S, D) - same as our format
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=mask,
        dropout_p=dropout_p,
        scale=scale,
        is_causal=is_causal,
    )


def log_attention_info() -> None:
    """Log information about available attention backends."""
    available = get_available_backends()
    current = get_attention_backend()
    cuda_cap = get_cuda_capability()

    logger.info("Attention backend configuration:")
    logger.info(f"  CUDA capability: SM{cuda_cap[0]}.{cuda_cap[1]}")
    logger.info(f"  Available: {available}")
    logger.info(f"  Current: {current}")

    if current == "flash_attn_3":
        logger.info("  Flash Attention 3 - fastest on Hopper GPUs (H100)")
    elif current == "flash_attn_2":
        logger.info("  Flash Attention 2 provides ~2x speedup on RTX 4090")
    elif current == "sage_int8_fp8_sm90":
        logger.info("  Sage INT8/FP8 SM90 - maximum memory efficiency on H100")
    elif current == "sage_int8_fp16":
        logger.info("  Sage INT8/FP16 - excellent memory efficiency on RTX 4090 (SM80+)")
        logger.info("  Uses INT8 quantization for Q/K, FP16 for P/V")
    elif current == "sage":
        logger.info("  Sage Attention (basic) - good balance of speed and memory")
    elif current == "xformers":
        logger.info("  xFormers memory efficient attention")
    elif current == "sdpa":
        logger.info("  Using PyTorch SDPA (install flash-attn or sageattention for better performance)")
        logger.info("  Install: pip install flash-attn --no-build-isolation")
        logger.info("  Or: pip install sageattention")
