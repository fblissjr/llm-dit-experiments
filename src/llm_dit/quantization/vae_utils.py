"""
VAE quantization utilities.

VAE models contain Conv2d layers which require different handling than Linear layers.
Only INT8/8-bit quantization is recommended for VAE (FP8 Conv2d poorly supported).

Example:
    from llm_dit.quantization.vae_utils import quantize_vae

    vae = AutoencoderKL.from_pretrained(...)
    vae = quantize_vae(vae, method="int8")  # ~50% VRAM reduction
"""

import logging
from typing import Literal

import torch

logger = logging.getLogger(__name__)

VAEQuantMethod = Literal["none", "int8"]


def quantize_vae(
    vae: torch.nn.Module,
    method: VAEQuantMethod,
    quantize_encoder: bool = False,
    quantize_decoder: bool = True,
) -> torch.nn.Module:
    """
    Quantize VAE for VRAM savings.

    VAE has Conv2d layers, so only int8 is supported:
    - int8: TorchAO dynamic quantization (Conv2d + Linear)

    Args:
        vae: VAE model to quantize
        method: Quantization method ("none", "int8")
        quantize_encoder: Whether to quantize encoder (default: False, rarely needed)
        quantize_decoder: Whether to quantize decoder (default: True, used in generation)

    Returns:
        Quantized VAE model

    Note:
        - FP8 is NOT recommended for Conv2d layers (poor support)
        - 4-bit is NOT recommended (quality degradation in VAE)
        - Encoder quantization rarely helps (only used in img2img/inpainting)
    """
    if method == "none":
        return vae

    if method == "int8":
        return _quantize_vae_int8(vae, quantize_encoder, quantize_decoder)

    raise ValueError(
        f"Unknown VAE quantization method: {method}. "
        f"Valid options: none, int8"
    )


def _quantize_vae_int8(
    vae: torch.nn.Module,
    quantize_encoder: bool,
    quantize_decoder: bool,
) -> torch.nn.Module:
    """Apply TorchAO INT8 dynamic quantization to VAE."""
    try:
        import torch.ao.quantization as tq
    except ImportError:
        logger.warning("torch.ao.quantization not available, skipping VAE quantization")
        return vae

    # Count parameters before
    params_before = sum(p.numel() for p in vae.parameters())

    if quantize_decoder and hasattr(vae, "decoder"):
        logger.info("Quantizing VAE decoder to INT8...")
        vae.decoder = tq.quantize_dynamic(
            vae.decoder,
            {torch.nn.Conv2d, torch.nn.Linear},
            dtype=torch.qint8,
        )

    if quantize_encoder and hasattr(vae, "encoder"):
        logger.info("Quantizing VAE encoder to INT8...")
        vae.encoder = tq.quantize_dynamic(
            vae.encoder,
            {torch.nn.Conv2d, torch.nn.Linear},
            dtype=torch.qint8,
        )

    # Estimate savings (rough, actual depends on layer types)
    logger.info(
        f"VAE quantization complete. "
        f"Estimated VRAM: ~{params_before * 2 / 1e6:.0f}MB -> ~{params_before / 1e6:.0f}MB"
    )

    return vae


def estimate_vae_vram(quantization: VAEQuantMethod) -> int:
    """
    Estimate VAE VRAM usage in MB.

    Args:
        quantization: Quantization method

    Returns:
        Estimated VRAM in MB
    """
    # Typical VAE sizes (approximate)
    base_vram = 500  # ~500MB for SDXL-style VAE

    if quantization == "none":
        return base_vram
    elif quantization == "int8":
        return base_vram // 2  # ~50% reduction
    else:
        return base_vram


def get_vae_quant_info(method: VAEQuantMethod) -> dict:
    """
    Get information about a VAE quantization method.

    Args:
        method: Quantization method

    Returns:
        Dict with method info
    """
    info = {
        "none": {
            "name": "No quantization",
            "vram_reduction": "0%",
            "quality": "100%",
            "supported_layers": "N/A",
        },
        "int8": {
            "name": "TorchAO INT8 Dynamic",
            "vram_reduction": "~50%",
            "quality": "99%+",
            "supported_layers": "Conv2d, Linear",
        },
    }
    return info.get(method, info["none"])
