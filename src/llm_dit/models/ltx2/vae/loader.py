"""
LTX-2 VAE Loader - Weight loading utilities for VideoDecoder.

Last Updated: 2026-01-19

Provides functions for loading official LTX-2 VAE checkpoints into our pure PyTorch
VideoDecoder implementation. Handles the key mapping between diffusers format and
our naming convention.

Usage:
    from llm_dit.models.ltx2.vae import load_ltx2_vae_decoder

    # Load from checkpoint directory
    decoder = load_ltx2_vae_decoder("models/LTX-2/vae/")

    # Load with specific dtype
    decoder = load_ltx2_vae_decoder(path, dtype=torch.bfloat16)
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Union

import torch

from .enums import NormLayerType, PaddingModeType
from .video_vae import VideoDecoder

logger = logging.getLogger(__name__)


def _load_safetensors(path: Path, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Load state dict from safetensors file.

    Args:
        path: Path to safetensors file or directory containing it
        device: Device to load tensors to

    Returns:
        State dictionary
    """
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("safetensors required. Install: pip install safetensors")

    # Handle both file and directory paths
    if path.is_dir():
        safetensors_path = path / "diffusion_pytorch_model.safetensors"
    else:
        safetensors_path = path

    if not safetensors_path.exists():
        raise FileNotFoundError(f"No safetensors file found at {safetensors_path}")

    state_dict = {}
    with safe_open(str(safetensors_path), framework="pt", device=device) as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    return state_dict


def _load_config(path: Path) -> dict:
    """
    Load VAE configuration from checkpoint directory.

    Args:
        path: Path to checkpoint directory

    Returns:
        Configuration dictionary
    """
    config_path = path / "config.json" if path.is_dir() else path.parent / "config.json"

    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)

    # Default config for LTX-2 VAE decoder
    return {
        "latent_channels": 128,
        "out_channels": 3,
        "patch_size": 4,
        "decoder_causal": False,
        "timestep_conditioning": False,
        "decoder_spatial_padding_mode": "reflect",
        "decoder_layers_per_block": [5, 5, 5, 5],
        "decoder_spatio_temporal_scaling": [True, True, True],
        "upsample_residual": [True, True, True],
    }


def _map_decoder_key(diffusers_key: str) -> str:
    """
    Map a diffusers VAE decoder key to our naming convention.

    Mapping structure:
        Diffusers                           → Ours
        decoder.conv_in.conv.*              → conv_in.conv.*
        decoder.mid_block.resnets.N.*       → up_blocks.0.res_blocks.N.*
        decoder.up_blocks.K.upsamplers.0.*  → up_blocks.{2K+1}.*
        decoder.up_blocks.K.resnets.N.*     → up_blocks.{2K+2}.res_blocks.N.*
        decoder.conv_out.conv.*             → conv_out.conv.*

    Args:
        diffusers_key: Key from diffusers checkpoint

    Returns:
        Mapped key for our implementation
    """
    key = diffusers_key

    # Strip 'decoder.' prefix
    if key.startswith("decoder."):
        key = key[8:]

    # Map mid_block.resnets → up_blocks.0.res_blocks
    if key.startswith("mid_block.resnets."):
        key = key.replace("mid_block.resnets.", "up_blocks.0.res_blocks.")
        return key

    # Map up_blocks.K.upsamplers.0 → up_blocks.{2K+1}
    # Map up_blocks.K.resnets.N → up_blocks.{2K+2}.res_blocks.N
    match = re.match(r"up_blocks\.(\d+)\.(upsamplers|resnets)\.", key)
    if match:
        block_idx = int(match.group(1))
        block_type = match.group(2)

        if block_type == "upsamplers":
            # upsamplers.0.conv.* → up_blocks.{2K+1}.*
            new_idx = 2 * block_idx + 1
            key = re.sub(
                r"up_blocks\.\d+\.upsamplers\.0\.",
                f"up_blocks.{new_idx}.",
                key
            )
        else:  # resnets
            # resnets.N.* → up_blocks.{2K+2}.res_blocks.N.*
            new_idx = 2 * block_idx + 2
            key = re.sub(
                r"up_blocks\.\d+\.resnets\.",
                f"up_blocks.{new_idx}.res_blocks.",
                key
            )

    return key


def _build_decoder_blocks(config: dict) -> list:
    """
    Build decoder_blocks list from config.

    The decoder architecture consists of:
    1. First res_x block (becomes mid_block in forward pass)
    2. Alternating: compress_all (upsample) + res_x blocks

    Args:
        config: VAE configuration dictionary

    Returns:
        List of (block_name, block_config) tuples
    """
    layers_per_block = config.get("decoder_layers_per_block", [5, 5, 5, 5])
    upsample_residual = config.get("upsample_residual", [True, True, True])

    decoder_blocks = []

    # First res_x block (mid_block)
    decoder_blocks.append(("res_x", {"num_layers": layers_per_block[0]}))

    # Alternating compress_all + res_x
    for residual, num_layers in zip(upsample_residual, layers_per_block[1:]):
        # Upsample block (compress_all because we're doing depth-to-space)
        decoder_blocks.append((
            "compress_all",
            {"residual": residual, "multiplier": 2}
        ))
        # Resnet block
        decoder_blocks.append(("res_x", {"num_layers": num_layers}))

    return decoder_blocks


def load_ltx2_vae_decoder(
    path: Union[str, Path],
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    strict: bool = False,
) -> VideoDecoder:
    """
    Load LTX-2 VAE decoder from official checkpoint.

    Loads weights from official LTX-2 VAE checkpoints and maps them to our
    pure PyTorch VideoDecoder implementation.

    Args:
        path: Path to checkpoint file or directory (e.g., "models/LTX-2/vae/")
        dtype: Model dtype (bf16 recommended)
        device: Device to load to initially (use 'cpu' then .to('cuda') for large models)
        strict: If True, raise error on missing/extra keys

    Returns:
        Loaded VideoDecoder model

    Example:
        # Load VAE decoder
        decoder = load_ltx2_vae_decoder("models/LTX-2/vae/")
        decoder = decoder.cuda()  # Move to GPU after loading

        # Decode latents
        latents = torch.randn(1, 128, 5, 16, 24)  # [B, C, F, H, W]
        video = decoder(latents)  # [B, 3, 33, 512, 768]
    """
    path = Path(path)

    # Load config
    config = _load_config(path)
    logger.info(f"Loading VAE decoder config: latent_channels={config.get('latent_channels', 128)}")

    # Build decoder blocks from config
    decoder_blocks = _build_decoder_blocks(config)

    # Parse padding mode
    padding_mode_str = config.get("decoder_spatial_padding_mode", "reflect")
    if padding_mode_str == "reflect":
        padding_mode = PaddingModeType.REFLECT
    elif padding_mode_str == "zeros":
        padding_mode = PaddingModeType.ZEROS
    else:
        padding_mode = PaddingModeType.REPLICATE

    # Create decoder model
    decoder = VideoDecoder(
        convolution_dimensions=3,
        in_channels=config.get("latent_channels", 128),
        out_channels=config.get("out_channels", 3),
        decoder_blocks=decoder_blocks,
        patch_size=config.get("patch_size", 4),
        norm_layer=NormLayerType.PIXEL_NORM,
        causal=config.get("decoder_causal", False),
        timestep_conditioning=config.get("timestep_conditioning", False),
        decoder_spatial_padding_mode=padding_mode,
    )

    # Load weights
    logger.info(f"Loading weights from {path}")
    diffusers_state_dict = _load_safetensors(path, device=device)

    # Map keys
    our_state_dict = {}
    skipped_keys = []

    for diffusers_key, tensor in diffusers_state_dict.items():
        # Handle latents_mean → per_channel_statistics.mean-of-means
        if diffusers_key == "latents_mean":
            our_state_dict["per_channel_statistics.mean-of-means"] = tensor.to(dtype)
            # Also copy to std-of-means for un_normalize to work correctly
            # Note: diffusers uses latents_std for the standard deviation
            continue

        # Handle latents_std → per_channel_statistics.std-of-means
        if diffusers_key == "latents_std":
            our_state_dict["per_channel_statistics.std-of-means"] = tensor.to(dtype)
            continue

        # Skip encoder keys
        if diffusers_key.startswith("encoder."):
            skipped_keys.append(diffusers_key)
            continue

        # Skip non-decoder keys
        if not diffusers_key.startswith("decoder."):
            skipped_keys.append(diffusers_key)
            continue

        our_key = _map_decoder_key(diffusers_key)
        our_state_dict[our_key] = tensor.to(dtype)

    # Load into model
    load_result = decoder.load_state_dict(our_state_dict, strict=strict)

    if skipped_keys:
        logger.info(f"Skipped {len(skipped_keys)} non-decoder keys")

    if load_result.missing_keys:
        # Filter out per_channel_statistics keys that we don't need
        relevant_missing = [k for k in load_result.missing_keys
                          if not k.startswith("per_channel_statistics.")]
        if relevant_missing:
            logger.warning(f"Missing keys: {relevant_missing[:10]}...")

    if load_result.unexpected_keys:
        logger.warning(
            f"Unexpected keys (ignored): {load_result.unexpected_keys[:10]}... "
            f"({len(load_result.unexpected_keys)} total)"
        )

    # Count parameters
    num_params = sum(p.numel() for p in decoder.parameters())
    logger.info(f"Loaded VAE decoder: {num_params / 1e6:.1f}M parameters")

    # Validate PerChannelStatistics buffers are loaded correctly
    std_buffer = decoder.per_channel_statistics.get_buffer("std-of-means")
    mean_buffer = decoder.per_channel_statistics.get_buffer("mean-of-means")

    # Check buffers are not empty/zero (would indicate failed loading)
    if std_buffer.abs().max() < 1e-6:
        logger.warning(
            "PerChannelStatistics std-of-means buffer appears empty! "
            "Latent denormalization may not work correctly."
        )
    if mean_buffer.abs().max() < 1e-6 and std_buffer.abs().max() > 1e-6:
        # mean-of-means can legitimately be near zero, only warn if std is also zero
        pass

    # Log buffer statistics for debugging
    logger.info(
        f"PerChannelStatistics loaded: "
        f"std-of-means range=[{std_buffer.min():.4f}, {std_buffer.max():.4f}], "
        f"mean-of-means range=[{mean_buffer.min():.4f}, {mean_buffer.max():.4f}]"
    )

    return decoder.to(dtype)
