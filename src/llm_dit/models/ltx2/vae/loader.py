"""
LTX-2.3 VAE Loader - Weight loading utilities for VideoDecoder.

Last Updated: 2026-03-06

Loads official LTX-2.3 VAE checkpoints into our pure PyTorch VideoDecoder.
V2.3 checkpoints use native key format (decoder.up_blocks.N.res_blocks.M)
and have a different architecture from V1 (9 up_blocks vs 7).

Usage:
    from llm_dit.models.ltx2.vae import load_ltx2_vae_decoder

    # Load from V2.3 split checkpoint
    decoder = load_ltx2_vae_decoder("models/LTX-2.3/ltx-2.3-video-vae.safetensors")
"""

import logging
from pathlib import Path
from typing import Dict, Union

import torch

from .enums import NormLayerType, PaddingModeType
from .video_vae import VideoDecoder

logger = logging.getLogger(__name__)


# V2.3 decoder_blocks specification (encoder/forward order, reversed for up_blocks).
# Inferred from checkpoint weight shapes:
#   up_blocks.0: res_x(2)@1024 -> up_blocks.1: compress_all(mult=2) 1024->512
#   up_blocks.2: res_x(2)@512  -> up_blocks.3: compress_all(mult=1) 512->512
#   up_blocks.4: res_x(4)@512  -> up_blocks.5: compress_time(mult=2) 512->256
#   up_blocks.6: res_x(6)@256  -> up_blocks.7: compress_space(mult=2) 256->128
#   up_blocks.8: res_x(4)@128
V23_DECODER_BLOCKS = [
    ("res_x", {"num_layers": 4}),
    ("compress_space", {"multiplier": 2}),
    ("res_x", {"num_layers": 6}),
    ("compress_time", {"multiplier": 2}),
    ("res_x", {"num_layers": 4}),
    ("compress_all", {"multiplier": 1}),
    ("res_x", {"num_layers": 2}),
    ("compress_all", {"multiplier": 2}),
    ("res_x", {"num_layers": 2}),
]


def _load_safetensors(path: Path, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Load state dict from safetensors file."""
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("safetensors required. Install: pip install safetensors")

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


def load_ltx2_vae_decoder(
    path: Union[str, Path],
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    strict: bool = False,
) -> VideoDecoder:
    """
    Load LTX-2.3 VAE decoder from checkpoint.

    V2.3 checkpoints use native key format -- keys map directly after
    stripping the 'decoder.' prefix. Architecture: 9 up_blocks with
    mixed compress_all/compress_time/compress_space upsamplers.

    Args:
        path: Path to safetensors file or directory.
        dtype: Model dtype (bf16 recommended).
        device: Device to load to.
        strict: If True, raise error on missing/extra keys.

    Returns:
        Loaded VideoDecoder model.
    """
    path = Path(path)

    # Create V2.3 decoder model
    decoder = VideoDecoder(
        convolution_dimensions=3,
        in_channels=128,
        out_channels=3,
        decoder_blocks=V23_DECODER_BLOCKS,
        patch_size=4,
        norm_layer=NormLayerType.PIXEL_NORM,
        causal=False,
        timestep_conditioning=False,
        decoder_spatial_padding_mode=PaddingModeType.REFLECT,
    )

    # Load weights
    logger.info(f"Loading VAE weights from {path}")
    raw_sd = _load_safetensors(path, device=device)

    # Map keys: V2.3 uses native format, just strip 'decoder.' prefix
    our_state_dict = {}
    skipped_keys = []

    for raw_key, tensor in raw_sd.items():
        # per_channel_statistics: direct mapping
        if raw_key.startswith("per_channel_statistics."):
            our_state_dict[raw_key] = tensor.to(dtype)
            continue

        # Skip encoder keys
        if raw_key.startswith("encoder."):
            skipped_keys.append(raw_key)
            continue

        # Strip 'decoder.' prefix for decoder keys
        if raw_key.startswith("decoder."):
            our_key = raw_key[len("decoder."):]
            our_state_dict[our_key] = tensor.to(dtype)
        else:
            skipped_keys.append(raw_key)

    # Load into model
    load_result = decoder.load_state_dict(our_state_dict, strict=strict)

    if skipped_keys:
        logger.info(f"Skipped {len(skipped_keys)} non-decoder keys")

    if load_result.missing_keys:
        relevant_missing = [k for k in load_result.missing_keys
                          if not k.startswith("per_channel_statistics.")]
        if relevant_missing:
            logger.warning(f"Missing keys: {relevant_missing[:10]}...")

    if load_result.unexpected_keys:
        logger.warning(
            f"Unexpected keys (ignored): {load_result.unexpected_keys[:10]}... "
            f"({len(load_result.unexpected_keys)} total)"
        )

    # Validate
    num_params = sum(p.numel() for p in decoder.parameters())
    logger.info(f"Loaded VAE decoder: {num_params / 1e6:.1f}M parameters")

    std_buffer = decoder.per_channel_statistics.get_buffer("std-of-means")
    mean_buffer = decoder.per_channel_statistics.get_buffer("mean-of-means")
    if std_buffer.abs().max() < 1e-6:
        logger.warning(
            "PerChannelStatistics std-of-means buffer appears empty! "
            "Latent denormalization may not work correctly."
        )
    logger.info(
        f"PerChannelStatistics loaded: "
        f"std-of-means range=[{std_buffer.min():.4f}, {std_buffer.max():.4f}], "
        f"mean-of-means range=[{mean_buffer.min():.4f}, {mean_buffer.max():.4f}]"
    )

    return decoder.to(dtype)
