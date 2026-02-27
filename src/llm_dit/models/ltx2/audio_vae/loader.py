"""
LTX-2 Audio VAE Loader - Weight Loading for AudioDecoder and Vocoder.

Last Updated: 2026-02-26

Loads official LTX-2 audio VAE and vocoder checkpoints into our pure
PyTorch implementations. Handles key mapping between diffusers format
and our naming convention.

Weight files:
    - models/LTX-2/audio_vae/diffusion_pytorch_model.safetensors (102MB)
    - models/LTX-2/vocoder/diffusion_pytorch_model.safetensors (107MB)

Key mappings:
    Audio VAE decoder keys map 1:1 (strip "decoder." prefix).
    Vocoder keys need remapping:
        conv_in    -> conv_pre
        conv_out   -> conv_post
        upsamplers -> ups
        resnets    -> resblocks

Usage:
    from llm_dit.models.ltx2.audio_vae import load_audio_decoder, load_vocoder

    decoder = load_audio_decoder("models/LTX-2/audio_vae/")
    vocoder = load_vocoder("models/LTX-2/vocoder/")
"""

import logging
from pathlib import Path
from typing import Dict, Union

import orjson
import torch

from .decoder import AudioDecoder
from .vocoder import Vocoder

logger = logging.getLogger(__name__)


def _load_safetensors(path: Path, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Load state dict from safetensors file."""
    from safetensors import safe_open

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
    """Load config.json from checkpoint directory."""
    config_path = path / "config.json" if path.is_dir() else path.parent / "config.json"

    if config_path.exists():
        with open(config_path, "rb") as f:
            return orjson.loads(f.read())

    return {}


# ---------------------------------------------------------------------------
# Audio VAE Decoder
# ---------------------------------------------------------------------------

def _map_decoder_key(diffusers_key: str) -> str:
    """Map a diffusers audio VAE decoder key to our naming.

    The checkpoint uses the same module hierarchy as our AudioDecoder,
    just with a "decoder." prefix that needs stripping.

    Checkpoint format:
        decoder.conv_in.conv.{weight,bias}
        decoder.mid.block_{1,2}.conv{1,2}.conv.{weight,bias}
        decoder.up.{level}.block.{idx}.conv{1,2}.conv.{weight,bias}
        decoder.up.{level}.block.{idx}.nin_shortcut.conv.{weight,bias}
        decoder.up.{level}.upsample.conv.conv.{weight,bias}
        decoder.conv_out.conv.{weight,bias}
    """
    if diffusers_key.startswith("decoder."):
        return diffusers_key[8:]  # Strip "decoder." prefix
    return diffusers_key


def load_audio_decoder(
    path: Union[str, Path],
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
) -> AudioDecoder:
    """Load LTX-2 audio VAE decoder from checkpoint.

    Args:
        path: Path to audio_vae checkpoint directory or safetensors file.
        dtype: Model dtype (bf16 recommended).
        device: Initial device for weight loading.

    Returns:
        Loaded AudioDecoder with per-channel statistics.
    """
    path = Path(path)
    config = _load_config(path)

    logger.info(
        f"Loading audio decoder: ch={config.get('base_channels', 128)}, "
        f"z_channels={config.get('latent_channels', 8)}, "
        f"mel_bins={config.get('mel_bins', 64)}"
    )

    # Parse config values
    ch_mult = tuple(config.get("ch_mult", [1, 2, 4]))
    attn_resolutions = set(config.get("attn_resolutions") or [])

    # Map norm_type string to enum
    norm_type_str = config.get("norm_type", "pixel")
    from ..vae.normalization import NormType
    norm_type = NormType.PIXEL if norm_type_str == "pixel" else NormType.GROUP

    # Map causality_axis string to enum
    from .blocks import CausalityAxis
    causality_str = config.get("causality_axis", "height")
    causality_map = {
        "height": CausalityAxis.HEIGHT,
        "width": CausalityAxis.WIDTH,
        "none": CausalityAxis.NONE,
        None: CausalityAxis.NONE,
    }
    causality_axis = causality_map.get(causality_str, CausalityAxis.HEIGHT)

    decoder = AudioDecoder(
        ch=config.get("base_channels", 128),
        out_ch=config.get("output_channels", config.get("in_channels", 2)),
        ch_mult=ch_mult,
        num_res_blocks=config.get("num_res_blocks", 2),
        attn_resolutions=attn_resolutions,
        resolution=config.get("resolution", 256),
        z_channels=config.get("latent_channels", 8),
        norm_type=norm_type,
        causality_axis=causality_axis,
        dropout=config.get("dropout", 0.0),
        mid_block_add_attention=config.get("mid_block_add_attention", False),
        sample_rate=config.get("sample_rate", 16000),
        mel_hop_length=config.get("mel_hop_length", 160),
        is_causal=config.get("is_causal", True),
        mel_bins=config.get("mel_bins", 64),
    )

    # Load weights
    diffusers_state_dict = _load_safetensors(path, device=device)

    our_state_dict = {}
    skipped_keys = []

    for key, tensor in diffusers_state_dict.items():
        # Map latents_mean/latents_std to per_channel_statistics buffers
        if key == "latents_mean":
            our_state_dict["per_channel_statistics.mean-of-means"] = tensor.to(dtype)
            continue
        if key == "latents_std":
            our_state_dict["per_channel_statistics.std-of-means"] = tensor.to(dtype)
            continue

        # Skip encoder keys
        if key.startswith("encoder."):
            skipped_keys.append(key)
            continue

        # Skip non-decoder keys (shouldn't happen but be safe)
        if not key.startswith("decoder."):
            skipped_keys.append(key)
            continue

        our_key = _map_decoder_key(key)
        our_state_dict[our_key] = tensor.to(dtype)

    load_result = decoder.load_state_dict(our_state_dict, strict=False)

    if skipped_keys:
        logger.info(f"Skipped {len(skipped_keys)} non-decoder keys (encoder, etc.)")

    if load_result.missing_keys:
        logger.warning(f"Missing keys: {load_result.missing_keys[:10]}...")

    if load_result.unexpected_keys:
        logger.warning(
            f"Unexpected keys: {load_result.unexpected_keys[:10]}... "
            f"({len(load_result.unexpected_keys)} total)"
        )

    # Validate per-channel statistics
    std_buffer = decoder.per_channel_statistics.get_buffer("std-of-means")
    mean_buffer = decoder.per_channel_statistics.get_buffer("mean-of-means")

    if std_buffer.abs().max() < 1e-6:
        logger.warning(
            "PerChannelStatistics std-of-means appears empty! "
            "Latent denormalization may not work correctly."
        )

    num_params = sum(p.numel() for p in decoder.parameters())
    logger.info(
        f"Loaded audio decoder: {num_params / 1e6:.1f}M params, "
        f"std range=[{std_buffer.min():.4f}, {std_buffer.max():.4f}], "
        f"mean range=[{mean_buffer.min():.4f}, {mean_buffer.max():.4f}]"
    )

    return decoder.to(dtype)


# ---------------------------------------------------------------------------
# Vocoder
# ---------------------------------------------------------------------------

# Vocoder checkpoint uses different names than our module
_VOCODER_KEY_MAP = {
    "conv_in.": "conv_pre.",
    "conv_out.": "conv_post.",
    "upsamplers.": "ups.",
    "resnets.": "resblocks.",
}


def _map_vocoder_key(checkpoint_key: str) -> str:
    """Map vocoder checkpoint key to our naming convention.

    Checkpoint format:
        conv_in.{weight,bias}         -> conv_pre.{weight,bias}
        conv_out.{weight,bias}        -> conv_post.{weight,bias}
        upsamplers.{idx}.{weight,bias} -> ups.{idx}.{weight,bias}
        resnets.{idx}.convs{1,2}.{j}.{weight,bias}
            -> resblocks.{idx}.convs{1,2}.{j}.{weight,bias}
    """
    key = checkpoint_key
    for old, new in _VOCODER_KEY_MAP.items():
        if key.startswith(old):
            key = new + key[len(old):]
            break
    return key


def load_vocoder(
    path: Union[str, Path],
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
) -> Vocoder:
    """Load LTX-2 vocoder from checkpoint.

    Args:
        path: Path to vocoder checkpoint directory or safetensors file.
        dtype: Model dtype (bf16 recommended).
        device: Initial device for weight loading.

    Returns:
        Loaded Vocoder model.
    """
    path = Path(path)
    config = _load_config(path)

    logger.info(
        f"Loading vocoder: hidden={config.get('hidden_channels', 1024)}, "
        f"upsample_factors={config.get('upsample_factors', [6,5,2,2,2])}, "
        f"output_rate={config.get('output_sampling_rate', 24000)}Hz"
    )

    vocoder = Vocoder(
        resblock_kernel_sizes=config.get("resnet_kernel_sizes", [3, 7, 11]),
        upsample_rates=config.get("upsample_factors", [6, 5, 2, 2, 2]),
        upsample_kernel_sizes=config.get("upsample_kernel_sizes", [16, 15, 8, 4, 4]),
        resblock_dilation_sizes=config.get("resnet_dilations", [[1, 3, 5]] * 3),
        upsample_initial_channel=config.get("hidden_channels", 1024),
        stereo=config.get("out_channels", 2) == 2,
        resblock="1",
        output_sample_rate=config.get("output_sampling_rate", 24000),
    )

    # Load weights
    checkpoint_state_dict = _load_safetensors(path, device=device)

    our_state_dict = {
        _map_vocoder_key(k): v.to(dtype)
        for k, v in checkpoint_state_dict.items()
    }

    load_result = vocoder.load_state_dict(our_state_dict, strict=True)

    if load_result.missing_keys:
        logger.warning(f"Missing vocoder keys: {load_result.missing_keys[:10]}...")
    if load_result.unexpected_keys:
        logger.warning(f"Unexpected vocoder keys: {load_result.unexpected_keys[:10]}...")

    num_params = sum(p.numel() for p in vocoder.parameters())
    logger.info(
        f"Loaded vocoder: {num_params / 1e6:.1f}M params, "
        f"upsample_factor={vocoder.upsample_factor}x"
    )

    return vocoder.to(dtype)
