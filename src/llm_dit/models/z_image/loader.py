"""
Weight loading utilities for Z-Image transformer and VAE.

Last updated: 2026-01-29

Provides functions for loading official Z-Image checkpoints into our pure PyTorch
implementation. Handles both diffusers sharded format and single safetensors files.

Supports:
- Diffusers sharded format (models/Z-Image-Turbo/transformer/)
- Single file safetensors
- Config-based variant selection (Turbo vs Base)

Usage:
    from llm_dit.models.z_image import load_z_image_transformer
    from llm_dit.models.z_image.vae import load_z_image_vae

    # Load transformer
    transformer = load_z_image_transformer("models/Z-Image-Turbo")

    # Load VAE
    vae_encoder, vae_decoder = load_z_image_vae("models/Z-Image-Turbo")
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch

from .transformer import ZImageDiT
from .vae.decoder import FluxVAEDecoder
from .vae.encoder import FluxVAEEncoder

logger = logging.getLogger(__name__)


# Key mapping from diffusers format to our implementation
# The Z-Image checkpoint uses the same naming convention as our implementation,
# so no mapping is needed. This dict is here for future compatibility.
DIFFUSERS_TO_OURS = {
    # Keys match exactly, no mapping needed
    # ".norm_q.": ".norm_q.",  # Same
    # ".norm_k.": ".norm_k.",  # Same
}


def map_key(diffusers_key: str) -> str:
    """
    Map a diffusers state dict key to our naming convention.

    Args:
        diffusers_key: Key from diffusers checkpoint

    Returns:
        Mapped key for our implementation
    """
    key = diffusers_key

    for old, new in DIFFUSERS_TO_OURS.items():
        key = key.replace(old, new)

    return key


def load_safetensors(path: Path, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Load state dict from safetensors file(s).

    Handles both single files and sharded checkpoints.

    Args:
        path: Path to safetensors file or directory containing shards
        device: Device to load tensors to

    Returns:
        State dictionary
    """
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("safetensors required. Install: pip install safetensors")

    state_dict = {}

    if path.is_file():
        # Single file
        with safe_open(str(path), framework="pt", device=device) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    else:
        # Sharded - look for index file
        index_path = path / "diffusion_pytorch_model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"No index file found at {index_path}")

        with open(index_path) as f:
            index = json.load(f)

        # Get unique shard files
        shard_files = set(index["weight_map"].values())

        for shard_name in shard_files:
            shard_path = path / shard_name
            with safe_open(str(shard_path), framework="pt", device=device) as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)

    return state_dict


def load_config(path: Path) -> dict:
    """
    Load model configuration from checkpoint directory.

    Args:
        path: Path to checkpoint directory

    Returns:
        Configuration dictionary
    """
    config_path = path / "config.json" if path.is_dir() else path.parent / "config.json"

    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)

    # Default config if no config file (Z-Image-Turbo defaults)
    return {
        "in_channels": 16,
        "dim": 3840,
        "n_layers": 30,
        "n_refiner_layers": 2,
        "n_heads": 30,
        "n_kv_heads": 30,
        "norm_eps": 1e-5,
        "qk_norm": True,
        "cap_feat_dim": 2560,
        "rope_theta": 256.0,
        "t_scale": 1000.0,
        "axes_dims": [32, 48, 48],
        "axes_lens": [1536, 512, 512],
        "all_patch_size": [2],
        "all_f_patch_size": [1],
    }


def create_model_from_config(config: dict, dtype: torch.dtype = torch.bfloat16) -> ZImageDiT:
    """
    Create ZImageDiT from config dictionary.

    Args:
        config: Model configuration
        dtype: Model dtype

    Returns:
        Initialized model (random weights)
    """
    model = ZImageDiT(
        all_patch_size=tuple(config.get("all_patch_size", [2])),
        all_f_patch_size=tuple(config.get("all_f_patch_size", [1])),
        in_channels=config.get("in_channels", 16),
        dim=config.get("dim", 3840),
        n_layers=config.get("n_layers", 30),
        n_refiner_layers=config.get("n_refiner_layers", 2),
        n_heads=config.get("n_heads", 30),
        n_kv_heads=config.get("n_kv_heads", 30),
        norm_eps=config.get("norm_eps", 1e-5),
        qk_norm=config.get("qk_norm", True),
        cap_feat_dim=config.get("cap_feat_dim", 2560),
        rope_theta=config.get("rope_theta", 256.0),
        t_scale=config.get("t_scale", 1000.0),
        axes_dims=config.get("axes_dims", [32, 48, 48]),
        axes_lens=config.get("axes_lens", [1536, 512, 512]),
        siglip_feat_dim=config.get("siglip_feat_dim", None),
    )

    return model.to(dtype)


def load_z_image_transformer(
    path: Union[str, Path],
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    variant: Optional[str] = None,
    strict: bool = False,
) -> ZImageDiT:
    """
    Load Z-Image transformer from checkpoint.

    Loads weights from official Z-Image checkpoints and maps them to our
    pure PyTorch implementation.

    Args:
        path: Path to model directory (e.g., "models/Z-Image-Turbo")
        dtype: Model dtype (bf16 recommended)
        device: Device to load to initially (use 'cpu' then .to('cuda') for large models)
        variant: Model variant ("turbo" or "base"). If None, auto-detected from path.
        strict: If True, raise error on missing/extra keys

    Returns:
        Loaded ZImageDiT model

    Example:
        # Load transformer
        model = load_z_image_transformer("models/Z-Image-Turbo")
        model = model.cuda()  # Move to GPU after loading

        # Load specific variant
        model = load_z_image_transformer("models/Z-Image-Base", variant="base")
    """
    path = Path(path)

    # Determine transformer directory
    if (path / "transformer").exists():
        transformer_path = path / "transformer"
    else:
        transformer_path = path

    # Load config
    config = load_config(transformer_path)
    logger.info(
        f"Loaded config: {config.get('n_layers', 30)} layers, "
        f"{config.get('n_heads', 30)} heads, dim={config.get('dim', 3840)}"
    )

    # Create model
    model = create_model_from_config(config, dtype)

    # Load weights
    logger.info(f"Loading weights from {transformer_path}")
    state_dict = load_safetensors(transformer_path, device=device)

    # Map keys (currently no mapping needed)
    our_state_dict = {}
    for key, tensor in state_dict.items():
        our_key = map_key(key)
        our_state_dict[our_key] = tensor.to(dtype)

    # Load into model
    load_result = model.load_state_dict(our_state_dict, strict=strict)

    if load_result.missing_keys:
        logger.warning(f"Missing keys: {load_result.missing_keys[:10]}...")
    if load_result.unexpected_keys:
        logger.warning(
            f"Unexpected keys: {load_result.unexpected_keys[:10]}... "
            f"({len(load_result.unexpected_keys)} total)"
        )

    logger.info(f"Loaded Z-Image transformer: {model.get_num_params() / 1e9:.2f}B parameters")

    return model


def load_z_image_vae(
    path: Union[str, Path],
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    strict: bool = False,
) -> Tuple[FluxVAEEncoder, FluxVAEDecoder]:
    """
    Load Z-Image VAE (encoder and decoder) from checkpoint.

    Args:
        path: Path to model directory (e.g., "models/Z-Image-Turbo")
        dtype: Model dtype (bf16 recommended)
        device: Device to load to initially
        strict: If True, raise error on missing/extra keys

    Returns:
        Tuple of (FluxVAEEncoder, FluxVAEDecoder)

    Example:
        encoder, decoder = load_z_image_vae("models/Z-Image-Turbo")
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    """
    path = Path(path)

    # Determine VAE directory
    if (path / "vae").exists():
        vae_path = path / "vae"
    else:
        vae_path = path

    # Find safetensors file
    safetensors_file = vae_path / "diffusion_pytorch_model.safetensors"
    if not safetensors_file.exists():
        raise FileNotFoundError(f"VAE checkpoint not found at {safetensors_file}")

    # Load weights
    logger.info(f"Loading VAE weights from {safetensors_file}")
    state_dict = load_safetensors(safetensors_file, device=device)

    # Create models
    encoder = FluxVAEEncoder().to(dtype)
    decoder = FluxVAEDecoder().to(dtype)

    # Split weights into encoder and decoder
    encoder_state_dict = {}
    decoder_state_dict = {}

    for key, tensor in state_dict.items():
        tensor = tensor.to(dtype)

        if key.startswith("encoder."):
            # Remove "encoder." prefix and map to our naming
            new_key = key[8:]  # Remove "encoder."
            encoder_state_dict[new_key] = tensor
        elif key.startswith("decoder."):
            # Remove "decoder." prefix and map to our naming
            new_key = key[8:]  # Remove "decoder."
            decoder_state_dict[new_key] = tensor
        else:
            # Shared weights (like quant_conv, post_quant_conv)
            # Z-Image VAE doesn't use these, so skip
            pass

    # Load encoder weights (may have missing keys due to architecture differences)
    if encoder_state_dict:
        enc_result = encoder.load_state_dict(encoder_state_dict, strict=False)
        if enc_result.missing_keys and strict:
            logger.warning(f"Encoder missing keys: {enc_result.missing_keys[:10]}...")

    # Load decoder weights
    if decoder_state_dict:
        dec_result = decoder.load_state_dict(decoder_state_dict, strict=False)
        if dec_result.missing_keys and strict:
            logger.warning(f"Decoder missing keys: {dec_result.missing_keys[:10]}...")

    logger.info("Loaded Z-Image VAE (encoder + decoder)")

    return encoder, decoder


def get_model_info(path: Union[str, Path]) -> dict:
    """
    Get information about a checkpoint without loading it.

    Args:
        path: Path to checkpoint

    Returns:
        Dict with model info (num_params, dtype, config, etc.)
    """
    path = Path(path)

    # Determine transformer directory
    if (path / "transformer").exists():
        transformer_path = path / "transformer"
    else:
        transformer_path = path

    config = load_config(transformer_path)

    # Calculate parameter count based on config
    dim = config.get("dim", 3840)
    n_layers = config.get("n_layers", 30)
    n_refiner = config.get("n_refiner_layers", 2)
    cap_dim = config.get("cap_feat_dim", 2560)

    # Rough estimate: attention + FFN per layer
    # attention: 4 * dim^2 (Q, K, V, O projections)
    # ffn: 3 * dim * (8/3 * dim) (SwiGLU)
    params_per_layer = 4 * dim * dim + 3 * dim * int(dim * 8 / 3)
    total_params = n_layers * params_per_layer + n_refiner * 2 * params_per_layer

    # Add embedders
    total_params += dim * cap_dim  # cap_embedder
    total_params += dim * 16 * 4  # x_embedder (patch_size=2)

    return {
        "config": config,
        "num_layers": n_layers,
        "n_refiner_layers": n_refiner,
        "hidden_dim": dim,
        "cap_feat_dim": cap_dim,
        "estimated_params": total_params,
        "estimated_size_bf16_gb": total_params * 2 / 1e9,
    }
