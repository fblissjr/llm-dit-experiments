"""
Z-Image DiT Models - Pure PyTorch Implementation.

Last updated: 2026-02-01

This module provides pure PyTorch implementations of the Z-Image
text-to-image model components:

- ZImageDiT: Main S3-DiT transformer
- ZImageTransformerBlock: Single transformer block with AdaLN
- FluxVAEEncoder/FluxVAEDecoder: VAE components (see vae submodule)

The diffusers-based implementations are preserved in
`llm_dit.pipelines.diffusers` for backward compatibility.

Usage:
    from llm_dit.models.z_image import load_z_image_transformer
    from llm_dit.models.z_image.vae import load_z_image_vae

    # Load transformer
    transformer = load_z_image_transformer("models/Z-Image-Turbo")

    # Load VAE
    vae_encoder, vae_decoder = load_z_image_vae("models/Z-Image-Turbo")
"""

from llm_dit.layers import RMSNorm
from .transformer import ZImageDiT, ZImageTransformerBlock
from .components import (
    FeedForward,
    FinalLayer,
    TimestepEmbedder,
)
from .attention import Attention
from .rope import RopeEmbedder, apply_rotary_emb
from .constants import ZImageConfig, FluxVAEConfig, ADALN_EMBED_DIM, SEQ_MULTI_OF

__all__ = [
    # Main model
    "ZImageDiT",
    "ZImageTransformerBlock",
    # Components
    "FeedForward",
    "FinalLayer",
    "RMSNorm",
    "TimestepEmbedder",
    "Attention",
    # RoPE
    "RopeEmbedder",
    "apply_rotary_emb",
    # Config
    "ZImageConfig",
    "FluxVAEConfig",
    "ADALN_EMBED_DIM",
    "SEQ_MULTI_OF",
]

# Lazy import for loader (requires safetensors)
def load_z_image_transformer(*args, **kwargs):
    """Load Z-Image transformer from checkpoint. See loader.py for details."""
    from .loader import load_z_image_transformer as _load
    return _load(*args, **kwargs)
