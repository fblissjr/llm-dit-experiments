"""
Z-Image VAE Models - Pure PyTorch Implementation.

Last updated: 2026-01-29

This module provides pure PyTorch implementations of the Flux VAE
used by Z-Image for latent encoding/decoding:

- FluxVAEEncoder: Image → 16-channel latents
- FluxVAEDecoder: 16-channel latents → Image

Usage:
    from llm_dit.models.z_image.vae import load_z_image_vae

    # Load VAE components
    vae_encoder, vae_decoder = load_z_image_vae("models/Z-Image-Turbo")

    # Encode image to latents
    latents = vae_encoder.encode(image)  # (B, 16, H/8, W/8)

    # Decode latents to image
    image = vae_decoder.decode(latents)  # (B, 3, H, W)
"""

from .decoder import FluxVAEDecoder
from .encoder import FluxVAEEncoder

__all__ = [
    "FluxVAEEncoder",
    "FluxVAEDecoder",
]


# Lazy import for loader (requires safetensors)
def load_z_image_vae(*args, **kwargs):
    """Load Z-Image VAE from checkpoint. See loader.py for details."""
    from .loader import load_z_image_vae as _load
    return _load(*args, **kwargs)
