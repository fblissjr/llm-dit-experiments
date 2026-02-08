"""
Weight loading utilities for Z-Image VAE.

Last updated: 2026-01-29

Re-exports the VAE loading function from the parent module's loader.

Usage:
    from llm_dit.models.z_image.vae import load_z_image_vae

    encoder, decoder = load_z_image_vae("models/Z-Image-Turbo")
"""

# Re-export from parent module
from ..loader import load_z_image_vae

__all__ = ["load_z_image_vae"]
