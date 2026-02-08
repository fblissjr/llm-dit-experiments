"""
Z-Image DiT architecture constants.

Last updated: 2026-01-29

These constants define the Z-Image S3-DiT architecture parameters.
Based on DiffSynth-Studio reference and diffusers model configs.
"""

# AdaLN embedding dimension (used for timestep modulation)
ADALN_EMBED_DIM = 256

# Sequence padding multiple (for efficient batching/attention)
SEQ_MULTI_OF = 32

# Padding dimension for empty image tokens (Omni mode)
X_PAD_DIM = 64


# Default model configurations
class ZImageConfig:
    """Configuration for Z-Image DiT models."""

    # Base variant (full model, ~6B params)
    BASE = {
        "in_channels": 16,  # VAE latent channels
        "dim": 3840,  # Hidden dimension
        "n_layers": 30,  # Main transformer layers
        "n_refiner_layers": 2,  # Refiner layers for noise/context
        "n_heads": 30,  # Attention heads
        "n_kv_heads": 30,  # KV heads (same as n_heads = MHA)
        "norm_eps": 1e-5,
        "qk_norm": True,
        "cap_feat_dim": 2560,  # Caption feature dim (Qwen3-4B hidden)
        "rope_theta": 256.0,  # RoPE theta
        "t_scale": 1000.0,  # Timestep scaling
        "axes_dims": [32, 48, 48],  # RoPE dimensions per axis
        "axes_lens": [1024, 512, 512],  # Max positions per axis
        "patch_size": 2,
        "f_patch_size": 1,  # Frame patch size (1 for image)
    }

    # Turbo variant (same architecture, distilled)
    TURBO = {
        **BASE,
        # Turbo uses same architecture but different weights
    }

    # Omni variant (with SigLIP for vision conditioning)
    OMNI = {
        **BASE,
        "siglip_feat_dim": 1152,  # SigLIP feature dimension
    }


# VAE configuration (Flux VAE)
class FluxVAEConfig:
    """Configuration for Flux VAE used by Z-Image."""

    DECODER = {
        "in_channels": 16,
        "out_channels": 3,
        "block_out_channels": [128, 256, 512, 512],
        "scaling_factor": 0.3611,
        "shift_factor": 0.1159,
    }

    ENCODER = {
        "in_channels": 3,
        "out_channels": 16,  # Returns 32, but first 16 are mean
        "block_out_channels": [128, 256, 512, 512],
        "scaling_factor": 0.3611,
        "shift_factor": 0.1159,
    }
