"""
FLUX.2 Klein Model Components.

Last Updated: 2026-01-23

Pure PyTorch implementation of the FLUX.2 Klein diffusion transformer for
image generation. This module provides granular access to all components
for research experiments.

Architecture Overview:
- Double→Single stream: 8 double-stream blocks merge into 24 single-stream blocks
- 32 attention heads with 128-dim head (4096 inner dim for 9B)
- 12288-dim text conditioning (3 x Qwen3-8B layers) for 9B
- 4D RoPE position embeddings (t, h, w, l)
- Shared AdaLN modulation (computed once, shared across blocks)
- Guidance-distilled for fast 4-step generation

Primary Exports:
    Flux2Transformer: Main transformer model
    Klein9BParams, Klein4BParams: Model configurations
    load_flux2_transformer: Load from HF checkpoints
    Flux2VAE: VAE for latent ↔ pixel conversion

Components:
    transformer.py: Main transformer model (Flux2, DoubleStreamBlock, SingleStreamBlock)
    constants.py: Klein9BParams, Klein4BParams, generation defaults
    rope.py: 4D rotary position embeddings
    vae.py: AutoEncoder with patchify/BatchNorm
    loader.py: Weight loading with FP8 support
"""

# Constants and parameters
from llm_dit.models.flux2.constants import (
    # Parameter classes
    Flux2Params,
    Klein9BParams,
    Klein4BParams,
    # Generation defaults
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_NUM_STEPS_DISTILLED,
    DEFAULT_GUIDANCE_DISTILLED,
    DEFAULT_NUM_STEPS_BASE,
    DEFAULT_GUIDANCE_BASE,
    DEFAULT_SEED,
    # VAE config
    VAE_Z_CHANNELS,
    TOTAL_SPATIAL_COMPRESSION,
    LATENT_CHANNELS_AFTER_PATCHIFY,
    # Model registry
    FLUX2_MODEL_INFO,
    # Helpers
    get_model_params,
    get_generation_defaults,
    calculate_latent_shape,
    calculate_num_tokens,
)

# RoPE components
from llm_dit.models.flux2.rope import (
    EmbedND,
    rope,
    apply_rope,
    attention,
    create_image_ids,
    create_text_ids,
    create_reference_ids,
)

# VAE components
from llm_dit.models.flux2.vae import (
    AutoEncoder,
    AutoEncoderParams,
    Flux2VAE,
)

# Transformer components
from llm_dit.models.flux2.transformer import (
    Flux2Transformer,
    Flux2,
    DoubleStreamBlock,
    SingleStreamBlock,
    Modulation,
    LastLayer,
    QKNorm,
    RMSNorm,
    timestep_embedding,
)

# Loader utilities
from llm_dit.models.flux2.loader import (
    load_flux2_transformer,
    load_flux2_vae,
    get_model_info,
    list_available_models,
)

__all__ = [
    # Parameter classes
    "Flux2Params",
    "Klein9BParams",
    "Klein4BParams",
    # Generation defaults
    "DEFAULT_WIDTH",
    "DEFAULT_HEIGHT",
    "DEFAULT_NUM_STEPS_DISTILLED",
    "DEFAULT_GUIDANCE_DISTILLED",
    "DEFAULT_NUM_STEPS_BASE",
    "DEFAULT_GUIDANCE_BASE",
    "DEFAULT_SEED",
    # VAE config
    "VAE_Z_CHANNELS",
    "TOTAL_SPATIAL_COMPRESSION",
    "LATENT_CHANNELS_AFTER_PATCHIFY",
    # Model registry
    "FLUX2_MODEL_INFO",
    # Helper functions
    "get_model_params",
    "get_generation_defaults",
    "calculate_latent_shape",
    "calculate_num_tokens",
    # RoPE
    "EmbedND",
    "rope",
    "apply_rope",
    "attention",
    "create_image_ids",
    "create_text_ids",
    "create_reference_ids",
    # VAE
    "AutoEncoder",
    "AutoEncoderParams",
    "Flux2VAE",
    # Transformer
    "Flux2Transformer",
    "Flux2",
    "DoubleStreamBlock",
    "SingleStreamBlock",
    "Modulation",
    "LastLayer",
    "QKNorm",
    "RMSNorm",
    "timestep_embedding",
    # Loader
    "load_flux2_transformer",
    "load_flux2_vae",
    "get_model_info",
    "list_available_models",
]
