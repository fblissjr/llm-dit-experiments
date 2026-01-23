"""
FLUX.2 Klein model constants and parameter configurations.

Last Updated: 2026-01-23

Defines parameter dataclasses for FLUX.2 Klein models (4B and 9B variants),
as well as generation defaults and VAE configuration.

Ported from: coderef/flux2/src/flux2/model.py

Usage:
    from llm_dit.models.flux2.constants import Klein9BParams, Klein4BParams

    # Get default parameters for 9B model
    params = Klein9BParams()
    print(params.hidden_size)  # 4096
"""

from dataclasses import dataclass, field


@dataclass
class Flux2Params:
    """
    Parameters for full FLUX.2 model (27B, with Mistral encoder).

    This is the largest FLUX.2 variant using Mistral-Small-3.2 for text encoding.
    Outputs context_in_dim of 15360 (3 layers x 5120 dim).
    """
    in_channels: int = 128  # After 2x2 patchify: 32 * 4 = 128
    context_in_dim: int = 15360  # 3 x 5120 (Mistral hidden dim)
    hidden_size: int = 6144
    num_heads: int = 48
    depth: int = 8  # Double-stream blocks
    depth_single_blocks: int = 48  # Single-stream blocks
    axes_dim: list[int] = field(default_factory=lambda: [32, 32, 32, 32])  # 4D RoPE
    theta: int = 2000  # RoPE base frequency
    mlp_ratio: float = 3.0  # FFN expansion ratio
    use_guidance_embed: bool = True  # Whether to use guidance embedding


@dataclass
class Klein9BParams:
    """
    Parameters for FLUX.2 Klein 9B model.

    Uses Qwen3-8B for text encoding (3 layers x 4096 dim = 12288).
    Guidance-distilled variant uses 4 steps with CFG=1.0.

    Architecture:
        - 8 double-stream blocks (joint attention, separate img/txt)
        - 24 single-stream blocks (merged attention)
        - 4096 hidden dimension, 32 attention heads
        - 4D RoPE with theta=2000
    """
    in_channels: int = 128  # After 2x2 patchify: 32 * 4 = 128
    context_in_dim: int = 12288  # 3 x 4096 (Qwen3-8B hidden dim)
    hidden_size: int = 4096
    num_heads: int = 32
    depth: int = 8  # Double-stream blocks
    depth_single_blocks: int = 24  # Single-stream blocks
    axes_dim: list[int] = field(default_factory=lambda: [32, 32, 32, 32])  # 4D RoPE
    theta: int = 2000  # RoPE base frequency (different from LTX-2's 10000)
    mlp_ratio: float = 3.0  # FFN expansion ratio
    use_guidance_embed: bool = False  # Distilled model: no guidance embedding


@dataclass
class Klein4BParams:
    """
    Parameters for FLUX.2 Klein 4B model.

    Uses Qwen3-4B for text encoding (3 layers x 2560 dim = 7680).
    Smallest Klein variant, suitable for consumer GPUs.

    Architecture:
        - 5 double-stream blocks
        - 20 single-stream blocks
        - 3072 hidden dimension, 24 attention heads
    """
    in_channels: int = 128  # After 2x2 patchify
    context_in_dim: int = 7680  # 3 x 2560 (Qwen3-4B hidden dim)
    hidden_size: int = 3072
    num_heads: int = 24
    depth: int = 5  # Double-stream blocks
    depth_single_blocks: int = 20  # Single-stream blocks
    axes_dim: list[int] = field(default_factory=lambda: [24, 24, 24, 24])  # 4D RoPE (smaller head dim)
    theta: int = 2000  # RoPE base frequency
    mlp_ratio: float = 3.0  # FFN expansion ratio
    use_guidance_embed: bool = False  # Distilled model: no guidance embedding


# =============================================================================
# Generation Defaults
# =============================================================================

# Default image dimensions (16x compression: 1024 -> 64 latent)
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024

# Guidance-distilled defaults (Klein models)
DEFAULT_NUM_STEPS_DISTILLED = 4
DEFAULT_GUIDANCE_DISTILLED = 1.0  # CFG baked into distilled models

# Base model defaults (non-distilled)
DEFAULT_NUM_STEPS_BASE = 50
DEFAULT_GUIDANCE_BASE = 4.0

# Seed
DEFAULT_SEED = 42


# =============================================================================
# VAE Configuration
# =============================================================================

# VAE parameters (from AutoEncoderParams)
VAE_RESOLUTION = 256  # Reference resolution
VAE_IN_CHANNELS = 3  # RGB input
VAE_CH = 128  # Base channel count
VAE_OUT_CH = 3  # RGB output
VAE_CH_MULT = [1, 2, 4, 4]  # Channel multipliers per level
VAE_NUM_RES_BLOCKS = 2  # Residual blocks per level
VAE_Z_CHANNELS = 32  # Latent channels before patchify

# Compression factors
VAE_SPATIAL_COMPRESSION = 8  # 8x spatial compression from encoder
PATCHIFY_COMPRESSION = 2  # 2x spatial compression from patchify
TOTAL_SPATIAL_COMPRESSION = 16  # Total: 8 * 2 = 16x

# Latent dimensions after patchify
LATENT_CHANNELS_AFTER_PATCHIFY = 128  # 32 * 2 * 2 = 128


# =============================================================================
# Model Registry Info
# =============================================================================

FLUX2_MODEL_INFO = {
    "klein-4b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-4B",
        "filename": "flux-2-klein-4b.safetensors",
        "filename_ae": "ae.safetensors",
        "params_cls": Klein4BParams,
        "text_encoder": "Qwen/Qwen3-4B-FP8",
        "distilled": True,
        "defaults": {"guidance": 1.0, "num_steps": 4},
    },
    "klein-9b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-9B",
        "filename": "flux-2-klein-9b.safetensors",
        "filename_ae": "ae.safetensors",
        "params_cls": Klein9BParams,
        "text_encoder": "Qwen/Qwen3-8B-FP8",
        "distilled": True,
        "defaults": {"guidance": 1.0, "num_steps": 4},
    },
    "klein-base-4b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-base-4B",
        "filename": "flux-2-klein-base-4b.safetensors",
        "filename_ae": "ae.safetensors",
        "params_cls": Klein4BParams,
        "text_encoder": "Qwen/Qwen3-4B-FP8",
        "distilled": False,
        "defaults": {"guidance": 4.0, "num_steps": 50},
    },
    "klein-base-9b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-base-9B",
        "filename": "flux-2-klein-base-9b.safetensors",
        "filename_ae": "ae.safetensors",
        "params_cls": Klein9BParams,
        "text_encoder": "Qwen/Qwen3-8B-FP8",
        "distilled": False,
        "defaults": {"guidance": 4.0, "num_steps": 50},
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_model_params(model_name: str):
    """
    Get parameter dataclass for a model variant.

    Args:
        model_name: Model variant name (e.g., "klein-9b")

    Returns:
        Instantiated parameter dataclass
    """
    model_name = model_name.lower()
    if model_name not in FLUX2_MODEL_INFO:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(FLUX2_MODEL_INFO.keys())}"
        )
    return FLUX2_MODEL_INFO[model_name]["params_cls"]()


def get_generation_defaults(model_name: str) -> dict:
    """
    Get default generation parameters for a model.

    Args:
        model_name: Model variant name

    Returns:
        Dict with guidance and num_steps defaults
    """
    model_name = model_name.lower()
    if model_name not in FLUX2_MODEL_INFO:
        raise ValueError(f"Unknown model: {model_name}")
    return FLUX2_MODEL_INFO[model_name]["defaults"].copy()


def calculate_latent_shape(height: int, width: int) -> tuple[int, int, int]:
    """
    Calculate latent dimensions for a given image size.

    Args:
        height: Image height in pixels
        width: Image width in pixels

    Returns:
        Tuple of (channels, latent_height, latent_width)
    """
    # Total compression is 16x (8x VAE + 2x patchify)
    latent_h = height // TOTAL_SPATIAL_COMPRESSION
    latent_w = width // TOTAL_SPATIAL_COMPRESSION
    return (LATENT_CHANNELS_AFTER_PATCHIFY, latent_h, latent_w)


def calculate_num_tokens(height: int, width: int) -> int:
    """
    Calculate number of latent tokens for an image.

    Args:
        height: Image height in pixels
        width: Image width in pixels

    Returns:
        Number of tokens (latent_h * latent_w)
    """
    _, latent_h, latent_w = calculate_latent_shape(height, width)
    return latent_h * latent_w
