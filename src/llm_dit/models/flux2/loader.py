"""
FLUX.2 Model Loading Utilities.

Last Updated: 2026-01-23

Provides functions to load FLUX.2 Klein models and VAE from HuggingFace
or local paths. Supports both BF16 and FP8 weights.

Note: FP8 model folders don't include configs - must get from non-FP8 folders.

Ported from: coderef/flux2/src/flux2/util.py

Usage:
    from llm_dit.models.flux2.loader import (
        load_flux2_transformer,
        load_flux2_vae,
        FLUX2_MODEL_INFO,
    )

    # Load Klein 9B model
    model = load_flux2_transformer("klein-9b", device="cuda")

    # Load VAE
    vae = load_flux2_vae("klein-9b", device="cuda")
"""

import os
import sys
from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import load_file as load_sft

from llm_dit.models.flux2.constants import (
    FLUX2_MODEL_INFO,
    Klein9BParams,
    Klein4BParams,
    Flux2Params,
)
from llm_dit.models.flux2.transformer import Flux2Transformer
from llm_dit.models.flux2.vae import AutoEncoder, AutoEncoderParams


# Environment variable names for local model paths
MODEL_PATH_ENV_VARS = {
    "klein-4b": "FLUX2_KLEIN_4B_PATH",
    "klein-9b": "FLUX2_KLEIN_9B_PATH",
    "klein-base-4b": "FLUX2_KLEIN_BASE_4B_PATH",
    "klein-base-9b": "FLUX2_KLEIN_BASE_9B_PATH",
}

VAE_PATH_ENV_VAR = "FLUX2_VAE_PATH"


def _try_huggingface_download(repo_id: str, filename: str) -> str | None:
    """
    Attempt to download a file from HuggingFace Hub.

    Returns the local path if successful, None otherwise.
    """
    try:
        import huggingface_hub
        return huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
        )
    except ImportError:
        print("huggingface_hub not installed. Install with: pip install huggingface_hub")
        return None
    except Exception as e:
        print(f"Failed to download {filename} from {repo_id}: {e}")
        return None


def _get_model_weight_path(model_name: str) -> str:
    """
    Get the path to model weights, checking environment and HuggingFace.

    Priority:
    1. Environment variable (e.g., FLUX2_KLEIN_9B_PATH)
    2. HuggingFace Hub download
    """
    model_name = model_name.lower()
    if model_name not in FLUX2_MODEL_INFO:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(FLUX2_MODEL_INFO.keys())}")

    config = FLUX2_MODEL_INFO[model_name]

    # Check environment variable
    env_var = MODEL_PATH_ENV_VARS.get(model_name)
    if env_var and env_var in os.environ:
        weight_path = os.environ[env_var]
        if os.path.exists(weight_path):
            return weight_path
        print(f"Warning: {env_var} set but path doesn't exist: {weight_path}")

    # Try HuggingFace download
    weight_path = _try_huggingface_download(config["repo_id"], config["filename"])
    if weight_path:
        return weight_path

    raise RuntimeError(
        f"Could not find weights for {model_name}. "
        f"Set {env_var} environment variable or ensure HuggingFace Hub access."
    )


def _get_vae_weight_path(model_name: str) -> str:
    """
    Get the path to VAE weights.

    All FLUX.2 models use the same VAE weights (ae.safetensors).
    """
    model_name = model_name.lower()
    if model_name not in FLUX2_MODEL_INFO:
        raise ValueError(f"Unknown model: {model_name}")

    config = FLUX2_MODEL_INFO[model_name]

    # Check environment variable
    if VAE_PATH_ENV_VAR in os.environ:
        weight_path = os.environ[VAE_PATH_ENV_VAR]
        if os.path.exists(weight_path):
            return weight_path
        print(f"Warning: {VAE_PATH_ENV_VAR} set but path doesn't exist: {weight_path}")

    # Try HuggingFace download
    weight_path = _try_huggingface_download(config["repo_id"], config["filename_ae"])
    if weight_path:
        return weight_path

    raise RuntimeError(
        f"Could not find VAE weights. "
        f"Set {VAE_PATH_ENV_VAR} environment variable or ensure HuggingFace Hub access."
    )


def load_flux2_transformer(
    model_name: str,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    debug_mode: bool = False,
) -> Flux2Transformer:
    """
    Load a FLUX.2 transformer model.

    Args:
        model_name: Model variant ("klein-4b", "klein-9b", "klein-base-4b", "klein-base-9b")
        device: Target device
        dtype: Model dtype (default bfloat16)
        debug_mode: If True, create minimal model (1 block each) for testing

    Returns:
        Loaded Flux2Transformer
    """
    model_name = model_name.lower()
    if model_name not in FLUX2_MODEL_INFO:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(FLUX2_MODEL_INFO.keys())}")

    config = FLUX2_MODEL_INFO[model_name]
    params = config["params_cls"]()

    if debug_mode:
        # Minimal model for testing
        params.depth = 1
        params.depth_single_blocks = 1
        print(f"Debug mode: creating minimal {model_name} model (1 double, 1 single block)")
        with torch.device(device):
            return Flux2Transformer(params).to(dtype)

    # Get weights path
    weight_path = _get_model_weight_path(model_name)
    print(f"Loading {model_name} from {weight_path}")

    # Create model on meta device for memory efficiency
    with torch.device("meta"):
        model = Flux2Transformer(params).to(dtype)

    # Load weights
    sd = load_sft(weight_path, device=str(device))
    model.load_state_dict(sd, strict=True, assign=True)

    return model.to(device)


def load_flux2_vae(
    model_name: str = "klein-9b",
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> AutoEncoder:
    """
    Load the FLUX.2 VAE (AutoEncoder).

    All FLUX.2 models share the same VAE architecture and weights.

    Args:
        model_name: Model variant (used to find weights, all share same VAE)
        device: Target device
        dtype: Model dtype (default bfloat16)

    Returns:
        Loaded AutoEncoder
    """
    # Get weights path
    weight_path = _get_vae_weight_path(model_name)
    print(f"Loading VAE from {weight_path}")

    # Create model on meta device
    with torch.device("meta"):
        vae = AutoEncoder(AutoEncoderParams())

    # Load weights
    sd = load_sft(weight_path, device=str(device))
    vae.load_state_dict(sd, strict=True, assign=True)

    return vae.to(device).to(dtype)


def get_model_info(model_name: str) -> dict:
    """
    Get information about a model without loading it.

    Args:
        model_name: Model variant name

    Returns:
        Dict with model information (params, defaults, etc.)
    """
    model_name = model_name.lower()
    if model_name not in FLUX2_MODEL_INFO:
        raise ValueError(f"Unknown model: {model_name}")

    config = FLUX2_MODEL_INFO[model_name]
    params = config["params_cls"]()

    # Calculate model size
    hidden_size = params.hidden_size
    num_heads = params.num_heads
    depth = params.depth
    depth_single = params.depth_single_blocks

    # Rough parameter estimation
    # Double blocks: 2 * (attn + mlp) per block
    # Single blocks: attn + mlp per block
    params_per_double = 4 * hidden_size * hidden_size * 3  # Rough estimate
    params_per_single = 2 * hidden_size * hidden_size * 3  # Rough estimate
    total_params = depth * params_per_double + depth_single * params_per_single

    return {
        "model_name": model_name,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "depth": depth,
        "depth_single_blocks": depth_single,
        "context_in_dim": params.context_in_dim,
        "distilled": config["distilled"],
        "defaults": config["defaults"],
        "estimated_params_b": total_params / 1e9,
        "estimated_size_bf16_gb": (total_params * 2) / 1e9,
        "text_encoder": config["text_encoder"],
    }


def list_available_models() -> list[str]:
    """List all available model variants."""
    return list(FLUX2_MODEL_INFO.keys())
