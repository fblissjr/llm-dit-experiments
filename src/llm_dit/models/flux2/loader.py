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


def _get_model_weight_path(model_name: str, model_path: str | None = None) -> str:
    """
    Get the path to model weights, checking direct path, environment, and HuggingFace.

    Priority:
    1. Direct path (if provided)
    2. Environment variable (e.g., FLUX2_KLEIN_9B_PATH)
    3. HuggingFace Hub download

    Args:
        model_name: Model variant name (e.g., "klein-9b", "klein-9b-fp8")
        model_path: Direct path to weights file or directory containing weights
    """
    model_name = model_name.lower()
    if model_name not in FLUX2_MODEL_INFO:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(FLUX2_MODEL_INFO.keys())}")

    config = FLUX2_MODEL_INFO[model_name]

    # 1. Check direct path (highest priority)
    if model_path:
        model_path_obj = Path(model_path)
        if model_path_obj.is_file() and model_path_obj.suffix == ".safetensors":
            return str(model_path_obj)
        # If it's a directory, look for the expected filename
        if model_path_obj.is_dir():
            # Check for transformer subdirectory (common in HF repos)
            transformer_dir = model_path_obj / "transformer"
            if transformer_dir.exists():
                weight_file = transformer_dir / config["filename"]
                if weight_file.exists():
                    return str(weight_file)
            # Check directly in the directory
            weight_file = model_path_obj / config["filename"]
            if weight_file.exists():
                return str(weight_file)
            # Fallback: look for any .safetensors file matching model pattern
            pattern = f"*{model_name.replace('-', '*')}*.safetensors"
            matches = list(model_path_obj.glob(pattern))
            if matches:
                return str(matches[0])
            # Last fallback: any safetensors file
            matches = list(model_path_obj.glob("*.safetensors"))
            if len(matches) == 1:
                return str(matches[0])
        raise ValueError(f"Could not find weights at {model_path}")

    # 2. Check environment variable
    env_var = MODEL_PATH_ENV_VARS.get(model_name.replace("-fp8", ""))  # Strip fp8 suffix for env var
    if env_var and env_var in os.environ:
        weight_path = os.environ[env_var]
        if os.path.exists(weight_path):
            return weight_path
        print(f"Warning: {env_var} set but path doesn't exist: {weight_path}")

    # 3. Try HuggingFace download
    weight_path = _try_huggingface_download(config["repo_id"], config["filename"])
    if weight_path:
        return weight_path

    raise RuntimeError(
        f"Could not find weights for {model_name}. Options:\n"
        f"  1. Use --flux2-model-path to specify local path\n"
        f"  2. Set {env_var} environment variable\n"
        f"  3. Ensure HuggingFace Hub access for {config['repo_id']}"
    )


def _get_vae_weight_path(model_name: str, vae_path: str | None = None) -> str:
    """
    Get the path to VAE weights.

    All FLUX.2 models use the same VAE weights (ae.safetensors).

    Args:
        model_name: Model variant name (used to find repo if downloading)
        vae_path: Direct path to VAE weights file or directory
    """
    model_name = model_name.lower().replace("-fp8", "")  # VAE is same for fp8/bf16
    if model_name not in FLUX2_MODEL_INFO:
        raise ValueError(f"Unknown model: {model_name}")

    config = FLUX2_MODEL_INFO[model_name]

    # 1. Check direct path (highest priority)
    if vae_path:
        vae_path_obj = Path(vae_path)
        if vae_path_obj.is_file() and vae_path_obj.suffix == ".safetensors":
            return str(vae_path_obj)
        # If it's a directory, look for ae.safetensors
        if vae_path_obj.is_dir():
            # Check for vae subdirectory
            vae_subdir = vae_path_obj / "vae"
            if vae_subdir.exists():
                ae_file = vae_subdir / "ae.safetensors"
                if ae_file.exists():
                    return str(ae_file)
            # Check directly
            ae_file = vae_path_obj / "ae.safetensors"
            if ae_file.exists():
                return str(ae_file)
            # Check diffusion_pytorch_model.safetensors (alternative name)
            alt_file = vae_path_obj / "diffusion_pytorch_model.safetensors"
            if alt_file.exists():
                return str(alt_file)
        raise ValueError(f"Could not find VAE weights at {vae_path}")

    # 2. Check environment variable
    if VAE_PATH_ENV_VAR in os.environ:
        weight_path = os.environ[VAE_PATH_ENV_VAR]
        if os.path.exists(weight_path):
            return weight_path
        print(f"Warning: {VAE_PATH_ENV_VAR} set but path doesn't exist: {weight_path}")

    # 3. Try HuggingFace download
    weight_path = _try_huggingface_download(config["repo_id"], config["filename_ae"])
    if weight_path:
        return weight_path

    raise RuntimeError(
        f"Could not find VAE weights. Options:\n"
        f"  1. Use --flux2-vae-path to specify local path\n"
        f"  2. Set {VAE_PATH_ENV_VAR} environment variable\n"
        f"  3. Ensure HuggingFace Hub access"
    )


def load_flux2_transformer(
    model_name: str,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    debug_mode: bool = False,
    model_path: str | None = None,
) -> Flux2Transformer:
    """
    Load a FLUX.2 transformer model.

    Args:
        model_name: Model variant (e.g., "klein-9b", "klein-9b-fp8")
        device: Target device
        dtype: Model dtype (default bfloat16)
        debug_mode: If True, create minimal model (1 block each) for testing
        model_path: Direct path to weights file or directory (overrides HF download)

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
    weight_path = _get_model_weight_path(model_name, model_path)
    print(f"Loading {model_name} from {weight_path}")

    # Check if this is an FP8 model (from registry or filename)
    is_fp8 = config.get("fp8", False) or "fp8" in weight_path.lower()

    # Create model on meta device for memory efficiency
    with torch.device("meta"):
        model = Flux2Transformer(params).to(dtype)

    # Load weights
    sd = load_sft(weight_path, device=str(device))

    # FP8 checkpoints contain extra scale tensors (input_scale, weight_scale)
    # that our model doesn't have. Filter them out and cast weights to target dtype.
    if is_fp8:
        # Filter out FP8 scale tensors - they're metadata, not model weights
        scale_keys = [k for k in sd.keys() if k.endswith(("_scale", ".input_scale", ".weight_scale"))]
        if scale_keys:
            print(f"FP8 checkpoint detected: removing {len(scale_keys)} scale tensors")
            for k in scale_keys:
                del sd[k]

        # Cast FP8 weights to target dtype (bf16)
        # FP8 weights are float8_e4m3fn which can't be used directly in matmul with bf16
        fp8_count = 0
        for k, v in sd.items():
            if v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                sd[k] = v.to(dtype)
                fp8_count += 1
        if fp8_count > 0:
            print(f"Cast {fp8_count} FP8 tensors to {dtype}")

    model.load_state_dict(sd, strict=True, assign=True)

    return model.to(device)


def load_flux2_vae(
    model_name: str = "klein-9b",
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    vae_path: str | None = None,
) -> AutoEncoder:
    """
    Load the FLUX.2 VAE (AutoEncoder).

    All FLUX.2 models share the same VAE architecture and weights.

    Args:
        model_name: Model variant (used to find weights, all share same VAE)
        device: Target device
        dtype: Model dtype (default bfloat16)
        vae_path: Direct path to VAE weights file or directory (overrides HF download)

    Returns:
        Loaded AutoEncoder
    """
    # Get weights path
    weight_path = _get_vae_weight_path(model_name, vae_path)
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
