"""
FLUX.2 Model Loading Utilities.

Last Updated: 2026-01-24

Pure PyTorch implementation for loading FLUX.2 Klein models and VAE.
Supports BF16 and FP8 weights in native single-file format only.

Note: This loader does NOT support Diffusers sharded format.
      Use native single-file weights (e.g., flux-2-klein-9b-fp8.safetensors).

Ported from: coderef/flux2/src/flux2/util.py

Usage:
    from llm_dit.models.flux2.loader import (
        load_flux2_transformer,
        load_flux2_vae,
        FLUX2_MODEL_INFO,
    )

    # Load Klein 9B FP8 model
    model = load_flux2_transformer("klein-9b-fp8", device="cuda")

    # Load VAE
    vae = load_flux2_vae("klein-9b", device="cuda")
"""

import gc
import logging
import os
from pathlib import Path

import torch
from safetensors.torch import load_file as load_sft

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def _format_memory_gb(bytes_val: int | float) -> str:
    """Format memory value in GB with 2 decimal places."""
    return f"{bytes_val / 1e9:.2f}GB"


def _log_memory_state(prefix: str = "") -> None:
    """Log current GPU and CPU memory state."""
    if not logger.isEnabledFor(logging.DEBUG):
        return

    msg_parts = [f"[FLUX2:Loader:{prefix}]" if prefix else "[FLUX2:Loader]"]

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        msg_parts.append(f"GPU allocated: {_format_memory_gb(allocated)}")
        msg_parts.append(f"reserved: {_format_memory_gb(reserved)}")

    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        mem_info = process.memory_info()
        msg_parts.append(f"CPU RSS: {_format_memory_gb(mem_info.rss)}")

    logger.debug(" → ".join(msg_parts))

from llm_dit.models.flux2.constants import FLUX2_MODEL_INFO
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
    Get the path to model weights (native single-file format only).

    Priority:
    1. Direct path (if provided)
    2. Environment variable (e.g., FLUX2_KLEIN_9B_PATH)
    3. HuggingFace Hub download

    Args:
        model_name: Model variant name (e.g., "klein-9b", "klein-9b-fp8")
        model_path: Direct path to weights file or directory containing weights

    Returns:
        Path to the single .safetensors weight file
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
            # Check for expected filename (e.g., flux-2-klein-9b-fp8.safetensors)
            weight_file = model_path_obj / config["filename"]
            if weight_file.exists():
                return str(weight_file)
            # Fallback: look for any .safetensors file matching model pattern
            pattern = f"*{model_name.replace('-', '*')}*.safetensors"
            matches = list(model_path_obj.glob(pattern))
            if matches:
                return str(matches[0])
            # Last fallback: any single safetensors file
            matches = list(model_path_obj.glob("*.safetensors"))
            if len(matches) == 1:
                return str(matches[0])
        raise ValueError(
            f"Could not find native weights at {model_path}.\n"
            f"Expected single-file format (e.g., {config['filename']}).\n"
            f"Diffusers sharded format is NOT supported."
        )

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
    Get the path to VAE weights (ae.safetensors format).

    All FLUX.2 models use the same VAE weights.

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
            # Check directly at root
            ae_file = vae_path_obj / "ae.safetensors"
            if ae_file.exists():
                return str(ae_file)
            # Check in vae subdirectory
            vae_subdir = vae_path_obj / "vae"
            if vae_subdir.exists():
                ae_file = vae_subdir / "ae.safetensors"
                if ae_file.exists():
                    return str(ae_file)
        raise ValueError(
            f"Could not find VAE weights at {vae_path}.\n"
            f"Expected ae.safetensors file."
        )

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
    block_offload: bool = False,
) -> Flux2Transformer:
    """
    Load a FLUX.2 transformer model (pure PyTorch).

    Args:
        model_name: Model variant (e.g., "klein-9b", "klein-9b-fp8")
        device: Target device
        dtype: Model dtype (default bfloat16)
        debug_mode: If True, create minimal model (1 block each) for testing
        model_path: Direct path to weights file or directory (overrides HF download)
        block_offload: If True, enable block-by-block GPU offloading (slower but uses less VRAM)

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

    # Get weights path (native single-file format only)
    weight_path = _get_model_weight_path(model_name, model_path)
    print(f"Loading {model_name} from {weight_path}")
    logger.debug(f"[FLUX2:Loader] Loading weights from {weight_path}")
    _log_memory_state("Before load")

    # Check if this is an FP8 model (from registry or filename)
    is_fp8 = config.get("fp8", False) or "fp8" in weight_path.lower()
    logger.debug(f"[FLUX2:Loader] FP8 model: {is_fp8}, block_offload: {block_offload}")

    # Create model on meta device for memory efficiency
    with torch.device("meta"):
        model = Flux2Transformer(params).to(dtype)

    # Load weights - for FP8, load to CPU first to avoid GPU memory spike during casting
    load_device = "cpu" if is_fp8 else str(device)
    logger.debug(f"[FLUX2:Loader] Loading safetensors to device: {load_device}")
    sd = load_sft(weight_path, device=load_device)
    _log_memory_state("After load_sft")

    # FP8 checkpoints contain extra scale tensors (input_scale, weight_scale)
    # that our model doesn't have. Filter them out and cast weights to target dtype.
    if is_fp8:
        # Filter out FP8 scale tensors - they're metadata, not model weights
        scale_keys = [k for k in sd.keys() if k.endswith(("_scale", ".input_scale", ".weight_scale"))]
        if scale_keys:
            print(f"FP8 checkpoint detected: removing {len(scale_keys)} scale tensors")
            for k in scale_keys:
                del sd[k]

        # Cast FP8 weights to target dtype (bf16) on CPU to avoid memory spike
        # FP8 weights are float8_e4m3fn which can't be used directly in matmul with bf16
        fp8_count = 0
        for k, v in sd.items():
            if v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                sd[k] = v.to(dtype)
                fp8_count += 1
        if fp8_count > 0:
            print(f"Cast {fp8_count} FP8 tensors to {dtype}")

        # Move to target device after casting - respect block_offload flag
        # BUG FIX: Previously moved ALL weights to GPU unconditionally (17GB for Klein-9B),
        # causing OOM even with block_offload=True. Now we check block_offload BEFORE
        # moving weights to avoid the memory spike.
        target_device = "cpu" if block_offload else device

        # Log memory before moving weights (P1: debug logging)
        pre_move = 0.0
        if torch.cuda.is_available():
            pre_move = torch.cuda.memory_allocated() / 1e9
            logger.debug(f"[Loader] GPU before state dict move: {pre_move:.2f}GB")

        sd = {k: v.to(target_device) for k, v in sd.items()}

        # Log memory after moving weights
        if torch.cuda.is_available():
            post_move = torch.cuda.memory_allocated() / 1e9
            delta = post_move - pre_move
            logger.debug(f"[Loader] GPU after state dict move: {post_move:.2f}GB (delta: {delta:.2f}GB)")

        # Clear temp tensors to maximize RAM for model loading
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elif block_offload:
        # For non-FP8 models with block offloading, ensure weights stay on CPU
        sd = {k: v.to("cpu") for k, v in sd.items()}

    # Log sample tensor dtypes and devices for debugging
    sample_keys = list(sd.keys())[:3]
    for k in sample_keys:
        logger.debug(f"[Loader] Sample tensor {k}: dtype={sd[k].dtype}, device={sd[k].device}")

    model.load_state_dict(sd, strict=True, assign=True)

    # Free state dict memory
    del sd
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if block_offload:
        # Enable block-by-block offloading (keeps blocks on CPU, small layers on GPU)
        print(f"Block offloading enabled: blocks will be moved to GPU one at a time")
        return model.enable_block_offload(device=device, offload_device="cpu")
    else:
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
