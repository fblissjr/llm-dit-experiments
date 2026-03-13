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

import logging
import os
from pathlib import Path

import torch
from safetensors.torch import load_file as load_sft

from llm_dit.utils.memory import cleanup_memory, format_memory_gb, log_memory_debug

logger = logging.getLogger(__name__)

from llm_dit.models.flux2.constants import FLUX2_MODEL_INFO
from llm_dit.models.flux2.transformer import Flux2Transformer
from llm_dit.models.flux2.vae import AutoEncoder, AutoEncoderParams


# Environment variable names for local model paths
MODEL_PATH_ENV_VARS = {
    "klein-4b": "FLUX2_KLEIN_4B_PATH",
    "klein-9b": "FLUX2_KLEIN_9B_PATH",
    "klein-base-4b": "FLUX2_KLEIN_BASE_4B_PATH",
    "klein-base-9b": "FLUX2_KLEIN_BASE_9B_PATH",
    "klein-9b-kv": "FLUX2_KLEIN_9B_KV_PATH",
}

VAE_PATH_ENV_VAR = "FLUX2_VAE_PATH"


def _convert_diffusers_vae_keys(state_dict: dict) -> dict:
    """
    Convert Diffusers VAE state_dict to native FLUX format.

    This handles both key renaming AND shape conversion:
    - Diffusers uses nn.Linear for attention (2D weights)
    - Native FLUX uses nn.Conv2d with kernel_size=1 (4D weights)

    Key mappings:
    - encoder.down_blocks.X.resnets.Y -> encoder.down.X.block.Y
    - encoder.down_blocks.X.downsamplers.0.conv -> encoder.down.X.downsample.conv
    - encoder.mid_block.resnets.0/1 -> encoder.mid.block_1/block_2
    - encoder.mid_block.attentions.0.to_q/k/v -> encoder.mid.attn_1.q/k/v
    - encoder.mid_block.attentions.0.to_out.0 -> encoder.mid.attn_1.proj_out
    - encoder.mid_block.attentions.0.group_norm -> encoder.mid.attn_1.norm
    - encoder.conv_norm_out -> encoder.norm_out
    - quant_conv -> encoder.quant_conv
    - conv_shortcut -> nin_shortcut
    - decoder.up_blocks.X reversed to decoder.up.(3-X) (indices are reversed)
    """
    import re

    # Check if conversion is needed (look for Diffusers-specific keys)
    has_diffusers_keys = any("down_blocks" in k or "up_blocks" in k for k in state_dict.keys())
    if not has_diffusers_keys:
        return state_dict  # Already in native format

    logger.info("Converting Diffusers VAE to native FLUX format (keys + shapes)...")

    new_sd = {}
    num_up_blocks = 4  # Standard VAE has 4 up/down blocks

    # Keys that need shape conversion from [N, M] to [N, M, 1, 1]
    attention_weight_patterns = [
        r"\.attn_1\.(q|k|v|proj_out)\.weight$",
    ]

    for old_key, value in state_dict.items():
        new_key = old_key

        # Top-level quant_conv -> encoder.quant_conv
        if old_key.startswith("quant_conv."):
            new_key = "encoder." + old_key
        # Top-level post_quant_conv -> decoder.post_quant_conv
        elif old_key.startswith("post_quant_conv."):
            new_key = "decoder." + old_key

        # Encoder down blocks
        elif old_key.startswith("encoder.down_blocks."):
            new_key = old_key
            # resnets -> block
            new_key = re.sub(r"\.resnets\.(\d+)\.", r".block.\1.", new_key)
            # downsamplers.0.conv -> downsample.conv
            new_key = re.sub(r"\.downsamplers\.0\.conv", ".downsample.conv", new_key)
            # conv_shortcut -> nin_shortcut
            new_key = new_key.replace(".conv_shortcut.", ".nin_shortcut.")
            # down_blocks -> down
            new_key = new_key.replace("encoder.down_blocks.", "encoder.down.")

        # Encoder mid block
        elif old_key.startswith("encoder.mid_block."):
            new_key = old_key
            # resnets.0 -> block_1, resnets.1 -> block_2
            new_key = re.sub(r"\.resnets\.0\.", ".block_1.", new_key)
            new_key = re.sub(r"\.resnets\.1\.", ".block_2.", new_key)
            # attentions.0.to_q/k/v -> attn_1.q/k/v
            new_key = re.sub(r"\.attentions\.0\.to_([qkv])\.", r".attn_1.\1.", new_key)
            # attentions.0.to_out.0 -> attn_1.proj_out
            new_key = re.sub(r"\.attentions\.0\.to_out\.0\.", ".attn_1.proj_out.", new_key)
            # attentions.0.group_norm -> attn_1.norm
            new_key = re.sub(r"\.attentions\.0\.group_norm\.", ".attn_1.norm.", new_key)
            # mid_block -> mid
            new_key = new_key.replace("encoder.mid_block.", "encoder.mid.")

        # Encoder conv_norm_out -> norm_out
        elif old_key.startswith("encoder.conv_norm_out."):
            new_key = old_key.replace("encoder.conv_norm_out.", "encoder.norm_out.")

        # Decoder up blocks (indices are reversed: up_blocks.0 -> up.3, etc.)
        elif old_key.startswith("decoder.up_blocks."):
            new_key = old_key
            # Extract block index and reverse it
            match = re.match(r"decoder\.up_blocks\.(\d+)\.", old_key)
            if match:
                idx = int(match.group(1))
                reversed_idx = num_up_blocks - 1 - idx
                # resnets -> block
                new_key = re.sub(r"\.resnets\.(\d+)\.", r".block.\1.", new_key)
                # upsamplers.0.conv -> upsample.conv
                new_key = re.sub(r"\.upsamplers\.0\.conv", ".upsample.conv", new_key)
                # conv_shortcut -> nin_shortcut
                new_key = new_key.replace(".conv_shortcut.", ".nin_shortcut.")
                # Replace index
                new_key = re.sub(r"decoder\.up_blocks\.\d+\.", f"decoder.up.{reversed_idx}.", new_key)

        # Decoder mid block
        elif old_key.startswith("decoder.mid_block."):
            new_key = old_key
            # resnets.0 -> block_1, resnets.1 -> block_2
            new_key = re.sub(r"\.resnets\.0\.", ".block_1.", new_key)
            new_key = re.sub(r"\.resnets\.1\.", ".block_2.", new_key)
            # attentions.0.to_q/k/v -> attn_1.q/k/v
            new_key = re.sub(r"\.attentions\.0\.to_([qkv])\.", r".attn_1.\1.", new_key)
            # attentions.0.to_out.0 -> attn_1.proj_out
            new_key = re.sub(r"\.attentions\.0\.to_out\.0\.", ".attn_1.proj_out.", new_key)
            # attentions.0.group_norm -> attn_1.norm
            new_key = re.sub(r"\.attentions\.0\.group_norm\.", ".attn_1.norm.", new_key)
            # mid_block -> mid
            new_key = new_key.replace("decoder.mid_block.", "decoder.mid.")

        # Decoder conv_norm_out -> norm_out
        elif old_key.startswith("decoder.conv_norm_out."):
            new_key = old_key.replace("decoder.conv_norm_out.", "decoder.norm_out.")

        # BatchNorm keys are kept as-is (bn.running_mean, bn.running_var, bn.num_batches_tracked)
        # These are critical for latent normalization!

        # Shape conversion: Attention weights need [N, M] -> [N, M, 1, 1] for Conv2d
        new_value = value
        for pattern in attention_weight_patterns:
            if re.search(pattern, new_key) and value.dim() == 2:
                new_value = value.unsqueeze(-1).unsqueeze(-1)
                break

        new_sd[new_key] = new_value

    # Add missing BatchNorm buffers (FLUX VAE has BatchNorm for latent normalization)
    # These are initialized as zeros and ones, and will be updated during first forward pass
    if "bn.running_mean" not in new_sd:
        # Default BatchNorm stats for 128-channel latent space (32 z_channels * 4 from patchify)
        new_sd["bn.running_mean"] = torch.zeros(128)
        new_sd["bn.running_var"] = torch.ones(128)
        new_sd["bn.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)

    logger.info(f"Converted {len(new_sd)} VAE keys (including shape fixes)")
    return new_sd


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
        logger.warning("huggingface_hub not installed. Install with: pip install huggingface_hub")
        return None
    except Exception as e:
        logger.warning(f"Failed to download {filename} from {repo_id}: {e}")
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
        logger.warning(f"{env_var} set but path doesn't exist: {weight_path}")

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
        # If it's a directory, look for VAE weights
        # Prefer native format (ae.safetensors) but also support Diffusers format
        # (diffusion_pytorch_model.safetensors) which will be converted automatically
        if vae_path_obj.is_dir():
            search_dirs = [vae_path_obj]

            # Also check vae subdirectory
            vae_subdir = vae_path_obj / "vae"
            if vae_subdir.exists():
                search_dirs.append(vae_subdir)

            # Check for native format first
            for search_dir in search_dirs:
                ae_file = search_dir / "ae.safetensors"
                if ae_file.exists():
                    return str(ae_file)

            # Fall back to Diffusers format (will be converted by _convert_diffusers_vae_keys)
            for search_dir in search_dirs:
                diffusers_file = search_dir / "diffusion_pytorch_model.safetensors"
                if diffusers_file.exists():
                    return str(diffusers_file)

        raise ValueError(
            f"Could not find VAE weights at {vae_path}.\n"
            f"Expected ae.safetensors or diffusion_pytorch_model.safetensors."
        )

    # 2. Check environment variable
    if VAE_PATH_ENV_VAR in os.environ:
        weight_path = os.environ[VAE_PATH_ENV_VAR]
        if os.path.exists(weight_path):
            return weight_path
        logger.warning(f"{VAE_PATH_ENV_VAR} set but path doesn't exist: {weight_path}")

    # 3. Try HuggingFace download from model's repo
    # FLUX.2-klein repos have vae/diffusion_pytorch_model.safetensors (Diffusers format)
    # which will be converted automatically by _convert_diffusers_vae_keys
    vae_repo_id = config["repo_id"]
    vae_filename = "vae/diffusion_pytorch_model.safetensors"

    weight_path = _try_huggingface_download(vae_repo_id, vae_filename)
    if weight_path:
        return weight_path

    raise RuntimeError(
        f"Could not find VAE weights. Options:\n"
        f"  1. Use --flux2-vae-path to specify local path\n"
        f"  2. Set {VAE_PATH_ENV_VAR} environment variable\n"
        f"  3. Ensure HuggingFace Hub access to {vae_repo_id}"
    )


def load_flux2_transformer(
    model_name: str,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    debug_mode: bool = False,
    model_path: str | None = None,
    block_offload: bool = False,
    validate: bool = True,
    quantize_to: str = "none",
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
        validate: If True, run sanity checks on loaded weights (catches FP8 dequant issues)
        quantize_to: Post-load quantization via torchao ("none", "fp8", "int8").
                     "fp8" replaces Linear weights with Float8Tensor for SM89 FP8 tensor cores.
                     Incompatible with block_offload (requires all weights on GPU).

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
        logger.debug(f"Debug mode: creating minimal {model_name} model (1 double, 1 single block)")
        with torch.device(device):
            return Flux2Transformer(params).to(dtype)

    # Get weights path (native single-file format only)
    weight_path = _get_model_weight_path(model_name, model_path)
    logger.info(f"Loading {model_name} from {weight_path}")
    logger.debug(f"[FLUX2:Loader] Loading weights from {weight_path}")
    log_memory_debug("Before load")

    # Check if this is an FP8 model (from registry or filename)
    is_fp8 = config.get("fp8", False) or "fp8" in weight_path.lower()
    logger.debug(f"[FLUX2:Loader] FP8 model: {is_fp8}, block_offload: {block_offload}")

    # Create model on meta device for memory efficiency
    with torch.device("meta"):
        model = Flux2Transformer(params).to(dtype)

    # Load weights to CPU when fp8 (small, moved to GPU after patching) or block_offload
    # (blocks stay on CPU, moved to GPU one at a time). Avoids GPU memory spike.
    load_device = "cpu" if (is_fp8 or block_offload) else str(device)
    logger.debug(f"[FLUX2:Loader] Loading safetensors to device: {load_device}")
    sd = load_sft(weight_path, device=load_device)
    log_memory_debug("After load_sft")

    # FP8-cast: keep weights as fp8, upcast to bf16 per-forward pass.
    # This drops transformer VRAM from ~18GB (dequant) to ~4.5GB (fp8).
    if is_fp8:
        # Extract weight scales from state dict
        scale_keys = [k for k in sd if k.endswith(".weight_scale")]
        weight_scales = {}
        for sk in scale_keys:
            weight_key = sk.replace(".weight_scale", ".weight")
            weight_scales[weight_key] = sd.pop(sk)

        # Remove input_scale and other non-weight scale keys
        for k in [k for k in sd if k.endswith(("_scale", ".input_scale"))]:
            del sd[k]

        if weight_scales:
            logger.info(f"[FLUX2:Loader] FP8-cast mode: {len(weight_scales)} weight scales")

        # Load fp8 weights directly (assign=True preserves fp8 dtype)
        logger.debug("[FLUX2:Loader] Loading fp8 state dict with assign=True")
        model.load_state_dict(sd, strict=False, assign=True)
        log_memory_debug("After fp8 load_state_dict")

        # Attach per-tensor weight scales to nn.Linear modules
        from llm_dit.quantization.fp8_cast import _attach_weight_scales, amend_forward_with_upcast
        _attach_weight_scales(model, weight_scales)

        # Patch forward methods for per-forward upcast (fp8 -> bf16)
        count = amend_forward_with_upcast(model)
        logger.info(f"[FLUX2:Loader] FP8-cast: {count} layers patched for per-forward upcast")

        # FP8 model is small (~4.5GB) -- can stay on GPU even with block_offload
        target_device = "cpu" if block_offload else device
        model = model.to(target_device)
        log_memory_debug("After fp8 model.to(device)")

        # Free scale dict
        del weight_scales, sd
        cleanup_memory()

    else:
        # Non-FP8 path: load bf16 weights, move to device
        # (block_offload models already loaded to CPU via load_device)

        # Log sample tensor dtypes and devices for debugging
        sample_keys = list(sd.keys())[:5]
        for k in sample_keys:
            tensor = sd[k]
            tensor_size = tensor.numel() * tensor.element_size()
            logger.debug(
                f"[FLUX2:Loader] Sample tensor {k}: dtype={tensor.dtype}, "
                f"device={tensor.device}, shape={list(tensor.shape)}, size={format_memory_gb(tensor_size)}"
            )

        logger.debug("[FLUX2:Loader] Calling load_state_dict with assign=True")
        model.load_state_dict(sd, strict=True, assign=True)
        log_memory_debug("After load_state_dict")

        # Free state dict memory
        del sd
        cleanup_memory()
        log_memory_debug("After freeing state dict")

        if block_offload:
            logger.info("[FLUX2:Loader] Block offload: blocks will move to GPU one at a time")
            model = model.enable_block_offload(device=device, offload_device="cpu")
            log_memory_debug("After enable_block_offload")
        else:
            logger.debug(f"[FLUX2:Loader] Moving entire model to {device}")
            model = model.to(device)
            log_memory_debug("After model.to(device)")

    # Validate loaded weights (catches FP8 dequantization issues)
    if validate:
        _validate_transformer_weights(model, is_fp8)

    # Post-load quantization via torchao (after model is on GPU).
    # Skip for fp8 models -- already using fp8-cast per-forward upcast.
    if quantize_to and quantize_to != "none" and not is_fp8 and not block_offload:
        from llm_dit.quantization import quantize_component
        logger.info(f"[FLUX2:Loader] Applying {quantize_to} quantization...")
        log_memory_debug("Before quantization")
        model, stats = quantize_component(  # type: ignore[assignment]
            model,
            method=quantize_to,
            component_type="transformer",
        )
        logger.info(
            f"[FLUX2:Loader] Quantization complete: "
            f"{stats['quantized_layers']}/{stats['total_layers']} layers quantized"
        )
        log_memory_debug("After quantization")
    elif quantize_to and quantize_to != "none" and block_offload:
        raise ValueError(
            f"quantize_to='{quantize_to}' is incompatible with block_offload=True. "
            "torchao quantization requires all weights on GPU. "
            "Set block_offload=false or quantization='none'."
        )

    return model


def _validate_transformer_weights(model: Flux2Transformer, is_fp8: bool) -> None:
    """
    Validate transformer weights after loading.

    Catches common issues:
    - FP8 dequantization failure (weights too large - scales not applied)
    - NaN/Inf values in weights
    - Wrong dtype

    Args:
        model: Loaded transformer model
        is_fp8: Whether model was loaded from FP8 checkpoint
    """
    # Sample a few parameters for validation
    sample_params = []
    for name, param in model.named_parameters():
        sample_params.append((name, param))
        if len(sample_params) >= 5:
            break

    for name, param in sample_params:
        # Skip fp8 params -- isnan/isinf not implemented for float8_e4m3fn.
        # FP8-cast models keep weights as fp8 and upcast per-forward.
        if param.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            continue

        # Check for NaN/Inf
        if param.isnan().any():
            raise ValueError(f"NaN detected in weight '{name}' - model is corrupted")
        if param.isinf().any():
            raise ValueError(f"Inf detected in weight '{name}' - model is corrupted")

        # Check weight magnitude (catches FP8 dequant failure)
        # Properly scaled weights should have std < 1.0 (typically 0.01-0.1)
        # Unscaled FP8 weights have std > 10.0
        param_std = param.float().std().item()
        if param_std > 5.0:
            logger.warning(
                f"[FLUX2:Loader:Validate] Weight '{name}' has high std={param_std:.2f} - "
                f"FP8 scale factors may not have been applied correctly"
            )
            if is_fp8:
                raise ValueError(
                    f"FP8 dequantization failed: weight '{name}' has std={param_std:.2f} "
                    f"(expected < 1.0). Scale factors were not applied correctly."
                )

    logger.debug("[FLUX2:Loader:Validate] Transformer weights validated successfully")


def load_flux2_vae(
    model_name: str = "klein-9b",
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    vae_path: str | None = None,
    validate: bool = True,
) -> AutoEncoder:
    """
    Load the FLUX.2 VAE (AutoEncoder).

    All FLUX.2 models share the same VAE architecture and weights.

    Args:
        model_name: Model variant (used to find weights, all share same VAE)
        device: Target device
        dtype: Model dtype (default bfloat16)
        vae_path: Direct path to VAE weights file or directory (overrides HF download)
        validate: If True, run sanity checks on loaded VAE (catches BatchNorm issues)

    Returns:
        Loaded AutoEncoder
    """
    # Get weights path
    weight_path = _get_vae_weight_path(model_name, vae_path)
    logger.info(f"Loading VAE from {weight_path}")

    # Create model on meta device
    with torch.device("meta"):
        vae = AutoEncoder(AutoEncoderParams())

    # Load weights
    sd = load_sft(weight_path, device=str(device))

    # Convert Diffusers VAE keys to native FLUX format if needed
    sd = _convert_diffusers_vae_keys(sd)

    vae.load_state_dict(sd, strict=True, assign=True)
    vae = vae.to(device).to(dtype)

    # Validate VAE (catches BatchNorm stats issues)
    if validate:
        _validate_vae(vae)

    return vae


def _validate_vae(vae: AutoEncoder) -> None:
    """
    Validate VAE after loading.

    Catches common issues:
    - BatchNorm running stats not loaded (would produce poor quality)
    - NaN/Inf values

    Args:
        vae: Loaded AutoEncoder
    """
    # Check BatchNorm running stats
    # If mean=0 and var=1, the stats weren't loaded - using identity normalization
    bn_mean = vae.bn.running_mean
    bn_var = vae.bn.running_var

    if bn_mean is None or bn_var is None:
        raise ValueError("VAE BatchNorm running stats are None - model not loaded correctly")

    mean_is_zero = (bn_mean.abs() < 1e-6).all()
    var_is_one = ((bn_var - 1.0).abs() < 1e-6).all()

    if mean_is_zero and var_is_one:
        logger.warning(
            "[FLUX2:Loader:Validate] VAE BatchNorm has identity stats (mean=0, var=1) - "
            "latent normalization may not work correctly. "
            "This can happen if BatchNorm keys weren't loaded from the weights file."
        )

    # Check for NaN/Inf
    if bn_mean.isnan().any() or bn_var.isnan().any():
        raise ValueError("VAE BatchNorm contains NaN values - model is corrupted")

    # Log actual stats for debugging
    logger.debug(
        f"[FLUX2:Loader:Validate] VAE BatchNorm stats: "
        f"mean range=[{bn_mean.min():.4f}, {bn_mean.max():.4f}], "
        f"var range=[{bn_var.min():.4f}, {bn_var.max():.4f}]"
    )
    logger.debug("[FLUX2:Loader:Validate] VAE validated successfully")


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
