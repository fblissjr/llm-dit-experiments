"""
Weight loading utilities for LTX-2 transformer.

Last Updated: 2026-01-18

Provides functions for loading official LTX-2 checkpoints into our pure PyTorch
implementation. Handles the key mapping between diffusers format and our naming.

Supports:
- Diffusers sharded format (models/LTX-2/transformer/)
- Single file safetensors (ltx-2-19b-dev.safetensors)
- FP8/FP4 quantized checkpoints

Usage:
    from llm_dit.models.ltx2 import load_ltx2_transformer

    # Load from diffusers format
    model = load_ltx2_transformer("models/LTX-2/transformer/")

    # Load from single safetensors
    model = load_ltx2_transformer("models/LTX-2/ltx-2-19b-dev.safetensors")

    # Load with specific dtype
    model = load_ltx2_transformer(path, dtype=torch.bfloat16)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import torch

from llm_dit.models.ltx2.transformer import LTX2Transformer, LTXModelType
from llm_dit.models.ltx2.rope import LTXRopeType

logger = logging.getLogger(__name__)


# Key mapping from diffusers format to our implementation
DIFFUSERS_TO_OURS = {
    # Input/output projections
    "proj_in.": "patchify_proj.",

    # Timestep embedding (AdaLayerNormSingle)
    "time_embed.emb.timestep_embedder.": "adaln_single.emb.timestep_embedder.",
    "time_embed.linear.": "adaln_single.linear.",

    # Attention norm naming (diffusers uses norm_q/norm_k, we use q_norm/k_norm)
    ".norm_q.": ".q_norm.",
    ".norm_k.": ".k_norm.",

    # The rest should match (caption_projection, transformer_blocks, etc.)
}

# Keys to skip when loading video-only (audio components)
AUDIO_KEYS_PREFIX = [
    "audio_",
    "av_cross_attn",
    "transformer_blocks.*.audio_",
    "transformer_blocks.*.a2v_",
    "transformer_blocks.*.v2a_",
]


def map_key(diffusers_key: str) -> str:
    """
    Map a diffusers state dict key to our naming convention.

    Args:
        diffusers_key: Key from diffusers checkpoint

    Returns:
        Mapped key for our implementation
    """
    key = diffusers_key

    for old, new in DIFFUSERS_TO_OURS.items():
        key = key.replace(old, new)

    return key


def is_audio_key(key: str) -> bool:
    """Check if a key is for audio components."""
    if key.startswith("audio_"):
        return True
    if key.startswith("av_cross_attn"):
        return True
    if "audio_attn" in key:
        return True
    if "audio_ff" in key:
        return True
    if "audio_a2v" in key or "audio_v2a" in key:
        return True
    if "a2v_cross_attn" in key or "v2a_cross_attn" in key:
        return True
    return False


def load_safetensors(path: Path, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Load state dict from safetensors file(s).

    Handles both single files and sharded checkpoints.

    Args:
        path: Path to safetensors file or directory containing shards
        device: Device to load tensors to

    Returns:
        State dictionary
    """
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("safetensors required. Install: pip install safetensors")

    state_dict = {}

    if path.is_file():
        # Single file
        with safe_open(str(path), framework="pt", device=device) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    else:
        # Sharded - look for index file
        index_path = path / "diffusion_pytorch_model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"No index file found at {index_path}")

        with open(index_path) as f:
            index = json.load(f)

        # Get unique shard files
        shard_files = set(index["weight_map"].values())

        for shard_name in shard_files:
            shard_path = path / shard_name
            with safe_open(str(shard_path), framework="pt", device=device) as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)

    return state_dict


def load_config(path: Path) -> dict:
    """
    Load model configuration from checkpoint directory.

    Args:
        path: Path to checkpoint directory

    Returns:
        Configuration dictionary
    """
    config_path = path / "config.json" if path.is_dir() else path.parent / "config.json"

    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)

    # Default config if no config file
    return {
        "num_attention_heads": 32,
        "attention_head_dim": 128,
        "in_channels": 128,
        "out_channels": 128,
        "num_layers": 48,
        "cross_attention_dim": 4096,
        "caption_channels": 3840,
        "rope_type": "split",
        "rope_theta": 10000.0,
        "rope_double_precision": True,
        "timestep_scale_multiplier": 1000,
        "pos_embed_max_pos": 20,
        "base_height": 2048,
        "base_width": 2048,
    }


def create_model_from_config(config: dict, dtype: torch.dtype = torch.bfloat16) -> LTX2Transformer:
    """
    Create LTX2Transformer from config dictionary.

    Args:
        config: Model configuration
        dtype: Model dtype

    Returns:
        Initialized model (random weights)
    """
    rope_type = LTXRopeType.SPLIT if config.get("rope_type") == "split" else LTXRopeType.INTERLEAVED

    model = LTX2Transformer(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=config.get("num_attention_heads", 32),
        attention_head_dim=config.get("attention_head_dim", 128),
        in_channels=config.get("in_channels", 128),
        out_channels=config.get("out_channels", 128),
        num_layers=config.get("num_layers", 48),
        cross_attention_dim=config.get("cross_attention_dim", 4096),
        caption_channels=config.get("caption_channels", 3840),
        positional_embedding_theta=config.get("rope_theta", 10000.0),
        positional_embedding_max_pos=[
            config.get("pos_embed_max_pos", 20),
            config.get("base_height", 2048),
            config.get("base_width", 2048),
        ],
        timestep_scale_multiplier=config.get("timestep_scale_multiplier", 1000),
        use_middle_indices_grid=True,
        rope_type=rope_type,
        double_precision_rope=config.get("rope_double_precision", True),
    )

    return model.to(dtype)


def load_ltx2_transformer(
    path: Union[str, Path],
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    video_only: bool = True,
    strict: bool = False,
) -> LTX2Transformer:
    """
    Load LTX-2 transformer from checkpoint.

    Loads weights from official LTX-2 checkpoints and maps them to our
    pure PyTorch implementation.

    Args:
        path: Path to checkpoint file or directory
        dtype: Model dtype (bf16 recommended)
        device: Device to load to initially (use 'cpu' then .to('cuda') for large models)
        video_only: If True, skip audio weights
        strict: If True, raise error on missing/extra keys

    Returns:
        Loaded LTX2Transformer model

    Example:
        # Load video-only model
        model = load_ltx2_transformer("models/LTX-2/transformer/")
        model = model.cuda()  # Move to GPU after loading

        # Load with specific dtype
        model = load_ltx2_transformer(path, dtype=torch.float16)
    """
    path = Path(path)

    # Load config
    config = load_config(path)
    logger.info(f"Loaded config: {config.get('num_layers', 48)} layers, "
                f"{config.get('num_attention_heads', 32)} heads")

    # Create model
    model = create_model_from_config(config, dtype)

    # Load weights
    logger.info(f"Loading weights from {path}")
    diffusers_state_dict = load_safetensors(path, device=device)

    # Map keys and filter audio if needed
    our_state_dict = {}
    skipped_keys = []
    missing_keys = []

    for diffusers_key, tensor in diffusers_state_dict.items():
        # Skip audio keys if video_only
        if video_only and is_audio_key(diffusers_key):
            skipped_keys.append(diffusers_key)
            continue

        our_key = map_key(diffusers_key)
        our_state_dict[our_key] = tensor.to(dtype)

    # Load into model
    load_result = model.load_state_dict(our_state_dict, strict=strict)

    if skipped_keys:
        logger.info(f"Skipped {len(skipped_keys)} audio keys (video_only=True)")

    if load_result.missing_keys:
        logger.warning(f"Missing keys: {load_result.missing_keys[:10]}...")
    if load_result.unexpected_keys:
        logger.warning(f"Unexpected keys: {load_result.unexpected_keys[:10]}...")

    logger.info(f"Loaded LTX-2 transformer: {model.get_num_params() / 1e9:.2f}B parameters")

    return model


def load_ltx2_from_diffusers(
    repo_or_path: str = "Lightricks/LTX-Video-2",
    dtype: torch.dtype = torch.bfloat16,
) -> LTX2Transformer:
    """
    Load LTX-2 transformer from HuggingFace diffusers checkpoint.

    This is a convenience wrapper that handles downloading from HuggingFace
    if needed.

    Args:
        repo_or_path: HuggingFace repo ID or local path
        dtype: Model dtype

    Returns:
        Loaded LTX2Transformer model
    """
    from pathlib import Path

    # Check if local path
    local_path = Path(repo_or_path)
    if local_path.exists():
        return load_ltx2_transformer(local_path, dtype=dtype)

    # Download from HuggingFace
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("huggingface_hub required. Install: pip install huggingface_hub")

    logger.info(f"Downloading {repo_or_path} from HuggingFace...")
    local_dir = snapshot_download(
        repo_or_path,
        allow_patterns=["transformer/*", "config.json"],
    )

    return load_ltx2_transformer(Path(local_dir) / "transformer", dtype=dtype)


def load_ltx2_transformer_quantized(
    path: Union[str, Path],
    precision: str = "fp8-quanto",
    dtype: torch.dtype = torch.bfloat16,
    video_only: bool = True,
    verbose: bool = True,
) -> LTX2Transformer:
    """
    Load and quantize LTX-2 transformer for memory-efficient inference.

    Uses block-by-block quantization strategy from LTX-2 official implementation.
    This enables loading the 13B model on 24GB GPUs like RTX 4090.

    Memory usage:
    - bf16 (default): ~26GB (won't fit on 24GB GPU)
    - fp8-quanto: ~13GB (fits on 24GB GPU with room for activations)
    - int8-quanto: ~13GB
    - int4-quanto: ~6.5GB

    Args:
        path: Path to checkpoint file or directory
        precision: Quantization precision (fp8-quanto, int8-quanto, int4-quanto)
        dtype: Original dtype before quantization (bf16 recommended)
        video_only: If True, skip audio weights
        verbose: Print progress during quantization

    Returns:
        Quantized LTX2Transformer model on CPU (call .to('cuda') after)

    Example:
        >>> model = load_ltx2_transformer_quantized("models/LTX-2/transformer/")
        >>> model = model.to("cuda")  # Now fits in 24GB VRAM

    Note:
        Requires optimum-quanto: pip install optimum-quanto
    """
    from llm_dit.utils.quantization import quantize_model, estimate_quantized_size

    # Load model to CPU first
    if verbose:
        logger.info(f"Loading model to CPU (dtype={dtype})")

    model = load_ltx2_transformer(
        path,
        dtype=dtype,
        device="cpu",
        video_only=video_only,
        strict=False,
    )

    # Estimate memory savings
    num_params = model.get_num_params()
    original_size = num_params * 2 / 1e9  # bf16
    quantized_size = estimate_quantized_size(num_params, precision)

    if verbose:
        logger.info(f"Model: {num_params / 1e9:.2f}B params")
        logger.info(f"Memory: {original_size:.1f}GB (bf16) → {quantized_size:.1f}GB ({precision})")

    # Quantize using block-by-block strategy
    model = quantize_model(
        model,
        precision=precision,
        quantize_activations=False,
        device="cuda",  # Quantize on GPU (fast)
        verbose=verbose,
    )

    return model


def get_model_info(path: Union[str, Path]) -> dict:
    """
    Get information about a checkpoint without loading it.

    Args:
        path: Path to checkpoint

    Returns:
        Dict with model info (num_params, dtype, config, etc.)
    """
    path = Path(path)
    config = load_config(path)

    # Calculate approximate parameter count
    num_layers = config.get("num_layers", 48)
    hidden_dim = config.get("num_attention_heads", 32) * config.get("attention_head_dim", 128)
    cross_dim = config.get("cross_attention_dim", 4096)

    # Rough estimate: ~800M params per layer + overhead
    estimated_params = num_layers * 800_000_000 + 500_000_000

    return {
        "config": config,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "cross_attention_dim": cross_dim,
        "estimated_params": estimated_params,
        "estimated_size_bf16_gb": estimated_params * 2 / 1e9,
    }
