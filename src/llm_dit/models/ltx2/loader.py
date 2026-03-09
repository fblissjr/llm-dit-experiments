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
    # Video input/output projections
    "proj_in.": "patchify_proj.",

    # Video timestep embedding (AdaLayerNormSingle)
    "time_embed.emb.timestep_embedder.": "adaln_single.emb.timestep_embedder.",
    "time_embed.linear.": "adaln_single.linear.",

    # Audio input/output projections
    "audio_proj_in.": "audio_patchify_proj.",

    # Audio timestep embedding
    "audio_time_embed.emb.timestep_embedder.": "audio_adaln_single.emb.timestep_embedder.",
    "audio_time_embed.linear.": "audio_adaln_single.linear.",

    # Cross-modal AdaLN modules
    "av_cross_attn_video_scale_shift.": "av_ca_video_scale_shift_adaln_single.",
    "av_cross_attn_audio_scale_shift.": "av_ca_audio_scale_shift_adaln_single.",
    "av_cross_attn_a2v_gate.": "av_ca_a2v_gate_adaln_single.",
    "av_cross_attn_v2a_gate.": "av_ca_v2a_gate_adaln_single.",

    # Attention norm naming (diffusers uses norm_q/norm_k, we use q_norm/k_norm)
    # This applies to all attention modules (video, audio, cross-modal)
    ".norm_q.": ".q_norm.",
    ".norm_k.": ".k_norm.",
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
    """Check if a key is for audio components.

    Filters out all audio-related weights when loading video-only model.
    These include cross-modal attention, audio FFN, scale/shift tables, etc.
    """
    # Direct audio prefixes
    if key.startswith("audio_"):
        return True
    # Cross-modal AV prefixes (diffusers format and our mapped format)
    if key.startswith(("av_cross_attn", "av_ca_")):
        return True
    # Audio in the key name (catches audio_attn, audio_to_video_attn, etc.)
    if "audio" in key:
        return True
    # Cross-modal attention variants (within transformer blocks)
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
    """Load model configuration from config.json or safetensors metadata.

    Resolution order:
      1. config.json in checkpoint directory (or parent if path is a file)
      2. Safetensors metadata 'config' key (extracts 'transformer' sub-dict)
      3. Hardcoded V2.3 defaults

    Returns:
        Configuration dictionary (flat key-value, ready for create_model_from_config).
    """
    config_path = path / "config.json" if path.is_dir() else path.parent / "config.json"

    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)

    # Try safetensors metadata
    sf_path = path if path.is_file() and path.suffix == ".safetensors" else None
    if sf_path is None:
        # Check parent dir for any safetensors with config metadata
        parent = path if path.is_dir() else path.parent
        candidates = sorted(parent.glob("*.safetensors"))
        for c in candidates:
            try:
                from safetensors import safe_open
                with safe_open(str(c), framework="pt") as f:
                    meta = f.metadata() or {}
                if "config" in meta:
                    sf_path = c
                    break
            except Exception:
                continue

    if sf_path is not None:
        try:
            from safetensors import safe_open
            with safe_open(str(sf_path), framework="pt") as f:
                meta = f.metadata() or {}
            if "config" in meta:
                full_config = json.loads(meta["config"])
                config = full_config.get("transformer", full_config)
                logger.info(f"Loaded config from safetensors metadata: {sf_path.name}")
                return config
        except Exception as e:
            logger.warning(f"Failed to read config from safetensors metadata: {e}")

    # Hardcoded V2.3 defaults (last resort)
    return {
        "num_attention_heads": 32,
        "attention_head_dim": 128,
        "in_channels": 128,
        "out_channels": 128,
        "num_layers": 48,
        "cross_attention_dim": 4096,
        "caption_channels": 3840,
        "rope_type": "split",
        "positional_embedding_theta": 10000.0,
        "positional_embedding_max_pos": [20, 2048, 2048],
        "timestep_scale_multiplier": 1000,
        "apply_gated_attention": True,
        "cross_attention_adaln": True,
        "audio_num_attention_heads": 32,
        "audio_attention_head_dim": 64,
        "audio_out_channels": 128,
        "audio_cross_attention_dim": 2048,
        "audio_positional_embedding_max_pos": [20],
        "frequencies_precision": "float64",
    }


def create_model_from_config(
    config: dict,
    dtype: torch.dtype = torch.bfloat16,
    model_type: LTXModelType = LTXModelType.VideoOnly,
    apply_gated_attention: bool = False,
    cross_attention_adaln: bool = False,
) -> LTX2Transformer:
    """
    Create LTX2Transformer from config dictionary.

    Args:
        config: Model configuration
        dtype: Model dtype
        model_type: Which variant to create (VideoOnly, AudioVideo, AudioOnly)
        apply_gated_attention: V2 per-head sigmoid gate on attention output.
        cross_attention_adaln: V2 AdaLN modulation on text cross-attention.

    Returns:
        Initialized model (random weights)
    """
    rope_type = LTXRopeType.SPLIT if config.get("rope_type") == "split" else LTXRopeType.INTERLEAVED

    # Resolve positional embedding params (safetensors metadata vs legacy keys)
    pos_max = config.get("positional_embedding_max_pos")
    if pos_max is None:
        pos_max = [
            config.get("pos_embed_max_pos", 20),
            config.get("base_height", 2048),
            config.get("base_width", 2048),
        ]

    rope_theta = config.get("positional_embedding_theta",
                            config.get("rope_theta", 10000.0))

    double_precision = config.get("rope_double_precision")
    if double_precision is None:
        double_precision = config.get("frequencies_precision", "") == "float64"

    # Resolve gated attention / cross_attention_adaln: explicit args override config
    gated = apply_gated_attention or config.get("apply_gated_attention", False)
    ca_adaln = cross_attention_adaln or config.get("cross_attention_adaln", False)

    # Audio parameters (only used when model_type includes audio)
    audio_kwargs = {}
    if model_type.is_audio_enabled():
        audio_pos_max = config.get("audio_positional_embedding_max_pos")
        if audio_pos_max is None:
            audio_pos_max = [config.get("audio_pos_embed_max_pos", 20)]

        audio_kwargs = dict(
            audio_num_attention_heads=config.get("audio_num_attention_heads", 32),
            audio_attention_head_dim=config.get("audio_attention_head_dim", 64),
            audio_in_channels=config.get("audio_in_channels", 128),
            audio_out_channels=config.get("audio_out_channels", 128),
            audio_cross_attention_dim=config.get("audio_cross_attention_dim", 2048),
            audio_positional_embedding_max_pos=audio_pos_max,
        )

    model = LTX2Transformer(
        model_type=model_type,
        num_attention_heads=config.get("num_attention_heads", 32),
        attention_head_dim=config.get("attention_head_dim", 128),
        in_channels=config.get("in_channels", 128),
        out_channels=config.get("out_channels", 128),
        num_layers=config.get("num_layers", 48),
        cross_attention_dim=config.get("cross_attention_dim", 4096),
        caption_channels=config.get("caption_channels", 3840),
        positional_embedding_theta=rope_theta,
        positional_embedding_max_pos=pos_max,
        timestep_scale_multiplier=config.get("timestep_scale_multiplier", 1000),
        use_middle_indices_grid=True,
        rope_type=rope_type,
        double_precision_rope=double_precision,
        apply_gated_attention=gated,
        cross_attention_adaln=ca_adaln,
        av_ca_timestep_scale_multiplier=config.get("av_ca_timestep_scale_multiplier", 1000.0),
        **audio_kwargs,
    )

    return model.to(dtype)


def load_ltx2_transformer(
    path: Union[str, Path],
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    video_only: bool = True,
    strict: bool = False,
    model_type: Optional[LTXModelType] = None,
) -> LTX2Transformer:
    """
    Load LTX-2 transformer from checkpoint.

    Loads weights from official LTX-2 checkpoints and maps them to our
    pure PyTorch implementation.

    Args:
        path: Path to checkpoint file or directory
        dtype: Model dtype (bf16 recommended)
        device: Device to load to initially (use 'cpu' then .to('cuda') for large models)
        video_only: If True, skip audio weights (ignored if model_type is set)
        strict: If True, raise error on missing/extra keys
        model_type: Explicit model type. If set, overrides video_only flag.

    Returns:
        Loaded LTX2Transformer model

    Example:
        # Load video-only model
        model = load_ltx2_transformer("models/LTX-2/transformer/")
        model = model.cuda()  # Move to GPU after loading

        # Load audio-video model
        model = load_ltx2_transformer(path, model_type=LTXModelType.AudioVideo)
    """
    path = Path(path)

    # Resolve model_type from video_only flag if not explicitly set
    if model_type is None:
        model_type = LTXModelType.VideoOnly if video_only else LTXModelType.AudioVideo
    video_only = not model_type.is_audio_enabled()

    # Load config
    config = load_config(path)
    logger.info(f"Loaded config: {config.get('num_layers', 48)} layers, "
                f"{config.get('num_attention_heads', 32)} heads")

    # Create model (V2.3 flags read from config metadata)
    model = create_model_from_config(
        config, dtype, model_type=model_type,
    )

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
        if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            logger.warning(f"FP8 tensor {our_key} loaded without scale dequantization -- "
                           "consider using transformer_file config for proper FP8 handling")
            our_state_dict[our_key] = tensor  # Preserve FP8, don't cast to BF16 without scales
        else:
            our_state_dict[our_key] = tensor.to(dtype)

    # Load into model (assign=True prevents silent fp8->bf16 cast)
    load_result = model.load_state_dict(our_state_dict, strict=strict, assign=True)

    if skipped_keys:
        logger.info(f"Skipped {len(skipped_keys)} audio keys (video_only=True)")

    if load_result.missing_keys:
        logger.warning(f"Missing keys in model (weights not loaded): {load_result.missing_keys[:10]}...")
    if load_result.unexpected_keys:
        # These are keys in the state dict that our model doesn't have parameters for
        # Usually indicates a key mapping issue or model architecture mismatch
        logger.warning(
            f"Unexpected keys in checkpoint (ignored): {load_result.unexpected_keys[:10]}... "
            f"({len(load_result.unexpected_keys)} total)"
        )

    logger.info(f"Loaded LTX-2 transformer: {model.get_num_params() / 1e9:.2f}B parameters")

    return model




def _attach_weight_scales(
    model: torch.nn.Module,
    weight_scales: Dict[str, torch.Tensor],
) -> int:
    """Attach per-tensor weight scales to nn.Linear modules as plain attributes.

    Scales are stored as plain attributes (not buffers/parameters) so they don't
    appear in state_dict(). This keeps the cache path clean -- weight_scales are
    stored separately in the cache dict and re-attached during reconstruction.

    Args:
        model: Model with nn.Linear layers.
        weight_scales: Dict mapping weight param names (e.g.
            "transformer_blocks.0.attn1.to_q.weight") to scale tensors.

    Returns:
        Number of scales attached.
    """
    if not weight_scales:
        return 0

    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        weight_key = f"{name}.weight"
        if weight_key in weight_scales:
            module._weight_scale = weight_scales[weight_key]  # type: ignore[attr-defined]
            count += 1

    if count != len(weight_scales):
        logger.warning(
            f"Weight scale mismatch: {len(weight_scales)} scales provided, "
            f"{count} attached to nn.Linear modules"
        )

    return count


def load_ltx2_transformer_fp8_cast(
    path: Union[str, Path],
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    video_only: bool = True,
    model_type: Optional[LTXModelType] = None,
) -> LTX2Transformer:
    """Load LTX-2.3 transformer with fp8-cast (official approach).

    Keeps FP8 weights as-is and patches nn.Linear.forward to upcast per-forward.
    Scale tensors are skipped -- the calibrated FP8 values in the checkpoint
    are used directly. This avoids the dequant->bf16->requant cycle.

    Peak memory: ~12GB (FP8 weights stay FP8, only one layer upcasted at a time).

    Args:
        path: Path to FP8 safetensors file.
        dtype: Dtype for non-quantized params (norms, embeddings). bf16 recommended.
        device: Device to load to.
        video_only: If True, skip audio weights.
        model_type: Explicit model type override.

    Returns:
        LTX2Transformer with fp8 weights and patched forward methods.
    """
    from llm_dit.quantization.fp8_cast import amend_forward_with_upcast
    from llm_dit.utils.meta_init import meta_init

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"FP8 checkpoint not found: {path}")

    if model_type is None:
        model_type = LTXModelType.VideoOnly if video_only else LTXModelType.AudioVideo
    video_only = not model_type.is_audio_enabled()

    logger.info(f"Loading FP8-cast transformer from {path} (model_type={model_type.value})")

    # Load raw state dict
    raw_sd = load_safetensors(path, device="cpu")

    # Filter to transformer keys, skip scales + non-transformer components
    prefix = "model.diffusion_model."
    non_transformer_prefixes = (
        "model.vae.", "model.vocoder.", "model.audio_codec.",
        "vae.", "vocoder.", "audio_codec.",
    )

    our_sd: Dict[str, torch.Tensor] = {}
    weight_scales: Dict[str, torch.Tensor] = {}
    skipped = {"audio": 0, "non_transformer": 0, "input_scale": 0}

    for key, tensor in raw_sd.items():
        if key.startswith(non_transformer_prefixes):
            skipped["non_transformer"] += 1
            continue

        stripped_key = key[len(prefix):] if key.startswith(prefix) else key

        if video_only and is_audio_key(stripped_key):
            skipped["audio"] += 1
            continue

        # Collect weight_scale for per-tensor dequantization (scaled FP8 checkpoints).
        # input_scale is only used by fp8-scaled-mm (TensorRT-LLM) -- skip it.
        if stripped_key.endswith(".weight_scale"):
            # Map the corresponding weight key name
            weight_key = stripped_key.replace(".weight_scale", ".weight")
            our_weight_key = map_key(weight_key)
            weight_scales[our_weight_key] = tensor
            continue
        if stripped_key.endswith(".input_scale"):
            skipped["input_scale"] += 1
            continue

        our_key = map_key(stripped_key)

        if tensor.dtype == torch.float8_e4m3fn:
            our_sd[our_key] = tensor  # Keep fp8 as-is
        else:
            our_sd[our_key] = tensor.to(dtype)

    logger.info(
        f"FP8-cast: {len(our_sd)} keys loaded, "
        f"{len(weight_scales)} weight_scales collected, "
        f"skipped {skipped['input_scale']} input_scales, "
        f"{skipped['audio']} audio, {skipped['non_transformer']} non-transformer"
    )

    # Create model shell with meta_init (zero memory)
    config = load_config(path)
    with meta_init():
        model = create_model_from_config(
            config, dtype, model_type=model_type,
        )

    # Load mixed-dtype state dict (fp8 linears + bf16 norms/embeddings)
    load_result = model.load_state_dict(our_sd, strict=False, assign=True)
    if load_result.missing_keys:
        logger.warning(f"Missing keys: {load_result.missing_keys[:10]}... ({len(load_result.missing_keys)} total)")
    if load_result.unexpected_keys:
        logger.warning(f"Unexpected keys: {load_result.unexpected_keys[:10]}... ({len(load_result.unexpected_keys)} total)")

    # Attach weight_scale to each nn.Linear for scaled FP8 dequantization.
    # The per-forward upcast multiplies: w_bf16 = fp8_weight.to(bf16) * weight_scale
    # Stored as plain attributes (not buffers/params) to avoid polluting state_dict.
    # The cache path stores these separately in the cache dict.
    scales_attached = _attach_weight_scales(model, weight_scales)
    if scales_attached > 0:
        logger.info(f"FP8-cast: attached {scales_attached} weight_scale values to nn.Linear layers")

    # Patch nn.Linear forwards for per-forward upcast
    patched = amend_forward_with_upcast(model)
    logger.info(f"FP8-cast: {patched} linear layers patched for per-forward upcast")

    fp8_params = sum(1 for p in model.parameters() if p.dtype == torch.float8_e4m3fn)
    bf16_params = sum(1 for p in model.parameters() if p.dtype in (torch.bfloat16, torch.float32))
    logger.info(f"Loaded LTX-2.3 transformer (fp8-cast): {fp8_params} fp8 + {bf16_params} bf16 params")

    if device != "cpu":
        model = model.to(device)

    return model




