"""
LTX-2 Pure PyTorch Generation Module.

Last Updated: 2026-01-19

Pure PyTorch implementation of the LTX-2 diffusion generation loop.
Used by both the pipeline and experiment infrastructure.

This module provides the core generation logic without diffusers dependency:
- Sigma schedule generation via LTX2Scheduler
- Euler denoising loop with CFG
- VAE decoding with our ported VideoDecoder

Original source: https://github.com/Lightricks/LTX-2
License: LTX-2 Community License
Copyright (c) 2025 Lightricks Ltd.

Memory Optimization (2026-01-19):
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True reduces fragmentation
- Native FP8 quantization eliminates quanto's frozen buffer memory leak
- Periodic cleanup in denoising loop prevents activation accumulation
"""

# CRITICAL: Set CUDA allocator config BEFORE importing torch
# This reduces memory fragmentation significantly
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from tqdm import tqdm

from llm_dit.conditioning import (
    ConditioningItem,
    LatentState,
    post_process_latent,
    timesteps_from_mask,
)
from llm_dit.models.ltx2 import (
    LTX2TextConnectors,
    LTX2Transformer,
    Modality,
    VideoDecoder,
)
from llm_dit.models.ltx2.constants import LTX2_DEFAULT_NEGATIVE_PROMPT
from llm_dit.schedulers import LTX2Scheduler
from llm_dit.quantization import QUANT_ALIASES
from llm_dit.utils.memory import cleanup_memory

logger = logging.getLogger(__name__)

# Diagnostic logger -- separate from main logger so it can be enabled independently
_diag = logging.getLogger(__name__ + ".diagnostics")


def _tensor_stats(name: str, t: torch.Tensor) -> str:
    """Format tensor stats for diagnostic logging."""
    f = t.float()
    return (
        f"[DIAG] {name}: shape={list(t.shape)}, dtype={t.dtype}, "
        f"mean={f.mean():.6f}, std={f.std():.6f}, "
        f"min={f.min():.6f}, max={f.max():.6f}"
    )


def _resolve_v23_component_path(model_path: Path, component: str) -> Path:
    """Resolve V2.3 component path with V1 fallback.

    V2.3 stores components as standalone safetensors files (e.g. ltx-2.3-video-vae.safetensors).
    V1 used subdirectories (e.g. model_path/vae/). This mirrors model_manager.py logic.
    """
    v23_names = {
        "vae": "ltx-2.3-video-vae.safetensors",
        "audio_vae": "ltx-2.3-audio-vae.safetensors",
        "vocoder": "ltx-2.3-vocoder.safetensors",
    }
    v23_name = v23_names.get(component)
    if v23_name:
        v23_path = model_path / v23_name
        if v23_path.exists():
            return v23_path
    return model_path / component


def _validate_two_stage_dimensions(height: int, width: int) -> None:
    """Validate that dimensions are divisible by 64 for two-stage generation.

    Two-stage halves both dimensions for stage 1 (height/2, width/2).
    Since the base model requires 32-divisibility, the full resolution
    must be 64-divisible so that the halved resolution is 32-divisible.
    """
    if height % 64 != 0 or width % 64 != 0:
        raise ValueError(
            f"Two-stage requires height and width divisible by 64, "
            f"got {height}x{width}"
        )


def _resolve_quantize(quantize: str) -> tuple[bool, str]:
    """Normalize quantize string shorthand to (should_quantize, precision) tuple.

    Handles aliases like "fp8" -> "fp8-dynamic" and falsy values.

    Args:
        quantize: Quantization method string. "none", "", or None disables
            quantization. "fp8" is aliased to "fp8-dynamic".

    Returns:
        (should_quantize, precision): Whether to quantize and the resolved method.
    """
    if quantize in (None, "", "none"):
        return False, "none"
    precision = QUANT_ALIASES.get(quantize, quantize)
    return True, precision


def _normalize_lora_args(
    lora_path: Optional[Union[str, Path, List[Union[str, Path]]]],
    lora_scale: Optional[Union[float, List[float]]],
) -> tuple[Optional[List[str]], Optional[List[float]]]:
    """Normalize LoRA path/scale args into parallel lists.

    Converts the flexible API-facing args (single or list, str or Path) into
    canonical (list[str], list[float]) form. Also validates length consistency.

    Returns:
        (paths, scales): Both None if no LoRA, or parallel lists of equal length.
    """
    if lora_path is None:
        return None, None

    if isinstance(lora_path, (str, Path)):
        paths = [str(lora_path)]
    else:
        paths = [str(p) for p in lora_path]

    if lora_scale is None:
        scales = [1.0] * len(paths)
    elif isinstance(lora_scale, (int, float)):
        scales = [float(lora_scale)] * len(paths)
    else:
        scales = list(lora_scale)

    if len(paths) != len(scales):
        raise ValueError(
            f"Number of LoRA paths ({len(paths)}) must match "
            f"number of scales ({len(scales)})"
        )

    return paths, scales


def _apply_distilled_lora_fp8(
    model: "LTX2Transformer",
    lora_path: str,
    scale: float,
) -> None:
    """Apply distilled LoRA to a live model with native fp8 weights.

    Uses state-dict-level fusion with proper FP8 dequantization:
    1. Extract weight_scales from model (per-tensor scale factors)
    2. Fuse LoRA: dequantize fp8 -> add delta in f32 -> re-quantize with new scale
    3. Reload state dict with assign=True
    4. Re-attach updated weight_scales

    The existing patched forwards (from amend_forward_with_upcast during cache
    reconstruction) survive load_state_dict(assign=True) because it replaces
    parameters, not forward methods -- closures access layer.weight at call time.
    """
    from llm_dit.models.ltx2.loader import _attach_weight_scales
    from llm_dit.utils.lora import fuse_lora_to_state_dict

    # Extract weight_scales (plain attributes on nn.Linear, not in state_dict)
    weight_scales: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if hasattr(module, "_weight_scale"):
            weight_scales[f"{name}.weight"] = module._weight_scale

    sd = model.state_dict()
    sd, new_scales = fuse_lora_to_state_dict(
        sd, [lora_path], [scale], weight_scales=weight_scales,
    )
    model.load_state_dict(sd, assign=True)

    # Re-attach updated weight_scales (scale factors change after re-quantization)
    if new_scales:
        attached = _attach_weight_scales(model, new_scales)
        logger.info(
            f"FP8-cast distilled LoRA: sd-level fusion applied "
            f"({attached} weight_scales updated)"
        )
    else:
        logger.info("FP8-cast distilled LoRA: sd-level fusion applied")


def _load_transformer_and_lora(
    *,
    cached_transformer: Optional[dict],
    model_path: Path,
    transformer_file: str,
    dtype: torch.dtype,
    transformer_device: str,
    effective_quantize: bool,
    effective_precision: str,
    granularity: str,
    lora_paths: Optional[List[str]],
    lora_scales: Optional[List[float]],
    video_only: bool = True,
) -> "LTX2Transformer":
    """Load transformer via cache or disk and apply LoRA.

    Two-branch dispatch:
    1. Cached: reconstruct from pinned state_dict (LoRA fused at sd level for fp8-cast)
    2. Disk: load from safetensors file (fallback)

    Returns:
        Loaded transformer model.
    """
    if cached_transformer is not None:
        model = _reconstruct_transformer_from_cache(
            cached_transformer, dtype, transformer_device,
            effective_quantize, effective_precision, granularity,
            lora_paths=lora_paths, lora_scales=lora_scales,
        )
    else:
        # Load from disk (fallback when no cache)
        from llm_dit.models.ltx2 import load_ltx2_transformer

        load_device = "cpu" if effective_quantize else transformer_device

        if transformer_file:
            tf_path = model_path / transformer_file
            if not tf_path.exists():
                logger.warning(f"transformer_file '{transformer_file}' not found at {tf_path}, "
                               "falling back to transformer/ directory")
                tf_path = model_path / "transformer"
        else:
            tf_path = model_path / "transformer"

        is_fp8_file = tf_path.is_file() and "fp8" in tf_path.name.lower()
        if is_fp8_file:
            from llm_dit.models.ltx2 import load_ltx2_transformer_fp8_cast
            model = load_ltx2_transformer_fp8_cast(
                tf_path, dtype=dtype, device=load_device, video_only=video_only,
            )
        else:
            model = load_ltx2_transformer(
                tf_path, dtype=dtype, device=load_device, video_only=video_only,
            )

            if effective_quantize and effective_precision != "none":
                from llm_dit.quantization import quantize_component
                model, stats = quantize_component(  # type: ignore[assignment]
                    model, method=effective_precision, component_type="transformer",
                    granularity=granularity,
                )
                logger.info(
                    f"Transformer quantized: {stats['quantized_layers']}/{stats['total_layers']} layers "
                    f"({effective_precision}, granularity={granularity})"
                )

        if load_device == "cpu":
            model = model.to(transformer_device)

    # Apply LoRA if specified (skip if already fused at state-dict level)
    lora_already_fused = getattr(model, "_lora_fused_at_sd_level", False)
    if lora_paths and lora_scales and not lora_already_fused:
        from llm_dit.utils.lora import load_lora as _load_lora
        total_updated = 0
        for path, scale in zip(lora_paths, lora_scales):
            updated = _load_lora(
                model, path, scale=scale,
                device=transformer_device, dtype=dtype,
            )
            total_updated += updated
        logger.info(f"LoRA loading complete: {total_updated} layers updated")

    return model


def _reconstruct_transformer_from_cache(
    cached_transformer: dict,
    dtype: torch.dtype,
    transformer_device: torch.device | str,
    effective_quantize: bool,
    effective_precision: str,
    granularity: str,
    lora_paths: Optional[List[str]] = None,
    lora_scales: Optional[List[float]] = None,
) -> "LTX2Transformer":
    """Reconstruct a transformer model from a cached state_dict.

    The cache dict is self-describing: it carries a 'video_only' flag so the
    correct model architecture (BasicTransformerBlock vs BasicAVTransformerBlock)
    is created. Legacy caches without the flag default to video-only.

    When 'fp8_cast' is True, the cached state dict contains mixed-dtype tensors
    (fp8 linears + bf16 norms/embeddings). Instead of torchao quantization, we
    patch nn.Linear forwards for per-forward upcast (official Lightricks approach).

    For fp8_cast models with LoRA, fusion happens at the state-dict level BEFORE
    load_state_dict (matching official LTX-2 fuse_loras.py). This avoids the
    unsupported fp8+bf16 addition that crashes fuse_lora_to_base_model().

    Args:
        cached_transformer: Dict with "config", "state_dict", and optionally
            "video_only" (defaults True for backward compat) and "fp8_cast".
        dtype: Model dtype (usually torch.bfloat16).
        transformer_device: Target device for the model after construction.
        effective_quantize: Whether to quantize after loading.
        effective_precision: Quantization method string (e.g. "fp8-dynamic").
        granularity: Quantization granularity (e.g. "per-row").
        lora_paths: Optional list of LoRA .safetensors paths to fuse (fp8_cast only).
        lora_scales: Optional list of scale factors, same length as lora_paths.

    Returns:
        Fully loaded (and optionally quantized) transformer on transformer_device.
    """
    from llm_dit.models.ltx2.loader import LTXModelType, create_model_from_config
    from llm_dit.utils.meta_init import meta_init

    is_fp8_cast = cached_transformer.get("fp8_cast", False)
    logger.info(
        "Using cached transformer weights, reconstructing model..."
        + (" (fp8-cast)" if is_fp8_cast else "")
    )
    cache_video_only = cached_transformer.get("video_only", True)
    model_type = LTXModelType.VideoOnly if cache_video_only else LTXModelType.AudioVideo

    # For fp8_cast models, fuse LoRA into the state dict BEFORE load_state_dict.
    # Native fp8 tensors can't do fp8+bf16 addition, so we dequantize using
    # weight_scales, fuse in f32, and re-quantize (matching official LTX-2
    # _fuse_delta_with_scaled_fp8 pattern).
    sd = cached_transformer["state_dict"]
    weight_scales = cached_transformer.get("weight_scales", {})
    lora_fused_at_sd_level = False
    if is_fp8_cast and lora_paths and lora_scales:
        from llm_dit.utils.lora import fuse_lora_to_state_dict
        sd, weight_scales = fuse_lora_to_state_dict(
            sd, lora_paths, lora_scales, weight_scales=weight_scales,
        )
        lora_fused_at_sd_level = True
        logger.info(f"LoRA fused at state-dict level ({len(lora_paths)} LoRA(s))")

    with meta_init():
        model = create_model_from_config(
            cached_transformer["config"], dtype, model_type=model_type,
        )
    model.load_state_dict(sd, assign=True)

    if is_fp8_cast:
        # Re-attach weight scales (updated by LoRA fusion or original from cache)
        if weight_scales:
            from llm_dit.models.ltx2.loader import _attach_weight_scales
            attached = _attach_weight_scales(model, weight_scales)
            logger.info(f"FP8-cast: attached {attached} weight_scales")

        # FP8-cast: patch forwards for per-forward upcast (no torchao)
        from llm_dit.quantization.fp8_cast import amend_forward_with_upcast
        patched = amend_forward_with_upcast(model)
        logger.info(f"FP8-cast: {patched} linear layers patched for per-forward upcast")

    model = model.to(transformer_device)

    # FP8 preservation guard: verify fp8 weights survived device transfer
    if is_fp8_cast:
        has_fp8 = any(p.dtype == torch.float8_e4m3fn for p in model.parameters())
        if not has_fp8:
            raise RuntimeError(
                f"FP8 weights lost during device transfer to {transformer_device} -- "
                "all parameters are now bf16. This hardware may not support float8. "
                "Use quantize='fp8-dynamic' (torchao runtime quantization) as an alternative."
            )

    # Tag model so callers know LoRA was already applied
    model._lora_fused_at_sd_level = lora_fused_at_sd_level  # type: ignore[attr-defined]

    return model


@dataclass
class GenerationConfig:
    """Configuration for pure PyTorch video generation."""

    num_frames: int = 33
    height: int = 512
    width: int = 768
    num_inference_steps: int = 50
    guidance_scale: float = 3.0
    seed: Optional[int] = None
    fps: float = 24.0  # LTX-2.3 native frame rate

    # Scheduler parameters
    base_shift: float = 0.95
    max_shift: float = 2.05
    stretch: bool = True
    terminal: float = 0.1

    # VAE normalization parameters (from diffusers model)
    latents_mean: Optional[torch.Tensor] = None
    latents_std: Optional[torch.Tensor] = None
    scaling_factor: float = 1.0

    @property
    def latent_dims(self) -> Tuple[int, int, int]:
        """Compute latent dimensions from video size."""
        t_latent = (self.num_frames - 1) // 8 + 1
        h_latent = self.height // 32
        w_latent = self.width // 32
        return t_latent, h_latent, w_latent

    @property
    def num_tokens(self) -> int:
        """Total number of latent tokens."""
        t, h, w = self.latent_dims
        return t * h * w


def create_position_indices(
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    device: torch.device,
    fps: float = 24.0,
    scale_factors: tuple[int, int, int] = (8, 32, 32),
    causal_fix: bool = True,
) -> torch.Tensor:
    """
    Create 3D position indices [B, 3, T, 2] in PIXEL space for RoPE.

    LTX-2's 3D RoPE requires positions in pixel coordinates, not latent indices.
    This function creates position bounds [start, end) for each patch, scaled
    to pixel space by multiplying by scale_factors.

    LTX-2 VAE compression: 32x spatial, 8x temporal.
    Reference: coderef/LTX-2/ltx-core/components/patchifiers.py:get_pixel_coords

    Args:
        batch_size: Batch size
        num_frames: Number of video frames
        height: Video height in pixels
        width: Video width in pixels
        device: Target device
        fps: Frames per second (for temporal scaling, default 24)
        scale_factors: (time, height, width) compression factors (default 8, 32, 32)
        causal_fix: Apply causal fix for temporal dimension (default True)

    Returns:
        Position indices tensor [B, 3, T, 2] where:
        - T = t_latent * h_latent * w_latent (flattened patches)
        - Last dim is [start, end] bounds in PIXEL coordinates
        - Temporal dim (positions[:, 0]) is scaled to seconds
    """
    # Calculate latent dimensions
    t_latent = (num_frames - 1) // 8 + 1
    h_latent = height // 32
    w_latent = width // 32

    # Create meshgrid of latent position indices (0, 1, 2...)
    t_indices = torch.arange(t_latent, device=device, dtype=torch.float32)
    h_indices = torch.arange(h_latent, device=device, dtype=torch.float32)
    w_indices = torch.arange(w_latent, device=device, dtype=torch.float32)

    # Create 3D grid: [t_latent, h_latent, w_latent]
    grid_t, grid_h, grid_w = torch.meshgrid(t_indices, h_indices, w_indices, indexing="ij")

    # Create start and end positions (each patch spans [start, start+1) in latent space)
    # Shape: [3, t_latent, h_latent, w_latent]
    patch_starts = torch.stack([grid_t, grid_h, grid_w], dim=0)
    patch_ends = patch_starts + 1.0

    # Stack start/end into bounds: [3, t_latent, h_latent, w_latent, 2]
    positions = torch.stack([patch_starts, patch_ends], dim=-1)

    # CRITICAL: Convert to pixel coordinates by multiplying by scale factors
    # This is the key fix - positions must be in pixel space for RoPE
    scale_tensor = torch.tensor(scale_factors, device=device, dtype=torch.float32)
    scale_tensor = scale_tensor.view(3, 1, 1, 1, 1)  # [3, 1, 1, 1, 1] for broadcast
    positions = positions * scale_tensor

    # Apply causal fix for temporal dimension (matches reference)
    # This adjusts temporal positions for causal attention
    if causal_fix:
        positions[0] = (positions[0] + 1 - scale_factors[0]).clamp(min=0)

    # Flatten spatial dims: [3, T, 2] where T = t_latent * h_latent * w_latent
    positions = positions.view(3, -1, 2)

    # Scale temporal positions to seconds (divide by fps)
    # This must happen AFTER pixel conversion
    positions[0] = positions[0] / fps

    # Expand for batch: [B, 3, T, 2]
    return positions.unsqueeze(0).expand(batch_size, -1, -1, -1)


def create_video_modality(
    latent: torch.Tensor,
    timestep: torch.Tensor,
    positions: torch.Tensor,
    prompt_embeds: torch.Tensor,
    sigma: Optional[torch.Tensor] = None,
    context_mask: Optional[torch.Tensor] = None,
) -> Modality:
    """
    Create Modality dataclass for transformer input.

    Args:
        latent: [B, T, D] latent tokens (D=128 for LTX-2)
        timestep: [B, T] per-token timesteps
        positions: [B, 3, T] position indices
        prompt_embeds: [B, seq_len, context_dim] text embeddings
        sigma: [B] scalar noise level for cross-attention AdaLN.
            If None, derived from timestep[:,0].
        context_mask: Optional [B, seq_len] attention mask

    Returns:
        Modality dataclass ready for transformer forward pass
    """
    if sigma is None:
        sigma = timestep[:, 0]
    return Modality(
        latent=latent,
        sigma=sigma,
        timesteps=timestep,
        positions=positions,
        context=prompt_embeds,
        enabled=True,
        context_mask=context_mask,
    )


def compute_audio_latent_frames(
    num_frames: int,
    fps: float = 24.0,
    sample_rate: int = 16000,
    hop_length: int = 160,
    downsample_factor: int = 4,
) -> int:
    """Compute number of audio latent frames from video parameters.

    Formula: round(video_duration_seconds * sample_rate / hop_length / downsample_factor)

    Args:
        num_frames: Number of video frames
        fps: Video frame rate (default 24.0)
        sample_rate: Audio sample rate in Hz (default 16000)
        hop_length: Mel spectrogram hop length (default 160)
        downsample_factor: Audio VAE temporal compression factor (default 4)

    Returns:
        Number of audio latent frames (= number of audio transformer tokens)
    """
    duration = num_frames / fps
    return round(duration * sample_rate / hop_length / downsample_factor)


def create_audio_position_indices(
    batch_size: int,
    audio_latent_frames: int,
    device: torch.device,
) -> torch.Tensor:
    """Create 1D temporal position indices [B, 1, T, 2] for audio RoPE.

    Uses AudioPatchifier.get_patch_grid_bounds() which returns timestamps
    in seconds, aligned with the causal VAE.

    Args:
        batch_size: Batch size
        audio_latent_frames: Number of audio latent frames
        device: Target device

    Returns:
        Position indices tensor [B, 1, T, 2] where T = audio_latent_frames
    """
    from llm_dit.models.ltx2.audio_vae.patchifier import AudioPatchifier
    from llm_dit.models.ltx2.audio_vae.types import AudioLatentShape

    patchifier = AudioPatchifier(patch_size=1)
    audio_shape = AudioLatentShape(
        batch=batch_size, channels=8, frames=audio_latent_frames, mel_bins=16,
    )
    return patchifier.get_patch_grid_bounds(audio_shape, device=device)


def create_audio_modality(
    latent: torch.Tensor,
    timestep: torch.Tensor,
    positions: torch.Tensor,
    prompt_embeds: torch.Tensor,
    sigma: Optional[torch.Tensor] = None,
    context_mask: Optional[torch.Tensor] = None,
) -> Modality:
    """Create Modality dataclass for audio transformer input.

    Args:
        latent: [B, T, D] audio latent tokens (D=128 = 8 channels * 16 mel_bins)
        timestep: [B, T] per-token timesteps (same sigmas as video)
        positions: [B, 1, T, 2] temporal position indices
        prompt_embeds: [B, seq_len, context_dim] audio text embeddings
        sigma: [B] scalar noise level for cross-attention AdaLN.
            If None, derived from timestep[:,0].
        context_mask: Optional [B, seq_len] attention mask

    Returns:
        Modality dataclass ready for transformer forward pass
    """
    if sigma is None:
        sigma = timestep[:, 0]
    return Modality(
        latent=latent,
        sigma=sigma,
        timesteps=timestep,
        positions=positions,
        context=prompt_embeds,
        enabled=True,
        context_mask=context_mask,
    )


def generate_video(
    model: LTX2Transformer,
    prompt_embeds: torch.Tensor,
    config: GenerationConfig,
    vae: Optional[torch.nn.Module] = None,  # VideoDecoder or diffusers AutoencoderKLLTXVideo
    connectors: Optional[LTX2TextConnectors] = None,
    attention_mask: Optional[torch.Tensor] = None,
    conditioning: Optional[List[ConditioningItem]] = None,
    callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    use_progress: bool = True,  # Use rich SamplingProgress display
    debug_latents: bool = False,
) -> torch.Tensor:
    """
    Generate video using pure PyTorch diffusion loop.

    This is the main generation entry point that handles:
    1. Connector processing (if embeddings are packed format)
    2. Noise initialization (with optional conditioning)
    3. Sigma schedule computation
    4. CFG-guided denoising loop (with per-token timesteps if conditioning)
    5. Optional VAE decoding

    Args:
        model: LTX2Transformer model
        prompt_embeds: Text embeddings
            - [B, T, 188160]: Packed format, requires connectors
            - [B, T, 3840]: Projected format, goes directly to transformer
        config: Generation configuration
        vae: Optional VideoDecoder for latent-to-pixel conversion.
            If None, returns latents.
        connectors: LTX2TextConnectors for processing packed embeddings.
            Required if prompt_embeds is [B, T, 188160].
        attention_mask: [B, T] attention mask for connector processing
        conditioning: Optional list of ConditioningItems for I2V or video continuation.
            - VideoConditionByLatentIndex: Replaces tokens at frame index (I2V)
            - VideoConditionByKeyframeIndex: Appends keyframe tokens (continuation)
        callback: Optional callback(step, total_steps, latents) for progress
        device: Override device (default: model device)
        dtype: Override dtype (default: model dtype)
        debug_latents: If True, log detailed latent/velocity statistics at key denoising steps.
            Useful for debugging generation quality issues without changing global log level.

    Returns:
        If vae provided: Video tensor [B, C, F, H, W] or [F, H, W, C] uint8
        If vae=None: Latent tensor [B, D, T_lat, H_lat, W_lat]

    Example with I2V conditioning:
        >>> from llm_dit.conditioning import VideoConditionByLatentIndex
        >>> image_latent = vae.encode(image)
        >>> cond = VideoConditionByLatentIndex(image_latent, latent_idx=0, strength=1.0)
        >>> video = generate_video(model, prompt, config, conditioning=[cond])
    """
    # Resolve device and dtype
    if device is None:
        device = next(model.parameters()).device
    if dtype is None:
        dtype = next(model.parameters()).dtype

    # Setup generator for reproducibility
    generator = None
    if config.seed is not None:
        generator = torch.Generator(device=device).manual_seed(config.seed)

    # Get dimensions
    t_latent, h_latent, w_latent = config.latent_dims
    num_tokens = config.num_tokens

    # =========================================================================
    # Step 0: Process embeddings through connectors if needed
    # =========================================================================
    embed_dim = prompt_embeds.shape[-1]
    if embed_dim == 188160:
        if connectors is None:
            raise RuntimeError(
                "Packed embeddings [B, T, 188160] require connectors. "
                "Pass connectors parameter or use projected embeddings."
            )
        # Process through connectors
        if attention_mask is None:
            attention_mask = torch.ones(
                prompt_embeds.shape[0],
                prompt_embeds.shape[1],
                device=prompt_embeds.device,
                dtype=prompt_embeds.dtype,
            )
        video_embeds, _, _ = connectors(
            prompt_embeds.to(device, dtype),
            attention_mask.to(device),
            additive_mask=False,
        )
        prompt_embeds = video_embeds
    else:
        prompt_embeds = prompt_embeds.to(device, dtype)

    # =========================================================================
    # Step 1: Initialize LatentState with noise or conditioning
    # =========================================================================
    if conditioning:
        # Use LatentState for conditioning support
        state = LatentState.create(
            shape=(1, num_tokens, 128),
            num_frames=config.num_frames,
            height=config.height,
            width=config.width,
            device=device,
            dtype=dtype,
        )

        # Apply all conditioning items
        for cond_item in conditioning:
            state = cond_item.apply_to(state)

        # Add noise respecting the denoise mask
        state = state.add_noise(generator=generator, noise_scale=1.0)

        # Extract for denoising loop
        latents = state.latent
        positions = state.positions
        denoise_mask = state.denoise_mask
        clean_latent = state.clean_latent
        # Update num_tokens in case conditioning appended tokens
        num_tokens = latents.shape[1]
    else:
        # Standard T2V path - pure noise
        latents = torch.randn(
            (1, num_tokens, 128),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        positions = create_position_indices(
            batch_size=1,
            num_frames=config.num_frames,
            height=config.height,
            width=config.width,
            device=device,
            fps=config.fps,
            scale_factors=(8, 32, 32),  # LTX-2 VAE compression factors
            causal_fix=True,
        )
        denoise_mask = None
        clean_latent = None

    # =========================================================================
    # Step 3: Get sigma schedule from scheduler
    # =========================================================================
    scheduler = LTX2Scheduler()

    sigmas = scheduler.execute(
        steps=config.num_inference_steps,
        latent=None,  # Use reference default (4096 tokens)
        max_shift=config.max_shift,
        base_shift=config.base_shift,
        stretch=config.stretch,
        terminal=config.terminal,
    )
    sigmas = sigmas.to(device, dtype)

    # =========================================================================
    # Step 4: Denoising loop (Euler method with velocity prediction)
    # =========================================================================
    # Clear memory before starting denoising to maximize available VRAM
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # CRITICAL: torch.no_grad() prevents autograd from holding intermediate tensors
    # Without this, memory usage during forward pass spikes dramatically
    model.train(False)  # PyTorch inference mode

    # Set up progress display - prefer rich SamplingProgress, fall back to tqdm
    num_steps = len(sigmas) - 1
    progress_mgr = None
    if use_progress:
        try:
            from llm_dit.utils.progress import SamplingProgress
            progress_mgr = SamplingProgress(num_steps=num_steps, desc="Denoising")
        except ImportError:
            pass  # Fall back to tqdm

    with torch.no_grad():
      # Enter progress context if available
      if progress_mgr is not None:
          progress_mgr.__enter__()

      step_iter = range(num_steps)
      if progress_mgr is None:
          step_iter = tqdm(step_iter, desc="Denoising")

      denoise_start = time.perf_counter()
      step_times: list[float] = []

      for i in step_iter:
        step_start = time.perf_counter()
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        # Timestep for model in [0, 1] range - sigma values directly
        # LTX-2 uses sigma values in [0, 1] as timesteps, NOT scaled by 1000
        if denoise_mask is not None:
            # Per-token timesteps: conditioned regions get lower timesteps
            timestep = timesteps_from_mask(denoise_mask, sigma)
            timestep = timestep.squeeze(-1)  # [B, T, 1] -> [B, T] for Modality
        else:
            # Uniform timesteps for standard T2V - sigma directly, no scaling!
            timestep = sigma.expand(1, num_tokens)

        # Track latent inter-token variation at key steps
        if debug_latents and i in [0, 20, len(sigmas) - 2]:
            latent_inter_token = latents.std(dim=1).mean()  # Variation across tokens
            latent_overall_std = latents.std()
            logger.debug(f"Step {i} latent stats: inter-token std={latent_inter_token:.4f}, overall std={latent_overall_std:.4f}")

        # Classifier-free guidance
        if config.guidance_scale > 1.0:
            # Set debug step for attention diagnostics (only on step 0)
            for block in model.transformer_blocks:
                block._debug_step = i
            # Enable cross-attention KV debug for block 0 at key steps
            if debug_latents and i in [0, 20, len(sigmas) - 2]:
                model.transformer_blocks[0].attn2._debug_cross_attn = True
                model.transformer_blocks[0].attn2._debug_cross_attn_post = True
                logger.debug(f"Step {i}: cross-attention debug enabled")

            # Unconditional pass (zero embeddings)
            uncond_embeds = torch.zeros_like(prompt_embeds)
            # Log embedding statistics at step 0 for CFG verification
            if debug_latents and i == 0:
                logger.debug(f"CFG embeddings: cond mean={prompt_embeds.mean():.4f} std={prompt_embeds.std():.4f}, uncond mean={uncond_embeds.mean():.4f} std={uncond_embeds.std():.4f}")
            uncond_modality = create_video_modality(latents, timestep, positions, uncond_embeds)
            velocity_uncond, _ = model(video=uncond_modality)
            del uncond_modality, uncond_embeds  # Free memory before conditional pass
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # Conditional pass
            if debug_latents and i in [0, 20, len(sigmas) - 2]:
                logger.debug(f"Step {i}: conditional pass")
                model.transformer_blocks[0].attn2._debug_cross_attn = True  # Re-enable for cond pass
                model.transformer_blocks[0].attn2._debug_cross_attn_post = True
                model.args_preprocessor._debug_context = True  # Caption projection debug
            cond_modality = create_video_modality(latents, timestep, positions, prompt_embeds)
            velocity_cond, _ = model(video=cond_modality)
            del cond_modality  # Free modality tensor

            # CFG blend
            velocity = velocity_cond + (config.guidance_scale - 1.0) * (
                velocity_cond - velocity_uncond
            )
            # Log velocity statistics at key steps
            if debug_latents and i in [0, len(sigmas) // 2, len(sigmas) - 2]:
                logger.debug(
                    f"Step {i} sigma={sigma:.4f}: "
                    f"v_cond mean={velocity_cond.mean():.4f} std={velocity_cond.std():.4f}, "
                    f"v_uncond mean={velocity_uncond.mean():.4f} std={velocity_uncond.std():.4f}, "
                    f"v_guided mean={velocity.mean():.4f} std={velocity.std():.4f} range=[{velocity.min():.4f}, {velocity.max():.4f}]"
                )
            del velocity_uncond, velocity_cond  # Free after blend
        else:
            modality = create_video_modality(latents, timestep, positions, prompt_embeds)
            velocity, _ = model(video=modality)

        # Euler step: x_{t-1} = x_t + v * dt
        # Use float32 for numerical stability (LTX-2 reference does this)
        dt = sigma_next - sigma
        denoised = (latents.float() + velocity.float() * dt).to(latents.dtype)

        # Post-process: blend with clean_latent based on denoise_mask
        # This preserves conditioned regions (mask=0) while denoising others
        if denoise_mask is not None and clean_latent is not None:
            latents = post_process_latent(denoised, denoise_mask, clean_latent)
        else:
            latents = denoised

        # Clear memory between steps to prevent fragmentation
        del velocity
        # Periodic cleanup every 5 steps (balances speed vs memory)
        # More frequent cleanup can slow down generation but prevents OOM
        if (i + 1) % 5 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Step timing
        step_elapsed = time.perf_counter() - step_start
        step_times.append(step_elapsed)
        if i == 0 and step_elapsed > 5.0:
            logger.info(f"[Denoise:Step 0] {step_elapsed:.1f}s (torch.compile warmup -- subsequent steps will be fast)")
        elif i == 0 or (i + 1) % 10 == 0 or i == num_steps - 1:
            logger.info(f"[Denoise:Step {i}] {step_elapsed:.2f}s")

        # Progress callback and progress manager update
        if callback is not None:
            callback(i + 1, num_steps, latents)
        if progress_mgr is not None:
            progress_mgr.advance()

      # Denoising summary
      denoise_elapsed = time.perf_counter() - denoise_start
      logger.info(f"[Denoise] {num_steps} steps in {denoise_elapsed:.1f}s ({', '.join(f'{t:.2f}s' for t in step_times)})")

      # Exit progress context after loop
      if progress_mgr is not None:
          progress_mgr.__exit__(None, None, None)

    # =========================================================================
    # Step 5: Reshape latents for VAE decode
    # From [B, T, D] to [B, D, T_lat, H_lat, W_lat]
    # =========================================================================
    latents = latents.transpose(1, 2)  # [B, D, T]
    latents = latents.reshape(1, 128, t_latent, h_latent, w_latent)

    # =========================================================================
    # Step 6: VAE decode (if provided)
    # =========================================================================
    if vae is None:
        return latents

    # Decode to pixel space
    # Support both diffusers VAE (decode method) and our VAE (direct call)
    decode_start = time.perf_counter()
    with torch.no_grad():
        if hasattr(vae, "decode"):
            # Diffusers VAE interface - requires external denormalization
            if hasattr(vae, "latents_mean") and vae.latents_mean is not None:
                latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(device, dtype)
                latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(device, dtype)
                scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
                latents = latents * latents_std / scaling_factor + latents_mean
            video = vae.decode(latents).sample
        else:
            # Our VideoDecoder - handles denormalization internally via
            # per_channel_statistics.un_normalize(), so DO NOT denormalize here
            video = vae(latents)
    decode_elapsed = time.perf_counter() - decode_start
    if decode_elapsed > 5.0:
        logger.info(f"[Decode] VAE decode {decode_elapsed:.1f}s (torch.compile warmup -- subsequent decodes will be fast)")
    else:
        logger.info(f"[Decode] VAE decode {decode_elapsed:.2f}s")

    # Convert to [F, H, W, C] uint8 format
    video = video.squeeze(0).permute(1, 2, 3, 0)  # [B, C, T, H, W] -> [T, H, W, C]
    video = ((video + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)

    return video


def _maybe_enhance_prompt(
    text_encoder,
    prompt: str,
    callback: Optional[Callable],
    enhance: bool,
) -> str:
    """Optionally enhance a prompt via Gemma3 text generation.

    When enabled, uses the encoder's .generate() to expand a terse prompt into
    a detailed video description before encoding. Returns the original prompt
    unchanged when disabled.
    """
    if not enhance:
        return prompt

    from llm_dit.encoders.gemma3 import LTX2_T2V_SYSTEM_PROMPT, clean_enhanced_prompt

    logger.info("Enhancing prompt via Gemma3...")
    if callback:
        callback("enhancing", 0, 1)
    enhanced = text_encoder.generate(
        prompt=f"user prompt: {prompt}",
        system_prompt=LTX2_T2V_SYSTEM_PROMPT,
        max_new_tokens=512,
        temperature=0.7,
    )
    result = clean_enhanced_prompt(enhanced)
    logger.info(f"Enhanced prompt ({len(result)} chars): {result[:200]}...")
    if callback:
        callback("enhancing", 1, 1, enhanced_prompt=result)
    return result


def generate_video_with_offloading(
    prompt: str,
    config: GenerationConfig,
    model_path: Union[str, Path] = "models/LTX-2",
    text_encoder_path: Optional[Union[str, Path]] = None,
    precomputed_embeddings: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.bfloat16,
    callback: Optional[Callable[[str, int, int], None]] = None,
    gemma_variant: str = "bf16",
    use_progress: bool = True,
    debug_latents: bool = False,
    lora_path: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
    lora_scale: Optional[Union[float, List[float]]] = None,
    text_encoder_device: str = "cpu",
    transformer_device: str = "cuda",
    vae_device: str = "cuda",
    quantize: str = "fp8",
    granularity: str = "per-row",
    transformer_file: str = "",
    skip_cleanup: bool = False,
    enhance_prompt: bool = False,
    text_encoder: Optional[Any] = None,
    cached_transformer: Optional[dict] = None,
    cached_vae: Optional[Any] = None,
) -> torch.Tensor:
    """
    Generate video with sequential component offloading for 24GB GPUs.

    This function implements the LTX-2 memory strategy:
    1. Load text encoder -> encode prompt -> unload (skipped if precomputed_embeddings)
    2. Load transformer -> denoise -> unload
    3. Load VAE -> decode latents -> unload

    Each component is loaded and unloaded sequentially to stay within 24GB VRAM.
    With FP8-cast, the 22B model fits with room for activations.

    Args:
        prompt: Text prompt for generation (can be empty if using precomputed_embeddings)
        config: Generation configuration
        model_path: Path to LTX-2 model directory (contains transformer/, text_encoder/, vae/)
        text_encoder_path: Optional separate path for text encoder
        precomputed_embeddings: Optional pre-computed text embeddings [seq_len, 3840].
            If provided, skips text encoding entirely.
        dtype: Base dtype for loading (bf16 recommended)
        callback: Optional callback(stage, step, total) for progress
        gemma_variant: Gemma3 variant: bf16, 8bit, q4-qat
        use_progress: Use rich progress display for denoising
        debug_latents: Log detailed latent/velocity statistics at key denoising steps.
        lora_path: Optional path to LoRA weights (.safetensors). Can be a single
            path or list of paths for stacking multiple LoRAs.
        lora_scale: LoRA scale factor(s). None defaults to 1.0 per LoRA.
        text_encoder_device: Device for Gemma3 text encoder.
        transformer_device: Device for DiT transformer.
        vae_device: Device for VAE decoder.
        quantize: Transformer quantization method (used for disk fallback path only).
        skip_cleanup: Skip memory cleanup between stages.
        cached_transformer: Pre-loaded transformer data from ModelManager. Dict with
            "config" (model config), "state_dict" (pinned bf16 tensors), and
            "video_only" (bool). Skips disk I/O when provided.
        cached_vae: Pre-loaded VAE decoder from ModelManager. Shuttled to GPU for
            decoding, then returned to CPU.

    Returns:
        Video tensor [F, H, W, C] in uint8 format

    Memory usage (RTX 4090, 24GB):
        - Text encoder (Gemma3): ~8GB peak
        - Transformer (FP8): ~13GB
        - VAE: ~2GB
        - Total per stage: <24GB
    """
    effective_quantize, effective_precision = _resolve_quantize(quantize)
    model_path = Path(model_path)
    if text_encoder_path is None:
        text_encoder_path = model_path / "text_encoder"

    logger.info(
        f"[LTX2] text_encoder={text_encoder_device}, transformer={transformer_device}, "
        f"vae={vae_device}, quantize={quantize}, variant={gemma_variant}"
    )

    # Stage 1: Text Encoding (skipped if precomputed_embeddings provided)
    gen_start = time.perf_counter()
    stage1_start = time.perf_counter()
    if precomputed_embeddings is not None:
        logger.info("Stage 1: Using precomputed embeddings (skipping text encoder)")
        # Precomputed embeddings are [seq_len, dim], need [1, seq_len, dim]
        if precomputed_embeddings.dim() == 2:
            prompt_embeds = precomputed_embeddings.unsqueeze(0)
        else:
            prompt_embeds = precomputed_embeddings
        # Move to transformer device with requested dtype
        prompt_embeds = prompt_embeds.to(transformer_device, dtype)
        # Create attention mask (all ones for precomputed embeddings)
        attention_mask = torch.ones(
            prompt_embeds.shape[0], prompt_embeds.shape[1],
            device=transformer_device, dtype=torch.long
        )
        logger.debug(f"Precomputed embeddings: {prompt_embeds.shape} on {transformer_device}")

        if callback:
            callback("text_encoder", 1, 1)
    else:
        if callback:
            callback("text_encoder", 0, 1)

        logger.info(f"Stage 1: Loading text encoder on {text_encoder_device}...")
        logger.debug(f"  Gemma variant: {gemma_variant}")

        # Use pre-loaded encoder if provided, otherwise create fresh
        encoder_is_borrowed = text_encoder is not None

        if not encoder_is_borrowed:
            if gemma_variant != "bf16":
                from llm_dit.encoders.gemma3_variants import create_gemma3_encoder
                text_encoder = create_gemma3_encoder(
                    variant=gemma_variant,
                    model_path=str(model_path),
                    text_encoder_path=str(text_encoder_path) if text_encoder_path else None,
                    device=text_encoder_device,
                    dtype=dtype,
                )
            else:
                from llm_dit.encoders.gemma3 import Gemma3Encoder
                text_encoder = Gemma3Encoder(
                    model_id=str(text_encoder_path),
                    device=text_encoder_device,
                    dtype=dtype,
                )
        else:
            # Shuttle borrowed encoder to GPU
            logger.info("Using cached encoder, shuttling to GPU...")
            text_encoder.to(torch.device(text_encoder_device))

        prompt = _maybe_enhance_prompt(text_encoder, prompt, callback, enhance_prompt)

        logger.info("Encoding prompt...")
        encoding_output = text_encoder.encode([prompt])
        prompt_embeds = encoding_output.embeddings[0].unsqueeze(0)  # [1, seq_len, dim]
        attention_mask = encoding_output.attention_masks[0].unsqueeze(0)  # [1, seq_len]
        logger.debug(f"Prompt embeddings: {prompt_embeds.shape}")

        # Move embeddings to transformer device, handle encoder lifecycle
        prompt_embeds = prompt_embeds.to(transformer_device, dtype)
        attention_mask = attention_mask.to(transformer_device)

        if encoder_is_borrowed:
            text_encoder.offload()  # Return to CPU pinned memory
            logger.info("Cached encoder returned to CPU")
        else:
            del text_encoder
            if not skip_cleanup:
                cleanup_memory()
            logger.info("Text encoder unloaded")

        if callback:
            callback("text_encoder", 1, 1)

    stage1_elapsed = time.perf_counter() - stage1_start
    logger.info(f"Stage 1 complete: {stage1_elapsed:.1f}s")

    # Stage 2: Transformer Denoising
    stage2_start = time.perf_counter()
    if callback:
        callback("transformer", 0, config.num_inference_steps)

    logger.info(f"Stage 2: Loading transformer on {transformer_device}...")

    _lora_paths, _lora_scales = _normalize_lora_args(lora_path, lora_scale)

    model = _load_transformer_and_lora(
        cached_transformer=cached_transformer,
        model_path=model_path,
        transformer_file=transformer_file,
        dtype=dtype,
        transformer_device=transformer_device,
        effective_quantize=effective_quantize,
        effective_precision=effective_precision,
        granularity=granularity,
        lora_paths=_lora_paths,
        lora_scales=_lora_scales,
        video_only=True,
    )

    # Only load connectors if embeddings need processing (188160 -> 3840 projection)
    # Our Gemma3Encoder already outputs 3840-dim via internal Embeddings1DConnector
    embed_dim = prompt_embeds.shape[-1]
    connectors = None
    if embed_dim == 188160:
        from llm_dit.models.ltx2.connectors import load_ltx2_connectors

        connectors_path = model_path / "connectors"
        connectors = load_ltx2_connectors(
            connectors_path,
            device="cuda",
            dtype=dtype,
        )
        logger.debug("Loaded connectors for 188160 -> 3840 projection")
    else:
        logger.debug(f"Skipping connectors (embed_dim={embed_dim} already projected)")

    logger.debug(f"Transformer loaded: {model.get_num_params() / 1e9:.2f}B params")

    # Generate latents (no VAE decode yet)
    def progress_callback(step, total, _latents):
        if callback:
            callback("transformer", step, total)

    latents = generate_video(
        model=model,
        prompt_embeds=prompt_embeds,
        config=config,
        vae=None,  # Don't decode yet
        connectors=connectors,
        attention_mask=attention_mask,
        callback=progress_callback,
        use_progress=use_progress,
        debug_latents=debug_latents,
    )

    # Unload transformer and connectors
    del model, connectors, prompt_embeds, attention_mask
    if not skip_cleanup:
        cleanup_memory()
        logger.info("Transformer unloaded")

    stage2_elapsed = time.perf_counter() - stage2_start
    logger.info(f"Stage 2 complete: {stage2_elapsed:.1f}s")

    if callback:
        callback("transformer", config.num_inference_steps, config.num_inference_steps)

    # Stage 3: VAE Decoding
    stage3_start = time.perf_counter()
    if callback:
        callback("vae", 0, 1)

    logger.info(f"Stage 3: Loading VAE decoder on {vae_device}...")

    vae_is_borrowed = cached_vae is not None
    if vae_is_borrowed:
        logger.info("Using cached VAE, shuttling to GPU...")
        vae = cached_vae.to(vae_device)
    else:
        from llm_dit.models.ltx2.vae import load_ltx2_vae_decoder
        vae = load_ltx2_vae_decoder(
            _resolve_v23_component_path(model_path, "vae"), dtype=dtype, device="cpu",
        ).to(vae_device)

    logger.info("Decoding latents to video...")

    # Decode latents to video
    # Note: our VideoDecoder handles denormalization internally via per_channel_statistics
    decode_start = time.perf_counter()
    with torch.no_grad():
        video = vae(latents)
    decode_elapsed = time.perf_counter() - decode_start
    if decode_elapsed > 5.0:
        logger.info(f"[Decode] VAE decode {decode_elapsed:.1f}s (torch.compile warmup -- subsequent decodes will be fast)")
    else:
        logger.info(f"[Decode] VAE decode {decode_elapsed:.2f}s")

    # Convert to [F, H, W, C] uint8
    video = video.squeeze(0).permute(1, 2, 3, 0)
    video = ((video + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)

    # Return or unload VAE
    if vae_is_borrowed:
        vae.to("cpu")  # Return to CPU pinned memory
        logger.info("Cached VAE returned to CPU")
    else:
        del vae
    del latents
    if not skip_cleanup:
        cleanup_memory()
    if not vae_is_borrowed:
        logger.info("VAE unloaded")

    stage3_elapsed = time.perf_counter() - stage3_start
    logger.info(f"Stage 3 complete: {stage3_elapsed:.1f}s")

    if callback:
        callback("vae", 1, 1)

    gen_elapsed = time.perf_counter() - gen_start
    logger.info(
        f"Generation complete: {gen_elapsed:.1f}s total "
        f"(encode={stage1_elapsed:.1f}s, denoise={stage2_elapsed:.1f}s, decode={stage3_elapsed:.1f}s)"
    )
    return video


# =============================================================================
# Two-Stage Generation (reference: TI2VidTwoStagesPipeline)
# =============================================================================


@dataclass
class StepContext:
    """Per-step parameters for the denoising loop.

    All denoising parameters that can vary per step live here.
    When guidance_scale <= 1.0, the loop runs a single forward pass.
    When > 1.0 and neg_embeds is provided, it runs CFG (double forward pass).
    When stg_scale > 0 and stg_blocks is set, adds a 3rd perturbed pass (STG).
    """
    guidance_scale: float = 1.0
    audio_guidance_scale: float = 0.0  # 0 = use guidance_scale for audio too
    neg_embeds: Optional[torch.Tensor] = None
    rescale_scale: float = 0.0
    ge_gamma: float = 0.0
    stg_scale: float = 0.0
    stg_blocks: Optional[List[int]] = None
    modality_scale: float = 1.0  # Cross-modal attention guidance (1.0=disabled, 3.0=reference)


# A callable that returns StepContext for a given (step_index, sigma_value)
StepSchedule = Callable[[int, float], StepContext]


def constant_schedule(
    guidance_scale: float = 1.0,
    audio_guidance_scale: float = 0.0,
    neg_embeds: Optional[torch.Tensor] = None,
    rescale_scale: float = 0.0,
    ge_gamma: float = 0.0,
    stg_scale: float = 0.0,
    stg_blocks: Optional[List[int]] = None,
    modality_scale: float = 1.0,
) -> StepSchedule:
    """Static parameters for all steps (default behavior)."""
    ctx = StepContext(
        guidance_scale=guidance_scale,
        audio_guidance_scale=audio_guidance_scale,
        neg_embeds=neg_embeds,
        rescale_scale=rescale_scale,
        ge_gamma=ge_gamma,
        stg_scale=stg_scale,
        stg_blocks=stg_blocks,
        modality_scale=modality_scale,
    )
    return lambda step, sigma: ctx


@dataclass
class TwoStageConfig:
    """Configuration for two-stage video generation.

    Stage 1: Denoise at half resolution (height/2, width/2) with full CFG.
    Stage 1.5: Spatial upsample latents 2x via learned upsampler.
    Stage 2: Refine at full resolution with distilled LoRA, no CFG, 3 steps.

    The base GenerationConfig specifies the FULL output resolution.
    Stage 1 resolution is computed automatically as (height/2, width/2).
    """

    # Stage 1 (low-res denoising)
    stage1_steps: int = 30
    guidance_scale: float = 3.0

    # Guidance options (stage 1 only)
    stg_scale: float = 1.0  # Spatio-temporal guidance scale (0=disabled, 1.0=reference)
    stg_blocks: list[int] = None  # type: ignore[assignment]
    rescale_scale: float = 0.7  # CFG rescaling
    modality_scale: float = 3.0  # Cross-modal attention guidance (1.0=disabled, 3.0=reference)

    negative_prompt: str = LTX2_DEFAULT_NEGATIVE_PROMPT

    # Gradient estimation
    ge_gamma: float = 0.0  # 0=disabled, 2.0=reference default

    # Stage 2 (high-res refinement)
    distilled_lora_path: str = ""
    distilled_lora_scale: float = 1.0

    # Spatial upsampler
    spatial_upsampler_file: str = "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"

    # FBCache
    fbcache_threshold: float = 0.0  # Block-skip threshold (0=disabled, 0.05=recommended)

    # Pipeline mode: "standard" (base+LoRA, 30 steps, full guidance) or
    # "distilled" (pre-distilled checkpoint, 8 steps, no guidance)
    pipeline_mode: str = "standard"

    def __post_init__(self):
        if self.stg_blocks is None:
            self.stg_blocks = [28]


def _compute_velocity(
    model: LTX2Transformer,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    positions: torch.Tensor,
    prompt_embeds: torch.Tensor,
    ctx: StepContext,
    fbcache_threshold: float = 0.0,
    step_index: int = 0,
    num_steps: int = 1,
) -> torch.Tensor:
    """Compute velocity prediction with optional CFG and STG.

    Pass structure depends on guidance parameters:
    - guidance_scale <= 1.0: single forward pass (no guidance)
    - guidance_scale > 1.0 with neg_embeds: 2-pass CFG
    - stg_scale > 0 with stg_blocks: adds 3rd perturbed pass (STG)

    The combined formula (reference MultiModalGuider):
      v = v_cond + (cfg-1)*(v_cond - v_uncond) + stg*(v_cond - v_perturbed)

    The perturbed pass uses the positive prompt but skips self-attention at
    specified blocks, producing a spatially/temporally degraded prediction.
    The delta between conditioned and perturbed predictions drives a guidance
    term that improves spatial coherence and temporal consistency.

    Args:
        model: LTX2Transformer on GPU.
        latents: [B, T, D] noisy latent tokens.
        timestep: [B, T] per-token timestep values.
        positions: [B, 3, T, 2] RoPE position indices.
        prompt_embeds: [B, seq_len, dim] positive text embeddings.
        ctx: Per-step denoising parameters.
        fbcache_threshold: FBCache block-skip threshold (0=disabled).
        step_index: Current denoising step (0-based).
        num_steps: Total denoising steps.

    Returns:
        Velocity prediction tensor [B, T, D].
    """
    # FBCache kwargs -- passed to all model() calls within this step
    fb_kwargs = {}
    if fbcache_threshold > 0.0:
        fb_kwargs = dict(
            fbcache_threshold=fbcache_threshold,
            step_index=step_index,
            num_steps=num_steps,
        )

    if ctx.guidance_scale > 1.0 and ctx.neg_embeds is not None:
        # Pass 1: Unconditional (negative prompt)
        uncond_modality = create_video_modality(latents, timestep, positions, ctx.neg_embeds)
        velocity_uncond, _ = model(video=uncond_modality, **fb_kwargs)
        del uncond_modality
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Pass 2: Conditional (positive prompt)
        cond_modality = create_video_modality(latents, timestep, positions, prompt_embeds)
        velocity_cond, _ = model(video=cond_modality, **fb_kwargs)

        # CFG blend
        velocity = velocity_cond + (ctx.guidance_scale - 1.0) * (velocity_cond - velocity_uncond)
        del velocity_uncond

        # Pass 3: Perturbed (STG) -- self-attention skipped at stg_blocks
        if ctx.stg_scale > 0 and ctx.stg_blocks:
            del cond_modality
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            stg_modality = create_video_modality(latents, timestep, positions, prompt_embeds)
            stg_blocks_set = set(ctx.stg_blocks)
            velocity_perturbed, _ = model(video=stg_modality, stg_blocks=stg_blocks_set, **fb_kwargs)
            del stg_modality

            velocity = velocity + ctx.stg_scale * (velocity_cond - velocity_perturbed)
            del velocity_perturbed
        else:
            del cond_modality

        # CFG rescaling (CFG* variant) to prevent over-saturation
        if ctx.rescale_scale > 0:
            factor = velocity_cond.std() / velocity.std()
            factor = ctx.rescale_scale * factor + (1.0 - ctx.rescale_scale)
            velocity = velocity * factor

        del velocity_cond
    else:
        # Simple denoising (no guidance)
        modality = create_video_modality(latents, timestep, positions, prompt_embeds)
        velocity, _ = model(video=modality, **fb_kwargs)
        del modality

    return velocity


def _compute_av_velocity(
    model: LTX2Transformer,
    video_latents: torch.Tensor,
    video_timestep: torch.Tensor,
    video_positions: torch.Tensor,
    video_prompt_embeds: torch.Tensor,
    audio_latents: torch.Tensor,
    audio_timestep: torch.Tensor,
    audio_positions: torch.Tensor,
    audio_prompt_embeds: torch.Tensor,
    ctx: StepContext,
    audio_neg_embeds: Optional[torch.Tensor] = None,
    fbcache_threshold: float = 0.0,
    step_index: int = 0,
    num_steps: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute velocity prediction for both video and audio streams.

    Mirrors _compute_velocity() but creates both modalities for each pass.
    Full guidance formula (up to 4 forward passes):
      v = cond + (cfg-1)*(cond - uncond)
          + stg*(cond - perturbed)
          + (modality_scale-1)*(cond - isolated)

    Pass 1: Unconditional (negative prompts, zeros for missing audio neg embeds)
    Pass 2: Conditional (positive prompts)
    Pass 3: STG perturbed (skip video + audio self-attention)
    Pass 4: Modality-isolated (skip cross-modal attention, when modality_scale > 1.0)

    Audio uses ctx.audio_guidance_scale when > 0, else falls back to ctx.guidance_scale.

    Args:
        model: LTX2Transformer (AudioVideo mode) on GPU.
        video_latents: [B, T_v, D_v] noisy video latent tokens.
        video_timestep: [B, T_v] per-token video timestep values.
        video_positions: [B, 3, T_v, 2] video RoPE position indices.
        video_prompt_embeds: [B, seq_len, dim] positive video text embeddings.
        audio_latents: [B, T_a, D_a] noisy audio latent tokens.
        audio_timestep: [B, T_a] per-token audio timestep values.
        audio_positions: [B, 1, T_a, 2] audio RoPE position indices.
        audio_prompt_embeds: [B, seq_len, dim] positive audio text embeddings.
        ctx: Per-step denoising parameters.
        audio_neg_embeds: Optional audio negative embeddings for CFG.
        fbcache_threshold: FBCache block-skip threshold (0=disabled).
        step_index: Current denoising step (0-based).
        num_steps: Total denoising steps.

    Returns:
        Tuple of (video_velocity, audio_velocity) tensors.
    """
    from llm_dit.models.ltx2.transformer import (
        BatchedPerturbationConfig,
        Perturbation,
        PerturbationConfig,
        PerturbationType,
    )

    fb_kwargs = {}
    if fbcache_threshold > 0.0:
        fb_kwargs = dict(
            fbcache_threshold=fbcache_threshold,
            step_index=step_index,
            num_steps=num_steps,
        )

    if ctx.guidance_scale > 1.0 and ctx.neg_embeds is not None:
        # Pass 1: Unconditional (negative prompts for both modalities)
        if step_index == 0:
            has_neg = audio_neg_embeds is not None
            logger.debug(
                f"AV CFG: guidance={ctx.guidance_scale}, "
                f"audio_neg_embeds={'provided' if has_neg else 'ZEROS_FALLBACK'}"
            )
        uncond_video = create_video_modality(
            video_latents, video_timestep, video_positions, ctx.neg_embeds,
        )
        uncond_audio = create_audio_modality(
            audio_latents, audio_timestep, audio_positions,
            audio_neg_embeds if audio_neg_embeds is not None else torch.zeros_like(audio_prompt_embeds),
        )
        v_uncond, a_uncond = model(video=uncond_video, audio=uncond_audio, **fb_kwargs)
        del uncond_video, uncond_audio
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Pass 2: Conditional (positive prompts)
        cond_video = create_video_modality(
            video_latents, video_timestep, video_positions, video_prompt_embeds,
        )
        cond_audio = create_audio_modality(
            audio_latents, audio_timestep, audio_positions, audio_prompt_embeds,
        )
        v_cond, a_cond = model(video=cond_video, audio=cond_audio, **fb_kwargs)
        del cond_video, cond_audio

        # CFG blend -- separate guidance scale for audio when specified
        video_vel = v_cond + (ctx.guidance_scale - 1.0) * (v_cond - v_uncond)
        audio_cfg = ctx.audio_guidance_scale if ctx.audio_guidance_scale > 0 else ctx.guidance_scale
        audio_vel = a_cond + (audio_cfg - 1.0) * (a_cond - a_uncond)
        del v_uncond, a_uncond

        # Pass 3: Perturbed (STG) -- skip video AND audio self-attention
        if ctx.stg_scale > 0 and ctx.stg_blocks:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            stg_video = create_video_modality(
                video_latents, video_timestep, video_positions, video_prompt_embeds,
            )
            stg_audio = create_audio_modality(
                audio_latents, audio_timestep, audio_positions, audio_prompt_embeds,
            )
            perturbations = [
                Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=list(ctx.stg_blocks)),
                Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=list(ctx.stg_blocks)),
            ]
            perturb_config = BatchedPerturbationConfig(
                perturbations=[PerturbationConfig(perturbations=perturbations)],
            )
            v_perturbed, a_perturbed = model(
                video=stg_video, audio=stg_audio,
                perturbation_config=perturb_config,
                **fb_kwargs,
            )
            del stg_video, stg_audio

            video_vel = video_vel + ctx.stg_scale * (v_cond - v_perturbed)
            audio_vel = audio_vel + ctx.stg_scale * (a_cond - a_perturbed)
            del v_perturbed, a_perturbed

        # Pass 4: Modality-isolated (skip cross-modal attention)
        if ctx.modality_scale > 1.0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            iso_video = create_video_modality(
                video_latents, video_timestep, video_positions, video_prompt_embeds,
            )
            iso_audio = create_audio_modality(
                audio_latents, audio_timestep, audio_positions, audio_prompt_embeds,
            )
            mod_perturbations = [
                Perturbation(type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None),
                Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None),
            ]
            mod_perturb_config = BatchedPerturbationConfig(
                perturbations=[PerturbationConfig(perturbations=mod_perturbations)],
            )
            v_isolated, a_isolated = model(
                video=iso_video, audio=iso_audio,
                perturbation_config=mod_perturb_config,
                **fb_kwargs,
            )
            del iso_video, iso_audio

            video_vel = video_vel + (ctx.modality_scale - 1.0) * (v_cond - v_isolated)
            audio_vel = audio_vel + (ctx.modality_scale - 1.0) * (a_cond - a_isolated)
            del v_isolated, a_isolated

        # CFG rescaling
        if ctx.rescale_scale > 0:
            v_factor = v_cond.std() / video_vel.std()
            v_factor = ctx.rescale_scale * v_factor + (1.0 - ctx.rescale_scale)
            video_vel = video_vel * v_factor

            a_factor = a_cond.std() / audio_vel.std()
            a_factor = ctx.rescale_scale * a_factor + (1.0 - ctx.rescale_scale)
            audio_vel = audio_vel * a_factor

        del v_cond, a_cond
    else:
        # Simple denoising (no guidance)
        video_mod = create_video_modality(
            video_latents, video_timestep, video_positions, video_prompt_embeds,
        )
        audio_mod = create_audio_modality(
            audio_latents, audio_timestep, audio_positions, audio_prompt_embeds,
        )
        video_vel, audio_vel = model(video=video_mod, audio=audio_mod, **fb_kwargs)
        del video_mod, audio_mod

    return video_vel, audio_vel


def _denoise_stage(
    model: LTX2Transformer,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    sigmas: torch.Tensor,
    positions: torch.Tensor,
    stage_name: str,
    step_schedule: Optional[StepSchedule] = None,
    callback: Optional[Callable[[str, int, int], None]] = None,
    denoise_mask: Optional[torch.Tensor] = None,
    clean_latent: Optional[torch.Tensor] = None,
    fbcache_threshold: float = 0.0,
) -> torch.Tensor:
    """Run a single denoising stage (Euler method with per-step parameters).

    The step_schedule callable controls all per-step parameters (guidance,
    rescaling, GE gamma). Pass constant_schedule() for static parameters,
    or a custom callable for dynamic schedules.

    Args:
        model: LTX2Transformer on GPU.
        latents: [B, T, D] noisy latent tokens.
        prompt_embeds: [B, seq_len, dim] positive text embeddings.
        sigmas: [N+1] sigma schedule.
        positions: [B, 3, T, 2] RoPE position indices.
        stage_name: Name for logging (e.g., "stage1_denoise").
        step_schedule: Callable(step_index, sigma) -> StepContext. Defaults
            to constant_schedule() (guidance_scale=1.0, no CFG).
        callback: Optional progress callback(stage_name, step, total).
        denoise_mask: Optional per-token denoise mask for conditioning.
        clean_latent: Optional clean latent for conditioned regions.
        fbcache_threshold: FBCache block-skip threshold (0=disabled, 0.05=recommended).

    Returns:
        Denoised latent tensor [B, T, D].
    """
    if step_schedule is None:
        step_schedule = constant_schedule()

    dtype = latents.dtype
    num_tokens = latents.shape[1]
    num_steps = len(sigmas) - 1

    model.train(False)

    # Reset FBCache state for this stage
    if fbcache_threshold > 0.0:
        model.reset_fbcache()
        logger.info(f"[{stage_name}] FBCache enabled (threshold={fbcache_threshold})")

    prev_velocity: Optional[torch.Tensor] = None

    with torch.no_grad():
        denoise_start = time.perf_counter()
        step_times: list[float] = []

        for i in range(num_steps):
            step_start = time.perf_counter()
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            ctx = step_schedule(i, sigma.item())

            # Per-token or uniform timesteps
            if denoise_mask is not None:
                timestep = timesteps_from_mask(denoise_mask, sigma).squeeze(-1)
            else:
                timestep = sigma.expand(1, num_tokens)

            # -- DIAGNOSTIC: Checkpoint 5 - Per-step stats --
            _log_step = (i <= 2) or (i == num_steps - 1)
            if _log_step:
                _diag.info(
                    f"[DIAG] [{stage_name}:step {i}] PRE: latent_std={latents.float().std():.6f}, "
                    f"sigma={sigma:.6f}, sigma_next={sigma_next:.6f}, dt={sigma_next - sigma:.6f}"
                )

            velocity = _compute_velocity(
                model, latents, timestep, positions, prompt_embeds, ctx,
                fbcache_threshold=fbcache_threshold,
                step_index=i,
                num_steps=num_steps,
            )

            if _log_step:
                vf = velocity.float()
                _diag.info(
                    f"[DIAG] [{stage_name}:step {i}] velocity: mean={vf.mean():.6f}, "
                    f"std={vf.std():.6f}, min={vf.min():.6f}, max={vf.max():.6f}, "
                    f"has_nan={torch.isnan(vf).any().item()}, has_inf={torch.isinf(vf).any().item()}"
                )

            # Gradient estimation: save raw velocity before correction,
            # then apply GE. Reference stores pre-GE velocity for next delta.
            if ctx.ge_gamma > 0:
                raw_velocity = velocity.clone()
                if prev_velocity is not None:
                    delta_v = velocity - prev_velocity
                    velocity = ctx.ge_gamma * delta_v + prev_velocity
                prev_velocity = raw_velocity

            # Euler step: x_{t-1} = x_t + v * dt
            dt = sigma_next - sigma
            denoised = (latents.float() + velocity.float() * dt).to(dtype)

            # Post-process conditioned regions
            if denoise_mask is not None and clean_latent is not None:
                latents = post_process_latent(denoised, denoise_mask, clean_latent)
            else:
                latents = denoised

            if _log_step:
                _diag.info(
                    f"[DIAG] [{stage_name}:step {i}] POST: latent_std={latents.float().std():.6f}"
                )

            del velocity
            if (i + 1) % 5 == 0:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            step_elapsed = time.perf_counter() - step_start
            step_times.append(step_elapsed)
            if i == 0 or (i + 1) % 10 == 0 or i == num_steps - 1:
                logger.info(f"[{stage_name}:Step {i}] {step_elapsed:.2f}s")

            if callback:
                callback(stage_name, i + 1, num_steps)

        denoise_elapsed = time.perf_counter() - denoise_start
        logger.info(f"[{stage_name}] {num_steps} steps in {denoise_elapsed:.1f}s")

    return latents


def _denoise_av_stage(
    model: LTX2Transformer,
    video_latents: torch.Tensor,
    audio_latents: torch.Tensor,
    video_prompt_embeds: torch.Tensor,
    audio_prompt_embeds: torch.Tensor,
    sigmas: torch.Tensor,
    video_positions: torch.Tensor,
    audio_positions: torch.Tensor,
    stage_name: str,
    step_schedule: Optional[StepSchedule] = None,
    audio_neg_embeds: Optional[torch.Tensor] = None,
    callback: Optional[Callable[[str, int, int], None]] = None,
    denoise_mask: Optional[torch.Tensor] = None,
    clean_latent: Optional[torch.Tensor] = None,
    fbcache_threshold: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run denoising stage for both video and audio streams.

    Same Euler method and sigma schedule as _denoise_stage. Audio uses
    uniform timesteps (no denoise_mask support for audio -- audio has
    no spatial conditioning).

    Args:
        model: LTX2Transformer (AudioVideo mode) on GPU.
        video_latents: [B, T_v, D_v] noisy video latent tokens.
        audio_latents: [B, T_a, D_a] noisy audio latent tokens.
        video_prompt_embeds: [B, seq_len, dim] video text embeddings.
        audio_prompt_embeds: [B, seq_len, dim] audio text embeddings.
        sigmas: [N+1] sigma schedule (shared for both modalities).
        video_positions: [B, 3, T_v, 2] video RoPE position indices.
        audio_positions: [B, 1, T_a, 2] audio RoPE position indices.
        stage_name: Name for logging.
        step_schedule: Callable(step_index, sigma) -> StepContext.
        audio_neg_embeds: Optional audio negative embeddings for CFG.
        callback: Optional progress callback.
        denoise_mask: Optional per-token denoise mask (video only).
        clean_latent: Optional clean latent for conditioned regions (video only).
        fbcache_threshold: FBCache block-skip threshold (0=disabled).

    Returns:
        Tuple of (video_latents, audio_latents) after denoising.
    """
    if step_schedule is None:
        step_schedule = constant_schedule()

    dtype = video_latents.dtype
    video_num_tokens = video_latents.shape[1]
    audio_num_tokens = audio_latents.shape[1]
    num_steps = len(sigmas) - 1

    model.train(False)

    if fbcache_threshold > 0.0:
        model.reset_fbcache()
        logger.info(f"[{stage_name}] FBCache enabled (threshold={fbcache_threshold})")

    prev_video_vel: Optional[torch.Tensor] = None
    prev_audio_vel: Optional[torch.Tensor] = None

    with torch.no_grad():
        denoise_start = time.perf_counter()
        step_times: list[float] = []

        for i in range(num_steps):
            step_start = time.perf_counter()
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            ctx = step_schedule(i, sigma.item())

            # Video timesteps: per-token (with mask) or uniform
            if denoise_mask is not None:
                video_timestep = timesteps_from_mask(denoise_mask, sigma).squeeze(-1)
            else:
                video_timestep = sigma.expand(1, video_num_tokens)

            # Audio timesteps: always uniform (no spatial conditioning)
            audio_timestep = sigma.expand(1, audio_num_tokens)

            video_vel, audio_vel = _compute_av_velocity(
                model,
                video_latents, video_timestep, video_positions, video_prompt_embeds,
                audio_latents, audio_timestep, audio_positions, audio_prompt_embeds,
                ctx,
                audio_neg_embeds=audio_neg_embeds,
                fbcache_threshold=fbcache_threshold,
                step_index=i,
                num_steps=num_steps,
            )

            # Gradient estimation: save raw velocities before correction,
            # then apply GE. Reference stores pre-GE velocity for next delta.
            if ctx.ge_gamma > 0:
                raw_video_vel = video_vel.clone()
                raw_audio_vel = audio_vel.clone()
                if prev_video_vel is not None:
                    video_vel = ctx.ge_gamma * (video_vel - prev_video_vel) + prev_video_vel
                if prev_audio_vel is not None:
                    audio_vel = ctx.ge_gamma * (audio_vel - prev_audio_vel) + prev_audio_vel
                prev_video_vel = raw_video_vel
                prev_audio_vel = raw_audio_vel

            # Euler step: x_{t-1} = x_t + v * dt
            dt = sigma_next - sigma
            video_denoised = (video_latents.float() + video_vel.float() * dt).to(dtype)
            audio_denoised = (audio_latents.float() + audio_vel.float() * dt).to(dtype)

            # Post-process conditioned regions (video only)
            if denoise_mask is not None and clean_latent is not None:
                video_latents = post_process_latent(video_denoised, denoise_mask, clean_latent)
            else:
                video_latents = video_denoised
            audio_latents = audio_denoised

            del video_vel, audio_vel
            if (i + 1) % 5 == 0:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            step_elapsed = time.perf_counter() - step_start
            step_times.append(step_elapsed)
            if i == 0 or i == num_steps - 1 or (i + 1) % 5 == 0:
                logger.debug(
                    f"[{stage_name}:Step {i}] video_std={video_latents.std():.4f}, "
                    f"audio_std={audio_latents.std():.4f}, "
                    f"sigma={sigma:.4f}->{sigma_next:.4f}"
                )
            if i == 0 or (i + 1) % 10 == 0 or i == num_steps - 1:
                logger.info(f"[{stage_name}:Step {i}] {step_elapsed:.2f}s")

            if callback:
                callback(stage_name, i + 1, num_steps)

        denoise_elapsed = time.perf_counter() - denoise_start
        logger.info(f"[{stage_name}] {num_steps} steps in {denoise_elapsed:.1f}s")

    return video_latents, audio_latents


def generate_video_two_stage(
    prompt: str,
    config: GenerationConfig,
    two_stage: TwoStageConfig,
    model_path: Union[str, Path] = "models/LTX-2",
    text_encoder_path: Optional[Union[str, Path]] = None,
    dtype: torch.dtype = torch.bfloat16,
    callback: Optional[Callable[[str, int, int], None]] = None,
    gemma_variant: str = "bf16",
    lora_path: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
    lora_scale: Optional[Union[float, List[float]]] = None,
    text_encoder_device: str = "cpu",
    transformer_device: str = "cuda",
    vae_device: str = "cuda",
    quantize: str = "fp8",
    granularity: str = "per-row",
    transformer_file: str = "",
    skip_cleanup: bool = False,
    enhance_prompt: bool = False,
    text_encoder: Optional[Any] = None,
    cached_transformer: Optional[dict] = None,
    cached_vae: Optional[Any] = None,
    # Audio generation params (Phase 3)
    video_only: bool = True,
    audio_negative_prompt: str = "",
    audio_guidance_scale: float = 7.0,
    cached_audio_decoder: Optional[Any] = None,
    cached_vocoder: Optional[Any] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, int]]:
    """Generate video using two-stage pipeline with spatial upsampling.

    Reference: TI2VidTwoStagesPipeline from official LTX-2 repo.

    Flow:
      Stage 0: Encode text (positive + negative prompts)
      Stage 1: Denoise at half resolution with CFG guidance
      Stage 1.5: Spatial upsample latents 2x
      Stage 2: Refine at full resolution with distilled LoRA (no CFG, 3 steps)
      Stage 3: VAE decode to pixels
      Stage 4 (audio): Audio VAE decode + vocoder (when video_only=False)

    Only one major component is on GPU at a time (sequential offloading).

    Args:
        prompt: Text prompt for generation.
        config: GenerationConfig with FULL output resolution (height, width).
            Stage 1 runs at (height/2, width/2) automatically.
        two_stage: TwoStageConfig with stage-specific parameters.
        model_path: Path to LTX-2 model directory.
        text_encoder_path: Optional separate path for text encoder.
        dtype: Base computation dtype.
        callback: Optional progress callback(stage_name, step, total).
        gemma_variant: Gemma3 variant (bf16, 8bit, q4-qat).
        lora_path: Optional base LoRA(s) for stage 1.
        lora_scale: LoRA scale(s) for base LoRA.
        text_encoder_device: Device for Gemma3 text encoder.
        transformer_device: Device for DiT transformer.
        vae_device: Device for VAE decoder.
        quantize: Transformer quantization method: none, fp8, fp8-weight-only,
            fp8-dynamic, int8, int4.
        skip_cleanup: Skip memory cleanup between stages.
        cached_transformer: Pre-loaded transformer data from ModelManager. Dict with
            "config" (model config), "state_dict" (pinned bf16 tensors), and
            "video_only" (bool).
        cached_vae: Pre-loaded VAE decoder from ModelManager. Used for
            per_channel_statistics in Stage 1.5 and decoding in Stage 3.
        video_only: When True (default), generate video only. When False,
            generate both video and audio streams.
        audio_negative_prompt: Negative prompt for audio CFG guidance.
        cached_audio_decoder: Pre-loaded audio decoder from ModelManager.
        cached_vocoder: Pre-loaded vocoder from ModelManager.

    Returns:
        When video_only=True: Video tensor [F, H, W, C] in uint8 format.
        When video_only=False: Tuple of (video_tensor, audio_waveform).
    """
    if two_stage.distilled_lora_scale > 0 and not two_stage.distilled_lora_path:
        raise ValueError(
            "Two-stage generation requires a distilled LoRA for stage 2 refinement. "
            "Set distilled_lora_path in TwoStageConfig (e.g., 'ltx-2-19b-distilled-lora-384.safetensors'). "
            "Without it, the base model cannot denoise in 3 steps and will produce garbage output. "
            "Set distilled_lora_scale=0 to skip the distilled LoRA entirely."
        )

    _validate_two_stage_dimensions(config.height, config.width)

    effective_quantize, effective_precision = _resolve_quantize(quantize)
    model_path = Path(model_path)
    if text_encoder_path is None:
        text_encoder_path = model_path / "text_encoder"

    logger.info(
        f"[LTX2:TwoStage] text_encoder={text_encoder_device}, transformer={transformer_device}, "
        f"vae={vae_device}, quantize={quantize}, variant={gemma_variant}"
    )

    gen_start = time.perf_counter()

    # =========================================================================
    # Stage 0: Text Encoding (positive + negative prompts)
    # =========================================================================
    stage0_start = time.perf_counter()
    if callback:
        callback("encoding", 0, 2)

    logger.info(f"Stage 0: Loading text encoder on {text_encoder_device}...")

    # Use pre-loaded encoder if provided, otherwise create fresh
    encoder_is_borrowed = text_encoder is not None

    if not encoder_is_borrowed:
        if gemma_variant != "bf16":
            from llm_dit.encoders.gemma3_variants import create_gemma3_encoder
            text_encoder = create_gemma3_encoder(
                variant=gemma_variant,
                model_path=str(model_path),
                text_encoder_path=str(text_encoder_path),
                device=text_encoder_device,
                dtype=dtype,
            )
        else:
            from llm_dit.encoders.gemma3 import Gemma3Encoder
            text_encoder = Gemma3Encoder(
                model_id=str(text_encoder_path),
                device=text_encoder_device,
                dtype=dtype,
            )
    else:
        # Shuttle borrowed encoder to GPU
        logger.info("Using cached encoder, shuttling to GPU...")
        text_encoder.to(torch.device(text_encoder_device))

    prompt = _maybe_enhance_prompt(text_encoder, prompt, callback, enhance_prompt)

    # Encode positive prompt
    logger.info("Encoding positive prompt...")
    pos_output = text_encoder.encode([prompt])
    pos_embeds = pos_output.embeddings[0].unsqueeze(0)
    pos_mask = pos_output.attention_masks[0].unsqueeze(0)

    # -- DIAGNOSTIC: Checkpoint 1 - Text encoding output --
    _diag.info(_tensor_stats("pos_embeds", pos_embeds))
    _diag.info(f"[DIAG] pos_mask: shape={list(pos_mask.shape)}, sum={pos_mask.sum().item():.0f}")

    # Extract audio embeddings (2048-dim) from encoder output
    pos_audio_embeds: Optional[torch.Tensor] = None
    if not video_only and pos_output.audio_embeddings is not None:
        pos_audio_embeds = pos_output.audio_embeddings[0].unsqueeze(0)
        logger.info(f"Audio embeddings: {pos_audio_embeds.shape}")
        _diag.info(_tensor_stats("pos_audio_embeds", pos_audio_embeds))
    elif not video_only:
        logger.warning("Audio mode but encoder returned no audio embeddings")

    if callback:
        callback("encoding", 1, 2)

    # Encode negative prompt
    logger.info("Encoding negative prompt...")
    neg_output = text_encoder.encode([two_stage.negative_prompt])
    neg_embeds = neg_output.embeddings[0].unsqueeze(0)

    # Encode audio negative prompt if audio is enabled
    audio_neg_embeds = None
    if not video_only:
        if audio_negative_prompt:
            logger.info("Encoding audio negative prompt...")
            audio_neg_output = text_encoder.encode([audio_negative_prompt])
            if audio_neg_output.audio_embeddings is not None:
                audio_neg_embeds = audio_neg_output.audio_embeddings[0].unsqueeze(0)
            else:
                logger.warning("Audio negative prompt produced no audio embeddings, using zeros")
        elif neg_output.audio_embeddings is not None:
            # Fall back to negative prompt's audio embeddings
            audio_neg_embeds = neg_output.audio_embeddings[0].unsqueeze(0)

        # Final fallback: zeros matching positive audio shape (ensures CFG gradient is nonzero)
        if audio_neg_embeds is None and pos_audio_embeds is not None:
            audio_neg_embeds = torch.zeros_like(pos_audio_embeds)
            logger.warning("No audio negative embeddings from encoder, using zeros")
        elif audio_neg_embeds is not None:
            logger.info(f"Audio neg embeddings: {audio_neg_embeds.shape}")

    if callback:
        callback("encoding", 2, 2)

    # Move to transformer device, handle encoder lifecycle
    pos_embeds = pos_embeds.to(transformer_device, dtype)
    pos_mask = pos_mask.to(transformer_device)
    neg_embeds = neg_embeds.to(transformer_device, dtype)
    if pos_audio_embeds is not None:
        pos_audio_embeds = pos_audio_embeds.to(transformer_device, dtype)
    if audio_neg_embeds is not None:
        audio_neg_embeds = audio_neg_embeds.to(transformer_device, dtype)

    if encoder_is_borrowed:
        text_encoder.offload()  # Return to CPU pinned memory
        logger.info("Cached encoder returned to CPU")
    else:
        del text_encoder
        if not skip_cleanup:
            cleanup_memory("post_encoder_unload")
        logger.info("Text encoder unloaded")

    stage0_elapsed = time.perf_counter() - stage0_start
    logger.info(f"Stage 0 complete: {stage0_elapsed:.1f}s")

    # =========================================================================
    # Stage 1: Low-Resolution Denoising (height/2, width/2)
    # =========================================================================
    stage1_start = time.perf_counter()
    if callback:
        callback("stage1_denoise", 0, two_stage.stage1_steps)

    logger.info(f"Stage 1: Loading transformer on {transformer_device} for low-res denoising...")

    # Stage 1 config: half resolution
    stage1_config = GenerationConfig(
        num_frames=config.num_frames,
        height=config.height // 2,
        width=config.width // 2,
        num_inference_steps=two_stage.stage1_steps,
        guidance_scale=two_stage.guidance_scale,
        seed=config.seed,
        base_shift=config.base_shift,
        max_shift=config.max_shift,
        stretch=config.stretch,
        terminal=config.terminal,
    )

    _lora_paths, _lora_scales = _normalize_lora_args(lora_path, lora_scale)

    model = _load_transformer_and_lora(
        cached_transformer=cached_transformer,
        model_path=model_path,
        transformer_file=transformer_file,
        dtype=dtype,
        transformer_device=transformer_device,
        effective_quantize=effective_quantize,
        effective_precision=effective_precision,
        granularity=granularity,
        lora_paths=_lora_paths,
        lora_scales=_lora_scales,
        video_only=video_only,
    )

    # Guard: if audio requested but model doesn't support it, fall back to video-only
    if not video_only and not model.model_type.is_audio_enabled():
        logger.warning("Audio requested but transformer is video-only. Falling back to video-only.")
        video_only = True

    # Initialize latent noise at half resolution
    fps = config.fps
    t_latent, h_latent, w_latent = stage1_config.latent_dims
    num_tokens = stage1_config.num_tokens

    generator = None
    if config.seed is not None:
        generator = torch.Generator(device=transformer_device).manual_seed(config.seed)

    latents = torch.randn(
        (1, num_tokens, 128),
        generator=generator,
        device=transformer_device,
        dtype=dtype,
    )

    # -- DIAGNOSTIC: Checkpoint 2 - Noise initialization --
    _diag.info(_tensor_stats("stage1_noise", latents))
    _diag.info(
        f"[DIAG] stage1_latent_dims: t={t_latent}, h={h_latent}, w={w_latent}, "
        f"tokens={num_tokens}, half_res={stage1_config.height}x{stage1_config.width}"
    )

    positions = create_position_indices(
        batch_size=1,
        num_frames=config.num_frames,
        height=stage1_config.height,
        width=stage1_config.width,
        device=torch.device(transformer_device),
        fps=fps,
        scale_factors=(8, 32, 32),
        causal_fix=True,
    )

    # -- DIAGNOSTIC: Checkpoint 3 - Position indices --
    _diag.info(_tensor_stats("video_positions", positions))
    _diag.info(
        f"[DIAG] positions_detail: temporal=[{positions[0,0,:,0].min():.2f},{positions[0,0,:,1].max():.2f}], "
        f"height=[{positions[0,1,:,0].min():.2f},{positions[0,1,:,1].max():.2f}], "
        f"width=[{positions[0,2,:,0].min():.2f},{positions[0,2,:,1].max():.2f}]"
    )

    # Initialize audio latents and positions if audio enabled
    audio_latents: Optional[torch.Tensor] = None
    audio_positions: Optional[torch.Tensor] = None
    audio_noise: Optional[torch.Tensor] = None
    audio_latent_frames = 0
    if not video_only:
        audio_latent_frames = compute_audio_latent_frames(config.num_frames, fps=fps)
        audio_latents = torch.randn(
            (1, audio_latent_frames, 128),
            generator=generator,
            device=transformer_device,
            dtype=dtype,
        )
        audio_positions = create_audio_position_indices(
            batch_size=1,
            audio_latent_frames=audio_latent_frames,
            device=torch.device(transformer_device),
        )
        # Save audio noise for stage 2 re-noising
        audio_noise = torch.randn(
            audio_latents.shape,
            generator=generator,
            device=audio_latents.device,
            dtype=audio_latents.dtype,
        )
        logger.info(f"Audio: {audio_latent_frames} latent frames ({config.num_frames / fps:.2f}s)")

    # Sigma schedule for stage 1
    is_distilled = two_stage.pipeline_mode == "distilled"
    if is_distilled:
        from llm_dit.models.ltx2.constants import DISTILLED_SIGMA_VALUES
        sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, device=transformer_device, dtype=dtype)
        logger.info(f"Stage 1: Using distilled sigma schedule ({len(sigmas) - 1} steps, no CFG)")
    else:
        scheduler = LTX2Scheduler()
        sigmas = scheduler.execute(
            steps=two_stage.stage1_steps,
            latent=None,  # Use reference default (4096 tokens)
            max_shift=config.max_shift,
            base_shift=config.base_shift,
            stretch=config.stretch,
            terminal=config.terminal,
        ).to(transformer_device, dtype)
    logger.debug(
        f"Stage 1 sigmas: [{sigmas[0]:.4f} -> {sigmas[-1]:.4f}], "
        f"{len(sigmas) - 1} steps, mode={'AV' if not video_only else 'video-only'}"
    )

    # -- DIAGNOSTIC: Checkpoint 4 - Sigma schedule --
    _diag.info(
        f"[DIAG] stage1_sigmas: first={sigmas[0]:.6f}, last={sigmas[-1]:.6f}, "
        f"steps={len(sigmas)-1}, all={[f'{s:.4f}' for s in sigmas.tolist()]}"
    )

    # Denoise stage 1
    # Distilled mode: no CFG (guidance baked into model), no STG
    if is_distilled:
        schedule = constant_schedule(guidance_scale=1.0)
    else:
        schedule = constant_schedule(
            guidance_scale=two_stage.guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            neg_embeds=neg_embeds,
            rescale_scale=two_stage.rescale_scale,
            ge_gamma=two_stage.ge_gamma,
            stg_scale=two_stage.stg_scale,
            stg_blocks=two_stage.stg_blocks,
            modality_scale=two_stage.modality_scale,
        )

    # Guard: audio enabled but no audio embeddings from encoder -- fall back to video-only
    if not video_only and pos_audio_embeds is None:
        logger.error("Audio enabled but no audio embeddings from encoder. Falling back to video-only.")
        video_only = True

    if not video_only and audio_latents is not None and audio_positions is not None:
        latents, audio_latents = _denoise_av_stage(
            model=model,
            video_latents=latents,
            audio_latents=audio_latents,
            video_prompt_embeds=pos_embeds,
            audio_prompt_embeds=pos_audio_embeds,
            sigmas=sigmas,
            video_positions=positions,
            audio_positions=audio_positions,
            stage_name="stage1_denoise",
            step_schedule=schedule,
            audio_neg_embeds=audio_neg_embeds,
            callback=callback,
            fbcache_threshold=two_stage.fbcache_threshold,
        )
    else:
        latents = _denoise_stage(
            model=model,
            latents=latents,
            prompt_embeds=pos_embeds,
            sigmas=sigmas,
            positions=positions,
            stage_name="stage1_denoise",
            step_schedule=schedule,
            callback=callback,
            fbcache_threshold=two_stage.fbcache_threshold,
        )

    # -- DIAGNOSTIC: Checkpoint 6 - Post-stage1 latents --
    _diag.info(_tensor_stats("post_stage1_latents", latents))

    # Reshape to spatial format for upsampler: [B, T, D] -> [B, D, T_lat, H_lat, W_lat]
    latents = latents.transpose(1, 2).reshape(1, 128, t_latent, h_latent, w_latent)

    del sigmas
    stage1_elapsed = time.perf_counter() - stage1_start
    logger.info(f"Stage 1 complete: {stage1_elapsed:.1f}s")

    # Release denoising intermediates before loading upsampler.
    # Stage 1 attention/FFN buffers can reserve 5-8 GB in the CUDA cache;
    # without this cleanup, Stage 1.5 + Stage 2 may not have enough headroom.
    if not skip_cleanup:
        cleanup_memory("post_stage1")

    # =========================================================================
    # Stage 1.5: Spatial Upsampling (2x)
    # =========================================================================
    # Transformer stays on GPU -- upsampler (~950MB) fits alongside it in 24GB.
    stage15_start = time.perf_counter()
    if callback:
        callback("upsample", 0, 1)

    logger.info("Stage 1.5: Loading spatial upsampler...")

    from llm_dit.models.ltx2.upsampler import load_spatial_upsampler

    upsampler_path = model_path / two_stage.spatial_upsampler_file
    upsampler = load_spatial_upsampler(upsampler_path, dtype=dtype, device="cpu")
    upsampler = upsampler.to(transformer_device)

    # Get per_channel_statistics from cached VAE or load briefly from disk.
    if cached_vae is not None:
        per_channel_stats = cached_vae.per_channel_statistics.to(transformer_device)
    else:
        from llm_dit.models.ltx2.vae import load_ltx2_vae_decoder
        vae_for_stats = load_ltx2_vae_decoder(
            _resolve_v23_component_path(model_path, "vae"), dtype=dtype, device="cpu"
        )
        per_channel_stats = vae_for_stats.per_channel_statistics
        per_channel_stats = per_channel_stats.to(transformer_device)

    # Un-normalize, upsample, re-normalize
    latents = latents.to(transformer_device)
    latents = per_channel_stats.un_normalize(latents)
    _diag.info(_tensor_stats("pre_upsample_unnorm", latents))
    latents = upsampler(latents)
    _diag.info(_tensor_stats("post_upsample", latents))
    latents = per_channel_stats.normalize(latents)

    # -- DIAGNOSTIC: Checkpoint 7 - Post-upsample latents --
    _diag.info(_tensor_stats("post_upsample_renorm", latents))

    del upsampler
    if cached_vae is None:
        del vae_for_stats  # Only exists when loaded from disk
    # per_channel_stats is either a detached copy or the cached VAE's sub-module
    # moved to GPU; the reference is no longer needed either way.
    del per_channel_stats
    if not skip_cleanup:
        cleanup_memory("post_stage1.5")

    if callback:
        callback("upsample", 1, 1)

    stage15_elapsed = time.perf_counter() - stage15_start
    logger.info(
        f"Stage 1.5 complete: {stage15_elapsed:.1f}s "
        f"(latent upsampled from {h_latent}x{w_latent} to {latents.shape[3]}x{latents.shape[4]})"
    )

    # =========================================================================
    # Stage 2: High-Resolution Refinement (full resolution, distilled LoRA)
    # =========================================================================
    # Reuse the Stage 1 transformer -- base LoRA(s) are already fused.
    # Only the distilled LoRA needs to be applied for Stage 2's 3-step schedule.
    stage2_start = time.perf_counter()
    if callback:
        from llm_dit.models.ltx2.constants import STAGE_2_DISTILLED_SIGMA_VALUES
        callback("stage2_denoise", 0, len(STAGE_2_DISTILLED_SIGMA_VALUES) - 1)

    logger.info("Stage 2: Applying distilled LoRA for high-res refinement (reusing Stage 1 model)...")

    # Apply distilled LoRA to the existing model (base LoRA already fused from Stage 1)
    if two_stage.distilled_lora_path and two_stage.distilled_lora_scale > 0:
        distilled_path = Path(two_stage.distilled_lora_path)
        if not distilled_path.is_absolute():
            distilled_path = model_path / distilled_path
        # Flush CUDA cache before LoRA fusion to ensure the allocator starts
        # with defragmented free memory. Stage 1 denoising + Stage 1.5 upsampling
        # leave fragmented cached blocks that can cause OOM during fusion.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Loading distilled LoRA: {distilled_path} (scale={two_stage.distilled_lora_scale})")
        # Detect native fp8 weights (fp8-cast model)
        import itertools
        has_fp8_weights = any(
            p.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            for p in itertools.islice(model.parameters(), 5)
        )
        if has_fp8_weights:
            _apply_distilled_lora_fp8(
                model, str(distilled_path),
                scale=two_stage.distilled_lora_scale,
            )
        else:
            from llm_dit.utils.lora import load_lora as _load_lora
            _load_lora(
                model, distilled_path,
                    scale=two_stage.distilled_lora_scale,
                    device=transformer_device, dtype=dtype,
                )
    else:
        logger.info("Stage 2: Skipping distilled LoRA (scale=0 or no path)")

    # Stage 2 uses distilled sigma schedule (pre-computed, not from scheduler)
    from llm_dit.models.ltx2.constants import STAGE_2_DISTILLED_SIGMA_VALUES

    distilled_sigmas = torch.tensor(
        STAGE_2_DISTILLED_SIGMA_VALUES, device=transformer_device, dtype=dtype
    )
    logger.debug(
        f"Stage 2 sigmas: [{distilled_sigmas[0]:.4f} -> {distilled_sigmas[-1]:.4f}], "
        f"{len(distilled_sigmas) - 1} steps (distilled, no CFG)"
    )

    # Full-resolution latent dimensions
    t_lat_full = (config.num_frames - 1) // 8 + 1
    h_lat_full = config.height // 32
    w_lat_full = config.width // 32
    num_tokens_full = t_lat_full * h_lat_full * w_lat_full

    # Reshape upsampled latents to [B, T, D] for denoising
    latents_flat = latents.reshape(1, 128, -1).transpose(1, 2)  # [B, T, D]

    # Flow-matching interpolation at the first distilled sigma (0.909375).
    # x_t = (1 - t) * x_0 + t * eps  -- NOT additive noise.
    noise_scale = distilled_sigmas[0].item()
    if config.seed is not None:
        generator = torch.Generator(device=transformer_device).manual_seed(config.seed + 1)
    noise = torch.randn_like(latents_flat, generator=generator if config.seed is not None else None)
    latents_noisy = (1 - noise_scale) * latents_flat + noise_scale * noise
    del noise, latents_flat

    # -- DIAGNOSTIC: Checkpoint 8 - Stage 2 re-noised latents --
    _diag.info(_tensor_stats("stage2_renoised", latents_noisy))
    _diag.info(f"[DIAG] stage2_noise_scale={noise_scale:.6f}")

    # Re-noise audio latents for stage 2 (same flow-matching interpolation)
    if not video_only and audio_latents is not None and audio_noise is not None:
        audio_latents_noisy = (1 - noise_scale) * audio_latents + noise_scale * audio_noise
        del audio_noise
    else:
        audio_latents_noisy = None

    # Full-resolution positions
    positions_full = create_position_indices(
        batch_size=1,
        num_frames=config.num_frames,
        height=config.height,
        width=config.width,
        device=torch.device(transformer_device),
        fps=fps,
        scale_factors=(8, 32, 32),
        causal_fix=True,
    )

    # Stage 2: no CFG (distilled model, guidance_scale=1.0 default), so audio_neg_embeds not needed
    if not video_only and audio_latents_noisy is not None and audio_positions is not None:
        latents_refined, audio_latents = _denoise_av_stage(
            model=model,
            video_latents=latents_noisy,
            audio_latents=audio_latents_noisy,
            video_prompt_embeds=pos_embeds,
            audio_prompt_embeds=pos_audio_embeds,
            sigmas=distilled_sigmas,
            video_positions=positions_full,
            audio_positions=audio_positions,
            stage_name="stage2_denoise",
            callback=callback,
            fbcache_threshold=two_stage.fbcache_threshold,
        )
    else:
        latents_refined = _denoise_stage(
            model=model,
            latents=latents_noisy,
            prompt_embeds=pos_embeds,
            sigmas=distilled_sigmas,
            positions=positions_full,
            stage_name="stage2_denoise",
            callback=callback,
            fbcache_threshold=two_stage.fbcache_threshold,
        )

    # -- DIAGNOSTIC: Checkpoint 9 - Post-stage2 latents --
    _diag.info(_tensor_stats("post_stage2_latents", latents_refined))

    # Reshape back to spatial: [B, T, D] -> [B, D, T_lat, H_lat, W_lat]
    latents = latents_refined.transpose(1, 2).reshape(1, 128, t_lat_full, h_lat_full, w_lat_full)

    del model, latents_noisy, latents_refined, pos_embeds, neg_embeds, pos_mask
    if not skip_cleanup:
        cleanup_memory()
        logger.info("Stage 2 transformer unloaded")

    stage2_elapsed = time.perf_counter() - stage2_start
    logger.info(f"Stage 2 complete: {stage2_elapsed:.1f}s")

    # =========================================================================
    # Stage 3: VAE Decoding
    # =========================================================================
    stage3_start = time.perf_counter()
    if callback:
        callback("decode", 0, 1)

    logger.info(f"Stage 3: Loading VAE decoder on {vae_device}...")

    vae_is_borrowed = cached_vae is not None
    if vae_is_borrowed:
        logger.info("Using cached VAE, shuttling to GPU...")
        vae = cached_vae.to(vae_device)
    else:
        from llm_dit.models.ltx2.vae import load_ltx2_vae_decoder
        vae = load_ltx2_vae_decoder(
            _resolve_v23_component_path(model_path, "vae"), dtype=dtype, device="cpu"
        ).to(vae_device)

    logger.info("Decoding latents to video...")

    decode_start = time.perf_counter()
    with torch.no_grad():
        video = vae(latents.to(vae_device))
    decode_elapsed = time.perf_counter() - decode_start
    logger.info(f"[Decode] VAE decode {decode_elapsed:.1f}s")

    # -- DIAGNOSTIC: Checkpoint 10 - VAE output --
    _diag.info(_tensor_stats("vae_output_raw", video))

    # Convert to [F, H, W, C] uint8
    video = video.squeeze(0).permute(1, 2, 3, 0)
    video = ((video + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
    _diag.info(
        f"[DIAG] vae_output_uint8: shape={list(video.shape)}, "
        f"mean={video.float().mean():.1f}, min={video.min().item()}, max={video.max().item()}"
    )

    # Return or unload VAE
    if vae_is_borrowed:
        vae.to("cpu")  # Return to CPU pinned memory
        logger.info("Cached VAE returned to CPU")
    else:
        del vae
    del latents
    if not skip_cleanup:
        cleanup_memory()

    if callback:
        callback("decode", 1, 1)

    stage3_elapsed = time.perf_counter() - stage3_start
    logger.info(f"Stage 3 complete: {stage3_elapsed:.1f}s")

    # =========================================================================
    # Stage 4: Audio Decode (when audio is enabled)
    # =========================================================================
    audio_waveform = None
    if not video_only and audio_latents is not None:
        stage4_start = time.perf_counter()
        if callback:
            callback("audio_decode", 0, 1)

        logger.info("Stage 4: Decoding audio latents...")

        # Reshape: [B, T, D] -> [B, C, T_audio, mel_bins] = [B, 8, T, 16]
        audio_latents_4d = audio_latents.reshape(1, audio_latent_frames, 8, 16)
        audio_latents_4d = audio_latents_4d.permute(0, 2, 1, 3)  # [B, 8, T, 16]

        # Audio decoder: latents -> mel spectrogram
        if cached_audio_decoder is not None:
            audio_decoder = cached_audio_decoder.to(vae_device)
            with torch.no_grad():
                mel = audio_decoder(audio_latents_4d.to(vae_device))
            audio_decoder.to("cpu")
            logger.info("Cached audio decoder returned to CPU")
        else:
            from llm_dit.models.ltx2.audio_vae.loader import load_audio_decoder
            audio_decoder = load_audio_decoder(
                _resolve_v23_component_path(model_path, "audio_vae"), dtype=dtype, device=vae_device,
            )
            with torch.no_grad():
                mel = audio_decoder(audio_latents_4d.to(vae_device))
            del audio_decoder

        # Vocoder: mel -> waveform
        if cached_vocoder is not None:
            vocoder = cached_vocoder.to(vae_device)
            with torch.no_grad():
                audio_waveform = vocoder(mel)
            audio_sample_rate = vocoder.output_sample_rate
            vocoder.to("cpu")
            logger.info(f"Cached vocoder returned to CPU (output_rate={audio_sample_rate}Hz)")
        else:
            from llm_dit.models.ltx2.audio_vae.loader import load_vocoder
            vocoder = load_vocoder(
                _resolve_v23_component_path(model_path, "vocoder"), dtype=dtype, device=vae_device,
            )
            with torch.no_grad():
                audio_waveform = vocoder(mel)
            audio_sample_rate = vocoder.output_sample_rate
            del vocoder

        del mel, audio_latents_4d, audio_latents
        if not skip_cleanup:
            cleanup_memory("post_audio_decode")

        if callback:
            callback("audio_decode", 1, 1)

        stage4_elapsed = time.perf_counter() - stage4_start
        logger.info(f"Stage 4 complete: {stage4_elapsed:.1f}s")
    else:
        stage4_elapsed = 0.0
        audio_sample_rate = 24000

    gen_elapsed = time.perf_counter() - gen_start
    timing = (
        f"encode={stage0_elapsed:.1f}s, stage1={stage1_elapsed:.1f}s, "
        f"upsample={stage15_elapsed:.1f}s, stage2={stage2_elapsed:.1f}s, "
        f"decode={stage3_elapsed:.1f}s"
    )
    if stage4_elapsed > 0:
        timing += f", audio={stage4_elapsed:.1f}s"
    logger.info(f"Two-stage generation complete: {gen_elapsed:.1f}s total ({timing})")

    if audio_waveform is not None:
        return video, audio_waveform, audio_sample_rate
    return video
