"""
LTX-2 Pure PyTorch Generation Module.

Last Updated: 2026-01-18

Pure PyTorch implementation of the LTX-2 diffusion generation loop.
Used by both the pipeline and experiment infrastructure.

This module provides the core generation logic without diffusers dependency:
- Sigma schedule generation via LTX2Scheduler
- Euler denoising loop with CFG
- VAE decoding with our ported VideoDecoder

Original source: https://github.com/Lightricks/LTX-2
License: LTX-2 Community License
Copyright (c) 2025 Lightricks Ltd.
"""

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

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
from llm_dit.schedulers import LTX2Scheduler

logger = logging.getLogger(__name__)


def cleanup_memory() -> None:
    """Free GPU memory between stages."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@dataclass
class GenerationConfig:
    """Configuration for pure PyTorch video generation."""

    num_frames: int = 33
    height: int = 512
    width: int = 768
    num_inference_steps: int = 50
    guidance_scale: float = 3.0
    seed: Optional[int] = None

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
) -> torch.Tensor:
    """
    Create 3D position indices [B, 3, T, 2] for video (temporal, height, width).

    LTX-2 uses position bounds [start, end) for each patch to enable temporal
    interpolation with use_middle_indices_grid=True. The model computes the
    middle point of each patch's bounds for RoPE.

    LTX-2 VAE compression: 32x spatial, 8x temporal.
    Reference: coderef/LTX-2/ltx-core/components/patchifiers.py

    Args:
        batch_size: Batch size
        num_frames: Number of video frames
        height: Video height in pixels
        width: Video width in pixels
        device: Target device
        fps: Frames per second (for temporal scaling, default 24)

    Returns:
        Position indices tensor [B, 3, T, 2] where:
        - T = t_latent * h_latent * w_latent (flattened patches)
        - Last dim is [start, end] bounds
        - Temporal dim (positions[:, 0]) is scaled to seconds
    """
    t_latent = (num_frames - 1) // 8 + 1
    h_latent = height // 32
    w_latent = width // 32

    # Create meshgrid of position indices
    t_indices = torch.arange(t_latent, device=device)
    h_indices = torch.arange(h_latent, device=device)
    w_indices = torch.arange(w_latent, device=device)

    # Create 3D grid: [t_latent, h_latent, w_latent]
    # Order is (t, h, w) matching the official implementation
    grid_t, grid_h, grid_w = torch.meshgrid(t_indices, h_indices, w_indices, indexing="ij")

    # Create start and end positions (each patch spans [start, start+1))
    # Shape: [3, t_latent, h_latent, w_latent]
    patch_starts = torch.stack([grid_t, grid_h, grid_w], dim=0).float()
    patch_ends = patch_starts + 1.0

    # Stack start/end into bounds: [3, t_latent, h_latent, w_latent, 2]
    positions = torch.stack([patch_starts, patch_ends], dim=-1)

    # Flatten spatial dims: [3, T, 2] where T = t_latent * h_latent * w_latent
    positions = positions.view(3, -1, 2)

    # Scale temporal positions to seconds (divide by fps)
    # This matches official: positions[:, 0, ...] = positions[:, 0, ...] / self.fps
    positions[0] = positions[0] / fps

    # Expand for batch: [B, 3, T, 2]
    return positions.unsqueeze(0).expand(batch_size, -1, -1, -1)


def create_video_modality(
    latent: torch.Tensor,
    timestep: torch.Tensor,
    positions: torch.Tensor,
    prompt_embeds: torch.Tensor,
    context_mask: Optional[torch.Tensor] = None,
) -> Modality:
    """
    Create Modality dataclass for transformer input.

    Args:
        latent: [B, T, D] latent tokens (D=128 for LTX-2)
        timestep: [B, T] per-token timesteps
        positions: [B, 3, T] position indices
        prompt_embeds: [B, seq_len, context_dim] text embeddings
        context_mask: Optional [B, seq_len] attention mask

    Returns:
        Modality dataclass ready for transformer forward pass
    """
    return Modality(
        latent=latent,
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
            1, config.num_frames, config.height, config.width, device
        )
        denoise_mask = None
        clean_latent = None

    # =========================================================================
    # Step 3: Get sigma schedule from scheduler
    # =========================================================================
    scheduler = LTX2Scheduler()

    # Create mock latent for scheduler (it needs shape for token count)
    mock_latent = torch.empty(1, 128, t_latent, h_latent, w_latent)

    sigmas = scheduler.execute(
        steps=config.num_inference_steps,
        latent=mock_latent,
        max_shift=config.max_shift,
        base_shift=config.base_shift,
        stretch=config.stretch,
        terminal=config.terminal,
    )
    sigmas = sigmas.to(device, dtype)

    # =========================================================================
    # Step 4: Denoising loop (Euler method with velocity prediction)
    # =========================================================================
    for i in tqdm(range(len(sigmas) - 1), desc="Denoising"):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        # Timestep for model in [0, 1000] range (LTX-2 convention)
        # When conditioning is used, timesteps are per-token scaled by denoise_mask
        if denoise_mask is not None:
            # Per-token timesteps: conditioned regions get lower timesteps
            timestep = timesteps_from_mask(denoise_mask, sigma)
            timestep = timestep.squeeze(-1)  # [B, T, 1] -> [B, T] for Modality
        else:
            # Uniform timesteps for standard T2V
            timestep = (sigma * 1000).expand(1, num_tokens)

        # Classifier-free guidance
        if config.guidance_scale > 1.0:
            # Unconditional pass (zero embeddings)
            uncond_embeds = torch.zeros_like(prompt_embeds)
            uncond_modality = create_video_modality(latents, timestep, positions, uncond_embeds)
            velocity_uncond, _ = model(video=uncond_modality)

            # Conditional pass
            cond_modality = create_video_modality(latents, timestep, positions, prompt_embeds)
            velocity_cond, _ = model(video=cond_modality)

            # CFG blend
            velocity = velocity_cond + (config.guidance_scale - 1.0) * (
                velocity_cond - velocity_uncond
            )
        else:
            modality = create_video_modality(latents, timestep, positions, prompt_embeds)
            velocity, _ = model(video=modality)

        # Euler step: x_{t-1} = x_t + v * dt
        dt = sigma_next - sigma
        denoised = latents + velocity * dt

        # Post-process: blend with clean_latent based on denoise_mask
        # This preserves conditioned regions (mask=0) while denoising others
        if denoise_mask is not None and clean_latent is not None:
            latents = post_process_latent(denoised, denoise_mask, clean_latent)
        else:
            latents = denoised

        # Progress callback
        if callback is not None:
            callback(i + 1, len(sigmas) - 1, latents)

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

    # Denormalize latents before VAE decode
    # Check for diffusers VAE (has latents_mean attribute) or our VAE (uses config)
    if hasattr(vae, "latents_mean") and vae.latents_mean is not None:
        # Diffusers VAE - use its built-in normalization params
        latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(device, dtype)
        latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(device, dtype)
        scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
        latents = latents * latents_std / scaling_factor + latents_mean
    elif config.latents_mean is not None and config.latents_std is not None:
        # Our VAE - use config normalization params
        latents_mean = config.latents_mean.view(1, -1, 1, 1, 1).to(device, dtype)
        latents_std = config.latents_std.view(1, -1, 1, 1, 1).to(device, dtype)
        latents = latents * latents_std / config.scaling_factor + latents_mean

    # Decode to pixel space
    # Support both diffusers VAE (decode method) and our VAE (direct call)
    with torch.no_grad():
        if hasattr(vae, "decode"):
            # Diffusers VAE interface
            video = vae.decode(latents).sample
        else:
            # Our VideoDecoder - direct call
            video = vae(latents)

    # Convert to [F, H, W, C] uint8 format
    video = video.squeeze(0).permute(1, 2, 3, 0)  # [B, C, T, H, W] -> [T, H, W, C]
    video = ((video + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)

    return video


def generate_video_with_offloading(
    prompt: str,
    config: GenerationConfig,
    model_path: Union[str, Path] = "models/LTX-2",
    text_encoder_path: Optional[Union[str, Path]] = None,
    quantize: bool = True,
    precision: str = "fp8-quanto",
    dtype: torch.dtype = torch.bfloat16,
    callback: Optional[Callable[[str, int, int], None]] = None,
) -> torch.Tensor:
    """
    Generate video with sequential component offloading for 24GB GPUs.

    This function implements the LTX-2 memory strategy:
    1. Load text encoder -> encode prompt -> unload
    2. Load transformer (+ optional quantization) -> denoise -> unload
    3. Load VAE -> decode latents -> unload

    Each component is loaded and unloaded sequentially to stay within 24GB VRAM.
    With FP8 quantization, the 13B model fits with room for activations.

    Args:
        prompt: Text prompt for generation
        config: Generation configuration
        model_path: Path to LTX-2 model directory (contains transformer/, text_encoder/, vae/)
        text_encoder_path: Optional separate path for text encoder
        quantize: If True, quantize transformer to FP8 for memory efficiency
        precision: Quantization precision (fp8-quanto recommended)
        dtype: Base dtype for loading (bf16 recommended)
        callback: Optional callback(stage, step, total) for progress

    Returns:
        Video tensor [F, H, W, C] in uint8 format

    Example:
        video = generate_video_with_offloading(
            "A cat walking",
            GenerationConfig(num_frames=33, height=512, width=768),
            model_path="models/LTX-2",
            quantize=True,
        )

    Memory usage (RTX 4090, 24GB):
        - Text encoder (Gemma3): ~8GB peak
        - Transformer (FP8): ~13GB
        - VAE: ~2GB
        - Total per stage: <24GB
    """
    model_path = Path(model_path)
    if text_encoder_path is None:
        text_encoder_path = model_path / "text_encoder"

    # Stage 1: Text Encoding
    if callback:
        callback("text_encoder", 0, 1)

    logger.info("Stage 1: Loading text encoder...")
    from llm_dit.encoders.gemma3 import Gemma3Encoder

    text_encoder = Gemma3Encoder(
        model_id=str(text_encoder_path),
        device="cuda",
        dtype=dtype,
        load_in_8bit=True,  # 8-bit for memory efficiency
    )

    logger.info("Encoding prompt...")
    encoding_output = text_encoder.encode([prompt])
    # EncodingOutput has embeddings list and attention_masks list
    prompt_embeds = encoding_output.embeddings[0].unsqueeze(0)  # [1, seq_len, dim]
    attention_mask = encoding_output.attention_masks[0].unsqueeze(0)  # [1, seq_len]
    logger.info(f"Prompt embeddings: {prompt_embeds.shape}")

    # Keep embeddings on GPU, unload encoder
    prompt_embeds = prompt_embeds.to("cuda", dtype)
    attention_mask = attention_mask.to("cuda")

    del text_encoder
    cleanup_memory()
    logger.info("Text encoder unloaded")

    if callback:
        callback("text_encoder", 1, 1)

    # Stage 2: Transformer Denoising
    if callback:
        callback("transformer", 0, config.num_inference_steps)

    logger.info("Stage 2: Loading transformer...")

    if quantize:
        from llm_dit.models.ltx2 import load_ltx2_transformer_quantized

        model = load_ltx2_transformer_quantized(
            model_path / "transformer",
            precision=precision,
            dtype=dtype,
            video_only=True,
            verbose=True,
        )
        model = model.to("cuda")
    else:
        from llm_dit.models.ltx2 import load_ltx2_transformer

        model = load_ltx2_transformer(
            model_path / "transformer",
            dtype=dtype,
            device="cpu",
            video_only=True,
        )
        model = model.to("cuda")

    # Load connectors (these are in the text_encoder checkpoint, not transformer)
    from llm_dit.models.ltx2.connectors import load_ltx2_connectors

    connectors = load_ltx2_connectors(
        text_encoder_path,  # Connectors are part of text encoder weights
        device="cuda",
        dtype=dtype,
    )

    logger.info(f"Transformer loaded: {model.get_num_params() / 1e9:.2f}B params")

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
    )

    # Unload transformer and connectors
    del model, connectors, prompt_embeds, attention_mask
    cleanup_memory()
    logger.info("Transformer unloaded")

    if callback:
        callback("transformer", config.num_inference_steps, config.num_inference_steps)

    # Stage 3: VAE Decoding
    if callback:
        callback("vae", 0, 1)

    logger.info("Stage 3: Loading VAE decoder...")

    # Use diffusers VAE (our ported VAE doesn't have weight loader yet)
    from diffusers import AutoencoderKLLTXVideo

    vae = AutoencoderKLLTXVideo.from_pretrained(
        str(model_path / "vae"),
        dtype=dtype,
    ).to("cuda")

    logger.info("Decoding latents to video...")

    # Denormalize latents
    if hasattr(vae, "latents_mean") and vae.latents_mean is not None:
        latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to("cuda", dtype)
        latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to("cuda", dtype)
        scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
        latents = latents * latents_std / scaling_factor + latents_mean

    with torch.no_grad():
        video = vae.decode(latents).sample

    # Convert to [F, H, W, C] uint8
    video = video.squeeze(0).permute(1, 2, 3, 0)
    video = ((video + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)

    # Unload VAE
    del vae, latents
    cleanup_memory()
    logger.info("VAE unloaded")

    if callback:
        callback("vae", 1, 1)

    logger.info(f"Generation complete: {video.shape}")
    return video
