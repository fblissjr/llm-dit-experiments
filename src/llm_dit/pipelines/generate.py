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
from llm_dit.utils.memory import cleanup_memory

logger = logging.getLogger(__name__)

# Shorthand aliases for quantization method strings
_QUANT_ALIASES = {"fp8": "fp8-weight-only"}


def _resolve_quantize(quantize: str) -> tuple[bool, str]:
    """Normalize quantize string shorthand to (should_quantize, precision) tuple.

    Handles aliases like "fp8" -> "fp8-weight-only" and falsy values.

    Args:
        quantize: Quantization method string. "none", "", or None disables
            quantization. "fp8" is aliased to "fp8-weight-only".

    Returns:
        (should_quantize, precision): Whether to quantize and the resolved method.
    """
    if quantize in (None, "", "none"):
        return False, "none"
    precision = _QUANT_ALIASES.get(quantize, quantize)
    return True, precision


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
            fps=24.0,
            scale_factors=(8, 32, 32),  # LTX-2 VAE compression factors
            causal_fix=True,
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
    skip_cleanup: bool = False,
) -> torch.Tensor:
    """
    Generate video with sequential component offloading for 24GB GPUs.

    This function implements the LTX-2 memory strategy:
    1. Load text encoder -> encode prompt -> unload (skipped if precomputed_embeddings)
    2. Load transformer (+ optional quantization) -> denoise -> unload
    3. Load VAE -> decode latents -> unload

    Each component is loaded and unloaded sequentially to stay within 24GB VRAM.
    With FP8 quantization, the 13B model fits with room for activations.

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
        lora_scale: LoRA scale factor(s). None defaults to 0.8 per LoRA.
        text_encoder_device: Device for Gemma3 text encoder.
        transformer_device: Device for DiT transformer.
        vae_device: Device for VAE decoder.
        quantize: Transformer quantization method: none, fp8, fp8-weight-only,
            fp8-dynamic, int8, int4.
        skip_cleanup: Skip memory cleanup between stages.

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

        # Use variant factory for flexible Gemma3 loading
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
            # Default bf16 path - use original Gemma3Encoder for compatibility
            from llm_dit.encoders.gemma3 import Gemma3Encoder
            text_encoder = Gemma3Encoder(
                model_id=str(text_encoder_path),
                device=text_encoder_device,
                dtype=dtype,
            )

        logger.info("Encoding prompt...")
        encoding_output = text_encoder.encode([prompt])
        # EncodingOutput has embeddings list and attention_masks list
        prompt_embeds = encoding_output.embeddings[0].unsqueeze(0)  # [1, seq_len, dim]
        attention_mask = encoding_output.attention_masks[0].unsqueeze(0)  # [1, seq_len]
        logger.debug(f"Prompt embeddings: {prompt_embeds.shape}")

        # Move embeddings to transformer device, unload encoder
        prompt_embeds = prompt_embeds.to(transformer_device, dtype)
        attention_mask = attention_mask.to(transformer_device)

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

    # Always load in BF16 first, then apply quantization via unified system
    from llm_dit.models.ltx2 import load_ltx2_transformer

    # Load to CPU first if quantizing (reduces peak GPU memory)
    load_device = "cpu" if effective_quantize else transformer_device
    model = load_ltx2_transformer(
        model_path / "transformer",
        dtype=dtype,
        device=load_device,
        video_only=True,
    )

    if effective_quantize and effective_precision != "none":
        from llm_dit.quantization import quantize_component

        # Apply quantization on CPU, then move to GPU (smaller transfer)
        model, stats = quantize_component(  # type: ignore[assignment]
            model,
            method=effective_precision,
            component_type="transformer",
        )
        logger.info(
            f"Transformer quantized: {stats['quantized_layers']}/{stats['total_layers']} layers "
            f"({effective_precision})"
        )

    if load_device == "cpu":
        model = model.to(transformer_device)

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

    # Load LoRA weights if specified
    if lora_path is not None:
        from llm_dit.utils.lora import load_lora as _load_lora

        # Normalize to lists
        if isinstance(lora_path, (str, Path)):
            lora_paths = [lora_path]
        else:
            lora_paths = list(lora_path)

        if lora_scale is None:
            lora_scales = [0.8] * len(lora_paths)  # Default scale
        elif isinstance(lora_scale, (int, float)):
            lora_scales = [float(lora_scale)] * len(lora_paths)
        else:
            lora_scales = list(lora_scale)

        if len(lora_paths) != len(lora_scales):
            raise ValueError(
                f"Number of LoRA paths ({len(lora_paths)}) must match "
                f"number of scales ({len(lora_scales)})"
            )

        total_updated = 0
        for path, scale in zip(lora_paths, lora_scales):
            logger.info(f"Loading LoRA: {path} (scale={scale})")
            updated = _load_lora(
                model,
                path,
                scale=scale,
                device=transformer_device,
                dtype=dtype,
            )
            total_updated += updated
        logger.info(f"LoRA loading complete: {total_updated} layers updated")

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

    # Pure PyTorch VAE decoder (no diffusers dependency)
    from llm_dit.models.ltx2.vae import load_ltx2_vae_decoder

    vae = load_ltx2_vae_decoder(
        model_path / "vae",
        dtype=dtype,
        device="cpu",
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

    # Unload VAE
    del vae, latents
    if not skip_cleanup:
        cleanup_memory()
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
    neg_embeds: Optional[torch.Tensor] = None
    rescale_scale: float = 0.0
    ge_gamma: float = 0.0
    stg_scale: float = 0.0
    stg_blocks: Optional[List[int]] = None


# A callable that returns StepContext for a given (step_index, sigma_value)
StepSchedule = Callable[[int, float], StepContext]


def constant_schedule(
    guidance_scale: float = 1.0,
    neg_embeds: Optional[torch.Tensor] = None,
    rescale_scale: float = 0.0,
    ge_gamma: float = 0.0,
    stg_scale: float = 0.0,
    stg_blocks: Optional[List[int]] = None,
) -> StepSchedule:
    """Static parameters for all steps (default behavior)."""
    ctx = StepContext(
        guidance_scale=guidance_scale,
        neg_embeds=neg_embeds,
        rescale_scale=rescale_scale,
        ge_gamma=ge_gamma,
        stg_scale=stg_scale,
        stg_blocks=stg_blocks,
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
    stage1_steps: int = 40
    guidance_scale: float = 3.0

    # Guidance options (stage 1 only)
    stg_scale: float = 1.0  # Spatio-temporal guidance scale (0=disabled, 1.0=reference)
    stg_blocks: list[int] = None  # type: ignore[assignment]
    rescale_scale: float = 0.7  # CFG rescaling

    # Negative prompt (reference DEFAULT_NEGATIVE_PROMPT from constants.py)
    negative_prompt: str = (
        "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
        "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
        "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
        "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
        "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
        "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
        "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
        "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
        "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
        "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
        "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
    )

    # Gradient estimation
    ge_gamma: float = 0.0  # 0=disabled, 2.0=reference default

    # Stage 2 (high-res refinement)
    stage2_steps: int = 3
    distilled_lora_path: str = ""
    distilled_lora_scale: float = 1.0

    # Spatial upsampler
    spatial_upsampler_file: str = "ltx-2-spatial-upscaler-x2-1.0.safetensors"

    def __post_init__(self):
        if self.stg_blocks is None:
            self.stg_blocks = [29]


def _compute_velocity(
    model: LTX2Transformer,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    positions: torch.Tensor,
    prompt_embeds: torch.Tensor,
    ctx: StepContext,
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

    Returns:
        Velocity prediction tensor [B, T, D].
    """
    if ctx.guidance_scale > 1.0 and ctx.neg_embeds is not None:
        # Pass 1: Unconditional (negative prompt)
        uncond_modality = create_video_modality(latents, timestep, positions, ctx.neg_embeds)
        velocity_uncond, _ = model(video=uncond_modality)
        del uncond_modality
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Pass 2: Conditional (positive prompt)
        cond_modality = create_video_modality(latents, timestep, positions, prompt_embeds)
        velocity_cond, _ = model(video=cond_modality)

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
            velocity_perturbed, _ = model(video=stg_modality, stg_blocks=stg_blocks_set)
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
        velocity, _ = model(video=modality)
        del modality

    return velocity


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

    Returns:
        Denoised latent tensor [B, T, D].
    """
    if step_schedule is None:
        step_schedule = constant_schedule()

    dtype = latents.dtype
    num_tokens = latents.shape[1]
    num_steps = len(sigmas) - 1

    model.train(False)

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

            velocity = _compute_velocity(
                model, latents, timestep, positions, prompt_embeds, ctx,
            )

            # Gradient estimation correction
            if ctx.ge_gamma > 0 and prev_velocity is not None:
                delta_v = velocity - prev_velocity
                velocity = ctx.ge_gamma * delta_v + prev_velocity

            # Save velocity for GE (before Euler step modifies it)
            if ctx.ge_gamma > 0:
                prev_velocity = velocity.clone()

            # Euler step: x_{t-1} = x_t + v * dt
            dt = sigma_next - sigma
            denoised = (latents.float() + velocity.float() * dt).to(dtype)

            # Post-process conditioned regions
            if denoise_mask is not None and clean_latent is not None:
                latents = post_process_latent(denoised, denoise_mask, clean_latent)
            else:
                latents = denoised

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
    skip_cleanup: bool = False,
) -> torch.Tensor:
    """Generate video using two-stage pipeline with spatial upsampling.

    Reference: TI2VidTwoStagesPipeline from official LTX-2 repo.

    Flow:
      Stage 0: Encode text (positive + negative prompts)
      Stage 1: Denoise at half resolution with CFG guidance
      Stage 1.5: Spatial upsample latents 2x
      Stage 2: Refine at full resolution with distilled LoRA (no CFG, 3 steps)
      Stage 3: VAE decode to pixels

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

    Returns:
        Video tensor [F, H, W, C] in uint8 format.
    """
    if not two_stage.distilled_lora_path:
        raise ValueError(
            "Two-stage generation requires a distilled LoRA for stage 2 refinement. "
            "Set distilled_lora_path in TwoStageConfig (e.g., 'ltx-2-19b-distilled-lora-384.safetensors'). "
            "Without it, the base model cannot denoise in 3 steps and will produce garbage output."
        )

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

    # Encode positive prompt
    logger.info("Encoding positive prompt...")
    pos_output = text_encoder.encode([prompt])
    pos_embeds = pos_output.embeddings[0].unsqueeze(0)
    pos_mask = pos_output.attention_masks[0].unsqueeze(0)

    if callback:
        callback("encoding", 1, 2)

    # Encode negative prompt
    logger.info("Encoding negative prompt...")
    neg_output = text_encoder.encode([two_stage.negative_prompt])
    neg_embeds = neg_output.embeddings[0].unsqueeze(0)

    if callback:
        callback("encoding", 2, 2)

    # Move to transformer device, unload encoder
    pos_embeds = pos_embeds.to(transformer_device, dtype)
    pos_mask = pos_mask.to(transformer_device)
    neg_embeds = neg_embeds.to(transformer_device, dtype)

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

    from llm_dit.models.ltx2 import load_ltx2_transformer

    load_device = "cpu" if effective_quantize else transformer_device
    model = load_ltx2_transformer(
        model_path / "transformer",
        dtype=dtype,
        device=load_device,
        video_only=True,
    )

    if effective_quantize and effective_precision != "none":
        from llm_dit.quantization import quantize_component
        model, stats = quantize_component(
            model,
            method=effective_precision,
            component_type="transformer",
        )
        logger.info(
            f"Transformer quantized: {stats['quantized_layers']}/{stats['total_layers']} layers "
            f"({effective_precision})"
        )

    if load_device == "cpu":
        model = model.to(transformer_device)

    # Apply base LoRA(s) if provided
    if lora_path is not None:
        from llm_dit.utils.lora import load_lora as _load_lora

        if isinstance(lora_path, (str, Path)):
            lora_paths = [lora_path]
        else:
            lora_paths = list(lora_path)

        if lora_scale is None:
            lora_scales = [0.8] * len(lora_paths)
        elif isinstance(lora_scale, (int, float)):
            lora_scales = [float(lora_scale)] * len(lora_paths)
        else:
            lora_scales = list(lora_scale)

        for path, scale in zip(lora_paths, lora_scales):
            logger.info(f"Loading base LoRA: {path} (scale={scale})")
            _load_lora(model, path, scale=scale, device=transformer_device, dtype=dtype)

    # Initialize latent noise at half resolution
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

    positions = create_position_indices(
        batch_size=1,
        num_frames=config.num_frames,
        height=stage1_config.height,
        width=stage1_config.width,
        device=torch.device(transformer_device),
        fps=24.0,
        scale_factors=(8, 32, 32),
        causal_fix=True,
    )

    # Sigma schedule for stage 1
    scheduler = LTX2Scheduler()
    mock_latent = torch.empty(1, 128, t_latent, h_latent, w_latent)
    sigmas = scheduler.execute(
        steps=two_stage.stage1_steps,
        latent=mock_latent,
        max_shift=config.max_shift,
        base_shift=config.base_shift,
        stretch=config.stretch,
        terminal=config.terminal,
    ).to(transformer_device, dtype)

    # Denoise stage 1
    schedule = constant_schedule(
        guidance_scale=two_stage.guidance_scale,
        neg_embeds=neg_embeds,
        rescale_scale=two_stage.rescale_scale,
        ge_gamma=two_stage.ge_gamma,
        stg_scale=two_stage.stg_scale,
        stg_blocks=two_stage.stg_blocks,
    )
    latents = _denoise_stage(
        model=model,
        latents=latents,
        prompt_embeds=pos_embeds,
        sigmas=sigmas,
        positions=positions,
        stage_name="stage1_denoise",
        step_schedule=schedule,
        callback=callback,
    )

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

    # Load VAE briefly just for per_channel_statistics, then discard.
    from llm_dit.models.ltx2.vae import load_ltx2_vae_decoder

    vae_for_stats = load_ltx2_vae_decoder(
        model_path / "vae", dtype=dtype, device="cpu"
    )
    per_channel_stats = vae_for_stats.per_channel_statistics
    per_channel_stats = per_channel_stats.to(transformer_device)

    # Un-normalize, upsample, re-normalize
    latents = latents.to(transformer_device)
    latents = per_channel_stats.un_normalize(latents)
    latents = upsampler(latents)
    latents = per_channel_stats.normalize(latents)

    del upsampler, vae_for_stats, per_channel_stats
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
        callback("stage2_denoise", 0, two_stage.stage2_steps)

    logger.info("Stage 2: Applying distilled LoRA for high-res refinement (reusing Stage 1 model)...")

    # Apply distilled LoRA to the existing model (base LoRA already fused from Stage 1)
    from llm_dit.utils.lora import load_lora as _load_lora

    if two_stage.distilled_lora_path:
        distilled_path = Path(two_stage.distilled_lora_path)
        if not distilled_path.is_absolute():
            distilled_path = model_path / distilled_path
        # Flush CUDA cache before LoRA fusion to ensure the allocator starts
        # with defragmented free memory. Stage 1 denoising + Stage 1.5 upsampling
        # leave fragmented cached blocks that can cause OOM during fusion.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Loading distilled LoRA: {distilled_path} (scale={two_stage.distilled_lora_scale})")
        _load_lora(
            model,
            distilled_path,
            scale=two_stage.distilled_lora_scale,
            device=transformer_device,
            dtype=dtype,
        )

    # Stage 2 uses distilled sigma schedule (pre-computed, not from scheduler)
    from llm_dit.models.ltx2.constants import STAGE_2_DISTILLED_SIGMA_VALUES

    distilled_sigmas = torch.tensor(
        STAGE_2_DISTILLED_SIGMA_VALUES, device=transformer_device, dtype=dtype
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

    # Full-resolution positions
    positions_full = create_position_indices(
        batch_size=1,
        num_frames=config.num_frames,
        height=config.height,
        width=config.width,
        device=torch.device(transformer_device),
        fps=24.0,
        scale_factors=(8, 32, 32),
        causal_fix=True,
    )

    # Denoise stage 2 (no CFG, simple denoising -- defaults to constant_schedule())
    latents_refined = _denoise_stage(
        model=model,
        latents=latents_noisy,
        prompt_embeds=pos_embeds,
        sigmas=distilled_sigmas,
        positions=positions_full,
        stage_name="stage2_denoise",
        callback=callback,
    )

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

    from llm_dit.models.ltx2.vae import load_ltx2_vae_decoder

    vae = load_ltx2_vae_decoder(
        model_path / "vae", dtype=dtype, device="cpu"
    ).to(vae_device)

    logger.info("Decoding latents to video...")

    decode_start = time.perf_counter()
    with torch.no_grad():
        video = vae(latents.to(vae_device))
    decode_elapsed = time.perf_counter() - decode_start
    logger.info(f"[Decode] VAE decode {decode_elapsed:.1f}s")

    # Convert to [F, H, W, C] uint8
    video = video.squeeze(0).permute(1, 2, 3, 0)
    video = ((video + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)

    del vae, latents
    if not skip_cleanup:
        cleanup_memory()

    if callback:
        callback("decode", 1, 1)

    stage3_elapsed = time.perf_counter() - stage3_start
    logger.info(f"Stage 3 complete: {stage3_elapsed:.1f}s")

    gen_elapsed = time.perf_counter() - gen_start
    logger.info(
        f"Two-stage generation complete: {gen_elapsed:.1f}s total "
        f"(encode={stage0_elapsed:.1f}s, stage1={stage1_elapsed:.1f}s, "
        f"upsample={stage15_elapsed:.1f}s, stage2={stage2_elapsed:.1f}s, "
        f"decode={stage3_elapsed:.1f}s)"
    )
    return video
