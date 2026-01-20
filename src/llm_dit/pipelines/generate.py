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
    model.eval()  # PyTorch evaluation mode (not Python eval)
    with torch.no_grad():
      for i in tqdm(range(len(sigmas) - 1), desc="Denoising"):
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

        # Classifier-free guidance
        if config.guidance_scale > 1.0:
            # Set debug step for attention diagnostics (only on step 0)
            for block in model.transformer_blocks:
                block._debug_step = i
            # Also enable cross-attention KV debug for block 0 at step 0, 20, and 39
            if i in [0, 20, len(sigmas) - 2]:
                model.transformer_blocks[0].attn2._debug_cross_attn = True
                model.transformer_blocks[0].attn2._debug_cross_attn_post = True
                print(f"[DEBUG] Enabling cross-attn debug for step {i}")

            # Unconditional pass (zero embeddings)
            uncond_embeds = torch.zeros_like(prompt_embeds)
            # DEBUG: Verify conditional embeddings are different from unconditional
            if i == 0:
                print(f"[DEBUG CFG] prompt_embeds (cond): mean={prompt_embeds.mean():.4f}, std={prompt_embeds.std():.4f}")
                print(f"[DEBUG CFG] uncond_embeds: mean={uncond_embeds.mean():.4f}, std={uncond_embeds.std():.4f}")
                print(f"[DEBUG UNCOND PASS]")
            uncond_modality = create_video_modality(latents, timestep, positions, uncond_embeds)
            velocity_uncond, _ = model(video=uncond_modality)
            del uncond_modality, uncond_embeds  # Free memory before conditional pass
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # Conditional pass
            if i in [0, 20, len(sigmas) - 2]:
                print(f"[DEBUG COND PASS - Step {i}]")
                model.transformer_blocks[0].attn2._debug_cross_attn = True  # Re-enable for cond pass
                model.transformer_blocks[0].attn2._debug_cross_attn_post = True
            cond_modality = create_video_modality(latents, timestep, positions, prompt_embeds)
            velocity_cond, _ = model(video=cond_modality)
            del cond_modality  # Free modality tensor

            # CFG blend
            velocity = velocity_cond + (config.guidance_scale - 1.0) * (
                velocity_cond - velocity_uncond
            )
            # DEBUG: Print velocity statistics at key steps
            if i in [0, len(sigmas) // 2, len(sigmas) - 2]:
                print(f"[DEBUG] Step {i}: sigma={sigma:.4f}")
                print(f"  velocity_cond: mean={velocity_cond.mean():.4f}, std={velocity_cond.std():.4f}")
                print(f"  velocity_uncond: mean={velocity_uncond.mean():.4f}, std={velocity_uncond.std():.4f}")
                print(f"  velocity_guided: mean={velocity.mean():.4f}, std={velocity.std():.4f}, range=[{velocity.min():.4f}, {velocity.max():.4f}]")
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

        # Progress callback
        if callback is not None:
            callback(i + 1, len(sigmas) - 1, latents)

    # =========================================================================
    # DEBUG: Post-denoising latent statistics
    # =========================================================================
    print(f"[DEBUG] Post-denoising latents:")
    print(f"  Shape: {latents.shape}")
    print(f"  Mean: {latents.mean():.4f}, Std: {latents.std():.4f}")
    print(f"  Range: [{latents.min():.4f}, {latents.max():.4f}]")
    per_channel_mean = latents.mean(dim=(0, 1))
    print(f"  Per-channel mean range: [{per_channel_mean.min():.4f}, {per_channel_mean.max():.4f}]")
    # Find which channels have extreme means
    extreme_mask = (per_channel_mean.abs() > 1.0)
    if extreme_mask.any():
        extreme_idx = extreme_mask.nonzero(as_tuple=True)[0]
        print(f"  Channels with |mean| > 1.0: {extreme_idx[:10].tolist()}... (showing first 10)")

    # =========================================================================
    # Step 5: Reshape latents for VAE decode
    # From [B, T, D] to [B, D, T_lat, H_lat, W_lat]
    # =========================================================================
    latents = latents.transpose(1, 2)  # [B, D, T]
    latents = latents.reshape(1, 128, t_latent, h_latent, w_latent)
    print(f"[DEBUG] After reshape: shape={latents.shape}")

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

    # =========================================================================
    # DEBUG: VAE output statistics
    # =========================================================================
    print(f"[DEBUG] VAE output (pre-conversion):")
    print(f"  Shape: {video.shape}")
    print(f"  Mean: {video.mean():.4f}, Std: {video.std():.4f}")
    print(f"  Range: [{video.min():.4f}, {video.max():.4f}]")
    print(f"  Expected range: [-1, 1] for proper color conversion")

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
    precision: str = "fp8-native",  # Changed default: native FP8 has no memory leak
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
        quantize: If True, quantize transformer for memory efficiency
        precision: Quantization method:
            - "fp8-native" (default): Official LTX-2 approach, no memory leak
            - "fp8-quanto": Quanto FP8 (legacy, has memory leak issue)
            - "int8-quanto": Quanto INT8
            - "int4-quanto": Quanto INT4 (lowest quality)
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
        if precision == "fp8-native":
            # Native FP8: official LTX-2 approach with no memory leak
            from llm_dit.models.ltx2 import load_ltx2_transformer_fp8_native

            model = load_ltx2_transformer_fp8_native(
                model_path / "transformer",
                dtype=dtype,
                device="cuda",  # Load directly to GPU
                video_only=True,
                verbose=True,
            )
        else:
            # Legacy quanto quantization (fp8-quanto, int8-quanto, int4-quanto)
            from llm_dit.models.ltx2 import load_ltx2_transformer_quantized

            model = load_ltx2_transformer_quantized(
                model_path / "transformer",
                precision=precision,  # type: ignore[arg-type]
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

    # Only load connectors if embeddings need processing (188160 -> 3840 projection)
    # Our Gemma3Encoder already outputs 3840-dim via internal Embeddings1DConnector
    embed_dim = prompt_embeds.shape[-1]
    print(f"[DEBUG] Prompt embeddings shape: {prompt_embeds.shape}")
    print(f"[DEBUG] Prompt embeddings stats: mean={prompt_embeds.mean():.4f}, std={prompt_embeds.std():.4f}")
    print(f"[DEBUG] Prompt embeddings range: [{prompt_embeds.min():.4f}, {prompt_embeds.max():.4f}]")
    connectors = None
    if embed_dim == 188160:
        from llm_dit.models.ltx2.connectors import load_ltx2_connectors

        connectors_path = model_path / "connectors"
        connectors = load_ltx2_connectors(
            connectors_path,
            device="cuda",
            dtype=dtype,
        )
        logger.info("Loaded connectors for 188160 -> 3840 projection")
    else:
        logger.info(f"Skipping connectors (embed_dim={embed_dim} already projected)")

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

    # Pure PyTorch VAE decoder (no diffusers dependency)
    from llm_dit.models.ltx2.vae import load_ltx2_vae_decoder

    vae = load_ltx2_vae_decoder(
        model_path / "vae",
        dtype=dtype,
        device="cpu",
    ).to("cuda")

    logger.info("Decoding latents to video...")

    # DEBUG: Latents received from generate_video (already reshaped)
    print(f"[DEBUG] Latents for VAE decode:")
    print(f"  Shape: {latents.shape}")
    print(f"  Mean: {latents.mean():.4f}, Std: {latents.std():.4f}")
    print(f"  Range: [{latents.min():.4f}, {latents.max():.4f}]")

    # Decode latents to video
    # Note: our VideoDecoder handles denormalization internally via per_channel_statistics
    with torch.no_grad():
        video = vae(latents)

    # DEBUG: VAE output statistics
    print(f"[DEBUG] VAE output (pre-conversion):")
    print(f"  Shape: {video.shape}")
    print(f"  Mean: {video.mean():.4f}, Std: {video.std():.4f}")
    print(f"  Range: [{video.min():.4f}, {video.max():.4f}]")
    print(f"  Expected range: [-1, 1] for proper color conversion")

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
