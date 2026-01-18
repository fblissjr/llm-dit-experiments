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

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from tqdm import tqdm

from llm_dit.models.ltx2 import (
    LTX2Transformer,
    Modality,
    LTX2TextConnectors,
    VideoDecoder,
)
from llm_dit.schedulers import LTX2Scheduler

logger = logging.getLogger(__name__)


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
) -> torch.Tensor:
    """
    Create 3D position indices [B, 3, T] for video (temporal, height, width).

    LTX-2 VAE compression: 32x spatial, 8x temporal.
    Reference: coderef/LTX-2/ltx-core/components/patchifiers.py

    Args:
        batch_size: Batch size
        num_frames: Number of video frames
        height: Video height in pixels
        width: Video width in pixels
        device: Target device

    Returns:
        Position indices tensor [B, 3, T] where T = t_latent * h_latent * w_latent
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
    grid_t, grid_h, grid_w = torch.meshgrid(t_indices, h_indices, w_indices, indexing='ij')

    # Flatten and stack to [3, T]
    positions = torch.stack([
        grid_t.flatten(),
        grid_h.flatten(),
        grid_w.flatten(),
    ], dim=0)

    # Expand for batch: [B, 3, T]
    return positions.unsqueeze(0).expand(batch_size, -1, -1)


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
    callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Generate video using pure PyTorch diffusion loop.

    This is the main generation entry point that handles:
    1. Connector processing (if embeddings are packed format)
    2. Noise initialization
    3. Sigma schedule computation
    4. CFG-guided denoising loop
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
        callback: Optional callback(step, total_steps, latents) for progress
        device: Override device (default: model device)
        dtype: Override dtype (default: model dtype)

    Returns:
        If vae provided: Video tensor [B, C, F, H, W] or [F, H, W, C] uint8
        If vae=None: Latent tensor [B, D, T_lat, H_lat, W_lat]
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
                prompt_embeds.shape[0], prompt_embeds.shape[1],
                device=prompt_embeds.device, dtype=prompt_embeds.dtype
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
    # Step 1: Initialize noise [B, T, D] where D=128 (VAE latent channels)
    # =========================================================================
    latents = torch.randn(
        (1, num_tokens, 128),
        generator=generator,
        device=device,
        dtype=dtype,
    )

    # =========================================================================
    # Step 2: Create position indices
    # =========================================================================
    positions = create_position_indices(
        1, config.num_frames, config.height, config.width, device
    )

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
        timestep = (sigma * 1000).expand(1, num_tokens)

        # Classifier-free guidance
        if config.guidance_scale > 1.0:
            # Unconditional pass (zero embeddings)
            uncond_embeds = torch.zeros_like(prompt_embeds)
            uncond_modality = create_video_modality(
                latents, timestep, positions, uncond_embeds
            )
            velocity_uncond, _ = model(video=uncond_modality)

            # Conditional pass
            cond_modality = create_video_modality(
                latents, timestep, positions, prompt_embeds
            )
            velocity_cond, _ = model(video=cond_modality)

            # CFG blend
            velocity = velocity_cond + (config.guidance_scale - 1.0) * (
                velocity_cond - velocity_uncond
            )
        else:
            modality = create_video_modality(
                latents, timestep, positions, prompt_embeds
            )
            velocity, _ = model(video=modality)

        # Euler step: x_{t-1} = x_t + v * dt
        dt = sigma_next - sigma
        latents = latents + velocity * dt

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
    if hasattr(vae, 'latents_mean') and vae.latents_mean is not None:
        # Diffusers VAE - use its built-in normalization params
        latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(device, dtype)
        latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(device, dtype)
        scaling_factor = getattr(vae.config, 'scaling_factor', 1.0)
        latents = latents * latents_std / scaling_factor + latents_mean
    elif config.latents_mean is not None and config.latents_std is not None:
        # Our VAE - use config normalization params
        latents_mean = config.latents_mean.view(1, -1, 1, 1, 1).to(device, dtype)
        latents_std = config.latents_std.view(1, -1, 1, 1, 1).to(device, dtype)
        latents = latents * latents_std / config.scaling_factor + latents_mean

    # Decode to pixel space
    # Support both diffusers VAE (decode method) and our VAE (direct call)
    with torch.no_grad():
        if hasattr(vae, 'decode'):
            # Diffusers VAE interface
            video = vae.decode(latents).sample
        else:
            # Our VideoDecoder - direct call
            video = vae(latents)

    # Convert to [F, H, W, C] uint8 format
    video = video.squeeze(0).permute(1, 2, 3, 0)  # [B, C, T, H, W] -> [T, H, W, C]
    video = ((video + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)

    return video
