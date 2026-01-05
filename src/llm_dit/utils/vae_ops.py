"""VAE operations and differential diffusion utilities.

This module provides shared VAE encoding/decoding helpers and differential
diffusion mask preparation used across all pipelines.

Last updated: 2025-01-05
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np


def encode_image_to_latents(
    vae,
    image: Union[Image.Image, torch.Tensor],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Encode a PIL Image or tensor to VAE latent space.

    Args:
        vae: VAE model (AutoencoderKL or similar)
        image: PIL Image or tensor in [0, 1] range with shape (B, C, H, W) or (C, H, W)
        dtype: Target dtype (default: VAE's dtype)
        device: Target device (default: VAE's device)

    Returns:
        Latent tensor with shape (B, C, H//scale, W//scale)
    """
    # Get the underlying VAE (unwrap TiledVAEDecoder if needed)
    underlying_vae = vae.vae if hasattr(vae, 'vae') else vae

    # Determine device and dtype
    if device is None:
        device = next(underlying_vae.parameters()).device
    if dtype is None:
        dtype = next(underlying_vae.parameters()).dtype

    # Convert PIL to tensor
    if isinstance(image, Image.Image):
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
    else:
        image_tensor = image

    # Ensure correct shape
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    # Move to correct device and dtype
    image_tensor = image_tensor.to(device=device, dtype=dtype)

    # Normalize from [0, 1] to [-1, 1]
    image_tensor = 2.0 * image_tensor - 1.0

    # Encode
    with torch.no_grad():
        latent_dist = underlying_vae.encode(image_tensor)
        if hasattr(latent_dist, 'latent_dist'):
            latents = latent_dist.latent_dist.sample()
        elif hasattr(latent_dist, 'sample'):
            latents = latent_dist.sample()
        else:
            latents = latent_dist

    # Apply VAE scaling if config exists
    if hasattr(underlying_vae, 'config'):
        shift = getattr(underlying_vae.config, 'shift_factor', 0.0)
        scale = getattr(underlying_vae.config, 'scaling_factor', 1.0)
        latents = (latents - shift) * scale

    return latents


def prepare_differential_masks(
    mask_image: Union[Image.Image, torch.Tensor],
    num_inference_steps: int,
    latent_size: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    vae_scale_factor: int = 8,
) -> torch.Tensor:
    """Prepare time-dependent masks for differential diffusion.

    Differential diffusion allows per-pixel control over edit strength using
    a grayscale mask. This function converts the mask into time-dependent
    binary masks for each denoising step.

    The algorithm:
    1. Normalize mask to [0, 1] where 0 = preserve, 1 = fully edit
    2. Create thresholds for each timestep: [0/T, 1/T, 2/T, ..., (T-1)/T]
    3. For each step, create binary mask: mask > threshold[step]
       - At step 0: threshold=0, most pixels pass -> most change
       - At step T-1: threshold~=1, few pixels pass -> little change

    Args:
        mask_image: Grayscale mask (PIL Image or tensor)
            - Values 0 (black) = preserve original
            - Values 1 (white) = allow full editing
            - Intermediate values = partial editing
        num_inference_steps: Number of denoising steps
        latent_size: Target latent dimensions (height, width)
        device: Target device
        dtype: Target dtype for the output masks
        vae_scale_factor: VAE downscaling factor (default: 8)

    Returns:
        Boolean mask tensor with shape (num_steps, 1, latent_height, latent_width)
        where True means "preserve original" and False means "use denoised"

    Example:
        >>> mask = Image.open("mask.png").convert("L")  # Grayscale
        >>> masks = prepare_differential_masks(mask, 9, (128, 128), device)
        >>> # In denoising loop:
        >>> for i, t in enumerate(timesteps):
        ...     latents = denoise_step(latents, t)
        ...     if i < len(timesteps) - 1:
        ...         noised_orig = add_noise(init_latents, timesteps[i+1])
        ...         latents = noised_orig * masks[i] + latents * (~masks[i])
    """
    latent_height, latent_width = latent_size

    # Convert PIL Image to tensor
    if isinstance(mask_image, Image.Image):
        # Ensure grayscale
        mask_image = mask_image.convert("L")
        mask_array = np.array(mask_image).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    else:
        mask = mask_image.float()
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)

    # Resize to latent resolution
    mask = F.interpolate(
        mask,
        size=(latent_height, latent_width),
        mode="bilinear",
        align_corners=False,
    )

    mask = mask.to(device=device, dtype=dtype)

    # Create time-dependent thresholds
    # At step 0, threshold = 0 (all pixels above 0 will be edited)
    # At step T-1, threshold = (T-1)/T (only very bright pixels edited)
    thresholds = torch.arange(num_inference_steps, dtype=dtype, device=device) / num_inference_steps
    thresholds = thresholds.reshape(-1, 1, 1, 1)  # (T, 1, 1, 1)

    # Create boolean masks: True where we should PRESERVE original
    # Note: mask > threshold means "this pixel wants more change than available at this step"
    # So ~(mask > threshold) = preserve original
    # mask <= threshold means "preserve" (black areas where mask=0 are always <= threshold)
    masks = mask <= thresholds  # (T, 1, H, W) - True = preserve, False = edit

    return masks


def blend_differential_latents(
    latents: torch.Tensor,
    original_latents: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Blend denoised latents with original using differential mask.

    Args:
        latents: Denoised latents from current step
        original_latents: Original latents (noised to appropriate level)
        mask: Binary mask where True = preserve original, False = use denoised

    Returns:
        Blended latents
    """
    # Cast mask to latent dtype for blending
    mask_float = mask.to(dtype=latents.dtype)
    return original_latents * mask_float + latents * (1 - mask_float)


def scale_noise_for_timestep(
    scheduler,
    original_latents: torch.Tensor,
    timestep: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Re-noise original latents to a specific timestep level.

    This is used in differential diffusion to blend with denoised latents.

    Args:
        scheduler: Flow matching scheduler
        original_latents: Clean latents from VAE encoding
        timestep: Target timestep (from scheduler.timesteps)
        noise: Same noise used for initial noising

    Returns:
        Noised latents at the specified timestep level
    """
    if hasattr(scheduler, 'scale_noise'):
        # diffusers scheduler expects timestep as iterable, not scalar
        # Wrap scalar timesteps in a list
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        return scheduler.scale_noise(original_latents, timestep, noise)
    elif hasattr(scheduler, 'sigmas'):
        # Our custom scheduler - manual flow matching noise addition
        # Find sigma for this timestep
        # For flow matching: noised = (1 - sigma) * clean + sigma * noise
        step_idx = (scheduler.timesteps == timestep).nonzero()
        if len(step_idx) > 0:
            idx = step_idx[0].item()
            sigma = scheduler.sigmas[idx]
        else:
            # Fallback: estimate sigma from timestep
            sigma = 1.0 - (1000 - timestep.float()) / 1000
        sigma = sigma.to(device=original_latents.device, dtype=original_latents.dtype)
        return (1 - sigma) * original_latents + sigma * noise
    else:
        raise ValueError("Scheduler must have scale_noise method or sigmas attribute")


def get_vae_scale_factor(vae) -> int:
    """Get the VAE spatial downscaling factor.

    Args:
        vae: VAE model

    Returns:
        Scale factor (typically 8 or 16)
    """
    # Unwrap TiledVAEDecoder if needed
    underlying_vae = vae.vae if hasattr(vae, 'vae') else vae

    # Check config
    if hasattr(underlying_vae, 'config'):
        if hasattr(underlying_vae.config, 'scaling_factor'):
            # This is different from spatial scaling
            pass
        if hasattr(underlying_vae.config, 'block_out_channels'):
            # Estimate from architecture
            return 2 ** (len(underlying_vae.config.block_out_channels) - 1)

    # Default fallback
    return 8
