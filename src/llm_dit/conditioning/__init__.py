"""
LTX-2 Conditioning System.

Last Updated: 2026-01-18

Provides conditioning support for LTX-2 video generation:
- Image-to-Video (I2V) via VideoConditionByLatentIndex
- Video continuation via VideoConditionByKeyframeIndex
- Per-token denoising strength via LatentState.denoise_mask

The conditioning system works by:
1. Creating a LatentState with default denoise_mask = 1.0 (full denoising)
2. Applying conditioning items that modify tokens and their masks
3. During denoising, timesteps are scaled by mask (conditioned regions get lower timesteps)
4. After denoising, output is blended with clean_latent based on mask

Example usage:
    >>> from llm_dit.conditioning import (
    ...     LatentState,
    ...     VideoConditionByLatentIndex,
    ...     timesteps_from_mask,
    ...     post_process_latent,
    ... )
    >>>
    >>> # Create initial state
    >>> state = LatentState.create(
    ...     shape=(1, 256, 128),
    ...     num_frames=33,
    ...     height=512,
    ...     width=768,
    ...     device="cuda",
    ...     dtype=torch.bfloat16,
    ... )
    >>>
    >>> # Apply I2V conditioning (first frame from image)
    >>> image_latent = vae.encode(image)  # [1, 128, 1, 16, 24]
    >>> cond = VideoConditionByLatentIndex(image_latent, latent_idx=0, strength=1.0)
    >>> state = cond.apply_to(state)
    >>>
    >>> # Add noise respecting the mask
    >>> state = state.add_noise(generator=generator)
    >>>
    >>> # During denoising loop:
    >>> timesteps = timesteps_from_mask(state.denoise_mask, sigma)  # Per-token!
    >>> # ... model forward pass ...
    >>> denoised = post_process_latent(output, state.denoise_mask, state.clean_latent)
"""

from .exceptions import ConditioningError
from .types import ConditioningItem, LatentState
from .keyframe import VideoConditionByKeyframeIndex
from .latent import VideoConditionByLatentIndex
from .utils import (
    timesteps_from_mask,
    post_process_latent,
    patchify_latent,
    unpatchify_latent,
    create_position_bounds,
    get_tokens_per_frame,
    validate_spatial_match,
)

__all__ = [
    # Core types
    "LatentState",
    "ConditioningItem",
    # Conditioning classes
    "VideoConditionByKeyframeIndex",
    "VideoConditionByLatentIndex",
    # Exceptions
    "ConditioningError",
    # Utility functions
    "timesteps_from_mask",
    "post_process_latent",
    "patchify_latent",
    "unpatchify_latent",
    "create_position_bounds",
    "get_tokens_per_frame",
    "validate_spatial_match",
]
