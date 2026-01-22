"""
Latent conditioning for video generation.

Last Updated: 2026-01-18

Provides VideoConditionByLatentIndex which conditions video generation
by REPLACING tokens at a specific latent frame index. This is used for:
- Image-to-Video (I2V): Replace first frame with encoded image
- Video editing: Replace specific frames with edited content

The replacement tokens modify the existing sequence without changing length:
- Tokens at the specified frame index are replaced
- denoise_mask set to (1.0 - strength) for replaced tokens
- clean_latent stores the replacement for output blending
"""

from __future__ import annotations

import torch

from .exceptions import ConditioningError
from .types import LatentState
from .utils import patchify_latent


class VideoConditionByLatentIndex:
    """
    Conditions video generation by injecting latents at a specific frame index.

    This conditioning type REPLACES tokens at the specified latent frame index.
    The replacement does not change sequence length - it modifies tokens in place.
    This is the primary mechanism for Image-to-Video (I2V) generation.

    Use cases:
    - Image-to-Video: Set latent_idx=0 to use image as first frame
    - Last frame conditioning: Set latent_idx to last frame position
    - Interpolation: Condition both first and last frames

    Args:
        latent: [B, C, 1, H, W] encoded conditioning latent from VAE
            Must have spatial dimensions matching the target latent
        latent_idx: Latent frame index to replace (0 for first frame)
        strength: Conditioning strength (0.0 = no effect, 1.0 = full conditioning)
            - denoise_mask for replaced tokens = 1.0 - strength
            - strength=1.0 means mask=0.0 (preserve conditioning exactly)

    Example:
        >>> image_latent = vae.encode(image)  # [1, 128, 1, 4, 4]
        >>> cond = VideoConditionByLatentIndex(image_latent, latent_idx=0, strength=1.0)
        >>> new_state = cond.apply_to(state)
        >>> # First frame tokens now have mask=0.0 (will be preserved)
    """

    def __init__(
        self,
        latent: torch.Tensor,
        latent_idx: int,
        strength: float,
    ):
        self.latent = latent
        self.latent_idx = latent_idx
        self.strength = strength

    def apply_to(self, latent_state: LatentState) -> LatentState:
        """
        Apply latent conditioning by replacing tokens at frame index.

        Args:
            latent_state: Current latent state to condition

        Returns:
            New state with tokens replaced at specified frame

        Raises:
            ConditioningError: If spatial dimensions don't match
        """
        # Validate spatial shape match
        _, _, _, cond_height, cond_width = self.latent.shape
        if cond_height != latent_state.latent_height or cond_width != latent_state.latent_width:
            raise ConditioningError(
                f"Conditioning latent spatial shape ({cond_height}, {cond_width}) does not match "
                f"target latent shape ({latent_state.latent_height}, {latent_state.latent_width}). "
                "Ensure the conditioning image has been resized to match the target video dimensions."
            )

        # Convert conditioning latent to tokens: [B, C, F, H, W] -> [B, T, C]
        tokens = patchify_latent(self.latent)
        num_tokens_to_replace = tokens.shape[1]

        # Calculate token indices for the target frame
        tokens_per_frame = latent_state.tokens_per_frame
        start_token = self.latent_idx * tokens_per_frame
        end_token = start_token + num_tokens_to_replace

        # Clone state to avoid modifying original
        new_state = latent_state.clone()

        # Replace tokens at the specified frame
        new_state.latent[:, start_token:end_token] = tokens

        # Update denoise_mask for replaced tokens
        new_state.denoise_mask[:, start_token:end_token] = 1.0 - self.strength

        # Set up clean_latent for blending
        if new_state.clean_latent is None:
            clean_latent = torch.zeros_like(new_state.latent)
        else:
            clean_latent = new_state.clean_latent
        clean_latent[:, start_token:end_token] = tokens

        return LatentState(
            latent=new_state.latent,
            denoise_mask=new_state.denoise_mask,
            positions=new_state.positions,
            clean_latent=clean_latent,
            _latent_height=new_state._latent_height,
            _latent_width=new_state._latent_width,
            _num_frames=new_state._num_frames,
        )
