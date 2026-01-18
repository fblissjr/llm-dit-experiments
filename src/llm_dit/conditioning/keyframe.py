"""
Keyframe conditioning for video generation.

Last Updated: 2026-01-18

Provides VideoConditionByKeyframeIndex which conditions video generation
by APPENDING keyframe latents to the token sequence. This is used for:
- Video continuation (extend a video with new frames)
- Animation guidance (guide generation with specific keyframes)

The keyframe tokens are appended to the end of the sequence with:
- Positions offset by frame_idx for temporal alignment
- denoise_mask set to (1.0 - strength) for blending control
"""

from __future__ import annotations

from dataclasses import replace

import torch

from .types import LatentState
from .utils import patchify_latent, create_position_bounds


class VideoConditionByKeyframeIndex:
    """
    Conditions video generation on keyframe latents at a specific frame index.

    This conditioning type APPENDS keyframe tokens to the latent state.
    The appended tokens have positions offset by frame_idx for temporal
    alignment, and denoise strength controlled by the strength parameter.

    Use cases:
    - Video continuation: Set frame_idx to where continuation should start
    - Animation keyframes: Insert guidance frames at specific positions

    Args:
        keyframes: [B, C, F, H, W] encoded keyframe latents from VAE
        frame_idx: Frame index for temporal position offset
        strength: Conditioning strength (0.0 = no effect, 1.0 = full conditioning)
            - denoise_mask for appended tokens = 1.0 - strength
            - strength=0.8 means mask=0.2 (mostly preserve keyframe)

    Example:
        >>> keyframe = vae.encode(image)  # [1, 128, 1, 4, 4]
        >>> cond = VideoConditionByKeyframeIndex(keyframe, frame_idx=10, strength=0.9)
        >>> new_state = cond.apply_to(state)
        >>> # new_state.latent now has appended keyframe tokens
    """

    def __init__(
        self,
        keyframes: torch.Tensor,
        frame_idx: int,
        strength: float,
    ):
        self.keyframes = keyframes
        self.frame_idx = frame_idx
        self.strength = strength

    def apply_to(self, latent_state: LatentState) -> LatentState:
        """
        Apply keyframe conditioning by appending tokens.

        Args:
            latent_state: Current latent state to condition

        Returns:
            New state with keyframe tokens appended
        """
        # Get keyframe dimensions
        batch, channels, frames, height, width = self.keyframes.shape

        # Convert keyframe latent to tokens: [B, C, F, H, W] -> [B, T, C]
        tokens = patchify_latent(self.keyframes)
        num_new_tokens = tokens.shape[1]

        # Create positions for keyframe tokens with frame_idx offset
        new_positions = create_position_bounds(
            frames=frames,
            height=height,
            width=width,
            device=self.keyframes.device,
            frame_offset=self.frame_idx,
        )
        # Expand for batch
        new_positions = new_positions.expand(batch, -1, -1, -1)

        # Create denoise_mask for appended tokens: mask = 1.0 - strength
        new_denoise_mask = torch.full(
            (batch, num_new_tokens, 1),
            fill_value=1.0 - self.strength,
            device=self.keyframes.device,
            dtype=self.keyframes.dtype,
        )

        # Concatenate with existing state
        new_latent = torch.cat([latent_state.latent, tokens], dim=1)
        new_mask = torch.cat([latent_state.denoise_mask, new_denoise_mask], dim=1)
        new_pos = torch.cat([latent_state.positions, new_positions], dim=2)

        # Clean latent: original + keyframe tokens
        if latent_state.clean_latent is not None:
            new_clean = torch.cat([latent_state.clean_latent, tokens], dim=1)
        else:
            # Initialize clean_latent with zeros for original tokens + keyframes
            original_clean = torch.zeros_like(latent_state.latent)
            new_clean = torch.cat([original_clean, tokens], dim=1)

        return replace(
            latent_state,
            latent=new_latent,
            denoise_mask=new_mask,
            positions=new_pos,
            clean_latent=new_clean,
        )
