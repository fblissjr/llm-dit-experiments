"""
Conditioning system core types.

Last Updated: 2026-01-18

Provides the LatentState dataclass for managing latent state during
diffusion denoising with conditioning support.

The LatentState differs from the Modality dataclass in that:
- Modality is frozen and represents transformer input (unchanged)
- LatentState is mutable and manages conditioning state before/during denoising

Key concepts:
- denoise_mask: Per-token denoising strength [B, T, 1] where:
  - 1.0 = full denoising (unconditioned)
  - 0.0 = no denoising (fully conditioned, preserved from clean_latent)
- clean_latent: Reference latent for blending with conditioned regions
- positions: Position bounds [B, 3, T, 2] for RoPE embeddings
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Protocol, TYPE_CHECKING

import torch

from .utils import create_position_bounds

if TYPE_CHECKING:
    pass


class ConditioningItem(Protocol):
    """Protocol for conditioning items that modify latent state during diffusion."""

    def apply_to(self, latent_state: "LatentState") -> "LatentState":
        """
        Apply the conditioning to the latent state.

        Args:
            latent_state: The latent state to apply conditioning to.
                This state contains patchified (token) format latents.

        Returns:
            New LatentState after conditioning has been applied.

        Note:
            If the conditioning adds extra tokens (e.g., keyframes),
            they should be appended to the end of the sequence.
        """
        ...


@dataclass
class LatentState:
    """
    State of latents during the diffusion denoising process.

    This class manages the mutable state needed for conditioning during
    diffusion. It tracks:
    - The current noisy latent being denoised
    - Per-token denoising strength via denoise_mask
    - Position information for RoPE embeddings
    - Clean reference latent for blending conditioned regions

    Attributes:
        latent: [B, T, D] current noisy latent tensor being denoised
        denoise_mask: [B, T, 1] mask encoding denoising strength per token
            - 1.0 = full denoising
            - 0.0 = no denoising (preserve from clean_latent)
        positions: [B, 3, T, 2] positional indices with bounds for RoPE
        clean_latent: Optional [B, T, D] reference latent for blending.
            Set when conditioning is applied, used for output blending.
        _latent_height: Internal tracking of latent spatial height
        _latent_width: Internal tracking of latent spatial width
        _num_frames: Internal tracking of number of latent frames
    """

    latent: torch.Tensor
    denoise_mask: torch.Tensor
    positions: torch.Tensor
    clean_latent: Optional[torch.Tensor]
    _latent_height: int
    _latent_width: int
    _num_frames: int

    @classmethod
    def create(
        cls,
        shape: tuple[int, int, int],
        device: str | torch.device,
        dtype: torch.dtype,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        fps: float = 24.0,
    ) -> "LatentState":
        """
        Create a new LatentState with default values.

        Args:
            shape: (batch_size, num_tokens, latent_dim) for the latent tensor
            device: Target device
            dtype: Target dtype
            num_frames: Number of video frames (for position calculation)
            height: Video height in pixels (for position calculation)
            width: Video width in pixels (for position calculation)
            fps: Frames per second for temporal position scaling

        Returns:
            New LatentState initialized with:
            - Zero latent tensor
            - denoise_mask = 1.0 everywhere (full denoising)
            - positions calculated from dimensions
            - clean_latent = None
        """
        batch_size, num_tokens, latent_dim = shape
        device = torch.device(device)

        # Initialize latent to zeros (will be filled with noise later)
        latent = torch.zeros(batch_size, num_tokens, latent_dim, device=device, dtype=dtype)

        # Initialize denoise_mask to 1.0 (full denoising everywhere)
        denoise_mask = torch.ones(batch_size, num_tokens, 1, device=device, dtype=dtype)

        # Calculate latent dimensions from token count if not provided
        # LTX-2: 32x spatial, 8x temporal downsampling
        if num_frames is not None and height is not None and width is not None:
            t_latent = (num_frames - 1) // 8 + 1
            h_latent = height // 32
            w_latent = width // 32
        else:
            # Infer from token count (assume square spatial)
            # This is a fallback - prefer passing explicit dimensions
            t_latent = 1
            remaining = num_tokens
            # Try to find factors
            h_latent = int(remaining ** 0.5)
            while remaining % h_latent != 0 and h_latent > 1:
                h_latent -= 1
            w_latent = remaining // h_latent
            # Check if we need more frames
            if h_latent * w_latent != num_tokens:
                # Try with multiple frames
                for t in range(1, num_tokens + 1):
                    if num_tokens % t == 0:
                        spatial = num_tokens // t
                        h = int(spatial ** 0.5)
                        while spatial % h != 0 and h > 1:
                            h -= 1
                        w = spatial // h
                        if h * w * t == num_tokens:
                            t_latent, h_latent, w_latent = t, h, w
                            break

        # Create position bounds
        positions = create_position_bounds(
            frames=t_latent,
            height=h_latent,
            width=w_latent,
            device=device,
            fps=fps,
        )

        # Expand positions for batch
        positions = positions.expand(batch_size, -1, -1, -1).clone()

        return cls(
            latent=latent,
            denoise_mask=denoise_mask,
            positions=positions,
            clean_latent=None,
            _latent_height=h_latent,
            _latent_width=w_latent,
            _num_frames=t_latent,
        )

    def clone(self) -> "LatentState":
        """Create a deep copy of this state."""
        return LatentState(
            latent=self.latent.clone(),
            denoise_mask=self.denoise_mask.clone(),
            positions=self.positions.clone(),
            clean_latent=self.clean_latent.clone() if self.clean_latent is not None else None,
            _latent_height=self._latent_height,
            _latent_width=self._latent_width,
            _num_frames=self._num_frames,
        )

    def with_clean_latent(self, clean_latent: torch.Tensor) -> "LatentState":
        """Return new state with clean_latent set."""
        return replace(self, clean_latent=clean_latent)

    def add_noise(
        self,
        generator: Optional[torch.Generator] = None,
        noise_scale: float = 1.0,
    ) -> "LatentState":
        """
        Add noise to the latent, scaled by denoise_mask.

        The formula is:
            new_latent = noise * (mask * scale) + latent * (1 - mask * scale)

        This means:
        - mask=1.0, scale=1.0: pure noise
        - mask=0.0: original latent preserved
        - mask=0.5: blend of noise and latent

        Args:
            generator: Optional random generator for reproducibility
            noise_scale: Overall scaling factor for noise (typically 1.0)

        Returns:
            New state with noisy latent
        """
        noise = torch.randn(
            *self.latent.shape,
            device=self.latent.device,
            dtype=self.latent.dtype,
            generator=generator,
        )

        scaled_mask = self.denoise_mask * noise_scale

        # If we have a clean_latent, blend with it; otherwise blend with zeros
        reference = self.clean_latent if self.clean_latent is not None else torch.zeros_like(self.latent)

        new_latent = noise * scaled_mask + reference * (1 - scaled_mask)
        new_latent = new_latent.to(self.latent.dtype)

        return replace(self, latent=new_latent)

    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self.latent.shape[0]

    @property
    def num_tokens(self) -> int:
        """Get number of tokens."""
        return self.latent.shape[1]

    @property
    def latent_dim(self) -> int:
        """Get latent dimension."""
        return self.latent.shape[2]

    @property
    def device(self) -> torch.device:
        """Get device."""
        return self.latent.device

    @property
    def dtype(self) -> torch.dtype:
        """Get dtype."""
        return self.latent.dtype

    @property
    def latent_height(self) -> int:
        """Get latent height."""
        return self._latent_height

    @property
    def latent_width(self) -> int:
        """Get latent width."""
        return self._latent_width

    @property
    def num_latent_frames(self) -> int:
        """Get number of latent frames."""
        return self._num_frames

    @property
    def tokens_per_frame(self) -> int:
        """Get tokens per frame."""
        return self._latent_height * self._latent_width
