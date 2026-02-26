"""
LTX-2 Audio VAE Type Definitions.

Last Updated: 2026-02-26

Type definitions for audio latent shapes, analogous to VideoLatentShape
in the video VAE.

Ported from: DiffSynth-Studio ltx2_common.AudioLatentShape
License: LTX-2 Community License
Copyright (c) 2025 Lightricks Ltd.
"""

from dataclasses import dataclass

import torch


# VAE temporal downsampling factor: mel frames -> latent frames
AUDIO_LATENT_DOWNSAMPLE_FACTOR = 4


@dataclass(frozen=True)
class AudioLatentShape:
    """Shape descriptor for audio latent tensors.

    Audio latents have shape (batch, channels, frames, mel_bins) where:
        - batch: batch size
        - channels: latent channels (8 for LTX-2 audio)
        - frames: temporal frames (mel frames / AUDIO_LATENT_DOWNSAMPLE_FACTOR)
        - mel_bins: frequency bins (16 for LTX-2 audio, after VAE spatial compression)
    """
    batch: int
    channels: int
    frames: int
    mel_bins: int

    def to_torch_shape(self) -> tuple[int, int, int, int]:
        """Return (B, C, T, F) tuple for torch tensor creation."""
        return (self.batch, self.channels, self.frames, self.mel_bins)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "AudioLatentShape":
        """Create from a 4D tensor (B, C, T, F)."""
        if tensor.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {tensor.dim()}D")
        return cls(
            batch=tensor.shape[0],
            channels=tensor.shape[1],
            frames=tensor.shape[2],
            mel_bins=tensor.shape[3],
        )
