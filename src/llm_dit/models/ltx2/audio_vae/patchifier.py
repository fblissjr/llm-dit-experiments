"""
LTX-2 Audio Patchifier - Audio Latent Patchify/Unpatchify.

Last Updated: 2026-02-26

Handles conversion between 4D audio latent tensors (B, C, T, F) and
flattened patch sequences (B, T, C*F) for transformer input. Also computes
temporal position bounds aligned with real-time audio timestamps.

Ported from: DiffSynth-Studio ltx2_audio_vae.AudioPatchifier
License: LTX-2 Community License
Copyright (c) 2025 Lightricks Ltd.
"""

from typing import Optional, Tuple

import einops
import torch

from .types import AudioLatentShape, AUDIO_LATENT_DOWNSAMPLE_FACTOR


class AudioPatchifier:
    """Patchifier for audio latent tensors.

    Audio patchification is simpler than video: it flattens the channel and
    frequency dimensions into a single patch dimension while keeping the
    temporal dimension intact. This produces (B, T, C*F) tokens where each
    token represents one temporal frame across all channels and frequencies.

    The patchifier also computes temporal position bounds that map each latent
    frame back to real-time seconds, accounting for hop length, downsample
    factor, and causal alignment offsets.

    Args:
        patch_size: Unused for audio (kept for API compatibility). Audio patches
            are always 1D temporal.
        sample_rate: Audio sample rate in Hz (default 16000).
        hop_length: Mel spectrogram hop length in samples (default 160).
        audio_latent_downsample_factor: Ratio between mel frames and latent
            frames from VAE compression (default 4).
        is_causal: Whether to apply causal timing offset (default True).
        shift: Integer offset for latent indices, for overlapping windows.
    """

    def __init__(
        self,
        patch_size: int = 1,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = AUDIO_LATENT_DOWNSAMPLE_FACTOR,
        is_causal: bool = True,
        shift: int = 0,
    ):
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.audio_latent_downsample_factor = audio_latent_downsample_factor
        self.is_causal = is_causal
        self.shift = shift
        self._patch_size = (1, patch_size, patch_size)

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        return self._patch_size

    def get_token_count(self, tgt_shape: AudioLatentShape) -> int:
        """Return the number of transformer tokens for a given latent shape."""
        return tgt_shape.frames

    def _get_audio_latent_time_in_sec(
        self,
        start_latent: int,
        end_latent: int,
        dtype: torch.dtype,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Convert latent frame indices to real-time seconds.

        Accounts for hop length, downsample factor, and causal offset to
        produce timestamps aligned with the original waveform.

        Args:
            start_latent: Inclusive start index in latent sequence.
            end_latent: Exclusive end index.
            dtype: Floating-point dtype for the returned tensor.
            device: Target device (defaults to CPU).

        Returns:
            1D tensor of timestamps in seconds, shape (end_latent - start_latent,).
        """
        if device is None:
            device = torch.device("cpu")

        audio_latent_frame = torch.arange(start_latent, end_latent, dtype=dtype, device=device)
        audio_mel_frame = audio_latent_frame * self.audio_latent_downsample_factor

        if self.is_causal:
            # Frame offset for causal alignment -- ensures timestamp corresponds
            # to the first sample that is fully available
            causal_offset = 1
            audio_mel_frame = (
                audio_mel_frame + causal_offset - self.audio_latent_downsample_factor
            ).clip(min=0)

        return audio_mel_frame * self.hop_length / self.sample_rate

    def _compute_audio_timings(
        self,
        batch_size: int,
        num_steps: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Build (B, 1, T, 2) tensor of [start, end) timestamps per latent frame.

        Args:
            batch_size: Number of sequences to broadcast over.
            num_steps: Number of latent frames to convert.
            device: Target device for the tensor.

        Returns:
            Tensor of shape (B, 1, T, 2) with start/end timestamps.
        """
        resolved_device = device if device is not None else torch.device("cpu")

        start_timings = self._get_audio_latent_time_in_sec(
            self.shift,
            num_steps + self.shift,
            torch.float32,
            resolved_device,
        )
        start_timings = start_timings.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)

        end_timings = self._get_audio_latent_time_in_sec(
            self.shift + 1,
            num_steps + self.shift + 1,
            torch.float32,
            resolved_device,
        )
        end_timings = end_timings.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)

        return torch.stack([start_timings, end_timings], dim=-1)

    def patchify(self, audio_latents: torch.Tensor) -> torch.Tensor:
        """Flatten audio latent tensor from (B, C, T, F) to (B, T, C*F).

        Each temporal frame becomes a single token containing all channel
        and frequency information.

        Args:
            audio_latents: Tensor of shape (B, C, T, F).

        Returns:
            Flattened tensor of shape (B, T, C*F).
        """
        return einops.rearrange(audio_latents, "b c t f -> b t (c f)")

    def unpatchify(
        self,
        audio_latents: torch.Tensor,
        output_shape: AudioLatentShape,
    ) -> torch.Tensor:
        """Restore (B, C, T, F) from flattened (B, T, C*F) patches.

        Args:
            audio_latents: Flattened tensor of shape (B, T, C*F).
            output_shape: Target shape describing channels and mel_bins.

        Returns:
            Restored tensor of shape (B, C, T, F).
        """
        return einops.rearrange(
            audio_latents,
            "b t (c f) -> b c t f",
            c=output_shape.channels,
            f=output_shape.mel_bins,
        )

    def get_patch_grid_bounds(
        self,
        output_shape: AudioLatentShape,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Return temporal bounds [start, end) for every patch.

        The returned tensor has shape (B, 1, T, 2) where axis 3 stores
        the [start, end) timestamps per patch in seconds.

        Args:
            output_shape: Audio latent shape describing batch and frames.
            device: Target device for the returned tensor.

        Returns:
            Tensor of shape (B, 1, T, 2).
        """
        if not isinstance(output_shape, AudioLatentShape):
            raise ValueError(
                "AudioPatchifier expects AudioLatentShape when computing coordinates"
            )

        return self._compute_audio_timings(output_shape.batch, output_shape.frames, device)
