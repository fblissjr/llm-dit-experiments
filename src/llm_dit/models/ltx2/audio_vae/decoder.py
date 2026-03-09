"""
LTX-2 Audio VAE Decoder.

Last Updated: 2026-02-26

Decodes audio latent representations back to mel spectrograms.
The decoder mirrors the encoder structure: conv_in -> mid_block ->
upsampling path -> norm + conv_out.

Architecture (from audio_vae/config.json):
    - base_channels: 128
    - ch_mult: [1, 2, 4]  (3 resolution levels)
    - num_res_blocks: 2
    - latent_channels: 8
    - norm_type: pixel (PixelNorm)
    - causality_axis: height (causal along time)
    - mid_block_add_attention: false

Data flow:
    Input:  (B, 8, T, 16)   -- normalized audio latents
    Output: (B, 2, T', 64)  -- stereo mel spectrogram

Ported from: DiffSynth-Studio ltx2_audio_vae.LTX2AudioDecoder
License: LTX-2 Community License
Copyright (c) 2025 Lightricks Ltd.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_dit.utils.shuttle import PinnedShuttleMixin

from ..vae.normalization import NormType, build_normalization_layer

from .blocks import (
    AttentionType,
    CausalityAxis,
    PerChannelStatistics,
    build_mid_block,
    build_upsampling_path,
    make_conv2d,
    run_mid_block,
)
from .patchifier import AudioPatchifier
from .types import AudioLatentShape, AUDIO_LATENT_DOWNSAMPLE_FACTOR


class AudioDecoder(PinnedShuttleMixin, nn.Module):
    """Decodes audio latents to mel spectrograms.

    Symmetric decoder that reconstructs audio spectrograms from latent
    features. The architecture uses a series of upsampling blocks with
    residual connections and configurable causal convolutions.

    The decoder first denormalizes the latents using per-channel statistics
    (loaded from checkpoint), then runs them through the upsampling network.

    Args:
        ch: Base channel count (128).
        out_ch: Output channels (2 for stereo).
        ch_mult: Channel multipliers per resolution level.
        num_res_blocks: Residual blocks per level.
        attn_resolutions: Resolutions at which to apply attention.
        resolution: Input spatial resolution.
        z_channels: Latent channel count (8).
        norm_type: Normalization type (PIXEL for audio).
        causality_axis: Causal axis (HEIGHT = time-causal).
        dropout: Dropout rate.
        mid_block_add_attention: Whether mid block has attention.
        sample_rate: Audio sample rate in Hz.
        mel_hop_length: Mel spectrogram hop length.
        is_causal: Whether to use causal timing.
        mel_bins: Number of mel frequency bins in output.
    """

    def __init__(
        self,
        *,
        ch: int = 128,
        out_ch: int = 2,
        ch_mult: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: set[int] | None = None,
        resolution: int = 256,
        z_channels: int = 8,
        norm_type: NormType = NormType.PIXEL,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
        dropout: float = 0.0,
        mid_block_add_attention: bool = False,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        is_causal: bool = True,
        mel_bins: int | None = 64,
    ) -> None:
        nn.Module.__init__(self)
        self._init_shuttle_state()

        if attn_resolutions is None:
            attn_resolutions = set()
        resamp_with_conv = True
        attn_type = AttentionType.VANILLA

        # Per-channel statistics for denormalizing latents
        self.per_channel_statistics = PerChannelStatistics(latent_channels=ch)
        self.sample_rate = sample_rate
        self.mel_hop_length = mel_hop_length
        self.is_causal = is_causal
        self.mel_bins = mel_bins

        self.patchifier = AudioPatchifier(
            patch_size=1,
            audio_latent_downsample_factor=AUDIO_LATENT_DOWNSAMPLE_FACTOR,
            sample_rate=sample_rate,
            hop_length=mel_hop_length,
            is_causal=is_causal,
        )

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.out_ch = out_ch
        self.norm_type = norm_type
        self.z_channels = z_channels
        self.causality_axis = causality_axis
        self.attn_type = attn_type

        base_block_channels = ch * ch_mult[-1]

        # Input projection from latent channels to base block channels
        self.conv_in = make_conv2d(
            z_channels, base_block_channels, kernel_size=3, stride=1,
            causality_axis=self.causality_axis,
        )
        self.non_linearity = nn.SiLU()

        # Mid block
        self.mid = build_mid_block(
            channels=base_block_channels,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            add_attention=mid_block_add_attention,
        )

        # Upsampling path
        self.up, final_block_channels = build_upsampling_path(
            ch=ch,
            ch_mult=ch_mult,
            num_resolutions=self.num_resolutions,
            num_res_blocks=num_res_blocks,
            resolution=resolution,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            attn_resolutions=attn_resolutions,
            resamp_with_conv=resamp_with_conv,
            initial_block_channels=base_block_channels,
        )

        # Output projection
        self.norm_out = build_normalization_layer(final_block_channels, normtype=self.norm_type)
        self.conv_out = make_conv2d(
            final_block_channels, out_ch, kernel_size=3, stride=1,
            causality_axis=self.causality_axis,
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """Decode audio latents to mel spectrogram.

        Args:
            sample: Normalized audio latents (B, 8, T, 16).

        Returns:
            Mel spectrogram (B, 2, T', 64) where T' accounts for
            upsampling and causal trimming.
        """
        sample, target_shape = self._denormalize_latents(sample)

        h = self.conv_in(sample)
        h = run_mid_block(self.mid, h)
        h = self._run_upsampling_path(h)
        h = self._finalize_output(h)

        return self._adjust_output_shape(h, target_shape)

    def _denormalize_latents(
        self, sample: torch.Tensor
    ) -> tuple[torch.Tensor, AudioLatentShape]:
        """Denormalize latents using per-channel statistics.

        Patchifies the latent (B, C, T, F) -> (B, T, C*F), applies
        denormalization, then unpatchifies back.
        """
        latent_shape = AudioLatentShape.from_tensor(sample)

        sample_patched = self.patchifier.patchify(sample)
        sample_denormalized = self.per_channel_statistics.un_normalize(sample_patched)
        sample = self.patchifier.unpatchify(sample_denormalized, latent_shape)

        # Compute target output shape
        target_frames = latent_shape.frames * AUDIO_LATENT_DOWNSAMPLE_FACTOR
        if self.causality_axis != CausalityAxis.NONE:
            target_frames = max(target_frames - (AUDIO_LATENT_DOWNSAMPLE_FACTOR - 1), 1)

        target_shape = AudioLatentShape(
            batch=latent_shape.batch,
            channels=self.out_ch,
            frames=target_frames,
            mel_bins=self.mel_bins if self.mel_bins is not None else latent_shape.mel_bins,
        )

        return sample, target_shape

    def _run_upsampling_path(self, h: torch.Tensor) -> torch.Tensor:
        """Run through all upsampling stages in reverse resolution order."""
        for level in reversed(range(self.num_resolutions)):
            stage = self.up[level]
            for block_idx, block in enumerate(stage.block):
                h = block(h, temb=None)
                if stage.attn:
                    h = stage.attn[block_idx](h)

            if level != 0 and hasattr(stage, "upsample"):
                h = stage.upsample(h)

        return h

    def _finalize_output(self, h: torch.Tensor) -> torch.Tensor:
        """Apply final normalization, activation, and output projection."""
        h = self.norm_out(h)
        h = self.non_linearity(h)
        return self.conv_out(h)

    def _adjust_output_shape(
        self,
        decoded_output: torch.Tensor,
        target_shape: AudioLatentShape,
    ) -> torch.Tensor:
        """Crop or pad output to match target dimensions.

        Handles the common case where decoded audio spectrograms need
        resizing due to upsampling arithmetic not aligning exactly with
        the target shape.
        """
        _, _, current_time, current_freq = decoded_output.shape
        target_channels = target_shape.channels
        target_time = target_shape.frames
        target_freq = target_shape.mel_bins

        # Crop to target
        decoded_output = decoded_output[
            :, :target_channels,
            :min(current_time, target_time),
            :min(current_freq, target_freq),
        ]

        # Pad if needed
        time_pad = target_time - decoded_output.shape[2]
        freq_pad = target_freq - decoded_output.shape[3]

        if time_pad > 0 or freq_pad > 0:
            padding = (0, max(freq_pad, 0), 0, max(time_pad, 0))
            decoded_output = F.pad(decoded_output, padding)

        # Final safety crop
        return decoded_output[:, :target_channels, :target_time, :target_freq]
