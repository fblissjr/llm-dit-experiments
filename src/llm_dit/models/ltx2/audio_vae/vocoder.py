"""
LTX-2 Vocoder - HiFiGAN Mel-to-Waveform Synthesis.

Last Updated: 2026-02-26

HiFiGAN-based vocoder that converts mel spectrograms from the AudioDecoder
into 24kHz stereo waveforms. Uses transposed 1D convolutions for upsampling
with residual blocks for quality refinement.

Architecture (from vocoder/config.json):
    - hidden_channels: 1024
    - upsample_factors: [6, 5, 2, 2, 2]  (total: 240x upsampling)
    - resnet_kernel_sizes: [3, 7, 11]
    - resnet_dilations: [[1,3,5], [1,3,5], [1,3,5]]
    - output_sampling_rate: 24000 Hz
    - stereo: True (2-channel output)

Data flow:
    Input:  (B, 2, T, 64)   -- stereo mel spectrogram from AudioDecoder
    Output: (B, 2, T*240)   -- stereo waveform @ 24kHz

Ported from: DiffSynth-Studio ltx2_audio_vae.LTX2Vocoder
License: LTX-2 Community License
Copyright (c) 2025 Lightricks Ltd.
"""

import math
from typing import List, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


LRELU_SLOPE = 0.1


class ResBlock1(nn.Module):
    """HiFiGAN residual block with dilated convolutions.

    Each block applies 3 pairs of (dilated conv -> residual conv).
    The dilated convolutions capture multi-scale temporal patterns
    while the residual connections preserve information.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int, int] = (1, 3, 5),
    ):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding="same")
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding="same")
            for _ in dilation
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2, strict=True):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = conv1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = conv2(xt)
            x = xt + x
        return x


class ResBlock2(nn.Module):
    """Simpler HiFiGAN residual block with 2 dilated convolutions."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int] = (1, 3),
    ):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding="same")
            for d in dilation
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = conv(xt)
            x = xt + x
        return x


class Vocoder(nn.Module):
    """HiFiGAN vocoder for synthesizing audio from mel spectrograms.

    Progressively upsamples the mel spectrogram through transposed
    convolutions, refining quality at each stage with residual blocks.
    The upsample factors [6, 5, 2, 2, 2] produce a total 240x upsampling
    to go from mel frames to audio samples.

    For stereo input: mel channels (B, 2, T, 64) are rearranged to
    (B, 128, T) before processing, and the output is split back to
    (B, 2, audio_length).

    Args:
        resblock_kernel_sizes: Kernel sizes for residual blocks.
        upsample_rates: Upsampling factors per stage.
        upsample_kernel_sizes: Kernel sizes for transposed convolutions.
        resblock_dilation_sizes: Dilation patterns for residual blocks.
        upsample_initial_channel: Initial hidden channel count (1024).
        stereo: Whether to produce stereo output.
        resblock: ResBlock type ("1" for ResBlock1, "2" for ResBlock2).
        output_sample_rate: Output waveform sample rate (24000 Hz).
    """

    def __init__(
        self,
        resblock_kernel_sizes: List[int] | None = None,
        upsample_rates: List[int] | None = None,
        upsample_kernel_sizes: List[int] | None = None,
        resblock_dilation_sizes: List[List[int]] | None = None,
        upsample_initial_channel: int = 1024,
        stereo: bool = True,
        resblock: str = "1",
        output_sample_rate: int = 24000,
    ):
        super().__init__()

        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if upsample_rates is None:
            upsample_rates = [6, 5, 2, 2, 2]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [16, 15, 8, 4, 4]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        self.output_sample_rate = output_sample_rate
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        in_channels = 128 if stereo else 64
        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)

        resblock_class = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (stride, kernel_size) in enumerate(
            zip(upsample_rates, upsample_kernel_sizes, strict=True)
        ):
            self.ups.append(
                nn.ConvTranspose1d(
                    upsample_initial_channel // (2 ** i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size,
                    stride,
                    padding=(kernel_size - stride) // 2,
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilations in zip(
                resblock_kernel_sizes, resblock_dilation_sizes, strict=True
            ):
                self.resblocks.append(resblock_class(ch, kernel_size, dilations))

        out_channels = 2 if stereo else 1
        final_channels = upsample_initial_channel // (2 ** self.num_upsamples)
        self.conv_post = nn.Conv1d(final_channels, out_channels, 7, 1, padding=3)

        self.upsample_factor = math.prod(layer.stride[0] for layer in self.ups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Synthesize waveform from mel spectrogram.

        Args:
            x: Mel spectrogram tensor.
                3D: (B, time, mel_bins) for mono
                4D: (B, 2, time, mel_bins) for stereo

        Returns:
            Waveform tensor (B, out_channels, audio_length).
        """
        # (B, C, time, mel_bins) -> (B, C, mel_bins, time)
        x = x.transpose(2, 3)

        if x.dim() == 4:  # stereo: merge channels and mel_bins
            x = einops.rearrange(x, "b s c t -> b (s c) t")

        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            start = i * self.num_kernels
            end = start + self.num_kernels

            # Evaluate all resblocks independently then average
            block_outputs = torch.stack(
                [self.resblocks[idx](x) for idx in range(start, end)],
                dim=0,
            )
            x = block_outputs.mean(dim=0)

        x = self.conv_post(F.leaky_relu(x))
        return torch.tanh(x)


def decode_audio(
    latent: torch.Tensor,
    audio_decoder: nn.Module,
    vocoder: Vocoder,
) -> torch.Tensor:
    """Decode audio latent to waveform using decoder + vocoder pipeline.

    Args:
        latent: Audio latent tensor (B, 8, T, 16).
        audio_decoder: AudioDecoder model (latents -> mel).
        vocoder: Vocoder model (mel -> waveform).

    Returns:
        Waveform tensor as float (channels, audio_length).
    """
    decoded_audio = audio_decoder(latent)
    decoded_audio = vocoder(decoded_audio).squeeze(0).float()
    return decoded_audio
