"""
LTX-2.3 Vocoder - BigVGAN v2 Mel-to-Waveform Synthesis with BWE.

Last Updated: 2026-03-06

BigVGAN v2 vocoder that converts mel spectrograms from the AudioDecoder
into 48kHz stereo waveforms. Uses anti-aliased upsampling with SnakeBeta
activations and bandwidth extension (BWE) for high-fidelity output.

Architecture (V2.3):
    Base vocoder:
        - upsample_initial_channel: 1536
        - upsample_rates: [5, 2, 2, 2, 2, 2]  (total: 160x upsampling)
        - resblock: AMP1 (BigVGAN v2 with SnakeBeta activations)
        - output: 16kHz stereo
    BWE generator:
        - upsample_initial_channel: 512
        - upsample_rates: [6, 5, 2, 2, 2]  (total: 240x upsampling)
        - upsamples 16kHz -> 48kHz via mel re-analysis + residual

Data flow:
    Input:  (B, 2, T, 64)   -- stereo mel spectrogram from AudioDecoder
    Output: (B, 2, T_audio)  -- stereo waveform @ 48kHz

Ported from: ltx-core/model/audio_vae/vocoder.py (official LTX-2.3 reference)
License: LTX-2 Community License
Copyright (c) 2025-2026 Lightricks Ltd.
"""

import math
from typing import List, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


LRELU_SLOPE = 0.1


# ---------------------------------------------------------------------------
# Anti-aliased resampling helpers (kaiser-sinc filters) for BigVGAN v2
# ---------------------------------------------------------------------------


def _sinc(x: torch.Tensor) -> torch.Tensor:
    return torch.where(
        x == 0,
        torch.tensor(1.0, device=x.device, dtype=x.dtype),
        torch.sin(math.pi * x) / math.pi / x,
    )


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    amplitude = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if amplitude > 50.0:
        beta = 0.1102 * (amplitude - 8.7)
    elif amplitude >= 21.0:
        beta = 0.5842 * (amplitude - 21) ** 0.4 + 0.07886 * (amplitude - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    time = torch.arange(-half_size, half_size) + 0.5 if even else torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * _sinc(2 * cutoff * time)
        filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff: float = 0.5,
        half_width: float = 0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "replicate",
        kernel_size: int = 12,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, n_channels, _ = x.shape
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        return F.conv1d(x, self.filter.expand(n_channels, -1, -1), stride=self.stride, groups=n_channels)


class UpSample1d(nn.Module):
    def __init__(
        self,
        ratio: int = 2,
        kernel_size: int | None = None,
        persistent: bool = True,
        window_type: str = "kaiser",
    ) -> None:
        super().__init__()
        self.ratio = ratio
        self.stride = ratio

        if window_type == "hann":
            rolloff = 0.99
            lowpass_filter_width = 6
            width = math.ceil(lowpass_filter_width / rolloff)
            self.kernel_size = 2 * width * ratio + 1
            self.pad = width
            self.pad_left = 2 * width * ratio
            self.pad_right = self.kernel_size - ratio
            time_axis = (torch.arange(self.kernel_size) / ratio - width) * rolloff
            time_clamped = time_axis.clamp(-lowpass_filter_width, lowpass_filter_width)
            window = torch.cos(time_clamped * math.pi / lowpass_filter_width / 2) ** 2
            sinc_filter = (torch.sinc(time_axis) * window * rolloff / ratio).view(1, 1, -1)
        else:
            self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
            self.pad = self.kernel_size // ratio - 1
            self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
            self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
            sinc_filter = kaiser_sinc_filter1d(
                cutoff=0.5 / ratio,
                half_width=0.6 / ratio,
                kernel_size=self.kernel_size,
            )

        self.register_buffer("filter", sinc_filter, persistent=persistent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, n_channels, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        filt = self.filter.to(dtype=x.dtype, device=x.device).expand(n_channels, -1, -1)
        x = self.ratio * F.conv_transpose1d(x, filt, stride=self.stride, groups=n_channels)
        return x[..., self.pad_left : -self.pad_right]


class DownSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int | None = None) -> None:
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


class Snake(nn.Module):
    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = True,
    ) -> None:
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = nn.Parameter(torch.zeros(in_features) if alpha_logscale else torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.eps = 1e-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        return x + (1.0 / (alpha + self.eps)) * torch.sin(x * alpha).pow(2)


class SnakeBeta(nn.Module):
    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = True,
    ) -> None:
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = nn.Parameter(torch.zeros(in_features) if alpha_logscale else torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta = nn.Parameter(torch.zeros(in_features) if alpha_logscale else torch.ones(in_features) * alpha)
        self.beta.requires_grad = alpha_trainable
        self.eps = 1e-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return x + (1.0 / (beta + self.eps)) * torch.sin(x * alpha).pow(2)


class Activation1d(nn.Module):
    """Anti-aliased activation: upsample -> activate -> downsample."""

    def __init__(
        self,
        activation: nn.Module,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ) -> None:
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.act(x)
        return self.downsample(x)


# ---------------------------------------------------------------------------
# Residual blocks
# ---------------------------------------------------------------------------


class ResBlock1(nn.Module):
    """HiFiGAN residual block with dilated convolutions and LeakyReLU."""

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


class AMPBlock1(nn.Module):
    """BigVGAN v2 residual block with anti-aliased SnakeBeta activations."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, ...] = (1, 3, 5),
        activation: str = "snake",
    ) -> None:
        super().__init__()
        act_cls = SnakeBeta if activation == "snakebeta" else Snake

        def _get_padding(k: int, d: int) -> int:
            return int((k * d - d) / 2)

        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=_get_padding(kernel_size, d))
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=_get_padding(kernel_size, 1))
            for _ in dilation
        ])
        self.acts1 = nn.ModuleList([Activation1d(act_cls(channels)) for _ in range(len(self.convs1))])
        self.acts2 = nn.ModuleList([Activation1d(act_cls(channels)) for _ in range(len(self.convs2))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.acts1, self.acts2, strict=True):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = x + xt
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


# ---------------------------------------------------------------------------
# Vocoder
# ---------------------------------------------------------------------------


class Vocoder(nn.Module):
    """HiFiGAN / BigVGAN v2 vocoder for synthesizing audio from mel spectrograms.

    Supports both legacy HiFiGAN (resblock="1") and BigVGAN v2 (resblock="AMP1")
    with anti-aliased SnakeBeta activations.

    Args:
        resblock_kernel_sizes: Kernel sizes for residual blocks.
        upsample_rates: Upsampling factors per stage.
        upsample_kernel_sizes: Kernel sizes for transposed convolutions.
        resblock_dilation_sizes: Dilation patterns for residual blocks.
        upsample_initial_channel: Initial hidden channel count.
        stereo: Whether to produce stereo output.
        resblock: ResBlock type ("1" for ResBlock1, "AMP1" for AMPBlock1).
        output_sample_rate: Output waveform sample rate.
        activation: Activation type for AMP1 ("snake" or "snakebeta").
        use_tanh_at_final: Apply tanh at the output.
        apply_final_activation: Whether to apply the final activation.
        use_bias_at_final: Whether to use bias in the final conv layer.
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
        activation: str = "snake",
        use_tanh_at_final: bool = True,
        apply_final_activation: bool = True,
        use_bias_at_final: bool = True,
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
        self.use_tanh_at_final = use_tanh_at_final
        self.apply_final_activation = apply_final_activation
        self.is_amp = resblock == "AMP1"

        in_channels = 128 if stereo else 64
        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)

        resblock_cls = ResBlock1 if resblock == "1" else AMPBlock1

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

        final_channels = upsample_initial_channel // (2 ** self.num_upsamples)

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilations in zip(
                resblock_kernel_sizes, resblock_dilation_sizes, strict=True
            ):
                if self.is_amp:
                    self.resblocks.append(resblock_cls(ch, kernel_size, dilations, activation=activation))
                else:
                    self.resblocks.append(resblock_cls(ch, kernel_size, dilations))

        if self.is_amp:
            self.act_post: nn.Module = Activation1d(SnakeBeta(final_channels))
        else:
            self.act_post = nn.LeakyReLU()

        out_channels = 2 if stereo else 1
        self.conv_post = nn.Conv1d(final_channels, out_channels, 7, 1, padding=3, bias=use_bias_at_final)

        self.upsample_factor = math.prod(upsample_rates)

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
            if not self.is_amp:
                x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            start = i * self.num_kernels
            end = start + self.num_kernels

            block_outputs = torch.stack(
                [self.resblocks[idx](x) for idx in range(start, end)],
                dim=0,
            )
            x = block_outputs.mean(dim=0)

        x = self.act_post(x)
        x = self.conv_post(x)

        if self.apply_final_activation:
            x = torch.tanh(x) if self.use_tanh_at_final else torch.clamp(x, -1, 1)

        return x


# ---------------------------------------------------------------------------
# MelSTFT and VocoderWithBWE (V2.3)
# ---------------------------------------------------------------------------


class _STFTFn(nn.Module):
    """STFT as a convolution with precomputed DFT x Hann-window bases.

    Bases are loaded from the checkpoint, ensuring bit-identical mel values.
    """

    def __init__(self, filter_length: int, hop_length: int, win_length: int) -> None:
        super().__init__()
        self.hop_length = hop_length
        self.win_length = win_length
        n_freqs = filter_length // 2 + 1
        self.register_buffer("forward_basis", torch.zeros(n_freqs * 2, 1, filter_length))
        self.register_buffer("inverse_basis", torch.zeros(n_freqs * 2, 1, filter_length))

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if y.dim() == 2:
            y = y.unsqueeze(1)
        left_pad = max(0, self.win_length - self.hop_length)  # causal: left-only
        y = F.pad(y, (left_pad, 0))
        spec = F.conv1d(y, self.forward_basis, stride=self.hop_length, padding=0)
        n_freqs = spec.shape[1] // 2
        real, imag = spec[:, :n_freqs], spec[:, n_freqs:]
        magnitude = torch.sqrt(real**2 + imag**2)
        phase = torch.atan2(imag.float(), real.float()).to(real.dtype)
        return magnitude, phase


class MelSTFT(nn.Module):
    """Causal log-mel spectrogram module whose buffers are loaded from the checkpoint."""

    def __init__(
        self,
        filter_length: int,
        hop_length: int,
        win_length: int,
        n_mel_channels: int,
    ) -> None:
        super().__init__()
        self.stft_fn = _STFTFn(filter_length, hop_length, win_length)
        n_freqs = filter_length // 2 + 1
        self.register_buffer("mel_basis", torch.zeros(n_mel_channels, n_freqs))

    def mel_spectrogram(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        magnitude, phase = self.stft_fn(y)
        energy = torch.norm(magnitude, dim=1)
        mel = torch.matmul(self.mel_basis.to(magnitude.dtype), magnitude)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel, magnitude, phase, energy


class VocoderWithBWE(nn.Module):
    """Vocoder with bandwidth extension (BWE) upsampling.

    Chains a mel-to-wav vocoder with a BWE module that upsamples the output
    to a higher sample rate. The BWE computes a mel spectrogram from the
    vocoder output, runs it through a second generator to predict a residual,
    and adds it to a sinc-resampled skip connection.

    V2.3 config: base vocoder at 16kHz, BWE upsamples to 48kHz.
    """

    def __init__(
        self,
        vocoder: Vocoder,
        bwe_generator: Vocoder,
        mel_stft: MelSTFT,
        input_sampling_rate: int,
        output_sampling_rate: int,
        hop_length: int,
    ) -> None:
        super().__init__()
        self.vocoder = vocoder
        self.bwe_generator = bwe_generator
        self.mel_stft = mel_stft
        self.input_sampling_rate = input_sampling_rate
        self.output_sampling_rate = output_sampling_rate
        self.hop_length = hop_length
        # Compute resampler on CPU so sinc filter is materialized even on meta device
        with torch.device("cpu"):
            self.resampler = UpSample1d(
                ratio=output_sampling_rate // input_sampling_rate, persistent=False, window_type="hann"
            )

    @property
    def output_sample_rate(self) -> int:
        return self.output_sampling_rate

    def _compute_mel(self, audio: torch.Tensor) -> torch.Tensor:
        batch, n_channels, _ = audio.shape
        flat = audio.reshape(batch * n_channels, -1)
        mel, _, _, _ = self.mel_stft.mel_spectrogram(flat)
        return mel.reshape(batch, n_channels, mel.shape[1], mel.shape[2])

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Run the full vocoder + BWE forward pass.

        Args:
            mel_spec: Mel spectrogram of shape (B, 2, T, mel_bins) for stereo.

        Returns:
            Waveform tensor of shape (B, out_channels, T_out) clipped to [-1, 1].
        """
        x = self.vocoder(mel_spec)
        _, _, length_low_rate = x.shape
        output_length = length_low_rate * self.output_sampling_rate // self.input_sampling_rate

        # Pad to multiple of hop_length for exact mel frame count
        remainder = length_low_rate % self.hop_length
        if remainder != 0:
            x = F.pad(x, (0, self.hop_length - remainder))

        mel = self._compute_mel(x)
        # Vocoder.forward expects (B, C, T, mel_bins)
        mel_for_bwe = mel.transpose(2, 3)
        residual = self.bwe_generator(mel_for_bwe)
        skip = self.resampler(x)
        assert residual.shape == skip.shape, f"residual {residual.shape} != skip {skip.shape}"

        return torch.clamp(residual + skip, -1, 1)[..., :output_length]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def decode_audio(
    latent: torch.Tensor,
    audio_decoder: nn.Module,
    vocoder: Vocoder | VocoderWithBWE,
) -> torch.Tensor:
    """Decode audio latent to waveform using decoder + vocoder pipeline.

    Args:
        latent: Audio latent tensor (B, 8, T, 16).
        audio_decoder: AudioDecoder model (latents -> mel).
        vocoder: Vocoder or VocoderWithBWE model (mel -> waveform).

    Returns:
        Waveform tensor as float (channels, audio_length).
    """
    decoded_audio = audio_decoder(latent)
    decoded_audio = vocoder(decoded_audio).squeeze(0).float()
    return decoded_audio
