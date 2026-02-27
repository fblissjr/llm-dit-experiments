"""
LTX-2 Audio Processor - Waveform to Mel Spectrogram Conversion.

Last Updated: 2026-02-26

Converts audio waveforms to log-mel spectrograms for encoding.
Uses torchaudio for MelSpectrogram computation (lazy import -- only
needed for the encode direction, not for decode).

Ported from: DiffSynth-Studio ltx2_audio_vae.AudioProcessor
License: LTX-2 Community License
Copyright (c) 2025 Lightricks Ltd.
"""

import torch
import torch.nn as nn


class AudioProcessor(nn.Module):
    """Converts audio waveforms to log-mel spectrograms with optional resampling.

    This processor is only needed for the encode direction (waveform -> mel).
    The decode direction (latents -> mel -> waveform) uses AudioDecoder + Vocoder
    and does not require this class.

    Requires torchaudio (lazy-imported). If torchaudio is not available,
    construction will raise ImportError with installation instructions.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        mel_bins: int = 64,
        mel_hop_length: int = 160,
        n_fft: int = 1024,
    ) -> None:
        super().__init__()

        try:
            import torchaudio
        except ImportError:
            raise ImportError(
                "torchaudio is required for AudioProcessor. "
                "Install with: uv add torchaudio "
                "(requires matching CUDA version with PyTorch)"
            )

        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=mel_hop_length,
            f_min=0.0,
            f_max=sample_rate / 2.0,
            n_mels=mel_bins,
            window_fn=torch.hann_window,
            center=True,
            pad_mode="reflect",
            power=1.0,
            mel_scale="slaney",
            norm="slaney",
        )

    def resample_waveform(
        self,
        waveform: torch.Tensor,
        source_rate: int,
        target_rate: int,
    ) -> torch.Tensor:
        """Resample waveform to target sample rate if needed."""
        if source_rate == target_rate:
            return waveform
        import torchaudio
        resampled = torchaudio.functional.resample(waveform, source_rate, target_rate)
        return resampled.to(device=waveform.device, dtype=waveform.dtype)

    def waveform_to_mel(
        self,
        waveform: torch.Tensor,
        waveform_sample_rate: int,
    ) -> torch.Tensor:
        """Convert waveform to log-mel spectrogram.

        Args:
            waveform: Audio waveform tensor (batch, channels, samples)
            waveform_sample_rate: Sample rate of the input waveform

        Returns:
            Log-mel spectrogram (batch, channels, time, n_mels)
        """
        waveform = self.resample_waveform(waveform, waveform_sample_rate, self.sample_rate)

        mel = self.mel_transform(waveform)
        mel = torch.log(torch.clamp(mel, min=1e-5))

        mel = mel.to(device=waveform.device, dtype=waveform.dtype)
        # Permute from (B, C, n_mels, time) to (B, C, time, n_mels)
        return mel.permute(0, 1, 3, 2).contiguous()
