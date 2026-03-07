"""
LTX-2 Audio VAE Package.

Last Updated: 2026-02-26

Pure PyTorch implementation of the LTX-2 Audio VAE for decoding audio
latents to mel spectrograms and synthesizing waveforms via HiFiGAN vocoder.

Decode Pipeline:
    audio latents (B, 8, T, 16) -> AudioDecoder -> mel (B, 2, T', 64)
                                 -> Vocoder -> waveform (B, 2, samples) @ 24kHz

Key Components:
    - AudioDecoder: Decodes audio latents to stereo mel spectrograms
    - Vocoder: HiFiGAN synthesizer, mel to 24kHz stereo waveform
    - AudioPatchifier: 1D temporal patchify/unpatchify for transformer input
    - AudioProcessor: Waveform to mel (encode direction, requires torchaudio)

Example:
    ```python
    from llm_dit.models.ltx2.audio_vae import load_audio_decoder, load_vocoder

    decoder = load_audio_decoder("models/LTX-2/audio_vae/")
    vocoder = load_vocoder("models/LTX-2/vocoder/")

    # Decode latents to waveform
    audio_latents = torch.randn(1, 8, 35, 16)  # normalized latents
    mel = decoder(audio_latents)                 # (1, 2, T', 64)
    waveform = vocoder(mel)                      # (1, 2, samples)
    ```

Ported from: DiffSynth-Studio ltx2_audio_vae
Original source: https://github.com/Lightricks/LTX-2
License: LTX-2 Community License
Copyright (c) 2025 Lightricks Ltd.
"""

# Types
from .types import AudioLatentShape, AUDIO_LATENT_DOWNSAMPLE_FACTOR

# Patchifier
from .patchifier import AudioPatchifier

# Building blocks (selective exports)
from .blocks import (
    CausalityAxis,
    CausalConv2d,
    PerChannelStatistics,
    ResnetBlock,
)

# Core models
from .decoder import AudioDecoder
from .vocoder import Vocoder, VocoderWithBWE, decode_audio

# Loader
from .loader import load_audio_decoder, load_vocoder

__all__ = [
    # Types
    "AudioLatentShape",
    "AUDIO_LATENT_DOWNSAMPLE_FACTOR",
    # Patchifier
    "AudioPatchifier",
    # Building blocks
    "CausalityAxis",
    "CausalConv2d",
    "PerChannelStatistics",
    "ResnetBlock",
    # Core models
    "AudioDecoder",
    "Vocoder",
    "VocoderWithBWE",
    "decode_audio",
    # Loader
    "load_audio_decoder",
    "load_vocoder",
]
