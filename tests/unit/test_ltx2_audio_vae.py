"""
LTX-2 Audio VAE Unit Tests (STUB - Not yet implemented).

Last Updated: 2026-01-22

The Audio VAE will be needed for video-with-audio generation.
Model weights exist at: models/LTX-2/audio_vae/
Reference implementation: coderef/LTX-2/packages/ltx-core/src/ltx_core/model/audio_vae/

Run with: uv run pytest tests/unit/test_ltx2_audio_vae.py -v
"""

import pytest

# Skip all tests in this module - Audio VAE not yet implemented
pytestmark = pytest.mark.skip(reason="Audio VAE not yet implemented")


class TestAudioVAE:
    """Placeholder for Audio VAE tests."""

    def test_audio_vae_not_implemented(self):
        """Audio VAE implementation pending.

        When implementing Audio VAE, add tests for:
        - Audio encoding (waveform -> latents)
        - Audio decoding (latents -> waveform)
        - Vocoder integration
        - Sample rate handling (24kHz expected)
        - Temporal alignment with video
        """
        pytest.skip("Audio VAE not yet ported from coderef")


class TestAudioVAECompression:
    """Placeholder for compression ratio tests."""

    def test_audio_temporal_compression(self):
        """Test audio temporal compression ratio.

        Expected: Audio uses different compression than video VAE.
        Reference: coderef/LTX-2/packages/ltx-core/src/ltx_core/model/audio_vae/
        """
        pytest.skip("Audio VAE compression not yet implemented")


class TestAudioVideoSync:
    """Placeholder for audio-video synchronization tests."""

    def test_av_latent_alignment(self):
        """Test audio and video latents align temporally.

        When generating video with audio, the audio latents must
        align with video latents for proper synchronization.
        """
        pytest.skip("A/V sync not yet implemented")


# Run with: uv run pytest tests/unit/test_ltx2_audio_vae.py -v
