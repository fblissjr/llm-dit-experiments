"""
LTX-2 Audio VAE Unit Tests.

Last Updated: 2026-02-26

Tests for the Audio VAE decode pipeline:
    audio latents (B, 8, T, 16) -> AudioDecoder -> mel (B, 2, T', 64)
                                 -> Vocoder -> waveform (B, 2, samples) @ 24kHz

Run with: uv run pytest tests/unit/test_ltx2_audio_vae.py -v
"""

from pathlib import Path

import pytest
import torch

from llm_dit.models.ltx2.audio_vae import (
    AUDIO_LATENT_DOWNSAMPLE_FACTOR,
    AudioDecoder,
    AudioLatentShape,
    AudioPatchifier,
    PerChannelStatistics,
    Vocoder,
)


# ---------------------------------------------------------------------------
# AudioLatentShape
# ---------------------------------------------------------------------------


class TestAudioLatentShape:
    """Tests for AudioLatentShape type."""

    def test_to_torch_shape(self):
        shape = AudioLatentShape(batch=2, channels=8, frames=35, mel_bins=16)
        assert shape.to_torch_shape() == (2, 8, 35, 16)

    def test_from_tensor(self):
        t = torch.randn(1, 8, 35, 16)
        shape = AudioLatentShape.from_tensor(t)
        assert shape.batch == 1
        assert shape.channels == 8
        assert shape.frames == 35
        assert shape.mel_bins == 16

    def test_from_tensor_wrong_dims(self):
        t = torch.randn(1, 8, 35)
        with pytest.raises(ValueError, match="4D"):
            AudioLatentShape.from_tensor(t)

    def test_frozen(self):
        shape = AudioLatentShape(batch=1, channels=8, frames=35, mel_bins=16)
        with pytest.raises(AttributeError):
            shape.batch = 2


# ---------------------------------------------------------------------------
# AudioPatchifier
# ---------------------------------------------------------------------------


class TestAudioPatchifier:
    """Tests for audio patchify/unpatchify operations."""

    @pytest.fixture
    def patchifier(self):
        return AudioPatchifier(
            patch_size=1,
            sample_rate=16000,
            hop_length=160,
            audio_latent_downsample_factor=4,
            is_causal=True,
        )

    def test_patchify_shape(self, patchifier):
        """Patchify should flatten (B, C, T, F) -> (B, T, C*F)."""
        x = torch.randn(2, 8, 35, 16)
        result = patchifier.patchify(x)
        assert result.shape == (2, 35, 128)  # 8 * 16 = 128

    def test_unpatchify_shape(self, patchifier):
        """Unpatchify should restore (B, T, C*F) -> (B, C, T, F)."""
        shape = AudioLatentShape(batch=2, channels=8, frames=35, mel_bins=16)
        x = torch.randn(2, 35, 128)
        result = patchifier.unpatchify(x, shape)
        assert result.shape == (2, 8, 35, 16)

    def test_round_trip(self, patchifier):
        """patchify -> unpatchify should be identity."""
        x = torch.randn(1, 8, 35, 16)
        shape = AudioLatentShape.from_tensor(x)
        patched = patchifier.patchify(x)
        unpatched = patchifier.unpatchify(patched, shape)
        assert torch.allclose(x, unpatched)

    def test_round_trip_various_lengths(self, patchifier):
        """Round-trip should work for various temporal lengths."""
        for frames in [1, 10, 35, 100]:
            x = torch.randn(1, 8, frames, 16)
            shape = AudioLatentShape.from_tensor(x)
            result = patchifier.unpatchify(patchifier.patchify(x), shape)
            assert torch.allclose(x, result), f"Failed for frames={frames}"

    def test_token_count(self, patchifier):
        """Token count should equal number of temporal frames."""
        shape = AudioLatentShape(batch=1, channels=8, frames=35, mel_bins=16)
        assert patchifier.get_token_count(shape) == 35

    def test_patch_grid_bounds_shape(self, patchifier):
        """Grid bounds should have shape (B, 1, T, 2)."""
        shape = AudioLatentShape(batch=2, channels=8, frames=35, mel_bins=16)
        bounds = patchifier.get_patch_grid_bounds(shape)
        assert bounds.shape == (2, 1, 35, 2)

    def test_timing_monotonic(self, patchifier):
        """Timestamps should be monotonically increasing."""
        shape = AudioLatentShape(batch=1, channels=8, frames=35, mel_bins=16)
        bounds = patchifier.get_patch_grid_bounds(shape)
        starts = bounds[0, 0, :, 0]  # start times
        # Start times should be non-decreasing
        assert (starts[1:] >= starts[:-1]).all()

    def test_timing_non_causal(self):
        """Non-causal patchifier should produce different timings."""
        causal = AudioPatchifier(is_causal=True)
        non_causal = AudioPatchifier(is_causal=False)
        shape = AudioLatentShape(batch=1, channels=8, frames=10, mel_bins=16)
        causal_bounds = causal.get_patch_grid_bounds(shape)
        non_causal_bounds = non_causal.get_patch_grid_bounds(shape)
        # First frame timing should differ (causal clips to 0)
        assert not torch.allclose(causal_bounds, non_causal_bounds)


# ---------------------------------------------------------------------------
# PerChannelStatistics
# ---------------------------------------------------------------------------


class TestPerChannelStatistics:
    """Tests for audio latent normalization/denormalization."""

    @pytest.fixture
    def stats(self):
        s = PerChannelStatistics(latent_channels=128)
        # Set known statistics
        s.get_buffer("std-of-means").copy_(torch.ones(128) * 2.0)
        s.get_buffer("mean-of-means").copy_(torch.ones(128) * 1.0)
        return s

    def test_normalize_denormalize_round_trip(self, stats):
        """normalize -> un_normalize should be identity."""
        x = torch.randn(1, 35, 128)
        normalized = stats.normalize(x)
        recovered = stats.un_normalize(normalized)
        assert torch.allclose(x, recovered, atol=1e-5)

    def test_normalize_formula(self, stats):
        """Verify normalize applies (x - mean) / std."""
        x = torch.ones(1, 1, 128) * 5.0
        result = stats.normalize(x)
        # (5.0 - 1.0) / 2.0 = 2.0
        assert torch.allclose(result, torch.ones_like(result) * 2.0)

    def test_un_normalize_formula(self, stats):
        """Verify un_normalize applies x * std + mean."""
        x = torch.ones(1, 1, 128) * 2.0
        result = stats.un_normalize(x)
        # 2.0 * 2.0 + 1.0 = 5.0
        assert torch.allclose(result, torch.ones_like(result) * 5.0)


# ---------------------------------------------------------------------------
# AudioDecoder (construction and forward shapes)
# ---------------------------------------------------------------------------


class TestAudioDecoder:
    """Tests for AudioDecoder construction and forward pass shapes."""

    @pytest.fixture
    def decoder(self):
        """Small decoder for testing (no checkpoint needed)."""
        d = AudioDecoder(
            ch=32,  # small for testing
            out_ch=2,
            ch_mult=(1, 2),
            num_res_blocks=1,
            resolution=64,
            z_channels=8,
            mel_bins=16,
        )
        # Initialize per-channel stats with identity transform
        d.per_channel_statistics.get_buffer("std-of-means").fill_(1.0)
        d.per_channel_statistics.get_buffer("mean-of-means").fill_(0.0)
        return d

    def test_construction(self, decoder):
        """Decoder should construct without errors."""
        assert isinstance(decoder, AudioDecoder)

    def test_forward_shape(self, decoder):
        """Forward should produce expected output shape.

        Input F_latent must satisfy: z_channels * F_latent = ch (for
        PerChannelStatistics buffer alignment). ch=32, z_channels=8 -> F=4.
        """
        with torch.no_grad():
            x = torch.randn(1, 8, 10, 4)
            out = decoder(x)
            assert out.shape[0] == 1  # batch
            assert out.shape[1] == 2  # stereo channels
            assert out.dim() == 4     # (B, C, T, F)

    def test_output_channels(self, decoder):
        """Output should have correct number of channels."""
        with torch.no_grad():
            x = torch.randn(1, 8, 10, 4)
            out = decoder(x)
            assert out.shape[1] == 2

    def test_batch_dimension_preserved(self, decoder):
        """Batch dimension should be preserved through forward."""
        with torch.no_grad():
            x = torch.randn(3, 8, 10, 4)
            out = decoder(x)
            assert out.shape[0] == 3


# ---------------------------------------------------------------------------
# Vocoder (construction and forward shapes)
# ---------------------------------------------------------------------------


class TestVocoder:
    """Tests for Vocoder construction and forward pass shapes."""

    @pytest.fixture
    def vocoder(self):
        """Small vocoder for testing."""
        return Vocoder(
            resblock_kernel_sizes=[3],
            upsample_rates=[2, 2],
            upsample_kernel_sizes=[4, 4],
            resblock_dilation_sizes=[[1, 3, 5]],
            upsample_initial_channel=64,
            stereo=True,
            output_sample_rate=24000,
        )

    def test_construction(self, vocoder):
        """Vocoder should construct without errors."""
        assert isinstance(vocoder, Vocoder)

    def test_upsample_factor(self, vocoder):
        """Upsample factor should be product of rates."""
        assert vocoder.upsample_factor == 4  # 2 * 2

    def test_forward_stereo(self, vocoder):
        """Forward should handle stereo input (B, 2, T, mel)."""
        with torch.no_grad():
            x = torch.randn(1, 2, 10, 64)
            out = vocoder(x)
            assert out.shape[0] == 1
            assert out.shape[1] == 2  # stereo output
            assert out.shape[2] == 10 * vocoder.upsample_factor

    def test_output_sample_rate(self, vocoder):
        """Output sample rate should be set correctly."""
        assert vocoder.output_sample_rate == 24000


# ---------------------------------------------------------------------------
# Full decode pipeline (construction only, no weights)
# ---------------------------------------------------------------------------


class TestDecodeAudioPipeline:
    """Tests for the full decode pipeline with small models."""

    def test_decode_pipeline_shapes(self):
        """Decoder produces correct mel shape for vocoder consumption.

        The vocoder expects stereo mel (B, 2, T, 64) in production. Here
        we validate the decoder output is a valid 4D stereo mel tensor.
        Full end-to-end testing with real vocoder dimensions is done in
        TestFullDecodeWithWeights.
        """
        decoder = AudioDecoder(
            ch=32, out_ch=2, ch_mult=(1, 2), num_res_blocks=1,
            resolution=64, z_channels=8, mel_bins=16,
        )
        decoder.per_channel_statistics.get_buffer("std-of-means").fill_(1.0)
        decoder.per_channel_statistics.get_buffer("mean-of-means").fill_(0.0)

        with torch.no_grad():
            latents = torch.randn(1, 8, 10, 4)  # F=ch/z_channels=32/8=4
            mel = decoder(latents)
            assert mel.dim() == 4
            assert mel.shape[0] == 1   # batch
            assert mel.shape[1] == 2   # stereo channels
            assert mel.shape[3] == 16  # mel_bins


# ---------------------------------------------------------------------------
# Weight loading (requires model files on disk)
# ---------------------------------------------------------------------------


AUDIO_VAE_PATH = Path("models/LTX-2/audio_vae")
VOCODER_PATH = Path("models/LTX-2/vocoder")


@pytest.mark.skipif(
    not AUDIO_VAE_PATH.exists(),
    reason="Audio VAE weights not found at models/LTX-2/audio_vae/"
)
class TestAudioDecoderWeightLoading:
    """Tests requiring actual model weights on disk."""

    def test_load_audio_decoder(self):
        """Load audio decoder from checkpoint."""
        from llm_dit.models.ltx2.audio_vae import load_audio_decoder
        decoder = load_audio_decoder(AUDIO_VAE_PATH)
        assert sum(p.numel() for p in decoder.parameters()) > 0

        # Verify per-channel statistics loaded
        std = decoder.per_channel_statistics.get_buffer("std-of-means")
        assert std.abs().max() > 0.1, "std-of-means should be loaded from checkpoint"

    def test_audio_decoder_forward_shape(self):
        """Forward pass with real weights should produce correct shapes."""
        from llm_dit.models.ltx2.audio_vae import load_audio_decoder
        decoder = load_audio_decoder(AUDIO_VAE_PATH)

        with torch.no_grad():
            latents = torch.randn(1, 8, 35, 16, dtype=torch.bfloat16)
            mel = decoder(latents)
            assert mel.shape[0] == 1
            assert mel.shape[1] == 2     # stereo
            assert mel.shape[3] == 64    # mel bins
            # T' should be close to 35 * 4 - 3 = 137
            assert 130 <= mel.shape[2] <= 145


@pytest.mark.skipif(
    not VOCODER_PATH.exists(),
    reason="Vocoder weights not found at models/LTX-2/vocoder/"
)
class TestVocoderWeightLoading:
    """Tests requiring actual vocoder weights on disk."""

    def test_load_vocoder(self):
        """Load vocoder from checkpoint."""
        from llm_dit.models.ltx2.audio_vae import load_vocoder
        vocoder = load_vocoder(VOCODER_PATH)
        assert vocoder.upsample_factor == 240
        assert vocoder.output_sample_rate == 24000

    def test_vocoder_forward_shape(self):
        """Forward pass with real weights."""
        from llm_dit.models.ltx2.audio_vae import load_vocoder
        vocoder = load_vocoder(VOCODER_PATH)

        with torch.no_grad():
            mel = torch.randn(1, 2, 137, 64, dtype=torch.bfloat16)
            waveform = vocoder(mel)
            assert waveform.shape[0] == 1
            assert waveform.shape[1] == 2  # stereo
            # 137 * 240 = 32880 samples
            assert waveform.shape[2] == 137 * 240


@pytest.mark.skipif(
    not (AUDIO_VAE_PATH.exists() and VOCODER_PATH.exists()),
    reason="Audio VAE and vocoder weights required"
)
class TestFullDecodeWithWeights:
    """End-to-end decode pipeline test with real weights."""

    def test_full_decode_pipeline(self):
        """latents -> decoder -> vocoder -> waveform with real weights."""
        from llm_dit.models.ltx2.audio_vae import load_audio_decoder, load_vocoder

        decoder = load_audio_decoder(AUDIO_VAE_PATH)
        vocoder = load_vocoder(VOCODER_PATH)

        with torch.no_grad():
            latents = torch.randn(1, 8, 35, 16, dtype=torch.bfloat16)
            mel = decoder(latents)
            waveform = vocoder(mel)

            assert waveform.dim() == 3
            assert waveform.shape[1] == 2  # stereo
            duration = waveform.shape[2] / vocoder.output_sample_rate
            # For 35 latent frames, expect ~1.37s audio
            assert 1.0 <= duration <= 2.0, f"Unexpected duration: {duration:.3f}s"


# ---------------------------------------------------------------------------
# Audio-Video Sync
# ---------------------------------------------------------------------------


class TestAudioVideoSync:
    """Tests for audio-video temporal alignment."""

    def test_audio_latent_frames_for_video(self):
        """Audio latent frame count should match video duration.

        For 33 video frames at 24fps = 1.375s:
            mel frames = 1.375 * 16000 / 160 = 137.5 -> 138
            audio latent frames = 138 / 4 = 34.5 -> 35 (rounded up)
        """
        video_frames = 33
        fps = 24
        duration_s = video_frames / fps  # 1.375

        sample_rate = 16000
        hop_length = 160
        mel_frames = int(duration_s * sample_rate / hop_length)

        audio_latent_frames = mel_frames // AUDIO_LATENT_DOWNSAMPLE_FACTOR
        # Should be around 34-35
        assert 34 <= audio_latent_frames <= 36

    def test_downsample_factor(self):
        """Audio latent downsample factor should be 4."""
        assert AUDIO_LATENT_DOWNSAMPLE_FACTOR == 4
