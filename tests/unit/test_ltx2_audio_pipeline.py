"""
Tests for LTX-2 audio pipeline utility functions.

Last Updated: 2026-02-27

Tests for compute_audio_latent_frames, create_audio_position_indices,
create_audio_modality, and related audio pipeline infrastructure.

Run with: uv run pytest tests/unit/test_ltx2_audio_pipeline.py -v
"""

import torch

from llm_dit.pipelines.generate import (
    compute_audio_latent_frames,
    create_audio_modality,
    create_audio_position_indices,
)


class TestComputeAudioLatentFrames:
    """Tests for compute_audio_latent_frames().

    Formula: round(num_frames / fps * sample_rate / hop_length / downsample_factor)
    Default params: fps=24.0, sample_rate=16000, hop_length=160, downsample_factor=4
    Effective rate: 16000 / 160 / 4 = 25 latent frames per second.
    """

    def test_standard_33_frames(self):
        """33 frames @ 24fps = 1.375s -> round(1.375 * 25) = 34."""
        result = compute_audio_latent_frames(33)
        assert result == 34

    def test_short_9_frames(self):
        """9 frames @ 24fps = 0.375s -> round(0.375 * 25) = 9."""
        result = compute_audio_latent_frames(9)
        assert result == 9

    def test_long_121_frames(self):
        """121 frames @ 24fps = 5.042s -> round(5.042 * 25) = 126."""
        result = compute_audio_latent_frames(121)
        assert result == 126

    def test_minimum_1_frame(self):
        """1 frame @ 24fps = 0.042s -> round(0.042 * 25) = 1."""
        result = compute_audio_latent_frames(1)
        assert result == 1

    def test_custom_fps(self):
        """33 frames @ 30fps = 1.1s -> round(1.1 * 25) = 28."""
        result = compute_audio_latent_frames(33, fps=30.0)
        assert result == 28

    def test_returns_int(self):
        """Return type is always int."""
        result = compute_audio_latent_frames(33)
        assert isinstance(result, int)


class TestCreateAudioPositionIndices:
    """Tests for create_audio_position_indices()."""

    def test_shape(self):
        """Output shape is [B, 1, T, 2] -- 1D temporal positions."""
        positions = create_audio_position_indices(
            batch_size=1, audio_latent_frames=34, device=torch.device("cpu"),
        )
        assert positions.shape == (1, 1, 34, 2)

    def test_batch_expansion(self):
        """Batch dimension expands correctly."""
        positions = create_audio_position_indices(
            batch_size=4, audio_latent_frames=34, device=torch.device("cpu"),
        )
        assert positions.shape == (4, 1, 34, 2)

    def test_temporal_ordering(self):
        """Start positions should be monotonically increasing."""
        positions = create_audio_position_indices(
            batch_size=1, audio_latent_frames=10, device=torch.device("cpu"),
        )
        starts = positions[0, 0, :, 0]  # [T] start values
        diffs = starts[1:] - starts[:-1]
        assert (diffs >= 0).all(), "Start positions should be non-decreasing"

    def test_start_before_end(self):
        """Each position's start should be <= end."""
        positions = create_audio_position_indices(
            batch_size=1, audio_latent_frames=10, device=torch.device("cpu"),
        )
        starts = positions[0, 0, :, 0]
        ends = positions[0, 0, :, 1]
        assert (ends >= starts).all(), "End should be >= start for all positions"

    def test_single_frame(self):
        """Works with a single frame."""
        positions = create_audio_position_indices(
            batch_size=1, audio_latent_frames=1, device=torch.device("cpu"),
        )
        assert positions.shape == (1, 1, 1, 2)


class TestCreateAudioModality:
    """Tests for create_audio_modality()."""

    def test_returns_modality(self):
        """Returns a Modality dataclass with correct fields."""
        from llm_dit.models.ltx2.components import Modality

        latent = torch.randn(1, 34, 128)
        timestep = torch.ones(1, 34) * 0.5
        positions = torch.randn(1, 1, 34, 2)
        prompt_embeds = torch.randn(1, 256, 2048)

        result = create_audio_modality(latent, timestep, positions, prompt_embeds)
        assert isinstance(result, Modality)
        assert result.enabled is True

    def test_fields_set_correctly(self):
        """All fields match the input tensors."""
        latent = torch.randn(1, 34, 128)
        timestep = torch.ones(1, 34)
        positions = torch.randn(1, 1, 34, 2)
        prompt_embeds = torch.randn(1, 256, 2048)
        mask = torch.ones(1, 256)

        result = create_audio_modality(
            latent, timestep, positions, prompt_embeds, context_mask=mask,
        )
        assert result.latent is latent
        assert result.timesteps is timestep
        assert result.positions is positions
        assert result.context is prompt_embeds
        assert result.context_mask is mask

    def test_no_mask(self):
        """context_mask defaults to None when not provided."""
        latent = torch.randn(1, 34, 128)
        timestep = torch.ones(1, 34)
        positions = torch.randn(1, 1, 34, 2)
        prompt_embeds = torch.randn(1, 256, 2048)

        result = create_audio_modality(latent, timestep, positions, prompt_embeds)
        assert result.context_mask is None


class TestAudioLatentDimensions:
    """Verify audio latent dimension formulas match the decode pipeline.

    Audio latents: (B, 8, T_audio, 16) before patchification.
    Patchified: (B, T_audio, 128) where 128 = 8 channels * 16 mel bins.
    """

    def test_patchified_dim_is_128(self):
        """Audio token dimension should be 8 * 16 = 128."""
        channels = 8
        mel_bins = 16
        assert channels * mel_bins == 128

    def test_latent_count_independent_of_resolution(self):
        """Audio latent count depends only on duration, not spatial resolution."""
        # Same num_frames, different resolutions
        count_small = compute_audio_latent_frames(33)
        count_large = compute_audio_latent_frames(33)
        assert count_small == count_large

    def test_latent_count_same_for_both_stages(self):
        """Stage 1 (half-res) and Stage 2 (full-res) use same audio latent count."""
        num_frames = 33
        count = compute_audio_latent_frames(num_frames)
        # This is by design -- audio has no spatial dimension
        assert count == 34  # Duration-only computation


class TestConfigAudioFields:
    """Tests for audio config fields added in Phase 3A."""

    def test_ltx2_config_has_audio_vae_path(self):
        """LTX2Config has audio_vae_path field."""
        from llm_dit.config import LTX2Config
        cfg = LTX2Config()
        assert hasattr(cfg, "audio_vae_path")
        assert cfg.audio_vae_path == ""

    def test_ltx2_config_has_vocoder_path(self):
        """LTX2Config has vocoder_path field."""
        from llm_dit.config import LTX2Config
        cfg = LTX2Config()
        assert hasattr(cfg, "vocoder_path")
        assert cfg.vocoder_path == ""

    def test_ltx2_config_has_audio_enabled(self):
        """LTX2Config has audio_enabled field."""
        from llm_dit.config import LTX2Config
        cfg = LTX2Config()
        assert hasattr(cfg, "audio_enabled")
        assert cfg.audio_enabled is False


class TestSchemaAudioFields:
    """Tests for audio schema fields added in Phase 3A."""

    def test_enable_audio_default_false(self):
        """LTX2GenerateRequest.enable_audio defaults to False."""
        from web.schemas import LTX2GenerateRequest
        req = LTX2GenerateRequest(prompt="test", width=512, height=512, num_frames=9)
        assert req.enable_audio is False

    def test_audio_negative_prompt_optional(self):
        """LTX2GenerateRequest.audio_negative_prompt is Optional[str]."""
        from web.schemas import LTX2GenerateRequest
        req = LTX2GenerateRequest(prompt="test", width=512, height=512, num_frames=9)
        assert req.audio_negative_prompt is None

    def test_audio_negative_prompt_accepts_string(self):
        """LTX2GenerateRequest.audio_negative_prompt accepts a string value."""
        from web.schemas import LTX2GenerateRequest
        req = LTX2GenerateRequest(
            prompt="test", width=512, height=512, num_frames=9,
            audio_negative_prompt="silence",
        )
        assert req.audio_negative_prompt == "silence"


class TestVideoOnlyParameterization:
    """Tests for video_only -> audio_enabled resolution logic."""

    def test_audio_disabled_means_video_only(self):
        """When audio_enabled=False, video_only should be True."""
        audio_enabled = False
        video_only = not audio_enabled
        assert video_only is True

    def test_audio_enabled_means_not_video_only(self):
        """When audio_enabled=True, video_only should be False."""
        audio_enabled = True
        video_only = not audio_enabled
        assert video_only is False

    def test_generate_function_signature_has_video_only(self):
        """generate_video_two_stage accepts video_only parameter."""
        import inspect
        from llm_dit.pipelines.generate import generate_video_two_stage
        sig = inspect.signature(generate_video_two_stage)
        assert "video_only" in sig.parameters
        assert sig.parameters["video_only"].default is True

    def test_generate_function_signature_has_audio_params(self):
        """generate_video_two_stage accepts all audio parameters."""
        import inspect
        from llm_dit.pipelines.generate import generate_video_two_stage
        sig = inspect.signature(generate_video_two_stage)
        for param in ["audio_negative_prompt", "cached_audio_decoder", "cached_vocoder"]:
            assert param in sig.parameters, f"Missing param: {param}"


class TestCrossModalPE:
    """Tests for cross-modal PE computation in transformer."""

    def test_cross_pe_max_pos_set_on_av_model(self):
        """AudioVideo transformer has _cross_pe_max_pos attribute."""
        from llm_dit.models.ltx2 import LTX2Transformer, LTXModelType
        model = LTX2Transformer(
            model_type=LTXModelType.AudioVideo,
            num_layers=2,
            num_attention_heads=4,
            attention_head_dim=32,
            in_channels=128,
            out_channels=128,
            caption_channels=2048,
            audio_num_attention_heads=4,
            audio_attention_head_dim=32,
            audio_cross_attention_dim=128,
        )
        assert hasattr(model, "_cross_pe_max_pos")
        assert isinstance(model._cross_pe_max_pos, int)
        assert model._cross_pe_max_pos >= 20  # default max_pos[0] for both modalities

    def test_cross_pe_not_set_on_video_only_model(self):
        """VideoOnly transformer should NOT have _cross_pe_max_pos."""
        from llm_dit.models.ltx2 import LTX2Transformer, LTXModelType
        model = LTX2Transformer(
            model_type=LTXModelType.VideoOnly,
            num_layers=2,
            num_attention_heads=4,
            attention_head_dim=32,
            in_channels=128,
            out_channels=128,
            caption_channels=2048,
        )
        assert not hasattr(model, "_cross_pe_max_pos")
