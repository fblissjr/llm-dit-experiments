"""
Tests for AV velocity CFG audio negative embeddings handling.

Last Updated: 2026-03-07

Validates that _compute_av_velocity uses zeros (not positive embeds)
when audio_neg_embeds is None, preventing CFG contamination.

Run with: uv run pytest tests/unit/test_av_velocity_cfg.py -v
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

pytestmark = pytest.mark.unit


def _make_step_context(guidance_scale=3.0, audio_guidance_scale=0.0, neg_embeds=None):
    """Build a minimal StepContext for testing."""
    from llm_dit.pipelines.generate import StepContext
    return StepContext(
        guidance_scale=guidance_scale,
        audio_guidance_scale=audio_guidance_scale,
        rescale_scale=0.0,
        ge_gamma=0.0,
        stg_scale=0.0,
        stg_blocks=None,
        neg_embeds=neg_embeds,
    )


class TestAVVelocityCFGFallback:
    """Verify _compute_av_velocity audio negative embed fallback uses zeros."""

    def test_none_audio_neg_embeds_falls_back_to_zeros(self):
        """When audio_neg_embeds is None, unconditional audio pass should use
        zeros, not positive audio embeddings."""
        from llm_dit.pipelines.generate import _compute_av_velocity

        B, T_v, D_v = 1, 16, 128
        T_a, D_a = 8, 128
        seq_len, ctx_dim = 32, 2048

        video_latents = torch.randn(B, T_v, D_v)
        video_timestep = torch.full((B, T_v), 0.5)
        video_positions = torch.zeros(B, 3, T_v, 2)
        video_prompt_embeds = torch.randn(B, seq_len, 4096)
        audio_latents = torch.randn(B, T_a, D_a)
        audio_timestep = torch.full((B, T_a), 0.5)
        audio_positions = torch.zeros(B, 1, T_a, 2)
        audio_prompt_embeds = torch.randn(B, seq_len, ctx_dim)
        neg_embeds = torch.randn(B, seq_len, 4096)

        ctx = _make_step_context(guidance_scale=3.0, neg_embeds=neg_embeds)

        # Mock the model to capture what modalities are passed
        mock_model = MagicMock()
        mock_model.return_value = (
            torch.randn(B, T_v, D_v),  # video vel
            torch.randn(B, T_a, D_a),  # audio vel
        )

        # Patch create_audio_modality to capture the prompt_embeds argument
        captured_audio_contexts = []

        def capture_create_audio(latent, timestep, positions, prompt_embeds, **kwargs):
            captured_audio_contexts.append(prompt_embeds.clone())
            from llm_dit.pipelines.generate import Modality
            sigma = timestep[:, 0]
            return Modality(
                latent=latent,
                sigma=sigma,
                timesteps=timestep,
                positions=positions,
                context=prompt_embeds,
                enabled=True,
                context_mask=kwargs.get("context_mask"),
            )

        with patch(
            "llm_dit.pipelines.generate.create_audio_modality",
            side_effect=capture_create_audio,
        ):
            _compute_av_velocity(
                model=mock_model,
                video_latents=video_latents,
                video_timestep=video_timestep,
                video_positions=video_positions,
                video_prompt_embeds=video_prompt_embeds,
                audio_latents=audio_latents,
                audio_timestep=audio_timestep,
                audio_positions=audio_positions,
                audio_prompt_embeds=audio_prompt_embeds,
                ctx=ctx,
                audio_neg_embeds=None,  # The bug condition
            )

        # First call is unconditional pass, second is conditional pass
        assert len(captured_audio_contexts) >= 2
        uncond_audio_ctx = captured_audio_contexts[0]

        # Key assertion: unconditional audio context should be zeros, NOT positive embeds
        assert torch.all(uncond_audio_ctx == 0), (
            "Unconditional audio pass should use zero embeddings when audio_neg_embeds is None, "
            "not positive audio_prompt_embeds (which makes CFG gradient zero)"
        )

        # Sanity: conditional audio context should be the actual positive embeds
        cond_audio_ctx = captured_audio_contexts[1]
        assert torch.allclose(cond_audio_ctx, audio_prompt_embeds), (
            "Conditional audio pass should use actual positive audio embeddings"
        )

    def test_provided_audio_neg_embeds_used_directly(self):
        """When audio_neg_embeds is provided, it should be used as-is."""
        from llm_dit.pipelines.generate import _compute_av_velocity

        B, T_v, D_v = 1, 16, 128
        T_a, D_a = 8, 128
        seq_len = 32

        video_latents = torch.randn(B, T_v, D_v)
        video_timestep = torch.full((B, T_v), 0.5)
        video_positions = torch.zeros(B, 3, T_v, 2)
        video_prompt_embeds = torch.randn(B, seq_len, 4096)
        audio_latents = torch.randn(B, T_a, D_a)
        audio_timestep = torch.full((B, T_a), 0.5)
        audio_positions = torch.zeros(B, 1, T_a, 2)
        audio_prompt_embeds = torch.randn(B, seq_len, 2048)
        neg_embeds = torch.randn(B, seq_len, 4096)
        audio_neg_embeds = torch.randn(B, seq_len, 2048)  # Explicit neg audio

        ctx = _make_step_context(guidance_scale=3.0, neg_embeds=neg_embeds)

        mock_model = MagicMock()
        mock_model.return_value = (
            torch.randn(B, T_v, D_v),
            torch.randn(B, T_a, D_a),
        )

        captured_audio_contexts = []

        def capture_create_audio(latent, timestep, positions, prompt_embeds, **kwargs):
            captured_audio_contexts.append(prompt_embeds.clone())
            from llm_dit.pipelines.generate import Modality
            sigma = timestep[:, 0]
            return Modality(
                latent=latent,
                sigma=sigma,
                timesteps=timestep,
                positions=positions,
                context=prompt_embeds,
                enabled=True,
                context_mask=kwargs.get("context_mask"),
            )

        with patch(
            "llm_dit.pipelines.generate.create_audio_modality",
            side_effect=capture_create_audio,
        ):
            _compute_av_velocity(
                model=mock_model,
                video_latents=video_latents,
                video_timestep=video_timestep,
                video_positions=video_positions,
                video_prompt_embeds=video_prompt_embeds,
                audio_latents=audio_latents,
                audio_timestep=audio_timestep,
                audio_positions=audio_positions,
                audio_prompt_embeds=audio_prompt_embeds,
                ctx=ctx,
                audio_neg_embeds=audio_neg_embeds,
            )

        uncond_audio_ctx = captured_audio_contexts[0]
        assert torch.allclose(uncond_audio_ctx, audio_neg_embeds), (
            "Unconditional audio pass should use provided audio_neg_embeds"
        )

    def test_no_guidance_skips_uncond_pass(self):
        """When guidance_scale <= 1.0, no unconditional pass should run."""
        from llm_dit.pipelines.generate import _compute_av_velocity

        B, T_v, D_v = 1, 16, 128
        T_a, D_a = 8, 128
        seq_len = 32

        ctx = _make_step_context(guidance_scale=1.0, neg_embeds=None)

        mock_model = MagicMock()
        mock_model.return_value = (
            torch.randn(B, T_v, D_v),
            torch.randn(B, T_a, D_a),
        )

        _compute_av_velocity(
            model=mock_model,
            video_latents=torch.randn(B, T_v, D_v),
            video_timestep=torch.full((B, T_v), 0.5),
            video_positions=torch.zeros(B, 3, T_v, 2),
            video_prompt_embeds=torch.randn(B, seq_len, 4096),
            audio_latents=torch.randn(B, T_a, D_a),
            audio_timestep=torch.full((B, T_a), 0.5),
            audio_positions=torch.zeros(B, 1, T_a, 2),
            audio_prompt_embeds=torch.randn(B, seq_len, 2048),
            ctx=ctx,
            audio_neg_embeds=None,
        )

        # Only 1 model call (no uncond pass)
        assert mock_model.call_count == 1
