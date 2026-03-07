"""
Tests for AV velocity CFG, STG, and modality guidance.

Last Updated: 2026-03-07

Validates:
- _compute_av_velocity uses zeros (not positive embeds) when audio_neg_embeds is None
- STG pass creates both video + audio self-attention perturbations
- Modality guidance 4th pass runs when modality_scale > 1.0
- No guidance mode uses single pass

Run with: uv run pytest tests/unit/test_av_velocity_cfg.py -v
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

pytestmark = pytest.mark.unit


def _make_step_context(
    guidance_scale=3.0, audio_guidance_scale=0.0, neg_embeds=None,
    stg_scale=0.0, stg_blocks=None, modality_scale=1.0,
):
    """Build a minimal StepContext for testing."""
    from llm_dit.pipelines.generate import StepContext
    return StepContext(
        guidance_scale=guidance_scale,
        audio_guidance_scale=audio_guidance_scale,
        rescale_scale=0.0,
        ge_gamma=0.0,
        stg_scale=stg_scale,
        stg_blocks=stg_blocks,
        neg_embeds=neg_embeds,
        modality_scale=modality_scale,
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


class TestSTGPerturbationTypes:
    """Verify STG pass creates both video + audio self-attention perturbations."""

    def test_stg_creates_both_perturbation_types(self):
        """STG pass should skip both video AND audio self-attention."""
        from llm_dit.pipelines.generate import _compute_av_velocity
        from llm_dit.models.ltx2.transformer import PerturbationType

        B, T_v, D_v = 1, 16, 128
        T_a, D_a = 8, 128
        seq_len = 32

        neg_embeds = torch.randn(B, seq_len, 4096)
        ctx = _make_step_context(
            guidance_scale=3.0, neg_embeds=neg_embeds,
            stg_scale=1.0, stg_blocks=[28],
        )

        mock_model = MagicMock()
        mock_model.return_value = (
            torch.randn(B, T_v, D_v),
            torch.randn(B, T_a, D_a),
        )

        captured_perturb_configs = []
        original_call = mock_model.__call__

        def capture_call(*args, **kwargs):
            if "perturbation_config" in kwargs:
                captured_perturb_configs.append(kwargs["perturbation_config"])
            return (torch.randn(B, T_v, D_v), torch.randn(B, T_a, D_a))

        mock_model.side_effect = capture_call

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

        # STG should have sent a perturbation_config
        assert len(captured_perturb_configs) >= 1, "STG pass should create perturbation_config"
        stg_config = captured_perturb_configs[0]

        # Extract perturbation types from the config
        perturb_types = set()
        for pc in stg_config.perturbations:
            for p in pc.perturbations:
                perturb_types.add(p.type)

        assert PerturbationType.SKIP_VIDEO_SELF_ATTN in perturb_types, (
            "STG should include SKIP_VIDEO_SELF_ATTN perturbation"
        )
        assert PerturbationType.SKIP_AUDIO_SELF_ATTN in perturb_types, (
            "STG should include SKIP_AUDIO_SELF_ATTN perturbation"
        )


class TestModalityGuidance:
    """Verify 4th forward pass (modality guidance) behavior."""

    def test_modality_guidance_4th_pass_runs(self):
        """When modality_scale > 1.0, model should be called 4 times."""
        from llm_dit.pipelines.generate import _compute_av_velocity

        B, T_v, D_v = 1, 16, 128
        T_a, D_a = 8, 128
        seq_len = 32

        neg_embeds = torch.randn(B, seq_len, 4096)
        ctx = _make_step_context(
            guidance_scale=3.0, neg_embeds=neg_embeds,
            stg_scale=1.0, stg_blocks=[28],
            modality_scale=3.0,
        )

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

        # 4 passes: uncond, cond, STG perturbed, modality-isolated
        assert mock_model.call_count == 4, (
            f"Expected 4 model calls (uncond + cond + STG + modality), got {mock_model.call_count}"
        )

    def test_modality_guidance_skipped_at_1(self):
        """When modality_scale == 1.0, only 3 passes should run (no 4th)."""
        from llm_dit.pipelines.generate import _compute_av_velocity

        B, T_v, D_v = 1, 16, 128
        T_a, D_a = 8, 128
        seq_len = 32

        neg_embeds = torch.randn(B, seq_len, 4096)
        ctx = _make_step_context(
            guidance_scale=3.0, neg_embeds=neg_embeds,
            stg_scale=1.0, stg_blocks=[28],
            modality_scale=1.0,
        )

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

        # 3 passes: uncond, cond, STG perturbed (no modality at scale=1.0)
        assert mock_model.call_count == 3, (
            f"Expected 3 model calls (uncond + cond + STG), got {mock_model.call_count}"
        )

    def test_modality_guidance_uses_cross_attn_perturbations(self):
        """4th pass should use SKIP_A2V_CROSS_ATTN + SKIP_V2A_CROSS_ATTN."""
        from llm_dit.pipelines.generate import _compute_av_velocity
        from llm_dit.models.ltx2.transformer import PerturbationType

        B, T_v, D_v = 1, 16, 128
        T_a, D_a = 8, 128
        seq_len = 32

        neg_embeds = torch.randn(B, seq_len, 4096)
        # No STG, just modality guidance to isolate the 4th pass
        ctx = _make_step_context(
            guidance_scale=3.0, neg_embeds=neg_embeds,
            stg_scale=0.0, stg_blocks=None,
            modality_scale=3.0,
        )

        captured_perturb_configs = []

        def capture_call(*args, **kwargs):
            if "perturbation_config" in kwargs:
                captured_perturb_configs.append(kwargs["perturbation_config"])
            return (torch.randn(B, T_v, D_v), torch.randn(B, T_a, D_a))

        mock_model = MagicMock(side_effect=capture_call)

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

        # 3 passes: uncond (no perturb), cond (no perturb), modality (perturbed)
        assert mock_model.call_count == 3
        assert len(captured_perturb_configs) == 1, "Modality pass should send perturbation_config"

        mod_config = captured_perturb_configs[0]
        perturb_types = set()
        for pc in mod_config.perturbations:
            for p in pc.perturbations:
                perturb_types.add(p.type)

        assert PerturbationType.SKIP_A2V_CROSS_ATTN in perturb_types, (
            "Modality guidance should skip audio-to-video cross attention"
        )
        assert PerturbationType.SKIP_V2A_CROSS_ATTN in perturb_types, (
            "Modality guidance should skip video-to-audio cross attention"
        )

        # blocks=None means "all blocks" -- architecture-independent
        for pc in mod_config.perturbations:
            for p in pc.perturbations:
                assert p.blocks is None, (
                    f"Modality perturbation {p.type} should use blocks=None (all blocks), "
                    f"not an explicit list: {p.blocks}"
                )

    def test_no_guidance_no_stg_single_pass(self):
        """guidance=1.0, stg=0, modality=1.0 -> 1 model call."""
        from llm_dit.pipelines.generate import _compute_av_velocity

        B, T_v, D_v = 1, 16, 128
        T_a, D_a = 8, 128
        seq_len = 32

        ctx = _make_step_context(
            guidance_scale=1.0, neg_embeds=None,
            stg_scale=0.0, stg_blocks=None,
            modality_scale=1.0,
        )

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

        assert mock_model.call_count == 1, (
            f"Expected 1 model call (no guidance, no STG), got {mock_model.call_count}"
        )
