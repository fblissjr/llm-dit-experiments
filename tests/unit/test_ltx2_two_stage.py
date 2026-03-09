"""
Tests for LTX-2 two-stage pipeline configuration and components.

Last Updated: 2026-02-10

Run with: uv run pytest tests/unit/test_ltx2_two_stage.py -v
"""

import torch

from llm_dit.models.ltx2.constants import (
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
    TWO_STAGE_STEPS_STAGE2,
)
from llm_dit.pipelines.generate import (
    GenerationConfig,
    StepContext,
    TwoStageConfig,
    constant_schedule,
    create_position_indices,
)


class TestTwoStageConfig:
    """Tests for TwoStageConfig dataclass."""

    def test_default_values(self):
        cfg = TwoStageConfig()
        assert cfg.stage1_steps == 30  # V2.3 default (was 40 in V2.0)
        assert cfg.pipeline_mode == "standard"
        assert cfg.guidance_scale == 3.0  # ref: DEFAULT_VIDEO_GUIDER_PARAMS.cfg_scale
        assert cfg.stg_scale == 1.0  # ref: DEFAULT_VIDEO_GUIDER_PARAMS.stg_scale
        assert cfg.rescale_scale == 0.7
        assert cfg.ge_gamma == 0.0
        assert cfg.distilled_lora_scale == 1.0  # ref: distilled_lora_scale default

    def test_stg_blocks_default(self):
        """STG blocks should default to [28] via __post_init__."""
        cfg = TwoStageConfig()
        assert cfg.stg_blocks == [28]

    def test_custom_stg_blocks(self):
        cfg = TwoStageConfig(stg_blocks=[10, 20, 29])
        assert cfg.stg_blocks == [10, 20, 29]

    def test_negative_prompt_default(self):
        cfg = TwoStageConfig()
        assert "blurry" in cfg.negative_prompt
        assert "AI artifacts" in cfg.negative_prompt


class TestDistilledSigmaSchedules:
    """Tests for distilled sigma constant correctness."""

    def test_distilled_sigma_monotonically_decreasing(self):
        """Full distilled schedule should be monotonically decreasing."""
        for i in range(len(DISTILLED_SIGMA_VALUES) - 1):
            assert DISTILLED_SIGMA_VALUES[i] > DISTILLED_SIGMA_VALUES[i + 1], (
                f"Non-monotonic at index {i}: {DISTILLED_SIGMA_VALUES[i]} <= {DISTILLED_SIGMA_VALUES[i + 1]}"
            )

    def test_stage2_sigma_monotonically_decreasing(self):
        """Stage 2 distilled schedule should be monotonically decreasing."""
        for i in range(len(STAGE_2_DISTILLED_SIGMA_VALUES) - 1):
            assert STAGE_2_DISTILLED_SIGMA_VALUES[i] > STAGE_2_DISTILLED_SIGMA_VALUES[i + 1]

    def test_distilled_sigma_bounds(self):
        """Distilled sigmas should be in [0, 1]."""
        assert DISTILLED_SIGMA_VALUES[0] == 1.0
        assert DISTILLED_SIGMA_VALUES[-1] == 0.0

    def test_stage2_sigma_ends_at_zero(self):
        assert STAGE_2_DISTILLED_SIGMA_VALUES[-1] == 0.0

    def test_stage2_steps_match_sigma_count(self):
        """Stage 2 steps should be len(sigmas) - 1."""
        assert TWO_STAGE_STEPS_STAGE2 == len(STAGE_2_DISTILLED_SIGMA_VALUES) - 1

    def test_stage2_sigmas_are_subset_of_full(self):
        """Stage 2 sigmas should be a subset of the full distilled schedule."""
        for sigma in STAGE_2_DISTILLED_SIGMA_VALUES:
            assert sigma in DISTILLED_SIGMA_VALUES, (
                f"Stage 2 sigma {sigma} not found in full distilled schedule"
            )

    def test_stage2_starts_below_one(self):
        """Stage 2 should start at a partial noise level (not full noise)."""
        assert STAGE_2_DISTILLED_SIGMA_VALUES[0] < 1.0
        assert STAGE_2_DISTILLED_SIGMA_VALUES[0] == 0.909375


class TestHalfResolution:
    """Tests for stage 1 half-resolution computation."""

    def test_half_resolution_latent_dims(self):
        """Stage 1 uses half resolution -- latent dims should be halved."""
        full_config = GenerationConfig(height=512, width=768)
        half_config = GenerationConfig(height=256, width=384)

        t_full, h_full, w_full = full_config.latent_dims
        t_half, h_half, w_half = half_config.latent_dims

        assert h_half == h_full // 2
        assert w_half == w_full // 2
        # Temporal dimension stays the same (same num_frames)
        assert t_half == t_full

    def test_position_indices_half_resolution(self):
        """Position indices at half resolution should have correct shape."""
        positions = create_position_indices(
            batch_size=1,
            num_frames=33,
            height=256,  # half of 512
            width=384,  # half of 768
            device=torch.device("cpu"),
        )
        t_latent = (33 - 1) // 8 + 1  # 5
        h_latent = 256 // 32  # 8
        w_latent = 384 // 32  # 12
        num_tokens = t_latent * h_latent * w_latent  # 480

        assert positions.shape == (1, 3, num_tokens, 2)

    def test_two_stage_resolution_must_be_64_divisible(self):
        """Two-stage pipeline requires output resolution divisible by 64.

        This is because stage 1 uses height/2 and width/2, and the VAE
        requires dimensions divisible by 32. So the full resolution must
        be divisible by 64.
        """
        # 768 / 2 = 384 -> 384 / 32 = 12 (valid)
        # 512 / 2 = 256 -> 256 / 32 = 8 (valid)
        assert 768 % 64 == 0
        assert 512 % 64 == 0

        # 1536 / 2 = 768 -> 768 / 32 = 24 (valid)
        # 1024 / 2 = 512 -> 512 / 32 = 16 (valid)
        assert 1536 % 64 == 0
        assert 1024 % 64 == 0


class TestGenerationConfigLatents:
    """Tests for GenerationConfig latent dimension computation."""

    def test_latent_dims_standard(self):
        cfg = GenerationConfig(num_frames=33, height=512, width=768)
        t, h, w = cfg.latent_dims
        assert t == 5  # (33-1)//8 + 1
        assert h == 16  # 512//32
        assert w == 24  # 768//32

    def test_latent_dims_half(self):
        cfg = GenerationConfig(num_frames=33, height=256, width=384)
        t, h, w = cfg.latent_dims
        assert t == 5
        assert h == 8
        assert w == 12

    def test_num_tokens(self):
        cfg = GenerationConfig(num_frames=33, height=512, width=768)
        assert cfg.num_tokens == 5 * 16 * 24  # 1920


class TestStepContext:
    """Tests for StepContext dataclass."""

    def test_defaults(self):
        ctx = StepContext()
        assert ctx.guidance_scale == 1.0
        assert ctx.neg_embeds is None
        assert ctx.rescale_scale == 0.0
        assert ctx.ge_gamma == 0.0
        assert ctx.stg_scale == 0.0
        assert ctx.stg_blocks is None

    def test_custom_values(self):
        neg = torch.zeros(1, 10, 64)
        ctx = StepContext(
            guidance_scale=3.5,
            neg_embeds=neg,
            rescale_scale=0.7,
            ge_gamma=2.0,
            stg_scale=1.0,
            stg_blocks=[29],
        )
        assert ctx.guidance_scale == 3.5
        assert ctx.neg_embeds is neg
        assert ctx.rescale_scale == 0.7
        assert ctx.ge_gamma == 2.0
        assert ctx.stg_scale == 1.0
        assert ctx.stg_blocks == [29]

    def test_no_cfg_when_scale_one(self):
        """guidance_scale=1.0 means no CFG regardless of neg_embeds."""
        ctx = StepContext(guidance_scale=1.0, neg_embeds=torch.zeros(1, 10, 64))
        assert ctx.guidance_scale <= 1.0

    def test_stg_disabled_when_scale_zero(self):
        """stg_scale=0.0 means no STG even with stg_blocks set."""
        ctx = StepContext(stg_scale=0.0, stg_blocks=[29])
        assert ctx.stg_scale == 0.0


class TestStepSchedule:
    """Tests for StepSchedule callable protocol and constant_schedule factory."""

    def test_constant_schedule_returns_same_context(self):
        """constant_schedule should return the same StepContext for all steps."""
        schedule = constant_schedule(guidance_scale=3.5, rescale_scale=0.7)
        ctx0 = schedule(0, 1.0)
        ctx5 = schedule(5, 0.5)
        ctx39 = schedule(39, 0.01)
        assert ctx0 is ctx5
        assert ctx5 is ctx39

    def test_constant_schedule_default_is_simple(self):
        """Default constant_schedule should produce no-CFG context."""
        schedule = constant_schedule()
        ctx = schedule(0, 1.0)
        assert ctx.guidance_scale == 1.0
        assert ctx.neg_embeds is None

    def test_constant_schedule_with_neg_embeds(self):
        neg = torch.randn(1, 10, 64)
        schedule = constant_schedule(guidance_scale=4.0, neg_embeds=neg)
        ctx = schedule(0, 1.0)
        assert ctx.guidance_scale == 4.0
        assert ctx.neg_embeds is neg

    def test_custom_callable_schedule(self):
        """A custom callable that varies params by sigma phase."""
        def phased(step: int, sigma: float) -> StepContext:
            if sigma > 0.5:
                return StepContext(guidance_scale=4.0, rescale_scale=0.7)
            return StepContext(guidance_scale=2.0, rescale_scale=0.3)

        high = phased(0, 0.9)
        low = phased(30, 0.2)
        assert high.guidance_scale == 4.0
        assert low.guidance_scale == 2.0
        assert high.rescale_scale == 0.7
        assert low.rescale_scale == 0.3

    def test_constant_schedule_with_stg(self):
        """constant_schedule should pass STG fields through."""
        schedule = constant_schedule(
            guidance_scale=3.0,
            stg_scale=1.0,
            stg_blocks=[29],
        )
        ctx = schedule(0, 1.0)
        assert ctx.stg_scale == 1.0
        assert ctx.stg_blocks == [29]

    def test_schedule_from_two_stage_config(self):
        """constant_schedule should correctly capture TwoStageConfig values."""
        cfg = TwoStageConfig(guidance_scale=3.5, rescale_scale=0.7, ge_gamma=2.0)
        schedule = constant_schedule(
            guidance_scale=cfg.guidance_scale,
            rescale_scale=cfg.rescale_scale,
            ge_gamma=cfg.ge_gamma,
            stg_scale=cfg.stg_scale,
            stg_blocks=cfg.stg_blocks,
        )
        ctx = schedule(0, 1.0)
        assert ctx.guidance_scale == 3.5
        assert ctx.rescale_scale == 0.7
        assert ctx.ge_gamma == 2.0
        assert ctx.stg_scale == 1.0  # TwoStageConfig default
        assert ctx.stg_blocks == [28]  # TwoStageConfig default
