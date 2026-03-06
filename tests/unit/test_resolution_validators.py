"""
Resolution validator tests for Pydantic request schemas and dataclass configs.

last updated: 2026-02-07

Tests the @field_validator snapping on Flux2GenerateRequest (mod 16),
LTX2GenerateRequest (mod 64), and __post_init__ snapping on
Flux2GenerationConfig. Pydantic Field(ge=, le=) constraints run BEFORE
field_validators, so out-of-range inputs are rejected by Pydantic, not
snapped.

Run with: uv run pytest tests/unit/test_resolution_validators.py -v
"""

import pytest
from pydantic import ValidationError

from web.schemas import Flux2GenerateRequest, LTX2GenerateRequest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# FLUX.2 Pydantic request -- snap to 16 (within 256-2048 range)
# ---------------------------------------------------------------------------

class TestFlux2RequestSnapping:
    """Tests for Flux2GenerateRequest width/height snap_to_16 validator.

    Note: Pydantic Field(ge=256, le=2048) validates BEFORE @field_validator,
    so out-of-range values raise ValidationError, never reaching snap_to_16.
    Python round() uses banker's rounding (half-to-even).
    """

    @pytest.mark.parametrize("value", [256, 512, 1024, 1280, 2048])
    def test_valid_multiples_unchanged(self, value: int):
        """Values already multiples of 16 should pass through unchanged."""
        req = Flux2GenerateRequest(prompt="test", width=value, height=value)
        assert req.width == value
        assert req.height == value

    @pytest.mark.parametrize("input_val,expected", [
        # Python round() uses banker's rounding (half-to-even):
        # 1000/16=62.5, round(62.5)=62 (even), 62*16=992
        (1000, 992),
        (1001, 1008),   # 1001/16=62.5625, round=63, 63*16=1008
        (1007, 1008),   # 1007/16=62.9375, round=63, 63*16=1008
        (1009, 1008),   # 1009/16=63.0625, round=63, 63*16=1008
        (1015, 1008),   # 1015/16=63.4375, round=63, 63*16=1008
        (1017, 1024),   # 1017/16=63.5625, round=64, 64*16=1024
        (500, 496),     # 500/16=31.25, round=31, 31*16=496
        (508, 512),     # 508/16=31.75, round=32, 32*16=512
    ])
    def test_snap_to_nearest_16(self, input_val: int, expected: int):
        """Non-multiples of 16 should snap to nearest multiple."""
        req = Flux2GenerateRequest(prompt="test", width=input_val, height=1024)
        assert req.width == expected

    def test_below_min_rejected(self):
        """Values below 256 are rejected by Pydantic Field(ge=256)."""
        with pytest.raises(ValidationError):
            Flux2GenerateRequest(prompt="test", width=100, height=1024)

    def test_above_max_rejected(self):
        """Values above 2048 are rejected by Pydantic Field(le=2048)."""
        with pytest.raises(ValidationError):
            Flux2GenerateRequest(prompt="test", width=3000, height=1024)

    def test_exact_boundaries(self):
        """Exact min/max boundaries should work."""
        req_min = Flux2GenerateRequest(prompt="test", width=256, height=256)
        assert req_min.width == 256
        req_max = Flux2GenerateRequest(prompt="test", width=2048, height=2048)
        assert req_max.width == 2048

    def test_near_min_snaps_to_min(self):
        """Value just above min that rounds down should clamp to 256."""
        # 257/16=16.0625, round=16, 16*16=256, max(256, 256)=256
        req = Flux2GenerateRequest(prompt="test", width=257, height=1024)
        assert req.width == 256

    def test_near_max_snaps_to_max(self):
        """Value just below max that rounds up should clamp to 2048."""
        # 2047/16=127.9375, round=128, 128*16=2048
        req = Flux2GenerateRequest(prompt="test", width=2047, height=1024)
        assert req.width == 2048


# ---------------------------------------------------------------------------
# LTX-2 Pydantic request -- snap to 64 (within 256-1536 / 256-1024 range)
# ---------------------------------------------------------------------------

class TestLTX2RequestSnapping:
    """Tests for LTX2GenerateRequest width/height snap_to_64 validator."""

    @pytest.mark.parametrize("value", [256, 512, 768, 1024])
    def test_valid_multiples_unchanged(self, value: int):
        """Values already multiples of 64 should pass through unchanged."""
        req = LTX2GenerateRequest(prompt="test", width=value, height=value)
        assert req.width == value
        assert req.height == value

    @pytest.mark.parametrize("input_val,expected", [
        # 1000/64=15.625, round=16 (banker's: half-to-even), 16*64=1024
        (1000, 1024),
        # 1017/64=15.890625, round=16, 16*64=1024
        (1017, 1024),
        # 500/64=7.8125, round=8 (banker's: half-to-even), 8*64=512
        (500, 512),
        # 780/64=12.1875, round=12, 12*64=768
        (780, 768),
    ])
    def test_snap_to_nearest_64(self, input_val: int, expected: int):
        """Non-multiples of 64 should snap to nearest multiple."""
        req = LTX2GenerateRequest(prompt="test", width=input_val, height=512)
        assert req.width == expected

    def test_below_min_rejected(self):
        """Values below 256 are rejected by Pydantic Field(ge=256)."""
        with pytest.raises(ValidationError):
            LTX2GenerateRequest(prompt="test", width=100, height=512)

    def test_above_max_rejected(self):
        """Values above 1536 (width) / 1024 (height) are rejected."""
        with pytest.raises(ValidationError):
            LTX2GenerateRequest(prompt="test", width=2000, height=512)


# ---------------------------------------------------------------------------
# FLUX.2 Generation Config (dataclass) -- snap to 16
# ---------------------------------------------------------------------------

class TestFlux2GenerationConfigSnapping:
    """Tests for Flux2GenerationConfig.__post_init__ width/height snapping.

    The dataclass has no min/max constraints, so all values pass through
    to __post_init__ where round() is applied with banker's rounding.
    """

    def test_valid_no_snap(self):
        """Multiples of 16 should not be modified."""
        from llm_dit.pipelines.flux2_generate import Flux2GenerationConfig
        config = Flux2GenerationConfig(prompt="test", width=1024, height=768)
        assert config.width == 1024
        assert config.height == 768

    @pytest.mark.parametrize("input_val,expected", [
        # 1000/16=62.5, round=62 (banker's), 62*16=992
        (1000, 992),
        # 500/16=31.25, round=31, 31*16=496
        (500, 496),
        # 1017/16=63.5625, round=64, 64*16=1024
        (1017, 1024),
    ])
    def test_snap_with_warning(self, input_val: int, expected: int):
        """Non-multiples of 16 should be snapped in __post_init__."""
        from llm_dit.pipelines.flux2_generate import Flux2GenerationConfig
        config = Flux2GenerationConfig(prompt="test", width=input_val, height=1024)
        assert config.width == expected

    def test_latent_dimensions_after_snap(self):
        """Latent dimensions should be correct after snapping."""
        from llm_dit.pipelines.flux2_generate import Flux2GenerationConfig
        # 1000 -> 992 (banker's round), 992/16 = 62
        config = Flux2GenerationConfig(prompt="test", width=1000, height=1000)
        assert config.latent_width == 62
        assert config.latent_height == 62
        assert config.num_tokens == 62 * 62


# ---------------------------------------------------------------------------
# Cross-pipeline consistency
# ---------------------------------------------------------------------------

class TestCrossPipelineConsistency:
    """Verify all request types produce valid multiples for the same input."""

    @pytest.mark.parametrize("input_val", [999, 1001, 777, 513, 300, 1900])
    def test_flux2_always_produces_mod_16(self, input_val: int):
        """Flux2 output should always be a multiple of 16."""
        req = Flux2GenerateRequest(prompt="test", width=input_val, height=input_val)
        assert req.width % 16 == 0, f"width {req.width} not multiple of 16"
        assert req.height % 16 == 0, f"height {req.height} not multiple of 16"

    @pytest.mark.parametrize("input_val", [999, 777, 513, 300])
    def test_ltx2_always_produces_mod_64(self, input_val: int):
        """LTX-2 output should always be a multiple of 64."""
        req = LTX2GenerateRequest(prompt="test", width=input_val, height=input_val)
        assert req.width % 64 == 0, f"width {req.width} not multiple of 64"
        assert req.height % 64 == 0, f"height {req.height} not multiple of 64"

    @pytest.mark.parametrize("input_val", [999, 1001, 777, 513, 100, 3000])
    def test_flux2_config_always_produces_mod_16(self, input_val: int):
        """Flux2GenerationConfig should always produce multiples of 16."""
        from llm_dit.pipelines.flux2_generate import Flux2GenerationConfig
        config = Flux2GenerationConfig(prompt="test", width=input_val, height=input_val)
        assert config.width % 16 == 0, f"width {config.width} not multiple of 16"
        assert config.height % 16 == 0, f"height {config.height} not multiple of 16"
