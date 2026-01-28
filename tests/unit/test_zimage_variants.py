"""
Unit tests for Z-Image variant detection and configuration.

last updated: 2026-01-27

Tests:
- ZImageConfig dataclass creation and defaults
- detect_zimage_variant() from scheduler_config.json
- get_variant_defaults() returns correct parameters
- ZIMAGE_VARIANTS dictionary structure
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from llm_dit.config import ZImageConfig, Config
from llm_dit.models.zimage.constants import (
    ZIMAGE_VARIANTS,
    detect_zimage_variant,
    get_variant_defaults,
    get_variant_description,
)

pytestmark = pytest.mark.unit


class TestZImageVariants:
    """Test ZIMAGE_VARIANTS dictionary structure."""

    def test_variants_has_turbo_and_base(self):
        """Both turbo and base variants should be defined."""
        assert "turbo" in ZIMAGE_VARIANTS
        assert "base" in ZIMAGE_VARIANTS

    def test_turbo_variant_values(self):
        """Turbo variant has correct default values."""
        turbo = ZIMAGE_VARIANTS["turbo"]
        assert turbo["shift"] == 3.0
        assert turbo["distilled"] is True
        assert turbo["defaults"]["num_inference_steps"] == 9
        assert turbo["defaults"]["guidance_scale"] == 0.0
        assert turbo["defaults"]["negative_prompt"] is None

    def test_base_variant_values(self):
        """Base variant has correct default values."""
        base = ZIMAGE_VARIANTS["base"]
        assert base["shift"] == 6.0
        assert base["distilled"] is False
        assert base["defaults"]["num_inference_steps"] == 35
        assert base["defaults"]["guidance_scale"] == 4.0
        assert base["defaults"]["negative_prompt"] == ""  # Empty but not None

    def test_variants_have_descriptions(self):
        """Both variants have human-readable descriptions."""
        for variant_name, variant in ZIMAGE_VARIANTS.items():
            assert "description" in variant
            assert len(variant["description"]) > 10


class TestGetVariantDefaults:
    """Test get_variant_defaults() function."""

    def test_turbo_defaults(self):
        """Get turbo variant defaults."""
        defaults = get_variant_defaults("turbo")
        assert defaults["num_inference_steps"] == 9
        assert defaults["guidance_scale"] == 0.0
        assert defaults["shift"] == 3.0
        assert defaults["distilled"] is True
        assert defaults["negative_prompt"] is None

    def test_base_defaults(self):
        """Get base variant defaults."""
        defaults = get_variant_defaults("base")
        assert defaults["num_inference_steps"] == 35
        assert defaults["guidance_scale"] == 4.0
        assert defaults["shift"] == 6.0
        assert defaults["distilled"] is False
        assert defaults["negative_prompt"] == ""

    def test_case_insensitive(self):
        """Variant names should be case-insensitive."""
        turbo_lower = get_variant_defaults("turbo")
        turbo_upper = get_variant_defaults("TURBO")
        turbo_mixed = get_variant_defaults("Turbo")

        assert turbo_lower == turbo_upper == turbo_mixed

    def test_unknown_variant_raises(self):
        """Unknown variant should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_variant_defaults("unknown")
        assert "unknown" in str(exc_info.value).lower()
        assert "turbo" in str(exc_info.value) or "base" in str(exc_info.value)


class TestDetectZImageVariant:
    """Test detect_zimage_variant() function."""

    def test_detect_turbo_from_shift_3(self, tmp_path):
        """Detect turbo variant from shift=3.0 in scheduler_config.json."""
        scheduler_dir = tmp_path / "scheduler"
        scheduler_dir.mkdir()
        config_path = scheduler_dir / "scheduler_config.json"
        config_path.write_text(json.dumps({"shift": 3.0}))

        variant = detect_zimage_variant(str(tmp_path))
        assert variant == "turbo"

    def test_detect_base_from_shift_6(self, tmp_path):
        """Detect base variant from shift=6.0 in scheduler_config.json."""
        scheduler_dir = tmp_path / "scheduler"
        scheduler_dir.mkdir()
        config_path = scheduler_dir / "scheduler_config.json"
        config_path.write_text(json.dumps({"shift": 6.0}))

        variant = detect_zimage_variant(str(tmp_path))
        assert variant == "base"

    def test_detect_base_from_shift_5_5_plus(self, tmp_path):
        """Shift >= 5.5 should detect as base (tolerance)."""
        scheduler_dir = tmp_path / "scheduler"
        scheduler_dir.mkdir()
        config_path = scheduler_dir / "scheduler_config.json"
        config_path.write_text(json.dumps({"shift": 5.5}))

        variant = detect_zimage_variant(str(tmp_path))
        assert variant == "base"

    def test_fallback_to_name_turbo(self, tmp_path):
        """Fallback to name-based detection for turbo."""
        turbo_path = tmp_path / "Z-Image-Turbo"
        turbo_path.mkdir()

        variant = detect_zimage_variant(str(turbo_path))
        assert variant == "turbo"

    def test_fallback_to_name_turbo_lowercase(self, tmp_path):
        """Fallback should be case-insensitive."""
        turbo_path = tmp_path / "z-image-turbo-v2"
        turbo_path.mkdir()

        variant = detect_zimage_variant(str(turbo_path))
        assert variant == "turbo"

    def test_fallback_default_turbo(self, tmp_path):
        """Fallback to turbo when no scheduler_config and no turbo in name."""
        base_path = tmp_path / "my_model"
        base_path.mkdir()

        variant = detect_zimage_variant(str(base_path))
        assert variant == "turbo"  # Conservative default

    def test_scheduler_config_in_root(self, tmp_path):
        """Also check for scheduler_config.json in model root."""
        config_path = tmp_path / "scheduler_config.json"
        config_path.write_text(json.dumps({"shift": 6.0}))

        variant = detect_zimage_variant(str(tmp_path))
        assert variant == "base"

    def test_invalid_json(self, tmp_path):
        """Invalid JSON should fall back to name-based detection."""
        scheduler_dir = tmp_path / "scheduler"
        scheduler_dir.mkdir()
        config_path = scheduler_dir / "scheduler_config.json"
        config_path.write_text("not valid json")

        variant = detect_zimage_variant(str(tmp_path))
        assert variant == "turbo"  # Fallback


class TestGetVariantDescription:
    """Test get_variant_description() function."""

    def test_turbo_description(self):
        desc = get_variant_description("turbo")
        assert "turbo" in desc.lower() or "fast" in desc.lower() or "distilled" in desc.lower()

    def test_base_description(self):
        desc = get_variant_description("base")
        assert "base" in desc.lower() or "quality" in desc.lower()

    def test_unknown_variant(self):
        desc = get_variant_description("unknown")
        assert "unknown" in desc.lower()


class TestZImageConfig:
    """Test ZImageConfig dataclass."""

    def test_default_values(self):
        """ZImageConfig has sensible defaults."""
        config = ZImageConfig()
        assert config.model_path == ""
        assert config.text_encoder_path == ""
        assert config.variant == "auto"
        assert config.default_steps is None
        assert config.default_guidance_scale is None
        assert config.default_shift is None
        assert config.default_negative_prompt == ""
        assert config.default_cfg_normalization == 0.0

    def test_turbo_preset_values(self):
        """ZImageConfig with turbo values."""
        config = ZImageConfig(
            model_path="models/Z-Image-Turbo",
            variant="turbo",
            default_steps=9,
            default_guidance_scale=0.0,
            default_shift=3.0,
        )
        assert config.default_steps == 9
        assert config.default_guidance_scale == 0.0
        assert config.default_shift == 3.0

    def test_base_preset_values(self):
        """ZImageConfig with base model values."""
        config = ZImageConfig(
            model_path="models/Z-Image",
            variant="base",
            default_steps=35,
            default_guidance_scale=4.0,
            default_shift=6.0,
            default_negative_prompt="blur, artifacts, low quality",
        )
        assert config.default_steps == 35
        assert config.default_guidance_scale == 4.0
        assert config.default_shift == 6.0
        assert "blur" in config.default_negative_prompt


class TestZImageInConfig:
    """Test ZImageConfig integration with main Config class."""

    def test_config_has_zimage(self):
        """Main Config class includes zimage field."""
        config = Config()
        assert hasattr(config, "zimage")
        assert isinstance(config.zimage, ZImageConfig)

    def test_config_from_dict_with_zimage(self):
        """Config.from_dict() handles zimage section."""
        data = {
            "model_path": "/path/to/model",
            "zimage": {
                "model_path": "models/Z-Image",
                "variant": "base",
                "default_steps": 35,
            },
        }
        config = Config.from_dict(data)

        assert config.zimage.model_path == "models/Z-Image"
        assert config.zimage.variant == "base"
        assert config.zimage.default_steps == 35


class TestNegativePromptHandling:
    """Test negative prompt handling across variants."""

    def test_turbo_ignores_negative_prompt(self):
        """Turbo variant defaults to None for negative prompt."""
        defaults = get_variant_defaults("turbo")
        assert defaults["negative_prompt"] is None

    def test_base_supports_negative_prompt(self):
        """Base variant defaults to empty string (supports but optional)."""
        defaults = get_variant_defaults("base")
        assert defaults["negative_prompt"] == ""
        assert defaults["negative_prompt"] is not None  # Explicitly NOT None

    def test_base_guidance_scale_enables_cfg(self):
        """Base variant has non-zero guidance scale for CFG."""
        defaults = get_variant_defaults("base")
        assert defaults["guidance_scale"] > 0.0  # CFG is used
        assert defaults["guidance_scale"] == 4.0  # Specific value


class TestVariantAwareSteps:
    """Test step count differences between variants."""

    def test_turbo_steps_are_fast(self):
        """Turbo should use 9 steps (fast)."""
        defaults = get_variant_defaults("turbo")
        assert defaults["num_inference_steps"] == 9

    def test_base_steps_are_quality(self):
        """Base should use 35 steps (quality)."""
        defaults = get_variant_defaults("base")
        assert defaults["num_inference_steps"] == 35
        assert defaults["num_inference_steps"] > 20  # More than turbo


class TestSchedulerShiftDifferences:
    """Test scheduler shift differences between variants."""

    def test_turbo_shift(self):
        """Turbo uses shift=3.0."""
        defaults = get_variant_defaults("turbo")
        assert defaults["shift"] == 3.0

    def test_base_shift(self):
        """Base uses shift=6.0 (higher for quality)."""
        defaults = get_variant_defaults("base")
        assert defaults["shift"] == 6.0
        assert defaults["shift"] > get_variant_defaults("turbo")["shift"]
