"""
Unit tests for Z-Image web server request models.

last updated: 2026-01-27

Tests that GenerateRequest and Img2ImgRequest models properly accept
and validate negative_prompt and other Z-Image parameters.
"""

import pytest
from pydantic import ValidationError

# Import the request models from shared schemas
from web.schemas import GenerateRequest, Img2ImgRequest

pytestmark = pytest.mark.unit


class TestGenerateRequestModel:
    """Test GenerateRequest Pydantic model."""

    def test_basic_request(self):
        """Basic request with just prompt."""
        request = GenerateRequest(prompt="A cat sleeping in sunlight")
        assert request.prompt == "A cat sleeping in sunlight"
        assert request.negative_prompt is None
        assert request.width == 1024
        assert request.height == 1024
        assert request.steps == 9
        assert request.guidance_scale == 0.0  # Turbo default

    def test_request_with_negative_prompt(self):
        """Request includes negative_prompt for base model."""
        request = GenerateRequest(
            prompt="A professional portrait",
            negative_prompt="blur, artifacts, low quality, watermark",
            guidance_scale=4.0,
            steps=35,
        )
        assert request.prompt == "A professional portrait"
        assert request.negative_prompt == "blur, artifacts, low quality, watermark"
        assert request.guidance_scale == 4.0
        assert request.steps == 35

    def test_request_with_all_params(self):
        """Request with all generation parameters."""
        request = GenerateRequest(
            prompt="A detailed landscape",
            negative_prompt="ugly, deformed",
            width=1280,
            height=768,
            steps=50,
            guidance_scale=5.0,
            cfg_normalization=0.5,
            shift=6.0,
            seed=42,
        )
        assert request.width == 1280
        assert request.height == 768
        assert request.steps == 50
        assert request.cfg_normalization == 0.5
        assert request.shift == 6.0
        assert request.seed == 42

    def test_empty_negative_prompt(self):
        """Empty string is valid for negative prompt."""
        request = GenerateRequest(
            prompt="A cat",
            negative_prompt="",
        )
        assert request.negative_prompt == ""

    def test_none_negative_prompt(self):
        """None is valid for negative prompt (turbo mode)."""
        request = GenerateRequest(
            prompt="A cat",
            negative_prompt=None,
        )
        assert request.negative_prompt is None

    def test_turbo_mode_defaults(self):
        """Turbo mode typical configuration."""
        request = GenerateRequest(
            prompt="A cat",
            steps=9,
            guidance_scale=0.0,
            shift=3.0,
        )
        assert request.steps == 9
        assert request.guidance_scale == 0.0
        assert request.shift == 3.0
        # Negative prompt should be None for turbo
        assert request.negative_prompt is None

    def test_base_mode_configuration(self):
        """Base model typical configuration."""
        request = GenerateRequest(
            prompt="A detailed image",
            negative_prompt="low quality",
            steps=35,
            guidance_scale=4.0,
            shift=6.0,
        )
        assert request.steps == 35
        assert request.guidance_scale == 4.0
        assert request.shift == 6.0
        assert request.negative_prompt == "low quality"

    def test_cfg_normalization_range(self):
        """CFG normalization should be non-negative."""
        request = GenerateRequest(
            prompt="A cat",
            cfg_normalization=1.0,
        )
        assert request.cfg_normalization == 1.0

    def test_dynamic_shift_flag(self):
        """Dynamic shift flag works."""
        request = GenerateRequest(
            prompt="A cat",
            dynamic_shift=True,
        )
        assert request.dynamic_shift is True


class TestImg2ImgRequestModel:
    """Test Img2ImgRequest Pydantic model."""

    def test_basic_img2img_request(self):
        """Basic img2img request with required fields."""
        # Minimal base64 PNG (1x1 transparent pixel)
        minimal_png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        request = Img2ImgRequest(
            prompt="Enhance this image",
            image=minimal_png,
        )
        assert request.prompt == "Enhance this image"
        assert request.negative_prompt is None
        assert request.strength == 0.75  # Default

    def test_img2img_with_negative_prompt(self):
        """Img2img request with negative prompt."""
        minimal_png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        request = Img2ImgRequest(
            prompt="Enhance this image",
            negative_prompt="blur, noise, artifacts",
            image=minimal_png,
            strength=0.5,
            guidance_scale=4.0,
        )
        assert request.negative_prompt == "blur, noise, artifacts"
        assert request.strength == 0.5
        assert request.guidance_scale == 4.0

    def test_img2img_strength_bounds(self):
        """Strength should be between 0 and 1."""
        minimal_png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        # Valid bounds
        request = Img2ImgRequest(prompt="test", image=minimal_png, strength=0.0)
        assert request.strength == 0.0

        request = Img2ImgRequest(prompt="test", image=minimal_png, strength=1.0)
        assert request.strength == 1.0

    def test_img2img_with_base_model_params(self):
        """Img2img with base model parameters."""
        minimal_png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        request = Img2ImgRequest(
            prompt="Enhance image",
            negative_prompt="low quality",
            image=minimal_png,
            strength=0.6,
            steps=35,
            guidance_scale=4.0,
            shift=6.0,
            cfg_normalization=0.5,
        )
        assert request.steps == 35
        assert request.guidance_scale == 4.0
        assert request.shift == 6.0
        assert request.cfg_normalization == 0.5


class TestRequestValidation:
    """Test request validation and error handling."""

    def test_empty_prompt_allowed(self):
        """Empty prompt should work (some use cases need it)."""
        request = GenerateRequest(prompt="")
        assert request.prompt == ""

    def test_long_negative_prompt(self):
        """Long negative prompts should be accepted."""
        long_negative = ", ".join(["low quality"] * 100)
        request = GenerateRequest(
            prompt="A cat",
            negative_prompt=long_negative,
        )
        assert len(request.negative_prompt) > 1000

    def test_guidance_scale_bounds(self):
        """Guidance scale should be within valid range."""
        # Valid bounds
        request = GenerateRequest(prompt="test", guidance_scale=0.0)
        assert request.guidance_scale == 0.0

        request = GenerateRequest(prompt="test", guidance_scale=30.0)
        assert request.guidance_scale == 30.0

    def test_seed_accepts_none(self):
        """Seed should accept None for random."""
        request = GenerateRequest(prompt="test", seed=None)
        assert request.seed is None

    def test_seed_accepts_integer(self):
        """Seed should accept integer values."""
        request = GenerateRequest(prompt="test", seed=42)
        assert request.seed == 42
