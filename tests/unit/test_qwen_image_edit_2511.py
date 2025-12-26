"""
Unit tests for Qwen-Image-Edit-2511 features.

Tests the new edit_multi() method and updated defaults without requiring
GPU or real models (uses mocks where appropriate).
"""

import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

pytestmark = pytest.mark.unit


class TestQwenImageEditDefaults:
    """Test updated defaults for Qwen-Image-Edit-2511."""

    def test_default_steps_updated(self):
        """Test DEFAULT_STEPS is 40 (was 50 for 2509)."""
        from llm_dit.pipelines.qwen_image_diffusers import DEFAULT_STEPS

        assert DEFAULT_STEPS == 40

    def test_default_cfg_scale(self):
        """Test DEFAULT_CFG_SCALE is still 4.0."""
        from llm_dit.pipelines.qwen_image_diffusers import DEFAULT_CFG_SCALE

        assert DEFAULT_CFG_SCALE == 4.0


class TestQwenImageDiffusersPipelineImport:
    """Test QwenImageDiffusersPipeline imports correctly."""

    def test_import(self):
        """Test pipeline can be imported."""
        from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

        assert QwenImageDiffusersPipeline is not None

    def test_has_edit_multi_method(self):
        """Test pipeline class has edit_multi method."""
        from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

        # Check that the method exists
        assert hasattr(QwenImageDiffusersPipeline, 'edit_multi')


class TestEditMultiInputValidation:
    """Test edit_multi() input validation."""

    def test_edit_multi_requires_at_least_2_images(self):
        """Test edit_multi raises ValueError for less than 2 images."""
        from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

        # Create a mock pipeline instance
        mock_decompose_pipe = MagicMock()
        mock_edit_pipe = MagicMock()

        pipe = QwenImageDiffusersPipeline(
            decompose_pipe=mock_decompose_pipe,
            edit_pipe=mock_edit_pipe,
        )

        # Create single image
        single_image = Image.new("RGB", (512, 512), (255, 0, 0))

        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            pipe.edit_multi(
                images=[single_image],
                instruction="Combine them",
            )

        assert "at least 2 images" in str(exc_info.value)
        assert "edit_layer()" in str(exc_info.value)

    def test_edit_multi_requires_non_empty_list(self):
        """Test edit_multi raises ValueError for empty list."""
        from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

        mock_decompose_pipe = MagicMock()
        mock_edit_pipe = MagicMock()

        pipe = QwenImageDiffusersPipeline(
            decompose_pipe=mock_decompose_pipe,
            edit_pipe=mock_edit_pipe,
        )

        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            pipe.edit_multi(
                images=[],
                instruction="Combine them",
            )

        assert "at least 2 images" in str(exc_info.value)

    def test_edit_multi_accepts_2_images(self):
        """Test edit_multi accepts 2 images (minimum)."""
        from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

        mock_decompose_pipe = MagicMock()
        mock_edit_pipe = MagicMock()

        # Set up mock to return proper result
        mock_result = MagicMock()
        mock_result.images = [Image.new("RGB", (512, 512), (0, 255, 0))]
        mock_edit_pipe.return_value = mock_result

        pipe = QwenImageDiffusersPipeline(
            decompose_pipe=mock_decompose_pipe,
            edit_pipe=mock_edit_pipe,
        )

        # Create 2 images
        img1 = Image.new("RGB", (512, 512), (255, 0, 0))
        img2 = Image.new("RGB", (512, 512), (0, 0, 255))

        # Should not raise
        result = pipe.edit_multi(
            images=[img1, img2],
            instruction="Place them side by side",
        )

        assert result is not None

    def test_edit_multi_accepts_4_images(self):
        """Test edit_multi accepts 4 images."""
        from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

        mock_decompose_pipe = MagicMock()
        mock_edit_pipe = MagicMock()

        mock_result = MagicMock()
        mock_result.images = [Image.new("RGB", (512, 512), (0, 255, 0))]
        mock_edit_pipe.return_value = mock_result

        pipe = QwenImageDiffusersPipeline(
            decompose_pipe=mock_decompose_pipe,
            edit_pipe=mock_edit_pipe,
        )

        # Create 4 images
        images = [Image.new("RGB", (512, 512), (i * 50, 0, 0)) for i in range(4)]

        result = pipe.edit_multi(
            images=images,
            instruction="Group photo in a park",
        )

        assert result is not None


class TestEditMultiImageConversion:
    """Test edit_multi() handles image mode conversions."""

    def test_converts_rgba_to_rgb(self):
        """Test RGBA images are converted to RGB."""
        from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

        mock_decompose_pipe = MagicMock()
        mock_edit_pipe = MagicMock()

        mock_result = MagicMock()
        mock_result.images = [Image.new("RGB", (512, 512), (0, 255, 0))]
        mock_edit_pipe.return_value = mock_result

        pipe = QwenImageDiffusersPipeline(
            decompose_pipe=mock_decompose_pipe,
            edit_pipe=mock_edit_pipe,
        )

        # Create RGBA images
        img1 = Image.new("RGBA", (512, 512), (255, 0, 0, 128))
        img2 = Image.new("RGBA", (512, 512), (0, 0, 255, 128))

        pipe.edit_multi(
            images=[img1, img2],
            instruction="Combine",
        )

        # Check that the pipeline was called with RGB images
        call_kwargs = mock_edit_pipe.call_args[1]
        images_passed = call_kwargs["image"]
        for img in images_passed:
            assert img.mode == "RGB"

    def test_handles_mixed_modes(self):
        """Test handles mixed RGB and RGBA images."""
        from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

        mock_decompose_pipe = MagicMock()
        mock_edit_pipe = MagicMock()

        mock_result = MagicMock()
        mock_result.images = [Image.new("RGB", (512, 512), (0, 255, 0))]
        mock_edit_pipe.return_value = mock_result

        pipe = QwenImageDiffusersPipeline(
            decompose_pipe=mock_decompose_pipe,
            edit_pipe=mock_edit_pipe,
        )

        # Mix RGB and RGBA
        img1 = Image.new("RGB", (512, 512), (255, 0, 0))
        img2 = Image.new("RGBA", (512, 512), (0, 0, 255, 128))

        pipe.edit_multi(
            images=[img1, img2],
            instruction="Combine",
        )

        # Should complete without error
        mock_edit_pipe.assert_called_once()


class TestEditMultiParameters:
    """Test edit_multi() parameter handling."""

    def test_default_steps_is_40(self):
        """Test default num_inference_steps is 40."""
        from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

        mock_decompose_pipe = MagicMock()
        mock_edit_pipe = MagicMock()

        mock_result = MagicMock()
        mock_result.images = [Image.new("RGB", (512, 512), (0, 255, 0))]
        mock_edit_pipe.return_value = mock_result

        pipe = QwenImageDiffusersPipeline(
            decompose_pipe=mock_decompose_pipe,
            edit_pipe=mock_edit_pipe,
        )

        images = [Image.new("RGB", (512, 512)) for _ in range(2)]

        pipe.edit_multi(
            images=images,
            instruction="Test",
            # Don't specify num_inference_steps
        )

        call_kwargs = mock_edit_pipe.call_args[1]
        assert call_kwargs["num_inference_steps"] == 40

    def test_custom_steps(self):
        """Test custom num_inference_steps is respected."""
        from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

        mock_decompose_pipe = MagicMock()
        mock_edit_pipe = MagicMock()

        mock_result = MagicMock()
        mock_result.images = [Image.new("RGB", (512, 512), (0, 255, 0))]
        mock_edit_pipe.return_value = mock_result

        pipe = QwenImageDiffusersPipeline(
            decompose_pipe=mock_decompose_pipe,
            edit_pipe=mock_edit_pipe,
        )

        images = [Image.new("RGB", (512, 512)) for _ in range(2)]

        pipe.edit_multi(
            images=images,
            instruction="Test",
            num_inference_steps=50,
        )

        call_kwargs = mock_edit_pipe.call_args[1]
        assert call_kwargs["num_inference_steps"] == 50

    def test_cfg_scale_default(self):
        """Test default cfg_scale is 4.0."""
        from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

        mock_decompose_pipe = MagicMock()
        mock_edit_pipe = MagicMock()

        mock_result = MagicMock()
        mock_result.images = [Image.new("RGB", (512, 512), (0, 255, 0))]
        mock_edit_pipe.return_value = mock_result

        pipe = QwenImageDiffusersPipeline(
            decompose_pipe=mock_decompose_pipe,
            edit_pipe=mock_edit_pipe,
        )

        images = [Image.new("RGB", (512, 512)) for _ in range(2)]

        pipe.edit_multi(
            images=images,
            instruction="Test",
        )

        call_kwargs = mock_edit_pipe.call_args[1]
        assert call_kwargs["true_cfg_scale"] == 4.0

    def test_custom_cfg_scale(self):
        """Test custom cfg_scale is respected."""
        from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

        mock_decompose_pipe = MagicMock()
        mock_edit_pipe = MagicMock()

        mock_result = MagicMock()
        mock_result.images = [Image.new("RGB", (512, 512), (0, 255, 0))]
        mock_edit_pipe.return_value = mock_result

        pipe = QwenImageDiffusersPipeline(
            decompose_pipe=mock_decompose_pipe,
            edit_pipe=mock_edit_pipe,
        )

        images = [Image.new("RGB", (512, 512)) for _ in range(2)]

        pipe.edit_multi(
            images=images,
            instruction="Test",
            cfg_scale=6.0,
        )

        call_kwargs = mock_edit_pipe.call_args[1]
        assert call_kwargs["true_cfg_scale"] == 6.0

    def test_instruction_passed_as_prompt(self):
        """Test instruction is passed as prompt parameter."""
        from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

        mock_decompose_pipe = MagicMock()
        mock_edit_pipe = MagicMock()

        mock_result = MagicMock()
        mock_result.images = [Image.new("RGB", (512, 512), (0, 255, 0))]
        mock_edit_pipe.return_value = mock_result

        pipe = QwenImageDiffusersPipeline(
            decompose_pipe=mock_decompose_pipe,
            edit_pipe=mock_edit_pipe,
        )

        images = [Image.new("RGB", (512, 512)) for _ in range(2)]
        instruction = "Place both people in a forest setting"

        pipe.edit_multi(
            images=images,
            instruction=instruction,
        )

        call_kwargs = mock_edit_pipe.call_args[1]
        assert call_kwargs["prompt"] == instruction

    def test_images_passed_as_list(self):
        """Test images are passed as a list to the pipeline."""
        from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

        mock_decompose_pipe = MagicMock()
        mock_edit_pipe = MagicMock()

        mock_result = MagicMock()
        mock_result.images = [Image.new("RGB", (512, 512), (0, 255, 0))]
        mock_edit_pipe.return_value = mock_result

        pipe = QwenImageDiffusersPipeline(
            decompose_pipe=mock_decompose_pipe,
            edit_pipe=mock_edit_pipe,
        )

        images = [Image.new("RGB", (512, 512)) for _ in range(3)]

        pipe.edit_multi(
            images=images,
            instruction="Test",
        )

        call_kwargs = mock_edit_pipe.call_args[1]
        assert isinstance(call_kwargs["image"], list)
        assert len(call_kwargs["image"]) == 3


class TestEditMultiLazyLoading:
    """Test edit_multi() lazy loading behavior."""

    def test_loads_edit_model_if_not_loaded(self):
        """Test edit model is loaded if not already loaded."""
        from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

        mock_decompose_pipe = MagicMock()

        pipe = QwenImageDiffusersPipeline(
            decompose_pipe=mock_decompose_pipe,
            edit_pipe=None,  # Not loaded
        )
        pipe._edit_model_path = "Qwen/Qwen-Image-Edit-2511"

        images = [Image.new("RGB", (512, 512)) for _ in range(2)]

        # Mock load_edit_model to set edit_pipe
        with patch.object(pipe, 'load_edit_model') as mock_load:
            # Make load_edit_model set up a mock edit_pipe
            def set_mock_pipe():
                mock_edit_pipe = MagicMock()
                mock_result = MagicMock()
                mock_result.images = [Image.new("RGB", (512, 512))]
                mock_edit_pipe.return_value = mock_result
                pipe.edit_pipe = mock_edit_pipe

            mock_load.side_effect = set_mock_pipe

            pipe.edit_multi(
                images=images,
                instruction="Test",
            )

            mock_load.assert_called_once()


class TestEditMultiMethodSignature:
    """Test edit_multi() method signature."""

    def test_method_signature(self):
        """Test method has correct parameters."""
        from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline
        import inspect

        sig = inspect.signature(QwenImageDiffusersPipeline.edit_multi)
        params = sig.parameters

        # Required parameters
        assert "images" in params
        assert "instruction" in params

        # Optional parameters with defaults
        assert "num_inference_steps" in params
        assert params["num_inference_steps"].default == 40
        assert "cfg_scale" in params
        assert params["cfg_scale"].default == 4.0
        assert "seed" in params
        assert params["seed"].default is None

    def test_returns_pil_image(self):
        """Test method returns a PIL Image."""
        from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

        mock_decompose_pipe = MagicMock()
        mock_edit_pipe = MagicMock()

        output_image = Image.new("RGB", (1024, 1024), (0, 255, 0))
        mock_result = MagicMock()
        mock_result.images = [output_image]
        mock_edit_pipe.return_value = mock_result

        pipe = QwenImageDiffusersPipeline(
            decompose_pipe=mock_decompose_pipe,
            edit_pipe=mock_edit_pipe,
        )

        images = [Image.new("RGB", (512, 512)) for _ in range(2)]

        result = pipe.edit_multi(
            images=images,
            instruction="Test",
        )

        assert isinstance(result, Image.Image)
        assert result == output_image


class TestConfigDefaults:
    """Test configuration defaults are updated for 2511."""

    def test_qwen_image_config_default_steps(self):
        """Test QwenImageConfig.num_inference_steps is 40."""
        from llm_dit.config import QwenImageConfig

        config = QwenImageConfig()
        assert config.num_inference_steps == 40

    def test_runtime_config_default_steps(self):
        """Test RuntimeConfig.qwen_image_steps is 40."""
        from llm_dit.cli import RuntimeConfig

        config = RuntimeConfig()
        assert config.qwen_image_steps == 40
