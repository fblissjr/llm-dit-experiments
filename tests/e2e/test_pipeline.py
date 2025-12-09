"""
End-to-end pipeline tests.

These tests require:
- Z-Image model files (Z_IMAGE_MODEL_PATH)
- CUDA GPU with sufficient VRAM (RTX 4090 recommended)

Run on Linux server only:
    Z_IMAGE_MODEL_PATH=/path/to/model pytest tests/e2e/ -v
"""

import pytest
import torch
from PIL import Image

pytestmark = [pytest.mark.e2e, pytest.mark.requires_gpu, pytest.mark.requires_model]


class TestFullPipeline:
    """Test complete text-to-image generation pipeline."""

    @pytest.mark.slow
    def test_basic_generation(self, z_image_model_path, output_dir):
        """Generate a basic image."""
        from llm_dit.pipelines import ZImagePipeline

        pipe = ZImagePipeline.from_pretrained(
            z_image_model_path,
            torch_dtype=torch.bfloat16,
            encoder_device="cpu",
            dit_device="cuda",
            vae_device="cuda",
        )

        image = pipe(
            "A cat sleeping in sunlight",
            height=1024,
            width=1024,
            num_inference_steps=9,
            generator=torch.Generator().manual_seed(42),
        )

        assert isinstance(image, Image.Image)
        assert image.size == (1024, 1024)

        # Save for manual inspection
        output_path = output_dir / "basic_generation.png"
        image.save(output_path)
        assert output_path.exists()

    @pytest.mark.slow
    def test_generation_with_system_prompt(self, z_image_model_path, output_dir):
        """Generate with system prompt."""
        from llm_dit.pipelines import ZImagePipeline

        pipe = ZImagePipeline.from_pretrained(
            z_image_model_path,
            torch_dtype=torch.bfloat16,
            encoder_device="cpu",
            dit_device="cuda",
            vae_device="cuda",
        )

        image = pipe(
            "A portrait of a woman",
            system_prompt="You are an expert portrait photographer.",
            height=1024,
            width=1024,
            num_inference_steps=9,
            generator=torch.Generator().manual_seed(42),
        )

        assert isinstance(image, Image.Image)
        output_path = output_dir / "with_system_prompt.png"
        image.save(output_path)

    @pytest.mark.slow
    def test_generation_with_thinking(self, z_image_model_path, output_dir):
        """Generate with thinking content."""
        from llm_dit.pipelines import ZImagePipeline

        pipe = ZImagePipeline.from_pretrained(
            z_image_model_path,
            torch_dtype=torch.bfloat16,
            encoder_device="cpu",
            dit_device="cuda",
            vae_device="cuda",
        )

        image = pipe(
            "A mountain landscape",
            thinking_content="Golden hour lighting, dramatic clouds, snow-capped peaks.",
            enable_thinking=True,
            height=1024,
            width=1024,
            num_inference_steps=9,
            generator=torch.Generator().manual_seed(42),
        )

        assert isinstance(image, Image.Image)
        output_path = output_dir / "with_thinking.png"
        image.save(output_path)

    @pytest.mark.slow
    def test_different_resolutions(self, z_image_model_path, output_dir):
        """Test various resolution presets."""
        from llm_dit.pipelines import ZImagePipeline

        pipe = ZImagePipeline.from_pretrained(
            z_image_model_path,
            torch_dtype=torch.bfloat16,
            encoder_device="cpu",
            dit_device="cuda",
            vae_device="cuda",
        )

        resolutions = [
            (1024, 1024, "square"),
            (1280, 720, "landscape"),
            (720, 1280, "portrait"),
        ]

        for width, height, name in resolutions:
            image = pipe(
                "A test image",
                height=height,
                width=width,
                num_inference_steps=4,  # Fewer steps for speed
                generator=torch.Generator().manual_seed(42),
            )

            assert image.size == (width, height)
            output_path = output_dir / f"resolution_{name}.png"
            image.save(output_path)

    @pytest.mark.slow
    def test_reproducibility_with_seed(self, z_image_model_path):
        """Verify same seed produces same image."""
        from llm_dit.pipelines import ZImagePipeline

        pipe = ZImagePipeline.from_pretrained(
            z_image_model_path,
            torch_dtype=torch.bfloat16,
            encoder_device="cpu",
            dit_device="cuda",
            vae_device="cuda",
        )

        prompt = "A red apple on a table"
        seed = 12345

        # Generate twice with same seed
        image1 = pipe(
            prompt,
            height=512,
            width=512,
            num_inference_steps=4,
            generator=torch.Generator().manual_seed(seed),
        )

        image2 = pipe(
            prompt,
            height=512,
            width=512,
            num_inference_steps=4,
            generator=torch.Generator().manual_seed(seed),
        )

        # Images should be identical (or very close)
        import numpy as np

        arr1 = np.array(image1)
        arr2 = np.array(image2)
        assert np.allclose(arr1, arr2, atol=1)


class TestDistributedInference:
    """Test distributed inference with API backend."""

    @pytest.mark.requires_api
    @pytest.mark.slow
    def test_api_encoding_local_generation(
        self, z_image_model_path, api_server_url, output_dir
    ):
        """Test: API handles encoding, local handles generation."""
        from llm_dit.backends.api import APIBackend, APIBackendConfig
        from llm_dit.encoders import ZImageTextEncoder
        from llm_dit.pipelines import ZImagePipeline

        # Create API-backed encoder
        api_config = APIBackendConfig(
            base_url=api_server_url,
            model_id="Qwen3-4B-mlx",
        )
        api_backend = APIBackend(api_config)
        encoder = ZImageTextEncoder(backend=api_backend)

        # Load pipeline without encoder
        pipe = ZImagePipeline.from_pretrained_generator_only(
            z_image_model_path,
            torch_dtype=torch.bfloat16,
            dit_device="cuda",
            vae_device="cuda",
        )
        pipe.encoder = encoder

        # Generate
        image = pipe(
            "A sunset over mountains",
            height=1024,
            width=1024,
            num_inference_steps=9,
            generator=torch.Generator().manual_seed(42),
        )

        assert isinstance(image, Image.Image)
        output_path = output_dir / "distributed_inference.png"
        image.save(output_path)


@pytest.mark.requires_lora
class TestLoRAIntegration:
    """Test LoRA loading and generation."""

    @pytest.mark.slow
    def test_load_single_lora(self, z_image_model_path, lora_path, output_dir):
        """Load and apply a single LoRA."""
        from llm_dit.pipelines import ZImagePipeline

        pipe = ZImagePipeline.from_pretrained(
            z_image_model_path,
            torch_dtype=torch.bfloat16,
            encoder_device="cpu",
            dit_device="cuda",
            vae_device="cuda",
        )

        # Load LoRA
        updated_layers = pipe.load_lora(lora_path, scale=0.8)
        assert updated_layers > 0

        # Generate with LoRA
        image = pipe(
            "An anime character",
            height=1024,
            width=1024,
            num_inference_steps=9,
            generator=torch.Generator().manual_seed(42),
        )

        assert isinstance(image, Image.Image)
        output_path = output_dir / "with_lora.png"
        image.save(output_path)

    @pytest.mark.slow
    def test_lora_affects_output(self, z_image_model_path, lora_path):
        """Verify LoRA changes the output."""
        from llm_dit.pipelines import ZImagePipeline
        import numpy as np

        prompt = "An anime character with blue hair"
        seed = 42

        # Generate without LoRA
        pipe1 = ZImagePipeline.from_pretrained(
            z_image_model_path,
            torch_dtype=torch.bfloat16,
            encoder_device="cpu",
            dit_device="cuda",
            vae_device="cuda",
        )

        image_without = pipe1(
            prompt,
            height=512,
            width=512,
            num_inference_steps=4,
            generator=torch.Generator().manual_seed(seed),
        )

        # Generate with LoRA
        pipe2 = ZImagePipeline.from_pretrained(
            z_image_model_path,
            torch_dtype=torch.bfloat16,
            encoder_device="cpu",
            dit_device="cuda",
            vae_device="cuda",
        )
        pipe2.load_lora(lora_path, scale=0.8)

        image_with = pipe2(
            prompt,
            height=512,
            width=512,
            num_inference_steps=4,
            generator=torch.Generator().manual_seed(seed),
        )

        # Images should be different
        arr_without = np.array(image_without)
        arr_with = np.array(image_with)
        assert not np.allclose(arr_without, arr_with, atol=10)


class TestSchedulerSettings:
    """Test scheduler configuration."""

    @pytest.mark.slow
    def test_shift_parameter(self, z_image_model_path, output_dir):
        """Test different shift values."""
        from llm_dit.pipelines import ZImagePipeline

        pipe = ZImagePipeline.from_pretrained(
            z_image_model_path,
            torch_dtype=torch.bfloat16,
            encoder_device="cpu",
            dit_device="cuda",
            vae_device="cuda",
        )

        # Test different shift values
        for shift in [1.0, 3.0, 5.0]:
            image = pipe(
                "A test image",
                height=512,
                width=512,
                num_inference_steps=4,
                shift=shift,
                generator=torch.Generator().manual_seed(42),
            )

            assert isinstance(image, Image.Image)
            output_path = output_dir / f"shift_{shift}.png"
            image.save(output_path)
