"""
Pure PyTorch path tests for Z-Image pipeline.

Tests the pure PyTorch implementation (use_diffusers=False) including:
- Component loading (transformer, VAE, scheduler)
- Image generation quality (not noise)
- Variant config flow (shift parameter)
- Reproducibility with seeds

Requires:
- Z-Image model files (Z_IMAGE_MODEL_PATH environment variable)
- CUDA GPU with sufficient VRAM

Run:
    Z_IMAGE_MODEL_PATH=/path/to/model pytest tests/e2e/z_image/test_pure_pytorch.py -v -s

Last updated: 2026-02-01
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = [pytest.mark.e2e, pytest.mark.requires_gpu, pytest.mark.requires_model]


# DEPRECATED: Use zimage_base_preset fixture instead
# Kept temporarily for backwards compatibility with existing tests
# TODO: Remove once all tests migrate to preset fixtures
BASE_MODEL_DEFAULTS = {
    "guidance_scale": 4.0,  # Required for Base model
    "num_inference_steps": 30,
    "shift": 6.0,  # Base model default
}


def save_generation_metadata(
    output_dir: Path,
    image_path: Path,
    prompt: str,
    seed: int,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    shift: float,
    negative_prompt: str | None = None,
    variant: str = "base",
    **extra_params,
) -> Path:
    """Save generation metadata alongside the image.

    Creates a JSON file with all parameters used to generate the image,
    enabling reproducibility and regression testing.
    """
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "image_file": image_path.name,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "shift": shift,
        "variant": variant,
        **extra_params,
    }

    metadata_path = output_dir / f"{image_path.stem}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path


class TestPurePyTorchLoading:
    """Verify pure PyTorch components load correctly."""

    def test_transformer_is_zimage_dit(self, z_image_model_path):
        """Verify transformer is ZImageDiT, not diffusers."""
        from llm_dit.models.z_image import ZImageDiT
        from llm_dit.pipelines import ZImagePipeline

        pipe = ZImagePipeline.from_pretrained(
            z_image_model_path,
            use_diffusers=False,
            encoder_device="cpu",
            dit_device="cuda",
            vae_device="cuda",
        )
        assert isinstance(pipe.transformer, ZImageDiT), (
            f"Expected ZImageDiT, got {type(pipe.transformer).__name__}"
        )

    def test_vae_is_flux_decoder(self, z_image_model_path):
        """Verify VAE is FluxVAEDecoder."""
        from llm_dit.models.z_image.vae import FluxVAEDecoder
        from llm_dit.pipelines import ZImagePipeline

        pipe = ZImagePipeline.from_pretrained(
            z_image_model_path,
            use_diffusers=False,
            encoder_device="cpu",
            dit_device="cuda",
            vae_device="cuda",
        )
        assert isinstance(pipe.vae, FluxVAEDecoder), (
            f"Expected FluxVAEDecoder, got {type(pipe.vae).__name__}"
        )

    def test_scheduler_is_flow_match(self, z_image_model_path):
        """Verify scheduler is FlowMatchScheduler."""
        from llm_dit.pipelines import ZImagePipeline
        from llm_dit.schedulers import FlowMatchScheduler

        pipe = ZImagePipeline.from_pretrained(
            z_image_model_path,
            use_diffusers=False,
            encoder_device="cpu",
            dit_device="cuda",
            vae_device="cuda",
        )
        assert isinstance(pipe.scheduler, FlowMatchScheduler), (
            f"Expected FlowMatchScheduler, got {type(pipe.scheduler).__name__}"
        )


class TestPurePyTorchGeneration:
    """Verify pure PyTorch generates valid images."""

    @pytest.mark.slow
    def test_basic_generation_not_noise(self, z_image_model_path, output_dir, image_verifier):
        """Verify output is a valid image, not noise."""
        from llm_dit.pipelines import ZImagePipeline

        pipe = ZImagePipeline.from_pretrained(
            z_image_model_path,
            use_diffusers=False,
            encoder_device="cpu",
            dit_device="cuda",
            vae_device="cuda",
        )

        # Generation parameters - use Base model defaults (NOT Turbo)
        # Turbo: guidance_scale=0.0 (baked in), steps=9, shift=3.0
        # Base: guidance_scale=4.0+, steps=30+, shift=6.0
        prompt = "A cat sleeping in sunlight"
        seed = 42
        height = 512
        width = 512
        num_inference_steps = 30
        guidance_scale = BASE_MODEL_DEFAULTS["guidance_scale"]
        shift = BASE_MODEL_DEFAULTS["shift"]

        image = pipe(
            prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            shift=shift,
            generator=torch.Generator("cpu").manual_seed(seed),
        )

        # Save image
        output_path = output_dir / "pure_pytorch_basic.png"
        image.save(output_path)

        # Save metadata for reproducibility
        save_generation_metadata(
            output_dir=output_dir,
            image_path=output_path,
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            shift=shift,
            variant="base",
            scheduler_type=type(pipe.scheduler).__name__,
        )

        # Verify not noise using variance check
        result = image_verifier(image)
        assert result["is_valid"], (
            f"Image appears to be noise (variance={result['variance']:.1f})"
        )

        print(f"Image hash: {result['hash']}, size: {result['size']}")
        print(f"Variance: {result['variance']:.1f}")
        print(f"Parameters: guidance_scale={guidance_scale}, steps={num_inference_steps}, shift={shift}")

    @pytest.mark.slow
    def test_reproducibility_with_seed(self, z_image_model_path, image_verifier):
        """Same seed produces identical output."""
        from llm_dit.pipelines import ZImagePipeline

        pipe = ZImagePipeline.from_pretrained(
            z_image_model_path,
            use_diffusers=False,
            encoder_device="cpu",
            dit_device="cuda",
            vae_device="cuda",
        )

        gen1 = torch.Generator("cpu").manual_seed(42)
        image1 = pipe(
            "A cat",
            height=512,
            width=512,
            num_inference_steps=30,
            generator=gen1,
        )

        gen2 = torch.Generator("cpu").manual_seed(42)
        image2 = pipe(
            "A cat",
            height=512,
            width=512,
            num_inference_steps=30,
            generator=gen2,
        )

        hash1 = image_verifier(image1)["hash"]
        hash2 = image_verifier(image2)["hash"]

        assert hash1 == hash2, f"Hashes differ: {hash1} vs {hash2}"

    @pytest.mark.slow
    def test_different_seeds_produce_different_images(self, z_image_model_path, image_verifier):
        """Different seeds produce different outputs."""
        from llm_dit.pipelines import ZImagePipeline

        pipe = ZImagePipeline.from_pretrained(
            z_image_model_path,
            use_diffusers=False,
            encoder_device="cpu",
            dit_device="cuda",
            vae_device="cuda",
        )

        gen1 = torch.Generator("cpu").manual_seed(42)
        image1 = pipe(
            "A cat",
            height=512,
            width=512,
            num_inference_steps=30,
            generator=gen1,
        )

        gen2 = torch.Generator("cpu").manual_seed(123)
        image2 = pipe(
            "A cat",
            height=512,
            width=512,
            num_inference_steps=30,
            generator=gen2,
        )

        hash1 = image_verifier(image1)["hash"]
        hash2 = image_verifier(image2)["hash"]

        assert hash1 != hash2, f"Different seeds should produce different images"


class TestVariantConfig:
    """Verify variant config flows through correctly."""

    def test_base_variant_uses_shift_6(self, z_image_model_path):
        """Base model (no 'turbo' in path) uses shift=6.0."""
        from llm_dit.pipelines import ZImagePipeline

        # Skip if model path contains "turbo"
        if "turbo" in z_image_model_path.lower():
            pytest.skip("Test requires base (non-turbo) model")

        pipe = ZImagePipeline.from_pretrained(
            z_image_model_path,
            use_diffusers=False,
            encoder_device="cpu",
            dit_device="cuda",
            vae_device="cuda",
        )
        assert pipe.scheduler.shift == 6.0, (
            f"Expected shift=6.0 for base model, got {pipe.scheduler.shift}"
        )

    def test_turbo_variant_uses_shift_3(self):
        """Turbo model uses shift=3.0."""
        from llm_dit.pipelines import ZImagePipeline

        # This test only runs if turbo model path is available
        import os
        turbo_path = os.getenv("Z_IMAGE_TURBO_MODEL_PATH")
        if not turbo_path:
            pytest.skip("Z_IMAGE_TURBO_MODEL_PATH not set")

        pipe = ZImagePipeline.from_pretrained(
            turbo_path,
            use_diffusers=False,
            encoder_device="cpu",
            dit_device="cuda",
            vae_device="cuda",
        )
        assert pipe.scheduler.shift == 3.0, (
            f"Expected shift=3.0 for turbo model, got {pipe.scheduler.shift}"
        )

    def test_explicit_shift_override(self, z_image_model_path):
        """Explicit shift parameter overrides variant detection."""
        from llm_dit.pipelines import ZImagePipeline

        pipe = ZImagePipeline.from_pretrained(
            z_image_model_path,
            use_diffusers=False,
            encoder_device="cpu",
            dit_device="cuda",
            vae_device="cuda",
            shift=5.0,  # Explicit override
        )
        assert pipe.scheduler.shift == 5.0, (
            f"Expected shift=5.0 (explicit override), got {pipe.scheduler.shift}"
        )

    def test_variant_detection_from_path(self):
        """Verify _detect_variant_shift works correctly."""
        from llm_dit.pipelines import ZImagePipeline

        # Test turbo detection
        assert ZImagePipeline._detect_variant_shift("models/Z-Image-Turbo") == 3.0
        assert ZImagePipeline._detect_variant_shift("/path/to/turbo/model") == 3.0
        assert ZImagePipeline._detect_variant_shift("Tongyi-MAI/Z-Image-Turbo") == 3.0

        # Test base detection
        assert ZImagePipeline._detect_variant_shift("models/Z-Image") == 6.0
        assert ZImagePipeline._detect_variant_shift("/path/to/base/model") == 6.0
        assert ZImagePipeline._detect_variant_shift("Tongyi-MAI/Z-Image") == 6.0


class TestVisualVerification:
    """Visual verification using image statistics."""

    @pytest.mark.slow
    def test_output_looks_like_prompt(self, z_image_model_path, output_dir, image_verifier):
        """Verify output resembles prompt (not noise/artifacts)."""
        from llm_dit.pipelines import ZImagePipeline

        pipe = ZImagePipeline.from_pretrained(
            z_image_model_path,
            use_diffusers=False,
            encoder_device="cpu",
            dit_device="cuda",
            vae_device="cuda",
        )

        # Use Base model parameters (guidance_scale required for non-Turbo)
        prompt = "A bright red apple on a white background"
        seed = 42
        height = 512
        width = 512
        num_inference_steps = 30
        guidance_scale = BASE_MODEL_DEFAULTS["guidance_scale"]
        shift = BASE_MODEL_DEFAULTS["shift"]

        image = pipe(
            prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            shift=shift,
            generator=torch.Generator("cpu").manual_seed(seed),
        )

        output_path = output_dir / "visual_verification.png"
        image.save(output_path)

        # Save metadata for reproducibility
        save_generation_metadata(
            output_dir=output_dir,
            image_path=output_path,
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            shift=shift,
            variant="base",
        )

        arr = np.array(image)

        # Statistical checks for valid image (not noise)
        variance = np.var(arr)
        mean = np.mean(arr)

        # Valid images: variance 500-6000, mean 50-200
        # Noise: variance > 6000, mean ~127
        assert 500 < variance < 6000, f"Suspicious variance: {variance}"
        assert 30 < mean < 220, f"Suspicious mean: {mean}"

        print(f"Saved: {output_path}")
        print(f"Variance: {variance:.1f}, Mean: {mean:.1f}")
        print(f"Parameters: guidance_scale={guidance_scale}, steps={num_inference_steps}, shift={shift}")

    @pytest.mark.slow
    def test_prompt_affects_output(self, z_image_model_path, image_verifier):
        """Different prompts produce different images."""
        from llm_dit.pipelines import ZImagePipeline

        pipe = ZImagePipeline.from_pretrained(
            z_image_model_path,
            use_diffusers=False,
            encoder_device="cpu",
            dit_device="cuda",
            vae_device="cuda",
        )

        # Use Base model parameters
        guidance_scale = BASE_MODEL_DEFAULTS["guidance_scale"]
        shift = BASE_MODEL_DEFAULTS["shift"]

        # Same seed, different prompts
        gen1 = torch.Generator("cpu").manual_seed(42)
        image1 = pipe(
            "A red apple",
            height=512,
            width=512,
            num_inference_steps=30,
            guidance_scale=guidance_scale,
            shift=shift,
            generator=gen1,
        )

        gen2 = torch.Generator("cpu").manual_seed(42)
        image2 = pipe(
            "A blue car",
            height=512,
            width=512,
            num_inference_steps=30,
            guidance_scale=guidance_scale,
            shift=shift,
            generator=gen2,
        )

        # Images should be different
        arr1 = np.array(image1)
        arr2 = np.array(image2)

        # Calculate normalized difference
        diff = np.abs(arr1.astype(float) - arr2.astype(float)).mean()
        assert diff > 10, f"Different prompts should produce different images (diff={diff:.1f})"


class TestShiftImpactOnGeneration:
    """Test that shift parameter actually affects generation."""

    @pytest.mark.slow
    def test_different_shifts_produce_different_results(self, z_image_model_path, output_dir):
        """Different shift values produce different outputs."""
        from llm_dit.pipelines import ZImagePipeline

        # Use Base model parameters
        guidance_scale = BASE_MODEL_DEFAULTS["guidance_scale"]
        prompt = "A mountain landscape"
        seed = 42

        results = {}
        for shift in [3.0, 6.0]:
            pipe = ZImagePipeline.from_pretrained(
                z_image_model_path,
                use_diffusers=False,
                encoder_device="cpu",
                dit_device="cuda",
                vae_device="cuda",
                shift=shift,
            )

            # Verify the shift was set
            assert pipe.scheduler.shift == shift

            gen = torch.Generator("cpu").manual_seed(seed)
            image = pipe(
                prompt,
                height=512,
                width=512,
                num_inference_steps=30,
                guidance_scale=guidance_scale,
                generator=gen,
            )

            output_path = output_dir / f"shift_{shift}.png"
            image.save(output_path)

            # Save metadata
            save_generation_metadata(
                output_dir=output_dir,
                image_path=output_path,
                prompt=prompt,
                seed=seed,
                height=512,
                width=512,
                num_inference_steps=30,
                guidance_scale=guidance_scale,
                shift=shift,
                variant="base",
            )

            results[shift] = np.array(image)

        # Different shifts should produce different results
        diff = np.abs(results[3.0].astype(float) - results[6.0].astype(float)).mean()
        assert diff > 5, (
            f"Different shifts should affect output (diff={diff:.1f})"
        )


class TestPresetBasedGeneration:
    """Tests using the production preset system for configuration.

    These tests demonstrate using the same GenerationPreset infrastructure
    as the production web UI, ensuring test configs stay in sync.
    """

    @pytest.mark.slow
    def test_basic_generation_with_preset(
        self, z_image_model_path, output_dir, image_verifier, zimage_base_preset
    ):
        """Visual verification using production preset system."""
        from llm_dit.pipelines import ZImagePipeline

        preset = zimage_base_preset

        pipe = ZImagePipeline.from_pretrained(
            z_image_model_path,
            use_diffusers=False,
            encoder_device="cpu",
            dit_device="cuda",
            vae_device="cuda",
        )

        # Extract parameters from preset
        prompt = preset.metadata["prompt"]
        seed = preset.metadata["seed"]
        height = preset.metadata["height"]
        width = preset.metadata["width"]
        min_variance = preset.metadata["min_variance"]
        max_variance = preset.metadata["max_variance"]

        image = pipe(
            prompt,
            height=height,
            width=width,
            num_inference_steps=preset.steps,
            guidance_scale=preset.guidance_scale,
            shift=preset.shift,
            generator=torch.Generator("cpu").manual_seed(seed),
        )

        # Save image
        output_path = output_dir / "preset_based_generation.png"
        image.save(output_path)

        # Save metadata for reproducibility
        save_generation_metadata(
            output_dir=output_dir,
            image_path=output_path,
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_inference_steps=preset.steps,
            guidance_scale=preset.guidance_scale,
            shift=preset.shift,
            variant=preset.variant or "base",
            preset_name=preset.name,
        )

        # Validate using preset thresholds
        result = image_verifier(image, min_variance=min_variance, max_variance=max_variance)
        assert result["is_valid"], (
            f"Image failed preset validation (variance={result['variance']:.1f}, "
            f"expected {min_variance}-{max_variance})"
        )

        print(f"Preset: {preset.name}")
        print(f"Image hash: {result['hash']}, size: {result['size']}")
        print(f"Variance: {result['variance']:.1f} (range: {min_variance}-{max_variance})")
        print(f"Parameters: guidance_scale={preset.guidance_scale}, steps={preset.steps}, shift={preset.shift}")

    def test_preset_matches_hardcoded_defaults(self, zimage_base_preset):
        """Verify preset values match the legacy BASE_MODEL_DEFAULTS.

        This test ensures the preset system is configured correctly and
        matches the previously hardcoded values. Once migration is complete,
        this test can be removed.
        """
        preset = zimage_base_preset

        assert preset.guidance_scale == BASE_MODEL_DEFAULTS["guidance_scale"], (
            f"guidance_scale mismatch: preset={preset.guidance_scale}, "
            f"hardcoded={BASE_MODEL_DEFAULTS['guidance_scale']}"
        )
        assert preset.steps == BASE_MODEL_DEFAULTS["num_inference_steps"], (
            f"steps mismatch: preset={preset.steps}, "
            f"hardcoded={BASE_MODEL_DEFAULTS['num_inference_steps']}"
        )
        assert preset.shift == BASE_MODEL_DEFAULTS["shift"], (
            f"shift mismatch: preset={preset.shift}, "
            f"hardcoded={BASE_MODEL_DEFAULTS['shift']}"
        )

    def test_preset_has_required_metadata(self, zimage_base_preset):
        """Verify preset has all required test metadata fields."""
        preset = zimage_base_preset

        required_fields = ["prompt", "seed", "height", "width", "min_variance", "max_variance"]
        for field in required_fields:
            assert field in preset.metadata, f"Missing required metadata field: {field}"

    def test_preset_factory_works(self, preset_factory):
        """Verify the preset factory can load arbitrary presets."""
        # Load the zimage base test preset
        preset = preset_factory("zimage_base_test")
        assert preset.name == "zimage_base_test"
        assert preset.category == "testing"

        # Verify it has the expected parameters
        assert preset.guidance_scale == 4.0
        assert preset.steps == 30
        assert preset.shift == 6.0
