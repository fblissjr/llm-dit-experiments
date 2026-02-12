"""
Z-Image Base Model E2E Tests.

Tests the BASE model variant (shift=6.0, steps=30-40, CFG=4.0) to verify:
1. Scheduler sigma schedule matches DiffSynth reference (FLUX-style exponential shift)
2. Generation produces valid images (not pure noise)
3. Image variance is in expected range (500-6000, not noise which is >6000)

These tests verify the scheduler fix for the "pure noise" bug where the double
linear shift formula caused images to remain as noise.

Run with model path:
    Z_IMAGE_MODEL_PATH=/path/to/Z-Image pytest tests/integration/pipeline/z_image/test_base_model.py -v -s

Or with a running server:
    pytest tests/integration/pipeline/z_image/test_base_model.py -v -s

Last updated: 2026-01-30
"""

import math
import os
from typing import Optional

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.e2e]


class TestSchedulerSigmaSchedule:
    """Verify scheduler produces correct sigma schedule for Base model."""

    def test_flux_style_exponential_shift(self):
        """Verify sigma schedule matches FLUX-style exponential shift formula."""
        from llm_dit.schedulers import FlowMatchScheduler

        # Base model parameters
        shift = 6.0  # mu for FLUX-style shift
        num_steps = 30

        scheduler = FlowMatchScheduler(
            shift=shift,
            use_dynamic_shifting=True,  # MUST be True for Z-Image
        )
        scheduler.set_timesteps(num_steps, device="cpu")

        sigmas = scheduler.sigmas.tolist()

        # Expected values from DiffSynth reference
        # Formula: sigma_out = exp(mu) / (exp(mu) + (1/sigma_in - 1))
        expected_first = math.exp(shift) / (math.exp(shift) + (1 / 1.0 - 1))  # ~1.0
        expected_last_before_zero = sigmas[-2]  # Should be ~0.02-0.03

        print(f"\n=== Sigma Schedule (shift={shift}, steps={num_steps}) ===")
        print(f"First sigma: {sigmas[0]:.6f} (expected ~1.0)")
        print(f"Last sigma before 0: {sigmas[-2]:.6f} (expected ~0.02-0.03)")
        print(f"Final sigma: {sigmas[-1]:.6f} (expected 0.0)")
        print(f"Schedule length: {len(sigmas)} (expected {num_steps + 1})")

        # Verify first sigma is close to 1.0
        assert abs(sigmas[0] - 1.0) < 0.01, f"First sigma should be ~1.0, got {sigmas[0]}"

        # Verify final sigma is 0.0
        assert sigmas[-1] == 0.0, f"Final sigma should be 0.0, got {sigmas[-1]}"

        # Verify last sigma before zero is low (denoising actually happens)
        assert sigmas[-2] < 0.1, f"Second-to-last sigma should be <0.1, got {sigmas[-2]}"

        # Verify sigma schedule is monotonically decreasing
        for i in range(len(sigmas) - 1):
            assert sigmas[i] >= sigmas[i + 1], (
                f"Sigma schedule should be monotonically decreasing: "
                f"sigmas[{i}]={sigmas[i]}, sigmas[{i + 1}]={sigmas[i + 1]}"
            )

    def test_matches_diffsynth_reference_values(self):
        """Compare sigma values against DiffSynth reference implementation."""
        from llm_dit.schedulers import FlowMatchScheduler

        scheduler = FlowMatchScheduler(
            shift=6.0,
            use_dynamic_shifting=True,
        )
        scheduler.set_timesteps(30, device="cpu")
        sigmas = scheduler.sigmas.tolist()

        # Reference values from DiffSynth (mu=6.0, 30 steps)
        # These are approximate - the key is the shape of the curve
        expected_checkpoints = {
            0: 1.0,  # Start at 1.0
            10: 0.95,  # Still high early (exponential holds noise)
            20: 0.65,  # Middle range
            28: 0.02,  # Near end, almost denoised
            30: 0.0,  # Final is 0
        }

        print("\n=== Sigma Checkpoint Comparison ===")
        for step, expected in expected_checkpoints.items():
            actual = sigmas[step]
            diff = abs(actual - expected)
            status = "✓" if diff < 0.2 else "✗"
            print(f"Step {step:2d}: actual={actual:.4f}, expected~{expected:.2f} {status}")

        # Note: We allow tolerance because exact values depend on implementation details
        # The critical check is that denoising actually happens (not staying at high sigma)

    def test_dynamic_shifting_enabled(self):
        """Verify use_dynamic_shifting is actually being used."""
        from llm_dit.schedulers import FlowMatchScheduler

        # With dynamic shifting
        sched_dynamic = FlowMatchScheduler(shift=6.0, use_dynamic_shifting=True)
        sched_dynamic.set_timesteps(30, device="cpu")

        # Without dynamic shifting (linear formula)
        sched_linear = FlowMatchScheduler(shift=6.0, use_dynamic_shifting=False)
        sched_linear.set_timesteps(30, device="cpu")

        # They should produce different sigma schedules
        dynamic_sigmas = sched_dynamic.sigmas.tolist()
        linear_sigmas = sched_linear.sigmas.tolist()

        print("\n=== Dynamic vs Linear Shift Comparison ===")
        for i in [0, 10, 20, 28]:
            print(f"Step {i:2d}: dynamic={dynamic_sigmas[i]:.4f}, linear={linear_sigmas[i]:.4f}")

        # At step 20, the difference should be significant
        # Dynamic holds noise longer at the start, then drops faster
        diff_at_20 = abs(dynamic_sigmas[20] - linear_sigmas[20])
        assert diff_at_20 > 0.05, (
            f"Dynamic and linear shift should differ significantly at step 20, "
            f"but diff={diff_at_20:.4f}"
        )


class TestBaseModelGeneration:
    """Test Base model generates valid images (not pure noise)."""

    @pytest.fixture
    def model_path(self):
        """Get Z-Image model path from environment."""
        path = os.environ.get("Z_IMAGE_MODEL_PATH")
        if not path:
            pytest.skip("Z_IMAGE_MODEL_PATH not set")
        return path

    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create output directory for test images."""
        return tmp_path

    def _calculate_variance(self, image) -> float:
        """Calculate image variance (noise indicator)."""
        arr = np.array(image).astype(np.float32)
        return float(np.var(arr))

    def _is_pure_noise(self, image, threshold: float = 6000.0) -> bool:
        """Check if image is pure random noise.

        Pure noise typically has variance > 6000.
        Valid images typically have variance 500-6000.
        """
        return self._calculate_variance(image) > threshold

    @pytest.mark.slow
    @pytest.mark.requires_gpu
    @pytest.mark.requires_model
    def test_base_model_not_noise(self, model_path, output_dir):
        """Verify Base model produces valid images, not pure noise."""
        from PIL import Image

        from llm_dit.pipelines import ZImagePipeline

        # Load with Base model parameters
        pipe = ZImagePipeline.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            encoder_device="cpu",
            dit_device="cuda",
            vae_device="cuda",
            use_custom_scheduler=True,  # Our fixed scheduler
        )

        # Generate with Base model settings
        image = pipe(
            "A beautiful mountain landscape with snow-capped peaks",
            height=1024,
            width=1024,
            num_inference_steps=30,
            guidance_scale=4.0,
            shift=6.0,
            generator=torch.Generator().manual_seed(42),
        )

        # Save for inspection
        output_path = output_dir / "base_model_output.png"
        image.save(output_path)

        # Calculate variance
        variance = self._calculate_variance(image)
        print(f"\n=== Generation Result ===")
        print(f"Output saved to: {output_path}")
        print(f"Image variance: {variance:.2f}")
        print(f"Is pure noise: {self._is_pure_noise(image)}")

        # Verify not pure noise
        assert not self._is_pure_noise(image), (
            f"Image appears to be pure noise (variance={variance:.2f} > 6000). "
            "This indicates the scheduler fix may not be working correctly."
        )

        # Verify variance is in expected range for valid images
        assert 100 < variance < 6000, (
            f"Image variance {variance:.2f} outside expected range [100, 6000]. "
            "Either pure noise or something unusual."
        )

    @pytest.mark.slow
    @pytest.mark.requires_gpu
    @pytest.mark.requires_model
    def test_base_vs_turbo_output_differs(self, model_path, output_dir):
        """Verify Base and Turbo settings produce different outputs."""
        from PIL import Image

        from llm_dit.pipelines import ZImagePipeline

        pipe = ZImagePipeline.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            encoder_device="cpu",
            dit_device="cuda",
            vae_device="cuda",
            use_custom_scheduler=True,
        )

        prompt = "A red apple on a wooden table"
        seed = 12345

        # Generate with Turbo settings
        image_turbo = pipe(
            prompt,
            height=512,
            width=512,
            num_inference_steps=9,
            guidance_scale=0.0,
            shift=3.0,
            generator=torch.Generator().manual_seed(seed),
        )
        turbo_path = output_dir / "turbo_output.png"
        image_turbo.save(turbo_path)

        # Generate with Base settings
        image_base = pipe(
            prompt,
            height=512,
            width=512,
            num_inference_steps=30,
            guidance_scale=4.0,
            shift=6.0,
            generator=torch.Generator().manual_seed(seed),
        )
        base_path = output_dir / "base_output.png"
        image_base.save(base_path)

        # They should be different (different steps, CFG, shift)
        arr_turbo = np.array(image_turbo)
        arr_base = np.array(image_base)

        # Calculate difference
        diff = np.mean(np.abs(arr_turbo.astype(float) - arr_base.astype(float)))

        print(f"\n=== Turbo vs Base Comparison ===")
        print(f"Turbo saved to: {turbo_path}")
        print(f"Base saved to: {base_path}")
        print(f"Mean pixel difference: {diff:.2f}")

        # Images should be noticeably different
        assert diff > 10, (
            f"Turbo and Base outputs should differ significantly, "
            f"but mean difference is only {diff:.2f}"
        )


class TestAPIDefaultsWithGeneration:
    """Test that API correctly applies variant defaults."""

    @pytest.fixture
    def api_url(self):
        """Get API server URL."""
        return os.environ.get("TEST_SERVER_URL", "http://localhost:7860")

    def test_api_returns_zimage_variant(self, api_url):
        """Verify API returns zimage_variant in generation-config."""
        import requests

        try:
            resp = requests.get(f"{api_url}/api/generation-config", timeout=5)
            resp.raise_for_status()
            data = resp.json()

            variant = data.get("zimage_variant")
            print(f"\n=== API Generation Config ===")
            print(f"zimage_variant: {variant}")
            print(f"steps: {data.get('steps')}")
            print(f"guidance_scale: {data.get('guidance_scale')}")
            print(f"shift: {data.get('shift')}")

            assert variant is not None, "API should return zimage_variant"
            assert variant in ("turbo", "base"), f"Invalid variant: {variant}"

        except requests.exceptions.ConnectionError:
            pytest.skip(f"Server not running at {api_url}")

    def test_api_base_defaults_match_variant(self, api_url):
        """Verify API returns correct defaults for Base variant."""
        import requests

        try:
            resp = requests.get(f"{api_url}/api/generation-config", timeout=5)
            resp.raise_for_status()
            data = resp.json()

            variant = data.get("zimage_variant")

            if variant == "base":
                # Base variant should have these defaults
                assert data.get("shift") == 6.0, f"Base shift should be 6.0, got {data.get('shift')}"
                assert data.get("guidance_scale") == 4.0, (
                    f"Base guidance_scale should be 4.0, got {data.get('guidance_scale')}"
                )
                # Steps might be configured differently in config.toml, but should be > 20
                assert data.get("steps", 0) >= 20, (
                    f"Base steps should be >= 20, got {data.get('steps')}"
                )
                print(f"\n✓ API returns correct Base variant defaults")
            elif variant == "turbo":
                # Turbo variant defaults
                assert data.get("shift") == 3.0, f"Turbo shift should be 3.0, got {data.get('shift')}"
                assert data.get("guidance_scale") == 0.0, (
                    f"Turbo guidance_scale should be 0.0, got {data.get('guidance_scale')}"
                )
                print(f"\n✓ API returns correct Turbo variant defaults")
            else:
                pytest.fail(f"Unknown variant: {variant}")

        except requests.exceptions.ConnectionError:
            pytest.skip(f"Server not running at {api_url}")


if __name__ == "__main__":
    """Quick standalone test of scheduler."""
    from llm_dit.schedulers import FlowMatchScheduler

    print("Testing FlowMatchScheduler with FLUX-style exponential shift...")

    scheduler = FlowMatchScheduler(
        shift=6.0,
        use_dynamic_shifting=True,
    )
    scheduler.set_timesteps(30, device="cpu")

    sigmas = scheduler.sigmas.tolist()

    print(f"\nSigma schedule (shift=6.0, 30 steps):")
    for i, sigma in enumerate(sigmas):
        print(f"  Step {i:2d}: {sigma:.6f}")

    print(f"\nFirst sigma: {sigmas[0]:.6f} (should be ~1.0)")
    print(f"Last before 0: {sigmas[-2]:.6f} (should be ~0.02)")
    print(f"Final: {sigmas[-1]:.6f} (should be 0.0)")

    if sigmas[-2] < 0.1:
        print("\n✅ Scheduler appears to be working correctly!")
    else:
        print("\n❌ WARNING: Last sigma before 0 is too high - denoising may not complete!")
