"""
LTX-2 Reference Comparison Tests.

Last Updated: 2026-01-18

End-to-end tests using official LTX-2 parameters for 1:1 comparison
with the reference implementation. These tests produce actual video
outputs for visual inspection.

Reference Parameters (from coderef/LTX-2):
- height: 512, width: 768
- num_frames: 121 (or 33 for quick tests)
- num_inference_steps: 40
- guidance_scale: 4.0
- seed: 10

Output: outputs/tests/ltx2/{test_name}_{timestamp}/

Requirements:
- CUDA GPU with 24GB+ VRAM (FP8 quantization enabled)
- LTX-2 model weights at models/LTX-2/

Usage:
    # Run all reference tests
    uv run pytest tests/e2e/test_ltx2_reference.py -v

    # Run quick smoke test only
    uv run pytest tests/e2e/test_ltx2_reference.py -v -k smoke

    # Run with video output inspection
    uv run pytest tests/e2e/test_ltx2_reference.py -v -s
"""

import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytest
import torch

# Skip all tests if CUDA not available
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def output_base() -> Path:
    """Get output base directory for test results."""
    base = Path("outputs/tests/ltx2")
    base.mkdir(parents=True, exist_ok=True)
    return base


@pytest.fixture
def output_dir(output_base, request) -> Path:
    """Get timestamped output directory for this test."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_name = request.node.name.replace("[", "_").replace("]", "")
    out_dir = output_base / f"{test_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.fixture(autouse=True)
def cleanup_gpu():
    """Clean up GPU memory before and after each test."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def models_available() -> bool:
    """Check if LTX-2 models are available."""
    transformer_path = Path("models/LTX-2/transformer")
    encoder_path = Path("models/LTX-2/text_encoder")
    return transformer_path.exists() and encoder_path.exists()


def sufficient_vram() -> bool:
    """Check if GPU has enough VRAM for FP8 model (~16GB needed)."""
    if not torch.cuda.is_available():
        return False
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return total_vram >= 16


# =============================================================================
# Test Prompts
# =============================================================================

# Import from fixtures
try:
    from tests.fixtures.prompts import ltx2 as test_prompts
    SMOKE_PROMPT = test_prompts.get_smoke_test_prompt()
    REFERENCE_PROMPTS = test_prompts.get_reference_prompts()
except ImportError:
    # Fallback if fixtures not importable
    SMOKE_PROMPT = "A cat walking"
    REFERENCE_PROMPTS = {
        "cat_walking": "A cat walking",
        "cat_playing": "A cat playing with a ball",
    }


# =============================================================================
# Reference Constants
# =============================================================================

from llm_dit.models.ltx2 import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    DEFAULT_NUM_FRAMES,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_SEED,
    get_reference_config,
    get_quick_test_config,
)


# =============================================================================
# Test Classes
# =============================================================================

class TestLTX2ReferenceSmoke:
    """Quick smoke tests with reduced parameters."""

    @pytest.mark.skipif(not models_available(), reason="LTX-2 models not found")
    @pytest.mark.skipif(not sufficient_vram(), reason="Insufficient VRAM")
    def test_smoke_generation(self, output_dir):
        """Quick smoke test with minimal parameters."""
        from experiments.ltx2.base import LTX2ExperimentBase

        class SmokeTest(LTX2ExperimentBase):
            def __init__(self, out_dir: Path):
                super().__init__("smoke_test")
                self._custom_output_dir = out_dir

            def setup(self):
                self.load_model(quantize=True)
                self.load_encoder()

            def run_iteration(self, config: dict) -> dict:
                return config

        # Run smoke test
        test = SmokeTest(output_dir)
        test.setup()

        # Encode
        embeds = test.encode(SMOKE_PROMPT)
        assert embeds is not None

        # Generate with minimal params (fast)
        quick_config = get_quick_test_config()
        video = test.generate_video(
            embeds,
            num_frames=9,  # Even shorter than quick config
            num_inference_steps=4,  # Minimum viable
            seed=DEFAULT_SEED,
        )

        # Verify output
        assert video is not None
        assert not torch.isnan(video).any()
        assert not torch.isinf(video).any()

        # Save metadata
        metadata = {
            "prompt": SMOKE_PROMPT,
            "num_frames": 9,
            "num_inference_steps": 4,
            "height": DEFAULT_HEIGHT,
            "width": DEFAULT_WIDTH,
            "seed": DEFAULT_SEED,
            "test_type": "smoke",
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Cleanup
        test.cleanup()


class TestLTX2ReferenceComparison:
    """Tests using official LTX-2 reference parameters."""

    @pytest.mark.slow
    @pytest.mark.skipif(not models_available(), reason="LTX-2 models not found")
    @pytest.mark.skipif(not sufficient_vram(), reason="Insufficient VRAM")
    def test_reference_t2v_short(self, output_dir):
        """Reference T2V with official params but shorter video (33 frames)."""
        from experiments.ltx2.base import LTX2ExperimentBase

        class ReferenceTest(LTX2ExperimentBase):
            def __init__(self, out_dir: Path):
                super().__init__("reference_t2v")
                self._custom_output_dir = out_dir

            def setup(self):
                self.load_model(quantize=True)
                self.load_encoder()
                self.load_vae()

            def run_iteration(self, config: dict) -> dict:
                return config

        test = ReferenceTest(output_dir)
        test.setup()

        prompt = REFERENCE_PROMPTS["cat_walking"]
        embeds = test.encode(prompt)

        # Use reference params but shorter video
        video = test.generate_video(
            embeds,
            num_frames=33,  # Shorter than official 121
            height=DEFAULT_HEIGHT,
            width=DEFAULT_WIDTH,
            num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
            guidance_scale=DEFAULT_GUIDANCE_SCALE,
            seed=DEFAULT_SEED,
        )

        assert video is not None
        # Video shape: [F, H, W, C]
        assert video.shape[0] == 33
        assert video.shape[1] == DEFAULT_HEIGHT
        assert video.shape[2] == DEFAULT_WIDTH

        # Save video
        test.save_video(video, str(output_dir / "reference_t2v.mp4"))

        # Save metadata
        metadata = {
            "prompt": prompt,
            "num_frames": 33,
            "height": DEFAULT_HEIGHT,
            "width": DEFAULT_WIDTH,
            "num_inference_steps": DEFAULT_NUM_INFERENCE_STEPS,
            "guidance_scale": DEFAULT_GUIDANCE_SCALE,
            "seed": DEFAULT_SEED,
            "test_type": "reference_t2v_short",
            "reference_params": get_reference_config(),
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        test.cleanup()

    @pytest.mark.slow
    @pytest.mark.skipif(not models_available(), reason="LTX-2 models not found")
    @pytest.mark.skipif(not sufficient_vram(), reason="Insufficient VRAM")
    def test_reference_t2v_full(self, output_dir):
        """Full reference T2V with official 121 frames.

        This is the authoritative 1:1 comparison test.
        Output should match official LTX-2 implementation.
        """
        from experiments.ltx2.base import LTX2ExperimentBase

        class ReferenceTest(LTX2ExperimentBase):
            def __init__(self, out_dir: Path):
                super().__init__("reference_t2v_full")
                self._custom_output_dir = out_dir

            def setup(self):
                self.load_model(quantize=True)
                self.load_encoder()
                self.load_vae()

            def run_iteration(self, config: dict) -> dict:
                return config

        test = ReferenceTest(output_dir)
        test.setup()

        prompt = REFERENCE_PROMPTS["cat_walking"]
        embeds = test.encode(prompt)

        # Full official reference parameters
        ref_config = get_reference_config()
        video = test.generate_video(
            embeds,
            num_frames=DEFAULT_NUM_FRAMES,  # 121
            height=DEFAULT_HEIGHT,
            width=DEFAULT_WIDTH,
            num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
            guidance_scale=DEFAULT_GUIDANCE_SCALE,
            seed=DEFAULT_SEED,
        )

        assert video is not None
        assert video.shape[0] == DEFAULT_NUM_FRAMES

        # Save video
        test.save_video(video, str(output_dir / "reference_t2v_full.mp4"))

        # Save metadata
        metadata = {
            "prompt": prompt,
            **ref_config,
            "test_type": "reference_t2v_full",
            "is_authoritative": True,
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        test.cleanup()


class TestLTX2ReferenceI2V:
    """Image-to-Video tests with conditioning."""

    @pytest.mark.slow
    @pytest.mark.skipif(not models_available(), reason="LTX-2 models not found")
    @pytest.mark.skipif(not sufficient_vram(), reason="Insufficient VRAM")
    def test_i2v_with_conditioning(self, output_dir):
        """Test I2V generation using conditioning system."""
        from experiments.ltx2.base import LTX2ExperimentBase
        from llm_dit.conditioning import (
            LatentState,
            VideoConditionByLatentIndex,
        )

        class I2VTest(LTX2ExperimentBase):
            def __init__(self, out_dir: Path):
                super().__init__("i2v_conditioning")
                self._custom_output_dir = out_dir

            def setup(self):
                self.load_model(quantize=True)
                self.load_encoder()
                self.load_vae()

            def run_iteration(self, config: dict) -> dict:
                return config

        test = I2VTest(output_dir)
        test.setup()

        # For now, create a random "image" latent as placeholder
        # In real usage, this would be VAE-encoded input image
        h_latent = DEFAULT_HEIGHT // 32
        w_latent = DEFAULT_WIDTH // 32
        image_latent = torch.randn(
            1, 128, 1, h_latent, w_latent,
            device="cuda",
            dtype=torch.bfloat16,
        )

        # Create conditioning
        cond = VideoConditionByLatentIndex(
            latent=image_latent,
            latent_idx=0,  # First frame
            strength=1.0,  # Full conditioning
        )

        prompt = "A cat walking through a garden"
        embeds = test.encode(prompt)

        # Generate with conditioning
        video = test.generate_video(
            embeds,
            num_frames=33,
            height=DEFAULT_HEIGHT,
            width=DEFAULT_WIDTH,
            num_inference_steps=20,  # Fewer steps for I2V
            guidance_scale=DEFAULT_GUIDANCE_SCALE,
            seed=DEFAULT_SEED,
            conditioning=[cond],  # Pass conditioning
        )

        assert video is not None

        # Save video
        test.save_video(video, str(output_dir / "i2v_test.mp4"))

        # Save metadata
        metadata = {
            "prompt": prompt,
            "num_frames": 33,
            "height": DEFAULT_HEIGHT,
            "width": DEFAULT_WIDTH,
            "num_inference_steps": 20,
            "guidance_scale": DEFAULT_GUIDANCE_SCALE,
            "seed": DEFAULT_SEED,
            "test_type": "i2v_conditioning",
            "conditioning": {
                "type": "VideoConditionByLatentIndex",
                "latent_idx": 0,
                "strength": 1.0,
            },
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        test.cleanup()
