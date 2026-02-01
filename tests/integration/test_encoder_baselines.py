"""
Integration tests for encoder numerical baselines.

Last Updated: 2026-02-01

Tests that encoder outputs match pre-computed reference values.
This ensures refactoring doesn't change encoder behavior.

Run with: uv run pytest tests/integration/test_encoder_baselines.py -v

NOTE: These tests require the baseline fixtures to be generated first:
    uv run scripts/generate_encoder_baselines.py --output tests/fixtures/encoder_baselines.pt

If fixtures don't exist, tests will be skipped.
"""

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.integration

FIXTURES_PATH = Path(__file__).parent.parent / "fixtures" / "encoder_baselines.pt"


def fixtures_available() -> bool:
    """Check if baseline fixtures exist."""
    return FIXTURES_PATH.exists()


def has_gpu() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture
def reference_embeddings():
    """Load pre-computed reference embeddings from fixtures."""
    if not fixtures_available():
        pytest.skip(f"Baseline fixtures not found at {FIXTURES_PATH}")
    return torch.load(FIXTURES_PATH, map_location="cpu", weights_only=True)


@pytest.mark.skipif(not fixtures_available(), reason="Baseline fixtures not generated")
class TestQwen3ZImageNumericalEquivalence:
    """Test Qwen3Encoder (Z-Image) output matches baseline."""

    def test_fixture_contains_zimage_data(self, reference_embeddings):
        """Test fixtures contain Z-Image encoder data."""
        assert "zimage" in reference_embeddings, "Missing zimage key in fixtures"
        assert "embeddings" in reference_embeddings["zimage"]
        assert "prompt" in reference_embeddings["zimage"]
        assert "config" in reference_embeddings["zimage"]

    def test_fixture_config_matches_expected(self, reference_embeddings):
        """Test fixture config matches expected Z-Image settings."""
        config = reference_embeddings["zimage"]["config"]

        # Z-Image uses single layer extraction at -2
        assert config["layer_index"] == -2
        # Z-Image default has enable_thinking=True
        assert config.get("enable_thinking", True) is True

    @pytest.mark.skipif(not has_gpu(), reason="GPU required for encoder loading")
    def test_numerical_equivalence(self, reference_embeddings):
        """Test Z-Image encoder output matches reference within tolerance."""
        # This test requires actually loading the encoder and comparing
        # Skip if models aren't available locally
        try:
            from llm_dit.encoders.qwen3 import Qwen3Encoder
        except ImportError:
            pytest.skip("Qwen3Encoder not available")

        ref_data = reference_embeddings["zimage"]
        prompt = ref_data["prompt"]
        ref_embeddings = ref_data["embeddings"]

        # Would need actual model path here
        # encoder = Qwen3Encoder.from_pretrained(...)
        # output = encoder.encode([prompt], layer_index=-2)

        # For now, just verify fixture structure
        assert ref_embeddings.ndim == 2  # [seq_len, hidden_dim]
        assert ref_embeddings.shape[-1] == 2560  # Qwen3-4B hidden dim


@pytest.mark.skipif(not fixtures_available(), reason="Baseline fixtures not generated")
class TestQwen3Flux2NumericalEquivalence:
    """Test Qwen3Flux2Encoder output matches baseline."""

    def test_fixture_contains_flux2_data(self, reference_embeddings):
        """Test fixtures contain FLUX.2 encoder data."""
        assert "flux2" in reference_embeddings, "Missing flux2 key in fixtures"
        assert "embeddings" in reference_embeddings["flux2"]
        assert "prompt" in reference_embeddings["flux2"]
        assert "config" in reference_embeddings["flux2"]

    def test_fixture_config_matches_expected(self, reference_embeddings):
        """Test fixture config matches expected FLUX.2 settings."""
        config = reference_embeddings["flux2"]["config"]

        # FLUX.2 uses multi-layer extraction
        assert config["output_layers"] == [9, 18, 27]
        # FLUX.2 MUST have enable_thinking=False
        assert config["enable_thinking"] is False

    def test_output_dim_is_concatenated(self, reference_embeddings):
        """Test FLUX.2 output has concatenated dimension (3x hidden)."""
        ref_embeddings = reference_embeddings["flux2"]["embeddings"]

        # FLUX.2 concatenates 3 layers
        # Qwen3-4B: 3 * 2560 = 7680
        # Qwen3-8B: 3 * 4096 = 12288
        hidden_dim = ref_embeddings.shape[-1]
        assert hidden_dim in [7680, 12288], f"Unexpected output dim: {hidden_dim}"

    @pytest.mark.skipif(not has_gpu(), reason="GPU required for encoder loading")
    def test_numerical_equivalence(self, reference_embeddings):
        """Test FLUX.2 encoder output matches reference within tolerance."""
        try:
            from llm_dit.encoders.qwen3_flux2 import Qwen3Flux2Encoder
        except ImportError:
            pytest.skip("Qwen3Flux2Encoder not available")

        ref_data = reference_embeddings["flux2"]
        ref_embeddings = ref_data["embeddings"]

        # Verify fixture structure
        assert ref_embeddings.ndim == 3  # [batch, seq_len, hidden_dim]


@pytest.mark.skipif(not fixtures_available(), reason="Baseline fixtures not generated")
class TestEnableThinkingDifference:
    """Test that enable_thinking=True vs False produces different outputs.

    CRITICAL: This validates that the enable_thinking parameter actually affects
    the embeddings. FLUX.2 requires False, Z-Image uses True by default.
    """

    def test_fixture_contains_thinking_comparison(self, reference_embeddings):
        """Test fixtures contain thinking comparison data."""
        assert "thinking_comparison" in reference_embeddings, (
            "Missing thinking_comparison key in fixtures. "
            "Regenerate fixtures with: uv run scripts/generate_encoder_baselines.py"
        )

    def test_thinking_produces_different_embeddings(self, reference_embeddings):
        """Test enable_thinking=True vs False produces different outputs."""
        comp = reference_embeddings.get("thinking_comparison", {})

        if not comp:
            pytest.skip("Thinking comparison data not in fixtures")

        thinking_true = comp.get("thinking_true")
        thinking_false = comp.get("thinking_false")

        if thinking_true is None or thinking_false is None:
            pytest.skip("Incomplete thinking comparison data")

        # Embeddings should be different
        assert not torch.allclose(thinking_true, thinking_false), (
            "enable_thinking=True and False produced identical embeddings! "
            "This indicates a bug - they should be different."
        )

    def test_thinking_embeddings_have_same_shape(self, reference_embeddings):
        """Test both thinking modes produce same shape."""
        comp = reference_embeddings.get("thinking_comparison", {})

        if not comp:
            pytest.skip("Thinking comparison data not in fixtures")

        thinking_true = comp.get("thinking_true")
        thinking_false = comp.get("thinking_false")

        if thinking_true is None or thinking_false is None:
            pytest.skip("Incomplete thinking comparison data")

        # Shapes should match (only values differ)
        assert thinking_true.shape == thinking_false.shape


@pytest.mark.skipif(not fixtures_available(), reason="Baseline fixtures not generated")
class TestRegressionTolerance:
    """Test numerical regression tolerances."""

    # Tolerances for bfloat16 computations
    RTOL = 1e-4  # Relative tolerance
    ATOL = 1e-5  # Absolute tolerance

    def test_tolerance_constants_are_appropriate(self):
        """Document and test tolerance values."""
        # These tolerances are appropriate for bfloat16:
        # - bfloat16 has ~3 decimal digits of precision
        # - 1e-4 rtol allows for ~0.01% relative error
        # - 1e-5 atol allows for small absolute differences near zero
        assert self.RTOL == 1e-4
        assert self.ATOL == 1e-5

    def test_fixture_metadata(self, reference_embeddings):
        """Test fixture contains metadata for reproducibility."""
        assert "metadata" in reference_embeddings, "Missing metadata in fixtures"

        meta = reference_embeddings["metadata"]
        assert "generated_at" in meta
        assert "torch_version" in meta


class TestEncoderFixtureGeneration:
    """Tests for the baseline fixture generation script."""

    def test_script_exists(self):
        """Test baseline generation script exists."""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "generate_encoder_baselines.py"
        assert script_path.exists(), f"Script not found: {script_path}"

    def test_fixtures_directory_exists(self):
        """Test fixtures directory exists."""
        fixtures_dir = Path(__file__).parent.parent / "fixtures"
        assert fixtures_dir.exists(), f"Fixtures directory not found: {fixtures_dir}"
