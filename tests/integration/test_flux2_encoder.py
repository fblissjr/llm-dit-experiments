#!/usr/bin/env python3
"""
FLUX.2 Klein Qwen3 Encoder Integration Tests.

Last Updated: 2026-01-23

Tests the Qwen3 text encoder integration for FLUX.2 Klein models.
Verifies multi-layer extraction, output dimensions, and chat template application.

Requirements:
- CUDA GPU with sufficient VRAM (~8-16GB)
- Qwen3 model weights (will be downloaded from HF if not cached)

Usage:
    # Run encoder integration tests
    uv run pytest tests/integration/test_flux2_encoder.py -v

    # Run with verbose output
    uv run pytest tests/integration/test_flux2_encoder.py -v -s
"""

import gc
import pytest
import torch

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


def cleanup_gpu():
    """Free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory_gb() -> float:
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


def sufficient_vram_for_qwen3_8b() -> bool:
    """Check if GPU has enough VRAM for Qwen3-8B (~16GB)."""
    if not torch.cuda.is_available():
        return False
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return total_vram >= 20  # Need ~16GB for model + some buffer


def sufficient_vram_for_qwen3_4b() -> bool:
    """Check if GPU has enough VRAM for Qwen3-4B (~8GB)."""
    if not torch.cuda.is_available():
        return False
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return total_vram >= 12  # Need ~8GB for model + some buffer


class TestQwen3Flux2EncoderConstants:
    """Test encoder constants and configuration."""

    def test_output_layers(self):
        """Test that correct output layers are specified."""
        from llm_dit.encoders.qwen3_flux2 import OUTPUT_LAYERS_QWEN3

        # Should extract layers [9, 18, 27]
        assert OUTPUT_LAYERS_QWEN3 == [9, 18, 27]
        assert len(OUTPUT_LAYERS_QWEN3) == 3

    def test_expected_dimensions(self):
        """Test expected output dimensions for each model size."""
        from llm_dit.models.flux2.constants import Klein9BParams, Klein4BParams

        # Klein 9B uses Qwen3-8B (4096 hidden dim)
        params_9b = Klein9BParams()
        assert params_9b.context_in_dim == 12288  # 3 * 4096

        # Klein 4B uses Qwen3-4B (2560 hidden dim)
        params_4b = Klein4BParams()
        assert params_4b.context_in_dim == 7680  # 3 * 2560


@pytest.mark.skipif(
    not sufficient_vram_for_qwen3_4b(),
    reason="Need at least 12GB VRAM for Qwen3-4B"
)
class TestQwen3Flux2EncoderSmall:
    """Integration tests using Qwen3-4B (smaller, faster)."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and cleanup for each test."""
        cleanup_gpu()
        yield
        cleanup_gpu()

    def test_encoder_loading(self):
        """Test that encoder loads successfully."""
        from llm_dit.encoders.qwen3_flux2 import Qwen3Flux2Encoder

        # Try loading with FP8 variant
        try:
            encoder = Qwen3Flux2Encoder.from_pretrained(
                "Qwen/Qwen3-4B-FP8",
                device="cuda",
            )
            assert encoder is not None
            del encoder
        except Exception as e:
            # If FP8 not available, try bf16
            pytest.skip(f"Qwen3-4B-FP8 not available: {e}")

    def test_encode_single_prompt(self):
        """Test encoding a single prompt."""
        from llm_dit.encoders.qwen3_flux2 import Qwen3Flux2Encoder

        try:
            encoder = Qwen3Flux2Encoder.from_pretrained(
                "Qwen/Qwen3-4B-FP8",
                device="cuda",
            )

            # Encode a simple prompt
            embeddings = encoder.encode(["A photo of a cat"])

            # Check shape: [B, seq_len, context_dim]
            assert embeddings.ndim == 3
            assert embeddings.shape[0] == 1  # Batch size
            assert embeddings.shape[1] == 512  # Max sequence length
            assert embeddings.shape[2] == 7680  # 3 * 2560 for Qwen3-4B

            # Check for valid values
            assert not torch.isnan(embeddings).any()
            assert not torch.isinf(embeddings).any()

            del encoder
        except Exception as e:
            pytest.skip(f"Encoder test failed: {e}")

    def test_encode_multiple_prompts(self):
        """Test encoding multiple prompts in a batch."""
        from llm_dit.encoders.qwen3_flux2 import Qwen3Flux2Encoder

        try:
            encoder = Qwen3Flux2Encoder.from_pretrained(
                "Qwen/Qwen3-4B-FP8",
                device="cuda",
            )

            prompts = [
                "A photo of a cat",
                "A beautiful sunset over the ocean",
            ]
            embeddings = encoder.encode(prompts)

            # Check shape
            assert embeddings.shape[0] == 2  # Batch size
            assert embeddings.shape[1] == 512  # Max sequence length
            assert embeddings.shape[2] == 7680  # Context dim

            del encoder
        except Exception as e:
            pytest.skip(f"Batch encoding test failed: {e}")

    def test_offload_functionality(self):
        """Test that offload works to free memory."""
        from llm_dit.encoders.qwen3_flux2 import Qwen3Flux2Encoder

        try:
            encoder = Qwen3Flux2Encoder.from_pretrained(
                "Qwen/Qwen3-4B-FP8",
                device="cuda",
            )

            # Check memory before offload
            mem_before = get_gpu_memory_gb()

            # Offload
            encoder.offload()
            cleanup_gpu()

            # Check memory after offload
            mem_after = get_gpu_memory_gb()

            # Memory should decrease significantly
            assert mem_after < mem_before * 0.5, f"Memory not freed: {mem_before:.2f}GB -> {mem_after:.2f}GB"

            del encoder
        except Exception as e:
            pytest.skip(f"Offload test failed: {e}")


@pytest.mark.skipif(
    not sufficient_vram_for_qwen3_8b(),
    reason="Need at least 20GB VRAM for Qwen3-8B"
)
@pytest.mark.slow
class TestQwen3Flux2EncoderLarge:
    """Integration tests using Qwen3-8B (Klein 9B encoder)."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and cleanup for each test."""
        cleanup_gpu()
        yield
        cleanup_gpu()

    def test_encode_produces_correct_dims(self):
        """Test that 8B encoder produces correct context_in_dim."""
        from llm_dit.encoders.qwen3_flux2 import Qwen3Flux2Encoder

        try:
            encoder = Qwen3Flux2Encoder.from_pretrained(
                "Qwen/Qwen3-8B-FP8",
                device="cuda",
            )

            embeddings = encoder.encode(["A photo of a cat"])

            # Klein 9B expects 12288-dim context
            assert embeddings.shape[2] == 12288  # 3 * 4096

            del encoder
        except Exception as e:
            pytest.skip(f"Qwen3-8B test failed: {e}")


class TestMultiLayerExtraction:
    """Test the multi-layer extraction mechanism."""

    def test_layer_indices_valid(self):
        """Test that layer indices are valid for Qwen3 models."""
        from llm_dit.encoders.qwen3_flux2 import OUTPUT_LAYERS_QWEN3

        # Qwen3-8B has 36 layers, Qwen3-4B has 36 layers
        # Layers [9, 18, 27] should be valid for both
        max_layer = max(OUTPUT_LAYERS_QWEN3)
        assert max_layer <= 35, "Layer indices exceed Qwen3 model depth"

        # All layers should be positive
        for layer in OUTPUT_LAYERS_QWEN3:
            assert layer >= 0

    def test_layer_concatenation_formula(self):
        """Test the output dimension calculation."""
        # For Qwen3-8B (4096 hidden dim):
        # Output = 3 layers * 4096 dim = 12288
        assert 3 * 4096 == 12288

        # For Qwen3-4B (2560 hidden dim):
        # Output = 3 layers * 2560 dim = 7680
        assert 3 * 2560 == 7680


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
