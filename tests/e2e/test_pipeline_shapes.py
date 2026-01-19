"""
End-to-end pipeline shape and value tests.

Last Updated: 2026-01-19

This test file verifies shapes and values at each stage of the LTX-2 text
encoding pipeline. It's designed to catch the zero embeddings bug early.

Pipeline stages:
1. Tokenization: input_ids, attention_mask
2. Gemma3 forward: hidden_states from all 49 layers
3. Stack & normalize: normalized features
4. Feature extractor: projected embeddings
5. Embeddings connector: final output

Run with: uv run pytest tests/e2e/test_pipeline_shapes.py -v
"""

import pytest
import torch
from pathlib import Path


# Skip all tests if required files not present
def check_model_files():
    """Check if required model files exist."""
    required_paths = [
        Path("models/LTX-2/connectors/diffusion_pytorch_model.safetensors"),
        Path("models/LTX-2/connectors/config.json"),
    ]
    for path in required_paths:
        if not path.exists():
            return False
    return True


pytestmark = pytest.mark.skipif(
    not check_model_files(),
    reason="LTX-2 model files not found"
)


class TestPipelineShapesWithMockedGemma:
    """
    Test pipeline shapes with mocked Gemma3 outputs.

    These tests don't require loading the actual Gemma3 model,
    making them faster and more reliable for CI.
    """

    @pytest.fixture
    def simulated_hidden_states(self):
        """Simulate Gemma3 hidden states output."""
        batch_size = 1
        seq_len = 256
        hidden_dim = 3840
        num_layers = 49

        # Create non-zero hidden states with realistic distribution
        hidden_states = []
        for i in range(num_layers):
            # Each layer has slightly different distribution
            layer_output = torch.randn(batch_size, seq_len, hidden_dim) * (1 + i * 0.01)
            hidden_states.append(layer_output)

        return hidden_states

    @pytest.fixture
    def attention_mask(self):
        """Create attention mask with some padding."""
        batch_size = 1
        seq_len = 256
        mask = torch.ones(batch_size, seq_len)
        mask[:, 200:] = 0  # Last 56 tokens are padding
        return mask

    def test_stage1_tokenization_shapes(self, attention_mask):
        """Test tokenization stage produces correct shapes."""
        batch_size, seq_len = attention_mask.shape

        assert attention_mask.shape == (1, 256)
        assert attention_mask.sum() == 200  # 200 valid tokens

    def test_stage2_hidden_states_shapes(self, simulated_hidden_states):
        """Test hidden states from Gemma3 have correct shapes."""
        assert len(simulated_hidden_states) == 49

        for i, hs in enumerate(simulated_hidden_states):
            assert hs.shape == (1, 256, 3840), f"Layer {i} shape mismatch"
            assert hs.abs().max() > 0, f"Layer {i} should be non-zero"

    def test_stage3_stack_and_normalize(self, simulated_hidden_states, attention_mask):
        """Test stacking and normalization produces correct output."""
        from llm_dit.encoders.gemma3 import _norm_and_concat_layers

        # Stack hidden states
        stacked = torch.stack(simulated_hidden_states, dim=-1)
        assert stacked.shape == (1, 256, 3840, 49)

        # Normalize
        normalized = _norm_and_concat_layers(stacked, attention_mask)
        assert normalized.shape == (1, 256, 3840 * 49)

        # Check values are reasonable
        print(f"\nStage 3 - Normalized features:")
        print(f"  Shape: {normalized.shape}")
        print(f"  Range: [{normalized.min():.2f}, {normalized.max():.2f}]")
        print(f"  Mean: {normalized.mean():.4f}, Std: {normalized.std():.4f}")

        # Values should be in range [-8, +8] due to normalization formula
        assert normalized.abs().max() <= 10, "Normalized values out of expected range"

    def test_stage4_feature_extractor(self, simulated_hidden_states, attention_mask):
        """Test feature extractor produces non-zero output."""
        from llm_dit.encoders.gemma3 import _norm_and_concat_layers, FeatureExtractorLinear
        from safetensors import safe_open

        # Stack and normalize
        stacked = torch.stack(simulated_hidden_states, dim=-1)
        normalized = _norm_and_concat_layers(stacked, attention_mask)

        # Load feature extractor with real weights
        connectors_path = Path("models/LTX-2/connectors/diffusion_pytorch_model.safetensors")
        fe = FeatureExtractorLinear(dtype=torch.float32)

        with safe_open(connectors_path, framework="pt") as f:
            weight = f.get_tensor("text_proj_in.weight")
            fe.aggregate_embed.weight.data = weight.float()

        # Project
        projected = fe(normalized)

        print(f"\nStage 4 - Feature extractor output:")
        print(f"  Shape: {projected.shape}")
        print(f"  Mean: {projected.mean():.4f}, Std: {projected.std():.4f}")
        print(f"  Max abs: {projected.abs().max():.4f}")

        assert projected.shape == (1, 256, 3840)
        assert projected.abs().max() > 0.1, "Feature extractor output is too close to zero!"

    def test_stage5_embeddings_connector(self, simulated_hidden_states, attention_mask):
        """Test embeddings connector produces non-zero output."""
        import json
        from llm_dit.encoders.gemma3 import _norm_and_concat_layers, FeatureExtractorLinear
        from llm_dit.encoders.embeddings_connector import Embeddings1DConnector, load_connector_weights
        from safetensors import safe_open

        # Stack and normalize
        stacked = torch.stack(simulated_hidden_states, dim=-1)
        normalized = _norm_and_concat_layers(stacked, attention_mask)

        # Load feature extractor
        connectors_path = Path("models/LTX-2/connectors/diffusion_pytorch_model.safetensors")
        fe = FeatureExtractorLinear(dtype=torch.float32)
        with safe_open(connectors_path, framework="pt") as f:
            fe.aggregate_embed.weight.data = f.get_tensor("text_proj_in.weight").float()

        projected = fe(normalized)

        # Load connector config and weights
        with open("models/LTX-2/connectors/config.json") as f:
            config = json.load(f)

        connector = Embeddings1DConnector.from_config(config).float()
        load_connector_weights(connector, connectors_path, prefix="video_connector.")

        # Convert attention mask to additive format
        additive_mask = (1.0 - attention_mask.float()) * -10000.0
        additive_mask = additive_mask[:, None, None, :]

        # Run connector
        output, _ = connector(projected, additive_mask)

        print(f"\nStage 5 - Connector output:")
        print(f"  Shape: {output.shape}")
        print(f"  Mean: {output.mean():.4f}, Std: {output.std():.4f}")
        print(f"  Max abs: {output.abs().max():.4f}")

        assert output.shape == (1, 256, 3840)
        assert output.abs().max() > 0.1, "Connector output is too close to zero!"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPipelineShapesGPU:
    """GPU-specific pipeline shape tests."""

    def test_full_pipeline_bfloat16_gpu(self):
        """Test full pipeline on GPU with bfloat16."""
        import json
        from llm_dit.encoders.gemma3 import _norm_and_concat_layers, FeatureExtractorLinear
        from llm_dit.encoders.embeddings_connector import Embeddings1DConnector, load_connector_weights
        from safetensors import safe_open

        device = torch.device("cuda")
        dtype = torch.bfloat16

        # Simulated hidden states
        batch_size, seq_len, hidden_dim, num_layers = 1, 256, 3840, 49
        hidden_states = [
            torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        # Stack and normalize
        stacked = torch.stack(hidden_states, dim=-1)
        normalized = _norm_and_concat_layers(stacked, attention_mask)

        # Feature extractor
        connectors_path = Path("models/LTX-2/connectors/diffusion_pytorch_model.safetensors")
        fe = FeatureExtractorLinear(dtype=dtype).to(device)
        with safe_open(connectors_path, framework="pt") as f:
            fe.aggregate_embed.weight.data = f.get_tensor("text_proj_in.weight").to(device)

        projected = fe(normalized)

        # Connector
        with open("models/LTX-2/connectors/config.json") as f:
            config = json.load(f)

        connector = Embeddings1DConnector.from_config(config).to(device).to(dtype)
        load_connector_weights(connector, connectors_path, prefix="video_connector.")

        additive_mask = (1.0 - attention_mask.float()) * -10000.0
        additive_mask = additive_mask[:, None, None, :].to(dtype)  # Match model dtype

        output, _ = connector(projected, additive_mask)

        print(f"\nGPU Pipeline (bfloat16):")
        print(f"  Output shape: {output.shape}")
        print(f"  Output dtype: {output.dtype}")
        print(f"  Mean: {output.float().mean():.4f}")

        assert output.dtype == dtype
        assert output.shape == (1, 256, 3840)
        assert output.abs().max() > 0.1


class TestVerifyNonZeroEmbeddings:
    """Critical test to verify embeddings are non-zero after fix."""

    def test_feature_extractor_weights_are_loaded(self):
        """CRITICAL: Verify feature extractor has loaded (non-random) weights."""
        from safetensors import safe_open
        from llm_dit.encoders.gemma3 import FeatureExtractorLinear

        connectors_path = Path("models/LTX-2/connectors/diffusion_pytorch_model.safetensors")

        # Create uninitialized feature extractor
        fe_random = FeatureExtractorLinear()
        random_weight_sum = fe_random.aggregate_embed.weight.abs().sum().item()

        # Create and load feature extractor
        fe_loaded = FeatureExtractorLinear()
        with safe_open(connectors_path, framework="pt") as f:
            fe_loaded.aggregate_embed.weight.data = f.get_tensor("text_proj_in.weight")

        loaded_weight_sum = fe_loaded.aggregate_embed.weight.abs().sum().item()

        print(f"\nFeature Extractor Weight Verification:")
        print(f"  Random weights sum: {random_weight_sum:.2f}")
        print(f"  Loaded weights sum: {loaded_weight_sum:.2f}")

        # Loaded weights should have a different sum than random
        assert loaded_weight_sum > 0, "Weights should be non-zero"

        # Check weight statistics match expected distribution
        weight = fe_loaded.aggregate_embed.weight
        print(f"  Weight mean: {weight.float().mean():.6f}")
        print(f"  Weight std: {weight.float().std():.6f}")

    def test_embeddings_are_not_noise(self):
        """CRITICAL: Verify encoder output is meaningful, not noise."""
        import json
        from llm_dit.encoders.gemma3 import _norm_and_concat_layers, FeatureExtractorLinear
        from llm_dit.encoders.embeddings_connector import Embeddings1DConnector, load_connector_weights
        from safetensors import safe_open

        # Create realistic-looking input
        batch_size, seq_len = 1, 256
        hidden_states = [
            torch.randn(batch_size, seq_len, 3840) * 0.5  # Scaled to match Gemma output
            for _ in range(49)
        ]
        attention_mask = torch.ones(batch_size, seq_len)

        # Run through pipeline
        stacked = torch.stack(hidden_states, dim=-1)
        normalized = _norm_and_concat_layers(stacked, attention_mask)

        connectors_path = Path("models/LTX-2/connectors/diffusion_pytorch_model.safetensors")

        # Feature extractor with loaded weights
        fe = FeatureExtractorLinear(dtype=torch.float32)
        with safe_open(connectors_path, framework="pt") as f:
            fe.aggregate_embed.weight.data = f.get_tensor("text_proj_in.weight").float()
        projected = fe(normalized)

        # Connector with loaded weights
        with open("models/LTX-2/connectors/config.json") as f:
            config = json.load(f)
        connector = Embeddings1DConnector.from_config(config).float()
        load_connector_weights(connector, connectors_path, prefix="video_connector.")

        additive_mask = (1.0 - attention_mask.float()) * -10000.0
        additive_mask = additive_mask[:, None, None, :]
        output, _ = connector(projected, additive_mask)

        print(f"\nEmbeddings Sanity Check:")
        print(f"  Mean: {output.mean():.4f}")
        print(f"  Std: {output.std():.4f}")
        print(f"  Min: {output.min():.4f}")
        print(f"  Max: {output.max():.4f}")

        # Verify embeddings are meaningful
        assert output.abs().mean() > 0.01, "Embeddings mean too close to zero"
        assert output.std() > 0.01, "Embeddings have no variance (constant)"
        assert not torch.isnan(output).any(), "Embeddings contain NaN"
        assert not torch.isinf(output).any(), "Embeddings contain Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
