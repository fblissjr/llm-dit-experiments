"""
Tests for Gemma3 Encoder and Embeddings1DConnector.

Last Updated: 2026-01-19

Tests verify:
1. Feature extractor weight loading from checkpoint
2. Embeddings1DConnector shape and forward pass
3. RoPE implementation in connector
4. Full encoder pipeline shapes
5. Non-zero embeddings output

Run with: uv run pytest tests/unit/test_gemma3_encoder.py -v
"""

import json
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

# Import connector components
from llm_dit.encoders.embeddings_connector import (
    Embeddings1DConnector,
    RopeType,
    Attention,
    FeedForward,
    BasicTransformerBlock1D,
    GELUApprox,
    precompute_freqs_cis,
    apply_rotary_emb,
    rms_norm,
    load_connector_weights,
)

# Import encoder
from llm_dit.encoders.gemma3 import (
    Gemma3Encoder,
    FeatureExtractorLinear,
    GEMMA3_HIDDEN_DIM,
    GEMMA3_NUM_LAYERS,
    GEMMA3_FEATURE_DIM,
    GEMMA3_OUTPUT_DIM,
    _norm_and_concat_layers,
)


# ============================================================================
# RoPE Tests
# ============================================================================

class TestRoPE:
    """Tests for rotary position embedding utilities."""

    def test_precompute_freqs_shape(self):
        """Test precomputed frequencies have correct shape."""
        seq_len = 256
        dim = 3840
        num_heads = 30

        indices_grid = torch.arange(seq_len, dtype=torch.float32)[None, None, :]
        cos, sin = precompute_freqs_cis(
            indices_grid=indices_grid,
            dim=dim,
            out_dtype=torch.bfloat16,
            num_attention_heads=num_heads,
            rope_type=RopeType.SPLIT,
        )

        # Split RoPE: [B, H, T, D//2]
        assert cos.shape == (1, num_heads, seq_len, dim // 2 // num_heads)
        assert sin.shape == (1, num_heads, seq_len, dim // 2 // num_heads)

    def test_rope_values_bounded(self):
        """Test that cos/sin values are properly bounded."""
        indices_grid = torch.arange(100, dtype=torch.float32)[None, None, :]
        cos, sin = precompute_freqs_cis(
            indices_grid=indices_grid,
            dim=3840,
            out_dtype=torch.float32,
            num_attention_heads=30,
            rope_type=RopeType.SPLIT,
        )

        assert cos.abs().max() <= 1.0 + 1e-6
        assert sin.abs().max() <= 1.0 + 1e-6


# ============================================================================
# Component Tests
# ============================================================================

class TestGELUApprox:
    """Tests for GELU approximation module."""

    def test_gelu_output_shape(self):
        """Test GELU output shape."""
        gelu = GELUApprox(256, 1024)
        x = torch.randn(2, 10, 256)
        out = gelu(x)

        assert out.shape == (2, 10, 1024)


class TestRMSNorm:
    """Tests for RMS normalization function."""

    def test_rms_norm_output_shape(self):
        """Test RMSNorm preserves shape."""
        x = torch.randn(2, 10, 256)
        out = rms_norm(x)

        assert out.shape == x.shape

    def test_rms_norm_normalizes(self):
        """Test RMSNorm reduces variance."""
        x = torch.randn(2, 10, 256) * 100  # Large scale
        out = rms_norm(x)

        # Output should have unit RMS (approximately)
        rms = torch.sqrt((out ** 2).mean(dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), rtol=0.1)


class TestFeedForward:
    """Tests for feed-forward module."""

    def test_ff_output_shape(self):
        """Test FFN preserves shape."""
        ff = FeedForward(dim=3840, dim_out=3840)
        x = torch.randn(2, 256, 3840)
        out = ff(x)

        assert out.shape == x.shape


class TestAttention:
    """Tests for attention module."""

    def test_attention_output_shape(self):
        """Test attention output shape matches input."""
        dim = 3840
        attn = Attention(query_dim=dim, heads=30, dim_head=128)

        x = torch.randn(2, 256, dim)
        out = attn(x)

        assert out.shape == x.shape

    def test_attention_with_rope(self):
        """Test attention works with RoPE."""
        dim = 3840
        attn = Attention(query_dim=dim, heads=30, dim_head=128, rope_type=RopeType.SPLIT)

        batch, seq = 2, 256
        x = torch.randn(batch, seq, dim)

        # Generate RoPE frequencies
        indices_grid = torch.arange(seq, dtype=torch.float32)[None, None, :]
        pe = precompute_freqs_cis(
            indices_grid=indices_grid,
            dim=dim,
            out_dtype=x.dtype,
            num_attention_heads=30,
            rope_type=RopeType.SPLIT,
        )

        out = attn(x, pe=pe)

        assert out.shape == x.shape


class TestBasicTransformerBlock1D:
    """Tests for transformer block."""

    def test_block_output_shape(self):
        """Test block output shape matches input."""
        dim = 3840
        block = BasicTransformerBlock1D(dim=dim, heads=30, dim_head=128)

        x = torch.randn(2, 256, dim)
        out = block(x)

        assert out.shape == x.shape


# ============================================================================
# Embeddings1DConnector Tests
# ============================================================================

class TestEmbeddings1DConnector:
    """Tests for the full embeddings connector."""

    def test_connector_output_shape(self):
        """Test connector output shape."""
        connector = Embeddings1DConnector(
            attention_head_dim=128,
            num_attention_heads=30,
            num_layers=2,
            num_learnable_registers=128,
        )

        # Input must be divisible by num_learnable_registers
        batch, seq = 2, 256  # 256 is divisible by 128
        x = torch.randn(batch, seq, 3840)

        out, mask = connector(x)

        assert out.shape == (batch, seq, 3840)

    def test_connector_without_registers(self):
        """Test connector without learnable registers."""
        connector = Embeddings1DConnector(
            attention_head_dim=128,
            num_attention_heads=30,
            num_layers=2,
            num_learnable_registers=0,
        )

        batch, seq = 2, 256
        x = torch.randn(batch, seq, 3840)

        out, mask = connector(x)

        assert out.shape == (batch, seq, 3840)

    def test_connector_with_attention_mask(self):
        """Test connector with attention mask."""
        connector = Embeddings1DConnector(
            attention_head_dim=128,
            num_attention_heads=30,
            num_layers=2,
            num_learnable_registers=128,
        )

        batch, seq = 2, 256
        x = torch.randn(batch, seq, 3840)

        # Additive mask: 0 = valid, -10000 = padding
        mask = torch.zeros(batch, 1, 1, seq)
        mask[:, :, :, seq // 2:] = -10000

        out, new_mask = connector(x, attention_mask=mask)

        assert out.shape == (batch, seq, 3840)

    def test_connector_from_config(self):
        """Test connector creation from config dict."""
        config = {
            "video_connector_attention_head_dim": 128,
            "video_connector_num_attention_heads": 30,
            "video_connector_num_layers": 2,
            "video_connector_num_learnable_registers": 128,
            "rope_type": "split",
            "rope_theta": 10000.0,
            "rope_double_precision": True,
            "connector_rope_base_seq_len": 4096,
        }

        connector = Embeddings1DConnector.from_config(config)

        assert connector.num_attention_heads == 30
        assert connector.inner_dim == 3840
        assert connector.num_learnable_registers == 128
        assert connector.rope_type == RopeType.SPLIT


# ============================================================================
# Feature Extractor Tests
# ============================================================================

class TestFeatureExtractorLinear:
    """Tests for feature extractor linear projection."""

    def test_feature_extractor_shape(self):
        """Test feature extractor produces correct output shape."""
        fe = FeatureExtractorLinear()

        # Input: [B, T, 3840 * 49]
        x = torch.randn(2, 256, GEMMA3_FEATURE_DIM)
        out = fe(x)

        assert out.shape == (2, 256, GEMMA3_OUTPUT_DIM)

    def test_feature_extractor_dimensions(self):
        """Test feature extractor has correct weight dimensions."""
        fe = FeatureExtractorLinear()

        assert fe.aggregate_embed.in_features == GEMMA3_FEATURE_DIM  # 188160
        assert fe.aggregate_embed.out_features == GEMMA3_OUTPUT_DIM  # 3840
        assert fe.aggregate_embed.bias is None


# ============================================================================
# Normalization Tests
# ============================================================================

class TestNormAndConcatLayers:
    """Tests for layer normalization and concatenation."""

    def test_norm_concat_shape(self):
        """Test normalization produces correct output shape."""
        batch, seq, dim, layers = 2, 256, GEMMA3_HIDDEN_DIM, GEMMA3_NUM_LAYERS
        hidden_states = torch.randn(batch, seq, dim, layers)
        attention_mask = torch.ones(batch, seq)

        out = _norm_and_concat_layers(hidden_states, attention_mask)

        # Output should be [B, T, D * L]
        assert out.shape == (batch, seq, dim * layers)

    def test_norm_concat_with_padding(self):
        """Test normalization handles padding correctly."""
        batch, seq, dim, layers = 2, 256, GEMMA3_HIDDEN_DIM, GEMMA3_NUM_LAYERS
        hidden_states = torch.randn(batch, seq, dim, layers)

        # Half padding
        attention_mask = torch.ones(batch, seq)
        attention_mask[:, seq // 2:] = 0

        out = _norm_and_concat_layers(hidden_states, attention_mask)

        # Padding positions should be zero
        assert out[:, seq // 2:, :].abs().sum() == 0


# ============================================================================
# Integration Tests (require checkpoint files)
# ============================================================================

@pytest.fixture
def connectors_path():
    """Path to connectors checkpoint."""
    path = Path("models/LTX-2/connectors/diffusion_pytorch_model.safetensors")
    if not path.exists():
        pytest.skip(f"Checkpoint not found: {path}")
    return path


@pytest.fixture
def connectors_config():
    """Path to connectors config."""
    path = Path("models/LTX-2/connectors/config.json")
    if not path.exists():
        pytest.skip(f"Config not found: {path}")
    return path


class TestConnectorWeightLoading:
    """Tests for loading weights from checkpoint."""

    def test_load_connector_weights(self, connectors_path, connectors_config):
        """Test loading connector weights from checkpoint."""
        # Load config
        with open(connectors_config) as f:
            config = json.load(f)

        # Create connector
        connector = Embeddings1DConnector.from_config(config)

        # Load weights
        load_connector_weights(connector, connectors_path, prefix="video_connector.")

        # Verify weights are non-zero
        param_sum = sum(p.abs().sum().item() for p in connector.parameters())
        assert param_sum > 0, "Weights should be non-zero after loading"

    def test_feature_extractor_weights_loaded(self, connectors_path):
        """Test that feature extractor weights load correctly."""
        from safetensors import safe_open

        fe = FeatureExtractorLinear(dtype=torch.bfloat16)

        # Load weight from checkpoint
        with safe_open(connectors_path, framework="pt") as f:
            if "text_proj_in.weight" in f.keys():
                weight = f.get_tensor("text_proj_in.weight")
                fe.aggregate_embed.weight.data = weight

        # Verify weight is non-zero
        assert fe.aggregate_embed.weight.abs().sum() > 0
        assert fe.aggregate_embed.weight.shape == (GEMMA3_OUTPUT_DIM, GEMMA3_FEATURE_DIM)


# ============================================================================
# Pipeline Shape Trace Tests
# ============================================================================

class TestPipelineShapes:
    """Tests for end-to-end pipeline shape verification."""

    def test_full_pipeline_shapes(self, connectors_path, connectors_config):
        """Test shapes through full encoding pipeline."""
        from safetensors import safe_open

        # Load config
        with open(connectors_config) as f:
            config = json.load(f)

        batch_size = 1
        seq_len = 256
        num_layers = GEMMA3_NUM_LAYERS

        # Stage 1: Simulated Gemma3 hidden states
        hidden_states = torch.randn(batch_size, seq_len, GEMMA3_HIDDEN_DIM, num_layers)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, 200:] = 0  # Some padding

        print(f"\n=== Pipeline Shape Trace ===")
        print(f"Stage 1 - Hidden states: {hidden_states.shape}")

        # Stage 2: Normalize and concatenate
        normalized = _norm_and_concat_layers(hidden_states, attention_mask)
        print(f"Stage 2 - Normalized: {normalized.shape}")
        assert normalized.shape == (batch_size, seq_len, GEMMA3_FEATURE_DIM)

        # Stage 3: Feature extractor
        fe = FeatureExtractorLinear(dtype=torch.float32)
        with safe_open(connectors_path, framework="pt") as f:
            if "text_proj_in.weight" in f.keys():
                fe.aggregate_embed.weight.data = f.get_tensor("text_proj_in.weight").float()

        projected = fe(normalized)
        print(f"Stage 3 - Projected: {projected.shape}")
        print(f"         Mean: {projected.mean():.4f}, Std: {projected.std():.4f}")
        assert projected.shape == (batch_size, seq_len, GEMMA3_OUTPUT_DIM)
        assert projected.abs().max() > 0.1, "Projections should be non-zero"

        # Stage 4: Embeddings connector
        connector = Embeddings1DConnector.from_config(config)
        load_connector_weights(connector, connectors_path, prefix="video_connector.")

        # Convert mask to additive format
        additive_mask = (1.0 - attention_mask.float()) * -10000.0
        additive_mask = additive_mask[:, None, None, :]

        output, _ = connector(projected, additive_mask)
        print(f"Stage 4 - Connector output: {output.shape}")
        print(f"         Mean: {output.mean():.4f}, Std: {output.std():.4f}")
        assert output.shape == (batch_size, seq_len, GEMMA3_OUTPUT_DIM)
        assert output.abs().max() > 0.1, "Connector output should be non-zero"


# ============================================================================
# GPU Tests (optional)
# ============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPU:
    """Tests that require GPU."""

    def test_connector_bfloat16(self):
        """Test connector works with bfloat16 on GPU."""
        connector = Embeddings1DConnector(
            attention_head_dim=128,
            num_attention_heads=30,
            num_layers=2,
            num_learnable_registers=128,
        ).cuda().to(torch.bfloat16)

        x = torch.randn(1, 256, 3840, device="cuda", dtype=torch.bfloat16)
        out, _ = connector(x)

        assert out.dtype == torch.bfloat16
        assert out.shape == (1, 256, 3840)

    def test_feature_extractor_bfloat16_gpu(self):
        """Test feature extractor on GPU with bfloat16."""
        fe = FeatureExtractorLinear(dtype=torch.bfloat16).cuda()

        x = torch.randn(1, 256, GEMMA3_FEATURE_DIM, device="cuda", dtype=torch.bfloat16)
        out = fe(x)

        assert out.dtype == torch.bfloat16
        assert out.shape == (1, 256, GEMMA3_OUTPUT_DIM)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
