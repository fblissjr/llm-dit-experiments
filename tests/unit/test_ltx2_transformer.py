"""
Tests for LTX-2 transformer implementation.

Last Updated: 2026-01-17

These tests verify the pure PyTorch LTX-2 transformer port matches
the expected behavior from the official implementation.

Run with: uv run pytest tests/unit/test_ltx2_transformer.py -v
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import patch

# Import our implementation
from llm_dit.models.ltx2_rope import (
    LTXRopeType,
    apply_rotary_emb,
)
from llm_dit.models.ltx2_components import (
    GELUApprox,
    FeedForward,
    Timesteps,
    TimestepEmbedding,
    AdaLayerNormSingle,
    PixArtAlphaTextProjection,
    rms_norm,
)
from llm_dit.models.ltx2_attention import (
    Attention,
    AttentionFunction,
)
from llm_dit.models.ltx2_transformer import (
    LTX2Transformer,
    LTXModelType,
    TransformerConfig,
    BasicTransformerBlock,
)
from llm_dit.models.ltx2_loader import (
    map_key,
    is_audio_key,
    load_config,
    get_model_info,
)


# ============================================================================
# RoPE Tests
# ============================================================================

class TestRoPE:
    """Tests for rotary position embedding implementation."""

    def test_rope_type_enum(self):
        """Test RoPE type enum values."""
        assert LTXRopeType.INTERLEAVED.value == "interleaved"
        assert LTXRopeType.SPLIT.value == "split"

    def test_apply_rotary_emb_interleaved(self):
        """Test interleaved RoPE application."""
        batch, seq = 2, 100
        head_dim = 128

        # Create input tensor [B, T, D]
        x = torch.randn(batch, seq, head_dim)

        # For interleaved RoPE, cos/sin frequencies should match full dimension
        # since each pair of adjacent dimensions uses the same cos/sin
        cos_freq = torch.ones(seq, head_dim)
        sin_freq = torch.zeros(seq, head_dim)

        # Apply RoPE
        out = apply_rotary_emb(x, (cos_freq, sin_freq), LTXRopeType.INTERLEAVED)

        assert out.shape == x.shape

    def test_apply_rotary_emb_split(self):
        """Test split RoPE application."""
        batch, seq = 2, 100
        head_dim = 128

        x = torch.randn(batch, seq, head_dim)
        # For split RoPE, frequencies match half the dimension (first/second half rotated together)
        half_dim = head_dim // 2
        cos_freq = torch.ones(seq, half_dim)
        sin_freq = torch.zeros(seq, half_dim)

        out = apply_rotary_emb(x, (cos_freq, sin_freq), LTXRopeType.SPLIT)

        assert out.shape == x.shape


# ============================================================================
# Component Tests
# ============================================================================

class TestComponents:
    """Tests for core transformer components."""

    def test_gelu_approx(self):
        """Test GELU approximation with projection."""
        dim_in, dim_out = 4096, 4096
        gelu = GELUApprox(dim_in=dim_in, dim_out=dim_out)
        x = torch.randn(2, 10, dim_in)
        y = gelu(x)
        assert y.shape == (2, 10, dim_out)
        # Should not be identity
        assert not torch.allclose(x, y)

    def test_feedforward_shape(self):
        """Test FFN output shape matches input."""
        dim = 4096
        ff = FeedForward(dim=dim, dim_out=dim, mult=4)

        x = torch.randn(2, 100, dim)
        y = ff(x)

        assert y.shape == x.shape

    def test_timesteps_embedding(self):
        """Test timestep sinusoidal embedding."""
        timesteps_mod = Timesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0  # Required parameter
        )
        t = torch.tensor([0.0, 0.5, 1.0])
        emb = timesteps_mod(t)
        assert emb.shape == (3, 256)

    def test_timestep_embedding_projection(self):
        """Test timestep MLP projection."""
        te = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=4096,  # Correct param name
        )
        x = torch.randn(2, 256)
        y = te(x)
        assert y.shape == (2, 4096)

    def test_adaln_single_output(self):
        """Test AdaLN produces correct timestep/embedded output."""
        embedding_dim = 4096
        embedding_coefficient = 6  # default: shift/scale/gate for attn + FFN
        adaln = AdaLayerNormSingle(embedding_dim=embedding_dim)

        # Forward pass with timestep
        timestep = torch.tensor([0.5, 0.5])
        scale_shift_values, embedded_timestep = adaln(timestep, hidden_dtype=torch.float32)

        # scale_shift_values is [B, embedding_coefficient * embedding_dim] (flat)
        assert scale_shift_values.shape == (2, embedding_coefficient * embedding_dim)
        # embedded_timestep is the raw embedding
        assert embedded_timestep.shape == (2, embedding_dim)

    def test_text_projection(self):
        """Test caption projection dimensions."""
        proj = PixArtAlphaTextProjection(
            in_features=3840,  # Gemma3 dim
            hidden_size=4096,  # LTX-2 hidden
        )
        x = torch.randn(2, 100, 3840)
        y = proj(x)
        assert y.shape == (2, 100, 4096)

    def test_rms_norm(self):
        """Test RMS normalization."""
        x = torch.randn(2, 100, 64)
        y = rms_norm(x, weight=None, eps=1e-6)
        assert y.shape == x.shape
        # Output should have roughly unit variance
        assert torch.abs(y.pow(2).mean() - 1.0) < 0.5


# ============================================================================
# Attention Tests
# ============================================================================

class TestAttention:
    """Tests for multi-head attention implementation."""

    def test_self_attention_shape(self):
        """Test self-attention output shape."""
        attn = Attention(
            query_dim=4096,
            heads=32,
            dim_head=128,
            # No bias or qk_norm params - they're built-in
        )

        x = torch.randn(2, 100, 4096)
        y = attn(x)

        assert y.shape == x.shape

    def test_cross_attention_shape(self):
        """Test cross-attention output shape."""
        attn = Attention(
            query_dim=4096,
            heads=32,
            dim_head=128,
            context_dim=4096,  # Cross-attention
        )

        x = torch.randn(2, 100, 4096)
        context = torch.randn(2, 50, 4096)  # Different seq length
        y = attn(x, context=context)

        assert y.shape == x.shape

    def test_attention_function_enum(self):
        """Test attention function enum has expected backends."""
        assert AttentionFunction.PYTORCH is not None
        assert AttentionFunction.XFORMERS is not None
        assert AttentionFunction.FLASH_ATTENTION_3 is not None  # Correct name
        assert AttentionFunction.DEFAULT is not None


# ============================================================================
# Transformer Block Tests
# ============================================================================

class TestTransformerBlock:
    """Tests for BasicTransformerBlock."""

    @pytest.fixture
    def config(self):
        """Create a minimal config for testing."""
        return TransformerConfig(
            dim=4096,
            heads=32,
            d_head=128,
            context_dim=4096,
        )

    def test_block_instantiation(self, config):
        """Test block can be instantiated with config."""
        # BasicTransformerBlock requires idx (block index in transformer)
        block = BasicTransformerBlock(config=config, idx=0)
        assert block is not None


# ============================================================================
# Key Mapping Tests
# ============================================================================

class TestKeyMapping:
    """Tests for diffusers to our key mapping."""

    def test_basic_mappings(self):
        """Test key mappings from DIFFUSERS_TO_OURS."""
        # Test proj_in -> patchify_proj
        assert map_key("proj_in.weight") == "patchify_proj.weight"

        # Test time_embed
        assert "adaln_single" in map_key("time_embed.linear.weight")

        # Test norm_q/norm_k -> q_norm/k_norm
        assert "q_norm" in map_key("attn.norm_q.weight")
        assert "k_norm" in map_key("attn.norm_k.weight")

    def test_audio_key_detection(self):
        """Test audio key filtering."""
        # Audio keys
        assert is_audio_key("audio_proj.weight")
        assert is_audio_key("av_cross_attn.weight")
        assert is_audio_key("transformer_blocks.0.audio_attn.weight")
        assert is_audio_key("a2v_cross_attn.weight")

        # Video keys (should NOT be detected as audio)
        assert not is_audio_key("video_proj.weight")
        assert not is_audio_key("transformer_blocks.0.attn1.weight")
        assert not is_audio_key("caption_projection.weight")


# ============================================================================
# Config Loading Tests
# ============================================================================

class TestConfigLoading:
    """Tests for config and checkpoint handling."""

    def test_default_config_values(self):
        """Test default config has expected structure."""
        # Create temp path that doesn't exist
        fake_path = Path("/nonexistent/path")

        with patch("builtins.open", side_effect=FileNotFoundError):
            config = load_config(fake_path)

        assert config["num_attention_heads"] == 32
        assert config["attention_head_dim"] == 128
        assert config["num_layers"] == 48
        assert config["cross_attention_dim"] == 4096
        assert config["caption_channels"] == 3840


# ============================================================================
# Transformer Integration Tests
# ============================================================================

class TestLTX2Transformer:
    """Integration tests for full transformer."""

    @pytest.fixture
    def mini_transformer(self):
        """Create a minimal LTX2Transformer for testing (1 layer)."""
        return LTX2Transformer(
            model_type=LTXModelType.VideoOnly,
            num_attention_heads=4,
            attention_head_dim=32,
            in_channels=16,
            out_channels=16,
            num_layers=1,  # Minimal for speed
            cross_attention_dim=128,
            caption_channels=64,
            positional_embedding_max_pos=[4, 8, 8],
        )

    def test_transformer_instantiation(self, mini_transformer):
        """Test transformer can be instantiated."""
        assert mini_transformer is not None
        assert mini_transformer.num_layers == 1

    def test_model_type_enum(self):
        """Test model type enum properties."""
        assert LTXModelType.VideoOnly.is_video_enabled()
        assert not LTXModelType.VideoOnly.is_audio_enabled()
        assert LTXModelType.AudioVideo.is_video_enabled()
        assert LTXModelType.AudioVideo.is_audio_enabled()

    def test_get_num_params(self, mini_transformer):
        """Test parameter counting."""
        num_params = mini_transformer.get_num_params()
        assert num_params > 0
        # Mini model should be small
        assert num_params < 10_000_000  # < 10M params


# ============================================================================
# Slow Tests (require checkpoint)
# ============================================================================

@pytest.mark.slow
class TestWithCheckpoint:
    """Tests that require actual LTX-2 checkpoint. Skip in CI."""

    @pytest.fixture
    def checkpoint_path(self):
        """Get checkpoint path, skip if not available."""
        paths = [
            Path.home() / "Storage/LTX-2/transformer",
            Path.home() / "models/LTX-2/transformer",
            Path("models/LTX-2/transformer"),
        ]
        for path in paths:
            if path.exists():
                return path
        pytest.skip("LTX-2 checkpoint not found")

    def test_config_loading(self, checkpoint_path):
        """Test loading config from real checkpoint."""
        config = load_config(checkpoint_path)
        assert config["num_layers"] == 48
        assert config["num_attention_heads"] == 32

    def test_model_info(self, checkpoint_path):
        """Test getting model info without loading weights."""
        info = get_model_info(checkpoint_path)
        assert info["num_layers"] == 48
        assert info["hidden_dim"] == 4096
        assert info["estimated_size_bf16_gb"] > 30  # Should be ~38GB


# Run with: uv run pytest tests/unit/test_ltx2_transformer.py -v
# Run slow tests: uv run pytest tests/unit/test_ltx2_transformer.py -v --slow
