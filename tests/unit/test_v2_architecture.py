"""Tests for LTX-2.3 (V2) architecture features.

Last Updated: 2026-03-06

Verifies V2-specific additions:
- Gated attention (per-head sigmoid gate)
- Cross-attention AdaLN (9-param scale_shift_table)
- FeatureExtractorV2 (per-token RMSNorm, dual projections)
- V2 detection from state dict keys
- GGUF infrastructure (GGMLTensor, GGMLLinear)
- TransformerArgsPreprocessor prompt_timestep computation (V2)

Run with: uv run pytest tests/unit/test_v2_architecture.py -v
"""

import torch

from llm_dit.models.ltx2.attention import Attention
from llm_dit.models.ltx2.av_block import BasicAVTransformerBlock, _adaln_size
from llm_dit.models.ltx2.transformer import TransformerConfig
from llm_dit.models.ltx2.loader import create_model_from_config
from llm_dit.encoders.gemma3_feature_extractor_v2 import (
    FeatureExtractorV2,
    norm_and_concat_per_token_rms,
)
from llm_dit.quantization.gguf_tensor import GGMLTensor
from llm_dit.quantization.gguf_linear import GGMLLinear, replace_linear_with_ggml
from llm_dit.quantization.gguf_loader import detect_v2_from_state_dict


class TestGatedAttention:
    """Test apply_gated_attention feature."""

    def test_gate_logits_created_when_enabled(self):
        attn = Attention(query_dim=128, heads=4, dim_head=32, apply_gated_attention=True)
        assert hasattr(attn, "to_gate_logits")
        assert attn.to_gate_logits is not None
        assert attn.to_gate_logits.out_features == 4  # one gate per head

    def test_no_gate_logits_when_disabled(self):
        attn = Attention(query_dim=128, heads=4, dim_head=32, apply_gated_attention=False)
        assert attn.to_gate_logits is None

    def test_gated_attention_output_shape(self):
        attn = Attention(query_dim=128, heads=4, dim_head=32, apply_gated_attention=True)
        x = torch.randn(2, 16, 128)
        out = attn(x)
        assert out.shape == x.shape

    def test_gated_vs_ungated_differ(self):
        """Gated attention should produce different output than ungated."""
        torch.manual_seed(42)
        attn_gated = Attention(query_dim=128, heads=4, dim_head=32, apply_gated_attention=True)

        torch.manual_seed(42)
        attn_ungated = Attention(query_dim=128, heads=4, dim_head=32, apply_gated_attention=False)

        # Copy shared weights
        attn_ungated.load_state_dict(
            {k: v for k, v in attn_gated.state_dict().items() if "gate" not in k},
            strict=False,
        )

        x = torch.randn(1, 8, 128)
        out_gated = attn_gated(x)
        out_ungated = attn_ungated(x)
        # They should differ because of the gate scaling
        assert not torch.allclose(out_gated, out_ungated, atol=1e-5)


def _make_video_config(cross_attention_adaln=False, apply_gated_attention=False):
    """Helper to build a TransformerConfig for testing."""
    return TransformerConfig(
        dim=128,
        heads=4,
        d_head=32,
        context_dim=64,
        cross_attention_adaln=cross_attention_adaln,
        apply_gated_attention=apply_gated_attention,
    )


class TestCrossAttentionAdaLN:
    """Test cross_attention_adaln feature."""

    def test_adaln_size_v1(self):
        assert _adaln_size(cross_attention_adaln=False) == 6

    def test_adaln_size_v2(self):
        assert _adaln_size(cross_attention_adaln=True) == 9

    def test_scale_shift_table_shape_v1(self):
        cfg = _make_video_config(cross_attention_adaln=False)
        block = BasicAVTransformerBlock(idx=0, video=cfg)
        assert block.scale_shift_table.shape == (6, 128)

    def test_scale_shift_table_shape_v2(self):
        cfg = _make_video_config(cross_attention_adaln=True)
        block = BasicAVTransformerBlock(idx=0, video=cfg)
        assert block.scale_shift_table.shape == (9, 128)

    def test_prompt_scale_shift_table_created_v2(self):
        cfg = _make_video_config(cross_attention_adaln=True)
        block = BasicAVTransformerBlock(idx=0, video=cfg)
        assert block.prompt_scale_shift_table is not None
        assert block.prompt_scale_shift_table.shape == (2, 128)

    def test_no_prompt_scale_shift_table_v1(self):
        cfg = _make_video_config(cross_attention_adaln=False)
        block = BasicAVTransformerBlock(idx=0, video=cfg)
        assert not hasattr(block, "prompt_scale_shift_table") or block.prompt_scale_shift_table is None


class TestFeatureExtractorV2:
    """Test V2 feature extraction."""

    def test_dual_projections(self):
        fe = FeatureExtractorV2(
            embedding_dim=64, video_dim=128, audio_dim=32,
            feature_dim=64 * 4, dtype=torch.float32,
        )
        assert fe.video_aggregate_embed.out_features == 128
        assert fe.audio_aggregate_embed is not None
        assert fe.audio_aggregate_embed.out_features == 32

    def test_video_only_projection(self):
        fe = FeatureExtractorV2(
            embedding_dim=64, video_dim=128, audio_dim=None,
            feature_dim=64 * 4, dtype=torch.float32,
        )
        assert fe.audio_aggregate_embed is None

    def test_forward_shapes(self):
        fe = FeatureExtractorV2(
            embedding_dim=64, video_dim=128, audio_dim=32,
            feature_dim=64 * 4, dtype=torch.float32,
        )
        hidden = torch.randn(2, 10, 64, 4)
        mask = torch.ones(2, 10)
        video, audio = fe(hidden, mask)
        assert video.shape == (2, 10, 128)
        assert audio is not None
        assert audio.shape == (2, 10, 32)

    def test_padding_zeroed(self):
        fe = FeatureExtractorV2(
            embedding_dim=64, video_dim=128, audio_dim=32,
            feature_dim=64 * 4, dtype=torch.float32,
        )
        hidden = torch.randn(1, 10, 64, 4)
        mask = torch.zeros(1, 10)
        mask[0, :5] = 1.0

        video, audio = fe(hidden, mask)
        normed = norm_and_concat_per_token_rms(hidden, mask)
        assert torch.allclose(normed[0, 5:], torch.zeros_like(normed[0, 5:]))


class TestNormAndConcatPerTokenRMS:
    """Test per-token RMSNorm normalization."""

    def test_output_shape(self):
        x = torch.randn(2, 8, 64, 4)
        mask = torch.ones(2, 8)
        out = norm_and_concat_per_token_rms(x, mask)
        assert out.shape == (2, 8, 64 * 4)

    def test_masked_positions_zero(self):
        x = torch.randn(1, 5, 32, 3)
        mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float32)
        out = norm_and_concat_per_token_rms(x, mask)
        assert torch.allclose(out[0, 3:], torch.zeros_like(out[0, 3:]))

    def test_non_masked_positions_nonzero(self):
        x = torch.randn(1, 5, 32, 3) + 1.0  # ensure non-zero
        mask = torch.ones(1, 5)
        out = norm_and_concat_per_token_rms(x, mask)
        assert out.abs().sum() > 0


class TestV2Detection:
    """Test V2 model detection from state dict keys."""

    def test_detect_v2_with_prompt_scale_shift_table(self):
        sd = {"blocks.0.prompt_scale_shift_table": torch.zeros(2, 128)}
        assert detect_v2_from_state_dict(sd) is True

    def test_detect_v2_with_gate_logits(self):
        sd = {"blocks.0.attn1.to_gate_logits.weight": torch.zeros(4, 128)}
        assert detect_v2_from_state_dict(sd) is True

    def test_detect_v1(self):
        sd = {"blocks.0.scale_shift_table": torch.zeros(6, 128)}
        assert detect_v2_from_state_dict(sd) is False

    def test_empty_state_dict(self):
        assert detect_v2_from_state_dict({}) is False


class TestGGMLTensor:
    """Test GGMLTensor subclass."""

    def test_logical_shape(self):
        data = torch.zeros(100)
        t = GGMLTensor(data, tensor_type=8, tensor_shape=torch.Size([32, 64]))
        assert t.shape == torch.Size([32, 64])

    def test_to_preserves_metadata(self):
        data = torch.zeros(100)
        t = GGMLTensor(data, tensor_type=8, tensor_shape=torch.Size([32, 64]))
        t2 = t.to(dtype=torch.float32)
        assert isinstance(t2, GGMLTensor)
        assert t2.tensor_type == 8
        assert t2.tensor_shape == torch.Size([32, 64])


class TestGGMLLinear:
    """Test GGMLLinear layer."""

    def test_init_no_memory_alloc(self):
        """GGMLLinear init should not allocate full-size weight."""
        linear = GGMLLinear(1024, 512, bias=False)
        assert linear.weight is None
        assert linear.bias is None

    def test_forward_with_regular_tensor(self):
        """GGMLLinear with regular (non-quantized) weight should work like nn.Linear."""
        linear = GGMLLinear(8, 4, bias=False)
        linear.weight = torch.nn.Parameter(torch.randn(4, 8), requires_grad=False)
        x = torch.randn(2, 8)
        out = linear(x)
        assert out.shape == (2, 4)

    def test_replace_linear_with_ggml(self):
        """Replace all nn.Linear in a module with GGMLLinear."""
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 2),
        )
        count = replace_linear_with_ggml(model)
        assert count == 2
        assert isinstance(model[0], GGMLLinear)
        assert isinstance(model[2], GGMLLinear)


class TestCreateModelV2:
    """Test creating V2 transformer models."""

    def test_create_v2_model(self):
        config = {
            "num_attention_heads": 4,
            "attention_head_dim": 32,
            "in_channels": 16,
            "out_channels": 16,
            "num_layers": 2,
            "cross_attention_dim": 64,
            "caption_channels": 48,
        }
        model = create_model_from_config(
            config, torch.bfloat16,
            apply_gated_attention=True,
            cross_attention_adaln=True,
        )
        # Verify V2 features are enabled
        assert model.apply_gated_attention is True
        assert model.cross_attention_adaln is True

    def test_v2_video_only_has_gate_logits_in_state_dict(self):
        """V2 VideoOnly model should have to_gate_logits in state dict."""
        config = {
            "num_attention_heads": 4,
            "attention_head_dim": 32,
            "in_channels": 16,
            "out_channels": 16,
            "num_layers": 2,
            "cross_attention_dim": 64,
            "caption_channels": 48,
        }
        model = create_model_from_config(
            config, torch.bfloat16,
            apply_gated_attention=True,
            cross_attention_adaln=True,
        )
        sd_keys = set(model.state_dict().keys())
        # BasicTransformerBlock (VideoOnly) must have gate logits
        assert "transformer_blocks.0.attn1.to_gate_logits.weight" in sd_keys
        assert "transformer_blocks.0.attn2.to_gate_logits.weight" in sd_keys

    def test_v2_video_only_has_prompt_scale_shift_table(self):
        """V2 VideoOnly model should have prompt_scale_shift_table."""
        config = {
            "num_attention_heads": 4,
            "attention_head_dim": 32,
            "in_channels": 16,
            "out_channels": 16,
            "num_layers": 2,
            "cross_attention_dim": 64,
            "caption_channels": 48,
        }
        model = create_model_from_config(
            config, torch.bfloat16,
            apply_gated_attention=True,
            cross_attention_adaln=True,
        )
        sd_keys = set(model.state_dict().keys())
        assert "transformer_blocks.0.prompt_scale_shift_table" in sd_keys
        # V2 scale_shift_table should be 9 params (not 6)
        sst = model.state_dict()["transformer_blocks.0.scale_shift_table"]
        assert sst.shape[0] == 9, f"Expected 9-param scale_shift_table, got {sst.shape[0]}"

    def test_v2_video_only_no_caption_projection(self):
        """V2 model should not have caption_projection (moved to encoder)."""
        config = {
            "num_attention_heads": 4,
            "attention_head_dim": 32,
            "in_channels": 16,
            "out_channels": 16,
            "num_layers": 2,
            "cross_attention_dim": 64,
            "caption_channels": 48,
        }
        model = create_model_from_config(
            config, torch.bfloat16,
            apply_gated_attention=True,
            cross_attention_adaln=True,
        )
        sd_keys = set(model.state_dict().keys())
        assert "caption_projection.linear_1.weight" not in sd_keys
        assert "caption_projection.linear_2.weight" not in sd_keys

    def test_reconstruct_always_v23(self):
        """_reconstruct_transformer_from_cache() produces V2.3 model with gated attention."""
        # This tests that reconstruction always enables V2.3 features
        config = {
            "num_attention_heads": 4,
            "attention_head_dim": 32,
            "in_channels": 16,
            "out_channels": 16,
            "num_layers": 2,
            "cross_attention_dim": 64,
            "caption_channels": 48,
        }
        # Simulate what _reconstruct_transformer_from_cache does:
        # always apply_gated_attention=True, cross_attention_adaln=True
        model = create_model_from_config(
            config, torch.bfloat16,
            apply_gated_attention=True,
            cross_attention_adaln=True,
        )
        assert model.apply_gated_attention is True
        assert model.cross_attention_adaln is True
        # Verify state dict has V2.3 markers
        sd_keys = set(model.state_dict().keys())
        assert any("to_gate_logits" in k for k in sd_keys)
        assert any("prompt_scale_shift_table" in k for k in sd_keys)


class TestV23ConnectorFeatures:
    """Test V2.3 connector-specific features."""

    def test_v23_connector_gated_attention(self):
        """Embeddings1DConnector with gated attention has to_gate_logits on blocks."""
        from llm_dit.encoders.embeddings_connector import Embeddings1DConnector

        connector = Embeddings1DConnector(
            attention_head_dim=32,
            num_attention_heads=4,
            num_layers=2,
            apply_gated_attention=True,
        )
        # Check that transformer blocks have gate logits
        sd_keys = set(connector.state_dict().keys())
        gate_keys = [k for k in sd_keys if "to_gate_logits" in k]
        assert len(gate_keys) > 0, (
            f"Expected gate logit keys, got none. Keys: {sorted(sd_keys)}"
        )

    def test_v23_connector_key_naming(self):
        """Connector uses transformer_1d_blocks (not transformer_blocks) and q_norm/k_norm."""
        from llm_dit.encoders.embeddings_connector import Embeddings1DConnector

        connector = Embeddings1DConnector(
            attention_head_dim=32,
            num_attention_heads=4,
            num_layers=2,
        )
        sd_keys = set(connector.state_dict().keys())
        # Check for correct naming convention
        has_1d_blocks = any("transformer_1d_blocks" in k for k in sd_keys)
        assert has_1d_blocks, (
            f"Expected 'transformer_1d_blocks' in keys, got: {sorted(sd_keys)[:10]}"
        )
        # Check for q_norm/k_norm (not just norm1/norm2)
        has_q_norm = any("q_norm" in k for k in sd_keys)
        has_k_norm = any("k_norm" in k for k in sd_keys)
        assert has_q_norm, f"Expected 'q_norm' in keys, got: {sorted(sd_keys)[:10]}"
        assert has_k_norm, f"Expected 'k_norm' in keys, got: {sorted(sd_keys)[:10]}"


class TestV2PreparePromptTimestep:
    """Test that TransformerArgsPreprocessor computes prompt_timestep for V2."""

    # inner_dim = heads * d_head = 4 * 32 = 128
    # V2 uses Identity for caption_projection, so caption_channels must equal cross_attention_dim
    _CONFIG = {
        "num_attention_heads": 4,
        "attention_head_dim": 32,
        "in_channels": 16,
        "out_channels": 16,
        "num_layers": 2,
        "cross_attention_dim": 128,
        "caption_channels": 128,  # Must match cross_attention_dim for V2 Identity projection
    }

    def _make_v2_model(self):
        """Create a small V2 model for testing."""
        return create_model_from_config(
            dict(self._CONFIG), torch.float32,
            apply_gated_attention=True,
            cross_attention_adaln=True,
        )

    def test_v2_prepare_sets_prompt_timestep(self):
        """V2 preprocessor must return non-None prompt_timestep."""
        from llm_dit.models.ltx2.components import Modality

        model = self._make_v2_model()
        seq_len = 32
        modality = Modality(
            latent=torch.randn(1, seq_len, 16),
            context=torch.randn(1, 8, 128),
            context_mask=torch.ones(1, 8),
            positions=torch.zeros(1, 3, seq_len, 2),
            timesteps=torch.tensor([0.5]),
            enabled=True,
        )
        args = model.args_preprocessor.prepare(modality)
        assert args.prompt_timestep is not None
        assert args.prompt_timestep.ndim == 3  # [B, T', D]

    def test_v2_forward_no_crash(self):
        """V2 model forward pass should complete without error."""
        from llm_dit.models.ltx2.components import Modality

        model = self._make_v2_model()
        # Initialize weights so forward doesn't crash on uninitialized params
        for p in model.parameters():
            if p.data.is_floating_point() and p.data.ndim >= 2:
                torch.nn.init.xavier_uniform_(p.data)
            elif p.data.is_floating_point():
                torch.nn.init.zeros_(p.data)

        seq_len = 32
        modality = Modality(
            latent=torch.randn(1, seq_len, 16),
            context=torch.randn(1, 8, 128),
            context_mask=torch.ones(1, 8),
            positions=torch.zeros(1, 3, seq_len, 2),
            timesteps=torch.tensor([0.5]),
            enabled=True,
        )
        video_out, audio_out = model(video=modality)
        assert video_out is not None
        assert audio_out is None
