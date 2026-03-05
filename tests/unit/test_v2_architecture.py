"""Tests for LTX-2.3 (V2) architecture features.

Verifies V2-specific additions:
- Gated attention (per-head sigmoid gate)
- Cross-attention AdaLN (9-param scale_shift_table)
- FeatureExtractorV2 (per-token RMSNorm, dual projections)
- V2 detection from state dict keys
- GGUF infrastructure (GGMLTensor, GGMLLinear)
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

    def test_create_v1_model_default(self):
        config = {
            "num_attention_heads": 4,
            "attention_head_dim": 32,
            "in_channels": 16,
            "out_channels": 16,
            "num_layers": 2,
            "cross_attention_dim": 64,
            "caption_channels": 48,
        }
        model = create_model_from_config(config, torch.bfloat16)
        assert model.apply_gated_attention is False
        assert model.cross_attention_adaln is False
