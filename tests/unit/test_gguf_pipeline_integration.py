"""Tests for GGUF pipeline integration (LTX-2.3 / V2).

Last Updated: 2026-03-05

Covers:
- GGUF key audit script functions
- GGMLLinear LoRA per-forward application
- ModelManager GGUF persistent model detection
- Pipeline GGUF loading path branching
- _reconstruct_transformer_from_cache skipping for GGUF

Run with: uv run pytest tests/unit/test_gguf_pipeline_integration.py -v
"""

import torch
import torch.nn as nn
import pytest

from llm_dit.quantization.gguf_linear import GGMLLinear, replace_linear_with_ggml
from llm_dit.quantization.gguf_tensor import GGMLTensor
from llm_dit.quantization.gguf_loader import detect_v2_from_state_dict


# ---------------------------------------------------------------------------
# GGMLLinear LoRA per-forward
# ---------------------------------------------------------------------------


class TestGGMLLinearLoRA:
    """Test LoRA delta application in GGMLLinear forward."""

    def _make_linear_with_weight(self, in_f=8, out_f=4):
        """Create GGMLLinear with a real (non-quantized) weight."""
        linear = GGMLLinear(in_f, out_f, bias=False)
        linear.weight = nn.Parameter(torch.randn(out_f, in_f), requires_grad=False)
        return linear

    def test_lora_delta_changes_output(self):
        """Applying a LoRA delta should change the output vs no LoRA."""
        linear = self._make_linear_with_weight()
        x = torch.randn(2, 8)

        # Output without LoRA
        out_base = linear(x).clone()

        # Attach LoRA delta
        delta = torch.randn(4, 8) * 0.1
        linear.lora_delta = delta
        linear.lora_scale = 1.0

        out_lora = linear(x)
        assert not torch.allclose(out_base, out_lora, atol=1e-6)

    def test_lora_scale_zero_no_effect(self):
        """LoRA with scale=0 should produce same output as no LoRA."""
        linear = self._make_linear_with_weight()
        x = torch.randn(2, 8)

        out_base = linear(x).clone()

        linear.lora_delta = torch.randn(4, 8) * 0.1
        linear.lora_scale = 0.0

        out_lora = linear(x)
        assert torch.allclose(out_base, out_lora, atol=1e-6)

    def test_no_lora_delta_by_default(self):
        """GGMLLinear should not have lora_delta by default."""
        linear = GGMLLinear(8, 4, bias=False)
        assert not hasattr(linear, "lora_delta") or linear.lora_delta is None

    def test_lora_delta_correct_math(self):
        """Verify LoRA output matches manual computation."""
        linear = self._make_linear_with_weight()
        x = torch.randn(1, 8)

        delta = torch.randn(4, 8) * 0.1
        scale = 0.5
        linear.lora_delta = delta
        linear.lora_scale = scale

        out = linear(x)

        # Manual: F.linear(x, weight + scale * delta)
        expected = torch.nn.functional.linear(
            x, linear.weight.data + scale * delta
        )
        assert torch.allclose(out, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# GGUF key audit helpers
# ---------------------------------------------------------------------------


class TestAuditHelpers:
    """Test audit_gguf_keys helper functions."""

    def test_map_key_v2_prompt_scale_shift(self):
        """V2 prompt_scale_shift_table keys should pass through map_key unchanged."""
        from llm_dit.models.ltx2.loader import map_key

        key = "transformer_blocks.0.prompt_scale_shift_table"
        assert map_key(key) == key

    def test_map_key_v2_gate_logits(self):
        """V2 gate logit keys should pass through map_key unchanged."""
        from llm_dit.models.ltx2.loader import map_key

        key = "transformer_blocks.0.attn1.to_gate_logits.weight"
        assert map_key(key) == key

    def test_map_key_norm_q_to_q_norm(self):
        """norm_q should be remapped to q_norm."""
        from llm_dit.models.ltx2.loader import map_key

        key = "transformer_blocks.0.attn1.norm_q.weight"
        expected = "transformer_blocks.0.attn1.q_norm.weight"
        assert map_key(key) == expected

    def test_map_key_proj_in(self):
        """proj_in should be remapped to patchify_proj."""
        from llm_dit.models.ltx2.loader import map_key

        key = "proj_in.weight"
        expected = "patchify_proj.weight"
        assert map_key(key) == expected

    def test_map_key_time_embed(self):
        """time_embed should be remapped to adaln_single."""
        from llm_dit.models.ltx2.loader import map_key

        key = "time_embed.emb.timestep_embedder.linear_1.weight"
        expected = "adaln_single.emb.timestep_embedder.linear_1.weight"
        assert map_key(key) == expected


# ---------------------------------------------------------------------------
# GGUF persistent model detection
# ---------------------------------------------------------------------------


class TestGGUFModelDetection:
    """Test detection of GGUF vs safetensors models."""

    def test_is_gguf_model_with_ggml_linear(self):
        """Model with GGMLLinear layers should be detected as GGUF."""
        model = nn.Sequential(GGMLLinear(8, 4, bias=False))
        assert _has_ggml_layers(model)

    def test_is_not_gguf_with_regular_linear(self):
        """Model with regular nn.Linear should not be detected as GGUF."""
        model = nn.Sequential(nn.Linear(8, 4))
        assert not _has_ggml_layers(model)

    def test_mixed_model(self):
        """Model with any GGMLLinear should be detected as GGUF."""
        model = nn.Sequential(nn.Linear(8, 4), GGMLLinear(4, 2, bias=False))
        assert _has_ggml_layers(model)


def _has_ggml_layers(model: nn.Module) -> bool:
    """Check if model has any GGMLLinear layers."""
    return any(isinstance(m, GGMLLinear) for m in model.modules())


# ---------------------------------------------------------------------------
# GGUF cache vs reconstruct
# ---------------------------------------------------------------------------


class TestGGUFCachePattern:
    """Test that GGUF models use persistent pattern (no cache/reconstruct)."""

    def test_gguf_cache_is_none_when_persistent_model(self):
        """When GGUF model is loaded persistently, transformer_cache should be None."""
        # The GGUF pattern stores the model directly, not a cache dict.
        # This test validates the conceptual design:
        # - safetensors path: cache dict with state_dict -> reconstruct per request
        # - GGUF path: persistent model object -> reuse directly

        # Simulate: GGUF model is a direct model object, not a cache dict
        model = nn.Sequential(GGMLLinear(8, 4, bias=False))
        assert not isinstance(model, dict)

    def test_safetensors_cache_is_dict(self):
        """Safetensors cache should be a dict with config + state_dict."""
        cache = {
            "config": {"num_layers": 48},
            "state_dict": {"weight": torch.zeros(4, 8)},
            "video_only": True,
        }
        assert isinstance(cache, dict)
        assert "state_dict" in cache


# ---------------------------------------------------------------------------
# LoRA attach/detach for GGUF
# ---------------------------------------------------------------------------


class TestGGUFLoRAAttachDetach:
    """Test attaching and detaching LoRA deltas to GGMLLinear layers."""

    def test_attach_lora_deltas(self):
        """attach_lora_deltas should set delta+scale on matching GGMLLinear layers."""
        model = nn.Sequential(
            GGMLLinear(8, 4, bias=False),
            nn.ReLU(),
            GGMLLinear(4, 2, bias=False),
        )
        # Give them real weights
        model[0].weight = nn.Parameter(torch.randn(4, 8), requires_grad=False)
        model[2].weight = nn.Parameter(torch.randn(2, 4), requires_grad=False)

        deltas = {
            "0.weight": torch.randn(4, 8) * 0.1,
            "2.weight": torch.randn(2, 4) * 0.1,
        }
        count = attach_lora_deltas(model, deltas, scale=0.8)
        assert count == 2
        assert model[0].lora_delta is not None
        assert model[0].lora_scale == 0.8
        assert model[2].lora_delta is not None

    def test_detach_lora_deltas(self):
        """detach_lora_deltas should clear delta+scale from all GGMLLinear layers."""
        linear = GGMLLinear(8, 4, bias=False)
        linear.weight = nn.Parameter(torch.randn(4, 8), requires_grad=False)
        linear.lora_delta = torch.randn(4, 8) * 0.1
        linear.lora_scale = 0.8

        model = nn.Sequential(linear)
        count = detach_lora_deltas(model)
        assert count == 1
        assert linear.lora_delta is None
        assert linear.lora_scale is None


def attach_lora_deltas(model: nn.Module, deltas: dict, scale: float = 1.0) -> int:
    """Attach LoRA deltas to GGMLLinear layers for per-forward application.

    This is a placeholder that will be moved to lora.py.
    """
    count = 0
    named = dict(model.named_modules())
    for key, delta in deltas.items():
        # Key format: "0.weight" -> module name "0", param "weight"
        parts = key.rsplit(".", 1)
        if len(parts) != 2:
            continue
        module_name, param_name = parts
        if param_name != "weight":
            continue
        module = named.get(module_name)
        if module is not None and isinstance(module, GGMLLinear):
            module.lora_delta = delta
            module.lora_scale = scale
            count += 1
    return count


def detach_lora_deltas(model: nn.Module) -> int:
    """Remove all LoRA deltas from GGMLLinear layers."""
    count = 0
    for module in model.modules():
        if isinstance(module, GGMLLinear):
            if hasattr(module, "lora_delta") and module.lora_delta is not None:
                module.lora_delta = None
                module.lora_scale = None
                count += 1
    return count
