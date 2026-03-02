"""
Unit tests for LoKR (LyCORIS Kronecker product) support in lora.py.

Last Updated: 2026-03-02

Run with: uv run pytest tests/unit/test_lora_lokr.py -v
"""

import torch
import torch.nn as nn

from llm_dit.utils.lora import LoRALoader


# ============================================================================
# Format Detection Tests
# ============================================================================


class TestDetectFormat:
    """Test _detect_format dispatches correctly between LoRA and LoKR."""

    def test_detects_lokr_format(self):
        """LoKR state dicts contain .lokr_w1 keys."""
        state_dict = {
            "diffusion_model.blocks.0.attn.to_q.lokr_w1": torch.randn(8, 8),
            "diffusion_model.blocks.0.attn.to_q.lokr_w2": torch.randn(480, 480),
            "diffusion_model.blocks.0.attn.to_q.alpha": torch.tensor(8.0),
        }
        loader = LoRALoader()
        assert loader._detect_format(state_dict) == "lokr"

    def test_detects_standard_lora_kohya(self):
        """Standard Kohya LoRA uses lora_up/lora_down."""
        state_dict = {
            "blocks.0.attn.to_q.lora_up.weight": torch.randn(32, 8),
            "blocks.0.attn.to_q.lora_down.weight": torch.randn(8, 32),
        }
        loader = LoRALoader()
        assert loader._detect_format(state_dict) == "lora"

    def test_detects_standard_lora_peft(self):
        """Standard PEFT LoRA uses lora_A/lora_B."""
        state_dict = {
            "blocks.0.attn.to_q.lora_A.weight": torch.randn(8, 32),
            "blocks.0.attn.to_q.lora_B.weight": torch.randn(32, 8),
        }
        loader = LoRALoader()
        assert loader._detect_format(state_dict) == "lora"

    def test_empty_state_dict_returns_lora(self):
        """Empty state dict defaults to lora format."""
        loader = LoRALoader()
        assert loader._detect_format({}) == "lora"


# ============================================================================
# LoKR Name Dict Tests
# ============================================================================


class TestGetLokrNameDict:
    """Test _get_lokr_name_dict key parsing."""

    def test_strips_diffusion_model_prefix(self):
        """diffusion_model. prefix should be removed from target names."""
        state_dict = {
            "diffusion_model.blocks.0.attn.to_q.lokr_w1": torch.randn(8, 8),
            "diffusion_model.blocks.0.attn.to_q.lokr_w2": torch.randn(480, 480),
        }
        loader = LoRALoader()
        name_dict = loader._get_lokr_name_dict(state_dict)

        assert "blocks.0.attn.to_q" in name_dict
        assert "diffusion_model.blocks.0.attn.to_q" not in name_dict

    def test_strips_transformer_prefix(self):
        """transformer. prefix should be removed from target names."""
        state_dict = {
            "transformer.blocks.0.attn.to_q.lokr_w1": torch.randn(8, 8),
            "transformer.blocks.0.attn.to_q.lokr_w2": torch.randn(480, 480),
        }
        loader = LoRALoader()
        name_dict = loader._get_lokr_name_dict(state_dict)

        assert "blocks.0.attn.to_q" in name_dict

    def test_strips_both_prefixes(self):
        """diffusion_model.transformer. double prefix should be stripped."""
        state_dict = {
            "diffusion_model.transformer.blocks.0.attn.to_q.lokr_w1": torch.randn(8, 8),
            "diffusion_model.transformer.blocks.0.attn.to_q.lokr_w2": torch.randn(480, 480),
        }
        loader = LoRALoader()
        name_dict = loader._get_lokr_name_dict(state_dict)

        assert "blocks.0.attn.to_q" in name_dict

    def test_no_prefix(self):
        """Keys without prefix should pass through."""
        state_dict = {
            "blocks.0.attn.to_q.lokr_w1": torch.randn(8, 8),
            "blocks.0.attn.to_q.lokr_w2": torch.randn(480, 480),
        }
        loader = LoRALoader()
        name_dict = loader._get_lokr_name_dict(state_dict)

        assert "blocks.0.attn.to_q" in name_dict

    def test_returns_w1_w2_key_pair(self):
        """Each entry maps target_name -> (w1_key, w2_key)."""
        state_dict = {
            "diffusion_model.blocks.0.attn.to_q.lokr_w1": torch.randn(8, 8),
            "diffusion_model.blocks.0.attn.to_q.lokr_w2": torch.randn(480, 480),
        }
        loader = LoRALoader()
        name_dict = loader._get_lokr_name_dict(state_dict)

        w1_key, w2_key = name_dict["blocks.0.attn.to_q"]
        assert w1_key == "diffusion_model.blocks.0.attn.to_q.lokr_w1"
        assert w2_key == "diffusion_model.blocks.0.attn.to_q.lokr_w2"

    def test_multiple_layers(self):
        """Should parse multiple LoKR layers."""
        state_dict = {
            "diffusion_model.blocks.0.attn.to_q.lokr_w1": torch.randn(8, 8),
            "diffusion_model.blocks.0.attn.to_q.lokr_w2": torch.randn(480, 480),
            "diffusion_model.blocks.0.attn.to_k.lokr_w1": torch.randn(8, 8),
            "diffusion_model.blocks.0.attn.to_k.lokr_w2": torch.randn(480, 480),
            "diffusion_model.blocks.0.attn.to_q.alpha": torch.tensor(8.0),
            "diffusion_model.blocks.0.attn.to_k.alpha": torch.tensor(8.0),
        }
        loader = LoRALoader()
        name_dict = loader._get_lokr_name_dict(state_dict)

        assert len(name_dict) == 2
        assert "blocks.0.attn.to_q" in name_dict
        assert "blocks.0.attn.to_k" in name_dict

    def test_ignores_alpha_and_w2_keys(self):
        """Only .lokr_w1 keys should create entries (w2 derived from w1)."""
        state_dict = {
            "diffusion_model.blocks.0.attn.to_q.lokr_w1": torch.randn(8, 8),
            "diffusion_model.blocks.0.attn.to_q.lokr_w2": torch.randn(480, 480),
            "diffusion_model.blocks.0.attn.to_q.alpha": torch.tensor(8.0),
        }
        loader = LoRALoader()
        name_dict = loader._get_lokr_name_dict(state_dict)

        # Only one entry, not three
        assert len(name_dict) == 1


# ============================================================================
# LoKR Fusion Tests
# ============================================================================


class TestFuseLokrToBaseModel:
    """Test LoKR weight fusion into base model."""

    def _make_model(self):
        """Create a small model for testing.

        blocks.0.attn.to_q is a Linear(16, 16) so kron(4x4, 4x4) = 16x16.
        """
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([
                    nn.ModuleDict({
                        "attn": nn.ModuleDict({
                            "to_q": nn.Linear(16, 16, bias=False),
                            "to_k": nn.Linear(16, 16, bias=False),
                        }),
                    }),
                ])

        return TinyModel()

    def test_lokr_fusion_changes_weights(self):
        """LoKR fusion should modify the target weights."""
        model = self._make_model()
        original_weight = model.blocks[0]["attn"]["to_q"].weight.clone()

        state_dict = {
            "diffusion_model.blocks.0.attn.to_q.lokr_w1": torch.randn(4, 4),
            "diffusion_model.blocks.0.attn.to_q.lokr_w2": torch.randn(4, 4),
        }

        loader = LoRALoader(device="cpu", dtype=torch.float32)
        updated = loader._fuse_lokr_to_base_model(model, state_dict, alpha=1.0)

        assert updated == 1
        assert not torch.allclose(
            model.blocks[0]["attn"]["to_q"].weight, original_weight
        )

    def test_lokr_fusion_correct_kronecker_product(self):
        """Verify the fused weight equals base + scale * kron(w1, w2)."""
        model = self._make_model()
        model.blocks[0]["attn"]["to_q"].weight.data.zero_()

        w1 = torch.randn(4, 4)
        w2 = torch.randn(4, 4)

        state_dict = {
            "diffusion_model.blocks.0.attn.to_q.lokr_w1": w1,
            "diffusion_model.blocks.0.attn.to_q.lokr_w2": w2,
        }

        loader = LoRALoader(device="cpu", dtype=torch.float32)
        loader._fuse_lokr_to_base_model(model, state_dict, alpha=1.0)

        expected = torch.kron(w1, w2)
        actual = model.blocks[0]["attn"]["to_q"].weight.data
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_lokr_user_scale_applied(self):
        """User scale (alpha param) should multiply the delta."""
        model = self._make_model()
        model.blocks[0]["attn"]["to_q"].weight.data.zero_()

        w1 = torch.randn(4, 4)
        w2 = torch.randn(4, 4)

        state_dict = {
            "diffusion_model.blocks.0.attn.to_q.lokr_w1": w1,
            "diffusion_model.blocks.0.attn.to_q.lokr_w2": w2,
        }

        loader = LoRALoader(device="cpu", dtype=torch.float32)
        loader._fuse_lokr_to_base_model(model, state_dict, alpha=0.5)

        expected = 0.5 * torch.kron(w1, w2)
        actual = model.blocks[0]["attn"]["to_q"].weight.data
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_lokr_stored_alpha_ignored(self):
        """The .alpha tensor in the state dict should NOT affect fusion scale."""
        model = self._make_model()
        model.blocks[0]["attn"]["to_q"].weight.data.zero_()

        w1 = torch.randn(4, 4)
        w2 = torch.randn(4, 4)

        state_dict = {
            "diffusion_model.blocks.0.attn.to_q.lokr_w1": w1,
            "diffusion_model.blocks.0.attn.to_q.lokr_w2": w2,
            "diffusion_model.blocks.0.attn.to_q.alpha": torch.tensor(999.0),
        }

        loader = LoRALoader(device="cpu", dtype=torch.float32)
        loader._fuse_lokr_to_base_model(model, state_dict, alpha=1.0)

        # Should be kron(w1, w2), NOT 999 * kron(w1, w2)
        expected = torch.kron(w1, w2)
        actual = model.blocks[0]["attn"]["to_q"].weight.data
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_lokr_multiple_layers(self):
        """Should fuse into multiple matching layers."""
        model = self._make_model()

        state_dict = {
            "diffusion_model.blocks.0.attn.to_q.lokr_w1": torch.randn(4, 4),
            "diffusion_model.blocks.0.attn.to_q.lokr_w2": torch.randn(4, 4),
            "diffusion_model.blocks.0.attn.to_k.lokr_w1": torch.randn(4, 4),
            "diffusion_model.blocks.0.attn.to_k.lokr_w2": torch.randn(4, 4),
        }

        loader = LoRALoader(device="cpu", dtype=torch.float32)
        updated = loader._fuse_lokr_to_base_model(model, state_dict, alpha=1.0)

        assert updated == 2

    def test_lokr_unmatched_layers_skipped(self):
        """Layers in LoKR that don't match model modules should be skipped."""
        model = self._make_model()

        state_dict = {
            "diffusion_model.blocks.0.attn.to_q.lokr_w1": torch.randn(4, 4),
            "diffusion_model.blocks.0.attn.to_q.lokr_w2": torch.randn(4, 4),
            # This layer doesn't exist in the model
            "diffusion_model.blocks.99.ffn.linear.lokr_w1": torch.randn(4, 4),
            "diffusion_model.blocks.99.ffn.linear.lokr_w2": torch.randn(4, 4),
        }

        loader = LoRALoader(device="cpu", dtype=torch.float32)
        updated = loader._fuse_lokr_to_base_model(model, state_dict, alpha=1.0)

        assert updated == 1


# ============================================================================
# Dispatch Tests (fuse_lora_to_base_model dispatches by format)
# ============================================================================


class TestFuseDispatch:
    """Test that fuse_lora_to_base_model dispatches to LoKR path."""

    def test_dispatch_lokr(self):
        """LoKR state dict should route through _fuse_lokr_to_base_model."""

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([
                    nn.ModuleDict({
                        "attn": nn.ModuleDict({
                            "to_q": nn.Linear(16, 16, bias=False),
                        }),
                    }),
                ])

        model = TinyModel()
        original_weight = model.blocks[0]["attn"]["to_q"].weight.clone()

        state_dict = {
            "diffusion_model.blocks.0.attn.to_q.lokr_w1": torch.randn(4, 4),
            "diffusion_model.blocks.0.attn.to_q.lokr_w2": torch.randn(4, 4),
        }

        loader = LoRALoader(device="cpu", dtype=torch.float32)
        updated = loader.fuse_lora_to_base_model(model, state_dict, alpha=0.8)

        assert updated == 1
        assert not torch.allclose(
            model.blocks[0]["attn"]["to_q"].weight, original_weight
        )

    def test_dispatch_standard_lora_unchanged(self):
        """Standard LoRA state dict should still work through the old path."""

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(32, 32)

        model = TinyModel()
        original_weight = model.layer.weight.clone()

        state_dict = {
            "layer.lora_B.weight": torch.randn(32, 8),
            "layer.lora_A.weight": torch.randn(8, 32),
        }

        loader = LoRALoader(device="cpu", dtype=torch.float32)
        updated = loader.fuse_lora_to_base_model(model, state_dict, alpha=1.0)

        assert updated == 1
        assert not torch.allclose(model.layer.weight, original_weight)
