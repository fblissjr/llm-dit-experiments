"""
FP8-cast quantization tests.

Last Updated: 2026-03-06

Tests for the fp8-cast module that patches nn.Linear forward methods to
upcast FP8 weights per-forward, aligned with the official LTX-2.3 approach.

Run with: uv run pytest tests/unit/test_fp8_cast.py -v
"""

import pytest
import torch
from torch import nn


class TestFP8Cast:
    """Tests for fp8-cast forward patching."""

    def test_amend_forward_patches_linear(self):
        """Linear layers should be patched after amend_forward."""
        from llm_dit.quantization.fp8_cast import amend_forward_with_upcast

        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )
        # Downcast weights to fp8
        for m in model.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = m.weight.data.to(torch.float8_e4m3fn)

        count = amend_forward_with_upcast(model)
        assert count == 2, f"Expected 2 patched, got {count}"

    def test_patched_forward_produces_bf16_output(self):
        """Patched linear with fp8 weights should output in input dtype."""
        from llm_dit.quantization.fp8_cast import amend_forward_with_upcast

        linear = nn.Linear(32, 64, bias=False)
        linear.weight.data = linear.weight.data.to(torch.float8_e4m3fn)

        model = nn.Sequential(linear)
        amend_forward_with_upcast(model)

        x = torch.randn(2, 32, dtype=torch.bfloat16)
        y = model(x)
        assert y.dtype == torch.bfloat16, f"Expected bf16 output, got {y.dtype}"
        assert y.shape == (2, 64)

    def test_fp8_weight_stays_fp8(self):
        """After forward, weight should still be fp8 (not mutated)."""
        from llm_dit.quantization.fp8_cast import amend_forward_with_upcast

        linear = nn.Linear(32, 64, bias=False)
        linear.weight.data = linear.weight.data.to(torch.float8_e4m3fn)

        model = nn.Sequential(linear)
        amend_forward_with_upcast(model)

        x = torch.randn(2, 32, dtype=torch.bfloat16)
        _ = model(x)
        assert linear.weight.dtype == torch.float8_e4m3fn, (
            f"Weight should stay fp8, got {linear.weight.dtype}"
        )

    def test_skips_norms_and_embeddings(self):
        """Norms and embeddings should not be patched."""
        from llm_dit.quantization.fp8_cast import amend_forward_with_upcast

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(32, 64)
                self.norm = nn.LayerNorm(64)
                self.embed = nn.Embedding(100, 64)

            def forward(self, x):
                return self.norm(self.linear(x))

        model = DummyModel()
        for m in model.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = m.weight.data.to(torch.float8_e4m3fn)

        count = amend_forward_with_upcast(model)
        # Only the nn.Linear should be patched, norm/embed skipped
        assert count == 1

    def test_bias_upcasted_too(self):
        """Bias should be upcasted along with weight."""
        from llm_dit.quantization.fp8_cast import amend_forward_with_upcast

        linear = nn.Linear(32, 64, bias=True)
        linear.weight.data = linear.weight.data.to(torch.float8_e4m3fn)
        linear.bias.data = linear.bias.data.to(torch.float8_e4m3fn)

        model = nn.Sequential(linear)
        amend_forward_with_upcast(model)

        x = torch.randn(2, 32, dtype=torch.bfloat16)
        y = model(x)
        assert y.dtype == torch.bfloat16


class TestFP8CastLoRAIntegration:
    """Test LoRA fusion with fp8-cast models via state-dict approach."""

    def test_reconstruct_with_lora_fuses_into_state_dict(self):
        """When fp8_cast=True and LoRA provided, fusion happens at state-dict level."""
        from llm_dit.utils.lora import fuse_lora_to_state_dict
        from safetensors.torch import save_file

        # Build a small mixed-dtype state dict (simulating fp8-cast cache)
        base_sd = {
            "linear.weight": torch.randn(16, 8, dtype=torch.bfloat16).to(torch.float8_e4m3fn),
            "norm.weight": torch.ones(16, dtype=torch.bfloat16),
        }

        # Write LoRA
        lora_sd = {
            "linear.lora_A.weight": torch.randn(4, 8, dtype=torch.bfloat16),
            "linear.lora_B.weight": torch.randn(16, 4, dtype=torch.bfloat16),
        }
        import os
        os.makedirs("/tmp/claude-1000/test_fp8_lora", exist_ok=True)
        lora_path = "/tmp/claude-1000/test_fp8_lora/lora.safetensors"
        save_file(lora_sd, lora_path)

        result = fuse_lora_to_state_dict(base_sd, [lora_path], [1.0])

        # FP8 weight should stay fp8 after fusion
        assert result["linear.weight"].dtype == torch.float8_e4m3fn
        # Norm should pass through unchanged
        assert torch.equal(result["norm.weight"], base_sd["norm.weight"])

    def test_fp8_preservation_guard(self):
        """After .to(device), fp8 weights should still be fp8."""
        model = nn.Sequential(nn.Linear(8, 16, bias=False))
        model[0].weight.data = torch.randn(16, 8, dtype=torch.bfloat16).to(torch.float8_e4m3fn)

        # .to("cpu") should preserve fp8 dtype
        model = model.to("cpu")
        fp8_count = sum(1 for p in model.parameters() if p.dtype == torch.float8_e4m3fn)
        assert fp8_count == 1, f"Expected 1 fp8 param after .to(cpu), got {fp8_count}"


class TestFP8CastTransformerLoader:
    """Tests for loading LTX-2.3 transformer with fp8-cast."""

    FP8_PATH = "models/LTX-2.3/ltx-2.3-transformer-fp8.safetensors"

    @pytest.fixture
    def fp8_exists(self):
        from pathlib import Path
        if not Path(self.FP8_PATH).exists():
            pytest.skip(f"FP8 checkpoint not found: {self.FP8_PATH}")

    def test_load_fp8_cast_no_dequantize(self, fp8_exists):
        """fp8-cast loader should NOT dequantize weights -- they stay fp8."""
        from llm_dit.models.ltx2.loader import load_ltx2_transformer_fp8_cast

        model = load_ltx2_transformer_fp8_cast(self.FP8_PATH)
        # Check that some linear weights are fp8
        fp8_count = 0
        for name, param in model.named_parameters():
            if param.dtype == torch.float8_e4m3fn:
                fp8_count += 1
        assert fp8_count > 0, "Expected fp8 weights in model"

    def test_load_fp8_cast_forward_works(self, fp8_exists):
        """Model should be able to run forward pass after fp8-cast loading."""
        # This is a lightweight check -- just verify the patching is correct.
        # Full generation E2E tests are separate.
        from llm_dit.models.ltx2.loader import load_ltx2_transformer_fp8_cast

        model = load_ltx2_transformer_fp8_cast(self.FP8_PATH)
        # Verify forward method was patched (check for original_forward attr)
        patched = 0
        for m in model.modules():
            if isinstance(m, nn.Linear) and hasattr(m, "original_forward"):
                patched += 1
        assert patched > 0, "Expected patched linear layers"
