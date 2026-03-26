"""
FP8-cast quantization tests.

Last Updated: 2026-03-25

Tests for the fp8-cast module that patches nn.Linear forward methods to
upcast FP8 weights per-forward, aligned with the official LTX-2.3 approach.
Includes tests for native FP8 tensor core matmul via torch._scaled_mm.

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


class TestPatchedForwardSurvivesLoadStateDict:
    """Verify that amend_forward_with_upcast closures survive load_state_dict(assign=True).

    This is the key assumption behind removing the redundant amend_forward_with_upcast
    call from _apply_distilled_lora_fp8. If this test fails, the simplify was wrong
    and fp8 models will produce garbage after distilled LoRA application.
    """

    def test_patched_forward_survives_assign_true(self):
        """Forward patch should still work after load_state_dict(assign=True)."""
        from llm_dit.quantization.fp8_cast import amend_forward_with_upcast

        model = nn.Sequential(nn.Linear(32, 64, bias=False))
        model[0].weight.data = model[0].weight.data.to(torch.float8_e4m3fn)

        # Patch the forward
        patched = amend_forward_with_upcast(model)
        assert patched == 1

        # Verify forward works before load_state_dict
        x = torch.randn(2, 32, dtype=torch.bfloat16)
        y_before = model(x)
        assert y_before.dtype == torch.bfloat16

        # Now replace parameters via load_state_dict(assign=True) -- same as
        # what _apply_distilled_lora_fp8 does after fusing LoRA deltas
        new_sd = {"0.weight": torch.randn(64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)}
        model.load_state_dict(new_sd, assign=True)

        # The patched forward should STILL work -- closures access layer.weight
        # at call time, so they pick up the new parameter automatically
        y_after = model(x)
        assert y_after.dtype == torch.bfloat16, (
            f"Patched forward broken after load_state_dict(assign=True): got {y_after.dtype}"
        )
        assert y_after.shape == (2, 64)

        # Weight should still be fp8
        assert model[0].weight.dtype == torch.float8_e4m3fn

    def test_patched_forward_uses_new_weights(self):
        """After load_state_dict(assign=True), forward should use the NEW weights."""
        from llm_dit.quantization.fp8_cast import amend_forward_with_upcast

        model = nn.Sequential(nn.Linear(32, 64, bias=False))
        # Use zeros so output is deterministic
        model[0].weight.data = torch.zeros(64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)

        amend_forward_with_upcast(model)

        x = torch.ones(1, 32, dtype=torch.bfloat16)
        y_zeros = model(x)
        assert torch.allclose(y_zeros, torch.zeros(1, 64, dtype=torch.bfloat16))

        # Replace with ones -- output should change
        new_sd = {"0.weight": torch.ones(64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)}
        model.load_state_dict(new_sd, assign=True)

        y_ones = model(x)
        # Each output neuron = sum of 32 ones = 32.0
        expected = torch.full((1, 64), 32.0, dtype=torch.bfloat16)
        assert torch.allclose(y_ones, expected, atol=0.5), (
            f"Forward not using new weights: got {y_ones[0, :4]}, expected ~32.0"
        )


class TestFP8CastDistilledLoRA:
    """Test distilled LoRA on fp8-cast models (Stage 2 fix)."""

    def test_distilled_lora_on_fp8_uses_state_dict_fusion(self, tmp_path):
        """Distilled LoRA on a model with native fp8 weights should use
        state-dict-level fusion and preserve fp8 dtype."""
        from llm_dit.pipelines.generate import _apply_distilled_lora_fp8
        from llm_dit.utils.lora import fuse_lora_to_state_dict
        from safetensors.torch import save_file

        # Build a model with fp8 weights
        model = nn.Sequential(nn.Linear(32, 64, bias=False))
        model[0].weight.data = model[0].weight.data.to(torch.float8_e4m3fn)

        # Write a LoRA file
        lora_sd = {
            "0.lora_A.weight": torch.randn(4, 32, dtype=torch.bfloat16),
            "0.lora_B.weight": torch.randn(64, 4, dtype=torch.bfloat16),
        }
        lora_path = str(tmp_path / "distilled.safetensors")
        save_file(lora_sd, lora_path)

        # Apply distilled LoRA to fp8 model
        _apply_distilled_lora_fp8(model, lora_path, scale=1.0)

        # Weight should still be fp8 after fusion
        assert model[0].weight.dtype == torch.float8_e4m3fn

    def test_distilled_lora_with_weight_scales(self, tmp_path):
        """Distilled LoRA on scaled fp8 model: dequant, fuse, re-quant with updated scales.

        This verifies the fix for the bug where raw fp8 values (~300) dwarfed
        the LoRA delta (~0.001), making the LoRA negligible.
        """
        from llm_dit.pipelines.generate import _apply_distilled_lora_fp8
        from safetensors.torch import save_file

        # Build a model with properly scaled fp8 weights
        model = nn.Sequential(nn.Linear(32, 64, bias=False))
        real_weight = torch.randn(64, 32, dtype=torch.float32) * 0.02
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        max_abs = real_weight.abs().amax()
        quant_scale = fp8_max / max_abs
        raw_fp8 = (real_weight * quant_scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
        weight_scale = quant_scale.reciprocal()

        model[0].weight = nn.Parameter(raw_fp8, requires_grad=False)
        model[0]._weight_scale = weight_scale

        # Dequantize to get baseline real values
        baseline_real = raw_fp8.to(torch.float32) * weight_scale

        # Write a LoRA file with meaningful magnitude
        lora_sd = {
            "0.lora_A.weight": torch.randn(4, 32, dtype=torch.bfloat16) * 0.1,
            "0.lora_B.weight": torch.randn(64, 4, dtype=torch.bfloat16) * 0.1,
        }
        lora_path = str(tmp_path / "distilled.safetensors")
        save_file(lora_sd, lora_path)

        _apply_distilled_lora_fp8(model, lora_path, scale=1.0)

        # Weight should still be fp8
        assert model[0].weight.dtype == torch.float8_e4m3fn
        # Should have an updated weight_scale
        assert hasattr(model[0], "_weight_scale")

        # Dequantize result to real values
        result_real = model[0].weight.to(torch.float32) * model[0]._weight_scale

        # The LoRA should have made a meaningful change
        delta_magnitude = (result_real - baseline_real).abs().mean()
        assert delta_magnitude > 1e-4, (
            f"LoRA delta too small ({delta_magnitude:.2e}) -- "
            "weight_scale not applied during fusion"
        )


class TestFP8PreservationGuard:
    """Test FP8 preservation guard raises RuntimeError."""

    def test_fp8_guard_raises_on_zero_fp8_count(self):
        """Guard should raise RuntimeError when fp8 count is 0 after reconstruction."""
        from unittest.mock import patch, MagicMock
        from llm_dit.pipelines.generate import _reconstruct_transformer_from_cache

        # Create a fake cache dict that claims fp8_cast=True
        fake_cache = {
            "config": {},
            "state_dict": {},
            "fp8_cast": True,
            "video_only": True,
        }

        # Mock the model creation chain so we can control what happens.
        # The model will have NO fp8 params, triggering the guard.
        mock_model = MagicMock()
        # parameters() returns bf16-only params (no fp8)
        mock_param = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.bfloat16))
        mock_model.parameters.return_value = [mock_param]
        mock_model.to.return_value = mock_model

        with patch("llm_dit.utils.meta_init.meta_init"):
            with patch("llm_dit.models.ltx2.loader.create_model_from_config", return_value=mock_model):
                with patch("llm_dit.quantization.fp8_cast.amend_forward_with_upcast", return_value=0):
                    with pytest.raises(RuntimeError, match="FP8 weights lost"):
                        _reconstruct_transformer_from_cache(
                            fake_cache,
                            dtype=torch.bfloat16,
                            transformer_device="cpu",
                            effective_quantize=False,
                            effective_precision="none",
                            granularity="per-row",
                        )


class TestScaledMM:
    """Tests for native FP8 tensor core matmul via torch._scaled_mm."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_scaled_mm_available(self):
        """torch._scaled_mm should be available."""
        from llm_dit.quantization.fp8_cast import is_scaled_mm_available

        # RTX 4090 (SM89) supports _scaled_mm
        assert is_scaled_mm_available() is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_scaled_mm_forward_produces_correct_output(self):
        """Scaled MM forward should match upcast forward numerically."""
        from llm_dit.quantization.fp8_cast import (
            _replace_fwd_with_scaled_mm,
            _replace_fwd_with_upcast,
            quantize_to_fp8_per_tensor,
        )

        # Create a linear with scaled fp8 weights
        linear_smm = nn.Linear(64, 128, bias=False, device="cuda")
        real_weight = linear_smm.weight.data.clone()

        # Quantize
        fp8_weight, weight_scale = quantize_to_fp8_per_tensor(real_weight)
        linear_smm.weight = nn.Parameter(fp8_weight.to("cuda"), requires_grad=False)
        linear_smm._weight_scale = weight_scale.to("cuda")

        # Create identical upcast linear for comparison
        linear_up = nn.Linear(64, 128, bias=False, device="cuda")
        linear_up.weight = nn.Parameter(fp8_weight.to("cuda").clone(), requires_grad=False)
        linear_up._weight_scale = weight_scale.to("cuda")

        _replace_fwd_with_scaled_mm(linear_smm)
        _replace_fwd_with_upcast(linear_up)

        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        y_smm = linear_smm(x)
        y_up = linear_up(x)

        assert y_smm.dtype == torch.bfloat16
        assert y_smm.shape == (4, 128)
        # Allow fp8 quantization noise (larger tolerance than bf16-bf16)
        assert (y_smm - y_up).abs().max() < 0.5, (
            f"scaled_mm vs upcast too different: max_diff={(y_smm - y_up).abs().max():.4f}"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_scaled_mm_without_weight_scale(self):
        """Scaled MM should work without weight_scale (naive fp8)."""
        from llm_dit.quantization.fp8_cast import _replace_fwd_with_scaled_mm

        linear = nn.Linear(64, 128, bias=False, device="cuda")
        linear.weight.data = linear.weight.data.to(torch.float8_e4m3fn)
        # No _weight_scale attr

        _replace_fwd_with_scaled_mm(linear)

        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        y = linear(x)
        assert y.dtype == torch.bfloat16
        assert y.shape == (4, 128)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_scaled_mm_with_bias(self):
        """Scaled MM should handle bias correctly."""
        from llm_dit.quantization.fp8_cast import _replace_fwd_with_scaled_mm

        linear = nn.Linear(64, 128, bias=True, device="cuda")
        linear.weight.data = linear.weight.data.to(torch.float8_e4m3fn)
        # Keep bias as bf16

        _replace_fwd_with_scaled_mm(linear)

        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        y = linear(x)
        assert y.dtype == torch.bfloat16
        assert y.shape == (4, 128)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_scaled_mm_3d_input(self):
        """Scaled MM should handle 3D inputs (batch, seq, dim)."""
        from llm_dit.quantization.fp8_cast import _replace_fwd_with_scaled_mm

        linear = nn.Linear(64, 128, bias=False, device="cuda")
        linear.weight.data = linear.weight.data.to(torch.float8_e4m3fn)

        _replace_fwd_with_scaled_mm(linear)

        x = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)
        y = linear(x)
        assert y.shape == (2, 16, 128)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_amend_forward_prefers_scaled_mm(self, caplog):
        """amend_forward_with_upcast should use scaled_mm on CUDA when available."""
        import logging
        from llm_dit.quantization.fp8_cast import amend_forward_with_upcast

        model = nn.Sequential(nn.Linear(64, 128, device="cuda"))
        model[0].weight.data = model[0].weight.data.to(torch.float8_e4m3fn)

        with caplog.at_level(logging.INFO, logger="llm_dit.quantization.fp8_cast"):
            count = amend_forward_with_upcast(model)

        assert count == 1
        assert "via scaled_mm" in caplog.text

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_weight_stays_fp8_after_scaled_mm_forward(self):
        """Weight should remain fp8 after forward (not mutated)."""
        from llm_dit.quantization.fp8_cast import _replace_fwd_with_scaled_mm

        linear = nn.Linear(64, 128, bias=False, device="cuda")
        linear.weight.data = linear.weight.data.to(torch.float8_e4m3fn)

        _replace_fwd_with_scaled_mm(linear)

        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        _ = linear(x)
        assert linear.weight.dtype == torch.float8_e4m3fn


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
