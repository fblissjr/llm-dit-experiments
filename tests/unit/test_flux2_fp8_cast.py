"""
Tests for FLUX.2 FP8-cast loading (per-forward upcast instead of dequant-at-load).

Last Updated: 2026-03-13

Tests that FP8 weights stay as FP8 dtype after loading, weight_scale attributes
are attached to nn.Linear modules, and forward passes produce finite output.

Run with: uv run pytest tests/unit/test_flux2_fp8_cast.py -v
"""

import torch
from torch import nn

from llm_dit.models.flux2.transformer import Flux2Transformer
from llm_dit.models.flux2.constants import Klein9BParams
from llm_dit.models.flux2.rope import create_image_ids, create_text_ids
from llm_dit.quantization.fp8_cast import _attach_weight_scales, amend_forward_with_upcast, quantize_to_fp8_per_tensor


# ============================================================================
# Helpers
# ============================================================================


def _build_mini_model() -> Flux2Transformer:
    """Create a minimal FLUX.2 transformer (1 double + 1 single block)."""
    params = Klein9BParams()
    params.depth = 1
    params.depth_single_blocks = 1
    model = Flux2Transformer(params)
    return model


def _build_fp8_state_dict(model: Flux2Transformer) -> tuple[dict, dict]:
    """Convert a model's state dict into a synthetic FP8 state dict with weight scales.

    Returns:
        (fp8_state_dict, weight_scales) where weight_scales maps
        "module.weight" -> scale tensor.
    """
    sd = model.state_dict()
    fp8_sd = {}
    weight_scales = {}

    for key, tensor in sd.items():
        if key.endswith(".weight") and tensor.ndim == 2:
            fp8_w, scale = quantize_to_fp8_per_tensor(tensor)
            fp8_sd[key] = fp8_w
            scale_key = key.replace(".weight", ".weight_scale")
            fp8_sd[scale_key] = scale
            weight_scales[key] = scale
        else:
            fp8_sd[key] = tensor
    return fp8_sd, weight_scales


def _setup_fp8_cast_model():
    """Build a mini model with fp8-cast weights (shared test setup)."""
    model = _build_mini_model()
    fp8_sd, weight_scales = _build_fp8_state_dict(model)

    # Strip scale keys from state dict (loader pops them before load_state_dict)
    for k in [k for k in fp8_sd if k.endswith(".weight_scale")]:
        del fp8_sd[k]

    model.load_state_dict(fp8_sd, strict=False, assign=True)

    # Attach weight scales to modules
    _attach_weight_scales(model, weight_scales)

    amend_forward_with_upcast(model)
    return model, weight_scales


def _model_inputs(dtype=torch.float32):
    """Create minimal test inputs for the mini model."""
    B = 1
    img_len, txt_len = 64, 16
    x = torch.randn(B, img_len, 128, dtype=dtype)
    x_ids = create_image_ids(B, 8, 8)
    timesteps = torch.tensor([0.5])
    ctx = torch.randn(B, txt_len, Klein9BParams().context_in_dim, dtype=dtype)
    ctx_ids = create_text_ids(B, txt_len)
    return x, x_ids, timesteps, ctx, ctx_ids


# ============================================================================
# FP8-cast Loading Tests
# ============================================================================


class TestFp8CastLoader:
    """Tests for FP8-cast loading of FLUX.2 transformer."""

    def test_fp8_weights_stay_fp8_after_load(self):
        """FP8 weights should remain float8_e4m3fn after load_state_dict(assign=True)."""
        model = _build_mini_model()
        fp8_sd, _ = _build_fp8_state_dict(model)

        for k in [k for k in fp8_sd if k.endswith(".weight_scale")]:
            del fp8_sd[k]

        model.load_state_dict(fp8_sd, strict=False, assign=True)

        fp8_count = 0
        for _, param in model.named_parameters():
            if param.dtype == torch.float8_e4m3fn:
                fp8_count += 1

        assert fp8_count > 0, "Expected FP8 weights to remain after assign=True load"

    def test_weight_scales_attached_to_linears(self):
        """After fp8-cast setup, nn.Linear modules should have _weight_scale attribute."""
        model, weight_scales = _setup_fp8_cast_model()

        scale_count = 0
        for _, module in model.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, "_weight_scale"):
                scale_count += 1

        assert scale_count > 0, "Expected weight scales to be attached to nn.Linear modules"
        assert scale_count == len(weight_scales), (
            f"Expected {len(weight_scales)} scales attached, got {scale_count}"
        )

    def test_amend_forward_patches_linears(self):
        """amend_forward_with_upcast should patch nn.Linear forward methods."""
        model = _build_mini_model()
        fp8_sd, weight_scales = _build_fp8_state_dict(model)
        for k in [k for k in fp8_sd if k.endswith(".weight_scale")]:
            del fp8_sd[k]
        model.load_state_dict(fp8_sd, strict=False, assign=True)
        _attach_weight_scales(model, weight_scales)

        count = amend_forward_with_upcast(model)
        assert count > 0, "Expected some nn.Linear layers to be patched"

        # Verify patched forward is a function (not bound method from nn.Linear)
        for _, module in model.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, "original_forward"):
                assert "forward" in module.__dict__, "Patched forward should be instance attr"

    def test_fp8_cast_forward_produces_finite_output(self):
        """Forward pass through fp8-cast model should produce finite output."""
        model, _ = _setup_fp8_cast_model()
        x, x_ids, timesteps, ctx, ctx_ids = _model_inputs()

        with torch.no_grad():
            out = model(x=x, x_ids=x_ids, timesteps=timesteps, ctx=ctx, ctx_ids=ctx_ids, guidance=None)

        assert out.shape == (1, 64, 128)
        assert torch.isfinite(out).all(), "Output should be finite"

    def test_fp8_cast_numerically_close_to_dequant(self):
        """FP8-cast (per-forward upcast) should produce similar results to dequant-at-load."""
        # Path A: dequant at load (old behavior)
        model_dequant = _build_mini_model()
        fp8_sd, weight_scales = _build_fp8_state_dict(model_dequant)

        dequant_sd = {}
        for k, v in fp8_sd.items():
            if k.endswith(".weight_scale"):
                continue
            if v.dtype == torch.float8_e4m3fn:
                scale_key = k.replace(".weight", ".weight_scale")
                if scale_key in fp8_sd:
                    dequant_sd[k] = v.to(torch.float32) * fp8_sd[scale_key].to(torch.float32)
                else:
                    dequant_sd[k] = v.to(torch.float32)
            else:
                dequant_sd[k] = v
        model_dequant.load_state_dict(dequant_sd, strict=True, assign=True)

        # Path B: fp8-cast (new behavior)
        model_fp8cast = _build_mini_model()
        fp8_only_sd = {k: v for k, v in fp8_sd.items() if not k.endswith(".weight_scale")}
        model_fp8cast.load_state_dict(fp8_only_sd, strict=False, assign=True)
        _attach_weight_scales(model_fp8cast, weight_scales)
        amend_forward_with_upcast(model_fp8cast)

        # Same input for both (float32 to match model dtype)
        torch.manual_seed(42)
        x, x_ids, timesteps, ctx, ctx_ids = _model_inputs(dtype=torch.float32)

        with torch.no_grad():
            out_dequant = model_dequant(x=x.clone(), x_ids=x_ids, timesteps=timesteps, ctx=ctx, ctx_ids=ctx_ids, guidance=None)
            out_fp8cast = model_fp8cast(x=x.clone(), x_ids=x_ids, timesteps=timesteps, ctx=ctx, ctx_ids=ctx_ids, guidance=None)

        max_diff = (out_dequant - out_fp8cast).abs().max().item()
        assert max_diff < 0.1, f"FP8-cast and dequant outputs diverged: max_diff={max_diff}"


class TestFp8CastValidation:
    """Tests for fp8-cast weight validation."""

    def test_validate_handles_fp8_params(self):
        """Validation should handle models with fp8 params without crashing."""
        from llm_dit.models.flux2.loader import _validate_transformer_weights

        model, _ = _setup_fp8_cast_model()

        # Should not crash on fp8 params (isnan/isinf not supported for fp8)
        _validate_transformer_weights(model, is_fp8=False)
