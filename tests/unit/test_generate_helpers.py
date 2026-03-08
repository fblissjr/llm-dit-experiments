"""
Tests for generate.py helper functions.

Last Updated: 2026-03-08

Tests for extracted helpers: _normalize_lora_args (in test_lora.py),
_load_transformer_and_lora.

Run with: uv run pytest tests/unit/test_generate_helpers.py -v
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

pytestmark = pytest.mark.unit


class TestLoadTransformerAndLora:
    """Test _load_transformer_and_lora helper."""

    def test_cache_branch_calls_reconstruct(self):
        """Cached transformer should route through _reconstruct_transformer_from_cache."""
        from llm_dit.pipelines.generate import _load_transformer_and_lora

        mock_reconstructed = MagicMock()

        with patch(
            "llm_dit.pipelines.generate._reconstruct_transformer_from_cache",
            return_value=mock_reconstructed,
        ) as mock_reconstruct:
            model = _load_transformer_and_lora(
                cached_transformer={"config": {}, "state_dict": {}},
                model_path=Path("/fake"),
                transformer_file="",
                dtype=torch.bfloat16,
                transformer_device="cuda",
                effective_quantize=False,
                effective_precision="none",
                granularity="per-row",
                lora_paths=["lora.safetensors"],
                lora_scales=[0.8],
            )

        mock_reconstruct.assert_called_once()
        assert model is mock_reconstructed

    def test_cached_model_with_lora_uses_load_lora(self):
        """Cached model with LoRA should use load_lora."""
        from llm_dit.pipelines.generate import _load_transformer_and_lora

        mock_model = MagicMock()
        mock_model._lora_fused_at_sd_level = False

        with patch(
            "llm_dit.pipelines.generate._reconstruct_transformer_from_cache",
            return_value=mock_model,
        ):
            with patch("llm_dit.utils.lora.load_lora", return_value=5) as mock_lora:
                model = _load_transformer_and_lora(
                    cached_transformer={"config": {}, "state_dict": {}},
                    model_path=Path("/fake"),
                    transformer_file="",
                    dtype=torch.bfloat16,
                    transformer_device="cuda",
                    effective_quantize=False,
                    effective_precision="none",
                    granularity="per-row",
                    lora_paths=["a.safetensors", "b.safetensors"],
                    lora_scales=[0.5, 0.3],
                )

        assert mock_lora.call_count == 2

    def test_sd_level_fusion_skips_post_load_lora(self):
        """When LoRA was fused at state-dict level, post-load LoRA is skipped."""
        from llm_dit.pipelines.generate import _load_transformer_and_lora

        mock_model = MagicMock()
        mock_model._lora_fused_at_sd_level = True

        with patch(
            "llm_dit.pipelines.generate._reconstruct_transformer_from_cache",
            return_value=mock_model,
        ):
            with patch("llm_dit.utils.lora.load_lora") as mock_lora:
                _load_transformer_and_lora(
                    cached_transformer={"config": {}, "state_dict": {}},
                    model_path=Path("/fake"),
                    transformer_file="",
                    dtype=torch.bfloat16,
                    transformer_device="cuda",
                    effective_quantize=False,
                    effective_precision="none",
                    granularity="per-row",
                    lora_paths=["lora.safetensors"],
                    lora_scales=[0.8],
                )

        mock_lora.assert_not_called()

    def test_video_only_passed_to_disk_loader(self):
        """video_only parameter should be forwarded to disk loaders."""
        from llm_dit.pipelines.generate import _load_transformer_and_lora

        mock_model = MagicMock()

        with patch(
            "llm_dit.models.ltx2.load_ltx2_transformer",
            return_value=mock_model,
        ) as mock_load:
            mock_model.to.return_value = mock_model
            mock_model._lora_fused_at_sd_level = False
            mock_model.get_num_params.return_value = 22e9
            # Need the path to exist as a directory (not file) so it takes non-fp8 path
            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "is_file", return_value=False):
                    _load_transformer_and_lora(
                        cached_transformer=None,
                        model_path=Path("/fake"),
                        transformer_file="",
                        dtype=torch.bfloat16,
                        transformer_device="cuda",
                        effective_quantize=False,
                        effective_precision="none",
                        granularity="per-row",
                        lora_paths=None,
                        lora_scales=None,
                        video_only=False,
                    )

        # Check video_only was passed through
        _, kwargs = mock_load.call_args
        assert kwargs.get("video_only") is False

    def test_no_lora_no_error(self):
        """No LoRA paths should not cause any issues with cached model."""
        from llm_dit.pipelines.generate import _load_transformer_and_lora

        mock_model = MagicMock()

        with patch(
            "llm_dit.pipelines.generate._reconstruct_transformer_from_cache",
            return_value=mock_model,
        ):
            model = _load_transformer_and_lora(
                cached_transformer={"config": {}, "state_dict": {}},
                model_path=Path("/fake"),
                transformer_file="",
                dtype=torch.bfloat16,
                transformer_device="cuda",
                effective_quantize=False,
                effective_precision="none",
                granularity="per-row",
                lora_paths=None,
                lora_scales=None,
            )

        assert model is mock_model


class TestTwoStageDivisibilityGuard:
    """Test 64-divisibility guard in generate_video_two_stage."""

    @pytest.mark.parametrize(
        "height,width",
        [
            (448, 640),   # 64-divisible: should pass
            (512, 768),   # 64-divisible: should pass
            (256, 384),   # 64-divisible: should pass
        ],
    )
    def test_valid_dimensions_pass_guard(self, height, width):
        """64-divisible dimensions should not raise ValueError."""
        from llm_dit.pipelines.generate import _validate_two_stage_dimensions

        # Should not raise
        _validate_two_stage_dimensions(height, width)

    @pytest.mark.parametrize(
        "height,width",
        [
            (480, 672),   # 672 % 64 == 32
            (544, 736),   # 736 % 64 == 32
            (352, 480),   # 352 % 64 == 32
            (96, 96),     # 96 % 64 == 32
        ],
    )
    def test_non_64_divisible_raises(self, height, width):
        """Non-64-divisible dimensions should raise ValueError."""
        from llm_dit.pipelines.generate import _validate_two_stage_dimensions

        with pytest.raises(ValueError, match="divisible by 64"):
            _validate_two_stage_dimensions(height, width)
