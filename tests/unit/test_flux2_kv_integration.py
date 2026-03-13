"""
Tests for FLUX.2 KV-cache model integration into schema, config, and loader.

Verifies that KV model variants appear in the frontend dropdown, have correct
dependent_defaults for steps/guidance, and that the loader can resolve weights
from sibling directories when model_path points to a different variant.
"""

import pytest


class TestKVModelsInSchema:
    """KV model variants must appear in the frontend dropdown."""

    def test_kv_models_in_dropdown_options(self):
        """FLUX2_MODELS list drives the model dropdown -- KV variants must be included."""
        from llm_dit.pipelines.schemas.flux2 import FLUX2_MODELS

        assert "klein-9b-kv" in FLUX2_MODELS
        assert "klein-9b-kv-fp8" in FLUX2_MODELS

    def test_kv_models_in_dependent_defaults_steps(self):
        """num_steps dependent_defaults must have entries for KV models."""
        from llm_dit.pipelines.schemas import PIPELINE_REGISTRY

        # Trigger registration
        import llm_dit.pipelines.schemas.flux2  # noqa: F401

        schema = PIPELINE_REGISTRY["flux2"]
        num_steps_param = next(p for p in schema.params if p.id == "num_steps")
        model_defaults = num_steps_param.dependent_defaults["model_name"]
        assert "klein-9b-kv" in model_defaults
        assert "klein-9b-kv-fp8" in model_defaults
        # KV models are distilled -- 4 steps
        assert model_defaults["klein-9b-kv"] == 4
        assert model_defaults["klein-9b-kv-fp8"] == 4

    def test_kv_models_in_dependent_defaults_guidance(self):
        """guidance dependent_defaults must have entries for KV models."""
        from llm_dit.pipelines.schemas import PIPELINE_REGISTRY

        import llm_dit.pipelines.schemas.flux2  # noqa: F401

        schema = PIPELINE_REGISTRY["flux2"]
        guidance_param = next(p for p in schema.params if p.id == "guidance")
        model_defaults = guidance_param.dependent_defaults["model_name"]
        assert "klein-9b-kv" in model_defaults
        assert "klein-9b-kv-fp8" in model_defaults
        # KV models are distilled -- guidance 1.0
        assert model_defaults["klein-9b-kv"] == 1.0
        assert model_defaults["klein-9b-kv-fp8"] == 1.0


class TestKVModelsSupportedModels:
    """KV models must appear in the status endpoint's supported_models list."""

    def test_kv_in_supported_models_list(self):
        """Schema FLUX2_MODELS must be a subset of FLUX2_MODEL_INFO registry."""
        from llm_dit.pipelines.schemas.flux2 import FLUX2_MODELS
        from llm_dit.models.flux2.constants import FLUX2_MODEL_INFO

        for model in FLUX2_MODELS:
            assert model in FLUX2_MODEL_INFO, (
                f"Schema model '{model}' not in FLUX2_MODEL_INFO registry"
            )


class TestModelPathSiblingResolution:
    """Loader should find weights in sibling directories when model_path
    doesn't directly contain the requested model's weights."""

    def test_finds_weight_in_sibling_directory(self, tmp_path):
        """If model_path points to klein-9b-kv/ but we request klein-9b-kv-fp8,
        loader should check sibling directories."""
        from llm_dit.models.flux2.loader import _get_model_weight_path

        # Create sibling directories mimicking user's layout
        kv_dir = tmp_path / "FLUX.2-klein-9b-kv"
        kv_fp8_dir = tmp_path / "FLUX.2-klein-9b-kv-fp8"
        kv_dir.mkdir()
        kv_fp8_dir.mkdir()

        # Put weight files in each
        (kv_dir / "flux-2-klein-9b-kv.safetensors").write_text("fake")
        (kv_fp8_dir / "flux-2-klein-9b-kv-fp8.safetensors").write_text("fake")

        # model_path points to kv dir, but requesting kv-fp8
        result = _get_model_weight_path("klein-9b-kv-fp8", str(kv_dir))
        assert "flux-2-klein-9b-kv-fp8.safetensors" in result
        assert str(kv_fp8_dir) in result

    def test_finds_weight_in_same_directory(self, tmp_path):
        """Baseline: model_path directly contains the right file."""
        from llm_dit.models.flux2.loader import _get_model_weight_path

        model_dir = tmp_path / "FLUX.2-klein-9b-kv"
        model_dir.mkdir()
        (model_dir / "flux-2-klein-9b-kv.safetensors").write_text("fake")

        result = _get_model_weight_path("klein-9b-kv", str(model_dir))
        assert "flux-2-klein-9b-kv.safetensors" in result

    def test_nonexistent_model_path_raises_descriptive_error(self, tmp_path):
        """Nonexistent model_path should raise about not finding weights."""
        from llm_dit.models.flux2.loader import _get_model_weight_path

        nonexistent = str(tmp_path / "does_not_exist")
        with pytest.raises(ValueError, match="Could not find"):
            _get_model_weight_path("klein-9b-kv", nonexistent)
