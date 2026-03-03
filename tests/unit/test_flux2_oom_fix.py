"""
Tests for FLUX.2 OOM fix on 24GB GPUs + LTX-2 quant_transformer bridge.

Last Updated: 2026-03-03

Covers:
- Flux2Config.quant_transformer property (bridges [flux2].quantization to quant resolution)
- LTX2Config.quant_transformer property (bridges [ltx2].quantize to quant resolution)
- get_pipeline_quant_config() respects pipeline-specific quantization fields
- Transformer offload sequence during encoding (persistent transformer)
- OOM warning when compile=true + quantization=none

Run with: uv run pytest tests/unit/test_flux2_oom_fix.py -v
"""

import pytest

pytestmark = pytest.mark.unit


class TestFlux2ConfigQuantTransformer:
    """Flux2Config.quant_transformer property bridges [flux2].quantization to get_pipeline_quant_config."""

    def test_quant_transformer_returns_none_when_none(self):
        """quant_transformer returns None when quantization is 'none' (no override)."""
        from llm_dit.config import Flux2Config

        config = Flux2Config(quantization="none")
        assert config.quant_transformer is None

    def test_quant_transformer_resolves_fp8_alias(self):
        """quant_transformer resolves 'fp8' alias to 'fp8-dynamic'."""
        from llm_dit.config import Flux2Config

        config = Flux2Config(quantization="fp8")
        assert config.quant_transformer == "fp8-dynamic"

    def test_quant_transformer_returns_fp8_dynamic(self):
        """quant_transformer forwards fp8-dynamic value."""
        from llm_dit.config import Flux2Config

        config = Flux2Config(quantization="fp8-dynamic")
        assert config.quant_transformer == "fp8-dynamic"

    def test_quant_transformer_returns_int8(self):
        """quant_transformer forwards int8 value."""
        from llm_dit.config import Flux2Config

        config = Flux2Config(quantization="int8")
        assert config.quant_transformer == "int8"


class TestFlux2PipelineQuantResolution:
    """get_pipeline_quant_config('flux2') respects [flux2].quantization via quant_transformer."""

    def test_flux2_quantization_propagates_to_transformer(self):
        """[flux2].quantization = 'fp8' should resolve to fp8-dynamic on transformer."""
        from llm_dit.config import Flux2Config, PipelineQuantConfig, RuntimeConfig

        rc = RuntimeConfig()
        rc.flux2 = Flux2Config(quantization="fp8")
        result = rc.get_pipeline_quant_config("flux2")

        assert isinstance(result, PipelineQuantConfig)
        assert result.transformer.method == "fp8-dynamic"

    def test_flux2_quantization_none_falls_through_to_global(self):
        """[flux2].quantization = 'none' should fall through to global [quantization].transformer."""
        from llm_dit.config import (
            ComponentQuantConfig,
            Flux2Config,
            RuntimeConfig,
        )

        rc = RuntimeConfig()
        rc.flux2 = Flux2Config(quantization="none")
        # Set a global transformer default
        rc.quant.transformer = ComponentQuantConfig(method="int8")
        result = rc.get_pipeline_quant_config("flux2")

        assert result.transformer.method == "int8"

    def test_flux2_quantization_overrides_global(self):
        """[flux2].quantization should override global [quantization].transformer."""
        from llm_dit.config import (
            ComponentQuantConfig,
            Flux2Config,
            RuntimeConfig,
        )

        rc = RuntimeConfig()
        rc.flux2 = Flux2Config(quantization="fp8-dynamic")
        # Set a different global default
        rc.quant.transformer = ComponentQuantConfig(method="int8")
        result = rc.get_pipeline_quant_config("flux2")

        # Pipeline-specific should win
        assert result.transformer.method == "fp8-dynamic"


class TestLTX2ConfigQuantTransformer:
    """LTX2Config.quant_transformer property bridges [ltx2].quantize to get_pipeline_quant_config."""

    def test_quant_transformer_returns_none_when_none(self):
        """quant_transformer returns None when quantize is 'none'."""
        from llm_dit.config import LTX2Config

        config = LTX2Config(quantize="none")
        assert config.quant_transformer is None

    def test_quant_transformer_returns_none_for_empty_string(self):
        """quant_transformer returns None when quantize is empty string."""
        from llm_dit.config import LTX2Config

        config = LTX2Config(quantize="")
        assert config.quant_transformer is None

    def test_quant_transformer_resolves_fp8_alias(self):
        """quant_transformer resolves 'fp8' alias to 'fp8-dynamic'."""
        from llm_dit.config import LTX2Config

        config = LTX2Config(quantize="fp8")
        assert config.quant_transformer == "fp8-dynamic"

    def test_quant_transformer_passes_through_fp8_dynamic(self):
        """quant_transformer forwards fp8-dynamic without alias resolution."""
        from llm_dit.config import LTX2Config

        config = LTX2Config(quantize="fp8-dynamic")
        assert config.quant_transformer == "fp8-dynamic"


class TestLTX2PipelineQuantResolution:
    """get_pipeline_quant_config('ltx2') respects [ltx2].quantize via quant_transformer."""

    def test_ltx2_quantize_propagates_to_transformer(self):
        """[ltx2].quantize = 'fp8' should resolve to fp8-dynamic on transformer."""
        from llm_dit.config import LTX2Config, PipelineQuantConfig, RuntimeConfig

        rc = RuntimeConfig()
        rc.ltx2 = LTX2Config(quantize="fp8")
        result = rc.get_pipeline_quant_config("ltx2")

        assert isinstance(result, PipelineQuantConfig)
        assert result.transformer.method == "fp8-dynamic"

    def test_ltx2_quantize_none_falls_through_to_global(self):
        """[ltx2].quantize = 'none' should fall through to global [quantization].transformer."""
        from llm_dit.config import (
            ComponentQuantConfig,
            LTX2Config,
            RuntimeConfig,
        )

        rc = RuntimeConfig()
        rc.ltx2 = LTX2Config(quantize="none")
        rc.quant.transformer = ComponentQuantConfig(method="int8")
        result = rc.get_pipeline_quant_config("ltx2")

        assert result.transformer.method == "int8"

    def test_ltx2_quantize_overrides_global(self):
        """[ltx2].quantize should override global [quantization].transformer."""
        from llm_dit.config import (
            ComponentQuantConfig,
            LTX2Config,
            RuntimeConfig,
        )

        rc = RuntimeConfig()
        rc.ltx2 = LTX2Config(quantize="fp8")
        rc.quant.transformer = ComponentQuantConfig(method="int8")
        result = rc.get_pipeline_quant_config("ltx2")

        # Pipeline-specific should win
        assert result.transformer.method == "fp8-dynamic"


class TestFlux2OOMWarning:
    """_load_flux2 should warn when compile=true + quantization=none may OOM."""

    def test_oom_warning_string_present_in_load_flux2(self):
        """_load_flux2 should contain the OOM warning for compile + no quantization."""
        import inspect
        from llm_dit.model_manager import ModelManager

        source = inspect.getsource(ModelManager._load_flux2)
        assert "compile=true with quantization='none' may OOM" in source, (
            "_load_flux2 should warn about compile + no quantization OOM risk"
        )

    def test_oom_warning_recommends_fp8(self):
        """The OOM warning should recommend fp8 quantization."""
        import inspect
        from llm_dit.model_manager import ModelManager

        source = inspect.getsource(ModelManager._load_flux2)
        assert "fp8" in source.lower(), (
            "_load_flux2 OOM warning should recommend fp8 quantization"
        )


class TestTransformerOffloadDuringEncoding:
    """Persistent transformer should be offloaded to CPU during text encoding to avoid OOM."""

    def test_persistent_transformer_offloaded_before_encoding(self):
        """When transformer_is_persistent=True, transformer.to('cpu') is called before encoding."""
        import inspect
        from llm_dit.pipelines.flux2_generate import generate_image

        source = inspect.getsource(generate_image)

        # The function should contain transformer offload logic
        assert "transformer_was_offloaded" in source, (
            "generate_image should track transformer offload state via transformer_was_offloaded"
        )
        assert 'transformer.to("cpu")' in source or "transformer.to('cpu')" in source, (
            "generate_image should offload transformer to CPU before encoding"
        )

    def test_offload_skipped_for_compiled_transformer(self):
        """Compiled transformers (with _orig_mod) should NOT be offloaded."""
        from llm_dit.pipelines.flux2_generate import generate_image
        import inspect

        source = inspect.getsource(generate_image)

        # Should check for _orig_mod (compiled indicator)
        assert "_orig_mod" in source, (
            "generate_image should check for _orig_mod to detect compiled transformers"
        )

    def test_transformer_reloaded_after_encoder_offload(self):
        """After encoder offloads, transformer should be moved back to GPU."""
        from llm_dit.pipelines.flux2_generate import generate_image
        import inspect

        source = inspect.getsource(generate_image)

        # Should reload transformer after encoder offload
        assert "transformer_was_offloaded" in source, (
            "generate_image should use transformer_was_offloaded flag to reload"
        )
        assert "if transformer_was_offloaded" in source, (
            "generate_image should conditionally reload transformer based on offload flag"
        )
        # Verify the reload comes AFTER the offload
        offload_pos = source.find("transformer_was_offloaded = True")
        reload_pos = source.find("if transformer_was_offloaded")
        assert offload_pos < reload_pos, (
            "Transformer reload should come after the offload flag is set"
        )
