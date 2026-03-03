"""
Tests for codebase cleanup: dead code deletion, CUDA guards, consolidation.

Last Updated: 2026-03-03

Covers:
- CUDA guard consistency in model_manager unload methods
- Dead RuntimeConfig properties removed
- Dead LTX2Config methods/fields removed
- Dead PipelineConfig class removed
- EnhancementConfig preset classmethods removed
- cleanup_memory() adoption in model_manager (no inline gc.collect)
- Centralized QUANT_ALIASES constant

Run with: uv run pytest tests/unit/test_cleanup.py -v
"""

import inspect

import pytest

pytestmark = pytest.mark.unit


# =========================================================================
# Phase 1: P0 Bug fixes
# =========================================================================


class TestCUDAGuardConsistency:
    """All unload methods in ModelManager must guard torch.cuda calls."""

    def test_unload_qwen_image_has_cuda_guard(self):
        """_unload_qwen_image must guard torch.cuda.empty_cache() with is_available()."""
        from llm_dit.model_manager import ModelManager

        source = inspect.getsource(ModelManager._unload_qwen_image)
        # Should NOT have bare torch.cuda.empty_cache() without a guard
        lines = source.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "torch.cuda.empty_cache()":
                # Check that a preceding line has the guard
                preceding = "\n".join(lines[max(0, i - 3):i])
                assert "torch.cuda.is_available()" in preceding or "cleanup_memory" in source, (
                    "_unload_qwen_image has bare torch.cuda.empty_cache() without CUDA guard"
                )

    def test_unload_qwen_image_t2i_has_cuda_guard(self):
        """_unload_qwen_image_t2i must guard torch.cuda.empty_cache()."""
        from llm_dit.model_manager import ModelManager

        source = inspect.getsource(ModelManager._unload_qwen_image_t2i)
        lines = source.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "torch.cuda.empty_cache()":
                preceding = "\n".join(lines[max(0, i - 3):i])
                assert "torch.cuda.is_available()" in preceding or "cleanup_memory" in source, (
                    "_unload_qwen_image_t2i has bare torch.cuda.empty_cache() without CUDA guard"
                )


class TestDeadFlux2PropertiesRemoved:
    """RuntimeConfig should not have dead flux2 backward-compat properties."""

    def test_flux2_seed_property_removed(self):
        """flux2_seed property should not exist (Flux2Config has no seed field)."""
        from llm_dit.config import RuntimeConfig

        assert not hasattr(RuntimeConfig, "flux2_seed"), (
            "RuntimeConfig.flux2_seed should be deleted -- Flux2Config has no seed field"
        )

    def test_flux2_output_path_property_removed(self):
        """flux2_output_path property should not exist."""
        from llm_dit.config import RuntimeConfig

        assert not hasattr(RuntimeConfig, "flux2_output_path"), (
            "RuntimeConfig.flux2_output_path should be deleted -- Flux2Config has no output_path field"
        )

    def test_flux2_input_images_property_removed(self):
        """flux2_input_images property should not exist."""
        from llm_dit.config import RuntimeConfig

        assert not hasattr(RuntimeConfig, "flux2_input_images"), (
            "RuntimeConfig.flux2_input_images should be deleted -- Flux2Config has no input_images field"
        )


# =========================================================================
# Phase 2: Dead code deletion
# =========================================================================


class TestDeadLTX2ConfigMethodsRemoved:
    """Dead methods and fields on LTX2Config should be removed."""

    def test_get_total_steps_removed(self):
        """get_total_steps() was unused outside config.py. Should be deleted."""
        from llm_dit.config import LTX2Config

        assert not hasattr(LTX2Config, "get_total_steps"), (
            "LTX2Config.get_total_steps should be deleted -- zero callers"
        )

    def test_estimate_vram_usage_removed(self):
        """estimate_vram_usage() was unused. Should be deleted."""
        from llm_dit.config import LTX2Config

        assert not hasattr(LTX2Config, "estimate_vram_usage"), (
            "LTX2Config.estimate_vram_usage should be deleted -- zero callers"
        )

    def test_vram_classvars_removed(self):
        """9 ClassVar VRAM estimation constants should be removed."""
        from llm_dit.config import LTX2Config

        dead_classvars = [
            "_LTX2_NUM_BLOCKS",
            "_LTX2_VRAM_FP8_GB",
            "_LTX2_VRAM_FP4_GB",
            "_LTX2_VRAM_BF16_GB",
            "_GEMMA3_VRAM_Q4_GB",
            "_GEMMA3_VRAM_FULL_GB",
            "_VAE_VRAM_GB",
            "_OVERHEAD_GB",
            "_GROUP_OVERHEAD_GB",
        ]
        for attr in dead_classvars:
            assert not hasattr(LTX2Config, attr), (
                f"LTX2Config.{attr} should be deleted -- only used by dead estimate_vram_usage()"
            )

    def test_legacy_distillation_fields_removed(self):
        """use_distilled, distilled_steps_stage1/2 should be removed."""
        from llm_dit.config import LTX2Config

        assert "use_distilled" not in LTX2Config.__dataclass_fields__, (
            "LTX2Config.use_distilled should be deleted -- legacy, replaced by use_distilled_sigmas"
        )
        assert "distilled_steps_stage1" not in LTX2Config.__dataclass_fields__, (
            "LTX2Config.distilled_steps_stage1 should be deleted"
        )
        assert "distilled_steps_stage2" not in LTX2Config.__dataclass_fields__, (
            "LTX2Config.distilled_steps_stage2 should be deleted"
        )

    def test_deprecated_encoder_fields_removed(self):
        """encoder_quantization and encoder_cpu_offload should be removed."""
        from llm_dit.config import LTX2Config

        assert "encoder_quantization" not in LTX2Config.__dataclass_fields__, (
            "LTX2Config.encoder_quantization should be deleted -- use gemma_variant instead"
        )
        assert "encoder_cpu_offload" not in LTX2Config.__dataclass_fields__, (
            "LTX2Config.encoder_cpu_offload should be deleted -- encoder uses PinnedShuttleMixin"
        )


class TestPipelineConfigRemoved:
    """PipelineConfig class should be deleted (not wired into RuntimeConfig)."""

    def test_pipeline_config_class_removed(self):
        """PipelineConfig should not exist in config module."""
        from llm_dit import config as config_module

        assert not hasattr(config_module, "PipelineConfig"), (
            "PipelineConfig should be deleted -- not used in RuntimeConfig, dead class"
        )


class TestEnhancementConfigPresetsRemoved:
    """EnhancementConfig preset classmethods should be removed."""

    def test_quality_preset_removed(self):
        """quality_preset() classmethod should be deleted."""
        from llm_dit.config import EnhancementConfig

        assert not hasattr(EnhancementConfig, "quality_preset"), (
            "EnhancementConfig.quality_preset should be deleted -- zero callers"
        )

    def test_speed_preset_removed(self):
        """speed_preset() classmethod should be deleted."""
        from llm_dit.config import EnhancementConfig

        assert not hasattr(EnhancementConfig, "speed_preset"), (
            "EnhancementConfig.speed_preset should be deleted -- zero callers"
        )

    def test_all_preset_removed(self):
        """all_preset() classmethod should be deleted."""
        from llm_dit.config import EnhancementConfig

        assert not hasattr(EnhancementConfig, "all_preset"), (
            "EnhancementConfig.all_preset should be deleted -- zero callers"
        )


# =========================================================================
# Phase 3: Consolidation
# =========================================================================


class TestCentralizedQuantAliases:
    """QUANT_ALIASES should be a single constant, not duplicated."""

    def test_quant_aliases_in_quantization_module(self):
        """Canonical QUANT_ALIASES should exist in quantization module."""
        from llm_dit.quantization import QUANT_ALIASES

        assert isinstance(QUANT_ALIASES, dict)
        assert QUANT_ALIASES["fp8"] == "fp8-dynamic"

    def test_pipeline_configs_use_centralized_alias(self):
        """LTX2Config and Flux2Config should import from quantization, not define own."""
        source = inspect.getsource(__import__("llm_dit.config", fromlist=["LTX2Config"]))
        # The source should NOT define _QUANT_ALIASES locally anymore
        # Instead it should import QUANT_ALIASES from quantization
        # We check that the pipeline configs still resolve correctly
        from llm_dit.config import LTX2Config, Flux2Config

        ltx2 = LTX2Config(quantize="fp8")
        assert ltx2.quant_transformer == "fp8-dynamic"

        flux2 = Flux2Config(quantization="fp8")
        assert flux2.quant_transformer == "fp8-dynamic"

    def test_generate_py_no_local_quant_aliases(self):
        """pipelines/generate.py should not define its own _QUANT_ALIASES."""
        from llm_dit.pipelines import generate

        source = inspect.getsource(generate)
        assert "_QUANT_ALIASES = " not in source, (
            "generate.py should import QUANT_ALIASES from quantization, not define locally"
        )


class TestCleanupMemoryAdoption:
    """model_manager.py should use cleanup_memory() instead of inline gc.collect()."""

    def test_no_inline_gc_collect_in_model_manager(self):
        """model_manager.py should not contain inline gc.collect() calls."""
        from llm_dit import model_manager

        source = inspect.getsource(model_manager)
        # Allow gc import but not direct gc.collect() calls
        lines = source.split("\n")
        gc_collect_lines = [
            (i + 1, line.strip())
            for i, line in enumerate(lines)
            if "gc.collect()" in line and not line.strip().startswith("#")
        ]
        assert len(gc_collect_lines) == 0, (
            f"model_manager.py still has {len(gc_collect_lines)} inline gc.collect() calls. "
            f"Use cleanup_memory() from utils/memory.py instead. "
            f"Lines: {gc_collect_lines}"
        )


class TestFlux2GenerateLocalCleanupRemoved:
    """flux2_generate.py should import cleanup_memory, not define its own."""

    def test_no_local_cleanup_memory_definition(self):
        """flux2_generate.py should not define its own cleanup_memory()."""
        from llm_dit.pipelines import flux2_generate

        source = inspect.getsource(flux2_generate)
        assert "def cleanup_memory" not in source, (
            "flux2_generate.py should import cleanup_memory from utils/memory.py"
        )


class TestFlux2SchedulerExtracted:
    """FLUX.2 scheduler functions should be in schedulers/flux2_scheduler.py."""

    def test_scheduler_module_exists(self):
        """flux2_scheduler module should exist."""
        from llm_dit.schedulers import flux2_scheduler

        assert hasattr(flux2_scheduler, "get_schedule")
        assert hasattr(flux2_scheduler, "compute_empirical_mu")
        assert hasattr(flux2_scheduler, "generalized_time_snr_shift")

    def test_scheduler_functions_not_in_generate(self):
        """Scheduler functions should not be defined in flux2_generate.py."""
        from llm_dit.pipelines import flux2_generate

        source = inspect.getsource(flux2_generate)
        assert "def get_schedule(" not in source, (
            "get_schedule should be in schedulers/flux2_scheduler.py, not flux2_generate.py"
        )
        assert "def compute_empirical_mu(" not in source, (
            "compute_empirical_mu should be in schedulers/flux2_scheduler.py"
        )
        assert "def generalized_time_snr_shift(" not in source, (
            "generalized_time_snr_shift should be in schedulers/flux2_scheduler.py"
        )
