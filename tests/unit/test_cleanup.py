"""
Tests for codebase cleanup: dead code deletion, CUDA guards, consolidation.

Last Updated: 2026-03-13

Covers:
- CUDA guard consistency in model_manager unload methods
- Dead RuntimeConfig properties removed
- Dead LTX2Config methods/fields removed
- Dead PipelineConfig class removed
- EnhancementConfig preset classmethods removed
- cleanup_memory() adoption in model_manager (no inline gc.collect)
- Centralized QUANT_ALIASES constant
- v0.9.27: Dead stage2_steps removal, stage1_steps default fix, scheduler
  token count fix, pipeline_mode restructure, dead code removal
- v0.9.28: Dead offload_mode/use_fp8 removal, VAE PinnedShuttleMixin
- v0.9.31: print() -> logger, _attach_weight_scales move, memory util
  consolidation, pipeline_mode="distilled" dead code removal

Run with: uv run pytest tests/unit/test_cleanup.py -v
"""

import ast
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


# =========================================================================
# v0.9.27: Pipeline cleanup
# =========================================================================


class TestDeadStage2StepsRemoved:
    """stage2_steps is a dead parameter -- only used for progress callback.
    Actual stage 2 always uses hardcoded STAGE_2_DISTILLED_SIGMA_VALUES.
    """

    def test_stage2_steps_not_in_two_stage_config(self):
        """TwoStageConfig should not have stage2_steps field."""
        from llm_dit.pipelines.generate import TwoStageConfig

        assert "stage2_steps" not in TwoStageConfig.__dataclass_fields__, (
            "TwoStageConfig.stage2_steps should be deleted -- "
            "stage 2 always uses hardcoded STAGE_2_DISTILLED_SIGMA_VALUES"
        )

    def test_stage2_num_inference_steps_not_in_ltx2_config(self):
        """LTX2Config should not have stage2_num_inference_steps field."""
        from llm_dit.config import LTX2Config

        assert "stage2_num_inference_steps" not in LTX2Config.__dataclass_fields__, (
            "LTX2Config.stage2_num_inference_steps should be deleted -- dead parameter"
        )

    def test_stage2_steps_not_in_web_schema(self):
        """Web schema should not have stage2_steps field."""
        from web.schemas import LTX2GenerateRequest

        assert "stage2_steps" not in LTX2GenerateRequest.model_fields, (
            "LTX2GenerateRequest.stage2_steps should be deleted -- dead parameter"
        )

    def test_stage2_steps_not_in_ui_schema(self):
        """UI schema should not have a stage2_steps ParamSchema."""
        from llm_dit.pipelines.schemas import get_pipeline

        schema = get_pipeline("ltx2")
        param_ids = [p.id for p in schema.params]
        assert "stage2_steps" not in param_ids, (
            "stage2_steps should be removed from LTX2 UI schema"
        )


class TestStage1StepsDefault:
    """TwoStageConfig.stage1_steps default should be 30 (V2.3), not 40 (V2.0)."""

    def test_two_stage_config_stage1_default_is_30(self):
        """TwoStageConfig stage1_steps default should match V2.3 (30 steps)."""
        from llm_dit.pipelines.generate import TwoStageConfig

        config = TwoStageConfig()
        assert config.stage1_steps == 30, (
            f"TwoStageConfig.stage1_steps default is {config.stage1_steps}, should be 30 "
            "(V2.3 default; 40 was V2.0)"
        )

    def test_ui_schema_stage1_default_is_30(self):
        """UI schema stage1_steps default should be 30."""
        from llm_dit.pipelines.schemas import get_pipeline

        schema = get_pipeline("ltx2")
        stage1_param = next(p for p in schema.params if p.id == "stage1_steps")
        assert stage1_param.default == 30, (
            f"UI schema stage1_steps default is {stage1_param.default}, should be 30"
        )


class TestSchedulerTokenCount:
    """Scheduler should use default_number_of_tokens=4096 when no latent
    provided, matching the official reference implementation."""

    def test_scheduler_default_tokens_is_4096(self):
        """LTX2Scheduler with latent=None should use 4096 tokens (MAX_SHIFT_ANCHOR)."""
        from llm_dit.schedulers.ltx2_scheduler import LTX2Scheduler, MAX_SHIFT_ANCHOR

        assert MAX_SHIFT_ANCHOR == 4096, (
            f"MAX_SHIFT_ANCHOR is {MAX_SHIFT_ANCHOR}, should be 4096"
        )

        # When no latent is passed, scheduler uses MAX_SHIFT_ANCHOR
        scheduler = LTX2Scheduler()
        sigmas = scheduler.execute(steps=30, latent=None)
        assert len(sigmas) == 31  # steps + 1

    def test_generate_does_not_pass_mock_latent(self):
        """generate.py should not create a mock_latent for scheduler.
        It should pass latent=None to use the reference 4096 default."""
        from llm_dit.pipelines import generate

        source = inspect.getsource(generate)
        assert "mock_latent" not in source, (
            "generate.py still creates a mock_latent for scheduler. "
            "Should pass latent=None to use reference 4096 default tokens."
        )


class TestPipelineMode:
    """use_distilled_sigmas should be replaced with pipeline_mode enum."""

    def test_use_distilled_sigmas_not_in_two_stage_config(self):
        """TwoStageConfig should not have use_distilled_sigmas."""
        from llm_dit.pipelines.generate import TwoStageConfig

        assert "use_distilled_sigmas" not in TwoStageConfig.__dataclass_fields__, (
            "TwoStageConfig.use_distilled_sigmas should be replaced with pipeline_mode"
        )

    def test_use_distilled_sigmas_not_in_ltx2_config(self):
        """LTX2Config should not have use_distilled_sigmas."""
        from llm_dit.config import LTX2Config

        assert "use_distilled_sigmas" not in LTX2Config.__dataclass_fields__, (
            "LTX2Config.use_distilled_sigmas should be replaced with pipeline_mode"
        )

    def test_use_distilled_sigmas_not_in_web_schema(self):
        """Web schema should not have use_distilled_sigmas."""
        from web.schemas import LTX2GenerateRequest

        assert "use_distilled_sigmas" not in LTX2GenerateRequest.model_fields, (
            "LTX2GenerateRequest.use_distilled_sigmas should be replaced with pipeline_mode"
        )

    def test_pipeline_mode_in_two_stage_config(self):
        """TwoStageConfig should have pipeline_mode field."""
        from llm_dit.pipelines.generate import TwoStageConfig

        assert "pipeline_mode" in TwoStageConfig.__dataclass_fields__, (
            "TwoStageConfig should have pipeline_mode field"
        )

    def test_pipeline_mode_default_is_standard(self):
        """Default pipeline_mode should be 'standard'."""
        from llm_dit.pipelines.generate import TwoStageConfig

        config = TwoStageConfig()
        assert config.pipeline_mode == "standard", (
            f"TwoStageConfig.pipeline_mode default is '{config.pipeline_mode}', "
            "should be 'standard'"
        )

    def test_pipeline_mode_in_ltx2_config(self):
        """LTX2Config should have pipeline_mode field."""
        from llm_dit.config import LTX2Config

        assert "pipeline_mode" in LTX2Config.__dataclass_fields__, (
            "LTX2Config should have pipeline_mode field"
        )

    def test_pipeline_mode_in_web_schema(self):
        """Web schema should have pipeline_mode field."""
        from web.schemas import LTX2GenerateRequest

        assert "pipeline_mode" in LTX2GenerateRequest.model_fields, (
            "LTX2GenerateRequest should have pipeline_mode field"
        )

    def test_use_distilled_sigmas_not_in_ui_schema(self):
        """UI schema should not have use_distilled_sigmas."""
        from llm_dit.pipelines.schemas import get_pipeline

        schema = get_pipeline("ltx2")
        param_ids = [p.id for p in schema.params]
        assert "use_distilled_sigmas" not in param_ids, (
            "use_distilled_sigmas should be replaced with pipeline_mode in UI schema"
        )

    def test_pipeline_mode_in_ui_schema(self):
        """UI schema should have pipeline_mode control."""
        from llm_dit.pipelines.schemas import get_pipeline

        schema = get_pipeline("ltx2")
        param_ids = [p.id for p in schema.params]
        assert "pipeline_mode" in param_ids, (
            "LTX2_PARAMS should have a pipeline_mode control"
        )


class TestDeadCodeRemoval:
    """Dead functions and exports should be removed."""

    def test_load_ltx2_transformer_quantized_removed(self):
        """load_ltx2_transformer_quantized should be removed from ltx2 loader.
        It imports deleted llm_dit.utils.quantization module."""
        from llm_dit.models.ltx2 import loader

        assert not hasattr(loader, "load_ltx2_transformer_quantized"), (
            "load_ltx2_transformer_quantized should be deleted -- imports deleted module"
        )

    def test_load_ltx2_from_diffusers_removed(self):
        """load_ltx2_from_diffusers should be removed. V2.3 doesn't use diffusers format."""
        from llm_dit.models.ltx2 import loader

        assert not hasattr(loader, "load_ltx2_from_diffusers"), (
            "load_ltx2_from_diffusers should be deleted -- V2.3 does not use diffusers"
        )

    def test_get_model_info_removed_from_ltx2(self):
        """get_model_info should be removed from ltx2 loader. Crude estimates, never called."""
        from llm_dit.models.ltx2 import loader

        assert not hasattr(loader, "get_model_info"), (
            "ltx2 loader.get_model_info should be deleted -- crude estimates, zero callers"
        )

    def test_fuse_delta_into_weight_removed(self):
        """_fuse_delta_into_weight should be removed from lora.py. Zero callers."""
        from llm_dit.utils import lora

        assert not hasattr(lora, "_fuse_delta_into_weight"), (
            "_fuse_delta_into_weight should be deleted -- zero callers, "
            "replaced by _fuse_delta with weight_scale handling"
        )

    def test_load_ltx2_full_removed(self):
        """_load_ltx2_full should be removed from ModelManager.
        Only called from deprecated _register_pipeline_loader()."""
        from llm_dit.model_manager import ModelManager

        assert not hasattr(ModelManager, "_load_ltx2_full"), (
            "ModelManager._load_ltx2_full should be deleted -- "
            "only called from deprecated _register_pipeline_loader()"
        )

    def test_dead_ui_controls_removed(self):
        """stg_start_step and stg_end_step should be removed from UI schema.
        They have config_mapped=False and no corresponding API fields."""
        from llm_dit.pipelines.schemas import get_pipeline

        schema = get_pipeline("ltx2")
        param_ids = [p.id for p in schema.params]
        assert "stg_start_step" not in param_ids, (
            "stg_start_step should be removed from UI schema -- "
            "config_mapped=False and no API field"
        )
        assert "stg_end_step" not in param_ids, (
            "stg_end_step should be removed from UI schema -- "
            "config_mapped=False and no API field"
        )

    def test_quantized_loader_not_in_ltx2_init_exports(self):
        """load_ltx2_transformer_quantized should not be in ltx2 __init__ exports."""
        import llm_dit.models.ltx2 as ltx2_pkg
        source = inspect.getsource(ltx2_pkg)
        assert "load_ltx2_transformer_quantized" not in source, (
            "load_ltx2_transformer_quantized should be removed from ltx2/__init__.py exports"
        )


class TestDuplicateStepFields:
    """LTX2Config should not have duplicate step count fields."""

    def test_num_inference_steps_removed(self):
        """num_inference_steps should be removed (duplicate of stage1_num_inference_steps).
        We only support two-stage since v0.9.20."""
        from llm_dit.config import LTX2Config

        assert "num_inference_steps" not in LTX2Config.__dataclass_fields__, (
            "LTX2Config.num_inference_steps should be deleted -- "
            "duplicate of stage1_num_inference_steps, single-stage removed in v0.9.20"
        )


# =========================================================================
# v0.9.28: Offloading audit & cleanup
# =========================================================================


class TestDeadOffloadModeRemoved:
    """offload_mode and num_blocks_per_group are dead config fields.
    No generation function ever reads them. The actual offloading is
    hardcoded as sequential component offloading with pinned memory."""

    def test_offload_mode_not_in_ltx2_config(self):
        """offload_mode should be removed from LTX2Config."""
        from llm_dit.config import LTX2Config

        assert "offload_mode" not in LTX2Config.__dataclass_fields__, (
            "LTX2Config.offload_mode should be deleted -- dead code, "
            "no generation function reads it"
        )

    def test_num_blocks_per_group_not_in_ltx2_config(self):
        """num_blocks_per_group should be removed from LTX2Config."""
        from llm_dit.config import LTX2Config

        assert "num_blocks_per_group" not in LTX2Config.__dataclass_fields__, (
            "LTX2Config.num_blocks_per_group should be deleted -- dead code, "
            "only used by offload_mode validation"
        )

    def test_offload_type_not_in_ui_schema(self):
        """UI schema should not have offload_type ParamSchema."""
        from llm_dit.pipelines.schemas import get_pipeline

        schema = get_pipeline("ltx2")
        param_ids = [p.id for p in schema.params]
        assert "offload_type" not in param_ids, (
            "offload_type should be removed from LTX2 UI schema -- dead control"
        )

    def test_offload_type_not_in_config_mgmt_map(self):
        """config_mgmt should not have offload_type mapping."""
        from web.routers.config_mgmt import _PARAM_NAME_MAPS

        ltx2_map = _PARAM_NAME_MAPS.get("ltx2", {})
        assert "offload_type" not in ltx2_map, (
            "offload_type mapping should be removed from config_mgmt._PARAM_NAME_MAPS"
        )

    def test_offload_options_constant_removed(self):
        """OFFLOAD_OPTIONS constant should be removed from ltx2 schema."""
        from llm_dit.pipelines.schemas import ltx2 as ltx2_schema

        assert not hasattr(ltx2_schema, "OFFLOAD_OPTIONS"), (
            "OFFLOAD_OPTIONS should be deleted from ltx2 schema module"
        )

    def test_config_migration_strips_offload_mode(self):
        """Old config.toml with offload_mode should not crash during parsing."""
        from llm_dit.config import Config

        # Simulate old config data with dead fields
        data = {
            "ltx2": {
                "model_path": "/tmp/test",
                "offload_mode": "model",
                "num_blocks_per_group": 2,
            }
        }
        config = Config.from_dict(data)
        assert not hasattr(config.ltx2, "offload_mode")


class TestDeadUseFp8Removed:
    """use_fp8 maps to LTX2Config.quantize via config_mgmt. The quantize field
    IS consumed, but use_fp8 is an infrastructure concern -- the transformer is
    loaded as fp8 or not at cache time. Changing per-generation is misleading."""

    def test_use_fp8_not_in_ui_schema(self):
        """UI schema should not have use_fp8 ParamSchema."""
        from llm_dit.pipelines.schemas import get_pipeline

        schema = get_pipeline("ltx2")
        param_ids = [p.id for p in schema.params]
        assert "use_fp8" not in param_ids, (
            "use_fp8 should be removed from LTX2 UI schema -- "
            "infrastructure concern, not a per-generation param"
        )

    def test_use_fp8_not_in_config_mgmt_map(self):
        """config_mgmt should not have use_fp8 mapping."""
        from web.routers.config_mgmt import _PARAM_NAME_MAPS

        ltx2_map = _PARAM_NAME_MAPS.get("ltx2", {})
        assert "use_fp8" not in ltx2_map, (
            "use_fp8 mapping should be removed from config_mgmt._PARAM_NAME_MAPS"
        )


class TestVAEPinnedShuttleMixin:
    """VideoDecoder should use PinnedShuttleMixin for proper pinned memory
    round-trips instead of _pin_model_memory + bare .to() calls."""

    def test_video_decoder_has_shuttle_mixin(self):
        """VideoDecoder should inherit from PinnedShuttleMixin."""
        from llm_dit.models.ltx2.vae.video_vae import VideoDecoder
        from llm_dit.utils.shuttle import PinnedShuttleMixin

        assert issubclass(VideoDecoder, PinnedShuttleMixin), (
            "VideoDecoder should inherit from PinnedShuttleMixin "
            "for proper pinned memory round-trips"
        )

    def test_video_decoder_has_offload_to_pinned(self):
        """VideoDecoder should have offload_to_pinned method (from mixin)."""
        from llm_dit.models.ltx2.vae.video_vae import VideoDecoder

        assert hasattr(VideoDecoder, "offload_to_pinned")

    def test_video_decoder_has_offload(self):
        """VideoDecoder should have offload method (from mixin)."""
        from llm_dit.models.ltx2.vae.video_vae import VideoDecoder

        assert hasattr(VideoDecoder, "offload")

    def test_video_decoder_has_to_device(self):
        """VideoDecoder should have to_device method (from mixin)."""
        from llm_dit.models.ltx2.vae.video_vae import VideoDecoder

        assert hasattr(VideoDecoder, "to_device")


class TestAudioComponentsShuttleMixin:
    """AudioDecoder and Vocoder/VocoderWithBWE should use PinnedShuttleMixin."""

    def test_audio_decoder_has_shuttle_mixin(self):
        """AudioDecoder should inherit from PinnedShuttleMixin."""
        from llm_dit.models.ltx2.audio_vae.decoder import AudioDecoder
        from llm_dit.utils.shuttle import PinnedShuttleMixin

        assert issubclass(AudioDecoder, PinnedShuttleMixin), (
            "AudioDecoder should inherit from PinnedShuttleMixin"
        )

    def test_vocoder_with_bwe_has_shuttle_mixin(self):
        """VocoderWithBWE should inherit from PinnedShuttleMixin."""
        from llm_dit.models.ltx2.audio_vae.vocoder import VocoderWithBWE
        from llm_dit.utils.shuttle import PinnedShuttleMixin

        assert issubclass(VocoderWithBWE, PinnedShuttleMixin), (
            "VocoderWithBWE should inherit from PinnedShuttleMixin"
        )


class TestPinModelMemoryRemoved:
    """_pin_model_memory should be removed after all callers migrate to
    PinnedShuttleMixin.offload_to_pinned()."""

    def test_pin_model_memory_removed(self):
        """_pin_model_memory should not exist on ModelManager."""
        from llm_dit.model_manager import ModelManager

        assert not hasattr(ModelManager, "_pin_model_memory"), (
            "ModelManager._pin_model_memory should be deleted -- "
            "all callers migrated to PinnedShuttleMixin.offload_to_pinned()"
        )


# =========================================================================
# v0.9.31: Tech debt cleanup
# =========================================================================


def _count_print_calls(filepath: str) -> list[int]:
    """AST-parse a file and return line numbers of print() calls."""
    with open(filepath) as f:
        tree = ast.parse(f.read(), filename=filepath)
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "print":
                lines.append(node.lineno)
    return lines


class TestNoPrintStatements:
    """Production code should use logger, not print()."""

    def test_no_print_in_flux2_loader(self):
        """flux2/loader.py should have zero print() calls."""
        from llm_dit.models.flux2 import loader
        filepath = inspect.getfile(loader)
        lines = _count_print_calls(filepath)
        assert lines == [], (
            f"flux2/loader.py has print() at lines {lines}. Use logger instead."
        )

    def test_no_print_in_wan_dit(self):
        """wan_dit.py should have zero print() calls."""
        from llm_dit.models import wan_dit
        filepath = inspect.getfile(wan_dit)
        lines = _count_print_calls(filepath)
        assert lines == [], (
            f"wan_dit.py has print() at lines {lines}. Use logger instead."
        )


class TestAttachWeightScalesLocation:
    """_attach_weight_scales should live in quantization/fp8_cast.py,
    not models/ltx2/loader.py."""

    def test_importable_from_fp8_cast(self):
        """_attach_weight_scales should be importable from quantization.fp8_cast."""
        from llm_dit.quantization.fp8_cast import _attach_weight_scales
        assert callable(_attach_weight_scales)

    def test_not_defined_in_ltx2_loader(self):
        """ltx2/loader.py should not define _attach_weight_scales."""
        from llm_dit.models.ltx2 import loader
        filepath = inspect.getfile(loader)
        with open(filepath) as f:
            tree = ast.parse(f.read(), filename=filepath)
        func_names = [
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        ]
        assert "_attach_weight_scales" not in func_names, (
            "_attach_weight_scales should be moved to quantization/fp8_cast.py"
        )


class TestFormatMemoryGb:
    """format_memory_gb should live in utils/memory.py."""

    def test_format_zero(self):
        from llm_dit.utils.memory import format_memory_gb
        assert format_memory_gb(0) == "0.00GB"

    def test_format_one_gb(self):
        from llm_dit.utils.memory import format_memory_gb
        assert format_memory_gb(1e9) == "1.00GB"

    def test_format_fractional(self):
        from llm_dit.utils.memory import format_memory_gb
        assert format_memory_gb(1.5e9) == "1.50GB"


class TestLogMemoryDebug:
    """log_memory_debug should not crash regardless of CUDA availability."""

    def test_no_crash(self):
        from llm_dit.utils.memory import log_memory_debug
        # Should not raise
        log_memory_debug("test", component="Test")


class TestMemoryFunctionConsolidation:
    """Neither flux2/loader.py nor flux2/transformer.py should define
    local _format_memory_gb or _log_memory_state functions."""

    def _get_local_func_names(self, module):
        filepath = inspect.getfile(module)
        with open(filepath) as f:
            tree = ast.parse(f.read(), filename=filepath)
        return [
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        ]

    def test_no_format_memory_gb_in_loader(self):
        from llm_dit.models.flux2 import loader
        names = self._get_local_func_names(loader)
        assert "_format_memory_gb" not in names, (
            "flux2/loader.py should import format_memory_gb from utils/memory.py"
        )

    def test_no_format_memory_gb_in_transformer(self):
        from llm_dit.models.flux2 import transformer
        names = self._get_local_func_names(transformer)
        assert "_format_memory_gb" not in names, (
            "flux2/transformer.py should import format_memory_gb from utils/memory.py"
        )

    def test_no_log_memory_state_in_loader(self):
        from llm_dit.models.flux2 import loader
        names = self._get_local_func_names(loader)
        assert "_log_memory_state" not in names, (
            "flux2/loader.py should import log_memory_debug from utils/memory.py"
        )

    def test_no_log_memory_state_in_transformer(self):
        from llm_dit.models.flux2 import transformer
        names = self._get_local_func_names(transformer)
        assert "_log_memory_state" not in names, (
            "flux2/transformer.py should import log_memory_debug from utils/memory.py"
        )
