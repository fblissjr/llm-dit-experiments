last updated: 2026-02-06

# changelog

All notable changes to this project will be documented in this file.
Uses [Semantic Versioning](https://semver.org/).

## 0.7.1

### removed
- Entire VL (Vision Conditioning) module: `src/llm_dit/vl/`, VL endpoints in server.py, VLConfig, VL CLI args, VL frontend JS, VL documentation
- VL experiment files: `experiments/qwen3_vl/`, `experiments/test_vl_ablation.py`, `experiments/qwen3_vl_poc.py`
- VL schema entries from Z-Image ParamSchemas (vl_enabled, vl_image, vl_strength)
- VL constants (QWEN3_VL_4B_CONFIG, QWEN3_VL_GENERATION_DEFAULTS)
- `load_in_4bit`/`load_in_8bit` boolean flags from Gemma3Encoder (replaced by `quantization_variant: str`)
- BitsAndBytes `load_in_8bit` parameter from 14 script/test files

### changed
- `config.toml.example` rewritten: added `[quantization]` section, all methods use unified torchao names, fixed `compile_mode` default
- Gemma3Encoder variant metadata: two booleans replaced by `_quantization_variant: str` ("bf16", "int8", "q4_0")
- Config presets use unified names: "8bit"->"int8", "fp8"->"fp8-weight-only", "4bit"->"int4"
- CLAUDE.md: added "DRY Configuration Principles" section with `--hidden-layer` reference

### fixed
- 27 unit test failures resolved (wrong mock patch targets, argument mismatches, outdated expectations)
- 3 missed VL CLI args in cli.py (`--rewriter-no-vl`, `--rewriter-preload-vl`, `--rewriter-vl-api-model`)
- config.toml/config.py warning messages referencing deleted method names

## 0.7.0

### added
- Unified `quantize_component()` entry point for all model components across all pipelines
- `ComponentQuantConfig` and `PipelineQuantConfig` dataclasses for type-safe quantization config
- Global `[quantization]` TOML section with per-pipeline overrides (resolution: pipeline override > global default)
- `get_pipeline_quant_config()` on RuntimeConfig for resolving effective quantization per pipeline
- `get_quant_compile_warnings()` for detecting dangerous quant + compile combinations
- `VALID_METHODS` constant: `none`, `fp8-dynamic`, `fp8-weight-only`, `int8`, `int4`
- New unit tests for `quantize_component()`, `VALID_METHODS`, compile warnings, and stats dict shape

### removed
- `fp8_native.py` (manual FP8 casting with allowlist) -- replaced by `Float8WeightOnlyConfig`
- `fp8_inference.py` (DiffSynth-style `F.linear` patching context manager) -- replaced by `Float8DynamicActivationFloat8WeightConfig`
- `quantization/config.py` (`QuantizationMethod` enum and BitsAndBytes helpers) -- no longer needed
- `quantize_model_torchao()` and `quantize_model_torchao_filtered()` from torchao_utils
- `create_fp8_filter_fn()` and `analyze_fp8_compatibility()` from torchao_utils
- All BitsAndBytes quantization paths (4bit, 8bit NF4) across all pipelines
- DiffSynth FP8 context manager usage from Qwen-Image pipelines
- `_build_quantization_config()` from `qwen_image_2512.py`
- `QUANTIZATION_PRESETS` dict from QwenImageConfig
- `test_fp8_inference.py` test file

### changed
- All 4 pipelines (FLUX.2, LTX-2, Z-Image, Qwen-Image) now use `quantize_component()` as sole quantization entry point
- torchao is the sole quantization backend (BitsAndBytes dependency removed from quantization paths)
- `get_recommended_method()` returns unified method names (`"fp8-weight-only"` instead of `"fp8"`, `"int8"` instead of `"8bit"`)
- Encoder quantization uses post-load pattern (load BF16 then quantize) instead of BnB during-load
- Config field names unified: removed `flux2_quantization`, `ltx2_quantize`, `qwen_image_quantize_*` in favor of `<pipeline>_quant_<component>`
- Updated `docs/reference/quantization.md` with migration table from old to new API
- **Missed layers cleanup**: Fixed remaining ~40% of codebase still referencing old method names
  - Fixed runtime crash bugs: `generate.py` default param, CLI choices, Qwen variant defaults, QwenImageConfig validation
  - Removed BnB from: EncoderConfig, BackendConfig, backends/qwen_image.py, encoders/gemma3.py, backends/transformers.py
  - Deleted dead code: `utils/quantization.py` (quanto module), `is_bitsandbytes_available()`, BnB migration tests
  - Fixed config wiring: `ltx2_quantize` TOML default from `"fp8"` to `"fp8-weight-only"`
  - Removed `bitsandbytes` from `pyproject.toml` dependencies
  - Updated `vae_utils.py` to remove `"8bit"` BnB method, only `"int8"` remains for VAE

## 0.6.3

### changed
- Default `compile_mode` from `max-autotune-no-cudagraphs` to `default` for FLUX.2 and global optimization
  - Eliminates 5+ min Triton autotune warmup when combined with FP8 quantization
  - `default` mode still applies Inductor graph optimizations (kernel fusion, dead code elimination)
  - Users wanting maximum throughput can still set `max-autotune-no-cudagraphs` in config.toml

### added
- Compile+FP8 autotune warning in ModelCard: warns when `max-autotune` modes are used with quantization

## 0.6.2

### added
- FLUX.2 config visibility in ModelCard: colored badges for active optimization settings (FP8, compile, block_offload)
- Proactive config validation: incompatible settings (compile+block_offload, quantization+block_offload) shown as warnings before loading
- Generic data-driven config tag/warning system: backend provides tags, frontend renders them (pipeline-agnostic)

### fixed
- ModelCard VRAM display: backend now returns `vram_mb` alias (frontend expected this but only `total_vram_mb` was returned)

## 0.6.1

### added
- FLUX.2 persistent model loading: models stored in memory across requests, eliminating ~5-10s load per request
- FLUX.2 torchao FP8 quantization: transformer VRAM drops from ~18GB (BF16) to ~9GB (FP8)
- FLUX.2 encoder pinned memory shuttle: DMA-based CPU<->GPU transfers for encoder, ~2-3x faster than default
- FLUX.2 torch.compile support for transformer and VAE decoder
- `get_encoder_preset()` helper in FLUX.2 constants for DRY preset resolution
- Loading lock (`_flux2_loading_lock`) for concurrent `/api/vram/load-flux2` request safety

### fixed
- Pinned memory lost after first CUDA round-trip: `offload()` now uses shadow buffer pattern to copy CUDA tensors directly into pre-allocated pinned buffers (0 allocations vs 2N per cycle)
- Partial model cleanup on load failure: leaked encoder/transformer memory now explicitly freed
- DMA transfer now overlaps with tokenization in encoder `forward()` (~0.5-1s latency reduction)

### changed
- `quantize_to` + `block_offload` now raises `ValueError` instead of silently skipping quantization
- Added `RuntimeError` guard in transformer `forward()` for compile + block_offload incompatibility
- All logging in transformer `forward()` wrapped with `torch.compiler.is_compiling()` guards

## 0.2.0

### added
- Z-Image pipeline with Qwen3-4B encoder
- LTX-2 video pipeline with Gemma3-12B encoder
- Unified Qwen3 encoder (`qwen3_unified.py`) with preset system
- Multi-pipeline VRAM management with load/unload endpoints
- Block offload support for FLUX.2 transformer

## 0.1.0

### added
- Initial FLUX.2 Klein pipeline (4B and 9B variants)
- FastAPI server with React frontend
- Config system (TOML -> dataclass -> RuntimeConfig)
