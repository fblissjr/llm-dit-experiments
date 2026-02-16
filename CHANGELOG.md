last updated: 2026-02-16

# changelog

All notable changes to this project will be documented in this file.
Uses [Semantic Versioning](https://semver.org/).

## 0.9.9

### added
- Unified parameter resolution: `resolve_param()` helper in `web/param_resolver.py`. Establishes `client-sent > config.toml > schema default` precedence across all 4 pipelines using Pydantic v2's `model_fields_set`. Foundation for composable workflow orchestration (L1 vision).
- `csv_to_int_list()` helper for parsing comma-separated config strings (e.g., `stg_blocks = "29,30"`)
- 18 unit tests for `resolve_param()` covering: falsy value preservation (0, 0.0, ""), `skip_none` behavior, list fields, config-None graceful handling

### fixed
- **LTX-2**: config.toml generation defaults (`guidance_scale`, `stg_scale`, `stg_blocks`, `rescale_scale`, `ge_gamma`, `negative_prompt`, `num_frames`, `width`, `height`, `fps`, `distilled_lora_path`, `distilled_lora_scale`, `stage1_steps`, `stage2_steps`) now respected for API requests when client omits them. Previously, schema defaults always won.
- **LTX-2**: `stg_scale=0` (disable STG) via API was silently ignored due to `or`-operator fallback treating 0 as falsy
- **Z-Image**: variant defaults (`steps`, `guidance_scale`, `shift`) now use `model_fields_set` instead of equality comparison against hardcoded Pydantic defaults. Fixes bug where client explicitly sending `steps=9` for the base variant would get it silently overwritten.
- **FLUX.2**: `num_steps` and `guidance` resolution now consults `config.toml` (`flux2.default_steps`, `flux2.default_guidance`) as intermediate layer between client and model-specific defaults. Fixed-param validation for distilled models now uses `model_fields_set` instead of `is not None`.
- **Qwen-Image**: `steps` and `cfg_scale` on all 3 endpoints (edit-layer, edit-multi, T2I generate) now respect `config.toml` values (`qwen_image.num_inference_steps`, `qwen_image.cfg_scale`). Previously, schema default (40 steps, 4.0 CFG) always won.

### removed
- `_ZIMAGE_PYDANTIC_DEFAULTS` dict in `web/routers/core.py` (replaced by `model_fields_set` detection)

## 0.9.8

### added
- Spatio-Temporal Guidance (STG): 3rd forward pass in denoising loop where self-attention is skipped at specified transformer blocks. The delta between conditioned and perturbed predictions drives spatial coherence and temporal consistency guidance. Reference formula: `v = v_cond + (cfg-1)*(v_cond-v_uncond) + stg*(v_cond-v_perturbed)`. Default: `stg_scale=1.0`, `stg_blocks=[29]` (matches reference).
- `skip_self_attn` parameter on `BasicTransformerBlock.forward()` for STG perturbed pass
- `stg_blocks` parameter on `LTX2Transformer.forward()` -- set of block indices where self-attention is skipped
- `stg_scale` and `stg_blocks` fields on `StepContext` and `constant_schedule()`
- `stg_blocks` field on `LTX2GenerateRequest` schema (API parameter)
- STG unit tests: transformer perturbation tests (4 tests) and pipeline config tests (3 tests)
- Reference doc sections: STG (4.8), Gradient Estimation (4.9), MultiModalGuider architecture (4.10), implementation gaps (4.11)

### fixed
- Reference doc: global `scale_shift_table` shape was `[6, 4096]` (per-block), corrected to `[2, 4096]` (output projection shift+scale)
- Reference doc: Stage 2 distilled schedule was "Fixed 8-step", corrected to "Fixed 3-step" (4 sigma values = 3 steps)

### changed
- `stg_scale` default changed from `0.0` to `1.0` across all layers (config.py, generate.py TwoStageConfig, schemas.py, config.toml, config.toml.example) to match reference `DEFAULT_VIDEO_GUIDER_PARAMS.stg_scale`
- STG wired through `constant_schedule()` call in Stage 1 of two-stage pipeline (was dead code)
- `stg_blocks` parsed from config string to list in ltx2 router (config uses comma-separated string, pipeline uses `list[int]`)

## 0.9.7

### fixed
- Two-stage noise formula: fixed flow-matching interpolation in Stage 2 noise addition. Was `x_0 + sigma * eps` (additive), now `(1-sigma) * x_0 + sigma * eps` (correct flow-matching interpolation). The wrong formula put the clean signal 11x too strong at sigma=0.909, causing green garbage output.
- LoRA fusion OOM: re-quantize each layer to fp8 immediately after fusion instead of batching at end. Prevents VRAM ballooning from ~13GB (fp8) toward ~26GB (bf16) when fusing large LoRAs (e.g., rank-384 distilled LoRA). Also updated from deprecated `float8_weight_only()` function API to `Float8WeightOnlyConfig`.
- CUDA fragmentation OOM during two-stage LoRA fusion: added `cleanup_memory()` between Stage 1 and Stage 1.5 to release denoising intermediate buffers (5-8 GB reserved). Added periodic `empty_cache()` every 100 layers during 487-layer fusion loop to prevent CUDA pool fragmentation.
- torchao availability check: updated `is_torchao_available()` to import current API (`quantize_`) instead of removed `int8_weight_only` function. Fixes "torchao not available" false negative with torchao 0.16+.
- Negative prompt: replaced 3-word placeholder with full reference DEFAULT_NEGATIVE_PROMPT (~1,300 chars) tuned for suppressing common diffusion failure modes
- Gemma config: cleared mismatched `encoder_model_id` in config.toml.example (Q4 QAT path doesn't apply to 8bit variant)
- Default `guidance_scale` 3.5 -> 3.0 to match reference `DEFAULT_VIDEO_GUIDER_PARAMS.cfg_scale` (was 17% over reference)
- Default `distilled_lora_scale` 0.8 -> 1.0 to match reference (Stage 2 was under-weighted)
- Default `height x width` 768x512 -> 512x768 to match reference landscape orientation (`DEFAULT_1_STAGE_HEIGHT=512, DEFAULT_1_STAGE_WIDTH=768`)
- Smoke test dimensions 384x256 -> 256x384 (landscape, consistent with reference orientation)

### changed
- Test constants: renamed legacy distilled field names to two-stage (`use_two_stage`, `stage1_num_inference_steps`, `stage2_num_inference_steps`) in SMOKE/STANDARD/FULL dicts and TOML overlays
- Test infrastructure: deleted duplicate `tests/e2e/conftest.py` (556-line copy of integration/pipeline/conftest.py)
- Test infrastructure: moved orphan test files to proper locations (`test_web_server.py` -> integration, `test_rewriter_parsing.py` -> unit, `test_qwen3_think_tokens.py` -> scripts)
- Z-Image API tests: rewrote from external `requests.get()` to in-process TestClient pattern; removed obsolete tests against dead endpoints (`/api/generation-config`, `/api/status`)

### added
- VRAM diagnostic logging at stage transitions in two-stage pipeline (post-encoder, post-stage1, post-stage1.5, pre-stage2). Logs allocated/reserved/free GPU memory for debugging OOM failures from logs alone.
- Qwen-Image E2E smoke test: `tests/e2e/api/test_qwenimage_smoke.py` with T2I generation, status/config endpoints, and validation
- Qwen-Image test config overlay: `tests/configs/qwenimage_smoke.toml`
- Z-Image API test: `tests/e2e/api/test_zimage_api.py` (config defaults, variant checks, generation, seed reproducibility)
- Config factory: expanded `qwen_image` model path extraction to include `cpu_offload`, `quantize_text_encoder`, `quantize_transformer`

## 0.9.6

### fixed
- Two-stage pipeline: set `distilled_lora_path` in config.toml (was empty, causing neon blob artifacts from base model doing 3-step distilled denoising without the distilled LoRA)
- Two-stage pipeline: wired `encoder_model_id` config through router to pipeline (was dead config, never passed as `text_encoder_path`)

### added
- Distilled LoRA guard in pipeline: raises clear `ValueError` when `distilled_lora_path` is empty instead of silently producing garbage
- Distilled LoRA guard in router: validates file exists before starting expensive generation

### changed
- Two-stage pipeline: eliminated second transformer load by reusing Stage 1 model for Stage 2 (~15-25s savings per generation). Distilled LoRA is fused into the existing model instead of reloading from disk.

## 0.9.5

### fixed
- LTX-2 video rendering in frontend: backend SSE response now uses standard `urls` array format (was `video_url` string)
- Config defaults endpoint: extracts nested pipeline sub-dicts from `RuntimeConfig.to_dict()` (LTX-2 config.toml values now reach frontend form defaults)
- Static file serving: added `/outputs` mount for LTX-2 video files (was returning 404)

### added
- Two-stage generation controls in LTX-2 pipeline schema: `use_two_stage`, `stage1_steps`, `stage2_steps`, `rescale_scale`, `ge_gamma`

### changed
- LTX-2 pipeline schema: replaced non-functional `num_inference_steps` param (no matching Pydantic field) with `stage1_steps` and `stage2_steps`

## 0.9.4

### fixed
- LTX-2 text encoder now runs on CUDA (was stuck on CPU due to config not reaching generation functions)

### removed
- `LTX2OptimizationConfig` eliminated -- dual-state config gap replaced with explicit parameters matching FLUX.2 pattern
- Dead config.toml fields: `encoder_quantization`, `encoder_cpu_offload` (never used)

### changed
- `generate_video_with_offloading()` and `generate_video_two_stage()` now take explicit `text_encoder_device`, `transformer_device`, `vae_device`, `quantize` (str), `skip_cleanup` parameters instead of `optimization: LTX2OptimizationConfig`
- Web router (`ltx2.py`) now passes `config.ltx2.*` optimization settings to generation functions
- Added `[LTX2]` and `[LTX2:TwoStage]` entry logging with device placement and quantization

## 0.9.3

### added
- Frontend: `React.memo` on `ParamControl` with custom comparator (prevents 20+ controls re-rendering on single value change)
- Frontend: slider debounce (50ms) with pointer-up commit pattern for smooth dragging
- Frontend: per-param memoized `onChange` callbacks in `PipelineForm` (stable refs for memo comparator)
- Frontend: `ErrorBoundary` wrapping `PipelineView` with retry button
- Frontend: `getResolvedValues()` module-level reference-equality cache (avoids recomputing on every selector call)
- Frontend: shared `PIPELINE_COLOR_MAP` constant (deduplicates appStore + HistoryCard)
- Frontend: lazy-loading on history thumbnails (`loading="lazy"`)
- Frontend: `React.memo` on `HistoryCard` with `item.id` comparator (prevents 500 cards re-rendering)
- Frontend: smoke render test (`App.smoke.test.tsx`) with route-based fetch mock and jsdom setup
- Backend: `GZipMiddleware` (Starlette built-in, minimum_size=1000, SSE auto-excluded)
- Backend: `Cache-Control` headers on `/api/pipelines` (5min), `/api/presets/preset/{name}` (5min), `/api/context` (5s)

### changed
- Frontend: unified duplicate `validateParam` (merged formStore + utils/validation.ts into single source of truth)
- Frontend: `PipelineForm.handleChange` no longer depends on `pipeline` reactive selector (uses `getState()` instead)
- Frontend: production sourcemaps disabled (saves ~744KB from dist)
- Upgraded `@hey-api/openapi-ts` 0.92.3 -> 0.92.4
- Upgraded `immer` 10 -> 11 (enableMapSet still required and functional)
- Upgraded `jsdom` 27 -> 28 (devDependency only)
- Upgraded `vite` 6 -> 7 (baseline-widely-available default target)
- Upgraded `@vitejs/plugin-react` 4 -> 5
- Upgraded `react` 18 -> 19, `react-dom` 18 -> 19, `@types/react` 18 -> 19, `@types/react-dom` 18 -> 19
- Migrated Tailwind CSS 3 -> 4: `@tailwind` directives -> `@import "tailwindcss"`, config.js -> CSS `@theme`, `@layer components` -> `@utility`, PostCSS -> `@tailwindcss/vite` plugin
- Tailwind 4 class name updates: `rounded` -> `rounded-sm`, `outline-none` -> `outline-hidden`, `flex-shrink-0` -> `shrink-0`, `min-w-[3.5rem]` -> `min-w-14`
- Frontend: replaced dynamic `import('./formStore')` in appStore with static import (eliminates Vite build warning)
- Frontend: removed Tailwind 4 border-color compat shim (all usages already have explicit colors)

### fixed
- Frontend: `StatusBar.tsx` crash when `ctx.pendingRestartFields` is undefined (added optional chaining)
- FLUX.2 prompt upsampling now reads `api_model` from `config.toml [rewriter].api_model` instead of using hardcoded default (both sync and streaming endpoints)

## 0.9.2

### added
- `fixed_params` field in FLUX.2 model registry (`constants.py`) -- distilled models declare which params are baked into weights
- `get_fixed_params()` and `is_distilled()` helper functions for FLUX.2 model introspection
- Fixed params validation in FLUX.2 generation endpoints -- overrides invalid params to model defaults with user-facing warnings
- `GET /api/flux2/models/{model_name}` endpoint returning model metadata (distilled, fixed_params, defaults, fp8)
- `warnings` field on `ImageGenerationResult` response model -- propagated through both POST and SSE endpoints
- `denoise_cfg()` function for FLUX.2 base models implementing true classifier-free guidance (doubled batch, uncond+cond forward passes, CFG formula)
- Unconditional text embedding preparation for base model CFG (encodes empty string, concatenates with prompt embeddings)
- `Flux2PromptUpsampler` class using BFL's official T2I and I2I system prompts for prompt enrichment via heylookitsanllm API
- `upsample_prompt` request field and pipeline schema checkbox for optional prompt upsampling before generation
- Frontend: distilled model controls (steps, guidance) disabled with "Fixed for distilled models" label when non-base model selected
- Frontend: generation warnings displayed in amber banner above result metadata

### changed
- FLUX.2 default resolution from 1024x1024 to 1360x768 (matches BFL's official default)
- FLUX.2 dimension preset list reordered with 1360x768 as first option
- FLUX.2 base model denoising now uses `denoise_cfg()` (explicit CFG) instead of `denoise()` (guidance embedding)

## 0.9.1

### added
- LTX-2 two-stage video generation pipeline (`generate_video_two_stage()`) matching reference `TI2VidTwoStagesPipeline` architecture
- `TwoStageConfig` dataclass for two-stage pipeline parameters (guidance, STG, rescaling, gradient estimation, distilled LoRA)
- `_denoise_stage()` shared denoising kernel with CFG, CFG rescaling, and gradient estimation support
- `load_spatial_upsampler()` loader for spatial upsampler model from safetensors checkpoints
- Distilled sigma schedule constants: `DISTILLED_SIGMA_VALUES`, `STAGE_2_DISTILLED_SIGMA_VALUES`
- Two-stage config fields in `LTX2Config`: `use_two_stage`, `stage1_num_inference_steps`, `stage2_num_inference_steps`, `spatial_upsampler_file`, `distilled_lora_path`, `distilled_lora_scale`, `stg_scale`, `stg_blocks`, `rescale_scale`, `ge_gamma`, `negative_prompt`
- Two-stage request fields in `LTX2GenerateRequest`: `use_two_stage`, `stage1_steps`, `stage2_steps`, `stg_scale`, `rescale_scale`, `distilled_lora_path`, `distilled_lora_scale`, `ge_gamma`
- Enhanced `vram_load_ltx2()` file validation: checks transformer, encoder, VAE, spatial upsampler, and distilled LoRA existence
- 17 new unit tests for two-stage config, distilled sigma schedules, half-resolution latents, and position indices

### changed
- LTX-2 default encoder quantization from `"none"` to `"fp8-weight-only"` (RTX 4090 has native FP8 tensor cores; INT4 is emulated)
- LTX-2 default transformer from distilled (`ltx-2-19b-distilled-fp8.safetensors`) to dev (`ltx-2-19b-dev-fp8.safetensors`)
- LTX-2 resolution snapping from 32-divisible to 64-divisible (two-stage requires half-res dimensions divisible by 32)
- `config.toml` `[ltx2]` section: full 1:1 alignment with `LTX2Config` dataclass fields
- `get_ltx2_model_path()` and `ltx2_status()`: removed TOML re-parsing, now use injected RuntimeConfig directly
- `ModelManager._load_ltx2()`: removed TOML re-parsing, uses injected config, validates all required files

### fixed
- `Config.load()` bug in ltx2 router (method is `Config.from_toml()`, but now bypassed entirely via injected config)

## 0.9.0

### added
- IndexedDB storage adapter (`idbStorage.ts`) for zustand persist middleware -- replaces localStorage with ~50MB+ async storage
- One-time `migrateFromLocalStorage()` function for seamless upgrade of existing history data
- OpenAPI TypeScript codegen pipeline: `npm run export-openapi && npm run gen-api` generates frontend types from FastAPI OpenAPI spec
- 3 new Pydantic response models: `ParamSchemaResponse`, `PipelineSchemaResponse`, `PresetDetailResponse`
- `_ensure_qwen_image_loaded()` and `_ensure_qwen_image_t2i_loaded()` helpers for on-demand pipeline loading via ModelManager
- `_get_zimage_encoder()` and `_ensure_zimage_loaded()` helpers in `core.py` for ModelManager access
- `_LOADED_PIPELINE_NAMES` mapping in `config_mgmt.py` for canonical ModelManager ID -> frontend API name translation
- `internal/state/backlog.md` -- prioritized improvement backlog

### changed
- **Dual state unification complete:** ModelManager is now the sole source of truth for all pipeline state across all 7 routers
- `core.py`: all 71 `srv.*` references migrated to ConfigDep/ManagerDep dependency injection
- `qwen_image.py`: 3 direct pipeline instantiation sites replaced with ModelManager `load()`/`get_pipeline()`
- `vram.py`: all unload functions use `manager.unload()` instead of server.py shims; all "is loaded?" checks use `manager.is_loaded()`
- `flux2.py`: all ~20 `srv.flux2_pipeline` references replaced with `manager.is_loaded("flux2")` / `manager.get_pipeline("flux2")`; `import web.server as srv` removed entirely
- `config_mgmt.py`: all ~10 pipeline reads replaced with `manager.is_loaded()` loop; `import web.server as srv` removed entirely
- Frontend types: hybrid strategy -- generated types re-exported where fit, hand-written kept where generated are too loose
- `server.py`: reduced from ~491 to ~296 lines
- History storage backend: localStorage -> IndexedDB (async, ~50MB+ quota, no main thread blocking)
- `MAX_HISTORY_ITEMS` increased from 100 to 500 (IndexedDB quota supports this comfortably)

### removed
- 6 pipeline globals from `server.py`: `pipeline`, `encoder`, `qwen_image_pipeline`, `qwen_image_t2i_pipeline`, `ltx2_pipeline`, `flux2_pipeline` -- ModelManager owns all pipeline state now
- 6 dead functions from `server.py` (~180 lines): `unload_zimage_pipeline()`, `unload_qwen_image_pipeline()`, `unload_qwen_image_t2i_pipeline()`, `unload_ltx2_pipeline()`, `get_vram_status()`, `load_zimage_pipeline_on_demand()`
- `_sync_globals_after_load()` and `_sync_globals_after_unload()` shim functions from `vram.py` + all 9 call sites
- Inline sync writes to `srv.pipeline`, `srv.encoder`, `srv.qwen_image_pipeline`, `srv.qwen_image_t2i_pipeline` from `core.py` and `qwen_image.py` helper functions
- `gc` and `torch` imports from `server.py` (no longer needed after unload function removal)
- `import web.server as srv` from `flux2.py` and `config_mgmt.py` (no longer access any server globals)
- `quotaHandlingStorage` from `sessionStore.ts` (~67 lines of localStorage quota error handling) -- replaced by IndexedDB adapter

### fixed
- `/api/context` returning 500 when server started without `--profile` flag -- `getattr(config, "current_profile")` returned `None` which Pydantic rejected as non-string
- `RuntimeError: Cannot set version_counter for inference tensor` during FP8-quantized generation -- reverted `@torch.inference_mode()` to `@torch.no_grad()` on all 4 generation executor functions (torchao's Float8Tensor dispatch requires version counter support that inference_mode disables)

## 0.8.9

### added
- `torch.no_grad()` wrappers on all generation executor functions (Z-Image, FLUX.2, LTX-2, Qwen-Image) to prevent autograd graph accumulation during inference
- `finally` cleanup blocks (`gc.collect()` + `torch.cuda.empty_cache()`) on all generation endpoints (streaming and non-streaming) to recover VRAM after errors
- `torch._dynamo.reset()` in FLUX.2 unload path to release compiled CUDA kernel cache (~3-5GB)
- `gc.collect()` before `empty_cache()` in Qwen-Image unload paths (server.py + model_manager.py)
- `AbortController` signal support in `generateStream()` for SSE cancellation
- 4 new Pydantic response models: `DyPEStatusResponse`, `PipelinesResponse`, `PipelineDefaultsResponse`, `ResolutionConfigResponse`
- `response_model=` applied to 4 remaining untyped endpoints (`/api/dype/status`, `/api/pipelines`, `/api/pipelines/{id}/defaults`, `/api/resolution-config`)
- `create_app()` factory function in server.py for OpenAPI spec extraction
- `scripts/export_openapi.py` for headless OpenAPI JSON export
- `openapi-ts.config.ts` and npm scripts (`export-openapi`, `gen-api`) for TypeScript codegen scaffolding

### changed
- Frontend context polling switched from `setInterval` to `setTimeout` chaining (prevents request pile-up during slow responses)
- Backend history entries no longer store `image_b64` (frontend IndexedDB is the image source of truth, saves ~150-250MB heap at 50 entries)
- Qwen-Image history entries no longer store base64 image data

### fixed
- CUDA memory leak: generation without `torch.no_grad()` built autograd graphs holding intermediate tensors (estimated 2-8GB per generation)
- CUDA memory leak: failed generations never called `empty_cache()`, leaving dead tensors in VRAM
- CUDA memory leak: FLUX.2 unload skipped `_dynamo.reset()`, leaving compiled kernels (~3-5GB) in VRAM
- CUDA memory leak: Qwen-Image unload skipped `gc.collect()`, leaving Python-held CUDA tensors unreclaimable

## 0.8.8

### added
- Pydantic `CamelModel` base class with automatic camelCase JSON serialization (`alias_generator=to_camel`)
- ~27 typed response models in `web/schemas.py` covering all JSON endpoints
- `response_model=` applied to all JSON endpoints across 7 routers (OpenAPI schema now fully typed)
- Shared `get_lora_info()` utility in `web/utils.py` for LoRA extraction
- Shared `formatUptime()` utility and `RestartWarning` component in frontend
- `LoRAFile`, `LoRAListResponse`, `ClearCacheResponse`, `PresetsResponse` types in `types.ts`

### changed
- All API responses now serialize as camelCase (e.g., `uptimeSeconds` instead of `uptime_seconds`)
- Frontend `client.ts` simplified: eliminated all manual snake-to-camel mapping functions (~60 lines removed)
- `fetchGenerationContext()`, `fetchVRAMStatus()`, `fetchModelStatus()`, `fetchPresets()`, `clearCache()` now direct typed passthrough
- `VRAMStatus` interface updated: `usedMB` -> `usedMb` (matches Pydantic `to_camel` output)
- `ModelStatusResponse` interface expanded to match full backend schema
- `LoRAInfo.layers_updated` -> `layersUpdated` across frontend
- `ModelStatusResponse.loras` field in schemas.py changed from `List[Dict]` to `List[LoRAInfo]`
- Consolidated 3 `if compile_enabled:` blocks in `vram.py` into single block
- Duplicate `LoRAFile`/`LoRAListResponse` types removed from `client.ts` (now in `types.ts` only)

### removed
- 2 stub endpoints: `GET /api/configs/available`, `POST /api/configs/load`
- 10 legacy load/unload routes from `vram.py` (superseded by unified `/api/models/{id}/load|unload`)
- 4 overlapping status endpoints merged into `/api/context`: `GET /api/system/status`, `GET /api/server/status`, `GET /api/generation-config`, `GET /api/rewriter-models`
- `web/static/` (dead v1 frontend) and `web/archive-frontend/` directories deleted
- Duplicate `_get_lora_info()` from `system.py` (replaced by shared `web/utils.get_lora_info()`)

## 0.8.7

### added
- `GET /api/context` endpoint: composite status aggregating model variant, LoRA fusion state, VRAM, quantization, compile, and session state
- `LoRAInfo` and `GenerationContextResponse` Pydantic models in `web/schemas.py`
- StatusBar component: persistent compact strip showing loaded model, LoRA badges, quant badge, VRAM bar with expand/collapse
- SettingsMenu component: server restart (with confirmation dialog), clear CUDA cache, system info, pending restart warnings
- ConfirmDialog reusable component for destructive actions
- Gear icon in LeftNav header for settings access on desktop
- Generation context polling (15s interval) in App.tsx
- ModelManager cards enriched with model variant name, LoRA badges, and config tags

### changed
- LoRA slider UX: added stepper buttons (44px touch targets with long-press acceleration), preset pills (0.25/0.50/0.75/1.00), slider moved to desktop-only secondary control
- `get_model_status()` in vram.py now returns `model_variant`, `loras`, `lora_summary` fields
- Model load/unload actions now refresh generation context
- VRAM poll augmented with composite context poll (15s interval)

## 0.8.6

### fixed
- LoRA post-fusion OOM: re-quantizes affected layers to fp8 after LoRA merge, reclaiming ~8GB VRAM on persistent models
- LoRA spec format mismatch: filters out empty-path LoRA entries from frontend before comparison
- FLUX.2 block_offload default: schema default changed from `true` to `false` to match `config.toml`

### added
- `log_prompts` config option: toggle prompt text logging (default: true) via `[logging]` section
- `log_generation_params` now actually gated in generation routers (was defined but never checked)
- HTTPS support for frontend-v2 dev server via `VITE_BACKEND_URL`, `VITE_SSL_CERT`, `VITE_SSL_KEY` env vars
- `.env.example` in `web/frontend-v2/` documenting HTTPS env vars

### removed
- Qwen-Image-Layered pipeline: all decomposition code, endpoints, schemas, tests, and docs deleted (~15 files modified/removed)
- `/api/qwen-image/decompose` endpoint
- `/api/qwen-image/status` and `/api/qwen-image/config` endpoints
- `QwenImagePipeline` (legacy pure-Python layered pipeline)
- `QwenImageDecomposeRequest` schema
- `qwenimage-layered` model type from CLI, config, and model manager

### changed
- README.md: updated pipeline table, added HTTPS setup section, removed Qwen-Image-Layered docs link
- CLAUDE.md: updated to v0.8.6, added DRY Configuration Principles section, HTTPS nav link

## 0.8.5

### fixed
- LoRA re-fusion OOM on persistent models: second request no longer dequantizes fp8 (9GB) to bf16 (18GB) again, preventing 26GB OOM when encoder shuttles to GPU
- `_infer_model_device_dtype` returns bfloat16 (compute dtype) instead of uint8/float8 (storage dtype) for quantized models, so LoRA math happens in correct precision
- Z-Image `load_lora()` no longer passes raw storage dtype to LoRA loader (delegates to `_infer_model_device_dtype`)

### added
- HTTPS support via `ssl_certfile` / `ssl_keyfile` config fields and CLI args (uvicorn-native TLS)
- Optional `ssl_ca_certs` for mutual TLS client certificate verification
- `FusedLoRAState` / `LoRAFusionRecord` dataclasses for pipeline-agnostic LoRA fusion tracking
- `get_fused_state(model)` attaches tracking state to any `nn.Module` -- works regardless of how the pipeline stores the model
- LoRA fusion guard in `flux2_generate.py`: skips re-fusion when LoRAs already match, raises `RuntimeError` on mismatch
- `_ensure_correct_model()` now checks LoRA specs in addition to model name; auto-reloads on LoRA mismatch
- HTTP 409 response for LoRA mismatch errors in FLUX.2 endpoints

## 0.8.4

### fixed
- FLUX.2 model switching: frontend model dropdown now actually triggers model reload instead of silently using whatever was loaded at startup
- LoRA crash on fp8-quantized models: `Float8Tensor + Tensor` now dequantizes before merge instead of hitting unimplemented `aten.add`
- VRAM race between generate and unload: mid-request unload returns 503 "model was unloaded" instead of OOM cascade

### added
- `ModelManager.reload_flux2(model_name)` for model-switching with proper lock coordination
- Loaded vs requested model_name logging in FLUX.2 generate endpoints

## 0.8.3

### added
- Preset card browser: horizontal scroll strips with visual cards replace the old `<select>` dropdown
- Active preset indicator with three states: none / active (checkmark) / modified (warning + Restore)
- Preset modification detection: compares active preset's original params against current resolved values
- `clearPreset()` and `restorePreset()` actions in formStore

### changed
- `applyPreset()` signature expanded to `(pipelineId, presetName, params)` -- records active preset and clears `userModified` for preset-touched params (synergy fix with dependent_defaults)
- appStore `loadPresets()` updated to use new `applyPreset` signature
- PipelineForm now renders `<PresetBrowser>` instead of a `<select>` dropdown

### fixed
- Preset + dependent_defaults synergy bug: applying a preset then switching models could leave stale preset values because `userModified` incorrectly blocked dependent_defaults updates

## 0.8.2

### added
- `compile_dynamic` config field for FLUX.2: `torch.compile(dynamic=True)` eliminates ~90s recompilation when resolution changes
- `dependent_defaults` on ParamSchema: schema-driven system for auto-updating form values when a trigger param changes (e.g., switching FLUX.2 model updates steps/guidance)
- FLUX.2 presets: `presets/flux2/distilled_fast.md` and `presets/flux2/base_quality.md`
- NumberInput auto-snap on blur: misaligned values (e.g., 1000 for a step=16 field) auto-correct when focus leaves the input
- `tests/unit/test_resolution_validators.py`: 51 tests for Pydantic and dataclass resolution snapping
- `snapToStep()` shared utility in `web/frontend-v2/src/utils/numbers.ts`
- `userModified` tracking in formStore for smart dependent default application
- Dynamic shapes section in `docs/guides/compile_and_resolution.md`

### changed
- Slider `commitInputValue()` refactored to use shared `snapToStep()` utility
- `getResolvedValues()` now layers dependent defaults between schema and server defaults
- PipelineForm `handleChange` triggers `applyDependentDefaults()` when a trigger param changes
- torch.compile uses `fullgraph=False` when `compile_dynamic=true` (safety for data-dependent branches)

## 0.8.1

### added
- `docs/guides/compile_and_resolution.md` -- comprehensive torch.compile and resolution guide with ROI math, compatibility matrix, VRAM budgets, and RTX 4090 config recommendations
- Resolution validation: Pydantic `@field_validator` snaps width/height to nearest VAE multiple (16 for FLUX.2, 32 for LTX-2) with `Field()` min/max constraints
- Defense-in-depth: `Flux2GenerationConfig.__post_init__` snaps invalid resolutions with warning
- Compile-aware logging in FLUX.2 router (latent token count, warmup notice)
- `compile_enabled` and `compile_vae_enabled` fields in `/api/flux2/status` response
- Compile warmup warning in pipeline config metadata (shown in model manager UI)
- Frontend: `dimension_preset` dropdown now drives width/height values (one-way sync)
- Frontend: per-image dimension display for multi-reference uploads in ImageUpload
- Frontend: step alignment validation warns when values are not multiples of step size

### changed
- torch.compile calls now use `fullgraph=True` to catch graph breaks at compile time
- FLUX.2 default config: `compile = false`, `compile_vae = false` (better default for 4-step distilled model)
- FLUX.2 schema: width/height step changed from 64 to 16, min from 512 to 256
- Dimension presets list now includes "Custom" option

## 0.8.0

### added
- 7 domain routers in `web/routers/`: core (Z-Image), flux2, ltx2, qwen_image, vram, config_mgmt, system
- `web/schemas.py`: all Pydantic request/response models extracted from server.py
- `web/utils.py`: shared helpers (output dirs, image saving, config merging)
- `web/dependencies.py`: FastAPI `Depends()` for `ConfigDep` and `ManagerDep`

### removed
- 5 dead load functions from server.py: `load_pipeline()`, `load_encoder_only()`, `load_api_encoder()`, `load_hybrid_pipeline()`, `load_api_pipeline()` (replaced by ModelManager)
- 17 unused backward-compat `@property` shims from RuntimeConfig: all `wan_*`, `flux2_quantization`, `flux2_encoder_device`, `fmtt_scale`
- Broken integration test fixtures (`client_with_model`, `client_with_pipeline`) and their test classes (`TestEncodeEndpoint`, `TestGenerateEndpoint`)

### changed
- `web/server.py` decomposed from 5744 lines to 465 lines (globals, unload functions, startup logic)
- All 68 API endpoints moved from monolithic server.py into 7 domain-specific router files
- Router registration deferred to `_register_routers()` in `main()` to avoid circular imports with `web.server as srv`

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
