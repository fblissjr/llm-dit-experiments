# agent context (v0.9.20)

*last updated: 2026-03-06*

Quick reference for LLM agents. Read only what you need.

**This is a hobbyist exploration platform** - not a product with a finish line. New models, experiments, and features are continuously added. The codebase evolves to support whatever we're curious about next.

## hardware

This machine has an **RTX 4090 with 24GB VRAM**. Tests requiring GPU should always work.

## core principle

**ALWAYS rely on retrieval and search over assumptions.**

Before implementing anything, take a retrieval based approach rather than making assumptions.
1. Check if config fields and values are exist and/or are accurate
2. Check the schema for a given module
3. Trace the full data flow from entry point to execution
4. Find similar implementations in other pipelines

Never assume you know where code is or what exists. Always verify by retrieval and search.

## onboarding (3 steps)

| Step | File | Purpose |
|------|------|---------|
| 1 | This file | Critical rules, quick reference |
| 2 | [internal/state/current.md](internal/state/current.md) | What's happening now, active pipelines, versions |
| 3 | Domain docs (see navigation below) | Based on your task |
| -- | [internal/log/](internal/log/) | Recent session logs (most recent `log_YYYY-MM-DD.md`) |

**Optional:** [VISION.md](VISION.md) for architecture philosophy (L1-L6 composability hierarchy).

## critical rules

- **no emojis** in code, docs, or output
- **use `uv`** for all Python ops (`uv add`, `uv run`, `uv sync`)
- **never commit** without explicit user approval
- **use `ModelManager`** for all model load/unload/reload -- never manipulate model globals directly
- **always update state** after significant work (see state management below)
- **use `bun`** for all frontend ops (`bun install`, `bun run`, `bunx`) -- never `npm` or `yarn`
- **always update config.toml AND config.toml.example** when adding or changing any pipeline config parameter -- for all pipelines, not just the one you're working on

### IndexedDB conventions (frontend)

The frontend uses IndexedDB for persistence (3 stores: `llm-dit-history`, `llm-dit-app`, `llm-dit-form`). Rules:
- **Never write migration scripts.** IndexedDB is a cache, not a database. Always provide a nuke path.
- **Strip base64 data URLs** from persisted history params and form values in Zustand `partialize`. Large data URLs exhaust IndexedDB quota.
- **"Reset Storage" button** in SettingsMenu wipes all IndexedDB stores and reloads. Use `ConfirmDialog` for destructive actions (clear history, reset storage).
- **New persisted fields must be optional** with sensible defaults so existing stores hydrate without errors.

### configuration hierarchy

Config values ALWAYS win. Code should:
1. Read from config.toml as source of truth
2. Allow explicit parameter overrides when needed
3. Never auto-detect when a config value exists

**Config architecture:** Composed sub-configs in `src/llm_dit/config.py`:
- Pipeline configs: `LTX2Config`, `Flux2Config`, `ZImageConfig`, `QwenImageConfig`
- Shared configs: `EncoderConfig`, `OptimizationConfig`, `PipelineQuantConfig`
- `RuntimeConfig` composes these: access via `config.flux2.model_path` (not flat `config.flux2_model_path`)
- **RuntimeConfig is in config.py** (cli.py re-exports it)
- **Check these BEFORE adding new fields** - they may already exist!
- Adding a new config parameter: only **2 touchpoints** (dataclass field + config.toml). Validate with `tests/unit/test_dry_config.py`.
- Key CLI flags: `--hidden-layer` (encoder hidden state extraction, default -2), `--model-type`, `--config`

### DRY Configuration Principles

Every parameter should have exactly one source of truth. The chain is:
`config.toml` -> `Config` dataclasses -> `RuntimeConfig` (composed) -> backend configs

When adding a new parameter, only 2 files need changes: the dataclass in `config.py` + `config.toml`. The DRY consistency test (`test_dry_config.py`) validates that all layers stay in sync.

### parameter resolution (routers)

All routers use `resolve_param()` from `web/param_resolver.py` for generation parameters:
- **Precedence:** client-sent > config.toml > schema default
- Uses Pydantic v2 `model_fields_set` to detect explicit client values (NOT `or`, NOT `is not None`, NOT equality comparison)
- `skip_none=True` for Optional fields where None means "no override" (e.g., `stage1_steps`, `distilled_lora_path`)
- Falsy values (0, 0.0, "") are preserved when client sends them explicitly
- **Infrastructure params** (model paths, devices, quantization) always come from config -- do NOT use `resolve_param` for them
- Schema-config field name mismatches exist (e.g., `stage1_steps` vs `stage1_num_inference_steps`). `test_dry_config.py::TestResolveParamFieldConsistency` validates these.

## navigation by task

| Task | Read |
|------|------|
| **Adding feature to existing pipeline** | Feature workflow (below) + [post_refactor_guide.md](internal/docs/architecture/post_refactor_guide.md) |
| **Adding new pipeline** | [pipeline_integration.md](internal/checklists/pipeline_integration.md) |
| **Writing/running tests** | [tests/CLAUDE.md](tests/CLAUDE.md) |
| **Research/experiments** | [experiments/CLAUDE.md](experiments/CLAUDE.md) |
| **Post-refactor architecture** | [post_refactor_guide.md](internal/docs/architecture/post_refactor_guide.md) |
| **Composability analysis** | [composability_analysis.md](internal/docs/architecture/composability_analysis.md) |
| **Architecture decisions** | [architectural_decisions.md](internal/principles/architectural_decisions.md) |
| **Planned improvements / tech debt** | [backlog.md](internal/state/backlog.md) |
| **Debugging** | [lessons_learned.md](internal/state/lessons_learned.md) |
| **Agent workflows** | [claude_workflow.md](internal/principles/claude_workflow.md) |
| **Quantization** | [quantization.md](docs/reference/quantization.md) |
| **Codebase map** | [codebase_map.md](internal/docs/architecture/codebase_map.md) |
| **Logging standards** | [logging_standards.md](internal/principles/logging_standards.md) |
| **Modular architecture (L1-L6)** | [modular_architecture.md](internal/principles/modular_architecture.md) |
| **API endpoints / OpenAPI** | `scripts/export_openapi.py` or `bun run export-openapi && bun run gen-api` from `web/frontend-v2/` |
| **E2E testing standard** | [tests/e2e/api/README.md](tests/e2e/api/README.md) |

## feature implementation workflow

**Before implementing ANY feature, trace the full data flow using search:**

### 1. identify all touchpoints
```bash
# Find where the feature name appears (it may already exist!)
grep -rn "<feature>" src/llm_dit/
grep -rn "<feature>" web/routers/ web/schemas.py

# Check config dataclass for existing fields
grep -A 50 "class <Pipeline>Config" src/llm_dit/config.py
```

### 2. trace entry point to execution
Every feature has a data flow chain. Trace it BEFORE coding:

| Layer | Location | What to grep |
|-------|----------|--------------|
| **API Request Model** | `web/schemas.py` | `class <Pipeline>GenerateRequest` (Pydantic) |
| **API Endpoint** | `web/routers/<pipeline>.py` | `@router.post("/api/<pipeline>/")` |
| **Config Defaults** | `src/llm_dit/config.py` | `class <Pipeline>Config` (dataclass) |
| **RuntimeConfig** | `src/llm_dit/config.py` | `class RuntimeConfig` (composed sub-configs) |
| **CLI (server)** | `src/llm_dit/cli.py` | Server startup CLI args |
| **CLI (gen)** | `scripts/gen.py` | CLI-over-API client (deprecated: `scripts/generate.py`) |
| **Model lifecycle** | `src/llm_dit/model_manager.py` | `ModelManager`, load/unload/reload |
| **Pipeline Function** | `src/llm_dit/pipelines/` | Main generation function |
| **Shared Utils** | `src/llm_dit/utils/` | LoRA, quantization, attention, memory |

**All layers that accept the parameter must be updated.** Don't just add to the pipeline function - check if the schema, router, and config also need the field.

### 3. find similar implementations
```bash
# Check if another pipeline already has this feature
grep -rn "<feature>" src/llm_dit/pipelines/
grep -rn "<feature>" web/routers/
ls src/llm_dit/utils/  # Check existing utilities
```

### 4. verify visual baseline before changes
For any code that affects generation output:
```bash
# Generate baseline BEFORE making changes
uv run pytest tests/integration/pipeline/test_<pipeline>_baselines.py -v -s
```

## multi-model platform

This is a **multi-workstream project**. Work happens in parallel across layers:

| Layer | What | Impact |
|-------|------|--------|
| **Pipelines** | Model-specific generation loops | Independent - each pipeline can evolve separately |
| **Encoders** | Text/vision encoding (Qwen3, Gemma3, etc.) | Shared - one encoder may serve multiple pipelines |
| **Core Infra** | Attention, quantization, memory, VAE ops | Universal - changes affect all pipelines |
| **UI/Server** | React frontend, FastAPI backend | Universal - serves all pipelines dynamically |

**Key insight:** Encoders and core infra are shared. Check which pipelines use a component before modifying it. See [current.md](internal/state/current.md) for active pipelines and parallel work.

### shared code locations

| Type | Location | Notes |
|------|----------|-------|
| Encoders | `src/llm_dit/encoders/` | One encoder may serve multiple pipelines |
| Attention | `src/llm_dit/utils/attention.py` | All pipelines use this |
| Quantization | `src/llm_dit/quantization/` | All pipelines use `quantize_component()` (sole entry point) |
| Layerwise fp8 | `src/llm_dit/quantization/layerwise_fp8.py` | Pure PyTorch fp8 hooks (no torchao); used by Gemma3 encoder |
| Encoder variants | `src/llm_dit/encoders/gemma3_variants.py` | Gemma3 variant factory: bf16, fp8, 8bit, q4-qat |
| LoRA | `src/llm_dit/utils/lora.py` | Pipeline-agnostic: `load_lora()`, `FusedLoRAState` tracking |
| Prompt rewriting | `src/llm_dit/utils/prompt_rewriter.py` | `PromptRewriter` (Qwen-Image), `Flux2PromptUpsampler` (FLUX.2) |
| Meta init | `src/llm_dit/utils/meta_init.py` | Zero-memory model construction; use with `load_state_dict(assign=True)` |
| Pinned shuttle | `src/llm_dit/utils/shuttle.py` | `PinnedShuttleMixin` -- pinned-memory CPU<->GPU shuttle for AutoEncoder, Qwen3, Gemma3 |
| Audio VAE | `src/llm_dit/models/ltx2/audio_vae/` | AudioDecoder (latents to mel), HiFiGAN Vocoder (mel to 24kHz waveform), AudioPatchifier |
| AV Blocks | `src/llm_dit/models/ltx2/av_block.py` | `BasicAVTransformerBlock` -- video-only, audio-only, or dual-stream with cross-modal attention |
| Param resolution | `web/param_resolver.py` | `resolve_param()` -- all routers use for generation param defaults |
| Model lifecycle | `src/llm_dit/model_manager.py` | `ModelManager` -- load/unload/reload any pipeline |
| API layer | `web/routers/`, `web/schemas.py` | 7 domain routers + Pydantic models (~730 lines) |
| Pipelines | `src/llm_dit/pipelines/` | Each pipeline has its own file |
| Frontend | `web/frontend-v2/` | React 19 + Zustand 5 + Vite 7 + Tailwind 4 + Bun. Schema-driven forms from OpenAPI. See [web CLAUDE.md](internal/web/CLAUDE.md) |
| Frontend logger | `web/frontend-v2/src/utils/logger.ts` | Namespaced logging factory; `VITE_LOG_LEVEL` env var; zero raw console calls |
| Media utilities | `web/frontend-v2/src/utils/media.ts` | `detectKind()`, `mediaItemFromResult()`, `mediaItemFromHistory()` -- unified `MediaItem` type |
| VRAM bar | `web/frontend-v2/src/components/common/VRAMBar.tsx` | Shared VRAM usage bar component used by StatusBar and SettingsMenu |
| CLI-over-API | `scripts/gen.py` | Thin httpx client: `flux2`, `zimage`, `ltx2`, `qwen`, `status` subcommands. Tests: `tests/unit/test_gen_cli.py` (52 tests) |
| Memory cleanup | `src/llm_dit/utils/memory.py` | `cleanup_memory()` -- centralized gc.collect + torch.cuda.empty_cache (CUDA guard) |
| Quant aliases | `src/llm_dit/quantization/__init__.py` | `QUANT_ALIASES` dict -- canonical `"fp8"` -> `"fp8-dynamic"` mapping (single source of truth) |
| FLUX.2 scheduler | `src/llm_dit/schedulers/flux2_scheduler.py` | `get_schedule()`, `compute_empirical_mu()`, `generalized_time_snr_shift()` |
| GGUF quantization | `src/llm_dit/quantization/gguf_*.py` | `GGMLLinear` (per-forward dequant), `GGMLTensor`, GGUF loader |
| V2 Feature Extractor | `src/llm_dit/encoders/gemma3_feature_extractor_v2.py` | Per-token RMSNorm, dual projections (video 4096, audio 2048) |
| GGUF audit | `scripts/audit_gguf_keys.py` | Validate GGUF keys against model state dict |

## architecture patterns (post-refactor)

**Circular import prevention:** Routers that need server state (generation_history, encoder_only_mode, rewriter_backend) do `import web.server as srv` at module level. Server imports routers in `_register_routers()` called from `main()`, NOT at module level. Never move router imports to top of server.py. Note: flux2.py and config_mgmt.py no longer import srv at all.

**Router decomposition:** server.py (~304 lines) holds only server state globals and startup. All ~49 API route handlers live in 7 domain routers under `web/routers/`. All routers use `ConfigDep`/`ManagerDep` dependency injection for pipeline access. No pipeline globals remain in server.py -- ModelManager is the sole source of truth.

**LoRA fusion tracking:** `model._fused_lora_state` (FusedLoRAState) tracks what's fused on persistent models. Prevents re-fusion OOM where fp8 (9GB) dequantizes to bf16 (18GB). Pipeline detects mismatch (raises RuntimeError), router handles recovery (reload).

**On-demand lazy loading:** When `default_pipeline = "none"`, routers trigger `manager.load("pipeline_id")` on first request. Method is `manager.load()` (NOT `load_pipeline()`). Idempotent -- double-check inside lock, returns early if already loaded. Unloads other pipelines to free VRAM.

**`gemma_variant` vs `quantize` are independent:** `quantize` controls transformer quantization (torchao fp8-dynamic at runtime). `gemma_variant` controls encoder loading strategy (pure PyTorch for fp8/fp8-safetensors, torchao for 8bit). The fp8-safetensors path has zero torchao dependency; torchao debug logs during load are transitive from transformers import.

**Unified quantization:** All pipelines use `quantize_component()` as the sole entry point. torchao is the only backend. `"fp8"` alias maps to `"fp8-dynamic"` (W8A8, FP8 tensor cores). Default granularity is `"per-row"`. Already-quantized weights (torchao subclasses or native FP8 dtypes) are auto-detected and skipped. See `docs/reference/quantization.md`.

**Prompt upsampling (FLUX.2):** `_upsample_prompt()` factory in `web/routers/flux2.py` reads URL + model from `RuntimeConfig.rewriter_api_url`/`rewriter_api_model` (sourced from `config.toml [rewriter]`). Creates `Flux2PromptUpsampler` which calls heylookitsanllm at `192.168.1.123:8080`. Two modes: T2I (creative expansion) and I2I (instruction compilation). Graceful fallback to original prompt on error. Used by both sync and streaming endpoints.

**FBCache (block skipping):** `LTX2Transformer` tracks residual norms between denoising steps. When `fbcache_threshold > 0`, blocks with small residual changes are skipped. Must call `model.reset_fbcache()` at the start of each generation. First/last steps always compute fully.

**Distilled sigma mode:** When `use_distilled_sigmas=True`, Stage 1 uses predefined `DISTILLED_SIGMA_VALUES` from `constants.py`. Forces `guidance_scale=1.0` (no CFG, no STG) -- guidance is baked into the distilled model weights.

**STG perturbation model:** `PerturbationType`, `PerturbationConfig`, and `BatchedPerturbationConfig` in `av_block.py` define per-sample attention skipping for Spatio-Temporal Guidance. `BasicAVTransformerBlock` uses these to selectively skip cross-modal attention (A2V/V2A) during denoising, enabling separate guidance scales for audio and video streams.

**Audio-video dual-stream:** `BasicAVTransformerBlock` extends `BasicTransformerBlock` with bidirectional cross-modal attention. Three modes: video-only (no audio latents), audio-only (no video latents), dual-stream (both with A2V + V2A cross-attention). Each modality has independent FBCache tracking. Audio connector uses 2048 dim vs video's 4096 dim.

**GGUF persistent model:** Unlike bf16 cache+reconstruct, GGUF models stay loaded persistently (`_ltx2_gguf_model` in ModelManager). LoRA is applied per-forward via `lora_delta`/`lora_scale` on `GGMLLinear` layers without mutating base weights. No cache/reconstruct cycle needed.

**V2 caption_projection identity:** V2 models use `nn.Identity()` for `caption_projection` because projection moved to the encoder (`FeatureExtractorV2`). V2 also adds `prompt_adaln_single` for cross-attention KV-side modulation, computed in `TransformerArgsPreprocessor.prepare()`.

**Frontend logging:** `web/frontend-v2/src/utils/logger.ts` provides namespaced console logging. Factory: `logger('API')` returns `{ debug, info, warn, error }` with `[API]` prefix. Log level controlled by `VITE_LOG_LEVEL` env var (defaults: `debug` in dev, `warn` in prod). All 22 console calls across 8 files migrated -- zero raw `console.*` calls remain outside logger.ts. Filter in DevTools by namespace (e.g., `[Generate]`, `[Model]`, `[Session]`).

For full post-refactor details: [post_refactor_guide.md](internal/docs/architecture/post_refactor_guide.md)

## request lifecycle (reference)

### startup flow

When you run `uv run web/server.py --config config.toml`:

```
server.py main()
    |
    v
cli.py: create_base_parser() - defines CLI args
    |
    v
cli.py: load_runtime_config(args)
    |
    v
config.py: Config.from_toml() - parses TOML into composed sub-configs
    |
    v
config.py: RuntimeConfig.from_toml_config(config) - composes sub-configs
    |
    v
server.py: stores as global `runtime_config`
    |
    v
model_manager.py: ModelManager(runtime_config) - manages model lifecycle
    |
    v
pipelines/*.py: ZImagePipeline, LTX2Pipeline, etc.
```

### api request flow

When client sends POST /api/flux2/generate:

```
HTTP Request (JSON body)
    |
    v
web/schemas.py: Pydantic model validates request (Flux2GenerateRequest, etc.)
    |
    v
web/routers/<pipeline>.py: resolve_param() merges request + config.toml defaults
    |
    v
pipelines/*.py: pipeline(prompt, **merged_params)
    |
    v
Model execution -> Response
```

### hot-reload vs restart

Some config changes apply immediately; others require server restart. See `HOT_RELOAD_SAFE` and `REQUIRES_RESTART` constants in `src/llm_dit/model_manager.py`. Config update API: `web/routers/config_mgmt.py` (PUT `/api/config/session`).

## debugging quick reference

For full debugging patterns, see [lessons_learned.md](internal/state/lessons_learned.md).

**Common failure modes:**

| Symptom | Likely Cause | Check |
|---------|--------------|-------|
| Silent wrong output | Tokenizer mismatch | Verify chat template matches training |
| OOM on generation | Component not offloaded | Check device placement in model_manager.py |
| OOM on LoRA re-fusion | LoRA dequantizing fp8->bf16 | Check `FusedLoRAState` in lora.py |
| Config not applied | Hot-reload vs restart | See `HOT_RELOAD_SAFE` in model_manager.py |
| config.toml generation defaults ignored | Missing `resolve_param()` | Verify router uses `resolve_param()` from `web/param_resolver.py` |
| Tests pass, bad output | Visual verification skipped | Always verify baselines visually |
| API returns error | Pydantic validation | Check request schema in `web/schemas.py` |
| Circular import | Router imports server at module level | See architecture patterns above |
| Prompt upsampling skipped | Empty `api_url` in config | Check `config.toml [rewriter].api_url` is set |
| Prompt upsampling silent fail | heylookitsanllm unreachable | Verify `192.168.1.123:8080` is live; check logs for `[FLUX2:Upsample] Failed` |
| Video thumbnails broken in history | SSE type assertion missing field | Check `eventData` type in `sessionStore.ts` (~line 170) -- uses `as unknown as` cast |
| Wrong transformer weights loaded | `transformer_file` config mismatch | Verify `[ltx2].transformer_file` points to correct safetensors; FP8 files use `load_ltx2_transformer_fp8_cast()` |
| fp8 weights silently become bf16 | `load_state_dict` missing `assign=True` | Mixed-dtype models MUST use `assign=True` to preserve fp8 dtype; without it, tensors are copied into existing bf16 params |
| LTX-2 encoder None in on-demand mode | `default_pipeline = "none"` skips preload | Router lazy-loads via `manager.load("ltx2")` on first request; check `manager.ltx2_encoder is None` guard |
| Model construction OOM spike | Missing `meta_init()` wrapper | Wrap `create_model_from_config()` in `meta_init()` context manager + use `assign=True` in `load_state_dict` |
| 4 pre-existing unit test failures | Not caused by recent changes | `test_resolution_validators.py` (2): snap_to_32 vs snap_to_64 mismatch. `test_pipeline.py` (1): unrelated. `test_config_consistency.py` (1): config field drift. Verify with `git stash && uv run pytest tests/unit/ -v` |
| Generate button disabled silently | Validation error not displayed to user | Check DevTools for `[Generate]` namespace logs; likely stale IndexedDB value exceeds schema min/max |
| Noisy/garbage LTX-2 output | Compounding optimizations | Disable `ge_gamma`, `fbcache_threshold`, `use_distilled_sigmas` in config.toml; test one at a time. All three together = no CFG + stale cached blocks + amplified velocity = runaway noise |
| Stale form values from IndexedDB | Schema range changed after values persisted | `getResolvedValues()` clamps automatically; use "Reset Storage" in Settings or clear IndexedDB via DevTools Application tab |
| Audio keys cause `load_state_dict` failure | Cache created with `audio_enabled=True` but reconstruction defaults to VideoOnly | Cache carries `video_only` flag; reconstruction uses `_reconstruct_transformer_from_cache()` helper |
| Falsy-zero in dict lookups | `d.get("a") or d.get("b")` skips 0, 0.0, "" | Use `if key in data` pattern; see `_get_camel()` in `scripts/gen.py` |
| GGUF model shows `available=false` | `gguf_transformer_path` not checked | Verify `[ltx2].gguf_transformer_path` in config.toml points to valid .gguf file |
| V2 cross-attention produces NaN/crash | `prompt_timestep` not populated | Fixed in v0.9.19: `TransformerArgsPreprocessor.prepare()` now computes `prompt_timestep` when `prompt_adaln_single` exists |
| LoRA silently not applied with GGUF | Key mismatch in `attach_lora_deltas` | Check logs for "0 of N delta keys matched" warning |

## quick test commands

```bash
# Unit tests (no GPU, fast)
uv run pytest tests/unit/ -v

# E2E API tests (requires GPU + models)
uv run pytest tests/e2e/api/test_flux2_smoke.py -v -s
uv run pytest tests/e2e/api/test_ltx2_smoke.py -v -s

# Pipeline integration smoke test (GPU required)
uv run pytest tests/integration/pipeline/test_baseline_portable.py::TestBaselineSmoke -v -s

# LoRA tests
uv run pytest tests/unit/ -v -k lora

# Config DRY validation
uv run pytest tests/unit/test_dry_config.py -v

# Parameter resolution logic
uv run pytest tests/unit/test_param_resolver.py -v

# Audio/AV tests (audio VAE + AV transformer blocks)
uv run pytest tests/unit/test_ltx2_audio_vae.py tests/unit/test_ltx2_av_transformer.py -v

# Quantization tests (alias, detection, recommended method)
uv run pytest tests/unit/test_ltx2_resolve_quantize.py tests/unit/test_quantization.py -v

# V2 architecture + GGUF pipeline integration
uv run pytest tests/unit/test_v2_architecture.py tests/unit/test_gguf_pipeline_integration.py -v

# GGUF E2E smoke test (GPU + GGUF checkpoint)
uv run pytest tests/e2e/api/test_ltx2_gguf_smoke.py -v -s

# CLI-over-API tool (gen.py arg parsing, body building, SSE handling)
uv run pytest tests/unit/test_gen_cli.py -v

# Frontend TypeScript check (from web/frontend-v2/)
cd web/frontend-v2 && bunx tsc --noEmit

# Frontend E2E tests (requires backend running, from web/frontend-v2/)
cd web/frontend-v2 && bun run test:e2e           # headless
cd web/frontend-v2 && bun run test:e2e:headed     # visible browser
cd web/frontend-v2 && bun run test:e2e:ui          # interactive UI

# Regenerate frontend types from API (from web/frontend-v2/)
cd web/frontend-v2 && bun run export-openapi && bun run gen-api
```

Full testing guide: [tests/CLAUDE.md](tests/CLAUDE.md)

## model quickstarts

| Model | Quickstart | Full Reference |
|-------|------------|----------------|
| **FLUX.2 Klein** | [quickstart](internal/docs/flux2-klein/quickstart.md) | [comprehensive ref](internal/docs/flux2-klein/flux2_klein_comprehensive_reference.md) |
| **LTX-2** | [quickstart](internal/docs/ltx2/quickstart.md) | [comprehensive ref](internal/docs/ltx2/ltx2_comprehensive_reference.md) |
| **Z-Image** | [quickstart](internal/docs/z_image/quickstart.md) | [guides](internal/guides/) |

## state management

| File | When to Update |
|------|----------------|
| `internal/state/current.md` | Major milestone or blocker |
| `internal/state/backlog.md` | New improvement identified, or item completed/deprioritized |
| `internal/state/lessons_learned.md` | After debugging |
| `internal/log/log_YYYY-MM-DD.md` | Every session |
