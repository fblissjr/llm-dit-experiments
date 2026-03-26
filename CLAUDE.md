# agent context (v0.9.28)

*last updated: 2026-03-09*

Quick reference for LLM agents. This is a hobbyist exploration platform -- not a product. The codebase evolves to support whatever we're curious about next.

## BEFORE: rules + onboarding

**Hardware:** RTX 4090, 24GB VRAM. Tests requiring GPU should always work.

**Core principle:** ALWAYS rely on retrieval and search over assumptions. Before implementing anything: (1) check if config fields exist and are accurate, (2) check the schema, (3) trace the full data flow from entry point to execution, (4) find similar implementations. Never assume -- always verify.

### critical rules

- **no emojis** in code, docs, or output
- **use `uv`** for all Python ops (`uv add`, `uv run`, `uv sync`)
- **never commit** without explicit user approval
- **use `ModelManager`** for all model load/unload/reload -- never manipulate model globals directly
- **always update state** after significant work (see AFTER section below)
- **use `bun`** for all frontend ops (`bun install`, `bun run`, `bunx`) -- never `npm` or `yarn`
- **always update config.toml AND config.toml.example** when adding or changing any pipeline config parameter -- for all pipelines, not just the one you're working on

### configuration rules

- `config.toml` is source of truth. Never auto-detect when a config value exists.
- Composed sub-configs in `src/llm_dit/config.py`: access via `config.flux2.model_path` (not flat `config.flux2_model_path`). RuntimeConfig is in config.py (cli.py re-exports it).
- Adding a new config parameter: only **2 touchpoints** (dataclass field + config.toml). Validate with `tests/unit/test_dry_config.py`. Check existing fields BEFORE adding new ones.
- Config type consistency: dataclass type, config.toml value type, and downstream consumers must agree. E.g., `fps: float = 24.0` everywhere, not `int` in config and `float` in pipeline.
- All routers use `resolve_param()` from `web/param_resolver.py`. Precedence: client-sent > config.toml > schema default. Uses `model_fields_set` (NOT `or`, NOT `is not None`). Use `skip_none=True` for Optional fields where None = "no override". Falsy values (0, 0.0, "") are preserved when explicitly sent.
- Infrastructure params (model paths, devices, quantization) always from config -- never through `resolve_param`.
- CLI overrides for encoder config: `--hidden-layer` (int, default -2) selects Gemma3 hidden layer for feature extraction.
- Details: [post_refactor_guide.md](internal/docs/architecture/post_refactor_guide.md)

### onboarding (3 steps)

| Step | File | Purpose |
|------|------|---------|
| 1 | This file | Critical rules, quick reference |
| 2 | [internal/state/current.md](internal/state/current.md) | What's happening now, active pipelines, versions |
| 3 | Domain docs (see navigation below) | Based on your task |
| -- | [internal/log/](internal/log/) | Recent session logs (most recent `log_YYYY-MM-DD.md`) |

**Optional:** [VISION.md](VISION.md) for architecture philosophy (L1-L6 composability hierarchy).

## DURING: navigate + build

### navigation by task

| Task | Read |
|------|------|
| **Adding feature to existing pipeline** | [feature_workflow.md](internal/docs/feature_workflow.md) + [post_refactor_guide.md](internal/docs/architecture/post_refactor_guide.md) |
| **Adding new pipeline** | [pipeline_integration.md](internal/checklists/pipeline_integration.md) |
| **Writing/running tests** | [tests/CLAUDE.md](tests/CLAUDE.md) |
| **Research/experiments** | [experiments/CLAUDE.md](experiments/CLAUDE.md) |
| **Post-refactor architecture** | [post_refactor_guide.md](internal/docs/architecture/post_refactor_guide.md) |
| **Composability analysis** | [composability_analysis.md](internal/docs/architecture/composability_analysis.md) |
| **Architecture decisions** | [architectural_decisions.md](internal/principles/architectural_decisions.md) |
| **Planned improvements / tech debt** | [backlog.md](internal/state/backlog.md) |
| **Debugging** | [debugging_reference.md](internal/docs/debugging_reference.md) + [lessons_learned.md](internal/state/lessons_learned.md) |
| **Agent workflows** | [claude_workflow.md](internal/principles/claude_workflow.md) |
| **Quantization** | [quantization.md](docs/reference/quantization.md) |
| **LTX-2 two-stage / distilled pipeline** | [ltx2_distilled_pipeline.md](docs/reference/ltx2_distilled_pipeline.md) |
| **Codebase map** | [codebase_map.md](internal/docs/architecture/codebase_map.md) |
| **Logging standards** | [logging_standards.md](internal/principles/logging_standards.md) |
| **Modular architecture (L1-L6)** | [modular_architecture.md](internal/principles/modular_architecture.md) |
| **API endpoints / OpenAPI** | `scripts/export_openapi.py` or `bun run export-openapi && bun run gen-api` from `web/frontend-v2/` |
| **E2E testing standard** | [tests/e2e/api/README.md](tests/e2e/api/README.md) |
| **Frontend / IndexedDB** | [internal/web/CLAUDE.md](internal/web/CLAUDE.md) |

### multi-model awareness

This is a multi-workstream project. Encoders and core infra are shared across pipelines -- check which pipelines use a component before modifying it. Pipelines evolve independently. See [current.md](internal/state/current.md) for active pipelines.

### shared code locations

| Type | Location | Notes |
|------|----------|-------|
| Encoders | `src/llm_dit/encoders/` | One encoder may serve multiple pipelines |
| Attention | `src/llm_dit/utils/attention.py` | All pipelines use this |
| Quantization | `src/llm_dit/quantization/` | All pipelines use `quantize_component()` (sole entry point) |
| Layerwise fp8 | `src/llm_dit/quantization/layerwise_fp8.py` | Pure PyTorch fp8 hooks (no torchao); used by Gemma3 encoder |
| Encoder variants | `src/llm_dit/encoders/gemma3_variants.py` | Gemma3 variant factory: bf16, fp8, 8bit, q4-qat |
| LoRA | `src/llm_dit/utils/lora.py` | Pipeline-agnostic: `load_lora()`, `FusedLoRAState` tracking, `fuse_lora_to_state_dict()` with scale-aware fp8 fusion |
| Prompt rewriting | `src/llm_dit/utils/prompt_rewriter.py` | `PromptRewriter` (Qwen-Image), `Flux2PromptUpsampler` (FLUX.2) |
| Meta init | `src/llm_dit/utils/meta_init.py` | Zero-memory model construction; use with `load_state_dict(assign=True)` |
| Pinned shuttle | `src/llm_dit/utils/shuttle.py` | `PinnedShuttleMixin` -- pinned-memory CPU<->GPU shuttle for AutoEncoder, Qwen3, Gemma3 |
| Audio VAE | `src/llm_dit/models/ltx2/audio_vae/` | AudioDecoder (latents to mel), HiFiGAN Vocoder (mel to 24kHz waveform), AudioPatchifier |
| AV Blocks | `src/llm_dit/models/ltx2/av_block.py` | `BasicAVTransformerBlock` -- video-only, audio-only, or dual-stream with cross-modal attention |
| Param resolution | `web/param_resolver.py` | `resolve_param()` -- all routers use for generation param defaults |
| Model lifecycle | `src/llm_dit/model_manager.py` | `ModelManager` -- load/unload/reload any pipeline |
| API layer | `web/routers/`, `web/schemas.py` | 7 domain routers + Pydantic models (~560 lines) |
| Pipelines | `src/llm_dit/pipelines/` | Each pipeline has its own file |
| Frontend | `web/frontend-v2/` | React 19 + Zustand 5 + Vite 7 + Tailwind 4 + Bun. Schema-driven forms from OpenAPI. See [web CLAUDE.md](internal/web/CLAUDE.md) |
| Frontend logger | `web/frontend-v2/src/utils/logger.ts` | Namespaced logging factory; `VITE_LOG_LEVEL` env var; zero raw console calls |
| Media utilities | `web/frontend-v2/src/utils/media.ts` | `detectKind()`, `mediaItemFromResult()`, `mediaItemFromHistory()` -- unified `MediaItem` type |
| VRAM bar | `web/frontend-v2/src/components/common/VRAMBar.tsx` | Shared VRAM usage bar component used by StatusBar and SettingsMenu |
| CLI-over-API | `scripts/gen.py` | Thin httpx client (~440 lines): `flux2`, `zimage`, `ltx2`, `qwen`, `status` subcommands. Tests: `tests/unit/test_gen_cli.py` (52 tests) |
| Memory cleanup | `src/llm_dit/utils/memory.py` | `cleanup_memory()` -- centralized gc.collect + torch.cuda.empty_cache (CUDA guard) |
| Quant aliases | `src/llm_dit/quantization/__init__.py` | `QUANT_ALIASES` dict -- canonical `"fp8"` -> `"fp8-dynamic"` mapping (single source of truth) |
| FP8 forward | `src/llm_dit/quantization/fp8_cast.py` | `amend_forward_with_upcast()` -- dual-path: `torch._scaled_mm` (SM89+, 2x faster) or bf16 upcast fallback |
| FLUX.2 scheduler | `src/llm_dit/schedulers/flux2_scheduler.py` | `get_schedule()`, `compute_empirical_mu()`, `generalized_time_snr_shift()` |
| V2 Feature Extractor | `src/llm_dit/encoders/gemma3_feature_extractor_v2.py` | Per-token RMSNorm, dual projections (video 4096, audio 2048) |

### feature workflow

Trace the full data flow before implementing any feature. 4 steps: identify touchpoints, trace entry-to-execution, find similar implementations, verify baseline. Full guide: [feature_workflow.md](internal/docs/feature_workflow.md)

### quick test commands

```bash
uv run pytest tests/unit/ -v                    # Unit tests (no GPU, fast)
uv run pytest tests/unit/test_dry_config.py -v   # Config DRY validation
uv run pytest tests/e2e/api/test_ltx2_smoke.py -v -s  # E2E smoke (GPU)
```

Full testing guide with all commands: [tests/CLAUDE.md](tests/CLAUDE.md)

### model quickstarts

| Model | Quickstart | Full Reference |
|-------|------------|----------------|
| **FLUX.2 Klein** | [quickstart](internal/docs/flux2-klein/quickstart.md) | [comprehensive ref](internal/docs/flux2-klein/flux2_klein_comprehensive_reference.md) |
| **LTX-2** | [quickstart](internal/docs/ltx2/quickstart.md) | [comprehensive ref](internal/docs/ltx2/ltx2_comprehensive_reference.md) |
| **Z-Image** | [quickstart](internal/docs/z_image/quickstart.md) | [guides](internal/guides/) |

### top-5 failure modes

| Symptom | Likely Cause | Check |
|---------|--------------|-------|
| OOM on generation | Component not offloaded | Check device placement in model_manager.py |
| Config not applied | Hot-reload vs restart | See `HOT_RELOAD_SAFE` in model_manager.py |
| config.toml defaults ignored | Missing `resolve_param()` | Verify router uses `resolve_param()` from `web/param_resolver.py` |
| Circular import | Router imports server at module level | See architecture pattern in [post_refactor_guide.md](internal/docs/architecture/post_refactor_guide.md) |
| fp8 weights silently become bf16 | `load_state_dict` missing `assign=True` | Always use `assign=True` for mixed-dtype models |
| `_scaled_mm` CUBLAS error | Wrong scale shapes or matrix layout | B needs `stride(0)==1` (`.T` view). Trailing dim divisible by 16. Scalar scales for TensorWise |
| Washed-out/noisy distilled output | LoRA fused without fp8 weight_scale | `fuse_lora_to_state_dict` must dequant with `weight_scale` before adding LoRA delta; see `_fuse_delta()` in lora.py |
| Stage 1 washed out with few steps | Wrong two-stage pipeline mode | We use TI2VidTwoStagesPipeline (base+LoRA), NOT DistilledPipeline. Stage 1 needs 30 steps + full CFG, not 8 steps. See [ltx2_distilled_pipeline.md](docs/reference/ltx2_distilled_pipeline.md) |

Full debugging table: [debugging_reference.md](internal/docs/debugging_reference.md)

## AFTER: update state

| File | When to Update |
|------|----------------|
| `internal/state/current.md` | Major milestone or blocker |
| `internal/state/backlog.md` | New improvement identified, or item completed/deprioritized |
| `internal/state/lessons_learned.md` | After debugging |
| `internal/log/log_YYYY-MM-DD.md` | Every session |

## deep dives

Content moved from this file to keep it compact. Search for the section heading in each file.

| Topic | File | Section to find |
|-------|------|-----------------|
| Architecture patterns (14 bullets) | [post_refactor_guide.md](internal/docs/architecture/post_refactor_guide.md) | `implementation patterns` |
| Request lifecycle (startup + API flow) | [post_refactor_guide.md](internal/docs/architecture/post_refactor_guide.md) | `request lifecycle` |
| Feature implementation workflow | [feature_workflow.md](internal/docs/feature_workflow.md) | (entire file) |
| Full debugging table (31 rows) | [debugging_reference.md](internal/docs/debugging_reference.md) | (entire file) |
| Full test commands | [tests/CLAUDE.md](tests/CLAUDE.md) | (entire file) |
| IndexedDB conventions | [internal/web/CLAUDE.md](internal/web/CLAUDE.md) | `IndexedDB conventions` |
| LTX-2 two-stage / distilled reference | [ltx2_distilled_pipeline.md](docs/reference/ltx2_distilled_pipeline.md) | (entire file) |
