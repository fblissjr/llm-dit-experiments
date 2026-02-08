# agent context

*last updated: 2026-02-08*

Quick reference for LLM agents. Read only what you need.

**This is a hobbyist exploration platform** - not a product with a finish line. New models, experiments, and features are continuously added. The codebase evolves to support whatever we're curious about next.

## hardware

This machine has an **RTX 4090 with 24GB VRAM**. Tests requiring GPU should always work.

## core principle

**ALWAYS rely on retrieval and search over assumptions.**

Before implementing anything, USE grep/search to:
1. Check if config fields already exist
2. Trace the full data flow from entry point to execution
3. Find similar implementations in other pipelines

Never assume you know where code is or what exists. Always verify by reading.

## onboarding (3 steps)

| Step | File | Purpose |
|------|------|---------|
| 1 | This file | Critical rules, quick reference |
| 2 | [internal/state/current.md](internal/state/current.md) | What's happening now |
| 3 | Domain docs (see below) | Based on your task |

**Optional:** [VISION.md](VISION.md) for architecture philosophy, [spec.md](spec.md) for backlog.

## request lifecycle (end-to-end)

Understanding the full request flow is critical before modifying any layer.

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
web/routers/<pipeline>.py: domain router endpoint merges request + runtime_config
    |
    v
pipelines/*.py: pipeline(prompt, **merged_params)
    |
    v
Model execution -> Response
```

### key files in the chain

| Layer | File | What to grep |
|-------|------|--------------|
| **Entry point** | `web/server.py` | `main()`, globals, startup |
| **API routers** | `web/routers/*.py` | Domain endpoints (core, flux2, ltx2, qwen_image, vram, config_mgmt, system) |
| **Pydantic schemas** | `web/schemas.py` | `class <Pipeline>GenerateRequest`, response models |
| **FastAPI deps** | `web/dependencies.py` | `ConfigDep`, `ManagerDep` |
| **Shared helpers** | `web/utils.py` | Output dirs, image saving, config merging |
| **CLI parsing** | `src/llm_dit/cli.py` | `create_base_parser`, CLI overrides |
| **Config + RuntimeConfig** | `src/llm_dit/config.py` | `class <Pipeline>Config`, `RuntimeConfig`, `from_toml` |
| **Model lifecycle** | `src/llm_dit/model_manager.py` | `ModelManager`, load/unload/reload, `HOT_RELOAD_SAFE` |
| **Execution** | `src/llm_dit/pipelines/*.py` | `generate_*`, `__call__` |

### hot-reload vs restart

Some config changes apply immediately; others require server restart.

**Hot-reload safe** (change without restart) - see `HOT_RELOAD_SAFE` in `src/llm_dit/model_manager.py`:
- Scheduler params: `shift`, `d_noise`, `dynamic_shift`
- Generation defaults: `height`, `width`, `steps`, `guidance_scale`
- Cache settings, tiled VAE, seed

**Requires restart** (model reload) - see `REQUIRES_RESTART` in `src/llm_dit/model_manager.py`:
- Model paths: `model_path`, `text_encoder_path`
- Device placement: `encoder_device`, `dit_device`, `vae_device`
- Quantization: `quantization`, `dtype`
- Memory: `cpu_offload`
- Compilation: `compile`, `flash_attn`
- LoRA: `lora_paths`, `lora_scales`

Config update API is in `web/routers/config_mgmt.py` (hot-reload via PUT `/api/config/session`).

## critical rules

- **no emojis** in code, docs, or output
- **use `uv`** for all Python ops (`uv add`, `uv run`, `uv sync`)
- **never commit** without explicit user approval
- **always update state** after significant work

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

## DRY Configuration Principles

New config parameter touchpoints: **2** (dataclass field + config.toml). Was 6 pre-refactor.

Config flow (post-refactor):

1. **config.toml** (user-facing) -> **Config dataclass** (config.py) -> **RuntimeConfig.from_toml_config()** (config.py) -> **Backend/Pipeline** (usage)
2. **CLI flag** (cli.py) -> overrides on **RuntimeConfig** (config.py) -> **Backend/Pipeline** (usage)

RuntimeConfig composes sub-configs -- access via `config.flux2.model_path` not flat `config.flux2_model_path`.

Key CLI flags and their config paths:

| CLI Flag | RuntimeConfig Access | Config Dataclass |
|----------|---------------------|------------------|
| `--hidden-layer` | `config.encoder.hidden_layer` | `EncoderConfig.hidden_layer` |
| `--quantization` | `config.encoder.quantization` | `EncoderConfig.quantization` |
| `--compile` | `config.optimization.compile` | `OptimizationConfig.compile` |
| `--flash-attn` | `config.optimization.flash_attn` | `OptimizationConfig.flash_attn` |

When adding a new parameter, always check `tests/unit/test_dry_config.py` which validates config consistency automatically.

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
| **CLI** | `src/llm_dit/cli.py` | `<pipeline>` subcommand |
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
uv run pytest tests/e2e/test_<pipeline>_baselines.py -v -s
```

## multi-model platform

This is a **multi-workstream project**. Work happens in parallel across layers:

| Layer | What | Impact |
|-------|------|--------|
| **Pipelines** | Model-specific generation loops | Independent - each pipeline can evolve separately |
| **Encoders** | Text/vision encoding (Qwen3, Gemma3, etc.) | Shared - one encoder may serve multiple pipelines |
| **Core Infra** | Attention, quantization, memory, VAE ops | Universal - changes affect all pipelines |
| **UI/Server** | React frontend, FastAPI backend | Universal - serves all pipelines dynamically |

**Key insight:** Encoders and core infra are shared. Check which pipelines use a component before modifying it. See `current.md` for active parallel work.

### current pipelines

| Pipeline | Task | Encoder | Status |
|----------|------|---------|--------|
| FLUX.2 Klein | text-to-image, editing | Qwen3-8B/4B | Production |
| LTX-2 | text-to-video | Gemma3-12B | Production |
| Z-Image | text-to-image | Qwen3-4B | Production |
| Wan Video | text-to-video | UMT5-XXL | Phase 1 |

*This list grows as new models become interesting. Check `spec.md` for what's coming next.*

### shared code locations

| Type | Location | Notes |
|------|----------|-------|
| Encoders | `src/llm_dit/encoders/` | One encoder may serve multiple pipelines |
| Attention | `src/llm_dit/utils/attention.py` | All pipelines use this |
| Quantization | `src/llm_dit/quantization/` | All pipelines use `quantize_component()` (sole entry point) |
| LoRA | `src/llm_dit/utils/lora.py` | Pipeline-agnostic: `load_lora()`, `FusedLoRAState` tracking |
| Model lifecycle | `src/llm_dit/model_manager.py` | `ModelManager` -- load/unload/reload any pipeline |
| API layer | `web/routers/`, `web/schemas.py` | 7 domain routers + Pydantic models |
| Pipelines | `src/llm_dit/pipelines/` | Each pipeline has its own file |

**Architecture:** See [VISION.md](VISION.md) for L1-L6 composability hierarchy.

## model quickstarts

| Model | Quickstart | Full Reference |
|-------|------------|----------------|
| **FLUX.2 Klein** | [quickstart](internal/docs/flux2-klein/quickstart.md) | [comprehensive ref](internal/docs/flux2-klein/flux2_klein_comprehensive_reference.md) |
| **LTX-2** | [quickstart](internal/docs/ltx2/quickstart.md) | [comprehensive ref](internal/docs/ltx2/ltx2_comprehensive_reference.md) |
| **Z-Image** | [quickstart](internal/docs/z_image/quickstart.md) | [guides](internal/guides/) |

## state management

| File | When to Update |
|------|----------------|
| `current.md` | Major milestone or blocker |
| `todos.md` | Session start/end |
| `spec.md` | Backlog item complete |
| `lessons_learned.md` | After debugging |
| `log_YYYY-MM-DD.md` | Every session |

## debugging quick reference

For full debugging patterns, see [lessons_learned.md](internal/state/lessons_learned.md).

**Common failure modes:**

| Symptom | Likely Cause | Check |
|---------|--------------|-------|
| Silent wrong output | Tokenizer mismatch | Verify chat template matches training |
| OOM on generation | Component not offloaded | Check device placement in model_manager.py |
| OOM on LoRA re-fusion | LoRA dequantizing fp8->bf16 | Check `FusedLoRAState` in lora.py |
| Config not applied | Hot-reload vs restart | See hot-reload table above |
| Tests pass, bad output | Visual verification skipped | Always verify baselines visually |
| API returns error | Pydantic validation | Check request schema in `web/schemas.py` |
| Circular import | Router imports server at module level | See "circular import" note below |

## architecture patterns (post-refactor)

**Circular import prevention:** Routers do `import web.server as srv` at module level. Server imports routers in `_register_routers()` called from `main()`, NOT at module level. Never move router imports to top of server.py.

**Router decomposition:** server.py (491 lines) holds only globals, unload functions, and startup. All 68 API endpoints live in 7 domain routers under `web/routers/`. Routers access server globals via `srv.runtime_config`, `srv.flux2_pipeline`, etc.

**LoRA fusion tracking:** `model._fused_lora_state` (FusedLoRAState) tracks what's fused on persistent models. Prevents re-fusion OOM where fp8 (9GB) dequantizes to bf16 (18GB). Pipeline detects mismatch (raises RuntimeError), router handles recovery (reload).

**Unified quantization:** All pipelines use `quantize_component()` as the sole entry point. torchao is the only backend. See `docs/reference/quantization.md`.

## navigation by task

| Task | Read |
|------|------|
| **Adding feature to existing pipeline** | **See "feature implementation workflow" above** |
| Web/UI development | [internal/web/CLAUDE.md](internal/web/CLAUDE.md) |
| Writing/running tests | [tests/CLAUDE.md](tests/CLAUDE.md) |
| Research/experiments | [experiments/CLAUDE.md](experiments/CLAUDE.md) |
| Post-refactor architecture | [internal/docs/architecture/post_refactor_guide.md](internal/docs/architecture/post_refactor_guide.md) |
| Architecture decisions | [internal/principles/architectural_decisions.md](internal/principles/architectural_decisions.md) |
| Agent workflows | [internal/principles/claude_workflow.md](internal/principles/claude_workflow.md) |
| Debugging | [internal/state/lessons_learned.md](internal/state/lessons_learned.md) |
| Adding new pipeline | [internal/checklists/pipeline_integration.md](internal/checklists/pipeline_integration.md) |
| Quantization reference | [docs/reference/quantization.md](docs/reference/quantization.md) |

## quick test commands

```bash
# Smoke test (GPU required)
uv run pytest tests/e2e/test_baseline_portable.py::TestBaselineSmoke -v -s

# Unit tests (no GPU)
uv run pytest tests/unit/ -v

# All LTX-2 tests
uv run pytest tests/ -v -k ltx2
```

Full testing guide: [tests/CLAUDE.md](tests/CLAUDE.md)

## research status symbols

| Symbol | Meaning |
|--------|---------|
| VALIDATED | Confirmed through experiments |
| OPEN | Needs testing |
| NEEDS-VERIFICATION | Previous results may have bugs |
| DEAD-END | Tested, doesn't work |
