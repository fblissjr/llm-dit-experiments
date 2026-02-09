# agent context (v0.9.0)

*last updated: 2026-02-09*

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
| 2 | [internal/state/current.md](internal/state/current.md) | What's happening now, active pipelines, versions |
| 3 | Domain docs (see navigation below) | Based on your task |

**Optional:** [VISION.md](VISION.md) for architecture philosophy (L1-L6 composability hierarchy).

## critical rules

- **no emojis** in code, docs, or output
- **use `uv`** for all Python ops (`uv add`, `uv run`, `uv sync`)
- **never commit** without explicit user approval
- **use `ModelManager`** for all model load/unload/reload -- never manipulate model globals directly
- **always update state** after significant work (see state management below)

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

## navigation by task

| Task | Read |
|------|------|
| **Adding feature to existing pipeline** | Feature workflow (below) + [post_refactor_guide.md](internal/docs/architecture/post_refactor_guide.md) |
| **Adding new pipeline** | [pipeline_integration.md](internal/checklists/pipeline_integration.md) |
| **Web/UI development** | [internal/web/CLAUDE.md](internal/web/CLAUDE.md) |
| **Writing/running tests** | [tests/CLAUDE.md](tests/CLAUDE.md) |
| **Research/experiments** | [experiments/CLAUDE.md](experiments/CLAUDE.md) |
| **Post-refactor architecture** | [post_refactor_guide.md](internal/docs/architecture/post_refactor_guide.md) |
| **Architecture decisions** | [architectural_decisions.md](internal/principles/architectural_decisions.md) |
| **Planned improvements / tech debt** | [backlog.md](internal/state/backlog.md) |
| **Debugging** | [lessons_learned.md](internal/state/lessons_learned.md) |
| **Agent workflows** | [claude_workflow.md](internal/principles/claude_workflow.md) |
| **Quantization** | [quantization.md](docs/reference/quantization.md) |
| **HTTPS setup** | [README.md](README.md#https-setup) |
| **Model-specific docs** | See quickstarts below |

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

**Key insight:** Encoders and core infra are shared. Check which pipelines use a component before modifying it. See [current.md](internal/state/current.md) for active pipelines and parallel work.

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

## architecture patterns (post-refactor)

**Circular import prevention:** Routers that need server state (generation_history, encoder_only_mode, rewriter_backend) do `import web.server as srv` at module level. Server imports routers in `_register_routers()` called from `main()`, NOT at module level. Never move router imports to top of server.py. Note: flux2.py and config_mgmt.py no longer import srv at all.

**Router decomposition:** server.py (~296 lines) holds only server state globals and startup. All 68 API endpoints live in 7 domain routers under `web/routers/`. All routers use `ConfigDep`/`ManagerDep` dependency injection for pipeline access. No pipeline globals remain in server.py -- ModelManager is the sole source of truth.

**LoRA fusion tracking:** `model._fused_lora_state` (FusedLoRAState) tracks what's fused on persistent models. Prevents re-fusion OOM where fp8 (9GB) dequantizes to bf16 (18GB). Pipeline detects mismatch (raises RuntimeError), router handles recovery (reload).

**Unified quantization:** All pipelines use `quantize_component()` as the sole entry point. torchao is the only backend. See `docs/reference/quantization.md`.

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
web/routers/<pipeline>.py: domain router endpoint merges request + runtime_config
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
| Tests pass, bad output | Visual verification skipped | Always verify baselines visually |
| API returns error | Pydantic validation | Check request schema in `web/schemas.py` |
| Circular import | Router imports server at module level | See architecture patterns above |

## quick test commands

```bash
# Unit tests (no GPU, fast)
uv run pytest tests/unit/ -v

# Smoke test (GPU required)
uv run pytest tests/e2e/test_baseline_portable.py::TestBaselineSmoke -v -s

# LoRA tests
uv run pytest tests/unit/ -v -k lora

# Config DRY validation
uv run pytest tests/unit/test_dry_config.py -v
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
