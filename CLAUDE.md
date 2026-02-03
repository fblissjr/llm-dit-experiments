# agent context

*last updated: 2026-02-03*

Quick reference for LLM agents. Read only what you need.

**This is a hobbyist exploration platform** - not a product with a finish line. New models, experiments, and features are continuously added. The codebase evolves to support whatever we're curious about next.

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

**Config architecture:** Model-specific dataclasses in `src/llm_dit/config.py`:
- `LTX2Config`, `Flux2Config`, `ZImageConfig`, `QwenImageConfig`, etc.
- **Check these BEFORE adding new fields** - they may already exist!

## feature implementation workflow

**Before implementing ANY feature, trace the full data flow using search:**

### 1. identify all touchpoints
```bash
# Find where the feature name appears (it may already exist!)
grep -rn "<feature>" src/llm_dit/
grep -rn "<feature>" web/server.py

# Check config dataclass for existing fields
grep -A 50 "class <Pipeline>Config" src/llm_dit/config.py
```

### 2. trace entry point to execution
Every feature has a data flow chain. Trace it BEFORE coding:

| Layer | Location | What to grep |
|-------|----------|--------------|
| **API Request Model** | `web/server.py` | `class <Pipeline>GenerateRequest` (Pydantic) |
| **API Endpoint** | `web/server.py` | `@app.post("/api/<pipeline>/")` |
| **Config Defaults** | `src/llm_dit/config.py` | `class <Pipeline>Config` (dataclass) |
| **CLI** | `src/llm_dit/cli.py` | `<pipeline>` subcommand |
| **Pipeline Function** | `src/llm_dit/pipelines/` | Main generation function |
| **Models/Utils** | `src/llm_dit/models/`, `utils/` | Component implementations |

**All layers that accept the parameter must be updated.** Don't just add to the pipeline function - check if the API request model and config also need the field.

### 3. find similar implementations
```bash
# Check if another pipeline already has this feature
grep -rn "<feature>" src/llm_dit/pipelines/
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
| Quantization | `src/llm_dit/quantization/` | All pipelines use this |
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

## navigation by task

| Task | Read |
|------|------|
| **Adding feature to existing pipeline** | **See "feature implementation workflow" above** |
| Web/UI development | [internal/web/CLAUDE.md](internal/web/CLAUDE.md) |
| Writing/running tests | [tests/CLAUDE.md](tests/CLAUDE.md) |
| Research/experiments | [experiments/CLAUDE.md](experiments/CLAUDE.md) |
| Architecture decisions | [internal/principles/architectural_decisions.md](internal/principles/architectural_decisions.md) |
| Agent workflows | [internal/principles/claude_workflow.md](internal/principles/claude_workflow.md) |
| Debugging | [internal/state/lessons_learned.md](internal/state/lessons_learned.md) |
| Adding new pipeline | [internal/checklists/pipeline_integration.md](internal/checklists/pipeline_integration.md) |

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
