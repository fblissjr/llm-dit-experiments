# agent context

*last updated: 2026-01-23*

Quick reference for LLM agents. Uses progressive disclosure - read only what you need.

## guiding principles

**[Read for architectural decisions and documentation protocols](internal/principles/guiding_principles.md)**

Core principles:
- **Principle 0**: Experimentation First - modularity, reproducibility, extensibility
- **Principle 1**: Documentation as First-Class Output - session logging, state files
- **Principle 2**: Progressive Disclosure for Onboarding - layered docs, domain AGENTS.md
- **Principle 3**: State Management for Session Continuity - handoff protocols

## onboarding path

New Claude session? Read in this order:

| Step | File | Purpose |
|------|------|---------|
| 1 | This file (`AGENTS.md`) | Critical rules, quick reference |
| 2 | `internal/state/current.md` | What's happening now |
| 3 | `internal/state/todos.md` | Immediate tasks, handoffs |
| 4 | `guiding_principles.md` | How we work (if making decisions) |
| 5 | Domain docs | Based on your task (see navigation) |

## start here (required)

Read these files when starting a new session:

| File | Purpose | Read | Update |
|------|---------|------|--------|
| **[internal/state/current.md](internal/state/current.md)** | Project status, blockers, recent work | ALWAYS first | After milestones |
| **[internal/state/todos.md](internal/state/todos.md)** | Active session tasks | Check for handoffs | Mark complete, add new |
| **[spec.md](spec.md)** | Product roadmap (P0-P5 backlog) | When picking new work | Check off completed items |
| **[internal/log/log_YYYY-MM-DD.md](internal/log/)** | Today's session log | Optional | Every session |

## critical rules

- **no emojis** in code, docs, or output (status symbols in tables OK)
- **use `uv`** for all Python ops (`uv add`, `uv run`, `uv sync`)
- **never commit** without explicit user approval
- **dtype conventions** - transformers: `dtype=`, diffusers: `torch_dtype=`
- **always update state** after significant work (see below)

## multi-model platform

| Pipeline | Task | Encoder | Status |
|----------|------|---------|--------|
| LTX-2 | text-to-video | Gemma3-12B | Active |
| Z-Image | text-to-image | Qwen3-4B | Production |
| Qwen-Image | editing/decomposition | Qwen2.5-VL-7B | Production |

## state management (required)

### file purposes

| File | What it tracks | Scope |
|------|----------------|-------|
| `current.md` | Project status, active focus, blockers | Big picture |
| `todos.md` | Session-level tasks, handoffs between sessions | Short-term |
| `spec.md` | Product roadmap, prioritized backlog (P0-P5) | Long-term |
| `lessons_learned.md` | Debugging insights, gotchas, solutions | Permanent |
| `log_YYYY-MM-DD.md` | What was done today, decisions made | Historical |

### when to update

| Event | Update |
|-------|--------|
| Completed a backlog item | `spec.md` (check off), `current.md` (if significant) |
| Found a bug or gotcha | `lessons_learned.md` |
| Starting/finishing session | `todos.md`, `log_YYYY-MM-DD.md` |
| Major milestone or blocker | `current.md` |
| Research or analysis complete | `log_YYYY-MM-DD.md`, relevant docs |

## ltx-2 quick reference

**Start here for LTX-2 work:**

| Doc | Purpose | When to Read |
|-----|---------|--------------|
| **[ltx2_comprehensive_reference.md](internal/docs/ltx2/ltx2_comprehensive_reference.md)** | **AUTHORITATIVE** - Full architecture, text conditioning, inference, memory | First! Covers everything |
| [model_file_structure.md](internal/research/ltx2/model_file_structure.md) | Model weights, tensor shapes | Loading/debugging weights |
| [model_inventory.md](internal/research/ltx2/model_inventory.md) | Directory structure, component sizes | Disk/memory planning |
| [paper_analysis.md](internal/research/ltx2/paper_analysis.md) | Paper claims, ablation gaps | Research directions |
| [memory_optimization_ltx2_24gb.md](internal/analysis/memory/memory_optimization_ltx2_24gb.md) | 24GB VRAM strategies | OOM issues |

### ltx-2 params

| Param | Value | Notes |
|-------|-------|-------|
| encoder | Gemma3-12B (Q4) | 3840 dim, 49 layers |
| cfg | 4.0 | with latent normalization |
| steps | 40 | T2V default |
| frames | 121 | default (33 for quick tests) |
| quantization | FP8-quanto | fits 24GB GPU |

### z-image (turbo)

| Param | Value | Notes |
|-------|-------|-------|
| encoder | Qwen3-4B | 2560 dim, layer -2 |
| cfg | 0.0 | baked in (Decoupled-DMD) |
| steps | 8-9 | turbo distilled |

## navigation

### by task

| Task | Read | Why |
|------|------|-----|
| Picking new work | [spec.md](spec.md) | Prioritized backlog (P0-P5) |
| Writing/running tests | [tests/AGENTS.md](tests/AGENTS.md) | Complete testing guide |
| Research/experiments | [experiments/AGENTS.md](experiments/AGENTS.md) | Research protocols |
| Architecture decisions | [internal/principles/guiding_principles.md](internal/principles/guiding_principles.md) | Design rationale |
| Debugging | [internal/state/lessons_learned.md](internal/state/lessons_learned.md) | Past solutions |
| **LTX-2 architecture** | [internal/docs/ltx2/ltx2_comprehensive_reference.md](internal/docs/ltx2/ltx2_comprehensive_reference.md) | **AUTHORITATIVE** reference |
| LTX-2 data flow | [internal/docs/ltx2/technical_data_flow.md](internal/docs/ltx2/technical_data_flow.md) | CLI to video trace |

### by model

| Model | Guide |
|-------|-------|
| LTX-2 | [internal/guides/ltx2_*.md](internal/guides/) |
| Z-Image | [internal/guides/z_image_*.md](internal/guides/) |
| All models | [internal/models/overview.md](internal/models/overview.md) |

### reference docs

| Topic | Doc |
|-------|-----|
| CLI flags | [docs/reference/cli_flags.md](docs/reference/cli_flags.md) |
| API endpoints | [docs/reference/api_endpoints.md](docs/reference/api_endpoints.md) |
| Configuration | [docs/reference/configuration.md](docs/reference/configuration.md) |

### technical deep-dives

| Model | Doc | Purpose |
|-------|-----|---------|
| LTX-2 | [internal/docs/ltx2/ltx2_comprehensive_reference.md](internal/docs/ltx2/ltx2_comprehensive_reference.md) | **AUTHORITATIVE** - Full architecture, text conditioning, inference |
| LTX-2 | [internal/docs/ltx2/technical_data_flow.md](internal/docs/ltx2/technical_data_flow.md) | CLI to video end-to-end |
| LTX-2 | [internal/docs/ltx2/data_flow_diagram.md](internal/docs/ltx2/data_flow_diagram.md) | Visual tensor shapes |
| LTX-2 | [internal/docs/ltx2/e2e_code_path.md](internal/docs/ltx2/e2e_code_path.md) | File-by-file code trace |
| FLUX.2 Klein | [internal/docs/flux2-klein/flux2_klein_comprehensive_reference.md](internal/docs/flux2-klein/flux2_klein_comprehensive_reference.md) | **AUTHORITATIVE** - Full architecture, Qwen3 text encoder, FP8 optimization |

Full index: [internal/docs/README.md](internal/docs/README.md)

## research status symbols

| Symbol | Meaning |
|--------|---------|
| VALIDATED | Confirmed through experiments |
| OPEN | Needs testing |
| NEEDS-VERIFICATION | Previous results may have bugs |
| DEAD-END | Tested, doesn't work |

Research tracking: [experiments/AGENTS.md](experiments/AGENTS.md)

## architecture

```
Text Prompt -> TextEncoder -> hidden_states[layer] -> DiT -> VAE -> Image/Video
```

Details: [internal/models/overview.md](internal/models/overview.md)

## quick test commands

```bash
# Smoke test (GPU required)
uv run pytest tests/e2e/test_baseline_portable.py::TestBaselineSmoke -v -s

# Unit tests (no GPU)
uv run pytest tests/unit/ -v

# All LTX-2 tests
uv run pytest tests/ -v -k ltx2
```

Full testing guide: [tests/AGENTS.md](tests/AGENTS.md)
