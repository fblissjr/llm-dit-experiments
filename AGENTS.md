# agent context

*last updated: 2026-01-24*

Quick reference for LLM agents. Uses progressive disclosure - read only what you need, but always read

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
| 4 | `internal/principles/guiding_principles.md` | How we work (if making decisions) |
| 5 | `internal/principles/modular_architecture.md` | Our approach and vision for a modular architecture and the levels of granularity |
| 6 | Domain docs | Based on your task (see navigation) |

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

## modular architecture
This table defines the structural hierarchy of the **llm-dit-experiments** platform, moving from top-level business logic down to hardware-level math primitives.

### Hierarchy of Granularity & Composability

| Level | Role | Key Components | Reusability / Composability | Dependency Logic |
| :--- | :--- | :--- | :--- | :--- |
| **L1: Orchestration** | **The Conductor** | `Orchestrator`, `ModelPool`, `PipelineSteps` | **Total (100%)** | Model-agnostic. Coordinates between multiple pipelines and manages VRAM. |
| **L2: Pipelines** | **The Workflow** | `ZImagePipeline`, `LTX2Pipeline`, `WanVideoPipeline` | **Paradigm-Based** | Reusable for any model sharing the same math (e.g., all "Flow Matching" models). |
| **L3: Backbones** | **The Models** | `DiT Transformer`, `Gemma3/Qwen3 LLMs`, `VideoVAE` | **Zero (Atomic)** | Tied to specific weights. These are the fixed blocks that everything else acts upon. |
| **L4: Behaviors** | **The Logic** | `Schedulers`, `Guidance (SLG/FMTT)`, `Conditioning` | **High (Structural)** | Pluggable into any backbone that has a standard structure (e.g., a list of `layers`). |
| **L5: Primitives** | **The Math** | `Attention Backends`, `Quantization`, `DyPE/YaRN` | **Absolute (Universal)** | Math primitives. They apply to the `Tensor` or `Linear` layer level regardless of model. |
| **L6: Foundations** | **The Plumbing** | `MemoryTracker`, `VaeOps`, `Logging`, `Templates` | **Universal** | Base-level utilities that support all other levels. |

## multi-model platform

| Pipeline | Task | Encoder | Status |
|----------|------|---------|--------|
| **FLUX.2 Klein** | **text-to-image, image editing** | Qwen3-8B/4B | **Production** (2026-01-24) |
| LTX-2 | text-to-video | Gemma3-12B | Production |
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

## flux.2 klein quick reference

**Start here for FLUX.2 Klein work:**

| Doc | Purpose | When to Read |
|-----|---------|--------------|
| **[flux2_klein_comprehensive_reference.md](internal/docs/flux2-klein/flux2_klein_comprehensive_reference.md)** | **AUTHORITATIVE** - Full architecture, Qwen3 encoder, inference | First! Covers everything |
| [flux2_implementation_plan.md](~/.claude/plans/) | Implementation plan, phases | Understanding scope |

### flux.2 params

| Param | Value | Notes |
|-------|-------|-------|
| encoder | Qwen3-8B (Klein 9B) | 12288 dim, layers [9,18,27] |
| encoder | Qwen3-4B (Klein 4B) | 7680 dim, layers [9,18,27] |
| guidance | 1.0 (distilled) | baked in (guidance-distilled) |
| steps | 4 (distilled) | or 50 for base models |
| resolution | 1024×1024 | default |

### flux.2 usage

```bash
# Text-to-image (with FP8 and block offload for 24GB GPU)
uv run scripts/generate.py --model-type flux2 \
    --flux2-model-name klein-9b-fp8 \
    --flux2-block-offload \
    --flux2-model-path models/FLUX.2-klein/FLUX.2-klein-9b-fp8/ \
    --flux2-vae-path models/FLUX.2-klein/FLUX.2-klein-9B/ \
    "A photo of a cat"

# Image editing with references (1-4+ images supported)
uv run scripts/generate.py --model-type flux2 \
    --flux2-model-name klein-9b-fp8 \
    --flux2-block-offload \
    --flux2-model-path models/FLUX.2-klein/FLUX.2-klein-9b-fp8/ \
    --flux2-vae-path models/FLUX.2-klein/FLUX.2-klein-9B/ \
    --flux2-input-image input1.jpg input2.jpg \
    "Combine subject from image 1 with background from image 2"
```

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
| **Adding new pipeline** | [internal/checklists/pipeline_integration.md](internal/checklists/pipeline_integration.md) | **Complete checklist** for config integration |
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
