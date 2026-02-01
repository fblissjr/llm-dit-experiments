# vision

*last updated: 2026-02-01*

> The strategic north star for `llm-dit-experiments`. This document changes rarely (1-2x/year). For day-to-day work, see [CLAUDE.md](CLAUDE.md).

---

## what is this?

A **high-granularity, model-agnostic experimentation platform** for Diffusion Transformers (DiT) and LLM-conditioned image/video generation. Unlike traditional repositories that silo logic inside specific model implementations, this framework treats every aspect of generation - from hardware-level math to high-level orchestration - as a composable unit.

---

## philosophy

This is a hobbyist exploration platform. Design choices prioritize:

1. **Modularity and Granularity** - Common components, models, and pipelines that can be reused across current and future experiments
2. **Reproducibility and Confidence** - Code that is accurate, reliable, and well-tested
3. **Extensibility** - Building for different models while reusing core components
4. **Understanding over Magic** - We want to understand *why* things work, not just *that* they work

---

## the problem

Existing DiT repositories share common limitations:

| Problem | Impact |
|---------|--------|
| Logic siloed inside model implementations | Can't reuse schedulers, attention, or guidance across models |
| Generation loops hardcoded to specific inputs | Each model requires a completely separate pipeline |
| No resource management | 24GB consumer GPUs can't run 19B+ models |
| Optimizations scattered and duplicated | RMSNorm, RoPE, attention backends reimplemented per model |
| Monolithic pipelines | Testing one component requires the whole stack |

---

## the solution: composable hierarchy

We organize code into six levels of granularity. **Reusability increases as you move down the stack.**

| Level | Role | Key Components | Reusability | Why |
|:------|:-----|:---------------|:------------|:----|
| **L1: Orchestration** | The Conductor | `Orchestrator`, `ModelPool`, `PipelineSteps` | **Total (100%)** | Model-agnostic. Coordinates pipelines and manages VRAM. |
| **L2: Pipelines** | The Workflow | `ZImagePipeline`, `LTX2Pipeline`, `WanVideoPipeline` | **Paradigm-Based** | Reusable for any model sharing the same math (e.g., Flow Matching). |
| **L3: Backbones** | The Models | `DiT Transformer`, `Gemma3/Qwen3 LLMs`, `VideoVAE` | **Zero (Atomic)** | Tied to specific weights. These are the fixed blocks. |
| **L4: Behaviors** | The Logic | `Schedulers`, `Guidance (SLG/FMTT)`, `Conditioning` | **High (Structural)** | Pluggable into any backbone with standard structure. |
| **L5: Primitives** | The Math | `Attention Backends`, `Quantization`, `DyPE/YaRN` | **Absolute (Universal)** | Math primitives at the Tensor/Linear level. |
| **L6: Foundations** | The Plumbing | `MemoryTracker`, `VaeOps`, `Logging`, `Templates` | **Universal** | Base utilities supporting all other levels. |

### the composable DAG

```
[ WORKFLOW ] (L1: Orchestrator)
      |
      |-- [ LOGIC ] (L2: Pipeline) <---------------------------.
      |      |   (Reusable for models in same math family)      |
      |      |                                                  |
      |      |-- [ BRAIN ] (L4: Scheduler / Sampler)            | (L4: GUIDANCE)
      |      |      (Composable if Paradigm matches)            |-- SLG
      |      |                                                  `-- FMTT
      |      |                                              (Wraps any DiT)
      |      |-- [ BACKBONE ] (L3: Model Weights)
      |      |      |   (Atomic: The specific model)
      |      |      |
      |      |      |-- [ COMPRESSION ] (L5: Quantization)
      |      |      |      (Composable: Applies to any Linear layer)
      |      |      |
      |      |      `-- [ COMPUTE ] (L5: Performance Kernels)
      |      |             (Universal: FA2, SageAttn, Compile)
      |      |
      |      `-- [ GEOMETRY ] (L5: Position Ops)
      |             (Universal: RoPE, DyPE, YaRN)
      |
      `-- [ PLUMBING ] (L6: Utils)
             (Universal: Memory, Logging, VAE Ops, Tiling)
```

---

## what makes this different

| Traditional Approach | Our Approach |
|---------------------|--------------|
| One repo per model | Multi-model platform with shared primitives |
| Model owns generation loop | Pipeline owns loop, model is a dumb forward pass |
| Optimizations scattered | Centralized L5 primitives |
| VRAM: hope it fits | L1 Orchestrator manages VRAM budget |
| Test the whole pipeline | Test each level independently |
| Copy-paste to add models | Implement L3 interface, reuse everything else |

---

## what this is NOT

- **Not a training framework** - Inference and experimentation only
- **Not a Hugging Face wrapper** - Pure PyTorch `nn.Module`s for full control
- **Not a production service** - Hobbyist exploration platform
- **Not a diffusers fork** - We build from scratch to understand and experiment

---

## current pipelines

| Pipeline | Task | Encoder | Status |
|----------|------|---------|--------|
| **FLUX.2 Klein** | text-to-image, image editing | Qwen3-8B/4B | Production |
| LTX-2 | text-to-video | Gemma3-12B | Production |
| Z-Image Turbo | text-to-image | Qwen3-4B | Production |
| Z-Image Base | text-to-image | Qwen3-4B | Production |
| Qwen-Image | editing/decomposition | Qwen2.5-VL-7B | Production |
| Wan Video | text-to-video | UMT5-XXL | Phase 1 |

---

## architecture at a glance

```
Text Prompt -> TextEncoder -> hidden_states[layer] -> DiT -> VAE -> Image/Video
```

The text encoder extracts semantic features, the DiT transforms noise conditioned on those features, and the VAE decodes latents to pixels.

---

## related documentation

| Doc | Purpose | When to Read |
|-----|---------|--------------|
| [CLAUDE.md](CLAUDE.md) | Agent quick reference, critical rules | Every session |
| [spec.md](spec.md) | Product roadmap, P0-P5 backlog | Picking new work |
| [internal/principles/architectural_decisions.md](internal/principles/architectural_decisions.md) | Protocol patterns, design decisions | Making ML decisions |
| [internal/principles/claude_workflow.md](internal/principles/claude_workflow.md) | Documentation, state management | Agent workflows |
