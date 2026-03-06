# vision

*last updated: 2026-02-17*

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
| No resource management | 24GB consumer GPUs can't run 22B+ models |
| Optimizations scattered and duplicated | RMSNorm, RoPE, attention backends reimplemented per model |
| Monolithic pipelines | Testing one component requires the whole stack |

---

## the solution: composable hierarchy

We organize code into six levels of granularity. **Reusability increases as you move down the stack.**

| Level | Role | Key Components | Reusability | Why |
|:------|:-----|:---------------|:------------|:----|
| **L1: Orchestration** | The Conductor | `Orchestrator`, `ModelPool`, `PipelineSteps`, `DAG Engine` | **Total (100%)** | Model-agnostic. Coordinates pipelines, manages VRAM, resolves parameters. |
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

## toward composable orchestration (L1 vision)

The gap between our current L2 (siloed pipelines) and full L1 Orchestration is now concrete, informed by analysis of real-world ComfyUI workflows: a 175-node LTX-2 video generation pipeline and a 72-node FLUX.2 inpainting pipeline. See `internal/research/comfyui_workflows/` for detailed analyses.

### what "composable" means in practice

Complex generation is not one pipeline -- it is a DAG of composable operations where each node has its own parameters, model configuration, and guidance strategy:

```
                      Prompt
                     /      \
              Video Branch   Audio Branch
              /    |    \         |
       LoRA-A  LoRA-B  LoRA-C   Audio Model
          |      |       |         |
       Stage 1  ...    Stage 2   Audio Denoise
          |              |         |
      Upsample     Refine        Decode
          \          /             |
           Composite              |
               \                 /
                Mux -> Final Output
```

Key patterns observed in real workflows:
- **Multi-LoRA branching:** One model load, three LoRA configs applied per-branch (not per-call)
- **Chained sampling:** unsample -> resample -> resample, each with separate scheduler/guidance configs
- **Conditional routing:** Input properties determine which branch executes (I2V vs T2V, photo vs illustration)
- **Cross-model chaining:** Output of one model feeds into a different model (generate -> upscale)
- **Parallel execution:** Audio and video denoise simultaneously with separate guidance

### three prerequisites

| Prerequisite | Level | Status |
|-------------|-------|--------|
| **Parameter resolution layers** | Cross-cutting | DONE (v0.9.9) -- `resolve_param()` establishes the precedence pattern |
| **DiTProtocol** | L3 | Not started -- standardized `forward(x, t, context, **kwargs)` across all DiT models |
| **UniversalFlowMatchLoop** | L2 | Not started -- generation loop decoupled from model-specific inputs |

### the parameter resolution foundation

In a composable DAG, each node needs a consistent way to receive its parameters. The three-layer model established in v0.9.9 generalizes directly:

| Current (API) | Future (DAG Node) |
|--------------|-------------------|
| Client-sent value | Node-specific override |
| config.toml default | Workflow-level default |
| Schema default | Pipeline default |

The `resolve_param()` function is the resolution primitive. Any future orchestration layer will reuse this pattern -- the only change is what provides each layer's values.

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
