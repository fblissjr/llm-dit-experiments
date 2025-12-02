# llm-dit-experiments

This file provides guidance to Claude Code when working with this repository.

## Project Overview

A standalone diffusers-based experimentation platform for LLM-DiT image generation, starting with Z-Image (Alibaba/Tongyi). Designed for hobbyist exploration with:
- Pluggable LLM backends (transformers, vLLM, SGLang, mlx)
- Rich template system (140+ templates from ComfyUI)
- Experiment infrastructure with reproducibility tracking
- Living documentation for session continuity

## Critical Rules

- **No emojis** in code, docs, or output
- **Use `uv`** for all Python operations (`uv add`, `uv run`, `uv sync`)
- **Never commit** without explicit user approval
- **Semantic versioning** in CHANGELOG.md (no dates)

## Architecture

```
Text Prompt
    |
    v
Qwen3Formatter (chat template with thinking blocks)
    |
    v
TextEncoderBackend (transformers/vllm/sglang)
    |
    v
hidden_states[-2] -> embeddings (2560 dim)
    |
    v
[diffusers handles: DiT context_refiner -> main layers -> VAE decode]
    |
    v
Image Output
```

## Key Technical Details (Z-Image)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Text encoder | Qwen3-4B | 2560 hidden dim, 36 layers |
| Embedding extraction | hidden_states[-2] | Penultimate layer |
| CFG scale | 0.0 | Baked in via Decoupled-DMD |
| Steps | 8-9 | Turbo distilled |
| Scheduler | FlowMatchEuler | shift=3.0 |
| VAE | 16-channel | Wan-family |
| Context refiner | 2 layers | No timestep modulation |

## Chat Template Format (Qwen3-4B)

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
<think>
{thinking_content}
</think>

{assistant_content}
```

- `add_think_block=True` by default (matches DiffSynth reference)
- Empty assistant_content = no closing `<|im_end|>` (model generating)
- With assistant_content = close with `<|im_end|>` (complete message)

## Directory Structure

```
src/llm_dit/
    backends/           # LLM backend abstraction (Protocol-based)
        protocol.py     # TextEncoderBackend Protocol
        config.py       # BackendConfig dataclass
        transformers.py # HuggingFace transformers backend
    conversation/       # Chat formatting
        types.py        # Message, Conversation dataclasses
        formatter.py    # Qwen3Formatter
    templates/          # Template loading system
        loader.py       # YAML frontmatter parsing
        registry.py     # Template caching
    encoder/            # Text encoding pipeline
        z_image.py      # ZImageTextEncoder
    pipeline/           # Diffusion pipeline wrappers
        z_image.py      # ZImagePipeline
    analysis/           # Experiment analysis utilities
        embedding_compare.py
        visualization.py

templates/z_image/      # 140+ prompt templates (markdown + YAML frontmatter)
experiments/            # Structured experiment tracking
docs/                   # Technical documentation
```

## Living Documentation

- **SESSION_CONTINUITY.md**: Current state, what's working, next steps
- **GUIDING_PRINCIPLES.md**: Architectural decisions and rationale
- **CHANGELOG.md**: Version history (semantic versioning)

## Commands

```bash
# Setup
uv sync

# Run smoke test
uv run scripts/smoke_test.py

# Run with specific backend
uv run scripts/generate.py --prompt "A cat" --template photorealistic

# Run experiment
uv run experiments/001_thinking_block_effect/run.py
```

## Related Projects

- **ComfyUI-QwenImageWanBridge**: Source of templates and ComfyUI implementation patterns
- **DiffSynth-Studio**: Reference implementation for Z-Image pipeline
- **diffusers**: Base library we build on

## Key Research References

Located in sibling repo at `~/workspace/ComfyUI-QwenImageWanBridge/internal/`:
- `z_image_paper_analysis/decoupled_dmd_training_report.md` - CFG baking mechanism
- `z_image_paper_alignment/paper_code_alignment.md` - Implementation alignment
- `z_image_context_refiner_deep_dive.md` - Context refiner architecture
- `z_image_paper_alignment/diffusers_port_considerations.md` - Backend considerations
