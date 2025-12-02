# llm-dit-experiments

A standalone diffusers-based experimentation platform for LLM-DiT image generation, starting with Z-Image (Alibaba/Tongyi).

## Features

- **Pluggable LLM backends**: transformers (now), vLLM/SGLang/mlx (future)
- **Rich template system**: 140+ templates ported from ComfyUI
- **Experiment infrastructure**: Reproducible experiments with config tracking
- **Living documentation**: Session continuity and architectural decisions

## Quick Start

```bash
# Install dependencies
uv sync

# Run smoke test
uv run scripts/smoke_test.py

# Test with model (requires Z-Image model)
uv run scripts/smoke_test.py --model-path /path/to/Tongyi-MAI_Z-Image-Turbo
```

## Project Structure

```
src/llm_dit/
    backends/           # LLM backend abstraction
    conversation/       # Chat template formatting
    templates/          # Template loading
    encoder/            # Text encoding pipeline (coming)
    pipeline/           # Diffusion pipeline (coming)
    analysis/           # Experiment analysis (coming)

templates/z_image/      # 140+ prompt templates
experiments/            # Structured experiments
docs/                   # Technical documentation
```

## Architecture

```
Prompt -> Qwen3Formatter -> TextEncoderBackend -> embeddings -> diffusers -> Image
```

We handle the LLM encoding layer with full template control. diffusers handles DiT + VAE.

## Key Technical Details (Z-Image)

| Parameter | Value |
|-----------|-------|
| Text encoder | Qwen3-4B (2560 dim) |
| Embedding extraction | hidden_states[-2] |
| CFG scale | 0.0 (baked in) |
| Steps | 8-9 |
| VAE | 16-channel |

## Documentation

- [SESSION_CONTINUITY.md](SESSION_CONTINUITY.md) - Where we left off
- [GUIDING_PRINCIPLES.md](GUIDING_PRINCIPLES.md) - Architectural decisions
- [CLAUDE.md](CLAUDE.md) - Project instructions

## Related Projects

- [ComfyUI-QwenImageWanBridge](../ComfyUI-QwenImageWanBridge) - Source of templates
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) - Reference implementation
