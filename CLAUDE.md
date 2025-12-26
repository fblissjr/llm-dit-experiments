# llm-dit-experiments

This file provides guidance to Claude Code when working with this repository.

## Critical Rules

- **No emojis** in code, docs, or output
- **Use `uv`** for all Python operations (`uv add`, `uv run`, `uv sync`)
- **Never commit** without explicit user approval
- **Semantic versioning** in CHANGELOG.md (no dates)
- **Update documentation** after any feature (see `docs/reference/documentation_checklist.md`)
- **dtype parameter conventions** - different libraries use different parameter names:
  - **transformers** (AutoModel, AutoModelForCausalLM, Qwen3VLForConditionalGeneration): use `dtype=`
  - **diffusers** (DiffusionPipeline, AutoencoderKL, FluxTransformer2DModel): use `torch_dtype=`
  - Using `torch_dtype=` with transformers causes deprecation warnings

## Project Overview

A standalone diffusers-based experimentation platform for LLM-DiT image generation, starting with Z-Image (Alibaba/Tongyi). Designed for hobbyist exploration with:
- Pluggable LLM backends (transformers, API/heylookitsanllm, vLLM, SGLang)
- Rich template system (140+ templates from ComfyUI)
- Distributed inference support (encode on Mac, generate on CUDA)
- Web UI and REST API for generation
- Granular device placement (encoder/DiT/VAE on CPU/GPU independently)
- LoRA support with automatic weight fusion
- TOML config file support with CLI overrides

## Architecture

```
Text Prompt → Qwen3Formatter → TextEncoderBackend → hidden_states[layer] → DiT → VAE → Image
```

The text encoder extracts embeddings from Qwen3-4B's hidden states (default layer -2, configurable). The DiT uses these embeddings with flow matching to generate latents, which the VAE decodes to images.

## Key Parameters

### Z-Image

| Parameter | Value | Notes |
|-----------|-------|-------|
| Text encoder | Qwen3-4B | 2560 hidden dim, 36 layers |
| Embedding layer | -2 | Penultimate (configurable via `--hidden-layer`) |
| **Max tokens** | **1504** | DiT RoPE limit - see `docs/reference/long_prompts.md` |
| CFG scale | 0.0 | Baked in via Decoupled-DMD |
| Steps | 8-9 | Turbo distilled |
| Scheduler | FlowMatchEuler | shift=3.0 |

### Qwen-Image-Layered

| Parameter | Value | Notes |
|-----------|-------|-------|
| Text encoder | Qwen2.5-VL-7B | 3584 hidden dim |
| CFG scale | 4.0 | Required (not baked in) |
| Steps | 50 | Non-distilled |
| Resolution | 640 or 1024 | Fixed constraint |

See `docs/qwen_image_guide.md` for detailed usage.

## Quick Start

```bash
# Install dependencies
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv sync

# Run web server
uv run web/server.py --config config.toml --profile default

# CLI generation
uv run scripts/generate.py --model-path /path/to/z-image "A cat sleeping"
```

## Directory Structure

```
src/llm_dit/
    backends/           # LLM backend abstraction (transformers, API, vLLM)
    constants/          # Token IDs, model configs
    conversation/       # Chat formatting (Qwen3Formatter)
    encoders/           # ZImageTextEncoder
    pipelines/          # ZImagePipeline, QwenImagePipeline
    schedulers/         # FlowMatchScheduler
    models/             # ContextRefiner, VAE, DiT components
    utils/              # LoRA, attention, tiled VAE, DyPE
    vl/                 # Vision conditioning (Qwen3-VL)
    cli.py              # CLI argument parser
    config.py           # Configuration dataclasses

scripts/
    generate.py         # CLI image generation
    profiler.py         # Performance testing

web/
    server.py           # FastAPI server
    index.html          # Web UI

templates/z_image/      # 140+ prompt templates
experiments/            # Ablation studies, VL experiments
tests/                  # Unit and integration tests
docs/                   # User-facing documentation
internal/               # Development docs, research, session logs
```

## Documentation Map

### Reference (specs and facts)

| Topic | File | When to Read |
|-------|------|--------------|
| CLI flags | `docs/reference/cli_flags.md` | Adding/modifying CLI arguments |
| API endpoints | `docs/reference/api_endpoints.md` | Working on REST API |
| Configuration | `docs/reference/configuration.md` | Adding new parameters (DRY principles) |
| Resolution | `docs/reference/resolution.md` | VAE constraints, presets |
| DyPE | `docs/reference/dype.md` | High-resolution generation (2K+) |
| Long prompts | `docs/reference/long_prompts.md` | 1504 token limit, compression modes |
| Chat templates | `docs/reference/chat_templates.md` | Qwen3/Qwen3-VL template format |
| Doc checklist | `docs/reference/documentation_checklist.md` | After implementing features |

### Guides (how-to docs)

| Topic | File | When to Read |
|-------|------|--------------|
| VL conditioning | `docs/guides/vl_conditioning.md` | Vision conditioning with Qwen3-VL |
| Prompt rewriting | `docs/guides/prompt_rewriting.md` | Qwen3 prompt expansion |
| LoRA | `docs/guides/lora.md` | Loading and fusing LoRAs |
| Distributed | `docs/guides/distributed.md` | Mac encode, CUDA generate |
| Profiler | `docs/guides/profiler.md` | Performance testing |

### Internal (research and development)

| Topic | File | Notes |
|-------|------|-------|
| Index | `internal/index.md` | Map of all internal docs |
| Session state | `internal/SESSION_CONTINUITY.md` | Current focus, blockers, next steps |
| Guiding principles | `internal/GUIDING_PRINCIPLES.md` | Architectural decisions (north star) |
| Hidden layers | `internal/research/hidden_layer_selection.md` | Layer selection experiments |
| Long prompts | `internal/research/long_prompt_research.md` | Compression research |
| VL integration | `internal/research/qwen3_vl_integration.md` | VL conditioning theory |
| Session logs | `internal/log/` | Chronological session history |

## Configuration

Config file (TOML) is the source of truth. CLI flags override config values.

```bash
# Use config file with profile
uv run web/server.py --config config.toml --profile rtx4090

# Override specific settings via CLI
uv run scripts/generate.py --config config.toml --hidden-layer -6 "Prompt"
```

See `config.toml.example` for all options and `docs/reference/configuration.md` for DRY principles.

## Common CLI Examples

```bash
# Basic generation
uv run scripts/generate.py --model-path /path/to/z-image "A cat sleeping"

# With LoRA
uv run scripts/generate.py --model-path /path/to/z-image --lora style.safetensors:0.8 "Prompt"

# High-res with DyPE
uv run scripts/generate.py --model-path /path/to/z-image --dype --width 2048 --height 2048 "Prompt"

# Image-to-image
uv run scripts/generate.py --model-path /path/to/z-image --img2img input.jpg --strength 0.7 "Prompt"

# Run profiler
uv run scripts/profiler.py --model-path /path/to/z-image --tests encode,generate
```

See `docs/reference/cli_flags.md` for complete flag reference.

## API Usage

```bash
# Generate image
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cat", "width": 1024, "height": 1024}'

# List templates
curl http://localhost:8000/api/templates

# Health check
curl http://localhost:8000/health
```

See `docs/reference/api_endpoints.md` for complete API reference.

## PyTorch-Native Components

We maintain pure PyTorch implementations to reduce diffusers dependency:

- **FlowMatchScheduler**: Custom scheduler with shift=3.0 for Z-Image
- **TiledVAEDecoder**: Decode large images (2K+) in tiles
- **ContextRefiner**: Standalone 2-layer text processor
- **Attention Backend**: Auto-detect Flash Attention 3/2, Sage, xFormers, SDPA

```python
from llm_dit import FlowMatchScheduler, setup_attention_backend

setup_attention_backend("auto")  # Best available
scheduler = FlowMatchScheduler(shift=3.0)
```

## Adding New Parameters

Follow DRY principles - parameters must flow through a single chain:

```
config.toml → Config dataclass → RuntimeConfig → startup.py → Backend configs
```

1. Add to `config.toml.example`
2. Add to Config dataclass in `src/llm_dit/config.py`
3. Add CLI argument in `src/llm_dit/cli.py`
4. Wire through `load_runtime_config()` and `startup.py`
5. Run test: `uv run pytest tests/unit/test_dry_config.py -v`

See `docs/reference/configuration.md` for detailed checklist.

## Internal Research Context

The `internal/` directory contains valuable research context:

- **Guiding principles** (`internal/GUIDING_PRINCIPLES.md`): North star for architectural decisions
- **Research notes**: Hypotheses, experiments, findings in `internal/research/`
- **Session logs**: Chronological history of changes in `internal/log/`
- **Session continuity**: Current state and next steps in `internal/SESSION_CONTINUITY.md`

When working on a feature, check relevant internal docs for context on past decisions and experiments.

## Experiments

VL conditioning experiments are in `experiments/qwen3_vl/`. See `docs/guides/vl_conditioning.md` for the main guide.

```bash
# Run VL experiment
cd experiments/qwen3_vl/scripts
uv run run_comparison.py -i input.png -p "Prompt" --sweep style_transfer
```

Experiment results go in `experiments/results/`. The comparison viewer runs on port 7861:
```bash
uv run experiments/viewer/server.py
```

### Experiment Conventions

All experiment scripts **must** use the shared utilities in `experiments/utils.py`:

```python
from experiments.utils import save_image_grid, save_metadata, create_comparison_grid

# Save comparison grids
save_image_grid(images, output_dir / "comparison.png", cols=4, labels=["A", "B", "C", "D"])

# Save metadata as JSON with automatic timestamp
save_metadata(output_dir / "metadata.json", prompt=prompt, steps=steps, cfg=cfg)

# Create grid without saving (for further processing)
grid = create_comparison_grid(images, cols=3)
```

**Required for all experiments:**
- Output directory: `experiments/results/<experiment_name>/`
- Grid images: Use `save_image_grid()` or `create_comparison_grid()`
- Metadata: Use `save_metadata()` (JSON format with timestamps)

**Do not** create custom grid/metadata functions - use the shared utilities.

## Living Documentation

- **Session logs**: Create `internal/log/log_YYYY-MM-DD.md` for each session
- **Session continuity**: Update `internal/SESSION_CONTINUITY.md` at session end
- **CHANGELOG.md**: Semantic versioning for all changes

After implementing any feature, see `docs/reference/documentation_checklist.md`.
