# llm-dit-experiments

This file provides guidance to Claude Code when working with this repository.

## Project Overview

A standalone diffusers-based experimentation platform for LLM-DiT image generation, starting with Z-Image (Alibaba/Tongyi). Designed for hobbyist exploration with:
- Pluggable LLM backends (transformers, API/heylookitsanllm, vLLM, SGLang)
- Rich template system (140+ templates from ComfyUI)
- Distributed inference support (encode on Mac, generate on CUDA)
- Web UI and REST API for generation
- Granular device placement (encoder/DiT/VAE on CPU/GPU independently)
- LoRA support with automatic weight fusion
- TOML config file support with CLI overrides

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

**Important**: Our `enable_thinking` parameter uses semantic meaning (True = add think block).

**Exact format when `enable_thinking=True`:**
- Empty think: `<think>\n\n</think>\n\n`
- With content: `<think>\n{content}\n</think>\n\n`
- With assistant: `<think>\n{content}\n</think>\n\n{assistant}<|im_end|>`

All 4 components exposed via API:
- `prompt` (required) - User message
- `system_prompt` (optional) - System message
- `thinking_content` (optional) - Content inside `<think>...</think>`
- `assistant_content` (optional) - Content after `</think>`

## Directory Structure

```
src/llm_dit/
    backends/           # LLM backend abstraction (Protocol-based)
        protocol.py     # TextEncoderBackend Protocol, EncodingOutput
        config.py       # BackendConfig dataclass
        transformers.py # HuggingFace transformers backend
        api.py          # API backend (heylookitsanllm)
    conversation/       # Chat formatting
        types.py        # Message, Conversation dataclasses
        formatter.py    # Qwen3Formatter
    templates/          # Template loading system
        loader.py       # YAML frontmatter parsing
        registry.py     # Template caching
    encoders/           # Text encoding pipeline
        z_image.py      # ZImageTextEncoder
    pipelines/          # Diffusion pipeline wrappers
        z_image.py      # ZImagePipeline
    utils/              # Utility modules
        lora.py         # LoRA loading and fusion
    cli.py              # Shared CLI argument parser and config loading
    config.py           # Configuration dataclasses

scripts/
    generate.py         # CLI image generation script

web/
    server.py           # FastAPI web server (history storage, all endpoints)
    index.html          # Web UI (mobile-friendly, history panel, resolution presets)

templates/z_image/      # 140+ prompt templates (markdown + YAML frontmatter)
config.toml             # Example configuration file

docs/                   # User-facing documentation
    distributed_inference.md  # Running encoder on Mac, DiT on CUDA
    web_server_api.md   # REST API reference
    models/             # Model-specific documentation

internal/               # Development and maintainer documentation
    guides/             # Development guides
        DEVICE_PLACEMENT_GUIDE.md  # Device placement strategies
        TESTING_GUIDE.md           # Testing guidelines
    research/           # Research notes and technical investigations
        heylookitsanllm_hidden_states_spec.md
        decoupled_dmd_training_report.md
        z_image_*.md    # Model analysis and design docs
    reports/            # End-of-day reports
    LOG.md              # Detailed session log with all changes
    SESSION_CONTINUITY.md  # Session state tracking
    GUIDING_PRINCIPLES.md  # Architectural decisions
```

## Living Documentation

- **internal/LOG.md**: Detailed session log with all changes and investigations
- **CHANGELOG.md**: Version history (semantic versioning)

## Configuration

Config file (TOML) is the source of truth. CLI flags override config values.

**Transformers v5 Note:** `load_in_8bit`/`load_in_4bit` are deprecated. Use `quantization` instead.

```toml
# config.toml
[default]
model_path = "/path/to/z-image-turbo"
templates_dir = "templates/z_image"

[default.encoder]
device = "auto"
torch_dtype = "bfloat16"
quantization = "none"  # v5 API: "none", "4bit", "8bit"
cpu_offload = false

[default.pipeline]
device = "cuda"

[default.generation]
width = 1024
height = 1024
steps = 9
guidance_scale = 0.0

[default.scheduler]
type = "flow_euler"  # flow_euler, flow_heun, dpm_solver, unipc
shift = 3.0

[default.optimization]
flash_attn = false
compile = false
cpu_offload = false

[default.lora]
paths = []
scales = []
```

## Web Server

```bash
# Install torch separately (not pinned in pyproject.toml)
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

# Sync other dependencies
uv sync

# Run web server with local encoder (RTX 4090 recommended)
uv run web/server.py \
  --model-path /path/to/z-image-turbo \
  --text-encoder-device cpu \
  --dit-device cuda \
  --vae-device cuda

# Run with config file
uv run web/server.py --config config.toml --profile default

# Run with API encoder (distributed inference)
uv run web/server.py \
  --model-path /path/to/z-image-turbo \
  --api-url http://mac-host:8080 \
  --api-model Qwen3-4B \
  --dit-device cuda \
  --vae-device cuda

# Run with LoRA
uv run web/server.py \
  --model-path /path/to/z-image-turbo \
  --lora /path/to/style.safetensors:0.8 \
  --dit-device cuda

# Enable debug logging for backend comparison
uv run web/server.py --model-path ... --debug
```

## CLI Script (scripts/generate.py)

```bash
# Basic generation
uv run scripts/generate.py \
  --model-path /path/to/z-image-turbo \
  --output image.png \
  "A cat sleeping in sunlight"

# With config file
uv run scripts/generate.py \
  --config config.toml \
  --profile default \
  "A sunset over mountains"

# With LoRA
uv run scripts/generate.py \
  --model-path /path/to/z-image \
  --lora style.safetensors:0.8 \
  "An anime character"

# Full control
uv run scripts/generate.py \
  --model-path /path/to/z-image \
  --text-encoder-device cpu \
  --dit-device cuda \
  --vae-device cuda \
  --width 1280 --height 720 \
  --steps 9 \
  --shift 3.0 \
  --system-prompt "You are a photographer." \
  --thinking-content "Natural lighting, sharp focus." \
  --enable-thinking \
  --output photo.png \
  "A portrait of a woman"
```

## CLI Flags (Shared between web/server.py and scripts/generate.py)

### Model & Config
| Flag | Description |
|------|-------------|
| `--model-path` | Path to Z-Image model |
| `--config` | Path to TOML config file |
| `--profile` | Config profile to use (default: "default") |
| `--templates-dir` | Path to templates directory |

### Device Placement
| Flag | Description |
|------|-------------|
| `--text-encoder-device` | cpu/cuda/mps/auto |
| `--dit-device` | cpu/cuda/mps/auto |
| `--vae-device` | cpu/cuda/mps/auto |

### API Backend
| Flag | Description |
|------|-------------|
| `--api-url` | URL for heylookitsanllm API |
| `--api-model` | Model ID for API backend |
| `--local-encoder` | Force local encoder with API (for A/B testing) |

### Optimization
| Flag | Description |
|------|-------------|
| `--cpu-offload` | Enable CPU offload for transformer |
| `--flash-attn` | Enable Flash Attention |
| `--compile` | Compile transformer with torch.compile |
| `--debug` | Enable debug logging (embedding stats, token IDs) |

### Generation
| Flag | Description |
|------|-------------|
| `--width` | Image width (default: 1024) |
| `--height` | Image height (default: 1024) |
| `--steps` | Inference steps (default: 9) |
| `--guidance-scale` | CFG scale (default: 0.0) |
| `--scheduler` | Scheduler type: flow_euler, flow_heun, dpm_solver, unipc |
| `--shift` | Scheduler shift/mu (default: 3.0) |
| `--seed` | Random seed |

### Prompt Control
| Flag | Description |
|------|-------------|
| `--system-prompt` | System message |
| `--thinking-content` | Content inside `<think>...</think>` |
| `--assistant-content` | Content after `</think>` |
| `--enable-thinking` | Add think block structure |
| `--template` | Template name to use |

### LoRA
| Flag | Description |
|------|-------------|
| `--lora` | LoRA path with optional scale (path:scale). Repeatable. |

## REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Generate image from prompt |
| `/api/encode` | POST | Encode prompt to embeddings |
| `/api/format-prompt` | POST | Preview formatted prompt (no encoding) |
| `/api/templates` | GET | List available templates |
| `/api/schedulers` | GET | List available scheduler types |
| `/api/save-embeddings` | POST | Save embeddings to file |
| `/api/history` | GET | Get generation history |
| `/api/history/{index}` | DELETE | Delete specific history item |
| `/api/history` | DELETE | Clear all history |
| `/health` | GET | Health check |

## API Request Fields

```json
{
  "prompt": "A cat sleeping",
  "system_prompt": "You are a painter.",
  "thinking_content": "Orange fur, green eyes.",
  "assistant_content": "Here is your cat:",
  "enable_thinking": true,
  "template": "photorealistic",
  "width": 1024,
  "height": 1024,
  "steps": 9,
  "seed": 42,
  "guidance_scale": 0.0,
  "shift": 3.0,
  "scheduler": "flow_euler"
}
```

## LoRA Support

LoRAs are loaded and fused into the transformer weights at startup.

**Via CLI:**
```bash
# Single LoRA
--lora style.safetensors:0.8

# Multiple LoRAs (stackable)
--lora style.safetensors:0.5 --lora detail.safetensors:0.3
```

**Via config.toml:**
```toml
[default.lora]
paths = ["style.safetensors", "detail.safetensors"]
scales = [0.5, 0.3]
```

**Via Python:**
```python
pipe = ZImagePipeline.from_pretrained(...)
pipe.load_lora("style.safetensors", scale=0.8)
pipe.load_lora(["lora1.safetensors", "lora2.safetensors"], scale=[0.5, 0.3])
```

Note: LoRAs are fused (permanently merged) into weights. To remove, reload the model.

## Related Projects

- **ComfyUI-QwenImageWanBridge**: Source of templates and ComfyUI implementation patterns
- **DiffSynth-Studio**: Reference implementation for Z-Image pipeline
- **diffusers**: Base library we build on

## Key Research References

Located in the ComfyUI-QwenImageWanBridge sibling repo under `internal/`:
- `z_image_paper_analysis/decoupled_dmd_training_report.md` - CFG baking mechanism
- `z_image_paper_alignment/paper_code_alignment.md` - Implementation alignment
- `z_image_context_refiner_deep_dive.md` - Context refiner architecture
- `z_image_paper_alignment/diffusers_port_considerations.md` - Backend considerations
