# llm-dit-experiments

This file provides guidance to Claude Code when working with this repository.

## Project Overview

A standalone diffusers-based experimentation platform for LLM-DiT image generation, starting with Z-Image (Alibaba/Tongyi). Designed for hobbyist exploration with:
- Pluggable LLM backends (transformers, API/heylookitsanllm, vLLM, SGLang)
- Rich template system (140+ templates from ComfyUI)
- Distributed inference support (encode on Mac, generate on CUDA)
- Web UI and REST API for generation
- Granular device placement (encoder/DiT/VAE on CPU/GPU independently)

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

**Important**: DiffSynth/diffusers use `enable_thinking=False` (no think block by default).
The Qwen3 tokenizer parameter is counterintuitive:
- `enable_thinking=True` in tokenizer = NO think block
- `enable_thinking=False` in tokenizer = ADDS empty think block

Our `enable_thinking` parameter matches the semantic meaning (True = add think block).

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

web/
    server.py           # FastAPI web server
    index.html          # Web UI

templates/z_image/      # 140+ prompt templates (markdown + YAML frontmatter)
internal/               # Session logs and debug notes
    LOG.md              # Detailed session log with all changes
docs/                   # Technical documentation
```

## Living Documentation

- **internal/LOG.md**: Detailed session log with all changes and investigations
- **CHANGELOG.md**: Version history (semantic versioning)

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

# Run with API encoder (distributed inference)
uv run web/server.py \
  --model-path /path/to/z-image-turbo \
  --api-url http://mac-host:8080 \
  --api-model Qwen3-4B \
  --dit-device cuda \
  --vae-device cuda

# Enable debug logging for backend comparison
uv run web/server.py --model-path ... --debug
```

## CLI Flags

| Flag | Description |
|------|-------------|
| `--model-path` | Path to Z-Image model |
| `--text-encoder-device` | cpu/cuda/mps/auto |
| `--dit-device` | cpu/cuda/mps/auto |
| `--vae-device` | cpu/cuda/mps/auto |
| `--api-url` | URL for heylookitsanllm API |
| `--api-model` | Model ID for API backend |
| `--local-encoder` | Force local encoder with API (for A/B testing) |
| `--debug` | Enable debug logging (embedding stats, token IDs) |
| `--cpu-offload` | Enable CPU offload for transformer |
| `--flash-attn` | Enable Flash Attention |
| `--compile` | Compile transformer with torch.compile |

## REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Generate image from prompt |
| `/api/encode` | POST | Encode prompt to embeddings |
| `/api/format-prompt` | POST | Preview formatted prompt (no encoding) |
| `/api/templates` | GET | List available templates |
| `/api/save-embeddings` | POST | Save embeddings to file |
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
  "seed": 42
}
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
