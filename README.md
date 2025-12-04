# llm-dit-experiments

Standalone diffusers-based platform for experimenting with LLM-DiT image generation models. Currently supports Z-Image (Alibaba/Tongyi) with Qwen3-4B text encoder, either locally via `transformers` or distributed via [`heylookitsanllm`](https://github.com/fblissjr/heylookitsanllm), an LLM API server (with hidden state output support) that can run Apple MLX models or llama.cpp GGUF models.

## Features

- Pluggable LLM backends (transformers local, heylookitsanllm API)
- Distributed inference (encode on Mac MPS, generate on CUDA)
- Web UI for generation and prompt testing
- CLI script for image generation
- TOML configuration file support
- 100+ prompt templates with system prompt + thinking tokens
- LoRA support with automatic weight fusion
- Granular device placement (encoder/DiT/VAE independently)
- Multiple scheduler support (flow_euler, flow_heun, dpm_solver, unipc)

## Quick Start

```bash
# Install dependencies
uv sync

# Run web server with local model
uv run web/server.py --model-path /path/to/z-image-turbo

# Generate via CLI
uv run scripts/generate.py \
  --model-path /path/to/z-image-turbo \
  --output image.png \
  "A cat sleeping in sunlight"
```

## Usage

### Web Server

```bash
# Local generation (requires CUDA/MPS)
uv run web/server.py --model-path /path/to/z-image-turbo

# Distributed: API encoder + local generation
uv run web/server.py \
  --api-url http://mac-host:8080 \
  --api-model Qwen3-4B \
  --model-path /path/to/z-image-turbo

# With config file
uv run web/server.py --config config.toml --profile default

# With LoRA
uv run web/server.py \
  --model-path /path/to/z-image-turbo \
  --lora style.safetensors:0.8
```

Access at http://localhost:7860

### CLI Generation

```bash
# Basic generation
uv run scripts/generate.py \
  --model-path /path/to/z-image-turbo \
  --output image.png \
  "A sunset over mountains"

# With template
uv run scripts/generate.py \
  --model-path /path/to/z-image-turbo \
  --template photorealistic \
  "A portrait"

# With config file
uv run scripts/generate.py \
  --config config.toml \
  --profile default \
  "A landscape"

# Granular device control
uv run scripts/generate.py \
  --model-path /path/to/z-image-turbo \
  --text-encoder-device cpu \
  --dit-device cuda \
  --vae-device cuda \
  "A scene"
```

### Configuration

Create a config.toml file:

```toml
[default]
model_path = "/path/to/z-image-turbo"
templates_dir = "templates/z_image"

[default.devices]
text_encoder = "cpu"
dit = "cuda"
vae = "cuda"

[default.generation]
width = 1024
height = 1024
steps = 9
guidance_scale = 0.0

[default.scheduler]
type = "flow_euler"  # flow_euler, flow_heun, dpm_solver, unipc
shift = 3.0

[default.lora]
paths = ["style.safetensors"]
scales = [0.8]
```

CLI flags override config values.

## Running Tests

```bash
# All tests
uv run pytest tests/ -v

# Specific test file
uv run pytest tests/test_web_server.py -v
```

## Project Structure

```
src/llm_dit/
    backends/           # LLM backend abstraction (Protocol-based)
    conversation/       # Chat template formatting (Qwen3)
    templates/          # Template loading system
    encoders/           # Text encoding pipeline
    pipelines/          # Diffusion pipeline wrappers
    utils/              # Utilities (LoRA, etc)
    cli.py              # Shared CLI argument parser
    config.py           # Configuration dataclasses

web/
    server.py           # FastAPI web server
    index.html          # Web UI

templates/z_image/      # 144 prompt templates
scripts/                # CLI tools
tests/                  # Test suite
```

## Architecture

```
Text Prompt
    |
    v
Qwen3Formatter (chat template with thinking blocks)
    |
    v
TextEncoderBackend (transformers/API)
    |
    v
hidden_states[-2] -> embeddings (2560 dim)
    |
    v
diffusers (DiT + VAE)
    |
    v
Image Output
```

## Key Technical Details

| Parameter | Value | Notes |
|-----------|-------|-------|
| Text encoder | Qwen3-4B | 2560 hidden dim, 36 layers |
| Embedding extraction | hidden_states[-2] | Penultimate layer |
| CFG scale | 0.0 | Baked in via Decoupled-DMD |
| Steps | 8-9 | Turbo distilled |
| Scheduler | FlowMatchEuler | shift=3.0 (default) |
| VAE | 16-channel | Wan-family |

### Available Schedulers

| Scheduler | Description |
|-----------|-------------|
| `flow_euler` | Default 1st order Euler for flow matching |
| `flow_heun` | 2nd order Heun, better quality but slower |
| `dpm_solver` | DPM++ 2M multistep, fast and configurable |
| `unipc` | Unified predictor-corrector, fast convergence |

## Documentation

- [docs/distributed_inference.md](docs/distributed_inference.md) - Running text encoder on Mac, DiT on CUDA
- [docs/web_server_api.md](docs/web_server_api.md) - REST API reference
- [CHANGELOG.md](CHANGELOG.md) - Version history

## Related Projects

- [ComfyUI-QwenImageWanBridge](https://github.com/fblissjr/ComfyUI-QwenImageWanBridge) - Original implementation for ComfyUI
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) - Reference architecture
- [heylookitsanllm](https://github.com/fredbliss/heylookitsanllm) - Custom MLX & llama.cpp LLM server
