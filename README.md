# llm-dit-experiments

A standalone diffusers-based experimentation platform for LLM-DiT image generation, starting with Z-Image (Alibaba/Tongyi).

## Features

- **Pluggable LLM backends**: transformers (local) or API (heylookitsanllm)
- **Rich template system**: 144 templates ported from ComfyUI
- **Distributed inference**: Encode on Mac MPS, generate on CUDA
- **Web UI**: HTML+Tailwind interface for testing
- **Experiment infrastructure**: Reproducible experiments with config tracking

## Quick Start

```bash
# Install dependencies
uv sync

# Run smoke test
uv run scripts/smoke_test.py

# Test with model (requires Z-Image model)
uv run scripts/smoke_test.py --model-path /path/to/Tongyi-MAI_Z-Image-Turbo
```

## Web Server

The web server provides a UI for encoding prompts and generating images.

### API Backend Mode (Mac - uses heylookitsanllm)

```bash
# Start heylookitsanllm first (in another terminal)
# Then run the web server pointing to it:
uv run web/server.py --api-url http://localhost:8080 --api-model Qwen3-4B-mxfp4-mlx

# Open http://127.0.0.1:7860 in your browser
```

### Encoder-Only Mode (local model, no generation)

```bash
uv run web/server.py --encoder-only --model-path /path/to/Tongyi-MAI_Z-Image-Turbo
```

### Full Pipeline Mode (CUDA - full generation)

```bash
uv run web/server.py --model-path /path/to/Tongyi-MAI_Z-Image-Turbo
```

### With Config File

```bash
# Copy and customize config
cp config.example.toml config.toml

# Run with config (reads [server] section for host/port)
uv run web/server.py --config config.toml --profile default
```

## CLI Generation

```bash
# Basic generation
uv run scripts/generate.py --prompt "A cat sleeping in sunlight" \
    --model-path /path/to/Tongyi-MAI_Z-Image-Turbo

# With template
uv run scripts/generate.py --prompt "A mountain landscape" \
    --template photorealistic \
    --model-path /path/to/Tongyi-MAI_Z-Image-Turbo

# Using config file
uv run scripts/generate.py --prompt "A sunset" --config config.toml
```

## Distributed Inference

### Option 1: Direct API Mode (Recommended)

Run heylookitsanllm on Mac, generate.py on CUDA box calls Mac for encoding:

**Mac (Terminal 1):**
```bash
cd ~/workspace/heylookitsanllm
uv run heylook --host 0.0.0.0 --port 8080
```

**CUDA Box:**
```bash
# Encode via Mac's heylookitsanllm, generate locally
uv run scripts/generate.py \
    --api-url http://mac-ip:8080 \
    --model-path /path/to/Tongyi-MAI_Z-Image-Turbo \
    "A beautiful sunset over the ocean"
```

### Option 2: Save/Load Embeddings

For offline or batch workflows:

```bash
# Step 1: Encode on Mac and save embeddings
uv run scripts/generate.py --prompt "A beautiful sunset" \
    --model-path /path/to/model \
    --save-embeddings embeddings/sunset.safetensors \
    --encode-only

# Step 2: Transfer embeddings to CUDA server, then generate
scp embeddings/sunset.safetensors cuda-box:/path/to/embeddings/

# Step 3: Generate from embeddings on CUDA
uv run scripts/generate.py \
    --load-embeddings embeddings/sunset.safetensors \
    --model-path /path/to/model
```

## Configuration

See `config.example.toml` for all options. Key sections:

```toml
[server]
host = "127.0.0.1"
port = 7860

[default]
model_path = "/path/to/z-image"
templates_dir = "templates/z_image"

[default.encoder]
device = "auto"
torch_dtype = "bfloat16"

[default.generation]
height = 1024
width = 1024
num_inference_steps = 9
```

## Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run web server tests only
uv run pytest tests/test_web_server.py -v
```

## Project Structure

```
src/llm_dit/
    backends/           # LLM backend abstraction (transformers, API)
    conversation/       # Chat template formatting (Qwen3)
    templates/          # Template loading from markdown
    encoders/           # Text encoding pipeline
    pipelines/          # Diffusion pipeline wrapper
    distributed/        # Save/load embeddings
    config/             # TOML config loading

web/
    server.py           # FastAPI web server
    index.html          # Tailwind UI

templates/z_image/      # 144 prompt templates
scripts/                # CLI tools
tests/                  # Pytest test suite
docs/                   # Technical documentation
```

## Architecture

```
Prompt -> Qwen3Formatter -> TextEncoderBackend -> embeddings -> diffusers -> Image
                                   |
                            (transformers or API)
```

The LLM encoding layer supports full template control. LLM inference can be local (transformers) or remote (heylookitsanllm API). diffusers handles DiT + VAE.

## Key Technical Details (Z-Image)

| Parameter | Value |
|-----------|-------|
| Text encoder | Qwen3-4B (2560 dim) |
| Embedding extraction | hidden_states[-2] |
| CFG scale | 0.0 (baked in via DMD) |
| Steps | 8-9 |
| VAE | 16-channel, Flux-derived |
| Resolution alignment | 16 pixels |

## Documentation

- [SESSION_CONTINUITY.md](SESSION_CONTINUITY.md) - Where we left off
- [GUIDING_PRINCIPLES.md](GUIDING_PRINCIPLES.md) - Architectural decisions
- [CLAUDE.md](CLAUDE.md) - Project instructions
- [docs/heylookitsanllm_hidden_states_spec.md](docs/heylookitsanllm_hidden_states_spec.md) - API spec

## Related Projects

- [ComfyUI-QwenImageWanBridge](../ComfyUI-QwenImageWanBridge) - Source of templates
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) - Reference implementation
- [heylookitsanllm](../heylookitsanllm) - MLX LLM server with hidden states API
