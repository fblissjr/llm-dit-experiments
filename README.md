# llm-dit-experiments

Diffusers-based platform for LLM-DiT image generation models. Supports multiple model types with pluggable backends.

## Supported Models

| Model | Task | Input | Output |
|-------|------|-------|--------|
| **Z-Image** (Tongyi) | Text-to-image, img2img | Text prompt | Single RGB image |
| **Qwen-Image-Layered** (Qwen) | Image decomposition | Image + text | Multiple RGBA layers |

## Quick Start

```bash
uv sync --inexact
```

### Z-Image (Text-to-Image)

```bash
# Web UI
uv run web/server.py --model-path /path/to/z-image-turbo

# CLI
uv run scripts/generate.py \
  --model-path /path/to/z-image-turbo \
  --output image.png \
  "A cat sleeping in sunlight"
```

### Qwen-Image-Layered (Image Decomposition)

```bash
# CLI
uv run scripts/generate.py \
  --model-type qwenimage \
  --qwen-image-model-path /path/to/Qwen-Image-Layered \
  --img2img input.jpg \
  "A cheerful child waving under a blue sky"

# Web UI (model selector in header)
uv run web/server.py \
  --model-path /path/to/z-image-turbo \
  --config config.toml
```

Configure `qwen_image.model_path` in config.toml to enable Qwen-Image in web UI.

## Configuration

```bash
cp config.toml.example config.toml
# Edit paths, then:
uv run web/server.py --config config.toml --profile rtx4090
```

Key sections in config.toml:
- `[profile.encoder]` - Text encoder device/dtype
- `[profile.generation]` - Resolution, steps, CFG
- `[profile.qwen_image]` - Qwen-Image-Layered settings
- `[profile.vl]` - Vision conditioning (Qwen3-VL)
- `[profile.rewriter]` - Prompt rewriting settings

See [config.toml.example](config.toml.example) for all options.

## Key Differences

| | Z-Image | Qwen-Image-Layered |
|-|---------|-------------------|
| Text encoder | Qwen3-4B (2560 dim) | Qwen2.5-VL-7B (3584 dim) |
| CFG scale | 0.0 (baked in) | 4.0 (required) |
| Steps | 8-9 (distilled) | 30-50 |
| Resolution | Flexible (16 multiple) | Fixed (640/1024) |
| LoRA | Supported | Not supported |

## CLI Reference

```bash
# Z-Image with device control
uv run scripts/generate.py \
  --model-path /path/to/z-image-turbo \
  --text-encoder-device cpu \
  --dit-device cuda \
  --vae-device cuda \
  "A mountain landscape"

# Z-Image with LoRA
uv run scripts/generate.py \
  --model-path /path/to/z-image-turbo \
  --lora style.safetensors:0.8 \
  "An anime character"

# High-resolution with DyPE (2K)
uv run scripts/generate.py \
  --model-path /path/to/z-image-turbo \
  --dype \
  --dype-scale 2.0 \
  --width 2048 --height 2048 \
  "A detailed landscape"

# Qwen-Image with custom parameters
uv run scripts/generate.py \
  --model-type qwenimage \
  --qwen-image-model-path /path/to/Qwen-Image-Layered \
  --qwen-image-layers 5 \
  --qwen-image-resolution 1024 \
  --qwen-image-cfg-scale 4.0 \
  --img2img input.jpg \
  "Scene description"

# Distributed: encoder on Mac, DiT on CUDA
uv run web/server.py \
  --api-url http://mac-host:8080 \
  --api-model Qwen3-4B \
  --model-path /path/to/z-image-turbo
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Z-Image generation |
| `/api/qwen-image/decompose` | POST | Qwen-Image decomposition |
| `/api/rewrite` | POST | Prompt rewriting |
| `/api/vl/generate` | POST | Z-Image with vision conditioning |

See [docs/web_server_api.md](docs/web_server_api.md) for full API reference.

## Experiments

```bash
# Run ablation sweeps
./experiments/sweep_hidden_layer.sh --quick

# Compare results
uv run experiments/compare.py --list
uv run experiments/compare.py -e shift_sweep --mode grid -o grid.png

# Interactive viewer (port 7861)
uv run experiments/viewer/server.py
```

See [experiments/README.md](experiments/README.md) for documentation.

## Documentation

- [CLAUDE.md](CLAUDE.md) - Technical reference
- [docs/qwen_image_guide.md](docs/qwen_image_guide.md) - Qwen-Image-Layered user guide
- [docs/models/qwen_image_layered.md](docs/models/qwen_image_layered.md) - Qwen-Image-Layered details
- [docs/distributed_inference.md](docs/distributed_inference.md) - Distributed setup
- [docs/web_server_api.md](docs/web_server_api.md) - REST API reference
- [config.toml.example](config.toml.example) - Full configuration
- [CHANGELOG.md](CHANGELOG.md) - Version history
