# llm-dit-experiments

PyTorch and Diffusers-based (depending on the models / pipeline) experimentation platform for LLM-DiT image generation. Pluggable backends, quantization, and quality of life features for research.

## Pipelines

| Pipeline | Task | Encoder | Steps | Notes |
|----------|------|---------|-------|-------|
| Z-Image | text-to-image, img2img | Qwen3-4B (2560 dim) | 8-9 | CFG=0 baked, 1504 token limit |
| Qwen-Image-Layered | image decomposition | Qwen2.5-VL-7B (3584 dim) | 50 | Fixed 640/1024 res, outputs RGBA layers |
| Qwen-Image-Edit-2511 | instruction editing | Qwen2.5-VL-7B (3584 dim) | 40 | Multi-image composition support |

## Architecture

```
Prompt -> Qwen3Formatter -> TextEncoder -> hidden_states[layer] -> DiT -> VAE -> Image
```

Text encoder extracts embeddings from LLM hidden states (default layer -2). DiT uses flow matching to generate latents. VAE decodes to RGB/RGBA.

## Quick Start

```bash
uv sync
```

```bash
# Z-Image (text-to-image)
uv run scripts/generate.py --model-path /path/to/z-image-turbo "A cat sleeping"

# Qwen-Image-Layered (decomposition)
uv run scripts/generate.py --model-type qwenimage \
  --qwen-image-model-path /path/to/Qwen-Image-Layered \
  --img2img input.jpg "Scene description"

# Web UI
uv run web/server.py --config config.toml
```

See [docs/reference/cli_flags.md](docs/reference/cli_flags.md) for full CLI reference.

## Features

**Quantization** (VRAM reduction):
- BitsAndBytes: `4bit` NF4 (~75%), `8bit` INT8 (~50%)
- TorchAO: `fp8` dynamic (~50%, RTX 4090+), `int8` weight-only (~50%)

**Generation**:
- Vision Conditioning via Qwen3-VL (zero-shot style transfer)
- Skip Layer Guidance for improved anatomy
- DyPE for high-resolution (2K-4K)
- Long prompt compression (4 modes for >1504 tokens)
- LoRA with multi-stack support

**Backends**:
- Attention: Flash Attention 2/3, SageAttention, xFormers, SDPA (auto-detect)
- Text Encoder: local (transformers), remote API, vLLM
- Distributed: encode on Mac, generate on CUDA

**Configuration**:
- TOML-based with hardware profiles
- Web UI config management (edit params, switch profiles, restart server)
- Modular component system
- CLI overrides

## Configuration

```bash
cp config.toml.example config.toml
uv run web/server.py --config config.toml --profile rtx4090
```

Key sections: `[encoder]`, `[generation]`, `[qwen_image]`, `[vl]`, `[rewriter]`

See [config.toml.example](config.toml.example) for all options.

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Z-Image generation |
| `/api/qwen-image/decompose` | POST | Image decomposition |
| `/api/qwen-image/edit` | POST | Instruction editing |
| `/api/vl/generate` | POST | Vision-conditioned generation |
| `/api/rewrite` | POST | Prompt expansion |
| `/api/config/session` | GET/PUT | Session config management |
| `/api/server/restart` | POST | Server restart with profile |

See [docs/reference/api_endpoints.md](docs/reference/api_endpoints.md) for full reference.

## Experiments

Ablation sweeps and comparison tools in `experiments/`. Interactive viewer on port 7861.

See [experiments/README.md](experiments/README.md).

## Documentation

**Models**:
- [Z-Image](docs/models/z_image.md) - performance tuning, device placement
- [Qwen-Image-Layered](docs/models/qwen_image_layered.md) - decomposition details
- [Qwen-Image-Edit-2511](docs/models/qwen_image_edit_2511.md) - instruction editing

**Guides**:
- [Config Management](docs/guides/config_management.md) - web UI config editing
- [VL Conditioning](docs/guides/vl_conditioning.md) - vision-based style transfer
- [LoRA](docs/guides/lora.md) - loading and fusing
- [Distributed](docs/guides/distributed.md) - multi-machine setup
- [Profiler](docs/guides/profiler.md) - performance testing

**Reference**:
- [CLI Flags](docs/reference/cli_flags.md) - all command-line options
- [API Endpoints](docs/reference/api_endpoints.md) - REST API
- [Configuration](docs/reference/configuration.md) - TOML structure
- [Web Architecture](docs/reference/web_architecture.md) - modular JS/CSS structure
- [DyPE](docs/reference/dype.md) - high-resolution generation
- [Long Prompts](docs/reference/long_prompts.md) - token compression

**Internal**: [CLAUDE.md](CLAUDE.md) for development reference.
