# llm-dit-experiments

PyTorch and Diffusers-based (depending on the models / pipeline) experimentation platform for LLM-DiT image and video generation. Pluggable backends, quantization, and quality of life features for research.

## Pipelines

| Pipeline | Task | Encoder | Steps | Notes |
|----------|------|---------|-------|-------|
| FLUX.2 Klein | text-to-image, image editing | Qwen3-8B/4B (12288/7680 dim) | 4 | Distilled, multi-layer extraction, configurable text encoding |
| Z-Image | text-to-image, img2img | Qwen3-4B (2560 dim) | 8-9 | CFG=0 baked, 1504 token limit |
| LTX-2 | text-to-video | Gemma3-12B (3840 dim) | 15-40 | Pure PyTorch impl, FP8 quantization |
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
# FLUX.2 Klein (text-to-image with FP8 and block offload for 24GB GPU)
uv run scripts/generate.py --model-type flux2 \
    --flux2-model-name klein-9b-fp8 \
    --flux2-block-offload \
    --flux2-model-path /path/to/FLUX.2-klein-9b-fp8 \
    --flux2-vae-path /path/to/FLUX.2-klein-9B \
    "A photo of a cat"

# FLUX.2 Klein with longer prompts (configurable text encoding)
uv run scripts/generate.py --model-type flux2 \
    --flux2-model-name klein-9b-fp8 \
    --flux2-block-offload \
    --flux2-max-text-length 1024 \
    --flux2-model-path /path/to/FLUX.2-klein-9b-fp8 \
    --flux2-vae-path /path/to/FLUX.2-klein-9B \
    "A highly detailed description of a complex scene..."

# FLUX.2 Klein image editing with multiple references
uv run scripts/generate.py --model-type flux2 \
    --flux2-model-name klein-9b-fp8 \
    --flux2-block-offload \
    --flux2-model-path /path/to/FLUX.2-klein-9b-fp8 \
    --flux2-vae-path /path/to/FLUX.2-klein-9B \
    --flux2-input-image ref1.jpg ref2.jpg ref3.jpg \
    "Combine elements from the reference images"

# Z-Image (text-to-image)
uv run scripts/generate.py --model-path /path/to/z-image-turbo "A cat sleeping"

# LTX-2 (text-to-video) - Pure PyTorch pipeline
uv run scripts/generate.py --model-type ltx2 \
  --ltx2-model-path /path/to/LTX-2 \
  --ltx2-num-frames 33 --width 768 --height 512 \
  "A cat walking through a sunny garden"

  # LTX-2 (text-to-video) - PyTorch pipeline with explicit device placement
  uv run python scripts/generate.py --model-type ltx2 \
  --ltx2-model-path /path/to/LTX-2 \
      --ltx2-text-encoder-device cpu \
      --ltx2-transformer-device cuda \
      --ltx2-quantize fp8 \
      "A cat walking"

# Web UI
uv run web/server.py --config config.toml
```

See [docs/reference/cli_flags.md](docs/reference/cli_flags.md) for full CLI reference.

## Features

**Quantization** (unified torchao, VRAM reduction):
- `fp8-dynamic`: FP8 weights + activations (~50%, RTX 4090+)
- `fp8-weight-only`: FP8 weights, BF16 compute (~50%, compile-safe)
- `int8`: INT8 weight-only (~50%, any GPU)
- `int4`: INT4 weight-only (~75%, max compression)

**Generation**:
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
- [LTX-2](docs/models/ltx2.md) - video generation with pure PyTorch pipeline
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
