# llm-dit-experiments

PyTorch and Diffusers-based experimentation platform for LLM-DiT image and video generation. Pluggable backends, quantization, LoRA fusion, and a React frontend for research on a single GPU.

## Pipelines

| Pipeline | Task | Encoder | Steps | Notes |
|----------|------|---------|-------|-------|
| FLUX.2 Klein | text-to-image, image editing | Qwen3-8B/4B (12288/7680 dim) | 4 | Distilled, multi-layer extraction, configurable text encoding |
| Z-Image | text-to-image, img2img | Qwen3-4B (2560 dim) | 8-9 | CFG=0 baked, 1504 token limit |
| LTX-2 | text-to-video | Gemma3-12B (3840 dim) | 15-40 | Pure PyTorch impl, FP8 quantization |
| Qwen-Image-Layered | image decomposition | Qwen2.5-VL-7B (3584 dim) | 50 | Fixed 640/1024 res, outputs RGBA layers |
| Wan Video | text-to-video | UMT5-XXL | 50 | Phase 1 integration |

## Architecture

```
                          config.toml
                              |
                              v
Client -> React UI -> FastAPI (7 domain routers) -> ModelManager -> Pipeline -> Output
                              |
                              v
                    Prompt -> Encoder -> hidden_states[layer] -> DiT -> VAE -> Image/Video
```

Text encoder extracts embeddings from LLM hidden states (default layer -2). DiT uses flow matching to generate latents. VAE decodes to RGB/RGBA. ModelManager handles load/unload/reload lifecycle for all pipelines. Routers live in `web/routers/` (core, flux2, ltx2, qwen_image, vram, config_mgmt, system).

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

# Z-Image (text-to-image)
uv run scripts/generate.py --model-path /path/to/z-image-turbo "A cat sleeping"

# LTX-2 (text-to-video with explicit device placement)
uv run scripts/generate.py --model-type ltx2 \
    --ltx2-model-path /path/to/LTX-2 \
    --ltx2-text-encoder-device cpu \
    --ltx2-transformer-device cuda \
    --ltx2-quantize fp8 \
    --ltx2-num-frames 33 --width 768 --height 512 \
    "A cat walking through a sunny garden"

# Web UI (HTTP)
uv run web/server.py --config config.toml

# Web UI (HTTPS)
uv run web/server.py --config config.toml \
    --ssl-certfile /path/to/cert.pem --ssl-keyfile /path/to/key.pem
```

See [docs/reference/cli_flags.md](docs/reference/cli_flags.md) for full CLI reference.

## Features

**Quantization** (unified torchao, VRAM reduction):
- `fp8-dynamic`: FP8 weights + activations (~50%, RTX 4090+)
- `fp8-weight-only`: FP8 weights, BF16 compute (~50%, compile-safe)
- `int8`: INT8 weight-only (~50%, any GPU)
- `int4`: INT4 weight-only (~75%, max compression)

**Generation**:
- LoRA with multi-stack support and fusion tracking (prevents re-fusion OOM on persistent models)
- DyPE for high-resolution (2K-4K)
- Long prompt compression (4 modes for >1504 tokens)

**Backends**:
- Attention: Flash Attention 2/3, SageAttention, xFormers, SDPA (auto-detect)
- Text Encoder: local (transformers), remote API, vLLM
- Distributed: encode on Mac, generate on CUDA

**Configuration**:
- TOML-based with hardware profiles
- HTTPS via SSL certificates (uvicorn-native)
- Web UI config management (edit params, switch profiles, restart server)
- CLI overrides for all config fields

## Configuration

```bash
cp config.toml.example config.toml
uv run web/server.py --config config.toml --profile rtx4090
```

Key sections: `[server]`, `[encoder]`, `[generation]`, `[quantization]`, `[rewriter]`

HTTPS:
```toml
[server]
host = "0.0.0.0"
port = 7860
ssl_certfile = "/path/to/cert.pem"
ssl_keyfile = "/path/to/key.pem"
```

See [config.toml.example](config.toml.example) for all options.

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Z-Image generation |
| `/api/flux2/generate` | POST | FLUX.2 generation (text-to-image, editing) |
| `/api/ltx2/generate/stream` | POST | LTX-2 video generation (streaming) |
| `/api/qwen-image/decompose` | POST | Image decomposition to RGBA layers |
| `/api/models/{id}/load` | POST | Load pipeline by ID |
| `/api/models/{id}/unload` | POST | Unload pipeline by ID |
| `/api/loras` | GET | List available LoRAs |
| `/api/config/session` | GET/PUT | Session config management |
| `/api/rewrite` | POST | Prompt expansion |
| `/health` | GET | Health check |

See [docs/reference/api_endpoints.md](docs/reference/api_endpoints.md) for full reference.

## Experiments

Ablation sweeps and comparison tools in `experiments/`. Interactive viewer on port 7861.

See [experiments/README.md](experiments/README.md).

## Documentation

**Models**:
- [Z-Image](docs/models/z_image.md) - performance tuning, device placement
- [LTX-2](docs/models/ltx2.md) - video generation with pure PyTorch pipeline
- [Qwen-Image-Layered](docs/models/qwen_image_layered.md) - decomposition details

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
- [Quantization](docs/reference/quantization.md) - torchao backend details
- [DyPE](docs/reference/dype.md) - high-resolution generation
- [Long Prompts](docs/reference/long_prompts.md) - token compression

**Internal**: [CLAUDE.md](CLAUDE.md) for development reference.
