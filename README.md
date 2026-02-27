# llm-dit-experiments

PyTorch experimentation platform for LLM-DiT image and video generation on a single GPU.

## Pipelines

| Pipeline | Task | Encoder | Notes |
|----------|------|---------|-------|
| FLUX.2 Klein | text-to-image, image editing | Qwen3-8B/4B | Distilled, multi-layer extraction, LoRA support |
| Z-Image | text-to-image, img2img | Qwen3-4B | CFG=0 baked, 1504 token limit |
| LTX-2 | text-to-video | Gemma3-12B | Pure PyTorch, FP8 quantization, persistent component caching |
| Qwen-Image-2512 | text-to-image | Qwen2.5-VL-7B | 39GB transformer, requires fp8 on 24GB |
| Qwen-Image-Edit-2511 | image editing, multi-image | Qwen2.5-VL-7B | Multi-image composition, instruction editing |

## Quick Start

```bash
uv sync
cp config.toml.example config.toml   # edit model paths
uv run web/server.py --config config.toml
```

Open `http://localhost:7860` -- the React UI auto-detects loaded pipelines.

CLI generation is also available:

```bash
uv run scripts/generate.py --model-type flux2 \
    --flux2-model-path /path/to/FLUX.2-klein-9b-fp8 \
    "A photo of a cat"
```

See [docs/reference/cli_flags.md](docs/reference/cli_flags.md) for full CLI reference.

## Features

- **Quantization:** fp8-dynamic, fp8-weight-only, int8, int4 (torchao for transformers, native fp8 layerwise casting for encoders)
- **LoRA:** multi-stack support with fusion tracking (prevents re-fusion OOM on persistent models)
- **Attention:** Flash Attention 2/3, SageAttention, xFormers, SDPA (auto-detect)
- **DyPE:** high-resolution generation (2K-4K)
- **Long prompts:** 4 compression modes for >1504 tokens
- **Text encoding:** local (transformers) or remote via [heylookitsanllm](http://github.com/fblissjr/heylookitsanllm)
- **Config management:** TOML-based with hardware profiles, live session editing, CLI overrides

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Z-Image generation |
| `/api/flux2/generate` | POST | FLUX.2 generation (text-to-image, editing) |
| `/api/ltx2/generate/stream` | POST | LTX-2 video generation (streaming) |
| `/api/qwen-image/edit-layer` | POST | Single image editing with instructions |
| `/api/qwen-image/edit-multi` | POST | Multi-image composition |
| `/api/qwen-image-2512/generate` | POST | Qwen-Image T2I generation |
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

## Reference

- [Configuration](docs/reference/configuration.md) -- TOML config, hardware profiles, HTTPS setup
- [CLI flags](docs/reference/cli_flags.md) -- all command-line options
- [API endpoints](docs/reference/api_endpoints.md) -- full request/response reference
- [Quantization](docs/reference/quantization.md) -- methods, tradeoffs, backend details
- [config.toml.example](config.toml.example) -- annotated example config
