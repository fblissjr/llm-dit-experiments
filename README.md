# llm-dit-experiments

Multi-pipeline LLM-DiT generation platform. LLM hidden states -> flow-matching DiT -> VAE decode. Single GPU (24GB).

**Backend:** PyTorch, FastAPI, TOML config. **Frontend:** React 19, Vite 7, Bun (`web/frontend-v2/`).

## Pipelines

| Pipeline | Task | Encoder | Notes |
|----------|------|---------|-------|
| FLUX.2 Klein | text-to-image, image editing | Qwen3-8B/4B | Distilled, multi-layer extraction, LoRA support |
| Z-Image | text-to-image, img2img | Qwen3-4B | CFG=0 baked, 1504 token limit |
| LTX-2 | text-to-video | Gemma3-12B | Pure PyTorch, FP8, persistent component caching |
| Qwen-Image-2512 | text-to-image | Qwen2.5-VL-7B | 39GB transformer, requires fp8 on 24GB |
| Qwen-Image-Edit-2511 | image editing, multi-image | Qwen2.5-VL-7B | Multi-image composition, instruction editing |

## Quick Start

### 1. Backend

```bash
uv sync
cp config.toml.example config.toml   # edit model paths
uv run web/server.py --config config.toml
```

API on port 7860.

### 2. Frontend

```bash
cd web/frontend-v2
bun install
bun run dev
```

UI on `http://localhost:5175`. Vite proxies `/api` to the backend.

### 3. CLI (optional)

```bash
# Requires server running (step 1)
uv run scripts/gen.py flux2 --prompt "A photo of a cat" --seed 42
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Z-Image generation |
| `/api/flux2/generate` | POST | FLUX.2 generation |
| `/api/ltx2/generate/stream` | POST | LTX-2 video (streaming) |
| `/api/qwen-image/edit-layer` | POST | Single image editing |
| `/api/qwen-image/edit-multi` | POST | Multi-image composition |
| `/api/qwen-image-2512/generate` | POST | Qwen-Image T2I |
| `/api/models/{id}/load` | POST | Load pipeline |
| `/api/models/{id}/unload` | POST | Unload pipeline |
| `/api/loras` | GET | List LoRAs |
| `/api/config/session` | GET/PUT | Session config |
| `/api/rewrite` | POST | Prompt expansion |
| `/health` | GET | Health check |

## Experiments

Ablation sweeps and comparison tools in `experiments/`. See [experiments/README.md](experiments/README.md).

## Reference

- [Configuration](docs/reference/configuration.md) -- TOML config, hardware profiles, HTTPS
- [CLI flags](docs/reference/cli_flags.md) -- all command-line options
- [API endpoints](docs/reference/api_endpoints.md) -- request/response reference
- [Quantization](docs/reference/quantization.md) -- methods, tradeoffs, backends
- [config.toml.example](config.toml.example) -- annotated example
