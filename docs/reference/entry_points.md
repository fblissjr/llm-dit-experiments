# entry points reference

*last updated: 2026-03-03*

This document describes the three ways to invoke generation and how they relate to each other.

## overview

| Entry Point | File | Use Case | Status |
|-------------|------|----------|--------|
| **CLI-over-API** | `scripts/gen.py` | Batch generation, scripting, automation | Active |
| **Web API** | `web/server.py` | Interactive use, frontend, integrations | Active |
| **Legacy CLI** | `scripts/generate.py` | CLI-only features (embedding precompute, encoder-only) | Deprecated (v0.9.17) |

> **`scripts/generate.py` is deprecated.** Use `scripts/gen.py` instead. The legacy script calls pipeline functions directly and has diverged from the API (~50-90% parity depending on pipeline). It will be removed in v1.0.

## scripts/gen.py (CLI-over-API)

Thin httpx client that talks to the running server. Uses the same API endpoints and request schemas that E2E tests validate.

```
argparse subcommands
    |
    v
Request body builder (args -> dict, omit None for resolve_param precedence)
    |
    v
httpx POST to server endpoint
    |
    v
Response handler (JSON, SSE, or raw PNG)
    |
    v
Output writer (save file, print metadata)
```

### subcommands

| Subcommand | Endpoint | Response Format |
|------------|----------|-----------------|
| `status` | `GET /api/context` | JSON (server status) |
| `flux2` | `POST /api/flux2/generate` | JSON (ImageGenerationResult) |
| `flux2 --stream` | `POST /api/flux2/generate/stream` | SSE |
| `zimage` | `POST /api/generate` | JSON (ImageGenerationResult) |
| `zimage --stream` | `POST /api/generate/stream` | SSE |
| `ltx2` | `POST /api/ltx2/generate/stream` | SSE (always streaming) |
| `qwen` | `POST /api/qwen-image-2512/generate` | Raw PNG bytes |

### usage

```bash
# Check server status
uv run scripts/gen.py status

# FLUX.2 image generation
uv run scripts/gen.py flux2 --prompt "a cat sleeping in sunlight" --seed 42

# FLUX.2 with streaming progress
uv run scripts/gen.py flux2 --prompt "a cat" --stream

# Z-Image generation
uv run scripts/gen.py zimage --prompt "a mountain" --width 512 --height 512

# LTX-2 video (always streaming)
uv run scripts/gen.py ltx2 --prompt "ocean waves" --num-frames 33 --seed 42

# Qwen-Image T2I
uv run scripts/gen.py qwen --prompt "a bird" --seed 42

# Custom server URL and output directory
uv run scripts/gen.py --server http://localhost:9000 --output /tmp/gen/ flux2 --prompt "test"

# JSON output instead of saving files
uv run scripts/gen.py --json flux2 --prompt "test"
```

### global flags

| Flag | Default | Description |
|------|---------|-------------|
| `--server URL` | `http://127.0.0.1:7860` | Server base URL |
| `--output PATH` | `outputs/gen/` | Output directory for saved files |
| `--timeout SECONDS` | `300` | Request timeout |
| `--no-save` | off | Print metadata only, don't save file |
| `--json` | off | Output raw JSON response |

### design

- CLI arg names map directly to Pydantic schema field names (hyphens to underscores)
- None-valued args are omitted from the request body to preserve `resolve_param()` precedence (client > config.toml > schema default)
- Pre-flight health check before generation (warns if no pipeline loaded)
- TTY-aware: progress bar in terminal, line-per-event when piped

## Web API path

```
HTTP POST /api/<pipeline>/generate
    |
    v
web/schemas.py: Pydantic model validates request body
    |
    v
web/routers/<pipeline>.py: endpoint handler
    |
    v
ModelManager: manages model lifecycle (load/cache/unload)
    |
    v
Pipeline function called with merged params (request + runtime defaults)
    |
    v
HTTP response (JSON with base64 image or SSE stream)
```

**Key characteristics:**
- Models persist in memory between requests (ModelManager cache)
- Config resolution: `RuntimeConfig` defaults merged with per-request overrides from JSON body
- Output: HTTP response (base64-encoded image, SSE stream for progress)
- Full model lifecycle management (hot-reload, LoRA fusion tracking, VRAM monitoring)

## legacy CLI path (deprecated)

```
scripts/generate.py
    |
    v
cli.py: create_base_parser() -> parse args
    |
    v
cli.py: load_runtime_config(args) -> RuntimeConfig
    |
    v
Pipeline function called directly (fresh model load every time)
    |
    v
Output saved to file
```

**Key characteristics:**
- Models are loaded fresh for each invocation (no persistence between runs)
- Config resolution: `config.toml` -> `RuntimeConfig` -> CLI flag overrides
- Output: files written to disk
- No model management layer -- loads and unloads within the script

**Still needed for:**
- Embedding precompute
- Encoder-only mode
- Distributed encoding

These features will be migrated to the API in a future version.

## key differences

| Aspect | gen.py (CLI-over-API) | Web API | generate.py (Legacy) |
|--------|----------------------|---------|---------------------|
| Model loading | Server-managed | Server-managed | Fresh each run |
| Feature parity | 100% (same API) | 100% | 50-90% |
| LoRA support | Full | Full | Limited |
| Prompt upsampling | Yes | Yes | No |
| Streaming progress | Yes (SSE) | Yes (SSE) | No |
| Model switching | Via API | Hot-swap | Restart required |
| Two-stage (LTX-2) | Yes | Yes | No |

## FLUX.2 request lifecycle (Web API)

Detailed trace of what happens when a client sends `POST /api/flux2/generate`:

```
 1. Client sends POST /api/flux2/generate with JSON body
 2. FastAPI validates body against Flux2GenerateRequest (Pydantic)
 3. flux2.py router: _ensure_correct_model() checks model name + LoRA specs
    - If model mismatch: triggers reload via ModelManager
    - If LoRA mismatch: reloads fresh model, re-fuses LoRA weights
 4. _upsample_prompt() optionally rewrites prompt via heylookitsanllm
    - Calls external LLM API at config.rewriter_api_url
    - Two modes: T2I (creative expansion) or I2I (instruction compilation)
    - Graceful fallback to original prompt on error
 5. Router merges request params with RuntimeConfig defaults
 6. Router calls generate_image() from flux2_generate.py
 7. generate_image() loads encoder to GPU (pinned memory shuttle)
 8. Qwen3 encoder tokenizes + encodes prompt -> pooled + sequence embeddings
 9. Encoder offloads back to CPU (pinned memory)
10. Transformer runs denoising loop (denoise() for distilled, denoise_cfg() for base)
11. VAE decodes latents to pixel space
12. Router packages result as ImageGenerationResult (base64 PNG + metadata)
```

## startup flow (Web API)

```
uv run web/server.py --config config.toml
    |
    v
cli.py: create_base_parser() -> parse args
    |
    v
cli.py: load_runtime_config(args) -> RuntimeConfig
    |
    v
config.py: Config.from_toml() -> composed sub-configs
    |
    v
config.py: RuntimeConfig.from_toml_config(config) -> composed RuntimeConfig
    |
    v
server.py: stores as global runtime_config
    |
    v
model_manager.py: ModelManager(runtime_config) -> manages model lifecycle
    |
    v
server.py: _register_routers() -> 7 domain routers
    |
    v
uvicorn starts serving
```

## related docs

- [feature_parity_matrix.md](feature_parity_matrix.md) -- feature comparison across entry points
- [scripts_inventory.md](scripts_inventory.md) -- status of all scripts
- [configuration.md](configuration.md) -- config system reference
- [cli_flags.md](cli_flags.md) -- CLI flag reference
- [api_endpoints.md](api_endpoints.md) -- REST API reference
