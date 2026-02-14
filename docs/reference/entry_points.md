# entry points reference

*last updated: 2026-02-14*

This document describes the two ways to invoke generation and how they relate to each other.

## overview

The platform has two entry points that both call the same underlying pipeline functions:

| Entry Point | File | Use Case |
|-------------|------|----------|
| **CLI** | `scripts/generate.py` | Batch generation, scripting, experiments |
| **Web API** | `web/server.py` | Interactive use, frontend, integrations |

> **Deprecation notice:** `scripts/generate.py` is planned for deprecation. The strategic direction is a CLI tool that calls the Web API, making the API the single source of truth for generation logic. See [feature_parity_matrix.md](feature_parity_matrix.md) for current gaps.

## shared boundary

Both entry points ultimately call the same pipeline functions:

| Pipeline | Shared Function | File |
|----------|----------------|------|
| FLUX.2 Klein | `generate_image()`, `generate_image_with_progress()` | `src/llm_dit/pipelines/flux2_generate.py` |
| LTX-2 | `generate_video_with_offloading()`, `generate_video_two_stage()` | `src/llm_dit/pipelines/generate.py` |
| Z-Image | `ZImagePipeline.__call__()` | `src/llm_dit/pipelines/z_image.py` |
| Qwen-Image Edit | `QwenImageDiffusersPipeline.edit_layer()`, `.edit_multi()` | `src/llm_dit/pipelines/qwen_image_diffusers.py` |
| Qwen-Image T2I | `QwenImage2512Pipeline.generate()` | `src/llm_dit/pipelines/qwen_image_2512.py` |

## CLI path

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

## key differences

| Aspect | CLI | Web API |
|--------|-----|---------|
| Model loading | Fresh each run | Persistent (ModelManager cache) |
| Config source | config.toml + CLI flags | RuntimeConfig + request JSON |
| Output format | File on disk | HTTP response (JSON/SSE) |
| LoRA support | Limited | Full (fusion tracking, hot-swap) |
| Prompt upsampling | No | Yes (FLUX.2 via heylookitsanllm) |
| Streaming progress | No | Yes (SSE) |
| Model switching | Restart required | Hot-swap via API |
| VRAM management | Manual | Automatic (load/unload endpoints) |
| Two-stage generation | No (LTX-2) | Yes (LTX-2) |

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
