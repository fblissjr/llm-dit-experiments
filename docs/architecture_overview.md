# architecture overview

*last updated: 2026-03-03*

High-level architecture of the LLM-DiT multi-model generation platform. For detailed post-refactor internals, see [internal/docs/architecture/post_refactor_guide.md](../internal/docs/architecture/post_refactor_guide.md).

## system overview

A multi-model image and video generation platform running on a single RTX 4090 (24GB VRAM). Supports 5 pipeline variants across 4 model families:

| Pipeline | Task | Encoder | Model |
|----------|------|---------|-------|
| **FLUX.2 Klein** | text-to-image, editing | Qwen3-8B/4B | 4B/9B distilled, persistent in GPU |
| **LTX-2** | text-to-video | Gemma3-12B | 19B DiT, fresh load per session |
| **Z-Image** | text-to-image | Qwen3-4B | Custom turbo/base DiT |
| **Qwen-Image Edit (2511)** | image editing | Qwen2.5-VL-7B | 8B instruction-following DiT |
| **Qwen-Image T2I (2512)** | text-to-image | Qwen2.5-VL-7B | 60-layer DiT |

Two entry points: CLI (`scripts/gen.py`, a thin client over the Web API) and Web API (`web/server.py`). The legacy `scripts/generate.py` is deprecated. See [docs/reference/entry_points.md](reference/entry_points.md) for detailed comparison.

## server architecture

### server.py (~296 lines)

After v0.7-v0.9 refactoring, server.py holds only:
- **Server state globals:** `runtime_config`, `model_manager`, `generation_history`, `encoder_only_mode`, `rewriter_backend`
- **Startup logic:** `main()` -- parses CLI, loads config, creates ModelManager, registers routers, starts uvicorn
- **Static file serving:** Mounts `web/frontend-v2/dist/`

All API endpoints live in 7 domain routers under `web/routers/`. No pipeline globals remain in server.py.

### domain routers (web/routers/)

| Router | File | Endpoints | Responsibility |
|--------|------|-----------|----------------|
| core | `core.py` | 12 | Z-Image generate, encode, img2img, DyPE, templates, rewriting |
| flux2 | `flux2.py` | 4 | FLUX.2 generate (sync + SSE), status, model metadata |
| ltx2 | `ltx2.py` | 2 | LTX-2 video generation (SSE), status |
| qwen_image | `qwen_image.py` | 6 | Qwen-Image edit (single + multi), T2I generate, status, config |
| vram | `vram.py` | 7 | Model load/unload, VRAM status, LoRA listing |
| config_mgmt | `config_mgmt.py` | 10 | Pipeline schemas, presets, session config, profiles, resolution |
| system | `system.py` | 7 | Health, context, restart, history, cache/FMTT cleanup |

**48 total endpoints** across 7 routers.

### dependency injection

All routers use `ConfigDep` and `ManagerDep` from `web/dependencies.py`:

```python
@router.post("/api/flux2/generate")
async def generate(request: Flux2GenerateRequest, config: ConfigDep, manager: ManagerDep):
    pipeline = manager.get_pipeline("flux2")
    ...
```

Routers that need server state (generation_history, encoder_only_mode) do `import web.server as srv` at module level. flux2.py and config_mgmt.py use only dependency injection.

### circular import prevention

Server imports routers in `_register_routers()` called from `main()`, NOT at module level. Routers import server at module level for state access. Never move router imports to top of server.py.

## config system

### composed RuntimeConfig

```
config.toml  -->  Config (composed dataclasses)  -->  RuntimeConfig.from_toml_config()
                                                           |
CLI args  ------------------------------------------>  _apply_cli_overrides()
                                                           |
                                                     RuntimeConfig (composed sub-configs)
```

RuntimeConfig composes pipeline-specific sub-configs:
- `config.flux2` -> `Flux2Config`
- `config.ltx2` -> `LTX2Config`
- `config.zimage` -> `ZImageConfig`
- `config.qwen_image` -> `QwenImageConfig`
- `config.encoder` -> `EncoderConfig`
- `config.optimization` -> `OptimizationConfig`
- `config.quant` -> `PipelineQuantConfig`

Adding a new parameter requires only **2 touchpoints**: the dataclass field in `config.py` + the TOML section in `config.toml`. Validated by `tests/unit/test_dry_config.py`.

## request lifecycle

### HTTP POST to image response (FLUX.2 example)

```
 1. Client sends POST /api/flux2/generate (JSON body)
 2. FastAPI validates body against Flux2GenerateRequest (Pydantic)
 3. Router: _ensure_correct_model() checks model name + LoRA specs
 4. Router: _upsample_prompt() optionally rewrites prompt via heylookitsanllm
 5. Router merges request params with RuntimeConfig defaults
 6. Pipeline: generate_image() in flux2_generate.py
 7. Encoder shuttle: loads to GPU via pinned memory DMA
 8. Qwen3 tokenizes + encodes prompt -> embeddings
 9. Encoder shuttle: offloads back to CPU
10. Transformer: denoising loop (denoise() or denoise_cfg())
11. VAE: decodes latents to pixel space
12. Router: packages ImageGenerationResult (base64 PNG + metadata)
```

### hot-reload vs restart

Config changes via `PUT /api/config/session` are classified by `HOT_RELOAD_SAFE` and `REQUIRES_RESTART` constants in `model_manager.py`:

- **Hot-reload safe:** shift, d_noise, steps, guidance_scale, width, height, hidden_layer, DyPE/SLG params
- **Requires restart:** model_path, device placements, quantization, LoRA, attention_backend, compile

## model lifecycle

### ModelManager (sole source of truth)

All model loading, unloading, and reloading goes through `ModelManager` in `model_manager.py`. No pipeline state exists outside of it.

```python
manager.load_flux2(config)      # -> dict (pipeline components)
manager.get_pipeline("flux2")   # -> dict or None
manager.is_loaded("flux2")      # -> bool
manager.unload_flux2()          # -> frees VRAM
manager.reload_flux2(model_name)  # -> atomic reload for model switching
```

### model persistence patterns

| Pattern | Used By | Description |
|---------|---------|-------------|
| **Persistent dict** | FLUX.2 | Models stay in GPU memory between requests. LoRA fusion tracked via `FusedLoRAState`. |
| **On-demand load** | LTX-2, Qwen-Image | Models loaded when needed, unloaded when switching. |
| **Encoder shuttle** | FLUX.2 encoder | Encoder loads to GPU for encoding, offloads to CPU between requests. Uses pinned memory for fast DMA transfers. |

### LoRA fusion tracking (FLUX.2)

`FusedLoRAState` attached to `model._fused_lora_state` tracks what LoRA weights are fused into persistent models. Prevents re-fusion OOM where fp8 (9GB) dequantizes to bf16 (18GB) during fusion.

## quantization

All pipelines use a single entry point:

```python
from llm_dit.quantization.torchao_utils import quantize_component
quantize_component(model, method="fp8-weight-only")
```

torchao is the sole backend. Valid methods: `none`, `fp8-dynamic`, `fp8-weight-only`, `int8`, `int4`. See [docs/reference/quantization.md](reference/quantization.md).

## frontend

React 19 + Zustand 5 + Vite 7 + Bun application in `web/frontend-v2/`:

- **Schema-driven forms:** Backend returns `ParamSchema` lists, frontend renders controls automatically
- **Zustand stores:** `appStore` (pipelines, presets), `formStore` (per-pipeline state), `sessionStore` (history)
- **Persist middleware:** All three stores persist via IndexedDB (`utils/idbStorage.ts`), falls back to localStorage
- **Media system:** Unified `MediaItem` type with `detectKind()`/`mediaItemFromResult()`/`mediaItemFromHistory()` utilities
- **OpenAPI codegen:** `bun run export-openapi && bun run gen-api` generates TypeScript types from the API spec
- **Build:** `cd web/frontend-v2 && bun run build` -> served from `dist/`

## strategic direction

The Web API is the authoritative path for generation. `scripts/gen.py` is the primary CLI -- a thin HTTP client that calls the running server's API endpoints. This makes the API the single source of truth and eliminates the feature parity problem that existed with the old `scripts/generate.py` (now deprecated). See [docs/reference/feature_parity_matrix.md](reference/feature_parity_matrix.md) for current gaps.

```bash
# gen.py requires the server to be running
uv run web/server.py --config config.toml &

# Then use gen.py as the CLI
uv run scripts/gen.py flux2 --prompt "A sunset over mountains" --seed 42
uv run scripts/gen.py zimage --prompt "A cat" --width 1024 --height 1024
uv run scripts/gen.py ltx2 --prompt "Ocean waves" --num-frames 33
uv run scripts/gen.py qwen --prompt "A bird"
uv run scripts/gen.py status
```

## related docs

| Doc | Purpose |
|-----|---------|
| [reference/entry_points.md](reference/entry_points.md) | CLI vs Web API comparison |
| [reference/feature_parity_matrix.md](reference/feature_parity_matrix.md) | Feature parity across entry points |
| [reference/api_endpoints.md](reference/api_endpoints.md) | REST API reference |
| [reference/configuration.md](reference/configuration.md) | Config system reference |
| [reference/cli_flags.md](reference/cli_flags.md) | CLI flags reference |
| [internal/docs/architecture/post_refactor_guide.md](../internal/docs/architecture/post_refactor_guide.md) | Detailed post-refactor internals |
| [internal/docs/architecture/codebase_map.md](../internal/docs/architecture/codebase_map.md) | Full codebase map |
