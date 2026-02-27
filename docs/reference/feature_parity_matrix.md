# feature parity matrix

*last updated: 2026-02-14*

Comparison of features available through the CLI (`scripts/generate.py`) vs the Web API (`web/server.py` + routers).

> **Strategic direction:** CLI feature gaps will not be backported. The planned approach is a CLI-over-API tool that calls the Web API, giving automatic feature parity. See [entry_points.md](entry_points.md) for architecture details.

## FLUX.2 Klein

| Feature | CLI | Web API | Notes |
|---------|-----|---------|-------|
| Basic text-to-image | Yes | Yes | Both call `generate_image()` in `flux2_generate.py` |
| Image editing (reference images) | Yes | Yes | Same code path |
| LoRA loading + fusion | -- | Yes | Web-only: `_ensure_correct_model()` handles hot-swap |
| LoRA fusion tracking | -- | Yes | `FusedLoRAState` prevents re-fusion OOM |
| Prompt upsampling | -- | Yes | Web-only: `_upsample_prompt()` via heylookitsanllm |
| Streaming progress (SSE) | -- | Yes | Web-only: `generate_image_with_progress()` + SSE |
| Model persistence | -- | Yes | ModelManager keeps model in GPU memory between requests |
| Model variant switching | -- | Yes | Hot-swap between klein-4b, klein-9b, etc. |
| Fixed params validation | -- | Yes | Distilled models auto-override invalid guidance/steps |
| fp8 quantization | Yes | Yes | Both use `quantize_component()` |

## LTX-2

| Feature | CLI | Web API | Notes |
|---------|-----|---------|-------|
| Basic text-to-video | Yes | Yes | Both call `generate_video_with_offloading()` |
| Two-stage generation | -- | Yes | Web-only: `generate_video_two_stage()` |
| Embedding precompute | Yes | -- | CLI-only: distributed inference use case |
| Gemma3 bf16 encoder | Yes | Yes | |
| Gemma3 8bit encoder | Yes | Yes | |
| Gemma3 q4-qat encoder | Yes | Yes | |
| Streaming progress (SSE) | -- | Yes | Web-only |
| Frame count / FPS control | Yes | Yes | |
| Guidance scale | Yes | Yes | |

## Z-Image

| Feature | CLI | Web API | Notes |
|---------|-----|---------|-------|
| Basic text-to-image | Yes | Yes | Both call `ZImagePipeline.__call__()` |
| DyPE (high-resolution) | Yes | Yes | Full DyPE config in both paths |
| Skip Layer Guidance (SLG) | Yes | Yes | |
| Tiled VAE | Yes | Yes | |
| Prompt rewriting | -- | Yes | Web-only via rewriter API |
| Image-to-image | Yes | Yes | Same `img2img` path |
| Think block control | Yes | Yes | |
| Streaming progress (SSE) | -- | Yes | Web-only |
| Template system | Yes | Yes | |
| Long prompt compression | Yes | Yes | All 4 modes (truncate, interpolate, pool, attention_pool) |

**Parity note:** Z-Image has the best feature parity (~90%) because it was the first pipeline and the CLI was built alongside it.

## Qwen-Image Edit (2511)

| Feature | CLI | Web API | Notes |
|---------|-----|---------|-------|
| Single-image editing | -- | Yes | CLI has placeholder only (TODO) |
| Multi-image editing | -- | Yes | Web-only: `edit_multi()` |
| Edit model status check | -- | Yes | Web-only: `/api/qwen-image/edit-status` |
| Prompt rewriting | -- | Yes | Web-only via PromptRewriter |
| CPU offload | Yes (config) | Yes | |

## Qwen-Image T2I (2512)

| Feature | CLI | Web API | Notes |
|---------|-----|---------|-------|
| Text-to-image generation | Yes | Yes | Both call `QwenImage2512Pipeline.generate()` |
| fp8 quantization | Yes | Yes | |
| CPU offload | Yes | Yes | |
| Status check | -- | Yes | Web-only: `/api/qwen-image-2512/status` |
| Config inspection | -- | Yes | Web-only: `/api/qwen-image-2512/config` |

## cross-cutting features

| Feature | CLI | Web API | Notes |
|---------|-----|---------|-------|
| Config profiles | Yes | Yes | `--profile` flag / session config API |
| Hot-reload config | -- | Yes | Web-only: `PUT /api/config/session` |
| VRAM monitoring | -- | Yes | Web-only: `/api/vram/status` |
| Model load/unload | -- | Yes | Web-only: `/api/models/{id}/load` |
| Generation history | -- | Yes | Web-only: stored in frontend (IndexedDB, 500 items) |
| Health check | -- | Yes | Web-only: `/health` |
| Server restart | -- | Yes | Web-only: `/api/server/restart` |

## summary

| Pipeline | CLI Parity | Web-Only Features |
|----------|------------|-------------------|
| FLUX.2 Klein | ~50% | LoRA, prompt upsampling, streaming, model switching |
| LTX-2 | ~60% | Two-stage, streaming |
| Z-Image | ~90% | Streaming, rewriting |
| Qwen-Image Edit | ~10% | Nearly all features are web-only |
| Qwen-Image T2I | ~80% | Status/config endpoints |
