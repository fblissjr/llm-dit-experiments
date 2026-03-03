# feature parity matrix

*last updated: 2026-03-03*

Comparison of features available through entry points.

> **`scripts/generate.py` is deprecated.** Use `scripts/gen.py` (CLI-over-API) instead. gen.py has automatic 100% feature parity with the Web API since it calls the same endpoints.

## entry points

| Entry Point | File | Parity | Notes |
|-------------|------|--------|-------|
| **gen.py** | `scripts/gen.py` | 100% | CLI-over-API -- same endpoints, same schemas |
| **Web API** | `web/server.py` + routers | 100% | Source of truth |
| **generate.py** | `scripts/generate.py` | 50-90% | Deprecated -- calls pipelines directly |

## FLUX.2 Klein

| Feature | gen.py | Web API | generate.py (deprecated) |
|---------|--------|---------|--------------------------|
| Basic text-to-image | Yes | Yes | Yes |
| Image editing (reference images) | Yes | Yes | Yes |
| LoRA loading + fusion | Yes | Yes | -- |
| LoRA fusion tracking | Yes | Yes | -- |
| Prompt upsampling | Yes | Yes | -- |
| Streaming progress (SSE) | Yes (`--stream`) | Yes | -- |
| Model persistence | Yes (server) | Yes | -- |
| Model variant switching | Yes | Yes | -- |
| Fixed params validation | Yes | Yes | -- |
| fp8 quantization | Yes | Yes | Yes |

## LTX-2

| Feature | gen.py | Web API | generate.py (deprecated) |
|---------|--------|---------|--------------------------|
| Basic text-to-video | Yes | Yes | Yes |
| Two-stage generation | Yes | Yes | -- |
| Embedding precompute | -- | -- | Yes (CLI-only) |
| Gemma3 encoder variants | Yes | Yes | Yes |
| Streaming progress (SSE) | Yes (always) | Yes | -- |
| Frame count / FPS control | Yes | Yes | Yes |
| Guidance scale | Yes | Yes | Yes |
| LoRA support | Yes | Yes | -- |
| FBCache block skipping | Yes | Yes | -- |
| Distilled sigma mode | Yes | Yes | -- |
| Audio generation | Yes | Yes | -- |

## Z-Image

| Feature | gen.py | Web API | generate.py (deprecated) |
|---------|--------|---------|--------------------------|
| Basic text-to-image | Yes | Yes | Yes |
| DyPE (high-resolution) | -- | Yes | Yes |
| Skip Layer Guidance (SLG) | -- | Yes | Yes |
| Tiled VAE | -- | Yes | Yes |
| Prompt rewriting | -- | Yes | -- |
| Image-to-image | -- | Yes | Yes |
| Think block control | -- | Yes | Yes |
| Streaming progress (SSE) | Yes (`--stream`) | Yes | -- |
| Template system | Yes | Yes | Yes |
| Long prompt compression | -- | Yes | Yes |
| LoRA support | Yes | Yes | -- |

**Note:** gen.py exposes the most commonly used Z-Image parameters. Advanced features (DyPE, SLG, img2img) are available through the Web API directly or can be added to gen.py as needed.

## Qwen-Image Edit (2511)

| Feature | gen.py | Web API | generate.py (deprecated) |
|---------|--------|---------|--------------------------|
| Single-image editing | -- | Yes | -- |
| Multi-image editing | -- | Yes | -- |
| Edit model status check | -- | Yes | -- |

**Note:** Qwen-Image Edit requires multi-part file uploads. Not yet in gen.py -- use the Web API directly.

## Qwen-Image T2I (2512)

| Feature | gen.py | Web API | generate.py (deprecated) |
|---------|--------|---------|--------------------------|
| Text-to-image generation | Yes | Yes | Yes |
| fp8 quantization | Yes (server) | Yes | Yes |
| CPU offload | Yes (server) | Yes | Yes |
| Status check | `gen.py status` | Yes | -- |

## cross-cutting features

| Feature | gen.py | Web API | generate.py (deprecated) |
|---------|--------|---------|--------------------------|
| Config profiles | Yes (server) | Yes | Yes |
| Hot-reload config | Yes (server) | Yes | -- |
| VRAM monitoring | `gen.py status` | Yes | -- |
| Model load/unload | Automatic (lazy) | Yes | -- |
| Generation history | -- | Yes | -- |
| Health check | Yes (pre-flight) | Yes | -- |

## summary

| Pipeline | gen.py | generate.py (deprecated) |
|----------|--------|--------------------------|
| FLUX.2 Klein | 100% | ~50% |
| LTX-2 | 100% (minus precompute) | ~60% |
| Z-Image | ~70% (common params) | ~90% |
| Qwen-Image Edit | 0% (needs file upload) | ~10% |
| Qwen-Image T2I | 100% | ~80% |

gen.py achieves 100% parity for the generation APIs it wraps. The gap in Z-Image is only for advanced params not yet exposed as CLI flags -- the server still handles them. Qwen-Image Edit needs multi-part upload support which is planned separately.
