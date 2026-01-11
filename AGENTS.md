# agent context

*last updated: 2026-01-11*

Quick reference for LLM agents working on this codebase.

## critical rules

- **no emojis** in code, docs, or output
- **use `uv`** for all Python ops (`uv add`, `uv run`, `uv sync`)
- **never commit** without explicit user approval
- **dtype conventions** - libraries differ:
  - transformers: use `dtype=`
  - diffusers: use `torch_dtype=`
- **max tokens: 1504** (DiT RoPE limit)

## architecture

```
Text Prompt -> Qwen3Formatter -> TextEncoderBackend -> hidden_states[layer] -> DiT -> VAE -> Image
```

Text encoder extracts embeddings from Qwen3-4B hidden states (default layer -2). DiT uses flow matching to generate latents, VAE decodes to images.

## key parameters

### z-image (turbo distilled)

| param | value | notes |
|-------|-------|-------|
| encoder | Qwen3-4B | 2560 hidden dim, 36 layers |
| layer | -2 | penultimate (`--hidden-layer`) |
| max tokens | 1504 | DiT RoPE limit |
| cfg | 0.0 | baked in (Decoupled-DMD) |
| steps | 8-9 | turbo distilled |
| scheduler | FlowMatchEuler | shift=3.0 |

### qwen-image-layered

| param | value | notes |
|-------|-------|-------|
| encoder | Qwen2.5-VL-7B | 3584 hidden dim |
| cfg | 4.0 | required |
| steps | 50 | non-distilled |
| resolution | 640/1024 | fixed |

### wan (humo video)

| param | value | notes |
|-------|-------|-------|
| transformer | HuMo-17B | 5120 hidden, 40 blocks |
| encoder | UMT5-XXL | 4096 hidden dim |
| vae | Wan2.1 | 8x spatial, 4x temporal |
| input channels | 36 | 16 noise + 16 image + 4 extra |
| output channels | 16 | latent dimension |
| audio | Whisper-large-v3 | optional, for audio-sync |

**Status:** Phase 1 complete (transformer + pipeline structure). Phase 2 pending (VAE + text encoder + scheduler integration).

## key files

| area | files |
|------|-------|
| pipeline | `src/llm_dit/pipelines/z_image.py`, `qwen_image.py`, `wan_video.py` |
| config | `src/llm_dit/config.py`, `cli.py` |
| encoder | `src/llm_dit/encoders/z_image_encoder.py` |
| models | `src/llm_dit/models/humo_transformer.py`, `wan_vae.py` |
| web | `web/server.py`, `static/js/`, `static/css/` |
| tests | `tests/unit/`, `tests/integration/` |

## navigation

### session state
- [session continuity](internal/state/session_continuity.md) - current focus, blockers, next steps
- [todos](internal/state/todos.md) - pending work across sessions
- [lessons learned](internal/state/lessons_learned.md) - aggregated insights

### architecture
- [guiding principles](internal/principles/guiding_principles.md) - architectural north star
- [full docs index](internal/index.md) - complete navigation map

### reference docs
- [cli flags](docs/reference/cli_flags.md) - all CLI arguments
- [api endpoints](docs/reference/api_endpoints.md) - REST API reference
- [configuration](docs/reference/configuration.md) - DRY config principles

## common cli

```bash
# generation
uv run scripts/generate.py --model-path /path/to/z-image "A cat sleeping"

# with lora
uv run scripts/generate.py --model-path /path/to/z-image --lora style.safetensors:0.8 "Prompt"

# high-res (DyPE)
uv run scripts/generate.py --model-path /path/to/z-image --dype --width 2048 --height 2048 "Prompt"

# wan video (when complete)
uv run scripts/generate.py --model-type wan --wan-humo-path ~/Storage/HuMo/HuMo-17B --wan-base-path ~/Storage/Wan2.1-T2V-1.3B "A man singing"

# web server
uv run web/server.py --config config.toml --profile default

# tests
uv run pytest tests/
```

## adding parameters

Config flows through single chain:
```
config.toml -> Config dataclass -> RuntimeConfig -> startup.py -> Backend configs
```

See `docs/reference/configuration.md` for checklist.
