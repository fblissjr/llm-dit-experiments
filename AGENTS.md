# agent context

*last updated: 2026-01-17*

Quick reference for LLM agents working on this codebase.

## start here

| Doc | Purpose |
|-----|---------|
| **[internal/hub.md](internal/hub.md)** | Central documentation hub |
| **[internal/state/current.md](internal/state/current.md)** | Current project state (read first) |
| **[experiments/AGENTS.md](experiments/AGENTS.md)** | Research status tracking |

## research status system

All research documentation uses a consistent status tracking system:

| Symbol | Status | Meaning |
|--------|--------|---------|
| ✅ | **Validated** | Confirmed through experiments or architecture analysis |
| 🔬 | **Open** | Hypothesis needs testing or re-testing |
| ⚠️ | **Needs Verification** | Previous results may have bugs |
| 🚫 | **Dead-End** | Tested, doesn't work |

**Where to find status tracking:**
- [experiments/AGENTS.md](experiments/AGENTS.md) - Top-level research navigation
- [experiments/ltx2/docs/findings/](experiments/ltx2/docs/findings/) - Consolidated research findings

## critical rules

- **no emojis** in code, docs, or output
- **use `uv`** for all Python ops (`uv add`, `uv run`, `uv sync`)
- **never commit** without explicit user approval
- **dtype conventions** - libraries differ:
  - transformers: use `dtype=`
  - diffusers: use `torch_dtype=`
- **max tokens: 1504** (z-image DiT RoPE limit, LTX-2 differs)

## architecture

```
Text Prompt -> TextEncoder -> hidden_states[layer(s)] -> DiT -> VAE -> Image/Video
```

Models use different text encoders (Qwen3-4B, Gemma3-12B, UMT5-XXL) and DiT variants. See [internal/models/overview.md](internal/models/overview.md) for comparison.

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

### ltx-2 (video, active development)

| param | value | notes |
|-------|-------|-------|
| encoder | Gemma3-12B (Q4) | 3840 hidden dim, 49 layers |
| transformer | LTX-2 19B | 48 blocks, 32 heads |
| cfg | 3.5-4.0 | with latent normalization |
| steps | 50 | standard |
| resolution | 512-1280 | video generation |
| frames | 121-369 | per ComfyUI workflow |
| rope | 3D INTERLEAVED | (T, H, W) positions |

**Status:** Pure PyTorch port complete. Verification tests pending.

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
| pipeline | `src/llm_dit/pipelines/z_image.py`, `qwen_image.py`, `ltx2.py`, `wan_video.py` |
| config | `src/llm_dit/config.py`, `cli.py` |
| encoder | `src/llm_dit/encoders/z_image_encoder.py`, `gemma3.py` |
| models | `src/llm_dit/models/ltx2_transformer.py`, `humo_transformer.py` |
| router | `src/llm_dit/router/token_layer_router.py` |
| experiments | `experiments/ltx2/` |
| web | `web/server.py`, `static/js/`, `static/css/` |
| tests | `tests/unit/`, `tests/integration/` |

## navigation

### primary (read first)
- **[hub.md](internal/hub.md)** - central documentation hub
- **[current.md](internal/state/current.md)** - current project state
- [models/](internal/models/) - per-model knowledge base

### session state
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
