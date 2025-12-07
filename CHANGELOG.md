# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0]

### Added
- PyTorch-native components (Phase 1 migration from diffusers)
  - Priority-based attention backend selector (Flash Attention 3 > FA2 > Sage > xFormers > SDPA)
    - CLI: `--attention-backend auto|flash_attn_2|flash_attn_3|sage|xformers|sdpa`
    - Environment override: `LLM_DIT_ATTENTION=sdpa`
  - Pure PyTorch FlowMatchScheduler with Z-Image specific shift transformation
    - CLI: `--use-custom-scheduler`
    - Exact match to reference implementation (sigma' = shift * sigma / (1 + (shift-1) * sigma))
  - Tiled VAE decoder for 2K+ image generation without OOM
    - CLI: `--tiled-vae`, `--tile-size 512`, `--tile-overlap 64`
    - Linear blending mask for smooth tile transitions
  - Standalone Context Refiner module (2-layer transformer, no timestep modulation)
    - RMSNorm, RoPE embeddings, Gated SiLU FFN
    - Loadable from Z-Image checkpoint via `ContextRefiner.from_pretrained()`
    - Gradient checkpointing support via `enable_gradient_checkpointing()`
- Embedding cache for text encoder (DiffSynth optimization)
  - Thread-safe LRU cache with configurable max size
  - CLI: `--embedding-cache`, `--cache-size 100`
  - Config: `[default.pytorch] embedding_cache = true`
  - Cache statistics tracking (hit rate, evictions)
  - Avoids re-encoding identical prompts for batch generation
- Image-to-image generation pipeline (`pipe.img2img()`)
  - Strength parameter controls noise level (0.0 = no change, 1.0 = full regeneration)
- Gradient checkpointing support for transformer
  - `pipe.enable_gradient_checkpointing(True)` for reduced VRAM during fine-tuning
- Config profiles in `config.example.toml`
  - `[default]` - Conservative settings for most hardware
  - `[rtx4090]` - Optimized for 24GB VRAM (Flash Attention, caching)
  - `[low_vram]` - 8-16GB GPUs (CPU offload, quantization)
  - `[cpu_only]` - CPU inference only
  - `[distributed]` - Encoder on Mac, DiT on CUDA

### Changed
- RuntimeConfig now includes PyTorch-native fields (attention_backend, embedding_cache, etc.)
- Config loader processes `[default.pytorch]` section from TOML files
- Documentation updated with Phase 1 migration components and usage examples

### Fixed
- Long prompts no longer crash with `vectorized_gather_kernel: index out of bounds`
  - DiT transformer has max text sequence length of 1024 tokens (RoPE axes_lens[0])
  - Prompts exceeding limit are automatically truncated with warning
  - `MAX_TEXT_SEQ_LEN` constant exported for programmatic access
  - Truncation applied in txt2img, img2img, encode_prompt, generate_from_embeddings

## [0.3.0]

### Added
- Prompt rewriting using the loaded Qwen3 model
  - Uses same model for both embedding extraction and text generation (no extra memory)
  - Rewriter templates in `templates/z_image/rewriter/` category
  - Web UI: Collapsible "Prompt Rewriter" section with style selection
  - API: `GET /api/rewriters` lists available rewriter templates
  - API: `POST /api/rewrite` rewrites prompts using selected template
- Text generation support in backends
  - `TransformersBackend.generate()` for local text generation
  - `APIBackend.generate()` via `/v1/chat/completions` endpoint
  - `supports_generation` property on backends

### Changed
- `TransformersBackend` now uses `AutoModelForCausalLM` instead of `AutoModel`
  - Enables both embedding extraction (hidden states) and text generation
  - No change to embedding output format
- Template loader now searches subdirectories recursively by default
  - Pattern changed from `*.md` to `**/*.md`
  - Supports template categories in subdirectories (e.g., `rewriter/`)

## [0.2.0]

### Added
- Unified CLI architecture: web server and generate script share argument parser
- Scheduler shift parameter (`--shift`, default 3.0) for timestep schedule control
- LoRA support (`--lora path:scale`, stackable) with automatic weight fusion
- Device placement flags in generate.py (`--text-encoder-device`, `--dit-device`, `--vae-device`)
- Optimization flags in generate.py (`--cpu-offload`, `--flash-attn`, `--compile`)
- Prompt control flags in generate.py (`--system-prompt`, `--thinking-content`, `--assistant-content`)
- Config file sections: `[optimization]`, `[scheduler]`, `[lora]`
- Mobile-friendly web UI with all generation settings exposed
- New utility module `src/llm_dit/utils/lora.py` for LoRA loading
- Transformers v5 compatibility: `quantization_config` API support

### Changed
- Config file is now source of truth; CLI flags override config values
- Renamed `--no-thinking` to `--enable-thinking` (positive semantics)
- `torch_dtype` now honored from config (was hardcoded bfloat16)
- Web UI redesigned with collapsible sections and responsive layout
- **Breaking**: `EncoderConfig.load_in_8bit`/`load_in_4bit` deprecated in favor of `quantization` field
  - Use `quantization = "none" | "4bit" | "8bit"` instead
  - Legacy fields still work but emit deprecation warnings
  - Config automatically migrates to new API
- `TransformersBackend.from_pretrained()` now accepts `quantization_config` parameter
- `config.toml` updated: use `quantization = "8bit"` instead of `load_in_8bit = true`

### Fixed
- CLI/Web feature parity - all flags available in both interfaces
- Template formatter closing tag logic now matches ComfyUI-QwenImageWanBridge patterns exactly
- Fixed `<|im_end|>` handling: now only omits when assistant content is empty AND is_final=True

### Migration Guide (transformers v5)

The `load_in_8bit` and `load_in_4bit` parameters are deprecated in transformers v5.
Update your configuration as follows:

**config.toml (before):**
```toml
[default.encoder]
load_in_8bit = true
```

**config.toml (after):**
```toml
[default.encoder]
quantization = "8bit"  # or "4bit" or "none"
```

**Python code (before):**
```python
config = EncoderConfig(load_in_8bit=True)
```

**Python code (after):**
```python
config = EncoderConfig(quantization="8bit")

# Or use BitsAndBytesConfig directly:
from transformers import BitsAndBytesConfig
quant_config = BitsAndBytesConfig(load_in_8bit=True)
backend = TransformersBackend.from_pretrained(
    model_path,
    quantization_config=quant_config,
)
```

The legacy API still works but will emit deprecation warnings.

## [0.1.0]

### Added
- Initial release
- ZImageTextEncoder with template support (140+ templates)
- ZImagePipeline for end-to-end generation
- Distributed inference support (API backend for remote encoding)
- Per-component device placement (encoder/DiT/VAE independent)
- Web UI and REST API
- Generation history management
- Qwen3 chat template formatter with thinking block support
