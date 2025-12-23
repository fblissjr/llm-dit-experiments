# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Skip Layer Guidance (SLG) for improved structure/anatomy in generated images
  - New `src/llm_dit/guidance/` module with `SkipLayerGuidance` class
  - Hook-based layer skipping with context manager pattern
  - Integrated into Z-Image pipeline with `skip_layer_guidance_scale`, `skip_layer_indices` parameters
  - CLI support: `--slg-scale`, `--slg-layers`, `--slg-start`, `--slg-stop`
  - Config sections: `[*.slg]` with enabled, scale, layers, start, stop
  - Web UI: Collapsible SLG section with scale slider, layer presets, timing controls
  - `/api/generation-config` returns SLG defaults from server config
  - Startup logging shows SLG status alongside DyPE
  - Z-Image optimized settings: scale=2.5, layers=[7,8,9,10,11,12], range 5-50% of steps
  - Wider range needed for turbo-distilled model (8-9 steps with shift 3.0-7.0)
  - Based on StabilityAI SD3.5 and Spatio-Temporal Guidance paper
- Training infrastructure for LoRA and full model fine-tuning
  - New `src/llm_dit/training/` module with complete training pipeline
  - `ZImageTrainingModule` for supervised fine-tuning (SFT) and distillation
  - `FlowMatchSFTLoss` with Gaussian timestep weighting
  - `TrainingDataset` supporting JSON/JSONL/CSV metadata formats
  - Gradient checkpointing with optional CPU offload
  - PEFT-based LoRA injection for efficient fine-tuning
  - Training runner using HuggingFace Accelerate (distributed, mixed precision)
  - `ModelLogger` for checkpoint saving with trainable-only export
  - Training script: `scripts/train.py` with full CLI interface
  - FlowMatchScheduler extended with `training_target()` and `training_weight()` methods
  - Optional dependencies: `peft`, `pandas`, `tqdm`, `tensorboard` (install via `uv pip install -e ".[training]"`)
- DyPE (Dynamic Position Extrapolation) for high-resolution generation beyond training dimensions
  - Vision YaRN implementation for position embedding extrapolation
  - Per-axis scaling with timestep-dependent modulation
  - Parameter sweep script: `experiments/scripts/sweep_dype_params.sh`
- Qwen-Image-Layered feature complete with decomposition and layer editing
  - New pipeline: `QwenImageDiffusersPipeline` wrapping official diffusers implementation
  - Image decomposition: Split images into 2-10 RGBA layers
  - Layer editing: Modify individual layers with text instructions using Qwen-Image-Edit-2509
  - Edit model lazy-loads on first use (auto-downloads from HuggingFace)
  - CLI support: `--qwen-image-model-path`, `--qwen-image-edit-model-path`, `--qwen-image-cpu-offload`, `--qwen-image-layers`, `--qwen-image-steps`, `--qwen-image-cfg-scale`, `--qwen-image-resolution`
  - Web UI: Decomposition with layer grid, edit modal for each layer, replace original functionality
  - REST API: `/api/qwen-image/decompose`, `/api/qwen-image/edit-layer`, `/api/qwen-image/edit-status`
  - Config section: `[*.qwen_image]` with model_path, edit_model_path, cpu_offload, layer_num, num_inference_steps, cfg_scale, resolution
  - Fixed resolutions only: 640x640 (recommended) or 1024x1024
  - CPU offload enabled by default (~5 GB VRAM usage)
  - History entries tagged with `model_type` for filtering
  - User guide: `docs/qwen_image_guide.md`
  - Technical reference: `internal/research/qwen_image_technical_report.md`
- Legacy custom implementation preserved: `QwenImagePipeline`, `QwenImageTextEncoderBackend`, `QwenImageVAE`, `QwenImageDiT`
- Caption length study experiment scripts
  - `sweep_caption_fill_modes.sh` - Compare padding strategies (pad_both, pad_left, pad_right) at fixed length 600
  - `sweep_caption_lengths.sh` - Test embedding lengths from 50 to 1504 tokens in steps of 50
  - `sweep_caption_hidden_layer.sh` - Sweep Qwen3-4B hidden layers (-2, -6, -8, -12, -16, -21)
  - `sweep_caption_vl.sh` - Compare VL embeddings vs text encoding, test token modes (text-only vs full)
  - `run_all_caption_sweeps.sh` - Master script to run all caption experiments with unified flags
  - All scripts support `--quick`, `--dry-run`, `--config`, `--profile` flags
  - Master script includes `--skip-vl` option to skip VL experiments
  - Scripts use `=` syntax for negative number arguments (e.g., `--hidden-layers="-2,-6"`)
- Comprehensive test suite for Qwen-Image
  - Unit tests: 59 tests covering latent packing, DiT, VAE, pipeline, and backend
  - Integration test: `tests/integration/test_qwen_diffusers_wrapper.py` for full pipeline validation
- Hidden Layer vs CFG interaction experiments
  - Tests whether CFG > 0 helps when using non-default hidden layers (OOD for distillation)
  - Hypothesis: middle layers (-10 to -18) may benefit from small CFG (1.5-2.5) to compensate for distribution mismatch
  - New experiments: `hidden_layer_cfg_grid`, `hidden_layer_cfg_quick`, `hidden_layer_cfg_shift_grid`
  - Sweep script: `experiments/sweep_hidden_layer_cfg.sh` with `--quick`, `--with-shift` options

### Fixed
- CUDA OOM from dtype parameter mismatch: diffusers uses `torch_dtype`, transformers uses `dtype`
  - Diffusers silently ignores unknown kwargs, causing float32 loading (2x memory)
  - Fixed all diffusers calls to use `torch_dtype`, transformers calls to use `dtype`
- SLG config not loading from TOML: Added `SLGConfig` to main `Config` class
  - `[profile.slg]` sections now properly parsed and applied
  - Startup now logs SLG status: `SLG: enabled (scale=X, layers=[...], range=[X%, Y%])`
- SLG "Could not find 'blocks'" error: Z-Image transformer uses `layers` attribute
  - Fixed `fqn="blocks"` to `fqn="layers"` in pipeline initialization
- RGBA to RGB conversion in `edit_layer()` - edit model VAE expects 3-channel input, now properly extracts alpha before editing and reapplies it after
- Unit test parameter mismatch - fixed `quantization` to use correct `text_encoder_quantization` and `dit_quantization` parameter names
- DyPE "cannot assign module before Module.__init__() call" error
  - `ZImageDyPERoPE` assigned `self.original_embedder` before `super().__init__()`
  - PyTorch's nn.Module requires parent init before attribute assignment
  - Fixed initialization order: extract values to local vars, call super().__init__(), then assign attributes
- Scheduler shift parameter ignored: diffusers `FlowMatchEulerDiscreteScheduler` has read-only `shift` property
  - `set_timesteps(mu=...)` was ignored when `use_dynamic_shifting=False`
  - Fixed to use `set_shift()` method for diffusers scheduler, direct assignment for our FlowMatchScheduler
  - Applied in `img2img()`, `__call__()`, and `generate_from_embeddings()` methods
- Attention backend benchmark not using diffusers attention dispatch
  - Benchmark only set our `attention_forward()` backend, not diffusers transformer attention
  - Fixed by using `attention_backend()` context manager from `diffusers.models.attention_dispatch`
  - Added config compile setting reading (previously hardcoded `--compile default`)
  - Added SAGE/xFormers incompatibility check for torch.compile
- Missing `cfg_normalization` and `cfg_truncation` in `GenerationConfig`
  - Config file had these parameters but dataclass was missing them
  - Fixed by adding both fields with defaults (0.0 and 1.0 respectively)

## [0.6.0]

### Added
- Vision conditioning using Qwen3-VL (zero-shot, no training required)
  - New `src/llm_dit/vl/` module with `VLEmbeddingExtractor` class
  - Embedding blending utilities: `blend_embeddings`, `scale_embeddings`, `blend_style_only`, `blend_per_token`, `blend_attention_weighted`
  - Web UI section with image upload, alpha slider, blend mode selector
  - REST API endpoints: `/api/vl/status`, `/api/vl/config`, `/api/vl/extract`, `/api/vl/generate`, `/api/vl/cache`
  - CLI flags: `--vl-model-path`, `--vl-device`, `--vl-alpha`, `--vl-hidden-layer`, `--vl-blend-mode`
  - Config section: `[default.vl]` with model_path, device, default_alpha, etc.
  - Embeddings caching for efficient multi-generation workflows
  - Blend modes: linear, style_only (preserves text content), graduated, attention_weighted
  - Research documentation in `experiments/qwen3_vl/` and `internal/research/vl_conditioning_hypotheses.md`
- VL Rewriter feature for vision-based prompt rewriting
  - Three model options: qwen3-4b (text-only), qwen3-vl (local VL), qwen3-vl-api (VL via remote API)
  - Image upload support for VL models in rewriter UI
  - API VL rewriting via heylookitsanllm for larger VL models (e.g., qwen2.5-vl-72b-mlx)
  - Three input modes: text-only, image-only, or combined image+text
  - CLI flags: `--rewriter-vl-api-model`, `--rewriter-timeout`
  - Config options: `rewriter.vl_api_model`, `rewriter.timeout` (default: 120.0 seconds)
  - Enhanced error handling with specific messages for timeout, HTTP errors, and connection failures
  - `/api/rewriter-models` endpoint now includes VL API option when configured
  - `/api/rewrite` endpoint accepts `model: "qwen3-vl-api"` for API-based VL rewriting

### Changed
- Config dataclasses include `VLConfig` for vision conditioning settings
- RuntimeConfig includes VL fields (vl_model_path, vl_device, vl_alpha, etc.)
- Web server loads Qwen3-VL on startup if configured
- `config.toml.example` includes VL sections for all profiles
- CLAUDE.md updated with VL documentation, CLI flags, and API endpoints

## [0.5.0]

### Added
- Experiment comparison tools for visual analysis of ablation studies
  - CLI tool (`experiments/compare.py`) for generating comparison images
    - `--list` to discover experiments from `results/`
    - `--mode grid` generates NxM grid (prompts x variable values)
    - `--mode side-by-side` for horizontal pair comparison
    - `--mode diff` for pixel difference visualization (highlight/absolute/heatmap)
  - Comparison module (`experiments/compare/`) with reusable components
    - `discovery.py` - Auto-discover experiments from timestamped directories
    - `grid.py` - PIL-based grid generation with metric overlays
    - `diff.py` - Image difference calculations (3 visualization modes)
    - `models.py` - `ExperimentImage`, `ExperimentRun`, `ComparisonSpec` dataclasses
  - Web viewer (`experiments/viewer/`) on port 7861
    - Standalone FastAPI server (no model loading)
    - Grid View - NxM thumbnail grid
    - Slider - Draggable divider between two images
    - A/B Toggle - Click to swap between images
    - Diff Overlay - Interactive pixel difference visualization

### Changed
- Documentation structure updated
  - `experiments/README.md` now includes comparison tools section
  - `CLAUDE.md` Directory Structure includes `experiments/` tree
  - `README.md` includes Experiments quick start section
  - `internal/log/` replaces `internal/LOG.md` (dated log files)

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
- Long prompt handling with experimental compression modes
  - CLI: `--long-prompt-mode truncate|interpolate|pool|attention_pool`
  - Config: `[default.pytorch] long_prompt_mode = "truncate"`
  - `truncate`: Cuts off at 1504 tokens (loses end of prompt)
  - `interpolate`: Resamples embeddings via linear interpolation
  - `pool`: Compresses via adaptive average pooling
  - `attention_pool`: Importance-weighted pooling (preserves key tokens)
- Image-to-image generation pipeline (`pipe.img2img()`)
  - Strength parameter controls noise level (0.0 = no change, 1.0 = full regeneration)
- Gradient checkpointing support for transformer
  - `pipe.enable_gradient_checkpointing(True)` for reduced VRAM during fine-tuning
- Config profiles in `config.toml.example`
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
  - DiT transformer has max text sequence length of 1504 tokens (discovered via binary search)
  - Config shows 1536 but fails at boundary - likely off-by-one in diffusers RoPE
  - Prompts exceeding limit use `long_prompt_mode` (default: interpolate)
  - `MAX_TEXT_SEQ_LEN` constant exported for programmatic access (1504)
  - Compression applied in txt2img, img2img, encode_prompt, generate_from_embeddings

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
