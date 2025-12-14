# llm-dit-experiments

This file provides guidance to Claude Code when working with this repository.

## Project Overview

A standalone diffusers-based experimentation platform for LLM-DiT image generation, starting with Z-Image (Alibaba/Tongyi). Designed for hobbyist exploration with:
- Pluggable LLM backends (transformers, API/heylookitsanllm, vLLM, SGLang)
- Rich template system (140+ templates from ComfyUI)
- Distributed inference support (encode on Mac, generate on CUDA)
- Web UI and REST API for generation
- Granular device placement (encoder/DiT/VAE on CPU/GPU independently)
- LoRA support with automatic weight fusion
- TOML config file support with CLI overrides

## Critical Rules

- **No emojis** in code, docs, or output
- **Use `uv`** for all Python operations (`uv add`, `uv run`, `uv sync`)
- **Never commit** without explicit user approval
- **Semantic versioning** in CHANGELOG.md (no dates)
- **Update documentation** after any feature or change (see checklist below)

## Documentation Requirements

**After implementing any feature or significant change, update these files:**

| File | When to Update |
|------|----------------|
| `CHANGELOG.md` | Every feature, fix, or breaking change (semantic versioning) |
| `README.md` | New user-facing features, quick start examples |
| `CLAUDE.md` | New directories, CLI flags, architecture changes, technical details |
| `pyproject.toml` | New dependencies, version bumps (keep version in sync with CHANGELOG) |
| `config.toml.example` | New configurable parameters (with comments) |
| `experiments/README.md` | Experiment-related features or tools |
| `internal/log/LOG_YYYY-MM-DD.md` | Every session (create dated file) |
| `docs/*.md` | Feature-specific documentation |

**Checklist for new features:**
1. Add entry to `CHANGELOG.md` under appropriate version
2. Update `README.md` if user-facing
3. Update `CLAUDE.md` Directory Structure if new files/directories
4. Update relevant `docs/*.md` or `experiments/README.md`
5. Create/update `internal/log/LOG_YYYY-MM-DD.md` with session details

**For new configurable parameters (CRITICAL - follow DRY Principles below):**
1. Add to `config.toml.example` with descriptive comment
2. Add to Config dataclass in `src/llm_dit/config.py`
3. Add CLI argument in `src/llm_dit/cli.py` (`create_argument_parser()`)
4. Add to `RuntimeConfig` in `src/llm_dit/cli.py`
5. Wire in `load_runtime_config()` (TOML -> RuntimeConfig)
6. Wire in `src/llm_dit/startup.py` (RuntimeConfig -> Backend configs)
7. Expose in `web/server.py` and `web/index.html` if user-facing
8. Run DRY config test: `uv run pytest tests/unit/test_dry_config.py -v`

**Claude Code instruction:** After completing a feature implementation, always ask "Should I update the documentation?" or proactively update the files listed above. For any new parameters, follow the DRY Configuration Principles section and run the automated test.

## DRY Configuration Principles

All configurable parameters must flow through a single chain to prevent disconnected settings:

```
config.toml (TOML)     CLI flags (argparse)
        \                    /
         v                  v
     Config dataclass  →  RuntimeConfig
              \              /
               v            v
            PipelineLoader / startup.py
                    |
                    v
            Backend configs (APIBackendConfig, etc.)
                    |
                    v
            Actual usage (API requests, model loading)
```

**When adding a new parameter:**

1. **Add to TOML config** (`config.toml.example`) in appropriate section
2. **Add to Config dataclass** (`src/llm_dit/config.py`) - e.g., `EncoderConfig`, `RewriterConfig`
3. **Add CLI argument** (`src/llm_dit/cli.py`) in `create_argument_parser()`
4. **Add to RuntimeConfig** (`src/llm_dit/cli.py`) with same name
5. **Wire in load_runtime_config()** - load from TOML config, allow CLI override
6. **Wire in startup.py** - pass to backend configs (`APIBackendConfig`, etc.)
7. **Expose in web UI** if user-facing (web/index.html, server endpoints)
8. **Document in CLAUDE.md** - CLI flags table, config sections

**Files to check when adding parameters:**

| Layer | File | What to update |
|-------|------|----------------|
| TOML schema | `config.toml.example` | Add parameter with comment |
| Config classes | `src/llm_dit/config.py` | Add to dataclass, `to_dict()` |
| CLI parser | `src/llm_dit/cli.py` | `create_argument_parser()` |
| Runtime config | `src/llm_dit/cli.py` | `RuntimeConfig`, `load_runtime_config()` |
| Pipeline loading | `src/llm_dit/startup.py` | Pass to backend configs |
| API backend | `src/llm_dit/backends/api.py` | `APIBackendConfig` if API-relevant |
| Web server | `web/server.py` | Endpoint parameters |
| Web UI | `web/index.html` | Form fields if user-facing |
| Documentation | `CLAUDE.md` | CLI flags, config examples |

**Anti-pattern to avoid:** Adding a parameter to CLI but not wiring it through `startup.py` to the actual backend that uses it (e.g., `hidden_layer` must reach `APIBackendConfig`).

**Automated verification:** Run the DRY configuration consistency test after adding any new parameter:

```bash
uv run pytest tests/unit/test_dry_config.py -v
```

This test verifies:
- TOML parameters exist in Config dataclasses
- CLI arguments map to RuntimeConfig fields
- Critical parameters are wired through to backend configs
- Key parameters are documented

**Claude Code instruction:** Always run this test after adding or modifying configuration parameters to catch wiring gaps early.

## Architecture

```
Text Prompt
    |
    v
Qwen3Formatter (chat template with thinking blocks)
    |
    v
TextEncoderBackend (transformers/vllm/sglang)
    |
    v
hidden_states[layer] -> embeddings (2560 dim)  [layer=-2 default, configurable]
    |
    v
[diffusers handles: DiT context_refiner -> main layers -> VAE decode]
    |
    v
Image Output
```

## Key Technical Details (Z-Image)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Text encoder | Qwen3-4B | 2560 hidden dim, 36 layers |
| Embedding extraction | hidden_states[-2] | Penultimate layer (configurable via `--hidden-layer`) |
| **Max text tokens** | **1504** | DiT RoPE limit (see below) |
| CFG scale | 0.0 | Baked in via Decoupled-DMD |
| Steps | 8-9 | Turbo distilled |
| Scheduler | FlowMatchEuler | shift=3.0 |
| VAE | 16-channel | Wan-family |
| Context refiner | 2 layers | No timestep modulation |

## Hidden Layer Selection

The default embedding extraction layer is **-2** (penultimate, layer 35 of 36). This is configurable via `--hidden-layer` (-35 to -1).

**SFT Impact Hypothesis:** Qwen3-4B is instruction-tuned, which modifies later layers to be good at "helpful assistant" responses. This may make late layers (-1 to -10) overly abstract for image generation, losing concrete visual details. Middle layers (~-15 to -21) may provide better prompt adherence because they:
- Have completed semantic processing
- Haven't been heavily overwritten by SFT objectives
- Retain more pre-training knowledge about visual concepts

**Recommendations:**
- **Default (-2)**: Safe choice, works reasonably well
- **Middle layers (-15 to -21)**: Try if prompt details aren't being captured
- **Web UI**: Use the layer selector to experiment

See `internal/research/hidden_layer_selection.md` for detailed analysis and experimental plans.

## Text Sequence Length Limits

The DiT transformer has a **maximum text sequence length of 1504 tokens**. The config specifies `axes_lens=[1536, 512, 512]` but the actual working limit is 1504 (47 * 32, where 32 is `axes_dims[0]`). This appears to be an off-by-one in RoPE frequency table indexing. Exceeding 1504 causes CUDA kernel crashes.

**This is a key research area for this repository.** We are actively experimenting with methods to work around or extend this limit. See `internal/research/long_prompt_research.md` for detailed research notes on approaches, quality tradeoffs, and future directions.

### Current Behavior (Default)

- Prompts exceeding 1504 tokens are **automatically truncated** with a warning
- The pipeline logs: `Text sequence length (N tokens) exceeds maximum (1504). Truncating to 1504 tokens.`
- Truncation happens after encoding, preserving the first 1504 tokens of embeddings

### Experimental Compression Modes

We have implemented several experimental modes for handling prompts beyond 1504 tokens. **Quality impact varies - this is an active research area:**

| Mode | CLI Flag | Status | Quality Impact |
|------|----------|--------|----------------|
| `truncate` | `--long-prompt-mode truncate` | Stable | Predictable, loses tail content |
| `interpolate` | `--long-prompt-mode interpolate` | **Default** | Preserves all content, smooth resampling |
| `pool` | `--long-prompt-mode pool` | Experimental | Under evaluation |
| `attention_pool` | `--long-prompt-mode attention_pool` | Experimental | Cosine similarity weighting |

**Research questions we're investigating:**
- At what compression ratio does quality noticeably degrade?
- Does `attention_pool` preserve "important" tokens better than uniform methods?
- Can we achieve good results at 2x or 3x compression?
- Are there prompt structures that compress better than others?

See the detailed research doc: `internal/research/long_prompt_research.md`

### Token Count Guidelines

| Content Type | Approximate Tokens |
|--------------|-------------------|
| Simple prompt ("A cat sleeping") | 10-20 tokens |
| Detailed prompt with style | 50-100 tokens |
| Template + system prompt | 100-200 tokens |
| Full format (system + think + assistant) | 150-300 tokens |
| Maximum safe prompt | ~1200-1300 tokens |

**Note:** The 1504 limit includes ALL tokens: system prompt, user prompt, think block, and assistant content.

### Strategies for Long Prompts

1. **Omit system prompt**: Default templates add ~50-100 tokens
   ```bash
   # Skip system prompt to save tokens
   uv run scripts/generate.py --model-path ... "Your long detailed prompt here"
   ```

2. **Skip think block**: Empty think block adds ~10 tokens
   ```bash
   # Default: no think block (recommended for long prompts)
   uv run scripts/generate.py --model-path ... "Long prompt"

   # vs. with think block (uses more tokens)
   uv run scripts/generate.py --model-path ... --force-think-block "Long prompt"
   ```

3. **Use concise descriptions**: Focus on key visual elements
   ```
   # Instead of:
   "A highly detailed photograph of a beautiful serene peaceful calm quiet mountain landscape with..."

   # Use:
   "Mountain landscape, serene, golden hour, detailed, 8k"
   ```

4. **Check token count before generation**:
   ```python
   from llm_dit import ZImageTextEncoder

   encoder = ZImageTextEncoder.from_pretrained("/path/to/model")
   output = encoder.encode("Your prompt here")
   token_count = output.token_counts[0]
   print(f"Token count: {token_count}/1504")
   ```

5. **Access the constant programmatically**:
   ```python
   from llm_dit import MAX_TEXT_SEQ_LEN
   print(f"Max tokens: {MAX_TEXT_SEQ_LEN}")  # 1504
   ```

6. **Use experimental compression modes** (active research area):
   ```bash
   uv run scripts/generate.py --long-prompt-mode interpolate "Very long prompt..."
   uv run scripts/generate.py --long-prompt-mode pool "Very long prompt..."
   uv run scripts/generate.py --long-prompt-mode attention_pool "Very long prompt..."
   ```

   See `internal/research/long_prompt_research.md` for detailed analysis of each mode.

### Why This Limit Exists

The Z-Image DiT uses multi-axis RoPE for position encoding:
- Axis 0 (1504 actual, 1536 configured): Text/time sequence positions
- Axis 1 (512): Image height positions
- Axis 2 (512): Image width positions

The config specifies `axes_lens=[1536, 512, 512]` and `axes_dims=[32, 48, 48]`. The actual limit is 1504 = 47 * 32, suggesting an off-by-one in the RoPE frequency table indexing. Exceeding 1504 causes CUDA kernel errors (`vectorized_gather_kernel: index out of bounds`).

**Note**: This limit is architectural, not a fundamental constraint. Future research directions include RoPE extrapolation, hierarchical encoding, and chunked attention. See the research doc for exploration plans.

## Chat Template Format (Qwen3-4B)

### Official HuggingFace Space Format (Default)

The official Z-Image HuggingFace Space uses this format by default:

```
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
```

**No think block by default.** This matches calling `tokenizer.apply_chat_template(enable_thinking=True)`.

### Full Format (All Components)

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
<think>
{thinking_content}
</think>

{assistant_content}
```

### Content-Driven Think Block Logic

Our implementation uses **content-driven** logic to match the official HF Space while exposing full control:

| Condition | Result |
|-----------|--------|
| Default (no thinking_content, no force_think_block) | No think block (matches official) |
| `thinking_content` provided | Add think block with content |
| `force_think_block=True` | Add empty think block |

**Exact format when think block is enabled:**
- Empty think: `<think>\n\n</think>\n\n`
- With content: `<think>\n{content}\n</think>\n\n`
- With assistant: `<think>\n{content}\n</think>\n\n{assistant}<|im_end|>`

### Qwen3 Tokenizer Behavior (Reference)

Note: Qwen3's `enable_thinking` parameter has counterintuitive naming:
- `enable_thinking=True` -> NO think block (model CAN think on its own)
- `enable_thinking=False` -> ADD empty `<think>\n\n</think>\n\n` (skip thinking)

The official HF Space uses `enable_thinking=True` which produces NO think block.
Our `force_think_block` parameter provides explicit control without this confusion.

### API Components

All 4 components exposed via API:
- `prompt` (required) - User message
- `system_prompt` (optional) - System message
- `thinking_content` (optional) - Content inside `<think>...</think>` (triggers think block)
- `assistant_content` (optional) - Content after `</think>`
- `force_think_block` (optional) - If True, add empty think block even without content

## Qwen3 Sampler Settings

Qwen3 models have specific optimal sampling parameters documented in the [official model card](https://huggingface.co/Qwen/Qwen3-4B#best-practices).

### Two Use Cases

| Use Case | Description | Sampler Settings |
|----------|-------------|------------------|
| **Text Encoding** | Extract hidden states for image generation | N/A (forward pass only, no sampling) |
| **Text Generation** | Prompt rewriting via chat completions | Use settings below |

For text encoding (embedding extraction), sampler settings are **irrelevant** since we only run a forward pass through the model to extract hidden states - no token generation occurs.

### Thinking Mode (Default for Rewriting)

When the model can use `<think>...</think>` blocks for reasoning:

| Parameter | Value | Notes |
|-----------|-------|-------|
| temperature | 0.6 | DO NOT use 0 (greedy) - causes endless repetition |
| top_p | 0.95 | Nucleus sampling threshold |
| top_k | 20 | Top-k filtering |
| min_p | 0.0 | Disabled |
| presence_penalty | 0.0-2.0 | Optional, helps reduce repetition |

### Non-Thinking Mode

For Qwen3 Instruct models without thinking capability (e.g., Qwen3-4B-Instruct-2507):

| Parameter | Value | Notes |
|-----------|-------|-------|
| temperature | 0.7 | Slightly higher than thinking mode |
| top_p | 0.8 | Tighter than thinking mode |
| top_k | 20 | Same as thinking mode |
| min_p | 0.0 | Disabled |

### Token IDs (Qwen3-4B)

| Token | ID | Usage |
|-------|-----|-------|
| `<think>` | 151667 | Start thinking block |
| `</think>` | 151668 | End thinking block |
| `<|im_start|>` | 151644 | Chat template start |
| `<|im_end|>` | 151645 | Chat template end |

### Compatible Qwen3 Models

For Z-Image text encoding, the model must have **2560 hidden dimensions** (matching the DiT's expected input):

| Model | Hidden Dim | Compatible |
|-------|-----------|------------|
| Qwen3-4B | 2560 | Yes |
| Qwen3-4B-Instruct-2507 | 2560 | Yes (non-thinking only) |
| Qwen3-8B | 4096 | No |
| Qwen3-14B | 5120 | No |
| Qwen3-32B | 5120 | No |
| Qwen3-72B | 8192 | No |

### Configuration Examples

**TOML config (recommended):**
```toml
[default.rewriter]
temperature = 0.6       # Qwen3 thinking mode
top_p = 0.95
top_k = 20
min_p = 0.0
presence_penalty = 0.0  # Increase to 1.0-2.0 if seeing repetition
max_tokens = 512
```

**CLI flags:**
```bash
uv run web/server.py \
  --rewriter-temperature 0.6 \
  --rewriter-top-p 0.95 \
  --rewriter-top-k 20 \
  --rewriter-presence-penalty 0.0
```

## Directory Structure

```
src/llm_dit/
    backends/           # LLM backend abstraction (Protocol-based)
        protocol.py     # TextEncoderBackend Protocol, EncodingOutput
        config.py       # BackendConfig dataclass
        transformers.py # HuggingFace transformers backend
        api.py          # API backend (heylookitsanllm)
    conversation/       # Chat formatting
        types.py        # Message, Conversation dataclasses
        formatter.py    # Qwen3Formatter
    templates/          # Template loading system
        loader.py       # YAML frontmatter parsing
        registry.py     # Template caching
    encoders/           # Text encoding pipeline
        z_image.py      # ZImageTextEncoder
    pipelines/          # Diffusion pipeline wrappers
        z_image.py      # ZImagePipeline (txt2img, img2img)
    schedulers/         # Pure PyTorch schedulers
        flow_match.py   # FlowMatchScheduler (shifted sigma schedule)
    models/             # Pure PyTorch model components
        context_refiner.py  # ContextRefiner (2-layer text processor)
    utils/              # Utility modules
        lora.py         # LoRA loading and fusion
        attention.py    # Priority-based attention backend selector
        tiled_vae.py    # TiledVAEDecoder for 2K+ images
    vl/                 # Vision conditioning (Qwen3-VL)
        __init__.py     # Module exports
        qwen3_vl.py     # VLEmbeddingExtractor class
        blending.py     # Embedding blending utilities
    cli.py              # Shared CLI argument parser and config loading
    config.py           # Configuration dataclasses

scripts/
    generate.py         # CLI image generation script
    profiler.py         # Profiling and stability testing script

web/
    server.py           # FastAPI web server (history storage, all endpoints)
    index.html          # Web UI (mobile-friendly, history panel, resolution presets)

templates/z_image/      # 140+ prompt templates (markdown + YAML frontmatter)
config.toml             # Example configuration file

experiments/            # Ablation studies and evaluation tools
    run_ablation.py     # Automated experiment runner
    sweep_*.sh          # Priority sweep scripts
    compare.py          # CLI comparison tool (grids, diffs)
    compare/            # Comparison module
        discovery.py    # Auto-discover experiments from results/
        grid.py         # PIL-based grid generation
        diff.py         # Image difference calculations
    viewer/             # Web-based comparison viewer (port 7861)
        server.py       # FastAPI standalone viewer
        index.html      # Interactive comparison UI
    prompts/            # Standard evaluation prompts
    metrics/            # ImageReward, SigLIP scoring
    results/            # Generated images and logs
    qwen3_vl/           # Vision conditioning experiments
        README.md       # Feature documentation
        CONDITIONING_GUIDE.md   # Usage guide
        RESEARCH_FINDINGS.md    # Technical discoveries

docs/                   # User-facing documentation
    distributed_inference.md  # Running encoder on Mac, DiT on CUDA
    web_server_api.md   # REST API reference
    models/             # Model-specific documentation

internal/               # Development and maintainer documentation
    guides/             # Development guides
        DEVICE_PLACEMENT_GUIDE.md  # Device placement strategies
        TESTING_GUIDE.md           # Testing guidelines
    research/           # Research notes and technical investigations
        heylookitsanllm_hidden_states_spec.md
        decoupled_dmd_training_report.md
        z_image_*.md    # Model analysis and design docs
    reports/            # End-of-day reports
    log/                # Session logs (dated files)
    SESSION_CONTINUITY.md  # Session state tracking
    GUIDING_PRINCIPLES.md  # Architectural decisions
```

## Living Documentation

- **internal/log/**: Dated session logs with all changes and investigations
- **CHANGELOG.md**: Version history (semantic versioning)

## Configuration

Config file (TOML) is the source of truth. CLI flags override config values.

**Transformers v5 Note:** `load_in_8bit`/`load_in_4bit` are deprecated. Use `quantization` instead.

See [config.toml.example](config.toml.example)

## Web Server

```bash
# Install torch separately (not pinned in pyproject.toml)
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

# Sync other dependencies
uv sync

# Run web server with local encoder (RTX 4090 recommended)
uv run web/server.py \
  --model-path /path/to/z-image-turbo \
  --text-encoder-device cpu \
  --dit-device cuda \
  --vae-device cuda

# Run with config file
uv run web/server.py --config config.toml --profile default

# Run with API encoder (distributed inference)
uv run web/server.py \
  --model-path /path/to/z-image-turbo \
  --api-url http://mac-host:8080 \
  --api-model Qwen3-4B \
  --dit-device cuda \
  --vae-device cuda

# Run with LoRA
uv run web/server.py \
  --model-path /path/to/z-image-turbo \
  --lora /path/to/style.safetensors:0.8 \
  --dit-device cuda

# Enable debug logging for backend comparison
uv run web/server.py --model-path ... --debug
```

## CLI Script (scripts/generate.py)

```bash
# Basic generation
uv run scripts/generate.py \
  --model-path /path/to/z-image-turbo \
  --output image.png \
  "A cat sleeping in sunlight"

# With config file
uv run scripts/generate.py \
  --config config.toml \
  --profile default \
  "A sunset over mountains"

# With LoRA
uv run scripts/generate.py \
  --model-path /path/to/z-image \
  --lora style.safetensors:0.8 \
  "An anime character"

# Full control (with thinking content - automatically adds think block)
uv run scripts/generate.py \
  --model-path /path/to/z-image \
  --text-encoder-device cpu \
  --dit-device cuda \
  --vae-device cuda \
  --width 1280 --height 720 \
  --steps 9 \
  --shift 3.0 \
  --system-prompt "You are a photographer." \
  --thinking-content "Natural lighting, sharp focus." \
  --output photo.png \
  "A portrait of a woman"

# Image-to-image generation
uv run scripts/generate.py \
  --model-path /path/to/z-image \
  --img2img input.jpg \
  --strength 0.7 \
  --output transformed.png \
  "Transform this into an oil painting"
```

## Profiler Script (scripts/profiler.py)

Profiling and stability testing script for the Z-Image pipeline.

```bash
# Run all tests with default config
uv run scripts/profiler.py --model-path /path/to/z-image-turbo

# Run specific tests
uv run scripts/profiler.py --model-path /path/to/z-image-turbo --tests encode,generate

# Test different optimization configs (FA2, device placement, dtype, etc.)
uv run scripts/profiler.py --model-path /path/to/z-image-turbo --sweep

# Test different device placements (encoder/DiT/VAE on CPU/GPU)
uv run scripts/profiler.py --model-path /path/to/z-image-turbo --sweep-devices

# Save results to JSON
uv run scripts/profiler.py --model-path /path/to/z-image-turbo --output results.json

# Verbose output with all timings
uv run scripts/profiler.py --model-path /path/to/z-image-turbo -v

# Show system/library info only (no tests)
uv run scripts/profiler.py --show-info
```

### Profiler-Specific Flags

| Flag | Description |
|------|-------------|
| `--tests` | Comma-separated list of tests to run |
| `--sweep` | Test multiple optimization configs |
| `--sweep-devices` | Test multiple device placements |
| `--output`, `-o` | Save results to JSON file |
| `--repeat` | Number of times to repeat test suite |
| `--show-info` | Show system info and exit |

### Available Tests

| Test | Description |
|------|-------------|
| `cuda_sync` | CUDA synchronization timing |
| `load_encoder` | Load text encoder only |
| `encode_short` | Encode short prompt |
| `encode_medium` | Encode medium-length prompt |
| `encode_with_template` | Encode with template |
| `encode_with_thinking` | Encode with thinking block |
| `generate_text` | Text generation (rewriting) |
| `repeated_encode` | Memory leak detection |
| `load_pipeline` | Load full pipeline |
| `full_generation` | Full image generation |

## CLI Flags (Shared between web/server.py and scripts/generate.py)

### Model & Config
| Flag | Description |
|------|-------------|
| `--model-path` | Path to Z-Image model |
| `--config` | Path to TOML config file |
| `--profile` | Config profile to use (default: "default") |
| `--templates-dir` | Path to templates directory |

### Device Placement
| Flag | Description |
|------|-------------|
| `--text-encoder-device` | cpu/cuda/mps/auto |
| `--dit-device` | cpu/cuda/mps/auto |
| `--vae-device` | cpu/cuda/mps/auto |

### API Backend
| Flag | Description |
|------|-------------|
| `--api-url` | URL for heylookitsanllm API |
| `--api-model` | Model ID for API backend |
| `--use-api-encoder` | Use API for encoding (local is default) |

### Optimization
| Flag | Description |
|------|-------------|
| `--cpu-offload` | Enable CPU offload for transformer |
| `--flash-attn` | Enable Flash Attention |
| `--compile` | Compile transformer with torch.compile |
| `--debug` | Enable debug logging (embedding stats, token IDs) |

### PyTorch Native (Phase 1)
| Flag | Description |
|------|-------------|
| `--attention-backend` | auto/flash_attn_2/flash_attn_3/sage/xformers/sdpa |
| `--use-custom-scheduler` | Use pure PyTorch FlowMatchScheduler |
| `--tiled-vae` | Enable tiled VAE decode for 2K+ images |
| `--tile-size` | Tile size in pixels (default: 512) |
| `--tile-overlap` | Overlap between tiles (default: 64) |
| `--embedding-cache` | Enable embedding cache for repeated prompts |
| `--cache-size` | Max cached embeddings (default: 100) |
| `--long-prompt-mode` | How to handle prompts >1504 tokens: truncate/interpolate/pool/attention_pool |
| `--hidden-layer` | Which hidden layer to extract embeddings from (default: -2, penultimate) |

### Generation
| Flag | Description |
|------|-------------|
| `--width` | Image width (default: 1024) |
| `--height` | Image height (default: 1024) |
| `--steps` | Inference steps (default: 9) |
| `--guidance-scale` | CFG scale (default: 0.0) |
| `--shift` | Scheduler shift/mu (default: 3.0) |
| `--seed` | Random seed |
| `--img2img` | Input image path for img2img generation |
| `--strength` | img2img strength: 0.0 (no change) to 1.0 (full regeneration) (default: 0.7) |

### Prompt Control
| Flag | Description |
|------|-------------|
| `--system-prompt` | System message |
| `--thinking-content` | Content inside `<think>...</think>` (triggers think block) |
| `--assistant-content` | Content after `</think>` |
| `--force-think-block` | Add empty think block even without content |
| `--template` | Template name to use |

### LoRA
| Flag | Description |
|------|-------------|
| `--lora` | LoRA path with optional scale (path:scale). Repeatable. |

### Rewriter
| Flag | Description |
|------|-------------|
| `--rewriter-use-api` | Use API backend for prompt rewriting |
| `--rewriter-api-url` | API URL for rewriter (defaults to --api-url) |
| `--rewriter-api-model` | Model ID for rewriter API (default: Qwen3-4B) |
| `--rewriter-vl-api-model` | Model ID for VL rewriting via API (e.g., qwen2.5-vl-72b-mlx) |
| `--rewriter-temperature` | Sampling temperature (default: 0.6) |
| `--rewriter-top-p` | Nucleus sampling threshold (default: 0.95) |
| `--rewriter-min-p` | Minimum probability threshold (default: 0.0, disabled) |
| `--rewriter-max-tokens` | Maximum tokens to generate (default: 512) |
| `--rewriter-timeout` | API request timeout in seconds (default: 120.0, VL models may need longer) |
| `--rewriter-no-vl` | Disable VL model selection in rewriter UI |
| `--rewriter-preload-vl` | Preload Qwen3-VL at startup for rewriting |

### Vision Conditioning (Qwen3-VL)
| Flag | Description |
|------|-------------|
| `--vl-model-path` | Path to Qwen3-VL-4B-Instruct model |
| `--vl-device` | Device for VL model (cpu recommended to save VRAM) |
| `--vl-alpha` | Default VL influence (0.0-1.0, default: 0.3) |
| `--vl-hidden-layer` | Hidden layer for VL extraction (default: -2, recommend: -6 for VL) |
| `--vl-auto-unload` | Unload VL model after extraction (default: true) |
| `--vl-blend-mode` | Blend mode: linear/adain/adain_per_dim/style_delta |
| `--vl-outlier-masking` | Outlier masking mode: none/zero/clamp/scale (default: none) |
| `--vl-outlier-threshold` | Std ratio threshold for outlier masking (default: 10.0) |

## REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Generate image from prompt |
| `/api/encode` | POST | Encode prompt to embeddings |
| `/api/format-prompt` | POST | Preview formatted prompt (no encoding) |
| `/api/templates` | GET | List available templates |
| `/api/rewriters` | GET | List available rewriter templates |
| `/api/rewriter-models` | GET | List available rewriter models (text/VL) |
| `/api/rewriter-config` | GET | Get rewriter default parameters |
| `/api/rewrite` | POST | Rewrite prompt using Qwen3 model |
| `/api/save-embeddings` | POST | Save embeddings to file |
| `/api/history` | GET | Get generation history |
| `/api/history/{index}` | DELETE | Delete specific history item |
| `/api/history` | DELETE | Clear all history |
| `/api/vl/status` | GET | Check VL availability and config |
| `/api/vl/config` | GET | Get VL default parameters |
| `/api/vl/extract` | POST | Extract VL embeddings from image |
| `/api/vl/generate` | POST | Generate with VL conditioning |
| `/api/vl/cache/{id}` | DELETE | Clear specific VL cache entry |
| `/api/vl/cache` | DELETE | Clear all VL cache |
| `/health` | GET | Health check |

## API Request Fields

```json
{
  "prompt": "A cat sleeping",
  "system_prompt": "You are a painter.",
  "thinking_content": "Orange fur, green eyes.",
  "assistant_content": "Here is your cat:",
  "force_think_block": false,
  "strip_quotes": false,
  "template": "photorealistic",
  "width": 1024,
  "height": 1024,
  "steps": 9,
  "seed": 42,
  "guidance_scale": 0.0,
  "shift": 3.0
}
```

**Think block behavior:**
- If `thinking_content` is provided, a think block is automatically added
- If `force_think_block` is true, an empty think block is added even without content
- Default: no think block (matches official HF Space)

**Content processing:**
- `strip_quotes`: Remove `"` characters from prompt (for JSON-type inputs, since Z-Image treats `"` as text to render)

## LoRA Support

LoRAs are loaded and fused into the transformer weights at startup.

**Via CLI:**
```bash
# Single LoRA
--lora style.safetensors:0.8

# Multiple LoRAs (stackable)
--lora style.safetensors:0.5 --lora detail.safetensors:0.3
```

**Via config.toml:**
```toml
[default.lora]
paths = ["style.safetensors", "detail.safetensors"]
scales = [0.5, 0.3]
```

**Via Python:**
```python
pipe = ZImagePipeline.from_pretrained(...)
pipe.load_lora("style.safetensors", scale=0.8)
pipe.load_lora(["lora1.safetensors", "lora2.safetensors"], scale=[0.5, 0.3])
```

Note: LoRAs are fused (permanently merged) into weights. To remove, reload the model.

## PyTorch-Native Components (Phase 1 Migration)

These components reduce diffusers dependency and optimize for RTX 4090.

### Attention Backend Selector

Priority-based detection: Flash Attention 3 > FA2 > Sage > xFormers > SDPA

```python
from llm_dit import setup_attention_backend

# Auto-detect best available backend
setup_attention_backend("auto")

# Force specific backend
setup_attention_backend("flash_attn_2")

# Environment variable override
# LLM_DIT_ATTENTION=sdpa python script.py
```

### FlowMatchScheduler (Pure PyTorch)

**IMPORTANT:** The custom scheduler is required for the `shift` parameter to work. Diffusers' `FlowMatchEulerDiscreteScheduler` ignores the `mu`/`shift` parameter entirely. Always use `use_custom_scheduler = true` in config or `--use-custom-scheduler` on CLI.

```python
from llm_dit import FlowMatchScheduler

scheduler = FlowMatchScheduler(shift=3.0)
scheduler.set_timesteps(num_inference_steps=9, device="cuda")

# Access sigma schedule
sigmas = scheduler.sigmas  # Shifted: sigma' = shift * sigma / (1 + (shift-1) * sigma)
```

### Tiled VAE Decoder (2K+ Images)

```python
from llm_dit.utils.tiled_vae import TiledVAEDecoder

# Wrap existing VAE
tiled_decoder = TiledVAEDecoder(
    vae=pipe.vae,
    tile_size=512,    # Pixels (latent = 512/8 = 64)
    tile_overlap=64,  # Overlap for smooth blending
)

# Decode large latents
image = tiled_decoder.decode(latents)  # Handles any size
```

### Embedding Cache (Repeated Prompts)

Thread-safe LRU cache for text embeddings. Avoids re-encoding identical prompts.

```python
from llm_dit.backends.transformers import TransformersBackend

# Enable cache on creation
backend = TransformersBackend.from_pretrained(
    "/path/to/model",
    enable_cache=True,
    cache_size=100,
)

# Or enable later
backend.enable_cache(max_size=100)

# Check cache stats
stats = backend.cache_stats
print(f"Hit rate: {stats.hit_rate:.1f}%")

# Clear cache
backend.clear_cache()
```

Use cases:
- Generating multiple images with same prompt (different seeds)
- Web server handling repeated requests
- Iterating on generation parameters without re-encoding

### Context Refiner (Standalone Module)

```python
from llm_dit import ContextRefiner

# Create standalone (matches Z-Image architecture)
refiner = ContextRefiner(
    dim=3840,      # Hidden dimension
    n_layers=2,    # 2 transformer layers
    n_heads=30,    # 30 attention heads (128 dim/head)
)

# Load from Z-Image checkpoint
refiner = ContextRefiner.from_pretrained("/path/to/z-image", device="cuda")

# Process text embeddings (3840 dim, already projected from 2560)
refined = refiner(text_embeddings)  # (batch, seq, 3840)

# Enable gradient checkpointing for training
refiner.enable_gradient_checkpointing()
```

### Image-to-Image Generation

```python
from llm_dit import ZImagePipeline
from PIL import Image

pipe = ZImagePipeline.from_pretrained("/path/to/z-image")

# Load input image
input_image = Image.open("photo.jpg")

# Generate with strength control
result = pipe.img2img(
    prompt="A cat sleeping in sunlight, oil painting style",
    image=input_image,
    strength=0.75,  # 0.0 = no change, 1.0 = full regeneration
    num_inference_steps=9,
)
result.images[0].save("output.png")
```

### Usage Examples

**RTX 4090 Optimized (CLI):**
```bash
uv run scripts/generate.py \
  --model-path /path/to/z-image-turbo \
  --text-encoder-device cpu \
  --dit-device cuda \
  --vae-device cuda \
  --attention-backend auto \
  --use-custom-scheduler \
  --flash-attn \
  "A mountain landscape"
```

**RTX 4090 Optimized (Config):**
```bash
uv run scripts/generate.py --config config.toml --profile rtx4090 "A mountain landscape"
```

**Large Image Generation (2K+):**
```bash
uv run scripts/generate.py \
  --model-path /path/to/z-image-turbo \
  --width 2048 --height 2048 \
  --tiled-vae \
  --tile-size 512 \
  --tile-overlap 64 \
  "A detailed cityscape"
```

**Low VRAM (8-16GB):**
```bash
uv run scripts/generate.py --config config.toml --profile low_vram "A cat"
```

## Prompt Rewriting

The loaded Qwen3 model can be used for prompt rewriting/expansion in addition to embedding extraction. This enables creative prompt enhancement without loading additional models.

The rewriter supports three models:
- **Qwen3-4B (Text)**: Default text-only rewriter using the loaded encoder
- **Qwen3-VL (Vision+Text)**: Vision-language model that can rewrite based on images (local)
- **Qwen3-VL API**: Vision-language model via remote API (for larger VL models)

### Model Selection

The rewriter can use text-only or vision-language models, locally or via API:

| Model | Input Types | Use Case |
|-------|-------------|----------|
| qwen3-4b | Text only | Standard prompt expansion |
| qwen3-vl | Text, Image, or both | Image-based prompts, describe images (local) |
| qwen3-vl-api | Text, Image, or both | Image-based prompts via API (e.g., qwen2.5-vl-72b-mlx) |

**Local VL Model Configuration:**
- Uses the same model path as VL conditioning (`vl.model_path`)
- Loads on-demand when user selects it in the UI
- Can optionally preload at startup with `--rewriter-preload-vl`

**API VL Model Configuration:**
- Configure with `rewriter.vl_api_model` in config.toml or `--rewriter-vl-api-model` on CLI
- Uses configured `rewriter.api_url` or falls back to main `api.url`
- Image is sent as base64 data URL to OpenAI-compatible API
- Requires API server that supports vision models (e.g., heylookitsanllm with qwen2.5-vl)

### Backend Selection

The rewriter can use either the local model or a remote API backend (heylookitsanllm).

- By default, uses the local encoder's Qwen3 model
- With `--rewriter-use-api`, uses a remote API backend for generation
- The API URL defaults to `--api-url` but can be overridden with `--rewriter-api-url`

**Configuration via TOML:**
```toml
[default.rewriter]
use_api = true                # Use API backend
api_url = "http://mac:8080"   # API endpoint (or leave empty to use [api].url)
api_model = "Qwen3-4B"        # Model ID for text-only rewriting
vl_api_model = "qwen2.5-vl-72b-mlx"  # Model ID for VL rewriting via API
temperature = 0.6             # Sampling temperature (Qwen3 thinking mode)
top_p = 0.95                  # Nucleus sampling
max_tokens = 512              # Max tokens to generate
timeout = 120.0               # API request timeout in seconds (VL models may need longer)
vl_enabled = true             # Allow VL model selection in rewriter UI
preload_vl = false            # Load Qwen3-VL at startup for rewriter
```

**Configuration via CLI:**
```bash
# Use API backend for rewriting with custom parameters
uv run web/server.py \
  --model-path /path/to/z-image \
  --rewriter-use-api \
  --rewriter-api-url http://mac:8080 \
  --rewriter-api-model Qwen3-4B \
  --rewriter-temperature 0.6 \
  --rewriter-top-p 0.95 \
  --rewriter-max-tokens 512 \
  --rewriter-timeout 120.0

# Enable local VL rewriter with preloading
uv run web/server.py \
  --model-path /path/to/z-image \
  --vl-model-path /path/to/Qwen3-VL-4B-Instruct \
  --rewriter-preload-vl

# Enable API VL rewriter for larger models (e.g., qwen2.5-vl-72b)
uv run web/server.py \
  --model-path /path/to/z-image \
  --rewriter-use-api \
  --rewriter-api-url http://mac:8080 \
  --rewriter-vl-api-model qwen2.5-vl-72b-mlx \
  --rewriter-timeout 180.0
```

**Rewriter Templates:**
Place templates in `templates/z_image/rewriter/` with `category: rewriter` in frontmatter.

```markdown
---
name: rewriter_character_generator
description: Character Generator (prompt rewriter)
model: z-image
category: rewriter
---
You are an expert character designer...
```

**Via Web UI:**
1. Enter a basic prompt (or skip for image-only with VL model)
2. Open "Prompt Rewriter (Qwen3)" section
3. Select a model (Qwen3-4B or Qwen3-VL if configured)
4. If using Qwen3-VL, optionally upload a reference image
5. Select a rewriter style
6. Click "Rewrite Prompt"
7. Click "Use This Prompt" to apply

**Via API:**
```bash
# List available rewriters
curl http://localhost:8000/api/rewriters

# List available rewriter models
curl http://localhost:8000/api/rewriter-models

# Rewrite with text-only model (default)
curl -X POST http://localhost:8000/api/rewrite \
  -H "Content-Type: application/json" \
  -d '{"prompt": "An Israeli woman", "rewriter": "rewriter_official"}'

# Rewrite with local VL model and image
curl -X POST http://localhost:8000/api/rewrite \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-vl", "prompt": "Describe this image", "image": "data:image/jpeg;base64,..."}'

# Rewrite with API VL model (for larger models)
curl -X POST http://localhost:8000/api/rewrite \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-vl-api", "prompt": "Describe this image", "image": "data:image/jpeg;base64,..."}'

# Image-only rewriting (VL model only)
curl -X POST http://localhost:8000/api/rewrite \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-vl", "image": "data:image/jpeg;base64,...", "rewriter": "rewriter_official"}'

# Override generation parameters
curl -X POST http://localhost:8000/api/rewrite \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cat", "rewriter": "rewriter_official", "temperature": 0.8, "top_p": 0.9, "max_tokens": 256}'
```

**Via Python:**
```python
# Using local encoder
backend = TransformersBackend.from_pretrained(...)
rewritten = backend.generate(
    prompt="A cat sleeping",
    system_prompt="You are an expert at writing image prompts...",
    max_new_tokens=512,
    temperature=1.0,
    top_p=0.95,
)

# Using API backend
backend = APIBackend.from_url("http://localhost:8000", "qwen3-4b")
rewritten = backend.generate(...)
```

## Vision Conditioning (Qwen3-VL) - EXPERIMENTAL

Zero-shot vision conditioning using Qwen3-VL embeddings. This is a novel approach that uses a reference image to influence the generated output's style/content without any training.

**Status: EXPERIMENTAL** - VL conditioning works but produces visible artifacts compared to pure text. See Research Findings below.

### Why It Works (In Theory)

Qwen3-VL-4B's text model shares architecture with Qwen3-4B:
- Both have `hidden_size=2560` (matching Z-Image's expected embedding dimension)
- Qwen3-VL projects vision features into the same embedding space via `PatchMerger.linear_fc2`
- Interpolating VL + text embeddings should produce coherent conditioning

### Current Implementation

**Optimal Settings (from experiments):**
```toml
[rtx4090.vl]
model_path = "/path/to/Qwen3-VL-4B-Instruct"
device = "cpu"                  # Recommended to save VRAM
default_alpha = 1.0             # Use text_tokens_only=true, pure VL
default_hidden_layer = -8       # Layer -8 produces cleaner results than -2
text_tokens_only = true         # CRITICAL: Image tokens cause severe artifacts
target_std = 70.0               # Match Qwen3-4B text embedding std
auto_unload = true              # Unload after extraction
```

**Key Technical Details:**

| Parameter | Optimal Value | Notes |
|-----------|---------------|-------|
| `hidden_layer` | **-8** | Layer -2 (default for text) produces heavy artifacts for VL |
| `text_tokens_only` | **true** | Image token positions cause severe grid/blocky artifacts |
| `target_std` | **70.0** | Qwen3-4B text embeddings have std ~70, VL has ~13.25 |
| `alpha` | **1.0** | With text_tokens_only=true, can use pure VL |

### Chat Template Requirements

Qwen3-VL does NOT support the `enable_thinking` parameter. We manually inject think block tokens after tokenization to match Qwen3-4B format:

```python
# Token IDs for manual injection (src/llm_dit/vl/qwen3_vl.py)
THINK_START_TOKEN_ID = 151667   # <think>
THINK_END_TOKEN_ID = 151668     # </think>
DOUBLE_NEWLINE_TOKEN_ID = 271   # \n\n (NOT 198 twice!)

# After tokenization:
# input_ids = [...tokens..., 151667, 271, 151668, 271]
```

### Embedding Statistics

| Source | Mean | Std | Notes |
|--------|------|-----|-------|
| Qwen3-4B text | ~0.0 | ~70 | Target for scaling |
| Qwen3-VL text tokens | ~0.0 | ~13.25 | Needs ~5.3x scaling |
| Qwen3-VL image tokens | ~0.0 | ~2.5-5.0 | Needs 10-25x scaling (problematic) |

The extreme scaling needed for image tokens likely causes the artifacts. Text tokens need only ~1.5x scaling with `target_std=70`.

### Research Findings (2025-12-12)

**Layer Selection:**
- Layer -2 (Z-Image default) loses text prompt content with VL conditioning
- Layer -6 produces crisper images and preserves text prompt (Homer appears correctly)
- VL fine-tuning overwrote later layers for vision tasks, making middle layers better

**Outlier Masking Results:**
- Outlier masking found NO outliers at layer -6 (the 617x outlier is layer -2 specific)
- Layer -6 is naturally cleaner than -2 for VL embeddings
- The fix to apply masking to image tokens only works but layer -6 doesn't have the outliers

**CRITICAL CORRECTION - `text_tokens_only` Setting:**
- `text_tokens_only=True` (our previous default) = **VL does NOTHING** (all image info stripped)
- `text_tokens_only=False` = **VL actually works** (1026 image tokens preserved)
- Image information is in IMAGE TOKEN POSITIONS, not text tokens
- Previous "landscape breakthrough" was just img2img working, not VL

**Layer -2 vs -6 with Full Image Tokens:**
- Layer -2 + full tokens = HEAVY GLITCH ARTIFACTS (617x outlier in dim 396)
- Layer -6 + full tokens = CLEAN RESULTS

**What VL Actually Does (When Configured Correctly):**
- alpha=0.3: Slight style influence from reference
- alpha=0.5: Strong style transfer (e.g., photorealistic from photo reference)
- alpha=0.7: Reference strongly influences output
- alpha=1.0: Essentially reconstructs reference scene

**Style Delta Results (FAILED):**
- Even at alpha 0.3, Homer was completely destroyed
- Replaced with random style-influenced content (woman on beach, cyan ball)
- The delta contains too much "content" information, not just style

**AdaIN Results (PARTIAL SUCCESS):**
- `per_token` mode: Preserves Homer perfectly but no visible style transfer
- `per_dim` mode: Transfers colors (orange shirt from hexagon) but corrupts subject (Homer becomes Bart)
- Fundamental tradeoff: style transfer strong enough to be visible also corrupts content

**Core Insight:**
The embedding space doesn't cleanly separate "style" from "content". VL influence strong enough to transfer style also corrupts semantic content.

### Correct VL Configuration (CRITICAL)

**Previous documentation was WRONG.** The correct settings are:

```python
vl_result = vl_extractor.extract(
    image,
    text=prompt,
    hidden_layer=-6,          # NOT -2 (has glitch artifacts)
    text_tokens_only=False,   # MUST be False to include image info!
    scale_to_text=True
)
```

| Setting | Wrong | Correct |
|---------|-------|---------|
| `text_tokens_only` | True | **False** |
| `hidden_layer` | -2 | **-6** |

**Why `text_tokens_only=False` is critical:**
- Image info is in IMAGE TOKEN positions (~1026 tokens for 1024x1024)
- `text_tokens_only=True` discards all image tokens
- With True, VL embeddings ≈ pure text embeddings (VL does nothing)

**Results at different alphas (with correct settings):**

| Alpha | Effect |
|-------|--------|
| 0.3 | Slight style influence |
| 0.5 | Strong style transfer (photorealistic from photo ref) |
| 0.7 | Reference strongly influences output |
| 1.0 | Reconstructs reference scene |

### Usage (Pure VL Blending)

**Web UI:**
1. Open "Vision Conditioning (Qwen3-VL)" section
2. Upload a reference image
3. Use settings: layer -6, **text_tokens_only=False** (critical!)
4. Adjust alpha: 0.3-0.5 for style influence, higher for scene reconstruction

**CLI:**
```bash
uv run web/server.py \
  --model-path /path/to/z-image \
  --vl-model-path /path/to/Qwen3-VL-4B-Instruct \
  --vl-device cpu \
  --vl-hidden-layer -6
```

**Python (correct settings):**
```python
vl_result = vl_extractor.extract(
    image,
    text=prompt,
    hidden_layer=-6,
    text_tokens_only=False,  # CRITICAL!
    scale_to_text=True
)
blended = blend_embeddings(vl_result.embeddings, text_emb, alpha=0.5)
```

### Memory Management

VL extraction workflow (recommended):
1. Load Qwen3-VL on CPU
2. Extract embeddings from reference image
3. Cache embeddings (keyed by image hash)
4. Unload Qwen3-VL
5. Generate using cached embeddings

This keeps VRAM free for the DiT.

### Experiment Runner

Run VL experiments with comparison grids:

```bash
cd experiments/qwen3_vl/scripts

# Run style transfer test (recommended)
uv run run_comparison.py \
    -i experiments/inputs/style_anime_girl.png \
    -p "Homer Simpson" \
    --sweep style_transfer \
    --steps 4 \
    -o experiments/results/style_test

# Run blend mode comparison
uv run run_comparison.py \
    -i experiments/inputs/style_anime_girl.png \
    -p "Your prompt" \
    --sweep blend_comparison \
    --steps 4

# Run think block comparison (with/without <think> tokens)
uv run run_comparison.py \
    -i experiments/inputs/style_anime_girl.png \
    -p "Your prompt" \
    --sweep think_comparison \
    --steps 4
```

Results are saved to `experiments/results/` with comparison grids.

**Profile Configuration (CRITICAL):**

`run_comparison.py` uses the `default` profile by default to match `test_all_blend_modes.py`. This is important because the profiles have different `long_prompt_mode` settings:

| Profile | `long_prompt_mode` | Effect |
|---------|-------------------|--------|
| `default` | `interpolate` | Simple linear resampling (recommended for VL) |
| `rtx4090` | `attention_pool` | Cosine similarity weighting (can cause issues with VL) |

If you see unexpected results, verify you're using the correct profile:
```bash
# Explicitly use default profile (recommended for VL experiments)
uv run run_comparison.py --profile default ...

# Or use rtx4090 for non-VL experiments with long prompts
uv run run_comparison.py --profile rtx4090 ...
```

**Recommended Style Transfer Settings:**
- `--steps 4` (better style fusion than 9)
- `--sweep style_transfer` (uses `adain_per_dim` blend mode, layer -6, alpha 0.3-0.5)
- Profile: `default` (uses `interpolate` long_prompt_mode)

### Outlier Dimension Masking

VL image tokens have extreme per-dimension outliers that cause artifacts:
- **Dimension 396:** 617x std ratio at layer -2 (vs Qwen3-4B reference)
- **Dimension 4:** 42x std ratio at layer -2
- **Layer -6:** NO outliers found (naturally cleaner than -2)

Three masking modes are available via `mask_outlier_dimensions()`:

| Mode | Description |
|------|-------------|
| `zero` | Zero out outlier dimensions entirely |
| `clamp` | Scale outlier dimensions to threshold level |
| `scale` | Proportionally reduce outlier dimension values |

**Note:** Fixed to apply masking to image tokens ONLY before combining with text tokens.

**Experiment Runner CLI:**
```bash
# Run outlier masking sweep (tests none, zero, clamp, scale)
uv run python experiments/qwen3_vl/scripts/run_comparison.py \
  -i experiments/inputs/test_scene.png \
  -p "Your prompt" \
  --sweep outlier \
  -o experiments/results/outlier_sweep

# Custom outlier masking modes
uv run python experiments/qwen3_vl/scripts/run_comparison.py \
  -i experiments/inputs/test_scene.png \
  -p "Your prompt" \
  --token-modes full \
  --outlier-masking zero clamp \
  --outlier-threshold 10.0 \
  -o experiments/results/outlier_test
```

**Python API:**
```python
from llm_dit.vl import mask_outlier_dimensions, get_outlier_dimensions

# Mask outlier dimensions
masked_emb, info = mask_outlier_dimensions(
    embeddings,
    threshold=10.0,  # Std ratio threshold
    mode="zero",     # "zero", "clamp", or "scale"
)
print(f"Masked {len(info['masked_dimensions'])} dimensions")

# Analysis helper
outliers = get_outlier_dimensions(embeddings, threshold=10.0)
for dim, ratio in outliers[:5]:
    print(f"Dimension {dim}: {ratio:.1f}x std ratio")
```

### Style Delta and AdaIN Blending

Two new blending modes implemented in `src/llm_dit/vl/blending.py`:

**Style Delta Arithmetic:**
```python
from llm_dit.vl import compute_style_delta, blend_with_style_delta

# Extract style by subtracting neutral from styled
style_delta = compute_style_delta(styled_vl_emb, neutral_vl_emb)

# Add style delta to text embeddings
result = blend_with_style_delta(text_emb, style_delta, alpha=0.3)
```

**AdaIN Blending:**
```python
from llm_dit.vl import blend_adain, blend_adain_per_dim

# Per-token AdaIN (preserves content)
result = blend_adain(text_emb, vl_emb, alpha=0.3)

# Per-dimension AdaIN (stronger style transfer, may corrupt content)
result = blend_adain_per_dim(text_emb, vl_emb, alpha=0.3)
```

**Test scripts:**
- `experiments/qwen3_vl/scripts/test_style_delta.py` - Style delta experiments
- `experiments/qwen3_vl/scripts/test_adain.py` - AdaIN experiments

**Results:** Both methods show fundamental tradeoff - style transfer strong enough to be visible also corrupts content.

### Future Research Directions

1. **Trained minimal adapter**: Single linear layer to bridge VL and text space
2. **Layer interpolation**: Blend multiple hidden layers
3. **Attention-based selection**: Use attention weights to select important tokens
4. **Alternative VL models**: Try different vision-language models
5. **IP-Adapter comparison**: Measure quality gap vs trained methods

### Files

| File | Description |
|------|-------------|
| `src/llm_dit/vl/qwen3_vl.py` | VLEmbeddingExtractor class |
| `src/llm_dit/vl/blending.py` | Blending utilities: normalization, outlier masking, style delta, AdaIN |
| `src/llm_dit/vl/qwen3_4b_stats.npz` | Per-dimension reference statistics |
| `src/llm_dit/pipelines/z_image.py` | ZImagePipeline with img2img support |
| `scripts/generate.py` | CLI script with img2img flags |
| `experiments/qwen3_vl/AGENTS.md` | Working task list and experiment priorities |
| `experiments/qwen3_vl/scripts/run_comparison.py` | Experiment runner with sweep presets |
| `experiments/qwen3_vl/scripts/test_style_delta.py` | Style delta arithmetic experiments |
| `experiments/qwen3_vl/scripts/test_adain.py` | AdaIN blending experiments |
| `experiments/qwen3_vl/scripts/test_vl_img2img.py` | VL + img2img combined (landscape composition) |

See `experiments/qwen3_vl/README.md` for complete technical documentation.
