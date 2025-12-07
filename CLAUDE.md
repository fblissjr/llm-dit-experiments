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
hidden_states[-2] -> embeddings (2560 dim)
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
| Embedding extraction | hidden_states[-2] | Penultimate layer |
| CFG scale | 0.0 | Baked in via Decoupled-DMD |
| Steps | 8-9 | Turbo distilled |
| Scheduler | FlowMatchEuler | shift=3.0 |
| VAE | 16-channel | Wan-family |
| Context refiner | 2 layers | No timestep modulation |

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
    cli.py              # Shared CLI argument parser and config loading
    config.py           # Configuration dataclasses

scripts/
    generate.py         # CLI image generation script

web/
    server.py           # FastAPI web server (history storage, all endpoints)
    index.html          # Web UI (mobile-friendly, history panel, resolution presets)

templates/z_image/      # 140+ prompt templates (markdown + YAML frontmatter)
config.toml             # Example configuration file

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
    LOG.md              # Detailed session log with all changes
    SESSION_CONTINUITY.md  # Session state tracking
    GUIDING_PRINCIPLES.md  # Architectural decisions
```

## Living Documentation

- **internal/LOG.md**: Detailed session log with all changes and investigations
- **CHANGELOG.md**: Version history (semantic versioning)

## Configuration

Config file (TOML) is the source of truth. CLI flags override config values.

**Transformers v5 Note:** `load_in_8bit`/`load_in_4bit` are deprecated. Use `quantization` instead.

```toml
# config.toml
[default]
model_path = "/path/to/z-image-turbo"
templates_dir = "templates/z_image"

[default.encoder]
device = "auto"
torch_dtype = "bfloat16"
quantization = "none"  # v5 API: "none", "4bit", "8bit"
cpu_offload = false

[default.pipeline]
device = "cuda"

[default.generation]
width = 1024
height = 1024
steps = 9
guidance_scale = 0.0

[default.scheduler]
shift = 3.0

[default.optimization]
flash_attn = false
compile = false
cpu_offload = false

[default.pytorch]
# PyTorch-native components (Phase 1 migration)
attention_backend = "auto"    # auto/flash_attn_2/flash_attn_3/sage/xformers/sdpa
use_custom_scheduler = false  # Use pure PyTorch FlowMatchScheduler
tiled_vae = false             # Enable for 2K+ images
tile_size = 512               # Tile size in pixels
tile_overlap = 64             # Overlap for smooth blending

[default.lora]
paths = []
scales = []
```

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
```

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

### Generation
| Flag | Description |
|------|-------------|
| `--width` | Image width (default: 1024) |
| `--height` | Image height (default: 1024) |
| `--steps` | Inference steps (default: 9) |
| `--guidance-scale` | CFG scale (default: 0.0) |
| `--shift` | Scheduler shift/mu (default: 3.0) |
| `--seed` | Random seed |

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

## REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Generate image from prompt |
| `/api/encode` | POST | Encode prompt to embeddings |
| `/api/format-prompt` | POST | Preview formatted prompt (no encoding) |
| `/api/templates` | GET | List available templates |
| `/api/rewriters` | GET | List available rewriter templates |
| `/api/rewrite` | POST | Rewrite prompt using Qwen3 model |
| `/api/save-embeddings` | POST | Save embeddings to file |
| `/api/history` | GET | Get generation history |
| `/api/history/{index}` | DELETE | Delete specific history item |
| `/api/history` | DELETE | Clear all history |
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
1. Enter a basic prompt
2. Open "Prompt Rewriter (Qwen3)" section
3. Select a rewriter style
4. Click "Rewrite Prompt"
5. Click "Use This Prompt" to apply

**Via API:**
```bash
# List available rewriters
curl http://localhost:8000/api/rewriters

# Rewrite a prompt
curl -X POST http://localhost:8000/api/rewrite \
  -H "Content-Type: application/json" \
  -d '{"prompt": "An Israeli woman", "rewriter": "rewriter_z_image_character_generator"}'
```

**Via Python:**
```python
# Using local encoder
backend = TransformersBackend.from_pretrained(...)
rewritten = backend.generate(
    prompt="A cat sleeping",
    system_prompt="You are an expert at writing image prompts...",
    max_new_tokens=512,
    temperature=0.7,
)

# Using API backend
backend = APIBackend.from_url("http://localhost:8000", "qwen3-4b")
rewritten = backend.generate(...)
```

## Related Projects

- **ComfyUI-QwenImageWanBridge**: Source of templates and ComfyUI implementation patterns
- **DiffSynth-Studio**: Reference implementation for Z-Image pipeline
- **diffusers**: Base library we build on

## Key Research References

Located in the ComfyUI-QwenImageWanBridge sibling repo under `internal/`:
- `z_image_paper_analysis/decoupled_dmd_training_report.md` - CFG baking mechanism
- `z_image_paper_alignment/paper_code_alignment.md` - Implementation alignment
- `z_image_context_refiner_deep_dive.md` - Context refiner architecture
- `z_image_paper_alignment/diffusers_port_considerations.md` - Backend considerations
