# Z-Image Web Server API

## Overview

The web server provides a REST API for Z-Image generation with the Qwen3-4B text encoder.

## Running the Server

```bash
# Install PyTorch separately (not pinned in dependencies)
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

# Sync dependencies
uv sync

# Run with local encoder (default, recommended for RTX 4090)
uv run web/server.py \
  --model-path /path/to/z-image-turbo \
  --text-encoder-device cpu \
  --dit-device cuda \
  --vae-device cuda \
  --port 7860

# Run with API encoder (distributed inference, requires --use-api-encoder)
uv run web/server.py \
  --model-path /path/to/z-image-turbo \
  --api-url http://mac-host:8080 \
  --api-model Qwen3-4B \
  --use-api-encoder \
  --dit-device cuda \
  --vae-device cuda

# Run with config file
uv run web/server.py --config config.toml --profile default

# Run with LoRA
uv run web/server.py \
  --model-path /path/to/z-image-turbo \
  --lora /path/to/style.safetensors:0.8 \
  --dit-device cuda
```

### Running as a Background Service

The server can run in the background to survive SSH disconnects:

```bash
# Start server in background
./scripts/start-server.sh --model-path /path/to/z-image --dit-device cuda

# View logs
tail -f logs/server.log

# Stop server
./scripts/stop-server.sh
```

## Configuration

The server supports both CLI flags and TOML config files. Config file is the source of truth; CLI flags override config values.

```toml
# config.toml
[default]
model_path = "/path/to/z-image-turbo"
templates_dir = "templates/z_image"

[default.devices]
text_encoder = "cpu"
dit = "cuda"
vae = "cuda"

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

[default.lora]
paths = ["/path/to/lora1.safetensors", "/path/to/lora2.safetensors"]
scales = [0.8, 0.5]
```

## CLI Flags

### Model & Device Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model-path` | str | required | Path to Z-Image model |
| `--config` | str | None | Path to TOML config file |
| `--profile` | str | default | Config profile to use |
| `--text-encoder-device` | str | auto | Device for text encoder (cpu/cuda/mps/auto) |
| `--dit-device` | str | auto | Device for DiT transformer (cpu/cuda/mps/auto) |
| `--vae-device` | str | auto | Device for VAE (cpu/cuda/mps/auto) |
| `--hidden-layer` | int | -2 | Which hidden layer to extract embeddings from (-1=last, -2=penultimate) |

### Server Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--host` | str | 127.0.0.1 | Server host |
| `--port` | int | 7860 | Server port |
| `--encoder-only` | flag | False | Load only encoder (no DiT/VAE) |
| `--use-api-encoder` | flag | False | Use API backend for encoding (local is default) |

### API Backend Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--api-url` | str | None | URL for heylookitsanllm API backend |
| `--api-model` | str | None | Model ID for API backend |

### Optimization Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--cpu-offload` | flag | False | Enable CPU offload for transformer |
| `--flash-attn` | flag | False | Enable Flash Attention |
| `--compile` | flag | False | Compile transformer with torch.compile |
| `--debug` | flag | False | Enable debug logging |

### Generation Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--shift` | float | 3.0 | Scheduler shift/mu parameter |
| `--guidance-scale` | float | 0.0 | CFG scale (0.0 for Z-Image Turbo) |
| `--long-prompt-mode` | str | interpolate | How to handle prompts >1504 tokens: truncate, interpolate, pool, attention_pool |

### LoRA Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--lora` | str | None | LoRA file path with optional scale (path:scale). Repeatable. |

Example: `--lora style.safetensors:0.8 --lora detail.safetensors:0.5`

### Rewriter Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--rewriter-use-api` | flag | False | Use API backend for prompt rewriting |
| `--rewriter-api-url` | str | None | API URL for rewriter (defaults to --api-url) |
| `--rewriter-api-model` | str | Qwen3-4B | Model ID for rewriter API |
| `--rewriter-temperature` | float | 1.0 | Sampling temperature |
| `--rewriter-top-p` | float | 0.95 | Nucleus sampling threshold |
| `--rewriter-min-p` | float | 0.0 | Minimum probability threshold (0.0 = disabled) |
| `--rewriter-max-tokens` | int | 512 | Maximum tokens to generate |

Example: `--rewriter-use-api --rewriter-api-url http://mac:8080 --rewriter-temperature 0.8`

## API Endpoints

### POST /api/generate

Generate an image from a text prompt.

**Request:**
```json
{
  "prompt": "A cat sleeping",
  "system_prompt": "You are a painter.",
  "thinking_content": "Orange fur, green eyes.",
  "assistant_content": "",
  "force_think_block": false,
  "strip_quotes": false,
  "template": null,
  "width": 1024,
  "height": 1024,
  "steps": 9,
  "seed": null,
  "guidance_scale": 0.0,
  "shift": 3.0,
  "long_prompt_mode": "interpolate",
  "hidden_layer": -2
}
```

**Fields:**
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| prompt | string | Yes | - | User message / image description |
| system_prompt | string | No | null | System message (e.g., "You are a painter.") |
| thinking_content | string | No | null | Content inside `<think>...</think>` (auto-enables think block) |
| assistant_content | string | No | null | Content after `</think>` |
| force_think_block | bool | No | false | Force empty `<think></think>` even without content |
| strip_quotes | bool | No | false | Remove `"` characters from prompt (for JSON inputs) |
| template | string | No | null | Template name |
| width | int | No | 1024 | Image width (divisible by 16) |
| height | int | No | 1024 | Image height (divisible by 16) |
| steps | int | No | 9 | Denoising steps |
| seed | int | No | null | Random seed for reproducibility |
| guidance_scale | float | No | 0.0 | CFG scale (0.0 recommended for Z-Image) |
| shift | float | No | 3.0 | Scheduler shift/mu parameter |
| long_prompt_mode | string | No | interpolate | How to handle prompts >1504 tokens: truncate, interpolate, pool, attention_pool |
| hidden_layer | int | No | -2 | Which hidden layer to extract embeddings from (-1=last, -2=penultimate) |

**Note:** The 1504 token limit is the maximum text sequence length for the DiT. Compression modes only trigger when prompts exceed this limit.

**Note:** Think block behavior is content-driven:
- If `thinking_content` is provided, a think block is automatically added
- If `force_think_block=true`, an empty think block is added
- Otherwise, no think block (matches official HF Space)

**Response:** PNG image stream

**Headers:**
- `X-Generation-Time`: Generation time in seconds
- `X-Seed`: Seed used (or "random")

---

### POST /api/encode

Encode a prompt to embeddings without generating an image.

**Request:**
```json
{
  "prompt": "A cat sleeping",
  "system_prompt": null,
  "thinking_content": null,
  "assistant_content": null,
  "force_think_block": false,
  "strip_quotes": false,
  "template": null
}
```

**Response:**
```json
{
  "shape": [10, 2560],
  "dtype": "torch.bfloat16",
  "encode_time": 0.123,
  "token_count": 10,
  "prompt": "A cat sleeping",
  "formatted_prompt": "<|im_start|>user\nA cat sleeping<|im_end|>\n<|im_start|>assistant\n"
}
```

---

### POST /api/format-prompt

Preview the formatted prompt without encoding (fast, no GPU).

**Request:**
```json
{
  "prompt": "Paint a cat",
  "system_prompt": "You are a painter.",
  "thinking_content": "Orange fur.",
  "assistant_content": null,
  "force_think_block": false,
  "strip_quotes": false,
  "template": null
}
```

**Response:**
```json
{
  "formatted_prompt": "<|im_start|>system\nYou are a painter.<|im_end|>\n<|im_start|>user\nPaint a cat<|im_end|>\n<|im_start|>assistant\n<think>\nOrange fur.\n</think>\n",
  "char_count": 112,
  "token_count": 42,
  "prompt": "Paint a cat",
  "system_prompt": "You are a painter.",
  "thinking_content": "Orange fur.",
  "assistant_content": null,
  "template": null,
  "force_think_block": false,
  "strip_quotes": false
}
```

---

### GET /api/templates

List available templates.

**Response:**
```json
{
  "templates": [
    {"name": "default", "has_thinking": false},
    {"name": "photorealistic", "has_thinking": true},
    {"name": "artistic", "has_thinking": false}
  ]
}
```

---

### GET /api/rewriters

List available rewriter templates (templates with `category: rewriter`).

**Response:**
```json
{
  "rewriters": [
    {"name": "rewriter_character_generator", "description": "Character Generator (prompt rewriter)"},
    {"name": "rewriter_scene_enhancer", "description": "Scene Enhancer (prompt rewriter)"}
  ]
}
```

---

### POST /api/rewrite

Rewrite/expand a prompt using a rewriter template or custom system prompt.

Uses either the local Qwen3 model or an API backend (if configured with `--rewriter-use-api`).

**Request:**
```json
{
  "prompt": "A cat sleeping",
  "rewriter": "rewriter_character_generator",
  "custom_system_prompt": null,
  "max_tokens": 512,
  "temperature": 1.0,
  "top_p": 0.95,
  "min_p": 0.0
}
```

**Fields:**
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| prompt | string | Yes | - | User prompt to rewrite/expand |
| rewriter | string | No* | null | Name of rewriter template |
| custom_system_prompt | string | No* | null | Ad-hoc system prompt for rewriting |
| max_tokens | int | No | from config (512) | Maximum tokens to generate |
| temperature | float | No | from config (1.0) | Sampling temperature |
| top_p | float | No | from config (0.95) | Nucleus sampling threshold |
| min_p | float | No | from config (0.0) | Minimum probability threshold (0.0 = disabled) |

*Either `rewriter` or `custom_system_prompt` must be provided.

**Response:**
```json
{
  "original_prompt": "A cat sleeping",
  "rewritten_prompt": "A fluffy orange tabby cat curled up in a warm sunbeam...",
  "thinking_content": "I should describe a cozy scene with warm lighting...",
  "rewriter": "rewriter_character_generator",
  "backend": "local",
  "gen_time": 2.5
}
```

**Note:** If the LLM generates `<think>...</think>` tags in its output, the content is extracted into `thinking_content` and removed from `rewritten_prompt`. This allows the thinking to be used directly in subsequent image generation.

**Backend Types:**
- `"local"`: Using local Qwen3 model (default)
- `"api"`: Using remote API backend (when `--rewriter-use-api` is set)

---

### POST /api/save-embeddings

Save embeddings to a safetensors file.

**Request:**
```json
{
  "prompt": "A cat sleeping",
  "force_think_block": false
}
```

**Response:**
```json
{
  "path": "/path/to/embeddings/embeddings_abc123.safetensors",
  "shape": [10, 2560],
  "encode_time": 0.123
}
```

---

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "pipeline_loaded": true,
  "encoder_loaded": true,
  "encoder_only_mode": false
}
```

---

### GET /api/history

Get generation history (stored in memory, cleared on server restart).

**Response:**
```json
{
  "history": [
    {
      "id": 0,
      "timestamp": 1733285000.0,
      "prompt": "A cat sleeping",
      "system_prompt": null,
      "thinking_content": null,
      "assistant_content": null,
      "force_think_block": false,
      "strip_quotes": false,
      "width": 1024,
      "height": 1024,
      "steps": 9,
      "seed": null,
      "template": null,
      "guidance_scale": 0.0,
      "shift": 3.0,
      "long_prompt_mode": "interpolate",
      "hidden_layer": -2,
      "gen_time": 12.5,
      "image_b64": "iVBORw0KGgo..."
    }
  ]
}
```

---

### DELETE /api/history/{index}

Delete a specific history item by index.

**Response:**
```json
{
  "deleted": {...},
  "remaining": 4
}
```

---

### DELETE /api/history

Clear all history.

**Response:**
```json
{
  "cleared": 5
}
```

## Chat Template Format

The Qwen3-4B chat template has 4 components:

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<think>
{thinking_content}
</think>
{assistant_content}
```

### Examples

**Basic (no system, no thinking):**
```json
{"prompt": "A cat sleeping"}
```
Result:
```
<|im_start|>user
A cat sleeping<|im_end|>
<|im_start|>assistant
```

**With system prompt:**
```json
{"prompt": "Paint a cat", "system_prompt": "You are a painter."}
```
Result:
```
<|im_start|>system
You are a painter.<|im_end|>
<|im_start|>user
Paint a cat<|im_end|>
<|im_start|>assistant
```

**With empty thinking block (forced):**
```json
{"prompt": "Draw a house", "force_think_block": true}
```
Result:
```
<|im_start|>user
Draw a house<|im_end|>
<|im_start|>assistant
<think>

</think>
```

**With thinking content (auto-enables think block):**
```json
{"prompt": "Paint a cat", "thinking_content": "Orange fur, green eyes."}
```
Result:
```
<|im_start|>user
Paint a cat<|im_end|>
<|im_start|>assistant
<think>
Orange fur, green eyes.
</think>
```

**Full example:**
```json
{
  "prompt": "Paint a sunset",
  "system_prompt": "You are an impressionist painter.",
  "thinking_content": "Warm orange and pink hues, soft clouds.",
  "assistant_content": "Here is my painting:"
}
```
Result:
```
<|im_start|>system
You are an impressionist painter.<|im_end|>
<|im_start|>user
Paint a sunset<|im_end|>
<|im_start|>assistant
<think>
Warm orange and pink hues, soft clouds.
</think>
Here is my painting:
```

## Vision Conditioning (Qwen3-VL)

The server supports vision-conditioned generation using Qwen3-VL embeddings. This is a zero-shot technique that uses a reference image to influence the generated output's style/content.

### Configuration

Add to your config.toml:

```toml
[default.vl]
model_path = "/path/to/Qwen3-VL-4B-Instruct"  # Required to enable
device = "cpu"                                  # Recommended to save VRAM
default_alpha = 0.3                             # 0.0=text only, 1.0=VL only
default_hidden_layer = -2                       # -2=penultimate (recommended)
auto_unload = true                              # Unload after extraction
target_std = 58.75                              # Scale VL embeddings to match text
```

Or use CLI flags:
```bash
uv run web/server.py \
  --model-path /path/to/z-image-turbo \
  --vl-model-path /path/to/Qwen3-VL-4B-Instruct \
  --vl-device cpu \
  --vl-alpha 0.3
```

---

### GET /api/vl/status

Check if VL conditioning is available.

**Response:**
```json
{
  "available": true,
  "configured": true,
  "model_path": "/path/to/Qwen3-VL-4B-Instruct",
  "device": "cpu",
  "default_alpha": 0.3,
  "default_hidden_layer": -2,
  "blend_modes": ["linear", "style_only", "graduated", "attention_weighted"],
  "cached_embeddings": ["vl_abc123_def4_L-2"]
}
```

---

### GET /api/vl/config

Get VL default parameters from server config.

**Response:**
```json
{
  "alpha": 0.3,
  "hidden_layer": -2,
  "auto_unload": true,
  "blend_mode": "linear"
}
```

---

### POST /api/vl/extract

Extract VL embeddings from an uploaded image. Returns a cache ID for use with `/api/vl/generate`.

**Request:**
```json
{
  "image": "base64_encoded_image_data",
  "text": "optional description with image",
  "hidden_layer": -2,
  "image_tokens_only": false,
  "scale_to_text": true
}
```

**Fields:**
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| image | string | Yes | - | Base64-encoded image (PNG, JPG) |
| text | string | No | null | Text description to process with image |
| hidden_layer | int | No | -2 | Hidden layer to extract (-1 to -6) |
| image_tokens_only | bool | No | false | Only extract image token embeddings |
| scale_to_text | bool | No | true | Scale embeddings to match text statistics |

**Response:**
```json
{
  "embeddings_id": "vl_abc123_def4_L-2",
  "num_tokens": 256,
  "shape": [256, 2560],
  "hidden_layer": -2,
  "original_std": 13.25,
  "scaled_std": 58.75,
  "extract_time": 2.34
}
```

---

### POST /api/vl/generate

Generate an image with VL conditioning.

**Request:**
```json
{
  "prompt": "A cat sleeping in sunlight",
  "vl_image": "base64_encoded_image",
  "vl_embeddings_id": null,
  "vl_alpha": 0.3,
  "vl_hidden_layer": -2,
  "vl_image_tokens_only": false,
  "vl_text": null,
  "vl_blend_mode": "linear",
  "width": 1024,
  "height": 1024,
  "steps": 9,
  "seed": null
}
```

**VL-specific Fields:**
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| vl_image | string | No | null | Base64 reference image (extracted on-the-fly) |
| vl_embeddings_id | string | No | null | Pre-extracted embeddings ID from /api/vl/extract |
| vl_alpha | float | No | 0.3 | VL influence: 0.0=text only, 1.0=VL only |
| vl_hidden_layer | int | No | -2 | Hidden layer for extraction |
| vl_image_tokens_only | bool | No | false | Only use image tokens |
| vl_text | string | No | null | Text description with reference image |
| vl_blend_mode | string | No | linear | Blend mode: linear, style_only, graduated, attention_weighted |

**Blend Modes:**
| Mode | Description |
|------|-------------|
| linear | Uniform interpolation (default) |
| style_only | Only blend style dimensions, preserve text content |
| graduated | More VL influence for later tokens |
| attention_weighted | Reduce VL for important text tokens (experimental) |

**Response:** PNG image stream (same as `/api/generate`)

**Headers:**
- `X-Generation-Time`: Generation time in seconds
- `X-Seed`: Seed used
- `X-VL-Alpha`: VL alpha value used
- `X-VL-Blend-Mode`: Blend mode used

---

### DELETE /api/vl/cache/{embeddings_id}

Clear a specific cached VL embedding.

**Response:**
```json
{
  "deleted": "vl_abc123_def4_L-2"
}
```

---

### DELETE /api/vl/cache

Clear all cached VL embeddings.

**Response:**
```json
{
  "cleared": 5
}
```

---

### VL Workflow Example

**Two-stage workflow (recommended for multiple generations with same reference):**

```python
import base64
import requests

# 1. Load and encode reference image
with open("reference.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# 2. Extract embeddings once
extract_resp = requests.post("http://localhost:7860/api/vl/extract", json={
    "image": image_b64,
    "hidden_layer": -2,
})
embeddings_id = extract_resp.json()["embeddings_id"]

# 3. Generate multiple images using cached embeddings
for seed in [42, 123, 456]:
    gen_resp = requests.post("http://localhost:7860/api/vl/generate", json={
        "prompt": "A cat sleeping in sunlight",
        "vl_embeddings_id": embeddings_id,
        "vl_alpha": 0.3,
        "vl_blend_mode": "style_only",
        "seed": seed,
    })
    with open(f"output_{seed}.png", "wb") as f:
        f.write(gen_resp.content)

# 4. Clean up
requests.delete(f"http://localhost:7860/api/vl/cache/{embeddings_id}")
```

**One-shot workflow (simpler but slower):**

```python
gen_resp = requests.post("http://localhost:7860/api/vl/generate", json={
    "prompt": "A cat sleeping in sunlight",
    "vl_image": image_b64,  # Extracted on-the-fly
    "vl_alpha": 0.3,
})
```

---

## Debug Mode

Run with `--debug` to enable detailed logging:

```bash
uv run web/server.py --model-path ... --debug
```

Debug output includes:
- Token count and first 20 token IDs
- Embedding shape, dtype
- Embedding stats (min, max, mean, std)
- First/last 5 embedding values

Useful for comparing embeddings between different backends (API vs local).
