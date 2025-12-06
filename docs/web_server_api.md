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

### LoRA Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--lora` | str | None | LoRA file path with optional scale (path:scale). Repeatable. |

Example: `--lora style.safetensors:0.8 --lora detail.safetensors:0.5`

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
  "shift": 3.0
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
