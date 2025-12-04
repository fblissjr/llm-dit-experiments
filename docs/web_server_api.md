# Z-Image Web Server API

## Overview

The web server provides a REST API for Z-Image generation with the Qwen3-4B text encoder.

## Running the Server

```bash
# Install PyTorch separately (not pinned in dependencies)
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

# Sync dependencies
uv sync

# Run with local encoder (recommended for RTX 4090)
uv run web/server.py \
  --model-path /path/to/z-image-turbo \
  --text-encoder-device cpu \
  --dit-device cuda \
  --vae-device cuda \
  --port 7860

# Run with API encoder (distributed inference)
uv run web/server.py \
  --model-path /path/to/z-image-turbo \
  --api-url http://mac-host:8080 \
  --api-model Qwen3-4B \
  --dit-device cuda \
  --vae-device cuda
```

## CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model-path` | str | required | Path to Z-Image model |
| `--host` | str | 127.0.0.1 | Server host |
| `--port` | int | 7860 | Server port |
| `--text-encoder-device` | str | auto | Device for text encoder (cpu/cuda/mps/auto) |
| `--dit-device` | str | auto | Device for DiT transformer (cpu/cuda/mps/auto) |
| `--vae-device` | str | auto | Device for VAE (cpu/cuda/mps/auto) |
| `--api-url` | str | None | URL for heylookitsanllm API backend |
| `--api-model` | str | None | Model ID for API backend |
| `--local-encoder` | flag | False | Force local encoder when using API (for A/B testing) |
| `--debug` | flag | False | Enable debug logging |
| `--cpu-offload` | flag | False | Enable CPU offload for transformer |
| `--flash-attn` | flag | False | Enable Flash Attention |
| `--compile` | flag | False | Compile transformer with torch.compile |

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
  "enable_thinking": false,
  "template": null,
  "width": 1024,
  "height": 1024,
  "steps": 9,
  "seed": null,
  "guidance_scale": 0.0
}
```

**Fields:**
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| prompt | string | Yes | - | User message / image description |
| system_prompt | string | No | null | System message (e.g., "You are a painter.") |
| thinking_content | string | No | null | Content inside `<think>...</think>` |
| assistant_content | string | No | null | Content after `</think>` |
| enable_thinking | bool | No | false | Add `<think></think>` structure |
| template | string | No | null | Template name |
| width | int | No | 1024 | Image width (divisible by 16) |
| height | int | No | 1024 | Image height (divisible by 16) |
| steps | int | No | 9 | Denoising steps |
| seed | int | No | null | Random seed for reproducibility |
| guidance_scale | float | No | 0.0 | CFG scale (0.0 recommended for Z-Image) |

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
  "enable_thinking": false,
  "template": null
}
```

**Response:**
```json
{
  "shape": [10, 2560],
  "dtype": "torch.bfloat16",
  "encode_time": 0.123,
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
  "enable_thinking": true,
  "template": null
}
```

**Response:**
```json
{
  "formatted_prompt": "<|im_start|>system\nYou are a painter.<|im_end|>\n<|im_start|>user\nPaint a cat<|im_end|>\n<|im_start|>assistant\n<think>\nOrange fur.\n</think>\n",
  "char_count": 112,
  "prompt": "Paint a cat",
  "system_prompt": "You are a painter.",
  "thinking_content": "Orange fur.",
  "assistant_content": null,
  "template": null,
  "enable_thinking": true
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
  "enable_thinking": false
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

**With empty thinking block:**
```json
{"prompt": "Draw a house", "enable_thinking": true}
```
Result:
```
<|im_start|>user
Draw a house<|im_end|>
<|im_start|>assistant
<think>

</think>
```

**With thinking content:**
```json
{"prompt": "Paint a cat", "enable_thinking": true, "thinking_content": "Orange fur, green eyes."}
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
  "enable_thinking": true,
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
