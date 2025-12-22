# prompt rewriting guide

*last updated: 2025-12-22*

The loaded Qwen3 model can be used for prompt rewriting/expansion in addition to embedding extraction. This enables creative prompt enhancement without loading additional models.

## supported models

| Model | Input Types | Use Case |
|-------|-------------|----------|
| qwen3-4b | Text only | Standard prompt expansion |
| qwen3-vl | Text, Image, or both | Image-based prompts, describe images (local) |
| qwen3-vl-api | Text, Image, or both | Image-based prompts via API (e.g., qwen2.5-vl-72b-mlx) |

## configuration

### toml config

```toml
[default.rewriter]
use_api = true                # Use API backend
api_url = "http://mac:8080"   # API endpoint
api_model = "Qwen3-4B"        # Model ID for text-only rewriting
vl_api_model = "qwen2.5-vl-72b-mlx"  # Model ID for VL rewriting via API
temperature = 0.6             # Sampling temperature (Qwen3 thinking mode)
top_p = 0.95                  # Nucleus sampling
max_tokens = 512              # Max tokens to generate
timeout = 120.0               # API request timeout in seconds
vl_enabled = true             # Allow VL model selection in rewriter UI
preload_vl = false            # Load Qwen3-VL at startup for rewriter
```

### cli flags

```bash
# Use API backend for rewriting
uv run web/server.py \
  --model-path /path/to/z-image \
  --rewriter-use-api \
  --rewriter-api-url http://mac:8080 \
  --rewriter-api-model Qwen3-4B \
  --rewriter-temperature 0.6

# Enable local VL rewriter with preloading
uv run web/server.py \
  --model-path /path/to/z-image \
  --vl-model-path /path/to/Qwen3-VL-4B-Instruct \
  --rewriter-preload-vl

# Enable API VL rewriter for larger models
uv run web/server.py \
  --model-path /path/to/z-image \
  --rewriter-use-api \
  --rewriter-vl-api-model qwen2.5-vl-72b-mlx \
  --rewriter-timeout 180.0
```

## rewriter templates

Place templates in `templates/z_image/rewriter/` with `category: rewriter` in frontmatter:

```markdown
---
name: rewriter_character_generator
description: Character Generator (prompt rewriter)
model: z-image
category: rewriter
---
You are an expert character designer...
```

## usage

### web ui

1. Enter a basic prompt (or skip for image-only with VL model)
2. Open "Prompt Rewriter (Qwen3)" section
3. Select a model (Qwen3-4B or Qwen3-VL if configured)
4. If using Qwen3-VL, optionally upload a reference image
5. Select a rewriter style
6. Click "Rewrite Prompt"
7. Click "Use This Prompt" to apply

### api

```bash
# List available rewriters
curl http://localhost:8000/api/rewriters

# Rewrite with text-only model
curl -X POST http://localhost:8000/api/rewrite \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cat", "rewriter": "rewriter_official"}'

# Rewrite with VL model and image
curl -X POST http://localhost:8000/api/rewrite \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-vl", "prompt": "Describe this image", "image": "data:image/jpeg;base64,..."}'

# Override generation parameters
curl -X POST http://localhost:8000/api/rewrite \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cat", "rewriter": "rewriter_official", "temperature": 0.8}'
```

### python

```python
# Using local encoder
backend = TransformersBackend.from_pretrained(...)
rewritten = backend.generate(
    prompt="A cat sleeping",
    system_prompt="You are an expert at writing image prompts...",
    max_new_tokens=512,
    temperature=0.6,
    top_p=0.95,
)

# Using API backend
backend = APIBackend.from_url("http://localhost:8000", "qwen3-4b")
rewritten = backend.generate(...)
```

## backend selection

- **Default**: Uses the local encoder's Qwen3 model
- **With `--rewriter-use-api`**: Uses a remote API backend (heylookitsanllm)
- The API URL defaults to `--api-url` but can be overridden with `--rewriter-api-url`

## vl model configuration

**Local VL:**
- Uses the same model path as VL conditioning (`vl.model_path`)
- Loads on-demand when user selects it in the UI
- Can optionally preload at startup with `--rewriter-preload-vl`

**API VL:**
- Configure with `rewriter.vl_api_model` in config.toml
- Image is sent as base64 data URL to OpenAI-compatible API
- Requires API server that supports vision models
