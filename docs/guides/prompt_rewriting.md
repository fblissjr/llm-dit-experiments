# prompt rewriting guide

*last updated: 2026-02-06*

The loaded Qwen3 model can be used for prompt rewriting/expansion in addition to embedding extraction. This enables creative prompt enhancement without loading additional models.

## supported models

| Model | Input Types | Use Case |
|-------|-------------|----------|
| qwen3-4b | Text only | Standard prompt expansion |

## configuration

### toml config

```toml
[default.rewriter]
use_api = true                # Use API backend
api_url = "http://mac:8080"   # API endpoint
api_model = "Qwen3-4B"        # Model ID for text-only rewriting
temperature = 0.6             # Sampling temperature (Qwen3 thinking mode)
top_p = 0.95                  # Nucleus sampling
max_tokens = 512              # Max tokens to generate
timeout = 120.0               # API request timeout in seconds
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

1. Enter a basic prompt
2. Open "Prompt Rewriter (Qwen3)" section
3. Select a rewriter style
4. Click "Rewrite Prompt"
5. Click "Use This Prompt" to apply

### api

```bash
# List available rewriters
curl http://localhost:8000/api/rewriters

# Rewrite with text-only model
curl -X POST http://localhost:8000/api/rewrite \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cat", "rewriter": "rewriter_official"}'

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
