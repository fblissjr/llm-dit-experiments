# Distributed Inference Guide

Run the text encoder (Qwen3-4B) on one machine and the DiT/VAE pipeline on another. This is useful when:

- Your Mac or another machine has the LLM but no CUDA GPU for fast image generation
- Your CUDA server has limited VRAM and can't fit both the LLM and DiT
- You want to leverage MLX or llama.cpp on Apple Silicon for efficient text encoding

## Architecture

```
Mac (Apple Silicon)                    CUDA Server (RTX 4090, etc.)
+------------------+                   +---------------------------+
| heylookitsanllm  |  -- HTTP/JSON --> | llm-dit-experiments       |
| Qwen3-4B (MLX)   |  (embeddings)     | DiT + VAE                 |
+------------------+                   +---------------------------+
     |                                        |
     v                                        v
Text -> Embeddings [seq, 2560]         Embeddings -> Image
```

## Setup

### 1. Mac: Run heylookitsanllm

Install and configure heylookitsanllm with Qwen3-4B:

```bash
# Clone and install
git clone https://github.com/fblissjr/heylookitsanllm
cd heylookitsanllm
uv sync

# Import Qwen3-4B with encoder profile (optimized for hidden states)
uv run heylook models import --folder /path/to/models --profile encoder

# Or manually add to models.toml:
# [[models]]
# id = "Qwen3-4B-mlx"
# provider = "mlx"
# tags = ["encoder", "qwen"]
# [models.config]
# model_path = "/path/to/Qwen3-4B-mlx"
# cache_type = "standard"
# max_tokens = 2048
```

Start the server:

```bash
# Bind to all interfaces so CUDA server can reach it
uv run heylook --host 0.0.0.0 --port 8080

# Or with specific model
uv run heylook --host 0.0.0.0 --port 8080 --model Qwen3-4B-mlx
```

The hidden states endpoint will be available at `http://<mac-ip>:8080/v1/hidden_states`.

### 2. CUDA Server: Run llm-dit-experiments

```bash
# Clone and install
git clone https://github.com/fblissjr/llm-dit-experiments
cd llm-dit-experiments
uv sync

# Install PyTorch with CUDA (not pinned in pyproject.toml)
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

#### Option A: Web Server

```bash
uv run web/server.py \
  --api-url http://<mac-ip>:8080 \
  --api-model Qwen3-4B-mlx \
  --model-path /path/to/z-image-turbo \
  --dit-device cuda \
  --vae-device cuda \
  --host 0.0.0.0
```

Access the web UI at `http://<cuda-server-ip>:7860`.

#### Option B: CLI

```bash
uv run scripts/generate.py \
  --api-url http://<mac-ip>:8080 \
  --api-model Qwen3-4B-mlx \
  --model-path /path/to/z-image-turbo \
  --dit-device cuda \
  --vae-device cuda \
  --output image.png \
  "A sunset over mountains"
```

#### Option C: Config File

Create `config.toml`:

```toml
[default]
model_path = "/path/to/z-image-turbo"
templates_dir = "templates/z_image"

[default.api]
base_url = "http://<mac-ip>:8080"
model_id = "Qwen3-4B-mlx"
encoding_format = "base64"  # 73% smaller than JSON floats

[default.devices]
dit = "cuda"
vae = "cuda"

[default.generation]
width = 1024
height = 1024
steps = 9
```

Then run:

```bash
uv run web/server.py --config config.toml
# or
uv run scripts/generate.py --config config.toml "A landscape"
```

## Optimal heylookitsanllm Settings

### Model Configuration

For Qwen3-4B used as a text encoder, these settings matter:

| Setting | Recommended | Why |
|---------|-------------|-----|
| `cache_type` | `"standard"` | No KV cache quantization for max embedding precision |
| `max_tokens` | `2048` | High limit (though hidden states uses per-request `max_length`) |
| Model quantization | Full precision or mxfp4 | Lower quantization = slightly less precise embeddings |

These settings are **ignored** for hidden states (no text generation happens):
- `temperature`, `top_k`, `top_p`, `min_p`
- `repetition_penalty`

### Example models.toml Entry

```toml
[[models]]
id = "Qwen3-4B-mlx"
provider = "mlx"
description = "Qwen3-4B for Z-Image text encoding"
tags = ["encoder", "qwen"]
enabled = true

  [models.config]
  model_path = "/path/to/Qwen3-4B-mlx"
  vision = false
  cache_type = "standard"
  max_tokens = 2048
```

### Performance Tips

1. **Use base64 encoding** - Set `encoding_format = "base64"` in your config. This reduces network payload by ~73% (7MB vs 26MB for 512 tokens).

2. **Keep the model loaded** - heylookitsanllm keeps models in memory. First request loads the model, subsequent requests are fast.

3. **Network latency** - The hidden states request adds ~10-20ms network overhead. For batch workflows, this is negligible compared to image generation time.

4. **Full precision vs quantized** - Full precision Qwen3-4B gives slightly better embeddings than mxfp4 quantized, but both work well. Use quantized if memory is tight.

## Troubleshooting

### Connection refused

```
httpx.ConnectError: Connection refused
```

- Ensure heylookitsanllm is running with `--host 0.0.0.0` (not `127.0.0.1`)
- Check firewall rules allow port 8080
- Verify the IP address is correct

### Model not found

```
404: Model 'Qwen3-4B-mlx' not found
```

- Check model ID matches exactly (case-sensitive)
- Run `uv run heylook models list` to see available models
- Ensure the model is `enabled = true` in models.toml

### Hidden states shape mismatch

```
Expected hidden_dim 2560, got X
```

- Ensure you're using Qwen3-4B (not Qwen2 or other variants)
- Z-Image specifically requires Qwen3-4B's 2560-dimensional embeddings

### Slow first request

First request loads the model into memory (~5-10 seconds for 4B model). Subsequent requests are fast (~50-100ms for encoding).

## Distributed Prompt Rewriting

In addition to distributed encoding, you can also run prompt rewriting via the API. This uses the same heylookitsanllm server for text generation.

### Configuration

```toml
[default.rewriter]
use_api = true                    # Use API for rewriting
api_url = "http://<mac-ip>:8080"  # Can be same as encoding API
api_model = "Qwen3-4B-mlx"        # Model ID
temperature = 1.0
top_p = 0.95
max_tokens = 512
```

Or via CLI:

```bash
uv run web/server.py \
  --api-url http://<mac-ip>:8080 \
  --api-model Qwen3-4B-mlx \
  --model-path /path/to/z-image-turbo \
  --rewriter-use-api \
  --dit-device cuda
```

### How It Works

1. User submits a prompt to `/api/rewrite` on the CUDA server
2. The server calls heylookitsanllm's `/v1/chat/completions` endpoint
3. The Mac generates expanded prompt text
4. The expanded prompt is returned for use in image generation

This keeps the Mac's LLM busy with both encoding and rewriting, while the CUDA server handles DiT/VAE processing.

## API Reference

The hidden states endpoint used by this integration:

```
POST /v1/hidden_states

Request:
{
  "input": "<formatted prompt with chat template>",
  "model": "Qwen3-4B-mlx",
  "layer": -2,
  "max_length": 512,
  "encoding_format": "base64"
}

Response:
{
  "hidden_states": "<base64 encoded float32 array>",
  "shape": [seq_len, 2560],
  "model": "Qwen3-4B-mlx",
  "layer": -2,
  "dtype": "bfloat16",
  "encoding_format": "base64"
}
```

The `layer` parameter specifies which hidden layer to extract embeddings from:
- `-1`: Final layer (default for most models)
- `-2`: Penultimate layer (default for Z-Image, often better quality)
- Any positive int: Specific layer index (0-based)

This can be configured via `--hidden-layer` CLI flag or in config.toml.

See [heylookitsanllm documentation](https://github.com/fblissjr/heylookitsanllm) for full API details.
