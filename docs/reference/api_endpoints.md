# rest api reference

*last updated: 2026-02-14*

## config management endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/config/session` | GET | Get current session config values |
| `/api/config/session` | PUT | Update hot-reload safe parameters |
| `/api/config/profiles` | GET | List available config profiles |
| `/api/context` | GET | Get composite generation context (model, LoRA, VRAM, uptime) |
| `/api/server/restart` | POST | Restart server (optionally with new profile) |

## Z-Image endpoints (core router)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Generate image from prompt |
| `/api/generate/stream` | POST | Generate image with SSE progress streaming |
| `/api/img2img` | POST | Image-to-image generation |
| `/api/encode` | POST | Encode prompt to embeddings |
| `/api/format-prompt` | POST | Preview formatted prompt (no encoding) |
| `/api/save-embeddings` | POST | Save embeddings to file |
| `/api/templates` | GET | List available templates |
| `/api/rewriters` | GET | List available rewriter templates |
| `/api/rewrite` | POST | Rewrite prompt using Qwen3 model |
| `/api/dype/config` | GET | Get DyPE configuration |
| `/api/dype/status` | GET | Get DyPE status |

## FLUX.2 endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/flux2/generate` | POST | Generate image (sync) |
| `/api/flux2/generate/stream` | POST | Generate image with SSE progress streaming |
| `/api/flux2/status` | GET | Get FLUX.2 pipeline status |
| `/api/flux2/models/{model_name}` | GET | Get model metadata (fixed params, variants) |

## LTX-2 endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ltx2/generate/stream` | POST | Generate video with SSE progress streaming |
| `/api/ltx2/status` | GET | Get LTX-2 pipeline status |

## Qwen-Image endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/qwen-image/edit-layer` | POST | Edit image with text instructions (2511) |
| `/api/qwen-image/edit-multi` | POST | Multi-image editing (2511) |
| `/api/qwen-image/edit-status` | GET | Check if edit model is loaded |
| `/api/qwen-image-2512/generate` | POST | Text-to-image generation (2512) |
| `/api/qwen-image-2512/status` | GET | Get 2512 pipeline status |
| `/api/qwen-image-2512/config` | GET | Get 2512 configuration |

## pipeline management endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/pipelines` | GET | List all pipeline schemas |
| `/api/pipelines/{id}` | GET | Get schema for a specific pipeline |
| `/api/pipelines/{id}/defaults` | GET | Get defaults for a pipeline |
| `/api/presets` | GET | Get all presets across pipelines |
| `/api/presets/{id}` | GET | Get presets for a pipeline |
| `/api/presets/preset/{name}` | GET | Get a specific preset by name |

## model management endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/vram/status` | GET | Get VRAM usage breakdown |
| `/api/models/{id}/status` | GET | Get model status for a pipeline |
| `/api/models/{id}/load` | POST | Load a model for a pipeline |
| `/api/models/{id}/unload` | POST | Unload a model for a pipeline |
| `/api/models/unload-all` | POST | Unload all loaded models |
| `/api/loras` | GET | List available LoRA files (all pipelines) |
| `/api/loras/{id}` | GET | List LoRA files for a pipeline |

## config management endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/config/session` | GET | Get current session config values |
| `/api/config/session` | PUT | Update hot-reload safe parameters |
| `/api/config/profiles` | GET | List available config profiles |
| `/api/resolution-config` | GET | Get resolution validation constants and presets |
| `/api/rewriter-config` | GET | Get rewriter config (includes available models) |

## system endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/context` | GET | Get composite generation context (model, LoRA, VRAM, uptime) |
| `/api/server/restart` | POST | Restart server (optionally with new profile) |
| `/api/history` | GET | Get generation history |
| `/api/history/{index}` | DELETE | Delete specific history item |
| `/api/history` | DELETE | Clear all history |
| `/api/system/unload-fmtt` | POST | Unload FMTT model |
| `/api/system/clear-cache` | POST | Clear embedding cache |

## generate request fields

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
  "shift": 3.0,
  "slg_scale": 0.0,
  "slg_layers": [7, 8, 9, 10, 11, 12],
  "slg_start": 0.05,
  "slg_stop": 0.50,
  "dype": {
    "enabled": true,
    "method": "vision_yarn",
    "multipass": "twopass",
    "dype_scale": 2.0,
    "dype_exponent": 2.0,
    "base_shift": 0.5,
    "max_shift": 1.15,
    "pass2_strength": 0.5,
    "pass3_strength": 0.4,
    "frequency_modulation": false
  }
}
```

## dype configuration

The `dype` object enables high-resolution generation (2K+):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | false | Enable DyPE |
| `method` | string | "vision_yarn" | RoPE method: vision_yarn, yarn, ntk |
| `multipass` | string | "single" | Generation mode: single, twopass, threepass |
| `dype_scale` | float | 2.0 | Magnitude of DyPE effect (2.0 for 2K, 4.0 for 4K) |
| `dype_exponent` | float | 2.0 | Decay speed (2.0 = quadratic) |
| `base_shift` | float | 0.5 | Noise schedule shift at 1024px |
| `max_shift` | float | 1.15 | Noise schedule shift at max resolution |
| `pass2_strength` | float | 0.5 | Second pass img2img strength (0.3-0.8) |
| `pass3_strength` | float | 0.4 | Third pass strength (threepass only) |
| `frequency_modulation` | bool | false | Experimental timestep-based RoPE scaling |

## think block behavior

- If `thinking_content` is provided, a think block is automatically added
- If `force_think_block` is true, an empty think block is added even without content
- Default: no think block (matches official HF Space)

## content processing

- `strip_quotes`: Remove `"` characters from prompt (for JSON-type inputs, since Z-Image treats `"` as text to render)

## rewrite request fields

```json
{
  "prompt": "A cat",
  "model": "qwen3-4b",
  "rewriter": "rewriter_official",
  "image": "data:image/jpeg;base64,...",
  "temperature": 0.6,
  "top_p": 0.95,
  "max_tokens": 512
}
```

Available models:
- `qwen3-4b` - Text-only rewriting (default)

## qwen-image decompose request

```json
{
  "image": "data:image/png;base64,...",
  "prompt": "A detailed description of the image",
  "num_layers": 4,
  "steps": 50,
  "cfg_scale": 4.0,
  "resolution": 640
}
```

## config session response

```json
{
  "values": {
    "shift": 3.0,
    "d_noise": 1.0,
    "steps": 9,
    "guidance_scale": 0.0,
    "width": 1024,
    "height": 1024
  },
  "profile": "rtx4090",
  "modified": ["shift"],
  "config_file": "config.toml"
}
```

## config session update request

Only hot-reload safe parameters can be updated without restart:

```json
{
  "shift": 3.5,
  "d_noise": 0.98,
  "steps": 12
}
```

Response:

```json
{
  "success": true,
  "updated": ["shift", "d_noise", "steps"],
  "pending_restart": []
}
```

## hot-reload safe parameters

These parameters can be changed without server restart:
- `shift`, `d_noise`, `dynamic_shift`
- `steps`, `guidance_scale`, `height`, `width`
- `hidden_layer`, `layer_weights`, `long_prompt_mode`
- `dype_*`, `slg_*`, `fmtt_*` (feature parameters)
- `tiled_vae`, `tile_size`, `embedding_cache`

## parameters requiring restart

These require server restart to take effect:
- `model_path`, `text_encoder_path`, device placements
- `quantization`, `cpu_offload`
- `attention_backend`, `flash_attn`, `compile`
- `lora_paths`, `lora_scales`

## profiles response

```json
{
  "profiles": ["default", "rtx4090", "low_vram"],
  "current": "rtx4090",
  "config_file": "config.toml"
}
```

## generation context response

`GET /api/context` returns a composite status snapshot (camelCase JSON):

```json
{
  "uptimeSeconds": 3600,
  "profile": "rtx4090",
  "activePipeline": "zimage",
  "pipelineDisplayName": "Z-Image",
  "modelVariant": "mini",
  "loras": [{"name": "my_lora", "path": "/path/to/my_lora.safetensors", "scale": 0.8, "layersUpdated": 24}],
  "loraSummary": "my_lora @0.80 (24 layers)",
  "quantization": {"transformer": "float8_e4m3"},
  "compileEnabled": true,
  "compileMode": "default",
  "blockOffload": false,
  "vramUsedGb": 18.2,
  "vramTotalGb": 24.0,
  "vramPercent": 75.8,
  "pendingRestartFields": [],
  "sessionModifiedFields": ["shift"],
  "fmttCached": false,
  "historyCount": 5
}
```

## server restart request

```json
{
  "reason": "user_request",
  "new_profile": "low_vram"
}
```

Response:

```json
{
  "success": true,
  "message": "Server restarting..."
}
```
