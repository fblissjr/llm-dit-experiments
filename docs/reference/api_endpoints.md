# rest api reference

*last updated: 2026-01-06*

## endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Generate image from prompt |
| `/api/encode` | POST | Encode prompt to embeddings |
| `/api/format-prompt` | POST | Preview formatted prompt (no encoding) |
| `/api/templates` | GET | List available templates |
| `/api/resolution-config` | GET | Get resolution validation constants and presets |
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
| `/api/qwen-image/decompose` | POST | Decompose image into layers (Qwen-Image-Layered) |
| `/api/qwen-image/edit-layer` | POST | Edit a decomposed layer with text instructions |
| `/api/qwen-image/edit-status` | GET | Check if edit model is loaded |
| `/api/qwen-image/config` | GET | Get Qwen-Image configuration |
| `/health` | GET | Health check |

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

## vl generate request fields

```json
{
  "prompt": "A cat in style of reference",
  "image": "data:image/jpeg;base64,...",
  "alpha": 0.3,
  "hidden_layer": -6,
  "text_tokens_only": false,
  "blend_mode": "linear",
  "width": 1024,
  "height": 1024,
  "steps": 9,
  "seed": 42
}
```

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
- `qwen3-vl` - Local VL model for image-based rewriting
- `qwen3-vl-api` - API VL model (e.g., qwen2.5-vl-72b-mlx)

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
