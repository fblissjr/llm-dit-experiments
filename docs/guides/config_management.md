# config management guide

*last updated: 2026-01-24*

The web UI includes a Config Management interface for editing generation parameters, managing profiles, and controlling the server without using the CLI.

## overview

The Settings modal (gear icon) has four tabs:

| Tab | Purpose |
|-----|---------|
| Status | GPU memory, pipeline status, cache info |
| Config | Edit session generation defaults |
| Profiles | View and load config profiles |
| Server | Server status, restart controls |

## session config

The Config tab lets you edit generation defaults that apply to all subsequent generations in the current session.

### editable parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| Shift | 1.0 - 6.0 | Scheduler shift (default 3.0) |
| D-Noise | 0.8 - 1.2 | Denoising factor (default 1.0) |
| Steps | 1 - 20 | Generation steps (default 9) |
| Guidance Scale | 0.0 - 10.0 | CFG scale (default 0.0 for Z-Image) |

### workflow

1. Open Settings modal (gear icon)
2. Click "Config" tab
3. Adjust sliders as needed
4. Click "Apply to Session" to activate changes
5. Changes persist until server restart

The "Reset" button reverts sliders to their original values from the config file.

### session vs persistent changes

- **Session changes**: Apply immediately, reset on server restart
- **Persistent changes**: Modify config.toml directly (requires restart)

The UI shows which profile is currently loaded and whether any fields have been modified from their file values.

## profiles

The Profiles tab lists all profiles defined in your config.toml:

```toml
[default]
steps = 9
shift = 3.0

[rtx4090]
steps = 9
long_prompt_mode = "attention_pool"

[low_vram]
cpu_offload = true
```

### loading a profile

Click "Load" next to a profile name. This triggers a server restart with the new profile.

## server control

The Server tab shows:

- **Status**: Running/restarting
- **Uptime**: Time since server started
- **Profile**: Currently loaded profile
- **Pending Changes**: Parameters requiring restart

### restart server

Click "Restart Server" to restart with the current or a different profile. Use the profile dropdown to select a specific profile.

Restarting interrupts any active generation.

## hot-reload vs restart

Parameters are classified by whether they can be changed without restarting:

### hot-reload safe (immediate effect)

- `shift`, `d_noise`, `dynamic_shift`
- `steps`, `guidance_scale`, `height`, `width`
- `hidden_layer`, `layer_weights`, `long_prompt_mode`
- All `dype_*`, `slg_*`, `fmtt_*` parameters
- `tiled_vae`, `tile_size`, `embedding_cache`

### requires restart (model reload)

- `model_path`, `text_encoder_path`, device placements
- `quantization`, `cpu_offload`
- `attention_backend`, `flash_attn`, `compile`
- `lora_paths`, `lora_scales`

The Server tab shows any pending restart-required changes.

## api endpoints

For programmatic access:

```bash
# Get current session config
curl http://localhost:8000/api/config/session

# Update session parameters
curl -X PUT http://localhost:8000/api/config/session \
  -H "Content-Type: application/json" \
  -d '{"shift": 3.5, "steps": 12}'

# List profiles
curl http://localhost:8000/api/config/profiles

# Get server status
curl http://localhost:8000/api/server/status

# Restart server
curl -X POST http://localhost:8000/api/server/restart \
  -H "Content-Type: application/json" \
  -d '{"new_profile": "rtx4090"}'
```

See [api_endpoints.md](../reference/api_endpoints.md) for full API reference.

## toml configuration structure

The config.toml file contains model-specific sections and generation defaults. Profile-specific overrides are declared with the format `[profile_name.section]`.

### general section

```toml
[default]
# Controls which model loads at server startup
default_pipeline = "none"  # Options: none, z-image, qwen-image, flux2, ltx2

# Common generation parameters
steps = 9
guidance_scale = 0.0
shift = 3.0
d_noise = 1.0
```

The `default_pipeline` setting determines which model pipeline automatically loads when the web server starts. Set to "none" to start without preloading any model (saves memory).

### flux.2 section

```toml
[default.flux2]
model_path = "/path/to/FLUX.2-klein/FLUX.2-klein-9b-fp8"  # Transformer weights
vae_path = "/path/to/FLUX.2-klein/FLUX.2-klein-9B"        # VAE weights (from full model)
default_model = "klein-9b-fp8"   # Default model variant
block_offload = true             # Enable block-by-block GPU offload for 24GB VRAM
default_steps = 4                # Default inference steps (4 for distilled, 50 for base)
default_guidance = 1.0           # Default CFG scale (1.0 for distilled, 4.0 for base)
```

FLUX.2 configuration fields:

| Field | Type | Description |
|-------|------|-------------|
| `model_path` | string | Path to FLUX.2 transformer weights (empty string = HuggingFace auto-download) |
| `vae_path` | string | Path to FLUX.2 VAE weights (empty string = HuggingFace auto-download) |
| `default_model` | string | Default variant: klein-9b-fp8, klein-9b, klein-4b-fp8, klein-4b, klein-base-9b-fp8, etc. |
| `block_offload` | boolean | Enable block-by-block GPU offloading (reduces VRAM to ~12-15GB, slower) |
| `default_steps` | integer | Denoising steps (4 for distilled, 50 for base models) |
| `default_guidance` | float | CFG scale (1.0 for distilled, 4.0 for base models) |

### configuration fallback pattern

The system uses a three-tier fallback for configuration values:

1. **TOML config file** - Values defined in config.toml (highest priority)
2. **RuntimeConfig object** - In-memory session config with hot-reload support
3. **Server endpoint defaults** - Hardcoded fallback values when neither TOML nor RuntimeConfig provides a value

Example for FLUX.2 model paths:
```python
# In web/server.py
model_path = request.model_path or \
             getattr(runtime_config.flux2, "model_path", None) or \
             ""  # Empty string triggers HuggingFace download
```

This allows:
- Using custom local models via TOML config
- Overriding paths via web UI (RuntimeConfig)
- Falling back to HuggingFace auto-download if no path specified

### profile-specific overrides

Create hardware-specific profiles:

```toml
[rtx4090]
default_pipeline = "flux2"  # Auto-load FLUX.2 on startup

[rtx4090.flux2]
model_path = "/models/FLUX.2-klein/FLUX.2-klein-9b-fp8"
vae_path = "/models/FLUX.2-klein/FLUX.2-klein-9B"
default_model = "klein-9b-fp8"
block_offload = false  # 24GB VRAM, no offload needed
default_steps = 4
default_guidance = 1.0

[rtx3080]
default_pipeline = "flux2"

[rtx3080.flux2]
model_path = "/models/FLUX.2-klein/FLUX.2-klein-4b-fp8"
vae_path = "/models/FLUX.2-klein/FLUX.2-klein-4B"
default_model = "klein-4b-fp8"
block_offload = true  # 16GB VRAM, need offload
default_steps = 4
default_guidance = 1.0
```

Load a profile with:
```bash
uv run web/server.py --profile rtx4090
```

## mobile access

The Config Management UI is designed to work on mobile devices, allowing you to:

- Adjust generation parameters from your phone
- Switch profiles remotely
- Monitor server status
- Restart the server if needed

Changes made from any device apply to all devices connecting to the same server.
