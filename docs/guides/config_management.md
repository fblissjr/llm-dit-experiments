# config management guide

*last updated: 2026-01-06*

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

## mobile access

The Config Management UI is designed to work on mobile devices, allowing you to:

- Adjust generation parameters from your phone
- Switch profiles remotely
- Monitor server status
- Restart the server if needed

Changes made from any device apply to all devices connecting to the same server.
