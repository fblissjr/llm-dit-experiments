# configuration reference

*last updated: 2026-03-03*

## config format

Uses `--profile` to select a config section from `config.toml`. The config and profile are server-side flags passed when starting the server:

```bash
# Start the server with a config profile
uv run web/server.py --config config.toml --profile rtx4090

# Then use gen.py to generate (talks to the running server)
uv run scripts/gen.py zimage --prompt "A cat"
uv run scripts/gen.py flux2 --prompt "A cat" --seed 42
```

> **Note:** The deprecated `scripts/generate.py` also accepted `--config` and `--profile` directly.

## dry configuration principles

All configurable parameters must flow through a single chain to prevent disconnected settings:

```
config.toml (TOML)     CLI flags (argparse)
        \                    /
         v                  v
     Config dataclass  →  RuntimeConfig.from_toml_config()
              \              /
               v            v
          RuntimeConfig (composed sub-configs)
                    |
                    v
            Actual usage (API requests, model loading)
```

## adding a new parameter

Only **2 touchpoints** required (was 6+ pre-refactor):

1. **Add to Config dataclass** (`src/llm_dit/config.py`) -- e.g., `Flux2Config`, `LTX2Config`, `EncoderConfig`
2. **Add to TOML config** (`config.toml`) in the appropriate section

`RuntimeConfig.from_toml_config()` picks up the new field automatically. CLI override is optional (add to `cli.py` if needed).

Validate with: `uv run pytest tests/unit/test_dry_config.py -v`

## files to check when adding parameters

| Layer | File | What to update |
|-------|------|----------------|
| Config classes | `src/llm_dit/config.py` | Add to pipeline dataclass |
| TOML config | `config.toml` | Add parameter in appropriate section |
| CLI parser (optional) | `src/llm_dit/cli.py` | Add CLI flag if command-line override is needed |
| Documentation (optional) | `docs/reference/cli_flags.md` | Document the CLI flag if added |

## anti-patterns to avoid

- Hardcoding defaults in multiple places instead of using Config dataclass defaults
- Adding a CLI flag without a corresponding Config dataclass field
- Exposing config in web UI without wiring through the pipeline

## automated verification

Run the DRY configuration consistency test after adding any new parameter:

```bash
uv run pytest tests/unit/test_dry_config.py -v
```

This test verifies:
- TOML parameters exist in Config dataclasses
- CLI arguments map to RuntimeConfig fields
- Critical parameters are wired through to backend configs
- Key parameters are documented

## config file format

Config file (TOML) is the source of truth. CLI flags override config values.

```toml
# config.toml.example structure

[default]
model_path = "/path/to/model"
width = 1024
height = 1024
steps = 9

[default.encoder]
device = "cpu"
hidden_layer = -2

[default.dit]
device = "cuda"

[default.vae]
device = "cuda"

[default.dype]
enabled = false
method = "vision_yarn"
scale = 2.0

[default.slg]
scale = 0.0
layers = [7, 8, 9, 10, 11, 12]  # Middle layers for Z-Image (30 layers)
start = 0.05
stop = 0.50  # Wider range for turbo model

[default.rewriter]
use_api = false
temperature = 0.6
top_p = 0.95
max_tokens = 512

[default.vl]
model_path = ""
device = "cpu"
default_alpha = 0.3
default_hidden_layer = -6

[rtx4090]
# Profile inherits from default, overrides specific values
long_prompt_mode = "attention_pool"

[low_vram]
cpu_offload = true
```

## profile inheritance

Profiles can override defaults. Common profiles:
- `default` - Basic setup
- `rtx4090` - Optimized for RTX 4090
- `low_vram` - CPU offload for limited VRAM
- `distributed` - API-based encoding for distributed inference

## hot-reload vs restart parameters

When changing config values via the web UI, parameters are classified by whether they require a server restart:

### hot-reload safe (immediate effect)

These can be changed at runtime without restart:
- `shift`, `d_noise`, `dynamic_shift`
- `steps`, `guidance_scale`, `height`, `width`
- `hidden_layer`, `layer_weights`, `long_prompt_mode`
- All `dype_*`, `slg_*`, `fmtt_*` parameters
- `tiled_vae`, `tile_size`, `embedding_cache`

### requires restart (model reload)

These require server restart to take effect:
- `model_path`, `text_encoder_path`, device placements
- `quantization`, `cpu_offload`
- `attention_backend`, `flash_attn`, `compile`
- `lora_paths`, `lora_scales`

See [config_management.md](../guides/config_management.md) for the UI guide.
