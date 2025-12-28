# configuration reference

*last updated: 2025-12-27*

## config formats

Two configuration formats are supported:

### legacy format (profile-based)

Uses `--profile` to select a config section. Each profile duplicates all settings.

```bash
uv run scripts/generate.py --config config.toml --profile rtx4090 "A cat"
```

### modular format (component-based)

Uses `--config-name` to select a combined config. Reduces duplication through:
- Component library (encoders, transformers, VAEs, schedulers defined once)
- Model type inheritance (e.g., `qwenimageedit` inherits from `qwenlayered`)
- Hardware profiles separate from model definitions
- Named quantization presets

```bash
uv run scripts/generate.py --config config_modular.toml --config-name rtx4090_zimage "A cat"
```

See `config_modular.toml.example` for the full modular format specification.

## dry configuration principles

All configurable parameters must flow through a single chain to prevent disconnected settings:

```
config.toml (TOML)     CLI flags (argparse)
        \                    /
         v                  v
     Config dataclass  →  RuntimeConfig
              \              /
               v            v
            PipelineLoader / startup.py
                    |
                    v
            Backend configs (APIBackendConfig, etc.)
                    |
                    v
            Actual usage (API requests, model loading)
```

## adding a new parameter

1. **Add to TOML config** (`config.toml.example`) in appropriate section
2. **Add to Config dataclass** (`src/llm_dit/config.py`) - e.g., `EncoderConfig`, `RewriterConfig`
3. **Add CLI argument** (`src/llm_dit/cli.py`) in `create_argument_parser()`
4. **Add to RuntimeConfig** (`src/llm_dit/cli.py`) with same name
5. **Wire in load_runtime_config()** - load from TOML config, allow CLI override
6. **Wire in startup.py** - pass to backend configs (`APIBackendConfig`, etc.)
7. **Expose in web UI** if user-facing (web/index.html, server endpoints)
8. **Document in docs/reference/cli_flags.md** - add to appropriate table

## files to check when adding parameters

| Layer | File | What to update |
|-------|------|----------------|
| TOML schema | `config.toml.example` | Add parameter with comment |
| Config classes | `src/llm_dit/config.py` | Add to dataclass, `to_dict()` |
| CLI parser | `src/llm_dit/cli.py` | `create_argument_parser()` |
| Runtime config | `src/llm_dit/cli.py` | `RuntimeConfig`, `load_runtime_config()` |
| Pipeline loading | `src/llm_dit/startup.py` | Pass to backend configs |
| API backend | `src/llm_dit/backends/api.py` | `APIBackendConfig` if API-relevant |
| Web server | `web/server.py` | Endpoint parameters |
| Web UI | `web/index.html` | Form fields if user-facing |
| Documentation | `docs/reference/cli_flags.md` | CLI flags table |

## anti-patterns to avoid

- Adding a parameter to CLI but not wiring it through `startup.py` to the actual backend that uses it (e.g., `hidden_layer` must reach `APIBackendConfig`)
- Hardcoding defaults in multiple places instead of using RuntimeConfig defaults
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

## modular config structure

The modular config format (`config_modular.toml`) has these sections:

```toml
# Hardware profiles - device placement + optimizations
[profiles.rtx4090]
dtype = "bfloat16"
attention_backend = "flash_attn_2"
[profiles.rtx4090.devices]
encoder = "cuda"
dit = "cuda"
vae = "cuda"

# Component library - define once, reference by ID
[encoders.qwen3_4b]
model_id = "Qwen/Qwen3-4B"
hidden_dim = 2560

[transformers.zimage_dit]
num_layers = 40

# Model types - pipeline + component refs + defaults
[models.zimage]
pipeline_class = "ZImagePipeline"
text_encoder = "qwen3_4b"  # Reference to [encoders.qwen3_4b]
dit = "zimage_dit"         # Reference to [transformers.zimage_dit]
default_steps = 9

[models.qwenimageedit]
inherits = "qwenlayered"   # Inheritance from another model
pipeline_class = "QwenImageEditPipeline"
default_steps = 40

# Quantization presets
[quantization.balanced]
text_encoder = "8bit"
transformer = "none"

# Feature configs
[features.dype]
enabled = false
method = "vision_yarn"

# Combined configs - final assemblies
[configs.rtx4090_zimage]
profile = "rtx4090"
model = "zimage"
quantization = "balanced"
features = []
```

Key files:
- `src/llm_dit/config_modular.py` - Modular config dataclasses and TOML parsing
- `config_modular.toml.example` - Complete example with all sections
- `tests/unit/test_config_modular.py` - 43 unit tests for modular config
