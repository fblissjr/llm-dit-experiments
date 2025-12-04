# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0]

### Added
- Runtime LoRA management with reversible loading/unloading (no model reload required)
  - ComfyUI-style backup+patch system: original weights backed up to CPU RAM
  - Per-LoRA strength control (0.0-2.0 range)
  - Trigger words automatically prepended to prompts
  - Load order control (order matters for stacking)
  - Directory scanning for LoRA discovery
- New CLI flags:
  - `--loras-dir PATH` - Directory to scan for LoRA files
  - `--lora-trigger WORDS` - Trigger words for preceding `--lora` flag
- New REST API endpoints:
  - `GET /api/loras` - List available LoRAs from configured directory
  - `GET /api/loras/active` - Get currently active LoRAs with settings
  - `POST /api/loras/apply` - Apply LoRA configuration
  - `POST /api/loras/clear` - Remove all LoRAs and restore base weights
- Web UI LoRA management panel:
  - Collapsible LoRA Settings section
  - Add LoRAs from dropdown of available files
  - Per-LoRA controls: enable/disable, strength slider, trigger words input
  - Apply/Clear buttons for batch operations
- LoRAManager class with thread-safe operations
- LoRAEntry dataclass for LoRA configuration
- Unit tests for LoRAManager

### Changed
- LoRA loading is now reversible (was permanent fusion)
- Config structure updated: `[default.lora]` now supports `loras_dir` and `[[entries]]`

## [0.3.0]

### Added
- Multiple scheduler support: `flow_euler` (default), `flow_heun`, `dpm_solver`, `unipc`
  - CLI: `--scheduler flow_heun`
  - Config: `[default.scheduler] type = "flow_heun"`
  - Web UI: Scheduler dropdown in "Scheduler Options"
  - Runtime: `pipeline.set_scheduler("flow_heun")`
- New `/api/schedulers` endpoint lists available scheduler types with descriptions
- Scheduler selection persisted in generation history

### Changed
- Scheduler shift description updated to clarify it applies to flow matching schedulers

## [0.2.0]

### Added
- Unified CLI architecture: web server and generate script share argument parser
- Scheduler shift parameter (`--shift`, default 3.0) for timestep schedule control
- LoRA support (`--lora path:scale`, stackable) with automatic weight fusion
- Device placement flags in generate.py (`--text-encoder-device`, `--dit-device`, `--vae-device`)
- Optimization flags in generate.py (`--cpu-offload`, `--flash-attn`, `--compile`)
- Prompt control flags in generate.py (`--system-prompt`, `--thinking-content`, `--assistant-content`)
- Config file sections: `[optimization]`, `[scheduler]`, `[lora]`
- Mobile-friendly web UI with all generation settings exposed
- New utility module `src/llm_dit/utils/lora.py` for LoRA loading
- Transformers v5 compatibility: `quantization_config` API support

### Changed
- Config file is now source of truth; CLI flags override config values
- Renamed `--no-thinking` to `--enable-thinking` (positive semantics)
- `torch_dtype` now honored from config (was hardcoded bfloat16)
- Web UI redesigned with collapsible sections and responsive layout
- **Breaking**: `EncoderConfig.load_in_8bit`/`load_in_4bit` deprecated in favor of `quantization` field
  - Use `quantization = "none" | "4bit" | "8bit"` instead
  - Legacy fields still work but emit deprecation warnings
  - Config automatically migrates to new API
- `TransformersBackend.from_pretrained()` now accepts `quantization_config` parameter
- `config.toml` updated: use `quantization = "8bit"` instead of `load_in_8bit = true`

### Fixed
- CLI/Web feature parity - all flags available in both interfaces
- Template formatter closing tag logic now matches ComfyUI-QwenImageWanBridge patterns exactly
- Fixed `<|im_end|>` handling: now only omits when assistant content is empty AND is_final=True

### Migration Guide (transformers v5)

The `load_in_8bit` and `load_in_4bit` parameters are deprecated in transformers v5.
Update your configuration as follows:

**config.toml (before):**
```toml
[default.encoder]
load_in_8bit = true
```

**config.toml (after):**
```toml
[default.encoder]
quantization = "8bit"  # or "4bit" or "none"
```

**Python code (before):**
```python
config = EncoderConfig(load_in_8bit=True)
```

**Python code (after):**
```python
config = EncoderConfig(quantization="8bit")

# Or use BitsAndBytesConfig directly:
from transformers import BitsAndBytesConfig
quant_config = BitsAndBytesConfig(load_in_8bit=True)
backend = TransformersBackend.from_pretrained(
    model_path,
    quantization_config=quant_config,
)
```

The legacy API still works but will emit deprecation warnings.

## [0.1.0]

### Added
- Initial release
- ZImageTextEncoder with template support (140+ templates)
- ZImagePipeline for end-to-end generation
- Distributed inference support (API backend for remote encoding)
- Per-component device placement (encoder/DiT/VAE independent)
- Web UI and REST API
- Generation history management
- Qwen3 chat template formatter with thinking block support
