# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
