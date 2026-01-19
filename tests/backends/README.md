# Portable Test Backends

**Last Updated:** 2026-01-19

This module provides a unified interface for LTX-2 video generation tests that work with **both**:
- Our `llm_dit` implementation (this repo)
- Official LTX-2 implementation (Lightricks repo)

## Purpose

Enable 1:1 baseline comparison by running **identical tests** with either implementation.
Same prompts, same configs, same assertions → comparable outputs for visual inspection.

## Quick Start

### Run in llm-dit-experiments (our implementation)

```bash
# Smoke test (~30s)
uv run pytest tests/e2e/test_baseline_portable.py::TestBaselineSmoke -v -s

# Short test (~2min)
uv run pytest tests/e2e/test_baseline_portable.py::TestBaselineT2V::test_t2v_short -v -s

# Full reference test (~10min)
uv run pytest tests/e2e/test_baseline_portable.py::TestBaselineT2V::test_t2v_reference -v -s --runslow
```

### Run in LTX-2 repo (official implementation)

1. Copy the required files to LTX-2 repo:
   ```bash
   # From llm-dit-experiments root
   cp -r tests/backends /path/to/LTX-2/tests/
   cp tests/e2e/test_baseline_portable.py /path/to/LTX-2/tests/e2e/
   cp tests/e2e/conftest.py /path/to/LTX-2/tests/e2e/
   ```

2. Run tests:
   ```bash
   cd /path/to/LTX-2
   pytest tests/e2e/test_baseline_portable.py -v -s
   ```

3. Compare outputs:
   ```
   llm-dit-experiments/outputs/tests/baseline/llm_dit/
   LTX-2/outputs/tests/baseline/ltx2/
   ```

## Module Structure

```
tests/backends/
├── __init__.py           # Auto-detection and exports
├── protocol.py           # Interface definitions (GenerationConfig, Backend, etc.)
├── llm_dit_backend.py    # Our implementation backend
├── ltx2_backend.py       # Official LTX-2 backend
└── README.md             # This file
```

## API

### Getting a Backend

```python
from tests.backends import get_backend, get_backend_name

# Auto-detect available backend
backend_name = get_backend_name()  # "llm_dit", "ltx2", or "none"
backend = get_backend()

# Force specific backend via environment variable
# LLM_DIT_TEST_BACKEND=ltx2 pytest ...
```

### GenerationConfig

Canonical parameters matching official LTX-2 defaults:

```python
from tests.backends import GenerationConfig

config = GenerationConfig(
    num_frames=121,          # Must be 8k+1 (e.g., 9, 17, 25, ..., 121)
    height=512,              # Divisible by 32
    width=768,               # Divisible by 32
    frame_rate=24.0,
    num_inference_steps=40,
    guidance_scale=4.0,      # CFG scale
    seed=10,                 # Default LTX-2 seed
    fp8=True,                # Use FP8 quantization
)
```

### Generating Video

```python
result = backend.generate_video(
    prompt="A cat walking through a garden",
    config=config,
    output_dir=Path("outputs/test/"),
    save_video=True,
)

# Access results
print(f"Video shape: {result.video.shape}")  # [F, H, W, C] uint8
print(f"Total time: {result.stats.total_time:.1f}s")

# Files saved:
# - outputs/test/video.mp4
# - outputs/test/metadata.json
```

## Output Structure

```
outputs/tests/baseline/{backend}/
└── {test_name}_{timestamp}/
    ├── video.mp4           # Generated video
    ├── metadata.json       # Config + stats + prompt
    └── embedding_info.json # Text embedding details (for comparison tests)
```

## Test Configs

| Config | Frames | Resolution | Steps | CFG | Time | VRAM |
|--------|--------|------------|-------|-----|------|------|
| `get_smoke_config()` | 9 | 256x384 | 2 | 1.0 | ~30s | ~14GB |
| `get_short_config()` | 33 | 384x512 | 10 | 3.0 | ~2min | ~16GB |
| `get_reference_config()` | 121 | 512x768 | 40 | 4.0 | ~10min | ~20GB |

## Comparison Workflow

1. **Generate with both backends:**
   ```bash
   # In llm-dit-experiments
   uv run pytest tests/e2e/test_baseline_portable.py::TestBaselineT2V::test_t2v_reference -v -s --runslow

   # In LTX-2 repo (after copying files)
   pytest tests/e2e/test_baseline_portable.py::TestBaselineT2V::test_t2v_reference -v -s --runslow
   ```

2. **Compare outputs:**
   - Visual inspection: Watch both videos side-by-side
   - Stats comparison: Compare `metadata.json` files
   - For debugging: Compare `embedding_info.json` for text encoding differences

3. **Identify divergences:**
   - Same visual quality → Parity achieved
   - Different quality → Trace through intermediate values
   - Use debug checkpoints (set `LLM_DIT_DEBUG=1`)

## Backend Detection Priority

1. Environment variable `LLM_DIT_TEST_BACKEND` (if set)
2. `llm_dit` package available → use `llm_dit` backend
3. `ltx_pipelines` package available → use `ltx2` backend
4. `coderef/LTX-2/` exists → add to path and use `ltx2` backend

## Requirements

- CUDA GPU with 16GB+ VRAM (24GB recommended for reference tests)
- LTX-2 model weights at `models/LTX-2/`
  - `transformer/`
  - `text_encoder/`
  - `vae/`

## Notes

- Both backends use FP8 quantization by default for the transformer
- Sequential offloading enables 121-frame generation on 24GB GPUs
- Seed 10 is the official LTX-2 default for reproducibility
