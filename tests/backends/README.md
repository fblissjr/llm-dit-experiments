# portable test backends

*last updated: 2026-01-22*

Unified interface for LTX-2 video generation tests that work with **both**:
- Our `llm_dit` implementation (this repo)
- Official LTX-2 implementation (Lightricks repo)

## purpose

Enable 1:1 baseline comparison by running **identical tests** with either implementation.
Same prompts, same configs, same assertions -> comparable outputs for visual inspection.

## quick start

### run in llm-dit-experiments (our implementation)

```bash
# Smoke test (~30s)
uv run pytest tests/integration/pipeline/test_baseline_portable.py::TestBaselineSmoke -v -s

# Short test (~2min)
uv run pytest tests/integration/pipeline/test_baseline_portable.py::TestBaselineT2V::test_t2v_short -v -s

# Full reference test (~10min)
uv run pytest tests/integration/pipeline/test_baseline_portable.py::TestBaselineT2V::test_t2v_reference -v -s --runslow
```

### run in LTX-2 repo (official implementation)

1. Copy the required files to LTX-2 repo:
   ```bash
   # From llm-dit-experiments root
   cp -r tests/backends /path/to/LTX-2/tests/
   cp tests/integration/pipeline/test_baseline_portable.py /path/to/LTX-2/tests/e2e/
   cp tests/integration/pipeline/conftest.py /path/to/LTX-2/tests/e2e/
   ```

2. Run tests:
   ```bash
   cd /path/to/LTX-2
   pytest tests/integration/pipeline/test_baseline_portable.py -v -s
   ```

3. Compare outputs:
   ```
   llm-dit-experiments/outputs/tests/baseline/llm_dit/
   LTX-2/outputs/tests/baseline/ltx2/
   ```

## module structure

```
tests/backends/
├── __init__.py           # Auto-detection and exports
├── protocol.py           # Interface definitions (GenerationConfig, Backend)
├── llm_dit_backend.py    # Our implementation backend
├── ltx2_backend.py       # Official LTX-2 backend
├── diagnostics.py        # Debugging utilities
└── README.md             # This file
```

## api

### getting a backend

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
    num_inference_steps=30,
    guidance_scale=4.0,      # CFG scale
    seed=10,                 # Default LTX-2 seed
    fp8=True,                # Use FP8 quantization
)
```

### generating video

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

## output structure

```
outputs/tests/baseline/{backend}/
└── {test_name}_{timestamp}/
    ├── video.mp4           # Generated video
    ├── metadata.json       # Config + stats + prompt
    └── embedding_info.json # Text embedding details (for comparison tests)
```

## test configs

| Config | Frames | Resolution | Steps | CFG | Time | VRAM |
|--------|--------|------------|-------|-----|------|------|
| `get_smoke_config()` | 9 | 256x384 | 2 | 1.0 | ~30s | ~14GB |
| `get_short_config()` | 33 | 384x512 | 10 | 3.0 | ~2min | ~16GB |
| `get_reference_config()` | 121 | 512x768 | 40 | 4.0 | ~10min | ~20GB |

## comparison workflow

1. **Generate with both backends:**
   ```bash
   # In llm-dit-experiments
   uv run pytest tests/integration/pipeline/test_baseline_portable.py::TestBaselineT2V::test_t2v_reference -v -s --runslow

   # In LTX-2 repo (after copying files)
   pytest tests/integration/pipeline/test_baseline_portable.py::TestBaselineT2V::test_t2v_reference -v -s --runslow
   ```

2. **Compare outputs:**
   - Visual inspection: Watch both videos side-by-side
   - Stats comparison: Compare `metadata.json` files
   - For debugging: Compare `embedding_info.json` for text encoding differences

3. **Identify divergences:**
   - Same visual quality -> Parity achieved
   - Different quality -> Trace through intermediate values
   - Use debug checkpoints (set `LLM_DIT_DEBUG=1`)

## backend detection priority

1. Environment variable `LLM_DIT_TEST_BACKEND` (if set)
2. `llm_dit` package available -> use `llm_dit` backend
3. `ltx_pipelines` package available -> use `ltx2` backend
4. `coderef/LTX-2/` exists -> add to path and use `ltx2` backend

## requirements

- CUDA GPU with 16GB+ VRAM (24GB recommended for reference tests)
- LTX-2 model weights at `models/LTX-2/`
  - `transformer/`
  - `text_encoder/`
  - `vae/`

## code independence

| Path | Diffusers | coderef/LTX-2 |
|------|-----------|---------------|
| `src/llm_dit/encoders/` | No | No |
| `src/llm_dit/models/ltx2/` | No | No |
| `src/llm_dit/pipelines/generate.py` | No | No |
| `tests/backends/llm_dit_backend.py` | No | No |
| `tests/backends/ltx2_backend.py` | Yes | Yes (intentional) |

## notes

- Both backends use FP8 quantization by default for the transformer
- Sequential offloading enables 121-frame generation on 24GB GPUs
- Seed 10 is the official LTX-2 default for reproducibility
