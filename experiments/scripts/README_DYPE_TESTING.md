# DyPE Testing Guide

last updated: 2025-12-21

## Overview

This directory contains comprehensive test scripts for DyPE (Dynamic Position Extrapolation) validation. DyPE is a training-free technique that enables generating images at resolutions higher than the model's training resolution by dynamically adjusting RoPE frequencies based on the diffusion timestep.

## Test Scripts

### 1. `test_dype.py` - Comprehensive Test Script

Full-featured test script that validates DyPE at multiple resolutions, compares all three methods, and tests multipass mode.

**Features:**
- Tests 5 resolutions: 1K, 1.5K, 2K, 3K, 4K (square)
- Compares all three DyPE methods: vision_yarn, yarn, ntk
- Tests multipass mode for ultra-high resolution (4K)
- Generates comparison grids for visual inspection
- Measures generation time and peak VRAM
- Exports results to CSV and human-readable summary

**Usage:**

```bash
# Full test suite (all resolutions, all methods)
uv run experiments/scripts/test_dype.py \
    --model-path /path/to/z-image-turbo \
    --output results/dype_test

# Quick test (1024 and 2048 only)
uv run experiments/scripts/test_dype.py \
    --model-path /path/to/z-image-turbo \
    --quick

# Test specific method
uv run experiments/scripts/test_dype.py \
    --model-path /path/to/z-image-turbo \
    --method vision_yarn

# Test multipass mode only (two-pass at 4K)
uv run experiments/scripts/test_dype.py \
    --model-path /path/to/z-image-turbo \
    --multipass-only

# Use config file
uv run experiments/scripts/test_dype.py \
    --config config.toml \
    --profile rtx4090
```

**Output:**
- Individual images: `{resolution}_{method}.png`
- Comparison grids: `{resolution}_comparison.png`
- CSV results: `results.csv`
- Summary: `summary.txt`

### 2. `sweep_dype_params.sh` - Parameter Sweep

Shell script that sweeps `dype_scale` and `dype_exponent` to find optimal values.

**Parameters:**
- `dype_scale` (1.0, 1.5, 2.0, 2.5, 3.0): Controls magnitude of DyPE effect (lambda_s)
- `dype_exponent` (1.0, 1.5, 2.0, 2.5, 3.0): Controls decay speed (lambda_t, quadratic=2.0)

**Usage:**

```bash
# Full sweep (25 combinations: 5 scales x 5 exponents)
./experiments/scripts/sweep_dype_params.sh

# Quick sweep (9 combinations: 3 scales x 3 exponents)
./experiments/scripts/sweep_dype_params.sh --quick

# Use config file
./experiments/scripts/sweep_dype_params.sh \
    --config config.toml \
    --profile rtx4090

# Custom resolution
./experiments/scripts/sweep_dype_params.sh \
    --resolution 1536 \
    --model-path /path/to/z-image
```

**Output:**
- Baseline image: `baseline.png`
- Parameter sweep images: `dype_s{scale}_e{exponent}.png`
- CSV results: `sweep_results.csv`

**Create comparison grid:**
```bash
uv run experiments/compare.py grid \
    --input results/dype_sweep/TIMESTAMP \
    --cols 5
```

## Unit Tests

Location: `tests/unit/test_dype.py`

50 comprehensive unit tests covering all DyPE components:

**Test Coverage:**
- `DyPEConfig` dataclass (8 tests)
  - Default values, validation, serialization
  - Parameter clamping (dype_start_sigma)
  - to_dict/from_dict round-trip

- `compute_dype_shift` (4 tests)
  - Base/max resolution behavior
  - Linear interpolation
  - Edge cases

- `compute_k_t` (5 tests)
  - Timestep-dependent scaling
  - Different scales and exponents
  - Quadratic decay behavior

- `compute_mscale` (5 tests)
  - Amplitude scaling computation
  - Timestep interpolation
  - Scale factor effects

- `axis_token_span` (5 tests)
  - Integer and fractional positions
  - 2D position grids
  - Edge cases

- Vision YaRN helpers (4 tests)
  - `find_correction_factor`
  - `find_correction_range`
  - `linear_ramp_mask`
  - `find_newbase_ntk`

- Position embedding functions (6 tests)
  - `get_1d_ntk_pos_embed`
  - `get_1d_vision_yarn_pos_embed`
  - `get_1d_yarn_pos_embed`
  - DyPE modulation
  - Timestep progression

- `ZImageDyPERoPE` wrapper (4 tests)
  - Initialization
  - Timestep setting
  - Scale hint
  - Delegation to original embedder

- `DyPEPosEmbed` base class (4 tests)
  - Initialization
  - Base patch grid computation
  - Timestep setting

- Integration helpers (5 tests)
  - `patch_zimage_rope`
  - `set_zimage_timestep`
  - Scale hint computation
  - Error handling

**Run Tests:**

```bash
# Run all DyPE tests
uv run pytest tests/unit/test_dype.py -v

# Run specific test class
uv run pytest tests/unit/test_dype.py::TestDyPEConfig -v

# Run with coverage
uv run pytest tests/unit/test_dype.py --cov=llm_dit.utils.dype
```

**All tests pass:**
```
50 passed in 0.80s
```

## DyPE Configuration

DyPE can be configured via TOML config, CLI arguments, or Python API.

### TOML Config

```toml
[default.dype]
enabled = true
method = "vision_yarn"     # vision_yarn | yarn | ntk
dype_scale = 2.0           # Magnitude of DyPE effect (lambda_s)
dype_exponent = 2.0        # Decay speed (lambda_t, quadratic)
dype_start_sigma = 1.0     # When to start decay (0-1, 1.0=from start)
base_shift = 0.5           # Noise schedule shift at base resolution
max_shift = 1.15           # Noise schedule shift at max resolution
base_resolution = 1024     # Training resolution
anisotropic = false        # Use per-axis scaling for extreme aspect ratios
```

### CLI Arguments

```bash
uv run scripts/generate.py \
    --model-path /path/to/z-image \
    --width 2048 --height 2048 \
    --dype \
    --dype-method vision_yarn \
    --dype-scale 2.0 \
    --dype-exponent 2.0 \
    "Homer Simpson eating a donut"
```

### Python API

```python
from llm_dit.pipelines.z_image import ZImagePipeline
from llm_dit.utils.dype import DyPEConfig

pipe = ZImagePipeline.from_pretrained("/path/to/z-image")

dype_config = DyPEConfig(
    enabled=True,
    method="vision_yarn",
    dype_scale=2.0,
    dype_exponent=2.0,
)

result = pipe(
    prompt="Homer Simpson eating a donut",
    width=2048,
    height=2048,
    num_inference_steps=9,
    dype_config=dype_config,
)
```

## DyPE Methods

### 1. Vision YaRN (Recommended)

Dual-mask blending of three frequency baselines:
- Linear scaling for low frequencies (aspect ratio robust)
- NTK scaling for mid frequencies (extends range)
- Original frequencies for high frequencies (stability)

**Best for:** Most use cases, especially non-square aspect ratios

### 2. YaRN

Simpler single-scale approach with dual masks.

**Best for:** Experimentation, comparing to Vision YaRN

### 3. NTK

Simple NTK scaling of the RoPE theta base with DyPE modulation.

**Best for:** Baseline comparison

## Expected Results

Based on ComfyUI-DyPE implementation and early testing:

**1024x1024 (Baseline):**
- No extrapolation needed
- DyPE should have minimal effect

**2048x2048 (2x):**
- Moderate extrapolation
- All methods should work well
- Vision YaRN typically produces best details

**4096x4096 (4x):**
- Heavy extrapolation
- Vision YaRN recommended
- Consider multipass mode for best quality

## Troubleshooting

### VRAM Issues

If you run out of VRAM at high resolutions:

1. Use CPU offload:
   ```bash
   uv run scripts/generate.py \
       --model-path /path/to/z-image \
       --text-encoder-device cpu \
       --dit-device cuda \
       --vae-device cuda \
       --dype \
       --width 2048 --height 2048 \
       "Prompt here"
   ```

2. Use tiled VAE for very large images:
   ```bash
   uv run scripts/generate.py \
       --model-path /path/to/z-image \
       --dype \
       --tiled-vae \
       --tile-size 512 \
       --width 4096 --height 4096 \
       "Prompt here"
   ```

3. Use multipass mode:
   ```bash
   # First pass at 2K
   uv run scripts/generate.py \
       --model-path /path/to/z-image \
       --dype \
       --width 2048 --height 2048 \
       --output pass1.png \
       "Prompt here"

   # Second pass at 4K using img2img
   uv run scripts/generate.py \
       --model-path /path/to/z-image \
       --dype \
       --img2img pass1.png \
       --width 4096 --height 4096 \
       --strength 0.5 \
       --output pass2.png \
       "Prompt here"
   ```

### Quality Issues

If generated images have artifacts or quality problems:

1. Try different methods:
   - `vision_yarn`: Best overall quality
   - `yarn`: Simpler, may work better for some prompts
   - `ntk`: Baseline, good for comparison

2. Adjust parameters:
   - Lower `dype_scale` (1.5 instead of 2.0) for less aggressive extrapolation
   - Higher `dype_exponent` (2.5 instead of 2.0) for slower decay

3. Check resolution is divisible by 16:
   ```python
   from llm_dit.constants import validate_resolution

   is_valid, error = validate_resolution(2048, 2048)
   if not is_valid:
       print(f"Invalid resolution: {error}")
   ```

## Performance Benchmarks

Expected timings on RTX 4090 (approximate):

| Resolution | DyPE Method | Time | Peak VRAM |
|------------|-------------|------|-----------|
| 1024x1024  | Baseline    | 3s   | 6 GB      |
| 1024x1024  | vision_yarn | 3s   | 6 GB      |
| 2048x2048  | Baseline    | 10s  | 12 GB     |
| 2048x2048  | vision_yarn | 10s  | 12 GB     |
| 4096x4096  | vision_yarn | 40s  | 24 GB     |

Note: Actual timings vary based on hardware, model, and settings.

## References

- ComfyUI-DyPE: https://github.com/Haidra-Org/ComfyUI-DyPE
- Vision YaRN paper: "YaRN: Efficient Context Window Extension of Large Language Models"
- NTK scaling: "Neural Tangent Kernel: Convergence and Generalization in Neural Networks"

## Next Steps

After running DyPE tests:

1. Review generated images in output directory
2. Compare quality across methods and resolutions
3. Identify optimal parameters for your use case
4. Check CSV results for performance metrics
5. Experiment with different prompts and seeds
6. Try multipass mode for ultra-high resolutions

## Contributing

When adding new DyPE features or tests:

1. Add unit tests to `tests/unit/test_dype.py`
2. Run all tests: `uv run pytest tests/unit/test_dype.py -v`
3. Update this README with new usage examples
4. Update `CLAUDE.md` if adding new configuration parameters
