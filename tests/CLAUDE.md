# testing guide for agents

*last updated: 2026-02-13*

Quick reference for LLM agents running tests in this codebase.

## e2e testing standard (API-first)

**Non-negotiable: all E2E tests go through the API** via `TestClient` (in-process HTTP).

This validates the full stack: Pydantic validation, router logic, dependency injection, ModelManager, and pipeline execution. Tests that call pipelines directly are classified as integration tests, not E2E.

### quick start
```bash
# Run FLUX.2 API smoke tests
uv run pytest tests/e2e/api/test_flux2_smoke.py -v -s

# Run all API E2E tests
uv run pytest tests/e2e/api/ -v -s

# Collect only (no GPU, verify imports)
uv run pytest tests/e2e/api/ --collect-only
```

### how it works
- Config overlays in `tests/configs/*.toml` are deep-merged over `config.toml.example`
- Model paths come from the real `config.toml`
- Each test module declares its overlay via `config_overlay` fixture
- `run_recorder` captures all metadata to `outputs/tests/runs/`

### every e2e test must
- [ ] Use `run_recorder` fixture (not direct pipeline calls)
- [ ] Send HTTP requests to real API endpoints
- [ ] Config loaded from TOML overlay in `tests/configs/`
- [ ] Assert on response status code
- [ ] Save output via `save_output()`
- [ ] Run validation via `validate()`
- [ ] Assert `validation.passed`

### anti-patterns (NEVER do these)

- **NEVER use `requests.post()`/`requests.get()` to an external server in tests.** This requires a running server, breaks CI, and doesn't validate the full FastAPI stack. Always use `TestClient` (in-process HTTP).
- **NEVER call pipeline functions directly in E2E tests.** `generate_video_with_offloading()`, `generate_video_two_stage()`, etc. are for integration tests only. E2E tests must go through the API.
- **NEVER bypass `run_recorder`.** It captures metadata, outputs, and logs. Without it, test artifacts are lost.

If a test needs to hit a real HTTP endpoint, it goes through `TestClient` which runs the full FastAPI app in-process -- same routers, same middleware, same DI, no external server needed.

**Gold standard pattern** (from `test_flux2_smoke.py`):
```python
@pytest.fixture(scope="module")
def config_overlay():
    return "flux2_smoke"

class TestFlux2APISmoke:
    def test_basic_generation(self, run_recorder):
        response = run_recorder.post("/api/flux2/generate", json={...})
        assert response.status_code == 200
        output_path = run_recorder.save_output(response.json())
        result = run_recorder.validate(output_path, expected_w=256, expected_h=256)
        assert result.passed
```

### full reference
See [tests/e2e/api/README.md](e2e/api/README.md) for architecture, config factory, validation thresholds, and step-by-step guide for adding new tests.

## core principle

**ALWAYS rely on retrieval and search over assumptions.**

Before writing tests, search for existing patterns:
```bash
grep -rn "class Test.*<Feature>" tests/
grep -rn "def test_.*<feature>" tests/
ls tests/baselines/   # Use existing baseline infrastructure
ls presets/testing/   # Use existing preset configs
```

## shared test infrastructure (USE THIS)

**Do NOT create new test frameworks. Use the existing infrastructure:**

| Infrastructure | Location | Purpose |
|----------------|----------|---------|
| **Constants** | `tests/constants/<pipeline>.py` | Canonical parameter values (single source of truth) |
| **Presets** | `presets/testing/<pipeline>_*.md` | Test configs (YAML frontmatter) |
| **Baselines** | `tests/baselines/` | Generation and comparison utilities |
| **Backends** | `tests/backends/` | Portable test protocol (derives from constants) |

### generating baselines
```python
from tests.baselines import generate_baseline, generate_baseline_from_preset

# Use preset (preferred)
result = generate_baseline_from_preset("ltx2_smoke_test")

# Or use config tier
result = generate_baseline(config_tier="smoke", seed=42)
```

### new feature test checklist
When adding tests for a new feature, cover:
- [ ] **Loading** - Feature loads without errors
- [ ] **Generation** - Feature works in full pipeline
- [ ] **Parameter effects** - Changing values changes output
- [ ] **Error handling** - Invalid inputs handled gracefully

## tdd workflow (test-driven development)

**Always write tests first when implementing new features or fixing bugs.**

### the cycle

```
1. Write test  ->  2. Watch it fail  ->  3. Implement  ->  4. Tests pass  ->  5. Commit
```

### step by step

1. **Write a test first** for each change (grouped logically in test files)
   ```bash
   # Create or edit test file
   # tests/unit/test_my_feature.py
   ```

2. **Run the specific test** to watch it fail
   ```bash
   uv run pytest tests/unit/test_my_feature.py::test_specific_case -v
   ```

3. **Implement the change** to make the test pass

4. **Run tests** to confirm they pass
   ```bash
   uv run pytest tests/unit/test_my_feature.py -v
   ```

5. **Run related tests** to ensure no regressions
   ```bash
   uv run pytest tests/unit/ -v -k "related_component"
   ```

6. **Commit** implementation + tests + docs together (single commit)

### code execution standards

```bash
# Always use uv for Python execution
uv run python -c "from llm_dit.models.ltx2 import ..."
uv run pytest tests/unit/test_foo.py -v

# Add dependencies
uv add package_name
uv add pytest-mock --dev  # dev dependencies

# Sync environment
uv sync
```

### testing best practices

- **Use fixtures** - Reuse setup code via pytest fixtures (see `conftest.py`)
- **Parametrize** - Use `@pytest.mark.parametrize` to avoid duplicated test code
- **Isolate** - Each test should be independent and not rely on test order
- **Name clearly** - `test_<action>_<expected_outcome>`

### commit strategy

- **Commit often** in sensible chunks
- **Single commit** should include: implementation + tests + documentation
- **Never commit** failing tests (except skip-marked stubs for future work)

### example tdd session

```bash
# 1. Create test for new VAE feature
cat > tests/unit/test_vae_new_feature.py << 'EOF'
def test_vae_handles_odd_dimensions():
    """VAE should handle non-standard input dimensions."""
    from llm_dit.models.ltx2.vae import VideoDecoder
    # ... test code
EOF

# 2. Run test - expect failure
uv run pytest tests/unit/test_vae_new_feature.py -v
# FAILED - function not implemented

# 3. Implement the feature in src/llm_dit/models/ltx2/vae/...

# 4. Run test - expect pass
uv run pytest tests/unit/test_vae_new_feature.py -v
# PASSED

# 5. Run broader tests for regressions
uv run pytest tests/unit/test_ltx2_video_vae.py -v
# All PASSED

# 6. Commit (user approval required)
```

## quick commands

```bash
# Run ALL tests (~1600 tests, ~5 min)
uv run pytest tests/ -v

# API E2E smoke tests (GPU + models required)
uv run pytest tests/e2e/api/test_flux2_smoke.py -v -s
uv run pytest tests/e2e/api/test_ltx2_smoke.py -v -s
uv run pytest tests/e2e/api/ -v -s  # all pipelines

# Pipeline integration smoke test (GPU required)
uv run pytest tests/integration/pipeline/test_baseline_portable.py::TestBaselineSmoke -v -s

# Unit tests only (no GPU, fast)
uv run pytest tests/unit/ -v

# LTX-2 specific tests
uv run pytest tests/ -v -k ltx2

# Run with slow tests enabled
uv run pytest tests/ -v --runslow
```

## test hierarchy

| Priority | What to Run | When | Can call pipelines directly? |
|----------|-------------|------|-----------------------------|
| **1. Unit** | `tests/unit/` | After any code change | Yes (component-level) |
| **2. Integration** | `tests/integration/` (incl. `pipeline/`) | After component changes | Yes (cross-component) |
| **3. E2E (API)** | `tests/e2e/api/` | Before commits, after major changes | **NO -- TestClient only** |

**Rule:** Only unit and integration tests may call pipeline functions directly. E2E tests MUST go through the API via `TestClient`. This ensures the full request lifecycle (Pydantic validation, router logic, DI, ModelManager) is exercised.

## test structure

```
tests/
├── unit/                          # Fast, no GPU required
│   ├── test_ltx2_*.py             # LTX-2 transformer, VAE, components
│   ├── test_gemma3_*.py           # Gemma3 encoder
│   ├── test_conditioning.py
│   ├── test_scheduler.py
│   └── ...
├── integration/                   # May require GPU
│   ├── test_ltx2_*.py             # Cross-component tests
│   ├── test_performance.py        # Memory/timing regression tests
│   ├── pipeline/                  # Former E2E tests (direct pipeline calls)
│   │   ├── test_baseline_portable.py
│   │   ├── test_flux2_generation.py
│   │   ├── test_ltx2_baselines.py
│   │   └── ...
│   └── ...
├── e2e/                           # API-first E2E tests (require GPU + models)
│   └── api/                       # TestClient-based API tests
│       ├── conftest.py            # TestClient + RunRecorder fixtures
│       ├── config_factory.py      # TOML overlay merging
│       ├── run_recorder.py        # Metadata capture
│       ├── validation.py          # Automated checks
│       ├── test_flux2_smoke.py    # FLUX.2 API tests
│       ├── test_ltx2_smoke.py     # LTX-2 API tests
│       └── test_zimage_smoke.py   # Z-Image API tests
├── constants/                     # Canonical parameter constants (single source of truth)
│   ├── ltx2.py                    # LTX-2 reference-aligned values
│   ├── flux2.py                   # FLUX.2 values
│   └── zimage.py                  # Z-Image values (turbo + base)
├── configs/                       # TOML overlays for E2E tests (derived from constants)
│   ├── flux2_smoke.toml
│   ├── ltx2_smoke.toml
│   └── ...
├── backends/                      # Portable test infrastructure
│   ├── protocol.py
│   └── ...
└── fixtures/                      # Test data
```

## when to run which tests

### after changing transformer code
```bash
uv run pytest tests/unit/test_ltx2_transformer.py -v
uv run pytest tests/integration/test_ltx2_connectors.py -v
```

### after changing VAE code
```bash
uv run pytest tests/unit/test_ltx2_video_vae.py -v
```

### after changing encoder code
```bash
uv run pytest tests/unit/test_gemma3_encoder.py -v
```

### after changing conditioning system
```bash
uv run pytest tests/unit/test_conditioning.py -v
uv run pytest tests/integration/test_conditioning_integration.py -v
```

### after changing audio/AV code
```bash
uv run pytest tests/unit/test_ltx2_audio_vae.py -v
uv run pytest tests/unit/test_ltx2_av_transformer.py -v
```

### after changing scheduler/sigma logic
```bash
uv run pytest tests/unit/test_scheduler.py -v
```

### before committing any LTX-2 changes
```bash
# Quick validation (~2 min with GPU)
uv run pytest tests/unit/test_ltx2_*.py tests/unit/test_conditioning.py -v
uv run pytest tests/integration/pipeline/test_baseline_portable.py::TestBaselineSmoke -v -s
```

### full validation (before PR)
```bash
uv run pytest tests/ -v --runslow
```

## test markers

| Marker | Purpose | How to Use |
|--------|---------|------------|
| `@pytest.mark.slow` | Tests taking >1min | Skip with default, include with `--runslow` |
| `@pytest.mark.skipif(not torch.cuda.is_available())` | GPU required | Auto-skipped without CUDA |
| `pytestmark = pytest.mark.skip` | Module-level skip | For stubs/future tests |

## understanding test output

### passing test
```
tests/unit/test_scheduler.py::TestSigmaSchedule::test_sigma_schedule_monotonic PASSED
```

### skipped test (expected)
```
tests/unit/test_ltx2_audio_vae.py::TestAudioVAE::test_audio_vae_not_implemented SKIPPED
```
Audio VAE tests are stubs - skip is expected.

### failed test
```
FAILED tests/unit/test_ltx2_video_vae.py::TestCausalConv3d::test_causal_conv3d_shape
```
Read the assertion error, check the actual vs expected values.

## gpu memory considerations

| Test Type | VRAM Needed | Notes |
|-----------|-------------|-------|
| Unit tests | 0 GB | CPU only |
| Integration (no models) | 2-4 GB | Small tensor ops |
| Integration (with models) | 12-16 GB | Model loading |
| E2E smoke | 14-16 GB | FP8 quantized |
| E2E reference | 20-24 GB | Full precision |

### if out of memory
```bash
# Run smoke config only (14GB)
uv run pytest tests/integration/pipeline/test_baseline_portable.py::TestBaselineSmoke -v -s

# Skip model-loading tests
uv run pytest tests/integration/ -v -k "not gpu"
```

## test configs (e2e)

| Config | Frames | Resolution | Steps | CFG | Notes |
|--------|--------|------------|-------|-----|-------|
| `ltx2_smoke.toml` (distilled) | 9 | 256x384 | 4 | 3.0 | Fastest validation |
| `ltx2_standard.toml` (distilled) | 33 | 512x768 | 12 | 3.0 | Standard quality |
| `ltx2_reference.toml` (full) | 121 | 512x768 | 40 | 3.0 | 1:1 reference repo match |
| SMOKE_CONFIG (protocol.py, full) | 33 | 512x768 | 30 | 3.0 | Backend comparison |
| REFERENCE_CONFIG (protocol.py, full) | 121 | 512x768 | 40 | 3.0 | Backend comparison |

All values derive from `tests/constants/ltx2.py`. See `tests/unit/test_config_consistency.py` for drift detection.

## debugging test failures

### 1. Get verbose output
```bash
uv run pytest tests/unit/test_ltx2_transformer.py -v --tb=long
```

### 2. Run single test with debug
```bash
uv run pytest tests/unit/test_ltx2_transformer.py::TestRoPE::test_apply_rotary_emb_interleaved -v -s
```

### 3. Check for import errors
```bash
uv run python -c "from tests.unit.test_ltx2_transformer import *"
```

### 4. Verify test collection
```bash
uv run pytest tests/unit/test_ltx2_transformer.py --collect-only
```

## adding new tests

### naming conventions
- `test_<component>_<what>.py` for files
- `Test<Component>` for classes
- `test_<action>_<expected>` for methods

### required elements
```python
"""
Module docstring with Last Updated date.

Last Updated: 2026-01-22

Run with: uv run pytest tests/unit/test_foo.py -v
"""

import pytest
import torch

class TestFoo:
    """Tests for Foo component."""

    def test_basic_functionality(self):
        """Test that basic operation works."""
        # Arrange
        x = torch.randn(1, 64, 8, 8)

        # Act
        result = foo(x)

        # Assert
        assert result.shape == x.shape
```

### skip markers for GPU/model tests
```python
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required"
)
def test_gpu_operation(self):
    ...

@pytest.mark.skipif(
    not models_available(),
    reason="LTX-2 models not found"
)
def test_with_weights(self):
    ...
```

## portable backend testing

For 1:1 comparison with official LTX-2 implementation, see [backends/README.md](backends/README.md).

```bash
# Run with our implementation
uv run pytest tests/integration/pipeline/test_baseline_portable.py -v -s

# Run with official LTX-2 (if available)
LLM_DIT_TEST_BACKEND=ltx2 uv run pytest tests/integration/pipeline/test_baseline_portable.py -v -s
```

## test coverage summary

| Category | Files | Tests | What's Validated |
|----------|-------|-------|------------------|
| **Transformer** | 2 | ~35 | RoPE, attention, FFN, AdaLN, key mapping |
| **Encoder** | 2 | ~30 | Gemma3 connector, feature extraction |
| **Conditioning** | 2 | ~50 | LatentState, I2V, denoise mask |
| **VAE** | 2 | ~28 | Compression ratios, tiling, components |
| **Scheduler** | 1 | ~12 | Sigma schedule, dynamic shift |
| **Integration** | 5 | ~60 | Numerical equivalence, memory |
| **E2E** | 3 | ~12 | Full pipeline, reference comparison |
| **LTX-2 Embeddings** | 2 | ~19 | Embedding precomputation CLI, save/load |
| **Audio VAE** | 1 | 31 | AudioDecoder, HiFiGAN vocoder, AudioPatchifier, weight loading |
| **AV Transformer** | 1 | 41 | BasicAVTransformerBlock, dual-stream, STG perturbation, FBCache |
| **Total** | - | **~1600** | Comprehensive regression protection |

## critical: visual verification

**Passing tests do NOT guarantee correct output.**

### when to verify visually

- After any change to: pipeline, scheduler, VAE, conditioning
- Before merging PRs that touch generation code
- When tests pass but you're uncertain about output quality

### what "passing" looks like

| Metric | Acceptable Range | Red Flag |
|--------|------------------|----------|
| SSIM vs baseline | > 0.85 | < 0.70 |
| PSNR | > 25 dB | < 20 dB |
| Visual artifacts | None | Any blocky/glitchy regions |
| Prompt adherence | Matches semantically | Missing key elements |

**Note:** These thresholds are guidelines. Some intentional changes (new features, quality improvements) will legitimately change output. Use judgment.

### before implementing features that affect generation
```bash
# Generate baseline with existing infrastructure
uv run pytest tests/integration/pipeline/test_<pipeline>_baselines.py::Test<Pipeline>Baselines::test_smoke_baseline_generation -v -s
```

### after e2e tests
Verify:
1. Video/image plays without artifacts
2. Content matches prompt semantically
3. Motion is temporally coherent (for video)

```bash
# Extract frames for inspection
ffmpeg -i outputs/baselines/*/video.mp4 -vf "select=eq(n\,0)+eq(n\,16)+eq(n\,32)" -vsync vfr /tmp/claude/frames_%02d.png
```

### updating baselines

Only update when:
1. Change is intentional (new feature, bug fix)
2. New output is visually correct (human verified)
3. You document WHY in commit message

```bash
# Generate new baseline
uv run pytest tests/integration/pipeline/test_<pipeline>_baselines.py -v -s

# Compare with previous (visual inspection required)
# There is no automated threshold - you must look at the output
```
