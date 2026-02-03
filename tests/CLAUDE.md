# testing guide for agents

*last updated: 2026-02-03*

Quick reference for LLM agents running tests in this codebase.

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
| **Presets** | `presets/testing/<pipeline>_*.md` | Test configs (YAML frontmatter) |
| **Baselines** | `tests/baselines/` | Generation and comparison utilities |
| **Backends** | `tests/backends/` | Portable test protocol |

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
# Run ALL tests (~1030 tests, ~5 min)
uv run pytest tests/ -v

# Quick smoke test (30s, requires GPU)
uv run pytest tests/e2e/test_baseline_portable.py::TestBaselineSmoke -v -s

# Unit tests only (no GPU, fast)
uv run pytest tests/unit/ -v

# LTX-2 specific tests
uv run pytest tests/ -v -k ltx2

# Run with slow tests enabled
uv run pytest tests/ -v --runslow
```

## test hierarchy

| Priority | What to Run | When |
|----------|-------------|------|
| **1. Unit** | `tests/unit/` | After any code change |
| **2. Integration** | `tests/integration/` | After component changes |
| **3. E2E** | `tests/e2e/` | Before commits, after major changes |

## test structure

```
tests/
├── unit/                  # Fast, no GPU required
│   ├── test_ltx2_*.py     # LTX-2 transformer, VAE, components
│   ├── test_gemma3_*.py   # Gemma3 encoder
│   ├── test_conditioning.py
│   ├── test_scheduler.py
│   └── ...
├── integration/           # May require GPU
│   ├── test_ltx2_*.py     # Cross-component tests
│   ├── test_performance.py # Memory/timing regression tests
│   └── ...
├── e2e/                   # Require GPU + models
│   ├── test_baseline_portable.py  # Main e2e test
│   ├── test_ltx2_reference.py
│   └── ...
├── backends/              # Portable test infrastructure
│   ├── protocol.py        # GenerationConfig, Backend interface
│   ├── llm_dit_backend.py # Our implementation
│   └── ltx2_backend.py    # Official LTX-2 reference
└── fixtures/              # Test data
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

### after changing scheduler/sigma logic
```bash
uv run pytest tests/unit/test_scheduler.py -v
```

### before committing any LTX-2 changes
```bash
# Quick validation (~2 min with GPU)
uv run pytest tests/unit/test_ltx2_*.py tests/unit/test_conditioning.py -v
uv run pytest tests/e2e/test_baseline_portable.py::TestBaselineSmoke -v -s
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
uv run pytest tests/e2e/test_baseline_portable.py::TestBaselineSmoke -v -s

# Skip model-loading tests
uv run pytest tests/integration/ -v -k "not gpu"
```

## test configs (e2e)

| Config | Frames | Resolution | Steps | CFG | Est. Time |
|--------|--------|------------|-------|-----|-----------|
| `get_smoke_config()` | 9 | 256x384 | 2 | 1.0 | ~30s |
| `get_short_config()` | 33 | 384x512 | 10 | 3.0 | ~2min |
| `get_reference_config()` | 121 | 512x768 | 40 | 4.0 | ~10min |

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
uv run pytest tests/e2e/test_baseline_portable.py -v -s

# Run with official LTX-2 (if available)
LLM_DIT_TEST_BACKEND=ltx2 uv run pytest tests/e2e/test_baseline_portable.py -v -s
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
| **Total** | - | **~1050** | Comprehensive regression protection |

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
uv run pytest tests/e2e/test_<pipeline>_baselines.py::Test<Pipeline>Baselines::test_smoke_baseline_generation -v -s
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
uv run pytest tests/e2e/test_<pipeline>_baselines.py -v -s

# Compare with previous (visual inspection required)
# There is no automated threshold - you must look at the output
```
