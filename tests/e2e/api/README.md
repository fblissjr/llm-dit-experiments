# api e2e testing framework

last updated: 2026-02-12

All E2E tests go through the real API via TestClient (in-process FastAPI).
This validates the full stack: Pydantic validation, router logic, dependency injection, ModelManager, and pipeline execution.

## architecture

```
TestClient (in-process HTTP)
    |
    v
FastAPI app (same as production)
    |
    v
Pydantic validation (schemas.py)
    |
    v
Router logic (web/routers/*.py)
    |
    v
Dependency injection (ConfigDep, ManagerDep)
    |
    v
ModelManager (model_manager.py)
    |
    v
Pipeline execution (pipelines/*.py)
```

## config system

Tests use TOML overlays (see `tests/configs/README.md`):

```
config.toml.example (base defaults)
    + tests/configs/<overlay>.toml (test overrides)
    + config.toml (real model paths)
    = RuntimeConfig (same path as production)
```

Each test module declares its overlay:
```python
@pytest.fixture(scope="module")
def config_overlay():
    return "flux2_smoke"
```

## run recorder

Every test gets a `run_recorder` fixture that:
1. Creates an output directory in `outputs/tests/runs/`
2. Captures `/api/context` before generation
3. Wraps API calls to record request/response JSON
4. Saves output artifacts (images, videos)
5. Runs automated validation
6. Writes `manifest.json` with full reproducibility metadata

## validation

### image (FLUX.2, Z-Image, Qwen-Image)
- `valid_format`: PIL can decode it
- `correct_dimensions`: matches requested size
- `not_noise`: pixel std in [5, 80]
- `not_blank`: mean not near 0 or 255

### video (LTX-2)
- `valid_size`: file > 1KB
- `correct_frame_count`: matches requested frames
- `correct_dimensions`: matches requested resolution
- `not_noise_frame_N`: per-frame pixel std in [5, 100]
- `not_frozen_I_J`: adjacent frame MSE > 10

## output structure

```
outputs/tests/runs/api_{pipeline}_{test_name}_{timestamp}/
    config_frozen.toml      # exact merged TOML used
    request.json            # full API request body
    response.json           # full API response
    context.json            # GET /api/context snapshot
    environment.json        # GPU, torch version, etc.
    manifest.json           # master record
    output.png / output.mp4 # generated artifact
    generation.log          # INFO+ logs
    debug.log               # DEBUG+ full trace
    errors.log              # WARNING+ issues only
    validation.json         # automated check results
```

## writing a new e2e test

### checklist

- [ ] Uses `run_recorder` fixture (not direct pipeline calls)
- [ ] Sends HTTP request to a real API endpoint
- [ ] Config loaded from a TOML overlay in `tests/configs/`
- [ ] Asserts on response status code
- [ ] Saves output artifact via `save_output()`
- [ ] Runs automated validation via `validate()`
- [ ] Asserts `validation.passed`
- [ ] All outputs land in `outputs/tests/runs/`

### step by step

1. Create overlay in `tests/configs/<pipeline>_<tier>.toml` if needed
2. Create test file in `tests/e2e/api/test_<pipeline>_<tier>.py`
3. Define `config_overlay` fixture returning the overlay name
4. Write test methods using `run_recorder.post()` and `run_recorder.validate()`
5. Run: `uv run pytest tests/e2e/api/test_<pipeline>_<tier>.py -v -s`
6. Check output: `ls outputs/tests/runs/api_<pipeline>_*/`

### template

```python
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

@pytest.fixture(scope="module")
def config_overlay():
    return "<pipeline>_smoke"

class TestPipelineAPISmoke:
    def test_basic_generation(self, run_recorder):
        response = run_recorder.post("/api/<endpoint>", json={
            "prompt": "...",
            "seed": 42,
            # ... pipeline-specific params
        })
        assert response.status_code == 200
        data = response.json()

        output_path = run_recorder.save_output(data)
        result = run_recorder.validate(output_path, expected_w=256, expected_h=256)
        assert result.passed, result.summary()
```

## quick commands

```bash
# FLUX.2 smoke tests
uv run pytest tests/e2e/api/test_flux2_smoke.py -v -s

# LTX-2 smoke tests
uv run pytest tests/e2e/api/test_ltx2_smoke.py -v -s

# Z-Image smoke tests
uv run pytest tests/e2e/api/test_zimage_smoke.py -v -s

# All API E2E tests
uv run pytest tests/e2e/api/ -v -s

# Collect only (no GPU needed, verifies imports)
uv run pytest tests/e2e/api/ --collect-only
```
