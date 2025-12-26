last updated: 2025-12-26

# qwen-image-edit-2511 testing guide

Complete guide for testing the Qwen-Image-Edit-2511 integration.

## quick start

```bash
# Run unit tests only (no GPU required)
uv run pytest tests/unit/test_qwen_image_edit_2511.py -v

# Run all qwen_image tests
uv run pytest tests/ -k "qwen_image" -v

# Run practical test script (requires GPU and models)
uv run scripts/test_qwen_image_edit_2511.py \
  --model-path ~/Storage/Qwen_Qwen-Image-Layered \
  --edit-model-path ~/Storage/Qwen_Qwen-Image-Edit-2511
```

## test levels

### level 1: unit tests (no gpu)

Fast, mock-based tests that verify API contracts without loading models.

```bash
uv run pytest tests/unit/test_qwen_image_edit_2511.py -v
```

**What's tested:**
- Default values (steps=40, cfg_scale=4.0)
- `edit_multi()` method signature
- Input validation (minimum 2 images)
- Parameter passing to underlying pipeline
- RGBA to RGB conversion
- Lazy loading behavior
- Config defaults in `QwenImageConfig` and `RuntimeConfig`

**Expected output:**
```
21 passed
```

### level 2: integration tests (requires gpu + models)

Full pipeline tests with real model loading.

```bash
# Decompose only (faster, no edit model download)
uv run python tests/integration/test_qwen_diffusers_wrapper.py \
  --model-path ~/Storage/Qwen_Qwen-Image-Layered \
  --skip-edit

# Full test including editing
uv run python tests/integration/test_qwen_diffusers_wrapper.py \
  --model-path ~/Storage/Qwen_Qwen-Image-Layered

# Include multi-image test with custom second image
uv run python tests/integration/test_qwen_diffusers_wrapper.py \
  --model-path ~/Storage/Qwen_Qwen-Image-Layered \
  --second-image experiments/inputs/style_art_deco.png
```

**What's tested:**
- Pipeline import and loading
- Image decomposition into layers
- Single-image layer editing
- Multi-image editing (2511 feature)
- Device management and VRAM usage

### level 3: practical test script

Interactive script for manual verification.

```bash
uv run scripts/test_qwen_image_edit_2511.py \
  --model-path ~/Storage/Qwen_Qwen-Image-Layered \
  --edit-model-path ~/Storage/Qwen_Qwen-Image-Edit-2511 \
  --test-image experiments/inputs/homer_art_deco.png \
  --output-dir experiments/results/qwen_edit_2511_test
```

## test cases

### defaults verification

| Setting | Expected | File |
|---------|----------|------|
| `DEFAULT_STEPS` | 40 | `qwen_image_diffusers.py` |
| `DEFAULT_CFG_SCALE` | 4.0 | `qwen_image_diffusers.py` |
| `QwenImageConfig.num_inference_steps` | 40 | `config.py` |
| `RuntimeConfig.qwen_image_steps` | 40 | `cli.py` |

### edit_multi() validation

| Input | Expected Result |
|-------|-----------------|
| 0 images | `ValueError: at least 2 images` |
| 1 image | `ValueError: at least 2 images` |
| 2 images | Success, combined output |
| 4 images | Success, combined output |
| RGBA images | Converted to RGB, success |
| Mixed RGB/RGBA | Success |

### web api endpoints

```bash
# Start server
uv run web/server.py --config config.toml --profile default

# Test single-image edit
curl -X POST http://localhost:8000/api/qwen-image/edit-layer \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/png;base64,<base64_data>",
    "instruction": "Make it brighter",
    "steps": 40,
    "cfg_scale": 4.0
  }'

# Test multi-image edit
curl -X POST http://localhost:8000/api/qwen-image/edit-multi \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["data:image/png;base64,...", "data:image/png;base64,..."],
    "instruction": "Combine both subjects in a park",
    "steps": 40,
    "cfg_scale": 4.0
  }'
```

## expected vram usage

| Operation | VRAM (no offload) | VRAM (cpu offload) |
|-----------|-------------------|--------------------|
| Load decompose model | ~14 GB | ~8 GB |
| Load edit model | ~10 GB | ~5 GB |
| Both models loaded | ~24 GB | ~10 GB peak |
| Decomposition run | +2-3 GB | +2-3 GB |
| Edit run | +2-3 GB | +2-3 GB |

**Recommendation:** Enable CPU offload for RTX 4090 (24 GB).

## common issues

### "edit model not loaded"

The edit model loads lazily on first `edit_layer()` or `edit_multi()` call. To preload:

```python
pipe = QwenImageDiffusersPipeline.from_pretrained(
    model_path,
    load_edit_model=True,  # Eager loading
)
```

### "at least 2 images required"

`edit_multi()` requires 2+ images. For single-image editing, use `edit_layer()` instead.

### network timeout during download

If auto-downloading the edit model fails:

```bash
# Pre-download manually
huggingface-cli download Qwen/Qwen-Image-Edit-2511 \
  --local-dir ~/Storage/Qwen_Qwen-Image-Edit-2511
```

Then specify the local path:

```python
pipe = QwenImageDiffusersPipeline.from_pretrained(
    model_path,
    edit_model_path="~/Storage/Qwen_Qwen-Image-Edit-2511",
)
```

### cuda out of memory

Enable CPU offload:

```python
pipe = QwenImageDiffusersPipeline.from_pretrained(
    model_path,
    cpu_offload=True,
)
```

Or in config.toml:

```toml
[default.qwen_image]
cpu_offload = true
```

## test output locations

| Test | Output Directory |
|------|------------------|
| Integration test | `experiments/results/qwen_layered_test/wrapper_test/` |
| Practical test | `experiments/results/qwen_edit_2511_test/` (configurable) |

## running all tests

```bash
# Quick validation (unit tests only)
uv run pytest tests/unit/test_qwen_image_edit_2511.py -v

# Full test suite
uv run pytest tests/ -k "qwen" -v

# Practical verification
uv run scripts/test_qwen_image_edit_2511.py \
  --model-path ~/Storage/Qwen_Qwen-Image-Layered \
  --full
```

## see also

- [Qwen-Image-Edit-2511 Documentation](qwen_image_edit_2511.md)
- [Qwen-Image Guide](../qwen_image_guide.md)
- [Integration Test](../../tests/integration/test_qwen_diffusers_wrapper.py)
- [Unit Tests](../../tests/unit/test_qwen_image_edit_2511.py)
