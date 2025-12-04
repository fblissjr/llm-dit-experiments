# Tests

Automated tests for the llm-dit-experiments project.

## Running Tests

Run all tests:
```bash
uv run pytest tests/
```

Run specific test file:
```bash
uv run pytest tests/test_web_server.py
```

Run with verbose output:
```bash
uv run pytest tests/ -v
```

Run specific test class or method:
```bash
uv run pytest tests/test_web_server.py::TestHealthEndpoint
uv run pytest tests/test_web_server.py::TestHealthEndpoint::test_health_full_pipeline
```

## Test Files

### test_web_server.py

Comprehensive tests for the Z-Image web server (`web/server.py`).

**Test Coverage:**
- Health endpoint (`/health`) - validates server mode detection
- Encode endpoint (`/api/encode`) - tests prompt encoding with templates
- Generate endpoint (`/api/generate`) - tests image generation (full pipeline mode only)
- Templates endpoint (`/api/templates`) - tests template listing
- Save embeddings endpoint (`/api/save-embeddings`) - tests embedding persistence
- Index endpoint (`/`) - validates HTML serving
- Request validation - tests parameter validation and defaults
- Mode detection - tests 3 server modes (full pipeline, encoder-only, no models)

**29 tests covering:**
1. Full pipeline mode (pipeline loaded)
2. Encoder-only mode (local encoder, no DiT/VAE)
3. API backend mode (uses heylookitsanllm for encoding)

All tests use mocked models to avoid loading actual LLM weights.

## Test Structure

Tests are organized by endpoint/feature using pytest classes:
- `TestHealthEndpoint`
- `TestEncodeEndpoint`
- `TestGenerateEndpoint`
- `TestTemplatesEndpoint`
- `TestSaveEmbeddingsEndpoint`
- `TestIndexEndpoint`
- `TestRequestValidation`
- `TestModeDetection`

## Fixtures

- `mock_encoder` - Mocked encoder with templates
- `mock_pipeline` - Mocked full pipeline with encoder
- `client_full_pipeline` - Test client with full pipeline loaded
- `client_encoder_only` - Test client in encoder-only mode
- `client_no_models` - Test client with no models loaded

## Dependencies

Test dependencies are in the `[project.optional-dependencies]` section of `pyproject.toml`:
- pytest
- httpx (for FastAPI TestClient)
- fastapi

Install with:
```bash
uv sync --extra dev
```
