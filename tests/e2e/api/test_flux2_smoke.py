"""FLUX.2 API smoke tests -- minimal generation through real HTTP API.

last updated: 2026-02-12

Run with:
    uv run pytest tests/e2e/api/test_flux2_smoke.py -v -s

Requires: CUDA GPU, FLUX.2 model paths in config.toml
Output: outputs/tests/runs/api_flux2_*/ (full metadata for reproducibility)
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


@pytest.fixture(scope="module")
def config_overlay():
    return "flux2_smoke"


class TestFlux2APISmoke:
    """Smoke tests: fast validation that FLUX.2 works through the API."""

    def test_basic_generation(self, run_recorder):
        """POST /api/flux2/generate with minimal params produces a valid image."""
        response = run_recorder.post("/api/flux2/generate", json={
            "prompt": "A photograph of a cat sitting on a windowsill",
            "model_name": "klein-9b-fp8",
            "width": 256,
            "height": 256,
            "num_steps": 2,
            "seed": 42,
        })
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        assert data["seed"] == 42

        output_path = run_recorder.save_output(data)
        result = run_recorder.validate(output_path, expected_w=256, expected_h=256)
        assert result.passed, f"Validation failed:\n{result.summary()}"

    def test_deterministic_seed(self, run_recorder):
        """Same seed + params produce identical output."""
        body = {
            "prompt": "A red apple on a white table",
            "model_name": "klein-9b-fp8",
            "width": 256,
            "height": 256,
            "num_steps": 2,
            "seed": 12345,
        }
        r1 = run_recorder.post("/api/flux2/generate", json=body)
        r2 = run_recorder.post("/api/flux2/generate", json=body)
        assert r1.status_code == 200
        assert r2.status_code == 200

        img1 = run_recorder.save_output(r1.json(), suffix="_run1")
        img2 = run_recorder.save_output(r2.json(), suffix="_run2")

        assert img1.read_bytes() == img2.read_bytes(), "Deterministic generation failed"

    def test_status_endpoint(self, run_recorder):
        """GET /api/flux2/status returns expected fields."""
        response = run_recorder.get("/api/flux2/status")
        assert response.status_code == 200
        data = response.json()
        assert data["available"] is True
        assert "supportedModels" in data

    def test_invalid_request_rejected(self, run_recorder):
        """Missing required field (prompt) returns 422."""
        response = run_recorder.post("/api/flux2/generate", json={
            "width": 256,
            "height": 256,
        })
        assert response.status_code == 422, "Expected 422 for missing prompt"
