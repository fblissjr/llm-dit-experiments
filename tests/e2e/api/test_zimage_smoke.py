"""Z-Image API smoke tests -- minimal generation through real HTTP API.

last updated: 2026-02-12

Run with:
    uv run pytest tests/e2e/api/test_zimage_smoke.py -v -s

Requires: CUDA GPU, Z-Image model path in config.toml
Output: outputs/tests/runs/api_zimage_*/ (full metadata for reproducibility)
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


@pytest.fixture(scope="module")
def config_overlay():
    return "zimage_smoke"


class TestZImageAPISmoke:
    """Smoke tests: fast validation that Z-Image works through the API."""

    def test_basic_generation(self, run_recorder):
        """POST /api/generate with minimal params produces a valid image."""
        response = run_recorder.post("/api/generate", json={
            "prompt": "A cat sleeping in warm sunlight",
            "width": 256,
            "height": 256,
            "steps": 9,
            "seed": 42,
            "guidance_scale": 0.0,
            "shift": 3.0,
        })
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert data["seed"] == 42

        output_path = run_recorder.save_output(data)
        result = run_recorder.validate(output_path, expected_w=256, expected_h=256)
        assert result.passed, f"Validation failed:\n{result.summary()}"

    def test_health_endpoint(self, run_recorder):
        """GET /health returns expected fields."""
        response = run_recorder.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
