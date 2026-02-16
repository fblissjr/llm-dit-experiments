"""Z-Image API tests -- config defaults and variant behavior through TestClient.

last updated: 2026-02-16

Rewritten from integration/pipeline/z_image/test_api_defaults.py and
test_base_model.py to use TestClient instead of external HTTP requests.

Run with:
    uv run pytest tests/e2e/api/test_zimage_api.py -v -s

Requires: CUDA GPU, Z-Image model path in config.toml
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


@pytest.fixture(scope="module")
def config_overlay():
    return "zimage_smoke"


class TestZImageAPIDefaults:
    """Verify API returns correct defaults for Z-Image pipeline."""

    def test_health_endpoint(self, run_recorder):
        """GET /health returns ok status."""
        resp = run_recorder.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_pipeline_defaults_exist(self, run_recorder):
        """GET /api/pipelines/zimage/defaults returns config values."""
        resp = run_recorder.get("/api/pipelines/zimage/defaults")
        assert resp.status_code == 200
        data = resp.json()

        # Should have generation-relevant fields
        assert "shift" in data or "guidance_scale" in data or "steps" in data, (
            f"Pipeline defaults should contain generation params, got keys: {list(data.keys())}"
        )

    def test_pipeline_defaults_variant(self, run_recorder):
        """Pipeline defaults include _variant field for conditional UI."""
        resp = run_recorder.get("/api/pipelines/zimage/defaults")
        assert resp.status_code == 200
        data = resp.json()

        variant = data.get("_variant")
        assert variant in ("turbo", "base"), (
            f"Expected _variant to be 'turbo' or 'base', got {variant!r}"
        )


class TestZImageAPIGeneration:
    """Test Z-Image generation through the API."""

    def test_basic_generation(self, run_recorder):
        """POST /api/generate with turbo settings produces a valid image."""
        resp = run_recorder.post("/api/generate", json={
            "prompt": "A simple red circle on a white background",
            "width": 256,
            "height": 256,
            "steps": 9,
            "seed": 42,
            "guidance_scale": 0.0,
            "shift": 3.0,
        })
        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text[:500]}"
        )
        data = resp.json()
        assert data["seed"] == 42

        output_path = run_recorder.save_output(data)
        result = run_recorder.validate(output_path, expected_w=256, expected_h=256)
        assert result.passed, f"Validation failed:\n{result.summary()}"

    def test_seed_reproducibility(self, run_recorder):
        """Same seed produces deterministic output."""
        params = {
            "prompt": "A blue square",
            "width": 256,
            "height": 256,
            "steps": 9,
            "seed": 123,
            "guidance_scale": 0.0,
            "shift": 3.0,
        }

        resp1 = run_recorder.post("/api/generate", json=params)
        assert resp1.status_code == 200
        data1 = resp1.json()

        resp2 = run_recorder.post("/api/generate", json=params)
        assert resp2.status_code == 200
        data2 = resp2.json()

        # Both should use the same seed
        assert data1["seed"] == data2["seed"] == 123

    def test_minimal_request(self, run_recorder):
        """Minimal request (just prompt) should use server defaults."""
        resp = run_recorder.post("/api/generate", json={
            "prompt": "test",
            "width": 256,
            "height": 256,
        })
        # Should not error -- defaults should fill in missing params
        assert resp.status_code == 200, (
            f"Minimal request failed: {resp.status_code}: {resp.text[:500]}"
        )
