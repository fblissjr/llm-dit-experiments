"""Qwen-Image API smoke tests -- T2I generation and status endpoints.

last updated: 2026-02-16

Run with:
    uv run pytest tests/e2e/api/test_qwenimage_smoke.py -v -s

Requires: CUDA GPU, Qwen-Image-2512 model path in config.toml
Output: outputs/tests/runs/api_qwenimage_*/ (full metadata for reproducibility)

Note: The T2I endpoint returns raw PNG bytes (StreamingResponse), not JSON
with base64 data URLs. Tests save response.content directly as PNG.
"""

import pytest
import torch
from PIL import Image
import io

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


@pytest.fixture(scope="module")
def config_overlay():
    return "qwenimage_smoke"


class TestQwenImageT2IStatus:
    """Status and config endpoints for Qwen-Image T2I."""

    def test_t2i_status(self, run_recorder):
        """GET /api/qwen-image-2512/status returns expected fields."""
        resp = run_recorder.get("/api/qwen-image-2512/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "available" in data
        assert "configured" in data

    def test_t2i_config(self, run_recorder):
        """GET /api/qwen-image-2512/config returns generation defaults."""
        resp = run_recorder.get("/api/qwen-image-2512/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "steps" in data
        assert "cfg_scale" in data
        assert "default_width" in data
        assert "default_height" in data

    def test_edit_status(self, run_recorder):
        """GET /api/qwen-image/edit-status returns availability."""
        resp = run_recorder.get("/api/qwen-image/edit-status")
        assert resp.status_code == 200
        data = resp.json()
        assert "available" in data


class TestQwenImageT2IGeneration:
    """T2I generation through the API (requires model loaded)."""

    def test_basic_generation(self, run_recorder):
        """POST /api/qwen-image-2512/generate produces a valid PNG image."""
        resp = run_recorder.post("/api/qwen-image-2512/generate", json={
            "prompt": "A simple red circle on a white background",
            "width": 512,
            "height": 512,
            "steps": 20,
            "cfg_scale": 4.0,
            "seed": 42,
        })
        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text[:500]}"
        )

        # T2I returns raw PNG bytes, not JSON
        assert resp.headers.get("content-type", "").startswith("image/png"), (
            f"Expected image/png, got {resp.headers.get('content-type')}"
        )

        # Save the raw PNG
        output_path = run_recorder.output_dir / "output.png"
        output_path.write_bytes(resp.content)
        run_recorder._outputs.append(output_path)

        # Validate it's a real image with expected dimensions
        img = Image.open(io.BytesIO(resp.content))
        assert img.size == (512, 512), f"Expected 512x512, got {img.size}"
        assert img.mode in ("RGB", "RGBA"), f"Unexpected mode: {img.mode}"

    def test_generation_returns_timing_header(self, run_recorder):
        """Response includes X-Inference-Time header."""
        resp = run_recorder.post("/api/qwen-image-2512/generate", json={
            "prompt": "A blue square",
            "width": 512,
            "height": 512,
            "steps": 20,
            "seed": 123,
        })
        assert resp.status_code == 200

        inference_time = resp.headers.get("X-Inference-Time")
        assert inference_time is not None, "Missing X-Inference-Time header"
        assert float(inference_time) > 0, "Inference time should be positive"

    def test_invalid_request_rejected(self, run_recorder):
        """Missing required field (prompt) returns 422."""
        resp = run_recorder.post("/api/qwen-image-2512/generate", json={
            "width": 512,
            "height": 512,
        })
        assert resp.status_code == 422, "Expected 422 for missing prompt"
