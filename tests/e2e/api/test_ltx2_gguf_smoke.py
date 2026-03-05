"""LTX-2 GGUF API smoke tests -- V2 (22B) generation through HTTP API.

last updated: 2026-03-05

Run with:
    uv run pytest tests/e2e/api/test_ltx2_gguf_smoke.py -v -s

Requires: CUDA GPU, GGUF checkpoint + Gemma3 encoder paths in config.toml
Output: outputs/tests/runs/api_ltx2_gguf_*/ (full metadata for reproducibility)
"""

import json

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


@pytest.fixture(scope="module")
def config_overlay():
    return "ltx2_gguf_smoke"


def _parse_sse_events(response) -> list[dict]:
    """Parse Server-Sent Events from a streaming response."""
    events = []
    for line in response.text.split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


class TestLTX2GGUFSmoke:
    """Smoke tests: fast validation that GGUF pipeline works through the API."""

    def test_status_available(self, run_recorder):
        """GET /api/ltx2/status reports available when GGUF path is configured."""
        response = run_recorder.get("/api/ltx2/status")
        assert response.status_code == 200
        data = response.json()
        assert data["available"] is True, (
            "LTX-2 GGUF not available. Check gguf_transformer_path in config.toml"
        )

    def test_gguf_generation(self, run_recorder):
        """POST /api/ltx2/generate/stream with GGUF produces a valid video."""
        response = run_recorder.post("/api/ltx2/generate/stream", json={
            "prompt": "A slow pan across a mountain landscape at sunset",
            "width": 384,
            "height": 256,
            "num_frames": 9,
            "seed": 42,
            "use_two_stage": False,
            "stage1_steps": 4,
            "stg_scale": 0.0,
        })
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text[:500]}"
        )

        events = _parse_sse_events(response)
        complete_events = [e for e in events if e.get("type") == "complete"]
        error_events = [e for e in events if e.get("type") == "error"]

        assert not error_events, f"Generation errors: {error_events}"
        assert complete_events, (
            f"No completion event in {len(events)} SSE events. "
            f"Types: {[e.get('type') for e in events]}"
        )

        complete = complete_events[-1]
        assert "url" in complete or "urls" in complete, (
            f"No video URL in completion event: {complete}"
        )

    def test_gguf_vram_peak(self, run_recorder):
        """Peak VRAM during GGUF forward should stay under 22GB."""
        response = run_recorder.post("/api/ltx2/generate/stream", json={
            "prompt": "A red ball bouncing on a white floor",
            "width": 384,
            "height": 256,
            "num_frames": 9,
            "seed": 123,
            "use_two_stage": False,
            "stage1_steps": 4,
            "stg_scale": 0.0,
        })
        assert response.status_code == 200

        # Check peak VRAM
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        assert peak_gb < 22.0, f"Peak VRAM {peak_gb:.1f}GB exceeds 22GB limit"
