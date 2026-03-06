"""LTX-2 API smoke tests -- minimal video generation through real HTTP API.

last updated: 2026-02-16

Run with:
    uv run pytest tests/e2e/api/test_ltx2_smoke.py -v -s

Requires: CUDA GPU, LTX-2 model paths in config.toml
Output: outputs/tests/runs/api_ltx2_*/ (full metadata for reproducibility)

Note: LTX-2 uses SSE streaming. The test POSTs to /api/ltx2/generate/stream
and collects the final SSE event containing the video URL.
"""

import json

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


@pytest.fixture(scope="module")
def config_overlay():
    return "ltx2_smoke"


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


class TestLTX2APISmoke:
    """Smoke tests: fast validation that LTX-2 works through the API."""

    def test_basic_generation(self, run_recorder):
        """POST /api/ltx2/generate/stream produces a valid video."""
        response = run_recorder.post("/api/ltx2/generate/stream", json={
            "prompt": "A cat walking through a garden",
            "width": 256,
            "height": 384,
            "num_frames": 9,
            "seed": 42,
            "use_two_stage": False,
            "num_steps": 4,
            # STG disabled: 4 steps is too few for STG guidance
            # to develop temporal variation. STG validated at standard tier (40 steps).
            "stg_scale": 0.0,
        })
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text[:500]}"
        )

        # Parse SSE events to find the completion event
        events = _parse_sse_events(response)
        complete_events = [e for e in events if e.get("type") == "complete"]
        error_events = [e for e in events if e.get("type") == "error"]

        assert not error_events, f"Generation errors: {error_events}"
        assert complete_events, (
            f"No completion event in {len(events)} SSE events. "
            f"Types: {[e.get('type') for e in events]}"
        )

        complete = complete_events[-1]
        # Router sends "url" and "urls" keys (not "videoUrl" or "video_url")
        assert "url" in complete or "urls" in complete, (
            f"No video URL in completion event: {complete}"
        )

        # Download and save the video to the test run directory
        video_url = complete.get("url") or (complete.get("urls") or [None])[0]
        if video_url:
            video_resp = run_recorder.get(video_url)
            assert video_resp.status_code == 200, (
                f"Failed to download video from {video_url}: {video_resp.status_code}"
            )
            video_path = run_recorder.output_dir / "output.mp4"
            video_path.write_bytes(video_resp.content)
            run_recorder._outputs.append(video_path)

            result = run_recorder.validate(
                video_path, expected_w=256, expected_h=384,
            )
            assert result.passed, f"Validation failed:\n{result.summary()}"

    def test_status_endpoint(self, run_recorder):
        """GET /api/ltx2/status returns expected fields."""
        response = run_recorder.get("/api/ltx2/status")
        assert response.status_code == 200
        data = response.json()
        assert "available" in data
