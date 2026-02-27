"""Run recorder: captures all metadata for a single E2E test run.

last updated: 2026-02-12

Wraps a TestClient to capture request/response pairs, server context,
environment info, and generated outputs. Writes everything to a structured
output directory for full reproducibility.

Output directory structure:
    outputs/tests/runs/api_{pipeline}_{test_name}_{timestamp}/
        config_frozen.toml      # Exact merged TOML used
        request.json            # Full API request body as sent
        response.json           # Full API response
        context.json            # GET /api/context before generation
        environment.json        # GPU name, torch version, etc.
        manifest.json           # Master record tying everything together
        output.png / output.mp4 # Generated artifact
        generation.log          # INFO+ logs captured during generation
        debug.log               # DEBUG+ full trace
        errors.log              # WARNING+ issues only
        validation.json         # Automated validation results
"""

from __future__ import annotations

import base64
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson
import torch

if TYPE_CHECKING:
    from httpx import Response
    from starlette.testclient import TestClient

from tests.e2e.api.validation import ValidationResult, validate_image, validate_video

logger = logging.getLogger(__name__)


def _get_environment_info() -> dict:
    """Collect environment info for reproducibility."""
    info = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_vram_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
        )
    return info


class RunRecorder:
    """Captures all metadata for a single E2E test run."""

    def __init__(
        self,
        client: TestClient,
        output_dir: Path,
        config_path: Path,
        pipeline: str,
        test_name: str,
        config_overlay: str,
    ):
        self.client = client
        self.output_dir = output_dir
        self.config_path = config_path
        self.pipeline = pipeline
        self.test_name = test_name
        self.config_overlay = config_overlay

        self._requests: list[dict] = []
        self._responses: list[dict] = []
        self._context: dict | None = None
        self._environment = _get_environment_info()
        self._start_time = time.monotonic()
        self._outputs: list[Path] = []

        # Copy frozen config to output dir
        if config_path.exists():
            frozen_dest = output_dir / "config_frozen.toml"
            frozen_dest.write_bytes(config_path.read_bytes())

        # Save environment info
        self._write_json("environment.json", self._environment)

    def capture_context(self) -> dict:
        """GET /api/context -- model variant, quant, VRAM, LoRA state."""
        resp = self.client.get("/api/context")
        if resp.status_code == 200:
            self._context = resp.json()
            self._write_json("context.json", self._context)
        return self._context or {}

    def post(self, endpoint: str, json: dict) -> Response:
        """POST to API, capturing request/response to disk."""
        request_record = {
            "method": "POST",
            "endpoint": endpoint,
            "body": json,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._requests.append(request_record)

        response = self.client.post(endpoint, json=json)

        response_record = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
        }
        try:
            response_record["body"] = response.json()
        except Exception:
            # For SSE streams, store the last 5000 chars (includes completion event)
            # plus the first 500 chars (includes initial status)
            text = response.text
            if len(text) > 6000:
                response_record["body"] = text[:500] + "\n...\n" + text[-5000:]
            else:
                response_record["body"] = text
        self._responses.append(response_record)

        # Write per-request files (numbered if multiple)
        suffix = f"_{len(self._requests)}" if len(self._requests) > 1 else ""
        self._write_json(f"request{suffix}.json", request_record)
        self._write_json(f"response{suffix}.json", response_record)

        return response

    def get(self, endpoint: str) -> Response:
        """GET from API (for status checks, etc.)."""
        return self.client.get(endpoint)

    def save_output(
        self,
        response_data: dict,
        suffix: str = "",
    ) -> Path:
        """Decode base64 image or save video URL, write to output dir.

        For image pipelines: extracts base64 from response url/urls field.
        For video pipelines: the response contains a URL path to download.

        Returns path to the saved file.
        """
        output_path: Path | None = None

        # Image: base64 data URL in response
        urls = response_data.get("urls") or []
        url = response_data.get("url") or (urls[0] if urls else "")

        if url.startswith("data:image/"):
            # Extract base64 payload after the comma
            b64_data = url.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_data)
            output_path = self.output_dir / f"output{suffix}.png"
            output_path.write_bytes(img_bytes)

        elif url.startswith("/") or url.startswith("http"):
            # Video URL -- fetch it from the test server
            video_resp = self.client.get(url)
            if video_resp.status_code == 200:
                output_path = self.output_dir / f"output{suffix}.mp4"
                output_path.write_bytes(video_resp.content)

        if output_path is not None:
            self._outputs.append(output_path)

        return output_path

    def validate(
        self,
        output_path: Path,
        expected_w: int | None = None,
        expected_h: int | None = None,
        expected_frames: int | None = None,
    ) -> ValidationResult:
        """Run automated validation checks on the output artifact."""
        if output_path is None:
            result = ValidationResult(passed=False)
            from tests.e2e.api.validation import CheckResult
            result.add(CheckResult(
                passed=False, name="output_saved",
                value="None", threshold="file must exist",
            ))
            return result

        if output_path.suffix == ".mp4":
            result = validate_video(
                output_path,
                expected_frames=expected_frames,
                expected_w=expected_w,
                expected_h=expected_h,
            )
        else:
            result = validate_image(
                output_path,
                expected_w=expected_w,
                expected_h=expected_h,
            )

        self._write_json("validation.json", result.to_dict())
        return result

    def finalize(self, status: str = "needs_review") -> Path:
        """Write manifest.json with all collected metadata. Returns manifest path."""
        elapsed = time.monotonic() - self._start_time

        # Build manifest
        manifest: dict[str, Any] = {
            "test_name": self.test_name,
            "pipeline": self.pipeline,
            "config_overlay": self.config_overlay,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "elapsed_seconds": round(elapsed, 2),
        }

        # Request/response
        if self._requests:
            manifest["request"] = self._requests[0]
        if self._responses:
            resp = self._responses[0]
            manifest["response"] = {
                "status_code": resp["status_code"],
            }
            body = resp.get("body", {})
            if isinstance(body, dict):
                manifest["response"]["seed"] = body.get("seed")
                manifest["response"]["generation_time"] = body.get(
                    "generationTime", body.get("generation_time")
                )
                manifest["response"]["pipeline_id"] = body.get(
                    "pipelineId", body.get("pipeline_id")
                )

        # Server context
        if self._context:
            manifest["server_context"] = {
                "active_pipeline": self._context.get("activePipeline"),
                "model_variant": self._context.get("modelVariant"),
                "quantization": self._context.get("quantization", {}),
                "compile_enabled": self._context.get("compileEnabled", False),
                "loras": self._context.get("loras", []),
                "vram_used_gb": self._context.get("vramUsedGb"),
                "vram_total_gb": self._context.get("vramTotalGb"),
            }

        # Files inventory
        manifest["files"] = {
            "config": "config_frozen.toml",
            "outputs": [p.name for p in self._outputs],
        }
        if (self.output_dir / "validation.json").exists():
            manifest["files"]["validation"] = "validation.json"
        for log_name in ("generation.log", "debug.log", "errors.log"):
            if (self.output_dir / log_name).exists():
                manifest["files"].setdefault("logs", []).append(log_name)

        manifest["environment"] = self._environment

        manifest_path = self.output_dir / "manifest.json"
        self._write_json("manifest.json", manifest)
        return manifest_path

    def _write_json(self, filename: str, data: Any) -> None:
        """Write JSON to output directory using orjson."""
        path = self.output_dir / filename
        path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
