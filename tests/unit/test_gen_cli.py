"""Unit tests for scripts/gen.py -- CLI-over-API tool.

Tests cover:
- Argument parsing for all subcommands
- Request body building (None omission for resolve_param precedence)
- Response handler dispatch
- Server URL construction
- Health check / status subcommand
"""


# ---------------------------------------------------------------------------
# Import helpers -- gen.py exposes these as module-level functions
# ---------------------------------------------------------------------------

from scripts.gen import (
    _get_camel,
    build_request_body,
    create_parser,
    get_endpoint,
    get_response_handler,
    handle_sse,
)


# ===========================================================================
# Argument Parsing
# ===========================================================================


class TestArgParsing:
    """Verify subcommand routing and argument collection."""

    def test_status_subcommand(self):
        parser = create_parser()
        args = parser.parse_args(["status"])
        assert args.subcommand == "status"

    def test_flux2_minimal(self):
        parser = create_parser()
        args = parser.parse_args(["flux2", "--prompt", "a cat"])
        assert args.subcommand == "flux2"
        assert args.prompt == "a cat"

    def test_flux2_all_flags(self):
        parser = create_parser()
        args = parser.parse_args([
            "flux2",
            "--prompt", "a cat",
            "--width", "1024",
            "--height", "768",
            "--num-steps", "20",
            "--seed", "42",
            "--guidance", "3.5",
            "--model-name", "klein-4b-fp8",
            "--upsample-prompt",
            "--loras", "style.safetensors:0.8",
            "--stream",
        ])
        assert args.width == 1024
        assert args.height == 768
        assert args.num_steps == 20
        assert args.seed == 42
        assert args.guidance == 3.5
        assert args.model_name == "klein-4b-fp8"
        assert args.upsample_prompt is True
        assert args.loras == ["style.safetensors:0.8"]
        assert args.stream is True

    def test_zimage_minimal(self):
        parser = create_parser()
        args = parser.parse_args(["zimage", "--prompt", "a mountain"])
        assert args.subcommand == "zimage"
        assert args.prompt == "a mountain"

    def test_zimage_with_dimensions(self):
        parser = create_parser()
        args = parser.parse_args([
            "zimage", "--prompt", "test", "--width", "512", "--height", "512",
            "--steps", "9", "--seed", "123",
        ])
        assert args.width == 512
        assert args.height == 512
        assert args.steps == 9
        assert args.seed == 123

    def test_ltx2_minimal(self):
        parser = create_parser()
        args = parser.parse_args(["ltx2", "--prompt", "ocean waves"])
        assert args.subcommand == "ltx2"
        assert args.prompt == "ocean waves"

    def test_ltx2_all_flags(self):
        parser = create_parser()
        args = parser.parse_args([
            "ltx2",
            "--prompt", "ocean waves",
            "--width", "768",
            "--height", "512",
            "--num-frames", "33",
            "--fps", "24",
            "--seed", "42",
            "--guidance-scale", "3.0",
            "--use-two-stage",
            "--stage1-steps", "25",
            "--stage2-steps", "10",
        ])
        assert args.num_frames == 33
        assert args.fps == 24.0
        assert args.guidance_scale == 3.0
        assert args.use_two_stage is True
        assert args.stage1_steps == 25
        assert args.stage2_steps == 10

    def test_qwen_minimal(self):
        parser = create_parser()
        args = parser.parse_args(["qwen", "--prompt", "a bird"])
        assert args.subcommand == "qwen"
        assert args.prompt == "a bird"

    def test_qwen_all_flags(self):
        parser = create_parser()
        args = parser.parse_args([
            "qwen", "--prompt", "a bird",
            "--width", "1024", "--height", "1024",
            "--steps", "40", "--cfg-scale", "4.0",
            "--seed", "99", "--max-sequence-length", "256",
        ])
        assert args.steps == 40
        assert args.cfg_scale == 4.0
        assert args.max_sequence_length == 256

    def test_global_server_flag(self):
        parser = create_parser()
        args = parser.parse_args(["--server", "http://localhost:9000", "status"])
        assert args.server == "http://localhost:9000"

    def test_global_output_flag(self):
        parser = create_parser()
        args = parser.parse_args(["--output", "/tmp/test/", "flux2", "--prompt", "test"])
        assert args.output == "/tmp/test/"

    def test_global_timeout_flag(self):
        parser = create_parser()
        args = parser.parse_args(["--timeout", "600", "flux2", "--prompt", "test"])
        assert args.timeout == 600

    def test_global_no_save_flag(self):
        parser = create_parser()
        args = parser.parse_args(["--no-save", "flux2", "--prompt", "test"])
        assert args.no_save is True

    def test_global_json_flag(self):
        parser = create_parser()
        args = parser.parse_args(["--json", "flux2", "--prompt", "test"])
        assert args.json is True

    def test_default_server(self):
        parser = create_parser()
        args = parser.parse_args(["status"])
        assert args.server == "http://127.0.0.1:7860"

    def test_default_output(self):
        parser = create_parser()
        args = parser.parse_args(["status"])
        assert args.output == "outputs/gen/"

    def test_default_timeout(self):
        parser = create_parser()
        args = parser.parse_args(["status"])
        assert args.timeout == 300

    def test_flux2_loras_multiple(self):
        parser = create_parser()
        args = parser.parse_args([
            "flux2", "--prompt", "test",
            "--loras", "a.safetensors:0.5", "b.safetensors:1.0",
        ])
        assert args.loras == ["a.safetensors:0.5", "b.safetensors:1.0"]


# ===========================================================================
# Request Body Building
# ===========================================================================


class TestBuildRequestBody:
    """Verify None-valued args are omitted (preserves resolve_param precedence)."""

    def test_flux2_minimal_body(self):
        """Only prompt should appear when nothing else specified."""
        parser = create_parser()
        args = parser.parse_args(["flux2", "--prompt", "a cat"])
        body = build_request_body(args)
        assert body == {"prompt": "a cat"}

    def test_flux2_full_body(self):
        parser = create_parser()
        args = parser.parse_args([
            "flux2", "--prompt", "a cat",
            "--width", "1024", "--height", "768",
            "--num-steps", "20", "--seed", "42",
        ])
        body = build_request_body(args)
        assert body["prompt"] == "a cat"
        assert body["width"] == 1024
        assert body["height"] == 768
        assert body["num_steps"] == 20
        assert body["seed"] == 42

    def test_none_values_omitted(self):
        """None-valued fields must NOT appear in body (resolve_param precedence)."""
        parser = create_parser()
        args = parser.parse_args(["flux2", "--prompt", "a cat"])
        body = build_request_body(args)
        assert "seed" not in body
        assert "width" not in body
        assert "num_steps" not in body

    def test_false_bool_included(self):
        """Explicit False booleans should appear in body."""
        parser = create_parser()
        args = parser.parse_args(["flux2", "--prompt", "test", "--no-upsample-prompt"])
        body = build_request_body(args)
        assert body.get("upsample_prompt") is False

    def test_zero_seed_included(self):
        """Seed of 0 is a valid value and must be included."""
        parser = create_parser()
        args = parser.parse_args(["flux2", "--prompt", "test", "--seed", "0"])
        body = build_request_body(args)
        assert body["seed"] == 0

    def test_ltx2_body(self):
        parser = create_parser()
        args = parser.parse_args([
            "ltx2", "--prompt", "ocean waves",
            "--num-frames", "33", "--seed", "42",
        ])
        body = build_request_body(args)
        assert body["prompt"] == "ocean waves"
        assert body["num_frames"] == 33
        assert body["seed"] == 42
        # Non-specified fields omitted
        assert "width" not in body
        assert "height" not in body

    def test_qwen_body(self):
        parser = create_parser()
        args = parser.parse_args([
            "qwen", "--prompt", "a bird", "--steps", "40", "--cfg-scale", "4.0",
        ])
        body = build_request_body(args)
        assert body["prompt"] == "a bird"
        assert body["steps"] == 40
        assert body["cfg_scale"] == 4.0

    def test_zimage_body_with_template(self):
        parser = create_parser()
        args = parser.parse_args([
            "zimage", "--prompt", "a mountain", "--template", "photorealistic",
        ])
        body = build_request_body(args)
        assert body["prompt"] == "a mountain"
        assert body["template"] == "photorealistic"

    def test_loras_included_when_set(self):
        parser = create_parser()
        args = parser.parse_args([
            "flux2", "--prompt", "test", "--loras", "style.safetensors:0.8",
        ])
        body = build_request_body(args)
        assert body["loras"] == ["style.safetensors:0.8"]

    def test_loras_omitted_when_not_set(self):
        parser = create_parser()
        args = parser.parse_args(["flux2", "--prompt", "test"])
        body = build_request_body(args)
        assert "loras" not in body

    def test_global_flags_excluded_from_body(self):
        """Global flags (server, output, timeout, etc.) must not leak into request body."""
        parser = create_parser()
        args = parser.parse_args([
            "--server", "http://localhost:9000",
            "--output", "/tmp/test/",
            "--timeout", "600",
            "--no-save",
            "--json",
            "flux2", "--prompt", "test",
        ])
        body = build_request_body(args)
        assert "server" not in body
        assert "output" not in body
        assert "timeout" not in body
        assert "no_save" not in body
        assert "json" not in body
        assert "subcommand" not in body
        assert "stream" not in body


# ===========================================================================
# Endpoint Resolution
# ===========================================================================


class TestEndpointResolution:
    """Verify correct endpoint URL for each subcommand."""

    def test_flux2_sync(self):
        parser = create_parser()
        args = parser.parse_args(["flux2", "--prompt", "test"])
        assert get_endpoint(args) == "/api/flux2/generate"

    def test_flux2_stream(self):
        parser = create_parser()
        args = parser.parse_args(["flux2", "--prompt", "test", "--stream"])
        assert get_endpoint(args) == "/api/flux2/generate/stream"

    def test_zimage_sync(self):
        parser = create_parser()
        args = parser.parse_args(["zimage", "--prompt", "test"])
        assert get_endpoint(args) == "/api/generate"

    def test_zimage_stream(self):
        parser = create_parser()
        args = parser.parse_args(["zimage", "--prompt", "test", "--stream"])
        assert get_endpoint(args) == "/api/generate/stream"

    def test_ltx2_always_stream(self):
        """LTX-2 is always streaming."""
        parser = create_parser()
        args = parser.parse_args(["ltx2", "--prompt", "test"])
        assert get_endpoint(args) == "/api/ltx2/generate/stream"

    def test_qwen(self):
        parser = create_parser()
        args = parser.parse_args(["qwen", "--prompt", "test"])
        assert get_endpoint(args) == "/api/qwen-image-2512/generate"

    def test_status(self):
        parser = create_parser()
        args = parser.parse_args(["status"])
        assert get_endpoint(args) == "/api/context"


# ===========================================================================
# Response Handler Dispatch
# ===========================================================================


class TestResponseHandlerDispatch:
    """Verify correct handler type is selected for each subcommand/mode."""

    def test_flux2_sync_uses_json_handler(self):
        parser = create_parser()
        args = parser.parse_args(["flux2", "--prompt", "test"])
        assert get_response_handler(args) == "json"

    def test_flux2_stream_uses_sse_handler(self):
        parser = create_parser()
        args = parser.parse_args(["flux2", "--prompt", "test", "--stream"])
        assert get_response_handler(args) == "sse"

    def test_zimage_sync_uses_json_handler(self):
        parser = create_parser()
        args = parser.parse_args(["zimage", "--prompt", "test"])
        assert get_response_handler(args) == "json"

    def test_zimage_stream_uses_sse_handler(self):
        parser = create_parser()
        args = parser.parse_args(["zimage", "--prompt", "test", "--stream"])
        assert get_response_handler(args) == "sse"

    def test_ltx2_uses_sse_handler(self):
        parser = create_parser()
        args = parser.parse_args(["ltx2", "--prompt", "test"])
        assert get_response_handler(args) == "sse"

    def test_qwen_uses_png_handler(self):
        parser = create_parser()
        args = parser.parse_args(["qwen", "--prompt", "test"])
        assert get_response_handler(args) == "png"

    def test_status_uses_status_handler(self):
        parser = create_parser()
        args = parser.parse_args(["status"])
        assert get_response_handler(args) == "status"


# ===========================================================================
# _get_camel helper (falsy-zero safety)
# ===========================================================================


class TestGetCamel:
    """Verify _get_camel uses 'in' checks, not 'or', to avoid falsy-zero bugs."""

    def test_zero_float_preserved(self):
        """0.0 is a valid value, not a signal to fall through."""
        data = {"vramUsedGb": 0.0}
        assert _get_camel(data, "vramUsedGb", "vram_used_gb") == 0.0

    def test_zero_int_preserved(self):
        data = {"uptimeSeconds": 0}
        assert _get_camel(data, "uptimeSeconds", "uptime_seconds") == 0

    def test_empty_string_preserved(self):
        data = {"activePipeline": ""}
        assert _get_camel(data, "activePipeline", "active_pipeline", "none") == ""

    def test_falls_through_to_snake_case(self):
        data = {"active_pipeline": "flux2"}
        assert _get_camel(data, "activePipeline", "active_pipeline") == "flux2"

    def test_returns_default_when_missing(self):
        data = {}
        assert _get_camel(data, "activePipeline", "active_pipeline", "none") == "none"

    def test_returns_none_default(self):
        data = {}
        assert _get_camel(data, "vramUsedGb", "vram_used_gb") is None

    def test_camel_takes_priority_over_snake(self):
        """When both keys exist, camelCase wins."""
        data = {"activePipeline": "flux2", "active_pipeline": "ltx2"}
        assert _get_camel(data, "activePipeline", "active_pipeline") == "flux2"


# ===========================================================================
# SSE error event handling
# ===========================================================================


class TestSSEErrorHandling:
    """Verify SSE handler surfaces server error events."""

    def _make_args(self, **overrides):
        """Build a minimal args namespace for SSE handler tests."""
        import argparse
        defaults = {
            "subcommand": "ltx2",
            "server": "http://127.0.0.1:7860",
            "output": "/tmp/test/",
            "timeout": 300,
            "no_save": True,
            "json": False,
            "stream": False,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def _make_sse_response(self, events: list[str]):
        """Create a mock httpx.Response that yields SSE lines."""
        from unittest.mock import MagicMock
        resp = MagicMock()
        resp.iter_lines.return_value = iter(events)
        return resp

    def test_error_event_returns_nonzero(self):
        """Server error events should cause a non-zero exit code."""
        args = self._make_args()
        resp = self._make_sse_response([
            'data: {"type": "progress", "step": 1, "totalSteps": 10}',
            'data: {"type": "error", "message": "CUDA out of memory"}',
        ])
        client = None  # no_save=True, won't download
        result = handle_sse(resp, args, client)
        assert result != 0

    def test_error_event_prints_message(self, capsys):
        """Server error message should be printed to the user."""
        args = self._make_args()
        resp = self._make_sse_response([
            'data: {"type": "error", "message": "CUDA out of memory"}',
        ])
        result = handle_sse(resp, args, client=None)
        captured = capsys.readouterr()
        assert "CUDA out of memory" in captured.out
