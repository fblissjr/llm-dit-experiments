#!/usr/bin/env python3
"""CLI-over-API tool for generation.

Thin client that talks to the running server via HTTP.
Uses the same API surface that E2E tests validate.

Usage:
    # Check server status
    uv run scripts/gen.py status

    # FLUX.2 image generation
    uv run scripts/gen.py flux2 --prompt "a cat sleeping in sunlight" --seed 42

    # FLUX.2 with streaming progress
    uv run scripts/gen.py flux2 --prompt "a cat" --stream

    # Z-Image generation
    uv run scripts/gen.py zimage --prompt "a mountain" --width 512 --height 512

    # LTX-2 video (always streaming)
    uv run scripts/gen.py ltx2 --prompt "ocean waves" --num-frames 33 --seed 42

    # Qwen-Image T2I
    uv run scripts/gen.py qwen --prompt "a bird" --seed 42

    # Custom server URL
    uv run scripts/gen.py --server http://localhost:9000 flux2 --prompt "test"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import orjson

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SERVER = "http://127.0.0.1:7860"
DEFAULT_OUTPUT = "outputs/gen/"
DEFAULT_TIMEOUT = 300

# Fields that are global flags, NOT part of the API request body.
_GLOBAL_FIELDS = frozenset({
    "subcommand",
    "server",
    "output",
    "timeout",
    "no_save",
    "json",
    "stream",
})

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="gen",
        description="CLI client for the generation server. "
        "Talks to the running API -- same endpoints E2E tests use.",
    )

    # Global flags
    parser.add_argument(
        "--server", default=DEFAULT_SERVER,
        help=f"Server base URL (default: {DEFAULT_SERVER})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output directory for saved files (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--no-save", action="store_true", default=False,
        help="Print metadata only, don't save file",
    )
    parser.add_argument(
        "--json", action="store_true", default=False,
        help="Output raw JSON response instead of saving file",
    )

    subs = parser.add_subparsers(dest="subcommand", required=True)

    # -- status --------------------------------------------------------
    subs.add_parser("status", help="Check server status and loaded pipeline")

    # -- flux2 ---------------------------------------------------------
    flux2 = subs.add_parser("flux2", help="FLUX.2 Klein image generation")
    flux2.add_argument("--prompt", required=True)
    flux2.add_argument("--width", type=int, default=None)
    flux2.add_argument("--height", type=int, default=None)
    flux2.add_argument("--num-steps", type=int, default=None)
    flux2.add_argument("--seed", type=int, default=None)
    flux2.add_argument("--guidance", type=float, default=None)
    flux2.add_argument("--model-name", default=None)
    flux2.add_argument("--upsample-prompt", action="store_true", default=None)
    flux2.add_argument("--no-upsample-prompt", dest="upsample_prompt", action="store_false")
    flux2.add_argument("--loras", nargs="+", default=None, help="LoRA specs (path:scale)")
    flux2.add_argument("--block-offload", action="store_true", default=None)
    flux2.add_argument("--max-text-length", type=int, default=None)
    flux2.add_argument("--reference-images", nargs="+", default=None)
    flux2.add_argument("--stream", action="store_true", default=False)

    # -- zimage --------------------------------------------------------
    zimage = subs.add_parser("zimage", help="Z-Image generation")
    zimage.add_argument("--prompt", required=True)
    zimage.add_argument("--width", type=int, default=None)
    zimage.add_argument("--height", type=int, default=None)
    zimage.add_argument("--steps", type=int, default=None)
    zimage.add_argument("--seed", type=int, default=None)
    zimage.add_argument("--guidance-scale", type=float, default=None)
    zimage.add_argument("--template", default=None)
    zimage.add_argument("--negative-prompt", default=None)
    zimage.add_argument("--hidden-layer", type=int, default=None)
    zimage.add_argument("--shift", type=float, default=None)
    zimage.add_argument("--loras", nargs="+", default=None)
    zimage.add_argument("--stream", action="store_true", default=False)

    # -- ltx2 ----------------------------------------------------------
    ltx2 = subs.add_parser("ltx2", help="LTX-2 video generation (always streaming)")
    ltx2.add_argument("--prompt", required=True)
    ltx2.add_argument("--width", type=int, default=None)
    ltx2.add_argument("--height", type=int, default=None)
    ltx2.add_argument("--num-frames", type=int, default=None)
    ltx2.add_argument("--fps", type=float, default=None)
    ltx2.add_argument("--seed", type=int, default=None)
    ltx2.add_argument("--guidance-scale", type=float, default=None)
    ltx2.add_argument("--use-two-stage", action="store_true", default=None)
    ltx2.add_argument("--no-two-stage", dest="use_two_stage", action="store_false")
    ltx2.add_argument("--stage1-steps", type=int, default=None)
    ltx2.add_argument("--stage2-steps", type=int, default=None)
    ltx2.add_argument("--negative-prompt", default=None)
    ltx2.add_argument("--stg-scale", type=float, default=None)
    ltx2.add_argument("--loras", nargs="+", default=None)
    ltx2.add_argument("--lora-path", default=None)
    ltx2.add_argument("--lora-scale", type=float, default=None)
    ltx2.add_argument("--distilled-lora-path", default=None)
    ltx2.add_argument("--distilled-lora-scale", type=float, default=None)
    ltx2.add_argument("--enhance-prompt", action="store_true", default=None)
    ltx2.add_argument("--ge-gamma", type=float, default=None)
    ltx2.add_argument("--fbcache-threshold", type=float, default=None)
    ltx2.add_argument("--use-distilled-sigmas", action="store_true", default=None)
    ltx2.add_argument("--enable-audio", action="store_true", default=None)

    # -- qwen ----------------------------------------------------------
    qwen = subs.add_parser("qwen", help="Qwen-Image T2I generation")
    qwen.add_argument("--prompt", required=True)
    qwen.add_argument("--width", type=int, default=None)
    qwen.add_argument("--height", type=int, default=None)
    qwen.add_argument("--steps", type=int, default=None)
    qwen.add_argument("--cfg-scale", type=float, default=None)
    qwen.add_argument("--seed", type=int, default=None)
    qwen.add_argument("--negative-prompt", default=None)
    qwen.add_argument("--max-sequence-length", type=int, default=None)

    return parser


# ---------------------------------------------------------------------------
# Request body building
# ---------------------------------------------------------------------------


def build_request_body(args: argparse.Namespace) -> dict[str, Any]:
    """Convert parsed args to a JSON-serializable request body.

    None-valued fields are omitted so that the server's resolve_param()
    falls through to config.toml defaults. Global flags (server, output,
    timeout, etc.) are excluded.

    CLI arg names map directly to Pydantic schema field names (argparse
    converts --num-steps to num_steps, which matches the schema).
    """
    body: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in _GLOBAL_FIELDS:
            continue
        if value is None:
            continue
        body[key] = value
    return body


# ---------------------------------------------------------------------------
# Endpoint resolution
# ---------------------------------------------------------------------------


def get_endpoint(args: argparse.Namespace) -> str:
    """Return the API endpoint path for the given subcommand."""
    cmd = args.subcommand
    stream = getattr(args, "stream", False)

    if cmd == "status":
        return "/api/context"
    elif cmd == "flux2":
        return "/api/flux2/generate/stream" if stream else "/api/flux2/generate"
    elif cmd == "zimage":
        return "/api/generate/stream" if stream else "/api/generate"
    elif cmd == "ltx2":
        return "/api/ltx2/generate/stream"  # always streaming
    elif cmd == "qwen":
        return "/api/qwen-image-2512/generate"
    else:
        raise ValueError(f"Unknown subcommand: {cmd}")


# ---------------------------------------------------------------------------
# Response handler dispatch
# ---------------------------------------------------------------------------


def get_response_handler(args: argparse.Namespace) -> str:
    """Return the handler type name for the given subcommand/mode."""
    cmd = args.subcommand
    stream = getattr(args, "stream", False)

    if cmd == "status":
        return "status"
    elif cmd == "qwen":
        return "png"
    elif cmd == "ltx2":
        return "sse"
    elif stream:
        return "sse"
    else:
        return "json"


# ---------------------------------------------------------------------------
# Response handlers
# ---------------------------------------------------------------------------


def _output_path(args: argparse.Namespace, ext: str) -> Path:
    """Build output file path with timestamp."""
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    return out_dir / f"{args.subcommand}_{ts}.{ext}"


def _absolute_url(server: str, path: str) -> str:
    """Resolve a potentially-relative URL against the server base."""
    if path.startswith("/"):
        return server.rstrip("/") + path
    return path


def _download_file(client: httpx.Client, url: str, save_path: Path) -> None:
    """Download a file from the server, streaming to disk."""
    with client.stream("GET", url) as resp:
        resp.raise_for_status()
        with save_path.open("wb") as f:
            for chunk in resp.iter_bytes(chunk_size=65536):
                f.write(chunk)


def _get_camel(data: dict, camel: str, snake: str, default: Any = None) -> Any:
    """Get a value from a response dict, trying camelCase first then snake_case.

    Uses ``in`` check instead of ``or``-chaining to avoid falsy-zero bugs
    (0.0, 0, "" are valid values that ``or`` would skip).
    """
    if camel in data:
        return data[camel]
    if snake in data:
        return data[snake]
    return default


def _print_result_and_save(
    data: dict[str, Any],
    args: argparse.Namespace,
    client: httpx.Client,
) -> int:
    """Print generation metadata and optionally download the output file.

    Shared between JSON and SSE response handlers.
    """
    seed = data.get("seed", -1)
    gen_time = _get_camel(data, "generationTime", "generation_time", 0)

    print(f"Seed:       {seed}")
    print(f"Time:       {gen_time:.1f}s")

    for w in data.get("warnings", []):
        print(f"Warning:    {w}")

    if args.no_save:
        return 0

    urls = data.get("urls", [])
    url = data.get("url") or (urls[0] if urls else None)
    if not url:
        print("Error: No URL in response")
        return 1

    url = _absolute_url(args.server, url)
    ext = "mp4" if args.subcommand == "ltx2" else "png"
    save_path = _output_path(args, ext)
    _download_file(client, url, save_path)
    print(f"Saved:      {save_path}")
    return 0


def handle_status(resp: httpx.Response, args: argparse.Namespace) -> int:
    """Handle /api/context response."""
    data = resp.json()
    pipeline = _get_camel(data, "activePipeline", "active_pipeline", "none")
    display = _get_camel(data, "pipelineDisplayName", "pipeline_display_name", pipeline)
    uptime = _get_camel(data, "uptimeSeconds", "uptime_seconds", 0)
    vram_used = _get_camel(data, "vramUsedGb", "vram_used_gb")
    vram_total = _get_camel(data, "vramTotalGb", "vram_total_gb")

    print(f"Pipeline: {display}")
    print(f"Uptime:   {uptime // 3600}h {(uptime % 3600) // 60}m")
    if vram_used is not None and vram_total is not None:
        print(f"VRAM:     {vram_used:.1f} / {vram_total:.1f} GB")

    profile = data.get("profile", "default")
    if profile != "default":
        print(f"Profile:  {profile}")

    lora_summary = _get_camel(data, "loraSummary", "lora_summary")
    if lora_summary:
        print(f"LoRAs:    {lora_summary}")

    return 0


def handle_json(
    resp: httpx.Response, args: argparse.Namespace, client: httpx.Client
) -> int:
    """Handle JSON response (FLUX.2 sync, Z-Image sync)."""
    data = resp.json()

    if args.json:
        sys.stdout.buffer.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
        sys.stdout.buffer.write(b"\n")
        return 0

    return _print_result_and_save(data, args, client)


def handle_sse(
    resp: httpx.Response, args: argparse.Namespace, client: httpx.Client
) -> int:
    """Handle SSE (Server-Sent Events) response."""
    is_tty = sys.stdout.isatty()
    last_data = None

    for line in resp.iter_lines():
        if not line:
            continue
        if line.startswith("data: "):
            try:
                event = orjson.loads(line[6:])
            except Exception:
                continue

            last_data = event
            event_type = event.get("type", "")

            if event_type == "progress":
                step = event.get("step", 0)
                total = _get_camel(event, "totalSteps", "total_steps", 0)
                pct = (step / total * 100) if total > 0 else 0
                if is_tty:
                    bar = "#" * int(pct / 2.5)
                    sys.stdout.write(f"\r  [{bar:<40}] {pct:5.1f}%  step {step}/{total}")
                    sys.stdout.flush()
                else:
                    print(f"progress: {step}/{total} ({pct:.0f}%)")

            elif event_type == "status":
                msg = event.get("message", "")
                if is_tty:
                    sys.stdout.write(f"\r  {msg:<60}")
                    sys.stdout.flush()
                else:
                    print(f"status: {msg}")

            elif event_type == "error":
                if is_tty:
                    sys.stdout.write("\n")
                msg = event.get("message") or event.get("detail", "Unknown error")
                print(f"Error: Server error: {msg}")
                return 1

            elif event_type == "complete":
                if is_tty:
                    sys.stdout.write("\n")

    if last_data is None:
        print("Error: No SSE events received")
        return 1

    if args.json:
        sys.stdout.buffer.write(orjson.dumps(last_data, option=orjson.OPT_INDENT_2))
        sys.stdout.buffer.write(b"\n")
        return 0

    # Extract result from completion event
    result = last_data if last_data.get("type") == "complete" else {}
    return _print_result_and_save(result, args, client)


def handle_png(
    resp: httpx.Response, args: argparse.Namespace
) -> int:
    """Handle raw PNG response (Qwen-Image T2I)."""
    gen_time = resp.headers.get("X-Inference-Time", "?")
    print(f"Time:       {gen_time}s")

    if args.no_save:
        return 0

    save_path = _output_path(args, "png")
    save_path.write_bytes(resp.content)
    print(f"Saved:      {save_path}")
    return 0


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def check_health(client: httpx.Client, args: argparse.Namespace) -> bool:
    """Pre-flight check: is the server up?"""
    try:
        resp = client.get(f"{args.server}/api/context")
        resp.raise_for_status()
        data = resp.json()
        pipeline = _get_camel(data, "activePipeline", "active_pipeline", "none")
        if pipeline == "none":
            print("Warning: No pipeline loaded. First request will trigger lazy-load.")
        return True
    except httpx.ConnectError:
        print(f"Error: Cannot connect to server at {args.server}")
        print("Is the server running? Start with: uv run web/server.py --config config.toml")
        return False
    except Exception as e:
        print(f"Error: Health check failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Base timeout uses --timeout for the read phase (sync JSON/PNG requests).
    # SSE streaming overrides with read=None since the server may be silent
    # during long denoising loops.
    base_timeout = httpx.Timeout(
        connect=10.0, read=float(args.timeout), write=10.0, pool=5.0,
    )
    stream_timeout = httpx.Timeout(
        connect=10.0, read=None, write=10.0, pool=5.0,
    )

    with httpx.Client(timeout=base_timeout) as client:
        # Status subcommand -- just print and exit
        if args.subcommand == "status":
            endpoint = get_endpoint(args)
            url = f"{args.server.rstrip('/')}{endpoint}"
            try:
                resp = client.get(url)
                resp.raise_for_status()
            except httpx.ConnectError:
                print(f"Error: Cannot connect to server at {args.server}")
                return 1
            except httpx.HTTPStatusError as e:
                print(f"Error: {e.response.status_code} {e.response.text}")
                return 1
            return handle_status(resp, args)

        # Health check for generation subcommands
        if not check_health(client, args):
            return 1

        # Build request
        body = build_request_body(args)
        endpoint = get_endpoint(args)
        url = f"{args.server.rstrip('/')}{endpoint}"
        handler_type = get_response_handler(args)

        print(f"Endpoint:   {endpoint}")
        print(f"Prompt:     {body.get('prompt', '')[:80]}")

        try:
            if handler_type == "sse":
                # Streaming: override with unbounded read timeout for SSE
                with client.stream("POST", url, json=body, timeout=stream_timeout) as resp:
                    resp.raise_for_status()
                    return handle_sse(resp, args, client)
            elif handler_type == "png":
                resp = client.post(url, json=body)
                resp.raise_for_status()
                return handle_png(resp, args)
            else:
                # JSON
                resp = client.post(url, json=body)
                resp.raise_for_status()
                return handle_json(resp, args, client)

        except httpx.ConnectError:
            print(f"Error: Lost connection to {args.server}")
            return 1
        except httpx.HTTPStatusError as e:
            print(f"Error: {e.response.status_code}")
            try:
                detail = e.response.json()
                print(f"Detail: {detail.get('detail', e.response.text)}")
            except Exception:
                print(f"Detail: {e.response.text[:500]}")
            return 1
        except httpx.ReadTimeout:
            print(f"Error: Request timed out after {args.timeout}s")
            print("Try increasing with --timeout")
            return 1


if __name__ == "__main__":
    sys.exit(main())
