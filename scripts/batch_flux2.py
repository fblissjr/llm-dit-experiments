#!/usr/bin/env python3
"""Batch FLUX.2 Klein generation -- same prompt, different reference images.

Reads server URL and default model from config.toml. Loops over all images
in --input-dir, POSTs each to the streaming API, saves output to --output-dir.
Supports resume (skips images whose output already exists).

Usage:
    # Basic (reads config.toml for server + model defaults)
    uv run scripts/batch_flux2.py \
      --input-dir /path/to/153/images \
      --prompt "make this look like a watercolor painting"

    # Override model + output dir
    uv run scripts/batch_flux2.py \
      --input-dir /path/to/images \
      --output-dir /path/to/outputs \
      --prompt "transform this" \
      --model-name klein-9b-kv

    # Match output size to input image
    uv run scripts/batch_flux2.py \
      --input-dir /data/in \
      --prompt "enhance" \
      --match-image-size "0 (First Image)"
"""

from __future__ import annotations

import argparse
import base64
import datetime
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import orjson

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
_STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=5.0)

# Fields that are batch-script-specific, NOT part of the API request body.
_LOCAL_FIELDS = frozenset({
    "input_dir",
    "output_dir",
    "server",
    "timeout",
    "no_resume",
    "config",
    "label",
})


# ---------------------------------------------------------------------------
# Config loading (lightweight TOML read)
# ---------------------------------------------------------------------------


def load_batch_config(config_path: Path) -> dict[str, str]:
    """Read server URL and model default from config.toml.

    Returns dict with 'server_url' and 'model_name'. Falls back to
    sensible defaults if the file or sections are missing.
    """
    defaults = {
        "server_url": "http://127.0.0.1:7860",
        "model_name": "klein-9b-fp8",
    }

    if not config_path.exists():
        return defaults

    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            return defaults

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    server = data.get("server", {})
    host = server.get("host", "127.0.0.1")
    port = server.get("port", 7860)
    defaults["server_url"] = f"http://{host}:{port}"

    flux2 = data.get("flux2", {})
    if "default_model" in flux2:
        defaults["model_name"] = flux2["default_model"]

    return defaults


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def collect_images(input_dir: Path) -> list[Path]:
    """Find all image files in input_dir, sorted alphabetically."""
    images = []
    for p in input_dir.iterdir():
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS:
            images.append(p)
    return sorted(images, key=lambda p: p.name)


def encode_image_b64(path: Path) -> str:
    """Read an image file and return its base64 encoding (no data: prefix)."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def output_path_for(output_dir: Path, input_path: Path) -> Path:
    """Generate output path preserving the input filename."""
    return output_dir / input_path.name


def should_skip(output_dir: Path, filename: str) -> bool:
    """Check if output already exists and is non-empty (for resume)."""
    out = output_dir / filename
    return out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Request body
# ---------------------------------------------------------------------------


def build_body(args: argparse.Namespace, image_b64: str) -> dict[str, Any]:
    """Build API request body from parsed args + a single base64 image.

    Omits None-valued fields so the server's resolve_param() falls
    through to config.toml defaults. Batch-specific fields are excluded.
    """
    body: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in _LOCAL_FIELDS:
            continue
        if value is None:
            continue
        body[key] = value
    body["reference_images"] = [image_b64]
    return body


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for batch generation."""
    parser = argparse.ArgumentParser(
        description="Batch FLUX.2 generation: same prompt, different reference images.",
    )
    parser.add_argument("--input-dir", required=True, help="Directory of input images")
    parser.add_argument("--output-dir", default="outputs/batch/", help="Output directory")
    parser.add_argument("--prompt", required=True, help="Generation prompt (same for all)")
    parser.add_argument("--config", default="config.toml", help="Path to config.toml")

    # API params (None = use server defaults from config.toml)
    parser.add_argument("--model-name", default=None, help="Model variant (default: from config.toml)")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--match-image-size", default=None, help='e.g. "0 (First Image)"')

    # Batch control
    parser.add_argument("--server", default=None, help="Server URL (default: from config.toml)")
    parser.add_argument("--timeout", type=int, default=300, help="Per-image timeout (seconds)")
    parser.add_argument("--no-resume", action="store_true", default=False,
                        help="Regenerate even if output exists")
    parser.add_argument("--label", default=None, help="Label for this run (used in metrics filename)")

    return parser


# ---------------------------------------------------------------------------
# SSE streaming handler
# ---------------------------------------------------------------------------


def _stream_generate(
    client: httpx.Client,
    server_url: str,
    body: dict[str, Any],
) -> dict | None:
    """POST to streaming endpoint and return the completion event data.

    Prints progress inline. Returns the final SSE completion payload,
    or None on error.
    """
    url = f"{server_url.rstrip('/')}/api/flux2/generate/stream"
    is_tty = sys.stdout.isatty()

    try:
        with client.stream("POST", url, json=body, timeout=_STREAM_TIMEOUT) as resp:
            resp.raise_for_status()
            last_data = None
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                try:
                    event = orjson.loads(line[6:])
                except Exception:
                    continue

                last_data = event
                etype = event.get("type", "")

                if etype == "progress":
                    step = event.get("step", 0)
                    total = event["totalSteps"] if "totalSteps" in event else event.get("total_steps", 0)
                    pct = (step / total * 100) if total > 0 else 0
                    if is_tty:
                        bar = "#" * int(pct / 2.5)
                        sys.stdout.write(f"\r    [{bar:<40}] {pct:5.1f}%")
                        sys.stdout.flush()
                elif etype == "error":
                    if is_tty:
                        sys.stdout.write("\n")
                    msg = event.get("message") or event.get("detail", "unknown")
                    print(f"    Error: {msg}")
                    return None
                elif etype == "complete":
                    if is_tty:
                        sys.stdout.write("\r" + " " * 60 + "\r")
                        sys.stdout.flush()

            return last_data

    except httpx.HTTPStatusError as e:
        print(f"    HTTP {e.response.status_code}: {e.response.text[:200]}")
        return None
    except httpx.ConnectError:
        print(f"    Error: Cannot connect to {server_url}")
        return None


def _save_image(client: httpx.Client, server_url: str, url_or_data: str, save_path: Path) -> bool:
    """Save an image from a data: URL or server path."""
    try:
        if url_or_data.startswith("data:"):
            # Inline base64 from SSE complete event -- no HTTP round-trip
            _, b64 = url_or_data.split(",", 1)
            save_path.write_bytes(base64.b64decode(b64))
        else:
            full_url = url_or_data if url_or_data.startswith("http") else f"{server_url.rstrip('/')}{url_or_data}"
            resp = client.get(full_url)
            resp.raise_for_status()
            save_path.write_bytes(resp.content)
        return True
    except Exception as e:
        print(f"    Save failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Load config.toml defaults
    cfg = load_batch_config(Path(args.config))

    # Apply config defaults where CLI didn't override
    if args.server is None:
        args.server = cfg["server_url"]
    if args.model_name is None:
        args.model_name = cfg["model_name"]

    server_url = args.server

    # Collect input images
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    images = collect_images(input_dir)
    if not images:
        print(f"Error: No images found in {input_dir}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build skip set once (single stat pass, reused in loop)
    total = len(images)
    skip_set: set[str] = set()
    if not args.no_resume:
        skip_set = {img.name for img in images if should_skip(output_dir, img.name)}
    skipped = len(skip_set)

    print(f"Batch FLUX.2 Generation")
    print(f"  Model:    {args.model_name}")
    print(f"  Server:   {server_url}")
    print(f"  Prompt:   {args.prompt[:80]}")
    print(f"  Images:   {total} ({skipped} already done, {total - skipped} remaining)")
    print(f"  Input:    {input_dir}")
    print(f"  Output:   {output_dir}")
    print()

    if total - skipped == 0:
        print("All images already generated. Use --no-resume to regenerate.")
        return 0

    base_timeout = httpx.Timeout(connect=10.0, read=float(args.timeout), write=10.0, pool=5.0)
    completed = 0
    failed = 0
    timings: list[dict[str, Any]] = []
    batch_start = time.monotonic()

    with httpx.Client(timeout=base_timeout) as client:
        for img_path in images:
            # Resume: skip if output exists
            if img_path.name in skip_set:
                continue

            completed_so_far = completed + skipped
            remaining = total - completed_so_far - failed
            elapsed = time.monotonic() - batch_start
            eta = ""
            if completed > 0:
                per_image = elapsed / completed
                eta_secs = per_image * remaining
                eta = f"  ETA: {eta_secs / 60:.1f}m" if eta_secs > 60 else f"  ETA: {eta_secs:.0f}s"

            print(f"[{completed_so_far + 1}/{total}] {img_path.name}{eta}")

            # Encode and send
            img_b64 = encode_image_b64(img_path)
            body = build_body(args, img_b64)

            t0 = time.monotonic()
            result = _stream_generate(client, server_url, body)
            gen_time = time.monotonic() - t0

            if result is None or result.get("type") != "complete":
                print(f"    FAILED ({gen_time:.1f}s)")
                timings.append({"file": img_path.name, "time": gen_time, "status": "failed"})
                failed += 1
                continue

            # Save output image
            urls = result.get("urls") or []
            url = result.get("url") or (urls[0] if urls else "")
            out_path = output_path_for(output_dir, img_path)
            if url and _save_image(client, server_url, url, out_path):
                print(f"    OK {gen_time:.1f}s -> {out_path.name}")
                timings.append({"file": img_path.name, "time": gen_time, "status": "ok"})
                completed += 1
            else:
                print(f"    FAILED to save ({gen_time:.1f}s)")
                timings.append({"file": img_path.name, "time": gen_time, "status": "save_failed"})
                failed += 1

    total_time = time.monotonic() - batch_start
    ok_times = [t["time"] for t in timings if t["status"] == "ok"]

    # Summary
    print()
    print(f"Done: {completed} completed, {failed} failed, {skipped} skipped")
    print(f"Total time: {total_time / 60:.1f}m")
    if ok_times:
        avg = sum(ok_times) / len(ok_times)
        fastest = min(ok_times)
        slowest = max(ok_times)
        its = len(ok_times) / sum(ok_times) if sum(ok_times) > 0 else 0
        print(f"Per image:  avg {avg:.1f}s, min {fastest:.1f}s, max {slowest:.1f}s")
        print(f"Throughput: {its:.2f} it/s")

    # Write metrics JSON
    metrics = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "label": args.label,
        "model_name": args.model_name,
        "prompt": args.prompt,
        "total": total,
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "total_time": round(total_time, 2),
        "avg_time": round(sum(ok_times) / len(ok_times), 2) if ok_times else None,
        "min_time": round(min(ok_times), 2) if ok_times else None,
        "max_time": round(max(ok_times), 2) if ok_times else None,
        "throughput_its": round(len(ok_times) / sum(ok_times), 3) if ok_times and sum(ok_times) > 0 else None,
        "per_image": timings,
    }
    label = args.label or args.model_name or "batch"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = output_dir / f"metrics_{label}_{ts}.json"
    metrics_path.write_bytes(orjson.dumps(metrics, option=orjson.OPT_INDENT_2))
    print(f"Metrics:    {metrics_path}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
