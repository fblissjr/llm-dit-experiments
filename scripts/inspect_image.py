#!/usr/bin/env python3
"""
Image inspection utility for testing generation outputs.

Features:
- Extract and decode base64 images from API responses
- Create thumbnails for quick visual inspection
- Display image metadata (size, format, hash)
- Save images for manual review

Usage:
    # Inspect a base64 response file
    uv run python scripts/inspect_image.py /path/to/response.json

    # Inspect from stdin
    curl ... | uv run python scripts/inspect_image.py -

    # Quick inline test with a generation
    uv run python scripts/inspect_image.py --generate --pipeline zimage --prompt "A cat"

last updated: 2026-01-26
"""

import argparse
import base64
import hashlib
import io
import json
import sys
from pathlib import Path

from PIL import Image


def decode_data_url(data_url: str) -> bytes:
    """Decode a data URL to bytes."""
    if not data_url.startswith("data:"):
        raise ValueError("Not a data URL")

    # Format: data:image/png;base64,<data>
    header, encoded = data_url.split(",", 1)
    return base64.b64decode(encoded)


def image_info(img: Image.Image) -> dict:
    """Get image metadata including noise detection metrics."""
    import numpy as np

    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_hash = hashlib.md5(img_bytes.getvalue()).hexdigest()[:12]

    # Noise detection metrics
    arr = np.array(img)
    variance = float(np.var(arr))
    mean = float(np.mean(arr))

    # Noise thresholds based on test_pure_pytorch.py
    # Valid images: variance 500-6000, mean 50-200
    # Noise: variance > 6000, mean ~127
    is_noise = variance > 6000 or not (30 < mean < 220)
    is_valid = 500 < variance < 6000 and 30 < mean < 220

    return {
        "size": f"{img.width}x{img.height}",
        "mode": img.mode,
        "format": img.format or "PNG",
        "hash": img_hash,
        "bytes": len(img_bytes.getvalue()),
        "variance": variance,
        "mean": mean,
        "is_valid": is_valid,
        "is_noise": is_noise,
    }


def create_thumbnail(img: Image.Image, max_size: int = 256) -> Image.Image:
    """Create a thumbnail for quick inspection."""
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img


def inspect_response(data: dict, output_dir: Path | None = None) -> dict:
    """Inspect a generation API response.

    Args:
        data: API response dict with 'url' containing data URL
        output_dir: Optional directory to save the image

    Returns:
        Inspection results dict
    """
    results = {
        "response_keys": list(data.keys()),
        "pipeline_id": data.get("pipeline_id"),
        "output_type": data.get("output_type"),
        "seed": data.get("seed"),
        "generation_time": data.get("generation_time"),
    }

    url = data.get("url", "")
    if not url:
        results["error"] = "No 'url' field in response"
        return results

    if not url.startswith("data:"):
        results["error"] = f"URL is not a data URL: {url[:50]}..."
        return results

    try:
        img_bytes = decode_data_url(url)
        img = Image.open(io.BytesIO(img_bytes))
        info = image_info(img)
        results["image"] = info
        results["data_url_length"] = len(url)

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            # Save full image
            full_path = output_dir / f"{info['hash']}_full.png"
            img.save(full_path)
            results["saved_full"] = str(full_path)

            # Save thumbnail
            thumb = create_thumbnail(img.copy())
            thumb_path = output_dir / f"{info['hash']}_thumb.png"
            thumb.save(thumb_path)
            results["saved_thumb"] = str(thumb_path)

    except Exception as e:
        results["error"] = str(e)

    return results


def generate_and_inspect(
    pipeline: str,
    prompt: str,
    steps: int = 4,
    width: int = 512,
    height: int = 512,
    shift: float | None = None,
    guidance_scale: float | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Generate an image and inspect it.

    Args:
        pipeline: Pipeline ID (zimage, flux2)
        prompt: Text prompt
        steps: Number of inference steps
        width: Image width
        height: Image height
        output_dir: Optional directory to save images

    Returns:
        Inspection results
    """
    import httpx

    endpoint = {
        "zimage": "http://localhost:7860/api/generate",
        "flux2": "http://localhost:7860/api/flux2/generate",
    }.get(pipeline)

    if not endpoint:
        return {"error": f"Unknown pipeline: {pipeline}"}

    params = {
        "prompt": prompt,
        "steps": steps,
        "width": width,
        "height": height,
    }
    # Add optional params if specified
    if shift is not None:
        params["shift"] = shift
    if guidance_scale is not None:
        params["guidance_scale"] = guidance_scale

    print(f"Generating with {pipeline}...")
    print(f"  Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
    print(f"  Steps: {steps}, Size: {width}x{height}")

    try:
        with httpx.Client(timeout=300) as client:
            response = client.post(endpoint, json=params)
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        return {"error": f"Request failed: {e}"}

    results = inspect_response(data, output_dir)
    results["request"] = params
    return results


def print_results(results: dict):
    """Pretty print inspection results."""
    print("\n" + "=" * 60)
    print("IMAGE INSPECTION RESULTS")
    print("=" * 60)

    if "error" in results:
        print(f"❌ ERROR: {results['error']}")
        return

    if "request" in results:
        print(f"\n📝 Request:")
        for k, v in results["request"].items():
            print(f"   {k}: {v}")

    print(f"\n📦 Response:")
    print(f"   pipeline_id: {results.get('pipeline_id')}")
    print(f"   output_type: {results.get('output_type')}")
    print(f"   seed: {results.get('seed')}")
    print(f"   generation_time: {results.get('generation_time')}")

    if "image" in results:
        print(f"\n🖼️  Image:")
        img = results["image"]
        print(f"   Size: {img['size']}")
        print(f"   Mode: {img['mode']}")
        print(f"   Format: {img['format']}")
        print(f"   Hash: {img['hash']}")
        print(f"   Bytes: {img['bytes']:,}")

        # Noise detection metrics
        variance = img.get('variance', 0)
        mean = img.get('mean', 0)
        is_valid = img.get('is_valid', False)
        is_noise = img.get('is_noise', False)

        print(f"\n📊 Quality Metrics:")
        print(f"   Variance: {variance:.1f} (valid: 500-6000, noise: >6000)")
        print(f"   Mean: {mean:.1f} (valid: 30-220, noise: ~127)")

        if is_noise:
            print(f"   ❌ LIKELY NOISE - Image may be pure noise/artifacts!")
        elif is_valid:
            print(f"   ✅ VALID IMAGE - Passes noise detection")
        else:
            print(f"   ⚠️  UNCLEAR - Check manually")

    if "saved_full" in results:
        print(f"\n💾 Saved:")
        print(f"   Full: {results['saved_full']}")
        print(f"   Thumb: {results.get('saved_thumb')}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Inspect generation images")
    parser.add_argument(
        "input",
        nargs="?",
        help="JSON response file path or '-' for stdin",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate a new image instead of inspecting existing",
    )
    parser.add_argument(
        "--pipeline",
        default="flux2",
        help="Pipeline for generation (zimage, flux2)",
    )
    parser.add_argument(
        "--prompt",
        default="A cat sleeping in warm sunlight",
        help="Prompt for generation",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Inference steps",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=None,
        help="Scheduler shift (6.0 for BASE, 3.0 for turbo). Uses server default if not specified.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="CFG scale (4.0 for BASE, 0.0 for turbo). Uses server default if not specified.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("/tmp/claude/inspect"),
        help="Output directory for saved images",
    )
    args = parser.parse_args()

    if args.generate:
        results = generate_and_inspect(
            pipeline=args.pipeline,
            prompt=args.prompt,
            steps=args.steps,
            width=args.width,
            height=args.height,
            shift=args.shift,
            guidance_scale=getattr(args, "guidance_scale", None),
            output_dir=args.output,
        )
    elif args.input:
        if args.input == "-":
            data = json.load(sys.stdin)
        else:
            data = json.loads(Path(args.input).read_text())
        results = inspect_response(data, args.output)
    else:
        parser.print_help()
        return

    print_results(results)


if __name__ == "__main__":
    main()
