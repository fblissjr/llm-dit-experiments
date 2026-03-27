"""Shared utilities used by multiple routers.

Extracted from server.py to avoid circular imports and duplication.
"""

import base64
import io
import time
from typing import Optional, Union

import orjson

from PIL import Image

from pathlib import Path

from web.schemas import ImageGenerationResult, LoRAInfo


def sse_event(data: dict) -> str:
    """Format a dict as an SSE data line. Used by all streaming endpoints."""
    return f"data: {orjson.dumps(data).decode()}\n\n"


def create_image_response(
    image=None,  # PIL Image (optional if img_b64 provided)
    pipeline_id: str = "unknown",
    seed: int | None = None,
    generation_time: float = 0.0,
    history_id: int | None = None,
    img_b64: str | None = None,  # Pre-computed base64 (avoids double-encoding)
    warnings: list[str] | None = None,
    enhanced_prompt: str | None = None,
) -> ImageGenerationResult:
    """Create standardized JSON response for image generation endpoints.

    This shared utility ensures all image endpoints (Z-Image, FLUX.2, etc.)
    return the same format that the React frontend expects.

    Args:
        image: PIL Image object to encode (not needed if img_b64 provided)
        pipeline_id: Pipeline identifier (e.g., "zimage", "flux2")
        seed: Generation seed (or None/-1 for random)
        generation_time: Time taken in seconds
        history_id: Optional history entry ID
        img_b64: Pre-computed base64 string (skips encoding if provided)
        warnings: Optional list of warning messages about param overrides
        enhanced_prompt: Upsampled/enhanced prompt (when prompt was modified)

    Returns:
        ImageGenerationResult with: id, output_type, url, urls, thumbnail_url, seed, generation_time
    """
    # Use pre-computed base64 or encode from PIL Image
    if img_b64 is None:
        if image is None:
            raise ValueError("Either image or img_b64 must be provided")
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_b64 = base64.b64encode(img_bytes.getvalue()).decode("ascii")

    data_url = f"data:image/png;base64,{img_b64}"

    return ImageGenerationResult(
        id=history_id if history_id is not None else f"gen-{int(time.time() * 1000)}",
        pipeline_id=pipeline_id,
        output_type="image",
        url=data_url,
        urls=[data_url],
        thumbnail_url=data_url,
        seed=seed if seed is not None else -1,
        generation_time=generation_time,
        warnings=warnings or [],
        enhanced_prompt=enhanced_prompt,
    )


def get_lora_info(pipeline_obj) -> tuple[list[LoRAInfo], str | None]:
    """Extract LoRA fusion state from a pipeline's transformer.

    Works for any pipeline that stores a transformer as a dict value
    or as an attribute with _fused_lora_state.

    Returns:
        Tuple of (list of LoRAInfo models, summary string or None)
    """
    transformer = None
    if isinstance(pipeline_obj, dict):
        transformer = pipeline_obj.get("transformer")
    elif hasattr(pipeline_obj, "transformer"):
        transformer = pipeline_obj.transformer

    if transformer is None or not hasattr(transformer, "_fused_lora_state"):
        return [], None

    from llm_dit.utils.lora import get_fused_state

    state = get_fused_state(transformer)
    if state.is_empty:
        return [], None

    loras = [
        LoRAInfo(
            name=Path(r.path).stem,
            path=r.path,
            scale=r.scale,
            layers_updated=r.layers_updated,
        )
        for r in state.records
    ]
    return loras, state.summary()


def decode_base64_image(data: str, mode: str = "RGB") -> Image.Image:
    """Decode a base64-encoded image string to PIL Image.

    Handles both raw base64 and data URL prefixed strings.

    Args:
        data: Base64-encoded image string (with or without data: prefix)
        mode: PIL image mode to convert to (default "RGB")

    Returns:
        PIL Image in the specified mode
    """
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    image_bytes = base64.b64decode(data)
    return Image.open(io.BytesIO(image_bytes)).convert(mode)


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string.

    Args:
        image: PIL Image object
        format: Output format (default "PNG")

    Returns:
        Base64-encoded string (no data: prefix)
    """
    img_bytes = io.BytesIO()
    image.save(img_bytes, format=format)
    return base64.b64encode(img_bytes.getvalue()).decode("ascii")
