#!/usr/bin/env python3
"""
Extract vision-conditioned embeddings from Qwen3-VL.

This script uses the core VLEmbeddingExtractor from src/llm_dit/vl/ and
respects the config.toml settings for model paths and defaults.

Usage:
    # Basic extraction using config.toml settings
    uv run experiments/qwen3_vl/extract_embeddings.py \
        --config config.toml --profile rtx4090 \
        --image reference.png \
        --output embeddings.pt

    # Override model path via CLI
    uv run experiments/qwen3_vl/extract_embeddings.py \
        --vl-model-path /path/to/Qwen3-VL-4B-Instruct \
        --image reference.png \
        --output embeddings.pt

    # With text description (recommended for better quality)
    uv run experiments/qwen3_vl/extract_embeddings.py \
        --config config.toml \
        --image reference.png \
        --text "A house with a red roof" \
        --output embeddings.pt

    # Extract only image tokens
    uv run experiments/qwen3_vl/extract_embeddings.py \
        --config config.toml \
        --image reference.png \
        --vl-image-tokens-only \
        --output embeddings.pt

    # Use different hidden layer
    uv run experiments/qwen3_vl/extract_embeddings.py \
        --config config.toml \
        --image reference.png \
        --vl-hidden-layer -3 \
        --output embeddings.pt

Note: This script uses the same CLI flags as the main tools (--vl-model-path,
--vl-hidden-layer, etc.) for consistency. Config.toml is the source of truth.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from PIL import Image

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from llm_dit.cli import load_runtime_config
from llm_dit.vl import VLEmbeddingExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Extract vision-conditioned embeddings using core VLEmbeddingExtractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config file options (same as main CLI)
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config.toml file",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="default",
        help="Config profile to use (default: 'default')",
    )

    # VL-specific flags (same names as main CLI for consistency)
    parser.add_argument(
        "--vl-model-path",
        type=str,
        help="Path to Qwen3-VL model (overrides config.toml)",
    )
    parser.add_argument(
        "--vl-device",
        type=str,
        help="Device for VL model (overrides config.toml)",
    )
    parser.add_argument(
        "--vl-hidden-layer",
        type=int,
        help="Hidden layer to extract (overrides config.toml, default: -2)",
    )
    parser.add_argument(
        "--vl-image-tokens-only",
        action="store_true",
        help="Only extract image token hidden states",
    )
    parser.add_argument(
        "--vl-no-scale",
        action="store_true",
        help="Don't scale embeddings to match text statistics",
    )

    # Script-specific args
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output path for embeddings (.pt file)",
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        default=None,
        help="Optional text description to include with image",
    )

    args = parser.parse_args()

    # Load config using the standard config loading (same as web/server.py and scripts/generate.py)
    # This creates a minimal args namespace for load_runtime_config
    config_args = argparse.Namespace(
        config=args.config,
        profile=args.profile,
        # VL overrides
        vl_model_path=args.vl_model_path,
        vl_device=args.vl_device,
        vl_hidden_layer=args.vl_hidden_layer,
        vl_auto_unload=True,  # Default for extraction scripts
        vl_alpha=0.3,  # Not used for extraction, but required
        vl_blend_mode="linear",  # Not used for extraction
        # Required fields with defaults
        model_path=None,
        encoder_device=None,
        dit_device=None,
        vae_device=None,
        templates_dir=None,
    )

    # Add all other required RuntimeConfig fields with None/defaults
    for field in [
        'torch_dtype', 'quantization', 'encoder_cpu_offload', 'encoder_max_length',
        'pipeline_torch_dtype', 'pipeline_device', 'enable_model_cpu_offload',
        'enable_sequential_cpu_offload', 'width', 'height', 'num_inference_steps',
        'guidance_scale', 'enable_thinking', 'default_template', 'shift',
        'flash_attn', 'compile', 'cpu_offload', 'attention_backend',
        'use_custom_scheduler', 'tiled_vae', 'tile_size', 'tile_overlap',
        'embedding_cache', 'cache_size', 'long_prompt_mode', 'hidden_layer',
        'api_url', 'api_model', 'use_api_encoder', 'lora_paths', 'lora_scales',
        'rewriter_use_api', 'rewriter_api_url', 'rewriter_api_model',
        'rewriter_temperature', 'rewriter_top_p', 'rewriter_top_k',
        'rewriter_min_p', 'rewriter_presence_penalty', 'rewriter_max_tokens',
        'debug',
    ]:
        if not hasattr(config_args, field):
            setattr(config_args, field, None)

    try:
        runtime_config = load_runtime_config(config_args)
    except Exception as e:
        logger.warning(f"Could not load config: {e}. Using CLI args only.")
        runtime_config = None

    # Determine VL model path (CLI overrides config)
    vl_model_path = args.vl_model_path
    if not vl_model_path and runtime_config:
        vl_model_path = runtime_config.vl_model_path

    if not vl_model_path:
        # Try common locations as fallback
        candidates = [
            Path.home() / "Storage" / "Qwen3-VL-4B-Instruct",
            Path.home() / "models" / "Qwen3-VL-4B-Instruct",
            Path("/models/Qwen3-VL-4B-Instruct"),
        ]
        for candidate in candidates:
            if candidate.exists():
                vl_model_path = str(candidate)
                logger.info(f"Auto-detected VL model at {vl_model_path}")
                break

    if not vl_model_path:
        logger.error(
            "Could not find Qwen3-VL model. Either:\n"
            "  1. Set vl.model_path in config.toml\n"
            "  2. Use --vl-model-path CLI flag\n"
            "  3. Place model in ~/Storage/Qwen3-VL-4B-Instruct"
        )
        return 1

    # Determine device (CLI overrides config)
    vl_device = args.vl_device
    if not vl_device and runtime_config:
        vl_device = runtime_config.vl_device
    if not vl_device:
        vl_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine hidden layer (CLI overrides config)
    hidden_layer = args.vl_hidden_layer
    if hidden_layer is None and runtime_config:
        hidden_layer = runtime_config.vl_hidden_layer
    if hidden_layer is None:
        hidden_layer = -2

    # Determine target std from config
    target_std = 58.75
    if runtime_config and hasattr(runtime_config, 'vl_target_std'):
        target_std = runtime_config.vl_target_std or 58.75

    # Load image
    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return 1

    image = Image.open(image_path).convert("RGB")
    logger.info(f"Loaded image: {image_path} ({image.size[0]}x{image.size[1]})")

    # Use the core VLEmbeddingExtractor
    logger.info(f"Loading VLEmbeddingExtractor from {vl_model_path}")
    logger.info(f"Device: {vl_device}, Hidden layer: {hidden_layer}")

    vl_dtype = torch.bfloat16 if vl_device == "cuda" else torch.float32
    extractor = VLEmbeddingExtractor.from_pretrained(
        vl_model_path,
        device=vl_device,
        torch_dtype=vl_dtype,
    )

    # Extract embeddings using the core module
    result = extractor.extract(
        image=image,
        text=args.text,
        hidden_layer=hidden_layer,
        image_tokens_only=args.vl_image_tokens_only,
        scale_to_text=not args.vl_no_scale,
        target_std=target_std,
    )

    # Build output dict matching the expected format
    output = {
        "embeddings": result.embeddings.cpu(),
        "shape": result.embeddings.shape,
        "hidden_layer": result.hidden_layer,
        "original_std": result.original_std,
        "scale_factor": result.scale_factor,
        "scaled_std": result.scaled_std,
        "image_tokens_only": result.image_tokens_only,
        "text": args.text,
        "num_tokens": result.num_tokens,
        "source_image": str(image_path.absolute()),
    }

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)

    logger.info(f"Saved embeddings to {output_path}")
    logger.info(f"  Shape: {output['shape']}")
    logger.info(f"  Tokens: {output['num_tokens']}")
    logger.info(f"  Original std: {output['original_std']:.2f}")
    logger.info(f"  Scaled std: {output['scaled_std']:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
