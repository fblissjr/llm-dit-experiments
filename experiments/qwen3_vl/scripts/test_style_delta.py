#!/usr/bin/env python3
"""
Test style delta arithmetic for VL conditioning.

Usage:
    uv run experiments/qwen3_vl/scripts/test_style_delta.py \
        --styled experiments/inputs/purple_hexagon.png \
        --neutral experiments/inputs/gray_1024.png \
        --prompt "Homer Simpson" \
        --alphas 0.1 0.3 0.5 0.7 1.0 \
        --output-dir experiments/results/style_delta_test
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm_dit.vl import VLEmbeddingExtractor, compute_style_delta, blend_with_style_delta
from llm_dit.startup import PipelineLoader
from llm_dit.cli import load_runtime_config, create_base_parser

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_neutral_image(size: int = 1024) -> Image.Image:
    """Create a neutral gray image."""
    return Image.new("RGB", (size, size), (128, 128, 128))


def main():
    parser = argparse.ArgumentParser(description="Test style delta VL conditioning")
    parser.add_argument("--styled", type=str, required=True, help="Path to styled image")
    parser.add_argument("--neutral", type=str, default=None, help="Path to neutral image (default: generate gray)")
    parser.add_argument("--prompt", "-p", type=str, required=True, help="Text prompt")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 1.0], help="Style alpha values")
    parser.add_argument("--output-dir", "-o", type=str, default="experiments/results/style_delta_test")
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--vl-model-path", type=str, default="/home/fbliss/Storage/Qwen3-VL-4B-Instruct")
    parser.add_argument("--hidden-layer", type=int, default=-6, help="Hidden layer for VL extraction")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load styled image
    styled_image = Image.open(args.styled).convert("RGB")
    logger.info(f"Loaded styled image: {args.styled} ({styled_image.size})")

    # Load or create neutral image
    if args.neutral:
        neutral_image = Image.open(args.neutral).convert("RGB")
        logger.info(f"Loaded neutral image: {args.neutral}")
    else:
        neutral_image = create_neutral_image(styled_image.size[0])
        # Save it for reference
        neutral_path = output_dir / "neutral_gray.png"
        neutral_image.save(neutral_path)
        logger.info(f"Created neutral gray image: {neutral_path}")

    # Load VL extractor
    logger.info("Loading Qwen3-VL...")
    vl_extractor = VLEmbeddingExtractor.from_pretrained(
        args.vl_model_path,
        device="cpu",
        torch_dtype=torch.bfloat16,
    )

    # Extract VL embeddings from both images
    logger.info(f"Extracting VL embeddings from styled image (layer {args.hidden_layer})...")
    styled_result = vl_extractor.extract(
        styled_image,
        text=args.prompt,
        hidden_layer=args.hidden_layer,
        text_tokens_only=False,  # Use all tokens for style
        scale_to_text=True,
    )
    styled_vl = styled_result.embeddings
    logger.info(f"  Styled VL: shape={styled_vl.shape}, std={styled_vl.std():.2f}")

    logger.info(f"Extracting VL embeddings from neutral image (layer {args.hidden_layer})...")
    neutral_result = vl_extractor.extract(
        neutral_image,
        text=args.prompt,
        hidden_layer=args.hidden_layer,
        text_tokens_only=False,
        scale_to_text=True,
    )
    neutral_vl = neutral_result.embeddings
    logger.info(f"  Neutral VL: shape={neutral_vl.shape}, std={neutral_vl.std():.2f}")

    # Compute style delta
    logger.info("Computing style delta...")
    style_delta = compute_style_delta(styled_vl, neutral_vl, normalize=True)
    logger.info(f"  Style delta: shape={style_delta.shape}, std={style_delta.std():.2f}")

    # Unload VL model to free memory
    vl_extractor.unload()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load Z-Image pipeline
    logger.info("Loading Z-Image pipeline...")

    # Create a minimal args object for config loading
    class ConfigArgs:
        config = args.config
        profile = "default"

    config = load_runtime_config(ConfigArgs())
    loader = PipelineLoader(config)
    result = loader.load_pipeline()
    pipe = result.pipeline

    # Get text embeddings
    logger.info(f"Encoding text prompt: {args.prompt}")
    text_output = pipe.encoder.encode(args.prompt)
    text_emb = text_output.embeddings[0]
    logger.info(f"  Text embeddings: shape={text_emb.shape}, std={text_emb.std():.2f}")

    # Generate with different alphas
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    # First, generate baseline (no style delta)
    logger.info("Generating baseline (alpha=0.0, no style)...")
    baseline_image = pipe(
        prompt_embeds=text_emb,
        height=config.height,
        width=config.width,
        num_inference_steps=config.steps,
        guidance_scale=config.guidance_scale,
        generator=generator,
    )
    baseline_path = output_dir / "baseline_no_style.png"
    baseline_image.save(baseline_path)
    logger.info(f"  Saved: {baseline_path}")

    # Generate with style delta at various alphas
    for alpha in args.alphas:
        logger.info(f"Generating with style delta alpha={alpha}...")

        # Reset generator for reproducibility
        generator = torch.Generator(device="cpu").manual_seed(args.seed)

        # Apply style delta to text embeddings
        styled_emb = blend_with_style_delta(text_emb, style_delta, alpha=alpha)
        logger.info(f"  Styled embeddings: std={styled_emb.std():.2f}")

        # Generate
        image = pipe(
            prompt_embeds=styled_emb,
            height=config.height,
            width=config.width,
            num_inference_steps=config.steps,
            guidance_scale=config.guidance_scale,
            generator=generator,
        )

        output_path = output_dir / f"style_delta_alpha_{alpha}.png"
        image.save(output_path)
        logger.info(f"  Saved: {output_path}")

    # Save styled image copy for reference
    styled_copy_path = output_dir / "reference_styled.png"
    styled_image.save(styled_copy_path)

    logger.info(f"\nDone! Results in {output_dir}/")
    logger.info("Compare baseline_no_style.png with style_delta_alpha_*.png")


if __name__ == "__main__":
    main()
