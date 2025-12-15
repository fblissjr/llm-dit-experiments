#!/usr/bin/env python3
"""
Test VL caption-to-image generation at different caption lengths.

This experiment:
1. Takes an input image
2. Uses Qwen3-VL (Thinking variant preferred) to generate captions at various lengths
3. Feeds those captions to Z-Image for generation
4. Compares results across caption lengths

Usage:
    uv run experiments/scripts/test_caption_lengths.py \
        --image experiments/inputs/sunset.png \
        --lengths 256,512,768,1024 \
        --seed 42 \
        -o experiments/results/caption_lengths_test
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_dit.vl import VLEmbeddingExtractor
from llm_dit.startup import PipelineLoader
from llm_dit.cli import load_config, load_runtime_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_caption(
    extractor: VLEmbeddingExtractor,
    image: Image.Image,
    max_new_tokens: int,
    system_prompt: str | None = None,
) -> tuple[str, int]:
    """
    Generate a caption for an image.

    Returns:
        Tuple of (caption_text, actual_token_count)
    """
    default_system = (
        "You are a professional image captioner. Describe this image in detail "
        "for use as an image generation prompt. Include visual elements, style, "
        "lighting, mood, composition, and any notable details. Be descriptive but concise."
    )

    result = extractor.generate(
        image=image,
        system_prompt=system_prompt or default_system,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        presence_penalty=1.5,
    )

    # Strip think block for the caption
    caption = result
    if "<think>" in caption and "</think>" in caption:
        # Extract content after </think>
        think_end = caption.find("</think>")
        caption = caption[think_end + len("</think>"):].strip()

    # Count tokens in the caption
    token_count = len(extractor.processor.tokenizer.encode(caption))

    return caption, token_count


def main():
    parser = argparse.ArgumentParser(description="Test caption lengths for image generation")
    parser.add_argument("--image", "-i", required=True, help="Input image path")
    parser.add_argument(
        "--lengths", "-l",
        default="256,512,768,1024",
        help="Comma-separated list of max_new_tokens values"
    )
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--config", default="config.toml", help="Config file")
    parser.add_argument("--profile", default="rtx4090", help="Config profile")
    parser.add_argument("--vl-model", help="Path to Qwen3-VL model (auto-detected if not specified)")
    parser.add_argument("--hidden-layer", type=int, default=-2, help="Hidden layer for Z-Image encoding")
    parser.add_argument("--steps", type=int, default=9, help="Inference steps")
    parser.add_argument("--width", type=int, default=1024, help="Output width")
    parser.add_argument("--height", type=int, default=1024, help="Output height")
    parser.add_argument("--system-prompt", help="Custom system prompt for captioning")
    args = parser.parse_args()

    # Parse lengths
    lengths = [int(x.strip()) for x in args.lengths.split(",")]
    logger.info(f"Testing caption lengths: {lengths}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load input image
    input_image = Image.open(args.image).convert("RGB")
    logger.info(f"Loaded input image: {args.image} ({input_image.size})")

    # Save input image to output
    input_image.save(output_dir / "input.png")

    # Find and load VL model
    if args.vl_model:
        vl_path = args.vl_model
        vl_variant = "unknown"
    else:
        vl_path, vl_variant = VLEmbeddingExtractor.find_model_path(prefer_variant="thinking")
        if not vl_path:
            logger.error("Could not find Qwen3-VL model. Specify with --vl-model")
            sys.exit(1)

    logger.info(f"Loading Qwen3-VL from {vl_path} (variant: {vl_variant})")
    vl_extractor = VLEmbeddingExtractor.from_pretrained(vl_path, device="cuda")
    logger.info(f"Qwen3-VL loaded (variant: {vl_extractor.model_variant})")

    # Generate captions at each length
    captions = {}
    for max_tokens in lengths:
        logger.info(f"Generating caption with max_new_tokens={max_tokens}...")
        caption, token_count = generate_caption(
            vl_extractor,
            input_image,
            max_new_tokens=max_tokens,
            system_prompt=args.system_prompt,
        )
        captions[max_tokens] = {
            "text": caption,
            "token_count": token_count,
            "max_new_tokens": max_tokens,
        }
        logger.info(f"  Generated {token_count} tokens: {caption[:100]}...")

    # Save captions
    with open(output_dir / "captions.json", "w") as f:
        json.dump(captions, f, indent=2)
    logger.info(f"Saved captions to {output_dir / 'captions.json'}")

    # Unload VL model to free VRAM
    vl_extractor.unload()
    torch.cuda.empty_cache()
    logger.info("Unloaded Qwen3-VL to free VRAM")

    # Load Z-Image pipeline
    logger.info("Loading Z-Image pipeline...")
    config = load_config(args.config)
    runtime_config = load_runtime_config(config, args.profile, argparse.Namespace(
        model_path=None,
        text_encoder_device=None,
        dit_device=None,
        vae_device=None,
        hidden_layer=args.hidden_layer,
    ))

    loader = PipelineLoader(runtime_config)
    result = loader.load_pipeline()
    pipe = result.pipeline
    logger.info("Z-Image pipeline loaded")

    # Generate images for each caption
    results = []
    for max_tokens, caption_data in captions.items():
        caption = caption_data["text"]
        token_count = caption_data["token_count"]

        logger.info(f"Generating image for {max_tokens}-token caption ({token_count} actual)...")

        # Set seed
        generator = torch.Generator(device="cpu").manual_seed(args.seed)

        # Generate
        output = pipe(
            prompt=caption,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            generator=generator,
        )

        # Save image
        output_path = output_dir / f"caption_{max_tokens}tokens_seed{args.seed}.png"
        output.images[0].save(output_path)
        logger.info(f"  Saved to {output_path}")

        results.append({
            "max_tokens": max_tokens,
            "actual_tokens": token_count,
            "caption": caption,
            "output_path": str(output_path),
        })

    # Save results metadata
    metadata = {
        "input_image": args.image,
        "seed": args.seed,
        "hidden_layer": args.hidden_layer,
        "steps": args.steps,
        "vl_model": vl_path,
        "vl_variant": vl_extractor.model_variant if hasattr(vl_extractor, 'model_variant') else vl_variant,
        "results": results,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Create comparison grid
    logger.info("Creating comparison grid...")
    create_grid(output_dir, lengths, args.seed)

    logger.info(f"Experiment complete! Results in {output_dir}")


def create_grid(output_dir: Path, lengths: list[int], seed: int):
    """Create a comparison grid of all generated images."""
    images = []
    labels = []

    # Load input image
    input_img = Image.open(output_dir / "input.png")
    input_img.thumbnail((512, 512), Image.LANCZOS)
    images.append(input_img)
    labels.append("Input")

    # Load generated images
    for max_tokens in lengths:
        img_path = output_dir / f"caption_{max_tokens}tokens_seed{seed}.png"
        if img_path.exists():
            img = Image.open(img_path)
            img.thumbnail((512, 512), Image.LANCZOS)
            images.append(img)
            labels.append(f"{max_tokens} tokens")

    if len(images) < 2:
        logger.warning("Not enough images for grid")
        return

    # Create grid
    padding = 10
    header_height = 40
    thumb_size = 512

    n_cols = len(images)
    grid_width = n_cols * (thumb_size + padding) + padding
    grid_height = thumb_size + header_height + 2 * padding

    grid = Image.new("RGB", (grid_width, grid_height), "white")
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    for i, (img, label) in enumerate(zip(images, labels)):
        x = padding + i * (thumb_size + padding)

        # Draw label
        draw.text((x + thumb_size // 2, padding), label, fill="black", anchor="mt", font=font)

        # Paste image
        grid.paste(img, (x, header_height + padding))

    grid.save(output_dir / "comparison_grid.png")
    logger.info(f"Saved grid to {output_dir / 'comparison_grid.png'}")


if __name__ == "__main__":
    main()
