#!/usr/bin/env python
"""
Test Qwen-Image-Layered using official diffusers from coderef.

This bypasses our custom implementation to verify the model works.
"""

import sys
from pathlib import Path

# Add coderef diffusers to path BEFORE importing anything
coderef_diffusers = Path(__file__).parent.parent.parent / "coderef" / "diffusers" / "src"
sys.path.insert(0, str(coderef_diffusers))

import argparse
import logging
import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test Qwen-Image-Layered with official diffusers")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path.home() / "Storage" / "Qwen_Qwen-Image-Layered"),
        help="Path to Qwen-Image-Layered model",
    )
    parser.add_argument(
        "--input-image",
        type=str,
        default="experiments/inputs/test_scene.png",
        help="Input image to decompose",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A scene with objects",
        help="Scene description prompt",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of layers to decompose into (1-7)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=640,
        choices=[640, 1024],
        help="Output resolution",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of diffusion steps",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/qwen_official_test",
        help="Output directory for layers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    # Check model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log VRAM before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"VRAM before loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Import from official diffusers
    logger.info("Importing official diffusers QwenImageLayeredPipeline...")
    from diffusers import QwenImageLayeredPipeline

    # Load pipeline
    logger.info(f"Loading pipeline from {model_path}...")
    pipe = QwenImageLayeredPipeline.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
    )

    # Enable memory optimizations
    logger.info("Enabling sequential CPU offload...")
    pipe.enable_sequential_cpu_offload()

    # Log VRAM after loading
    if torch.cuda.is_available():
        logger.info(f"VRAM after loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Load input image if exists
    input_path = Path(args.input_image)
    if input_path.exists():
        logger.info(f"Loading input image: {input_path}")
        image = Image.open(input_path).convert("RGB").convert("RGBA")
        logger.info(f"Input image size: {image.size}, mode: {image.mode}")
    else:
        logger.warning(f"Input image not found at {input_path}, creating dummy")
        image = Image.new("RGBA", (args.resolution, args.resolution), color=(128, 128, 128, 255))

    # Set seed
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # Run decomposition
    logger.info(f"Decomposing image into {args.num_layers} layers...")
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Resolution: {args.resolution}x{args.resolution}")
    logger.info(f"Steps: {args.steps}")

    result = pipe(
        image=image,
        prompt=args.prompt,
        layers=args.num_layers,
        resolution=args.resolution,
        num_inference_steps=args.steps,
        generator=generator,
    )

    # Log VRAM peak
    if torch.cuda.is_available():
        logger.info(f"VRAM peak: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    # Save layers (result.images is a list of lists)
    layers = result.images[0] if isinstance(result.images[0], list) else result.images
    logger.info(f"Saving {len(layers)} layers to {output_dir}")
    for i, layer in enumerate(layers):
        layer_path = output_dir / f"layer_{i:02d}.png"
        layer.save(layer_path)
        logger.info(f"  Saved: {layer_path} ({layer.size}, mode={layer.mode})")

    # Save input for comparison
    input_copy = output_dir / "input.png"
    image.save(input_copy)

    logger.info("Decomposition complete!")
    logger.info(f"Results saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
