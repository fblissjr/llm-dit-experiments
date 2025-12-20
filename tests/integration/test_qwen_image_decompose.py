#!/usr/bin/env python
"""
Integration test for Qwen-Image-Layered decomposition.

Tests the full pipeline with a real image.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test Qwen-Image-Layered decomposition")
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
        "--cfg-scale",
        type=float,
        default=4.0,
        help="CFG guidance scale",
    )
    parser.add_argument(
        "--text-encoder-quantization",
        type=str,
        default="none",
        choices=["none", "4bit", "8bit"],
        help="Text encoder quantization (encoder on CPU doesn't need it)",
    )
    parser.add_argument(
        "--dit-quantization",
        type=str,
        default="4bit",
        choices=["none", "4bit", "8bit"],
        help="DiT quantization (4bit recommended for RTX 4090)",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Enable CPU offload to save VRAM",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for DiT",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/qwen_image_test",
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

    # Check input image exists
    input_image = Path(args.input_image)
    if not input_image.exists():
        logger.error(f"Input image not found at {input_image}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load input image
    logger.info(f"Loading input image: {input_image}")
    image = Image.open(input_image).convert("RGB")
    logger.info(f"Input image size: {image.size}")

    # Log VRAM before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"VRAM before loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Import and create pipeline
    logger.info("Loading Qwen-Image-Layered pipeline...")
    from llm_dit.pipelines.qwen_image import QwenImagePipeline

    pipeline = QwenImagePipeline.from_pretrained(
        model_path=str(model_path),
        device="cuda",
        text_encoder_device="cpu",  # Keep encoder on CPU to save VRAM
        vae_device="cpu",  # Keep VAE on CPU initially (offload moves it)
        torch_dtype=torch.bfloat16,
        text_encoder_quantization=args.text_encoder_quantization,
        dit_quantization=args.dit_quantization,
        compile_model=args.compile,
        cpu_offload=args.cpu_offload,
    )

    # Log VRAM after loading
    if torch.cuda.is_available():
        logger.info(f"VRAM after loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Run decomposition
    logger.info(f"Decomposing image into {args.num_layers} layers...")
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Resolution: {args.resolution}x{args.resolution}")
    logger.info(f"Steps: {args.steps}, CFG: {args.cfg_scale}")

    layers = pipeline.decompose(
        image=image,
        prompt=args.prompt,
        layer_num=args.num_layers,
        height=args.resolution,
        width=args.resolution,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
    )

    # Log VRAM during generation
    if torch.cuda.is_available():
        logger.info(f"VRAM peak: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    # Save layers
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
