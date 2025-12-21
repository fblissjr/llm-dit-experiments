#!/usr/bin/env python
"""
layer_remix.py - decompose images and remix layers from different sources

last updated: 2025-12-21

workflow:
1. decompose: split images into RGBA layers
2. remix: composite layers from different decompositions

examples:
    # decompose multiple images
    uv run experiments/scripts/layer_remix.py decompose \
        experiments/inputs/homer_art_deco.png \
        experiments/inputs/style_cyberpunk_city.png \
        -o results/remix_test

    # remix layers (layer format: source_name:layer_index)
    uv run experiments/scripts/layer_remix.py remix \
        --layers homer_art_deco:1 style_cyberpunk_city:2 \
        --base-dir results/remix_test \
        -o results/remix_test/remixed.png

    # interactive mode - list available layers
    uv run experiments/scripts/layer_remix.py list --base-dir results/remix_test
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Add coderef diffusers to path BEFORE importing anything
coderef_diffusers = Path(__file__).parent.parent.parent / "coderef" / "diffusers" / "src"
sys.path.insert(0, str(coderef_diffusers))

from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_pipeline(model_path: str):
    """Load QwenImageDiffusersPipeline with CPU offload."""
    import torch
    from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        vram_before = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"VRAM before loading: {vram_before:.2f} GB")

    logger.info(f"Loading pipeline from {model_path}...")
    start = time.time()
    pipe = QwenImageDiffusersPipeline.from_pretrained(
        model_path,
        cpu_offload=True,
        load_edit_model=False,
    )
    logger.info(f"Pipeline loaded in {time.time() - start:.1f}s")

    return pipe


def decompose_image(
    pipe,
    image_path: Path,
    output_dir: Path,
    prompt: Optional[str] = None,
    layer_num: int = 4,
    resolution: int = 640,
    steps: int = 50,
    cfg_scale: float = 4.0,
    seed: Optional[int] = None,
) -> list[Image.Image]:
    """Decompose a single image into layers."""
    import torch

    image = Image.open(image_path).convert("RGBA")
    image_name = image_path.stem

    # Auto-generate prompt if not provided
    if prompt is None:
        prompt = f"A scene from {image_name.replace('_', ' ')}"

    logger.info(f"Decomposing: {image_path.name}")
    logger.info(f"  Prompt: {prompt}")
    logger.info(f"  Layers: {layer_num}, Resolution: {resolution}, Steps: {steps}")

    start = time.time()
    layers = pipe.decompose(
        image=image,
        prompt=prompt,
        layer_num=layer_num,
        resolution=resolution,
        num_inference_steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
    )
    decompose_time = time.time() - start

    logger.info(f"  Completed in {decompose_time:.1f}s, generated {len(layers)} layers")

    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"  Peak VRAM: {peak_vram:.2f} GB")
        torch.cuda.reset_peak_memory_stats()

    # Save layers to subdirectory named after the image
    layer_dir = output_dir / image_name
    layer_dir.mkdir(parents=True, exist_ok=True)

    for i, layer in enumerate(layers):
        if i == 0:
            layer_path = layer_dir / "composite.png"
        else:
            layer_path = layer_dir / f"layer_{i:02d}.png"
        layer.save(layer_path)
        logger.info(f"  Saved: {layer_path.name} ({layer.size}, mode={layer.mode})")

    # Save metadata
    metadata_path = layer_dir / "metadata.txt"
    with open(metadata_path, "w") as f:
        f.write(f"source: {image_path}\n")
        f.write(f"prompt: {prompt}\n")
        f.write(f"layer_num: {layer_num}\n")
        f.write(f"resolution: {resolution}\n")
        f.write(f"steps: {steps}\n")
        f.write(f"cfg_scale: {cfg_scale}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"decompose_time: {decompose_time:.1f}s\n")
        f.write(f"num_layers: {len(layers)}\n")

    return layers


def composite_layers(layers: list[Image.Image], order: str = "back_to_front") -> Image.Image:
    """Composite RGBA layers together.

    Args:
        layers: List of RGBA images to composite
        order: 'back_to_front' (first layer is background) or 'front_to_back'

    Returns:
        Composited RGBA image
    """
    if not layers:
        raise ValueError("No layers to composite")

    # Ensure all layers are same size (use first layer size)
    target_size = layers[0].size

    if order == "front_to_back":
        layers = list(reversed(layers))

    # Start with transparent canvas
    result = Image.new("RGBA", target_size, (0, 0, 0, 0))

    for layer in layers:
        # Resize if needed
        if layer.size != target_size:
            layer = layer.resize(target_size, Image.Resampling.LANCZOS)
        # Ensure RGBA
        if layer.mode != "RGBA":
            layer = layer.convert("RGBA")
        # Alpha composite
        result = Image.alpha_composite(result, layer)

    return result


def cmd_decompose(args):
    """Handle decompose subcommand."""
    import torch

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline once
    pipe = load_pipeline(args.model_path)

    # Process each input image
    for image_path_str in args.images:
        image_path = Path(image_path_str)
        if not image_path.exists():
            logger.warning(f"Image not found, skipping: {image_path}")
            continue

        try:
            decompose_image(
                pipe=pipe,
                image_path=image_path,
                output_dir=output_dir,
                prompt=args.prompt,
                layer_num=args.layers,
                resolution=args.resolution,
                steps=args.steps,
                cfg_scale=args.cfg_scale,
                seed=args.seed,
            )
        except Exception as e:
            logger.error(f"Failed to decompose {image_path}: {e}")
            import traceback
            traceback.print_exc()

        # Clear VRAM between images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"All decompositions saved to: {output_dir}")


def cmd_list(args):
    """Handle list subcommand - show available layers."""
    base_dir = Path(args.base_dir)

    if not base_dir.exists():
        logger.error(f"Directory not found: {base_dir}")
        return 1

    print(f"\nAvailable layers in {base_dir}:\n")
    print("-" * 60)

    # Find all decomposed image directories
    for image_dir in sorted(base_dir.iterdir()):
        if not image_dir.is_dir():
            continue

        # Check if this looks like a decomposition directory
        composite_path = image_dir / "composite.png"
        if not composite_path.exists():
            continue

        image_name = image_dir.name
        print(f"\n{image_name}/")

        # List layers (skip layer 0/composite - it's often buggy from diffusers)
        layer_files = sorted(image_dir.glob("layer_*.png"))
        for layer_file in layer_files:
            layer_idx = int(layer_file.stem.split("_")[1])
            img = Image.open(layer_file)
            print(f"  {image_name}:{layer_idx}  ({img.size[0]}x{img.size[1]}, {img.mode})")

        # Note about composite
        print(f"  {image_name}:0  (composite - NOTE: often buggy, use manual remix)")

    print("\n" + "-" * 60)
    print("\nUsage: remix --layers <source1>:<layer> <source2>:<layer> ...")
    print("Example: remix --layers homer_art_deco:1 homer_art_deco:2 homer_art_deco:3")
    print("\nNOTE: Layer 0 (composite) from diffusers is often broken.")
    print("      Use remix with layers 1+ to manually composite.\n")

    return 0


def cmd_remix(args):
    """Handle remix subcommand - composite layers from different sources."""
    base_dir = Path(args.base_dir)
    output_path = Path(args.output)

    if not base_dir.exists():
        logger.error(f"Base directory not found: {base_dir}")
        return 1

    # Parse layer specifications
    layers_to_load = []
    for layer_spec in args.layers:
        parts = layer_spec.split(":")
        if len(parts) != 2:
            logger.error(f"Invalid layer spec '{layer_spec}'. Use format: source_name:layer_index")
            return 1

        source_name, layer_idx = parts[0], int(parts[1])
        layers_to_load.append((source_name, layer_idx))

    # Load layers
    loaded_layers = []
    for source_name, layer_idx in layers_to_load:
        source_dir = base_dir / source_name

        if not source_dir.exists():
            logger.error(f"Source directory not found: {source_dir}")
            return 1

        if layer_idx == 0:
            layer_path = source_dir / "composite.png"
        else:
            layer_path = source_dir / f"layer_{layer_idx:02d}.png"

        if not layer_path.exists():
            logger.error(f"Layer not found: {layer_path}")
            return 1

        layer = Image.open(layer_path).convert("RGBA")
        loaded_layers.append(layer)
        logger.info(f"Loaded: {source_name}:{layer_idx} ({layer.size})")

    # Composite layers
    logger.info(f"Compositing {len(loaded_layers)} layers...")
    result = composite_layers(loaded_layers, order=args.order)

    # Save result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    logger.info(f"Saved remixed image: {output_path}")

    return 0


def cmd_batch(args):
    """Handle batch subcommand - decompose all images in a directory."""
    import torch

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    image_paths = [
        p for p in input_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ]

    if not image_paths:
        logger.error(f"No images found in {input_dir}")
        return 1

    logger.info(f"Found {len(image_paths)} images to decompose")

    # Load pipeline once
    pipe = load_pipeline(args.model_path)

    # Process each image
    for image_path in sorted(image_paths):
        try:
            decompose_image(
                pipe=pipe,
                image_path=image_path,
                output_dir=output_dir,
                prompt=None,  # Auto-generate from filename
                layer_num=args.layers,
                resolution=args.resolution,
                steps=args.steps,
                cfg_scale=args.cfg_scale,
                seed=args.seed,
            )
        except Exception as e:
            logger.error(f"Failed to decompose {image_path}: {e}")
            import traceback
            traceback.print_exc()

        # Clear VRAM between images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"All decompositions saved to: {output_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Decompose images and remix layers from different sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # decompose subcommand
    decompose_parser = subparsers.add_parser(
        "decompose",
        help="Decompose images into RGBA layers",
    )
    decompose_parser.add_argument(
        "images",
        nargs="+",
        help="Input image paths to decompose",
    )
    decompose_parser.add_argument(
        "-o", "--output",
        default="results/layer_remix",
        help="Output directory (default: results/layer_remix)",
    )
    decompose_parser.add_argument(
        "--model-path",
        default=str(Path.home() / "Storage" / "Qwen_Qwen-Image-Layered"),
        help="Path to Qwen-Image-Layered model",
    )
    decompose_parser.add_argument(
        "--prompt",
        help="Prompt for decomposition (default: auto-generate from filename)",
    )
    decompose_parser.add_argument(
        "--layers",
        type=int,
        default=4,
        help="Number of layers to decompose into (default: 4)",
    )
    decompose_parser.add_argument(
        "--resolution",
        type=int,
        default=640,
        choices=[640, 1024],
        help="Resolution for decomposition (default: 640)",
    )
    decompose_parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of diffusion steps (default: 50)",
    )
    decompose_parser.add_argument(
        "--cfg-scale",
        type=float,
        default=4.0,
        help="CFG scale (default: 4.0)",
    )
    decompose_parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )

    # batch subcommand
    batch_parser = subparsers.add_parser(
        "batch",
        help="Decompose all images in a directory",
    )
    batch_parser.add_argument(
        "input_dir",
        help="Directory containing images to decompose",
    )
    batch_parser.add_argument(
        "-o", "--output",
        default="results/layer_remix",
        help="Output directory (default: results/layer_remix)",
    )
    batch_parser.add_argument(
        "--model-path",
        default=str(Path.home() / "Storage" / "Qwen_Qwen-Image-Layered"),
        help="Path to Qwen-Image-Layered model",
    )
    batch_parser.add_argument(
        "--layers",
        type=int,
        default=4,
        help="Number of layers to decompose into (default: 4)",
    )
    batch_parser.add_argument(
        "--resolution",
        type=int,
        default=640,
        choices=[640, 1024],
        help="Resolution for decomposition (default: 640)",
    )
    batch_parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of diffusion steps (default: 50)",
    )
    batch_parser.add_argument(
        "--cfg-scale",
        type=float,
        default=4.0,
        help="CFG scale (default: 4.0)",
    )
    batch_parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )

    # list subcommand
    list_parser = subparsers.add_parser(
        "list",
        help="List available layers from decompositions",
    )
    list_parser.add_argument(
        "--base-dir",
        default="results/layer_remix",
        help="Base directory containing decompositions (default: results/layer_remix)",
    )

    # remix subcommand
    remix_parser = subparsers.add_parser(
        "remix",
        help="Composite layers from different decompositions",
    )
    remix_parser.add_argument(
        "--layers",
        nargs="+",
        required=True,
        help="Layer specs in format source_name:layer_index (e.g., homer:1 cyberpunk:2)",
    )
    remix_parser.add_argument(
        "--base-dir",
        default="results/layer_remix",
        help="Base directory containing decompositions (default: results/layer_remix)",
    )
    remix_parser.add_argument(
        "-o", "--output",
        default="results/layer_remix/remixed.png",
        help="Output path for remixed image",
    )
    remix_parser.add_argument(
        "--order",
        choices=["back_to_front", "front_to_back"],
        default="back_to_front",
        help="Layer compositing order (default: back_to_front)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "decompose":
        return cmd_decompose(args)
    elif args.command == "batch":
        return cmd_batch(args)
    elif args.command == "list":
        return cmd_list(args)
    elif args.command == "remix":
        return cmd_remix(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
