#!/usr/bin/env python
"""
decompose_recompose.py - decompose with qwen-image-layered, recompose with z-image

last updated: 2025-12-21

workflow (approach 3 from research):
1. decompose image into semantic layers using qwen-image-layered
2. for each layer:
   a. extract vl embeddings (style/content info) using qwen3-vl
   b. blend with text prompt embeddings
   c. run z-image img2img to regenerate the layer
3. composite regenerated layers

examples:
    # full workflow: decompose then recompose with new style
    uv run experiments/scripts/decompose_recompose.py \
        --input experiments/inputs/homer_art_deco.png \
        --style-prompt "cyberpunk neon style, glowing edges" \
        -o results/recompose_test

    # recompose only (from existing decomposition)
    uv run experiments/scripts/decompose_recompose.py \
        --decomposed-dir results/layer_remix_v2/homer_art_deco \
        --style-prompt "watercolor painting style" \
        -o results/recompose_watercolor
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, List

# Add coderef diffusers to path BEFORE importing anything
coderef_diffusers = Path(__file__).parent.parent.parent / "coderef" / "diffusers" / "src"
sys.path.insert(0, str(coderef_diffusers))

# Add experiments to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from PIL import Image

from experiments.utils import save_image_grid, save_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_qwen_image_pipeline(model_path: str):
    """Load Qwen-Image-Layered pipeline for decomposition."""
    from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

    logger.info(f"Loading Qwen-Image-Layered from {model_path}...")
    pipe = QwenImageDiffusersPipeline.from_pretrained(
        model_path,
        cpu_offload=True,
        load_edit_model=False,
    )
    return pipe


def load_z_image_pipeline(model_path: str, device: str = "cuda"):
    """Load Z-Image pipeline for recomposition."""
    from llm_dit.pipelines.z_image import ZImagePipeline

    logger.info(f"Loading Z-Image from {model_path}...")
    pipe = ZImagePipeline.from_pretrained(
        model_path,
        text_encoder_device="cpu",
        dit_device=device,
        vae_device=device,
    )
    return pipe


def load_vl_extractor(model_path: str, device: str = "cpu"):
    """Load Qwen3-VL for embedding extraction."""
    from llm_dit.vl import VLEmbeddingExtractor

    logger.info(f"Loading Qwen3-VL from {model_path}...")
    extractor = VLEmbeddingExtractor.from_pretrained(
        model_path,
        device=device,
    )
    return extractor


def decompose_image(
    pipe,
    image_path: Path,
    output_dir: Path,
    prompt: str,
    layer_num: int = 4,
    resolution: int = 640,
    steps: int = 50,
    seed: Optional[int] = None,
) -> List[Image.Image]:
    """Decompose image into layers."""
    image = Image.open(image_path).convert("RGBA")

    logger.info(f"Decomposing: {image_path.name}")
    logger.info(f"  Prompt: {prompt}")

    start = time.time()
    layers = pipe.decompose(
        image=image,
        prompt=prompt,
        layer_num=layer_num,
        resolution=resolution,
        num_inference_steps=steps,
        cfg_scale=4.0,
        seed=seed,
    )
    logger.info(f"  Completed in {time.time() - start:.1f}s")

    # Save layers (skip layer 0 - composite is often buggy)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, layer in enumerate(layers):
        if i == 0:
            path = output_dir / "composite_original.png"
        else:
            path = output_dir / f"layer_{i:02d}_original.png"
        layer.save(path)
        logger.info(f"  Saved: {path.name}")

    # Return only actual layers (skip buggy composite)
    return layers[1:] if len(layers) > 1 else layers


def load_decomposed_layers(decomposed_dir: Path) -> List[Image.Image]:
    """Load previously decomposed layers from directory."""
    layer_files = sorted(decomposed_dir.glob("layer_*.png"))
    if not layer_files:
        raise ValueError(f"No layer files found in {decomposed_dir}")

    layers = []
    for layer_file in layer_files:
        layer = Image.open(layer_file).convert("RGBA")
        layers.append(layer)
        logger.info(f"Loaded: {layer_file.name} ({layer.size})")

    return layers


def recompose_layer(
    z_image_pipe,
    vl_extractor,
    layer: Image.Image,
    layer_prompt: str,
    style_prompt: str,
    vl_alpha: float = 0.3,
    img2img_strength: float = 0.8,
    steps: int = 9,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Recompose a single layer using VL + img2img (Approach 3).

    Args:
        z_image_pipe: Z-Image pipeline
        vl_extractor: Qwen3-VL embedding extractor
        layer: RGBA layer to recompose
        layer_prompt: Description of layer content
        style_prompt: Style to apply
        vl_alpha: VL embedding influence (0.0-1.0)
        img2img_strength: How much to transform (0.0=keep, 1.0=full regen)
        steps: Z-Image inference steps
        seed: Random seed

    Returns:
        Recomposed RGBA layer
    """
    from llm_dit.vl import blend_interpolate

    # Store original alpha channel
    _, _, _, alpha_channel = layer.split()

    # Convert to RGB for processing
    rgb_layer = layer.convert("RGB")

    # Extract VL embeddings from layer
    logger.info(f"  Extracting VL embeddings (alpha={vl_alpha})...")
    vl_result = vl_extractor.extract(
        rgb_layer,
        text=layer_prompt,
        hidden_layer=-6,  # -6 is cleaner than -2 for VL
        text_tokens_only=False,  # Must include image tokens
        scale_to_text=True,
    )

    # Encode style prompt with Z-Image's encoder
    full_prompt = f"{layer_prompt}, {style_prompt}"
    logger.info(f"  Encoding prompt: {full_prompt[:60]}...")

    # Get text embeddings from Z-Image pipeline (returns tensor directly)
    text_embeds = z_image_pipe.encode_prompt(full_prompt)

    # Blend VL and text embeddings
    logger.info(f"  Blending embeddings...")

    # Handle shape mismatch - VL may have different sequence length
    vl_embeds = vl_result.embeddings
    if vl_embeds.shape[1] != text_embeds.shape[1]:
        # Interpolate VL to match text length
        import torch.nn.functional as F
        vl_embeds = F.interpolate(
            vl_embeds.permute(0, 2, 1),
            size=text_embeds.shape[1],
            mode='linear',
            align_corners=False,
        ).permute(0, 2, 1)

    blended = blend_interpolate(vl_embeds, text_embeds, alpha=vl_alpha)

    # Run img2img with blended embeddings
    logger.info(f"  Running Z-Image img2img (strength={img2img_strength})...")

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    result = z_image_pipe.img2img(
        prompt_embeds=blended,
        image=rgb_layer,
        strength=img2img_strength,
        num_inference_steps=steps,
        generator=generator,
    )

    # Get result and restore alpha (img2img returns Image directly)
    new_rgb = result
    r, g, b = new_rgb.split()
    new_rgba = Image.merge("RGBA", (r, g, b, alpha_channel))

    return new_rgba


def composite_layers(layers: List[Image.Image]) -> Image.Image:
    """Composite RGBA layers (back to front)."""
    if not layers:
        raise ValueError("No layers to composite")

    target_size = layers[0].size
    result = Image.new("RGBA", target_size, (0, 0, 0, 0))

    for layer in layers:
        if layer.size != target_size:
            layer = layer.resize(target_size, Image.Resampling.LANCZOS)
        if layer.mode != "RGBA":
            layer = layer.convert("RGBA")
        result = Image.alpha_composite(result, layer)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Decompose with Qwen-Image-Layered, recompose with Z-Image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options (choose one)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", "-i",
        type=Path,
        help="Input image to decompose and recompose",
    )
    input_group.add_argument(
        "--decomposed-dir",
        type=Path,
        help="Directory with already-decomposed layers",
    )

    # Model paths
    parser.add_argument(
        "--qwen-image-path",
        default=str(Path.home() / "Storage" / "Qwen_Qwen-Image-Layered"),
        help="Path to Qwen-Image-Layered model",
    )
    parser.add_argument(
        "--z-image-path",
        default=str(Path.home() / "Storage" / "Z-Image-Turbo"),
        help="Path to Z-Image model",
    )
    parser.add_argument(
        "--vl-path",
        default=str(Path.home() / "Storage" / "Qwen3-VL-4B-Instruct"),
        help="Path to Qwen3-VL model",
    )

    # Decomposition options
    parser.add_argument(
        "--decompose-prompt",
        help="Prompt for decomposition (describes input image)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=4,
        help="Number of layers (default: 4)",
    )
    parser.add_argument(
        "--decompose-steps",
        type=int,
        default=50,
        help="Decomposition steps (default: 50)",
    )

    # Recomposition options
    parser.add_argument(
        "--style-prompt",
        default="detailed, high quality",
        help="Style to apply during recomposition",
    )
    parser.add_argument(
        "--layer-prompts",
        nargs="+",
        help="Per-layer prompts (one per layer, or single prompt for all)",
    )
    parser.add_argument(
        "--vl-alpha",
        type=float,
        default=0.3,
        help="VL embedding influence 0.0-1.0 (default: 0.3)",
    )
    parser.add_argument(
        "--img2img-strength",
        type=float,
        default=0.8,
        help="img2img strength 0.0-1.0 (default: 0.8)",
    )
    parser.add_argument(
        "--recompose-steps",
        type=int,
        default=9,
        help="Z-Image steps (default: 9)",
    )

    # Output
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("experiments/results/recompose"),
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Step 1: Get decomposed layers
    if args.input:
        # Need to decompose first
        if not args.decompose_prompt:
            logger.error("--decompose-prompt required when using --input")
            return 1

        qwen_image_pipe = load_qwen_image_pipeline(args.qwen_image_path)
        layers = decompose_image(
            qwen_image_pipe,
            args.input,
            args.output / "decomposed",
            prompt=args.decompose_prompt,
            layer_num=args.layers,
            steps=args.decompose_steps,
            seed=args.seed,
        )
        # Free VRAM
        del qwen_image_pipe
        torch.cuda.empty_cache()
    else:
        # Load existing decomposition
        layers = load_decomposed_layers(args.decomposed_dir)

    logger.info(f"Working with {len(layers)} layers")

    # Step 2: Load Z-Image and VL models
    z_image_pipe = load_z_image_pipeline(args.z_image_path)
    vl_extractor = load_vl_extractor(args.vl_path)

    # Step 3: Recompose each layer
    recomposed_layers = []

    # Setup layer prompts
    if args.layer_prompts:
        if len(args.layer_prompts) == 1:
            layer_prompts = [args.layer_prompts[0]] * len(layers)
        elif len(args.layer_prompts) >= len(layers):
            layer_prompts = args.layer_prompts[:len(layers)]
        else:
            logger.warning("Not enough layer prompts, reusing last one")
            layer_prompts = args.layer_prompts + [args.layer_prompts[-1]] * (len(layers) - len(args.layer_prompts))
    else:
        layer_prompts = [f"Layer {i+1} content" for i in range(len(layers))]

    for i, (layer, layer_prompt) in enumerate(zip(layers, layer_prompts)):
        logger.info(f"\nRecomposing layer {i+1}/{len(layers)}...")
        logger.info(f"  Layer prompt: {layer_prompt}")
        logger.info(f"  Style prompt: {args.style_prompt}")

        try:
            new_layer = recompose_layer(
                z_image_pipe,
                vl_extractor,
                layer,
                layer_prompt=layer_prompt,
                style_prompt=args.style_prompt,
                vl_alpha=args.vl_alpha,
                img2img_strength=args.img2img_strength,
                steps=args.recompose_steps,
                seed=args.seed + i if args.seed else None,
            )

            # Save individual recomposed layer
            layer_path = args.output / f"layer_{i+1:02d}_recomposed.png"
            new_layer.save(layer_path)
            logger.info(f"  Saved: {layer_path.name}")

            recomposed_layers.append(new_layer)

        except Exception as e:
            logger.error(f"  Failed to recompose layer {i+1}: {e}")
            import traceback
            traceback.print_exc()
            # Keep original layer on failure
            recomposed_layers.append(layer)

    # Step 4: Composite final result
    logger.info("\nCompositing final result...")
    final = composite_layers(recomposed_layers)
    final_path = args.output / "final_recomposed.png"
    final.save(final_path)
    logger.info(f"Saved: {final_path}")

    # Also save original composite for comparison
    original_composite = composite_layers(layers)
    original_path = args.output / "original_composite.png"
    original_composite.save(original_path)
    logger.info(f"Saved: {original_path}")

    # Create comparison grid
    logger.info("Creating comparison grid...")
    grid_path = save_image_grid(
        [original_composite, final],
        args.output / "comparison.png",
        cols=2,
        labels=["Original Composite", "Recomposed"],
        cell_size=512,
    )
    logger.info(f"Saved comparison grid: {grid_path}")

    # Save metadata
    save_metadata(
        args.output / "metadata.json",
        input_image=str(args.input) if args.input else None,
        decomposed_dir=str(args.decomposed_dir) if args.decomposed_dir else None,
        decompose_prompt=args.decompose_prompt,
        style_prompt=args.style_prompt,
        layer_prompts=args.layer_prompts,
        num_layers=len(layers),
        vl_alpha=args.vl_alpha,
        img2img_strength=args.img2img_strength,
        decompose_steps=args.decompose_steps,
        recompose_steps=args.recompose_steps,
        seed=args.seed,
    )

    logger.info(f"\nAll outputs saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
