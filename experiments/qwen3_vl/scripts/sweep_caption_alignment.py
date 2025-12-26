#!/usr/bin/env python3
"""
Sweep hidden layers and alpha blends for VL caption alignment experiment.

Tests VL+Caption vs VL Generic across:
- Hidden layers: -2, -3, -4, -5, -6
- Alphas: 0.0, 0.25, 0.5, 0.75, 1.0
"""

import argparse
import json
import logging
import sys
import torch
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm_dit.vl.qwen3_vl import VLEmbeddingExtractor
from llm_dit.vl.blending import blend_interpolate
from llm_dit.cli import load_runtime_config
from llm_dit.startup import PipelineLoader
from llm_dit.constants import MAX_TEXT_SEQ_LEN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ConfigArgs:
    """Mock args object for load_runtime_config."""
    def __init__(self):
        self.config = "config.toml"
        self.profile = "default"
        # All other args default to None (will use config file values)
        for attr in [
            "model_path", "text_encoder_device", "dit_device", "vae_device",
            "api_url", "api_model", "use_api_encoder", "cpu_offload", "flash_attn",
            "compile", "debug", "attention_backend", "use_custom_scheduler",
            "tiled_vae", "tile_size", "tile_overlap", "embedding_cache", "cache_size",
            "long_prompt_mode", "hidden_layer", "width", "height", "steps",
            "guidance_scale", "shift", "seed", "system_prompt", "thinking_content",
            "assistant_content", "force_think_block", "template", "lora",
            "templates_dir", "rewriter_use_api", "rewriter_api_url", "rewriter_api_model",
            "rewriter_temperature", "rewriter_top_p", "rewriter_min_p",
            "rewriter_max_tokens", "rewriter_timeout", "vl_model_path", "vl_device",
            "vl_alpha", "vl_hidden_layer", "vl_auto_unload", "vl_blend_mode",
            "rewriter_no_vl", "rewriter_preload_vl", "rewriter_vl_api_model",
            "vl_outlier_masking", "vl_outlier_threshold"
        ]:
            setattr(self, attr, None)


def generate_caption(vl_extractor, image, min_tokens=512, max_tokens=2500):
    """Generate detailed caption using VL model."""
    caption_system_prompt = f"""You are an expert image captioner. Generate an EXTREMELY detailed description of this image.

CRITICAL REQUIREMENT: Your description MUST be at least {min_tokens} tokens long.

Cover ALL of the following in exhaustive detail:
- Every person/character: exact pose, clothing, colors, patterns, accessories, hair, expression
- Background: every object, furniture, architecture, natural elements
- Lighting: direction, intensity, shadows, highlights, time of day
- Colors: specific names (cerulean, burgundy, etc.), relationships, saturation
- Artistic style: medium, technique, quality indicators
- Fine details: textures, small objects, text, imperfections

Write in flowing, descriptive prose. Be exhaustive - more detail is always better."""

    logger.info(f"Generating caption (min {min_tokens} tokens)...")

    raw_caption = vl_extractor.generate(
        image=image,
        prompt="Describe this image in extreme detail.",
        system_prompt=caption_system_prompt,
        max_new_tokens=max_tokens,
        temperature=0.6,
        top_p=0.95,
    )

    # Parse thinking block if present (generate returns a string)
    if "<think>" in raw_caption and "</think>" in raw_caption:
        think_start = raw_caption.find("<think>") + len("<think>")
        think_end = raw_caption.find("</think>")
        thinking = raw_caption[think_start:think_end].strip()
        caption = raw_caption[think_end + len("</think>"):].strip()
    else:
        thinking = ""
        caption = raw_caption.strip()

    # Count tokens
    caption_tokens = len(vl_extractor.processor.tokenizer.encode(caption))

    return caption, thinking, caption_tokens


def create_grid(images, labels, title, cell_size=512):
    """Create a labeled grid of images."""
    n_images = len(images)
    cols = min(n_images, 5)
    rows = (n_images + cols - 1) // cols

    # Calculate dimensions
    label_height = 30
    title_height = 40
    padding = 10

    grid_width = cols * cell_size + (cols + 1) * padding
    grid_height = title_height + rows * (cell_size + label_height) + (rows + 1) * padding

    grid = Image.new("RGB", (grid_width, grid_height), "white")
    draw = ImageDraw.Draw(grid)

    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()
        title_font = font

    # Draw title
    draw.text((grid_width // 2, 10), title, fill="black", font=title_font, anchor="mt")

    # Draw images and labels
    for i, (img, label) in enumerate(zip(images, labels)):
        row = i // cols
        col = i % cols

        x = padding + col * (cell_size + padding)
        y = title_height + padding + row * (cell_size + label_height + padding)

        # Resize image
        img_resized = img.resize((cell_size, cell_size), Image.Resampling.LANCZOS)
        grid.paste(img_resized, (x, y))

        # Draw label
        label_y = y + cell_size + 5
        draw.text((x + cell_size // 2, label_y), label, fill="black", font=font, anchor="mt")

    return grid


def run_sweep(args):
    """Run the parameter sweep."""

    # Load input image
    input_image = Image.open(args.input).convert("RGB")
    logger.info(f"Input image: {args.input} ({input_image.size[0]}x{input_image.size[1]})")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize VL extractor
    vl_extractor = VLEmbeddingExtractor.from_pretrained(args.vl_model_path, device="cuda")
    logger.info(f"VL model loaded: {args.vl_model_path}")

    # Generate caption once
    caption, thinking, caption_tokens = generate_caption(
        vl_extractor, input_image,
        min_tokens=args.caption_tokens,
        max_tokens=args.max_tokens
    )
    logger.info(f"Caption generated: {caption_tokens} tokens")

    # Save caption
    (output_dir / "caption.txt").write_text(caption)
    if thinking:
        (output_dir / "thinking.txt").write_text(thinking)

    # Extract VL embeddings for all hidden layers
    hidden_layers = args.hidden_layers
    alphas = args.alphas

    vl_aligned_cache = {}  # layer -> embeddings
    vl_generic_cache = {}  # layer -> embeddings

    logger.info(f"Extracting VL embeddings for layers {hidden_layers}...")

    for layer in hidden_layers:
        logger.info(f"  Layer {layer}...")

        # Aligned (caption + image)
        vl_aligned = vl_extractor.extract(
            image=input_image,
            text=caption,
            hidden_layer=layer,
            text_tokens_only=False,
            scale_to_text=True,
        )
        vl_aligned_cache[layer] = vl_aligned.embeddings

        # Generic (short prompt + image)
        vl_generic = vl_extractor.extract(
            image=input_image,
            text="Describe this image",
            hidden_layer=layer,
            text_tokens_only=False,
            scale_to_text=True,
        )
        vl_generic_cache[layer] = vl_generic.embeddings

    logger.info(f"  Aligned embeddings shape: {vl_aligned_cache[hidden_layers[0]].shape}")
    logger.info(f"  Generic embeddings shape: {vl_generic_cache[hidden_layers[0]].shape}")

    # Unload VL model
    vl_extractor.unload()
    logger.info("VL model unloaded")

    # Load Z-Image pipeline
    mock_args = ConfigArgs()
    runtime_config = load_runtime_config(mock_args)
    loader = PipelineLoader(runtime_config)
    pipeline_result = loader.load_pipeline()
    pipeline = pipeline_result.pipeline
    text_encoder = pipeline_result.encoder

    # Encode caption with text encoder
    logger.info("Encoding caption with text encoder...")
    text_result = text_encoder.encode(caption)
    text_embeddings = text_result.embeddings[0]  # Remove batch dim
    logger.info(f"Text embeddings shape: {text_embeddings.shape}")

    # Generation settings
    seed = args.seed
    steps = args.steps

    # Results storage
    results = {
        "input_image": str(args.input),
        "timestamp": timestamp,
        "caption_tokens": caption_tokens,
        "hidden_layers": hidden_layers,
        "alphas": alphas,
        "seed": seed,
        "steps": steps,
        "generations": []
    }

    # Generate images for each combination
    all_images = []
    all_labels = []

    # Also create per-layer grids
    layer_grids = {}

    for layer in hidden_layers:
        layer_images = []
        layer_labels = []

        for alpha in alphas:
            for method in ["aligned", "generic"]:
                logger.info(f"Generating: layer={layer}, alpha={alpha}, method={method}")

                # Get VL embeddings
                if method == "aligned":
                    vl_emb = vl_aligned_cache[layer]
                else:
                    vl_emb = vl_generic_cache[layer]

                # Blend embeddings
                if alpha == 0.0:
                    # Pure text
                    blended = text_embeddings.clone()
                elif alpha == 1.0:
                    # Pure VL - need to match text length first
                    blended = vl_emb.clone()
                else:
                    # Blend
                    blended = blend_interpolate(vl_emb, text_embeddings, alpha)

                # Interpolate to max tokens if needed
                if blended.shape[0] > MAX_TEXT_SEQ_LEN:
                    blended = torch.nn.functional.interpolate(
                        blended.T.unsqueeze(0),
                        size=MAX_TEXT_SEQ_LEN,
                        mode="linear",
                        align_corners=False
                    ).squeeze(0).T

                # Generate image
                # Note: Pipeline expects 2D embeddings (seq_len, hidden_dim) without batch dimension
                generator = torch.Generator("cpu").manual_seed(seed)

                image = pipeline(
                    prompt_embeds=blended.to(pipeline.device, pipeline.dtype),
                    width=1024,
                    height=1024,
                    num_inference_steps=steps,
                    generator=generator,
                )

                # Save individual image
                method_label = "caption" if method == "aligned" else "generic"
                filename = f"layer{layer}_alpha{alpha}_{method_label}.png"
                image.save(output_dir / filename)

                # Add to collections
                label = f"L{layer} a={alpha}"
                if method == "aligned":
                    label += " (caption)"
                else:
                    label += " (generic)"

                all_images.append(image)
                all_labels.append(label)

                layer_images.append(image)
                layer_labels.append(f"a={alpha} {'caption' if method == 'aligned' else 'generic'}")

                results["generations"].append({
                    "layer": layer,
                    "alpha": alpha,
                    "method": method,
                    "filename": filename
                })

        # Create per-layer grid
        layer_grid = create_grid(
            layer_images, layer_labels,
            f"Hidden Layer {layer}",
            cell_size=384
        )
        layer_grid.save(output_dir / f"grid_layer{layer}.png")
        layer_grids[layer] = layer_grid

    # Create alpha comparison grids (one per alpha, comparing layers)
    for alpha in alphas:
        alpha_images = []
        alpha_labels = []

        for layer in hidden_layers:
            for method in ["aligned", "generic"]:
                method_label = "caption" if method == "aligned" else "generic"
                filename = f"layer{layer}_alpha{alpha}_{method_label}.png"
                img = Image.open(output_dir / filename)
                alpha_images.append(img)
                alpha_labels.append(f"L{layer} {'cap' if method == 'aligned' else 'gen'}")

        alpha_grid = create_grid(
            alpha_images, alpha_labels,
            f"Alpha={alpha} - Layer Comparison",
            cell_size=384
        )
        alpha_grid.save(output_dir / f"grid_alpha{alpha}.png")

    # Create method comparison grids
    for method in ["aligned", "generic"]:
        method_label = "caption" if method == "aligned" else "generic"
        method_images = []
        method_labels = []

        for layer in hidden_layers:
            for alpha in alphas:
                filename = f"layer{layer}_alpha{alpha}_{method_label}.png"
                img = Image.open(output_dir / filename)
                method_images.append(img)
                method_labels.append(f"L{layer} a={alpha}")

        method_grid = create_grid(
            method_images, method_labels,
            f"Method: {method_label.upper()} - All Combinations",
            cell_size=256
        )
        method_grid.save(output_dir / f"grid_method_{method_label}.png")

    # Save input image for reference
    input_image.save(output_dir / "input.png")

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Sweep complete!")
    logger.info(f"Results: {output_dir}")
    logger.info(f"  - {len(results['generations'])} images generated")
    logger.info(f"  - Per-layer grids: grid_layer*.png")
    logger.info(f"  - Per-alpha grids: grid_alpha*.png")
    logger.info(f"  - Per-method grids: grid_method_*.png")
    logger.info(f"{'='*60}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Sweep hidden layers and alphas for VL caption alignment")
    parser.add_argument("-i", "--input", required=True, help="Input image path")
    parser.add_argument("-o", "--output", default="experiments/results/caption_alignment_sweep",
                        help="Output directory")
    parser.add_argument("--vl-model-path", required=True,
                        help="Path to Qwen3-VL model")
    parser.add_argument("--hidden-layers", type=str, default="-2,-3,-4,-5,-6",
                        help="Hidden layers to test (comma-separated, e.g., -2,-3,-4,-5,-6)")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0],
                        help="Alpha blend values to test")
    parser.add_argument("--caption-tokens", type=int, default=512,
                        help="Minimum caption tokens")
    parser.add_argument("--max-tokens", type=int, default=2500,
                        help="Max tokens for caption generation")
    parser.add_argument("--steps", type=int, default=9,
                        help="Inference steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Parse hidden layers from comma-separated string
    args.hidden_layers = [int(x.strip()) for x in args.hidden_layers.split(",")]

    output_dir = run_sweep(args)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
