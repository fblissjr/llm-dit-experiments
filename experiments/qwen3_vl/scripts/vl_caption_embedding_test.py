#!/usr/bin/env python3
"""
VL Caption + Embedding Alignment Test

This experiment tests whether:
1. Generating a detailed caption with Qwen3-VL-Thinking
2. Then extracting VL embeddings using that caption + original image

...produces better image reconstruction than direct VL embedding injection.

The idea is that aligning the text with the image content should produce
richer, more coherent embeddings for Z-Image generation.

Usage:
    uv run experiments/qwen3_vl/scripts/vl_caption_embedding_test.py \
        -i experiments/inputs/test_photo.jpg \
        -o experiments/results/caption_test \
        --caption-tokens 1024 \
        --steps 9

The script will:
1. Generate a detailed caption of the input image (~1024 tokens)
2. Extract VL embeddings using that caption + image
3. Generate an image with those embeddings
4. Also generate with a short generic prompt for comparison
"""

import argparse
import gc
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from llm_dit.vl import VLEmbeddingExtractor
from llm_dit.vl.blending import blend_interpolate

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def parse_thinking_content(text: str) -> tuple[str, str | None]:
    """
    Parse thinking content from VL-Thinking model output.

    Returns:
        Tuple of (caption_without_thinking, thinking_content_or_none)
    """
    # Match <think>...</think> block (may have newlines)
    match = re.search(r"<think>\s*(.*?)\s*</think>\s*", text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        caption = text[match.end():].strip()
        return caption, thinking
    return text.strip(), None


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def main():
    parser = argparse.ArgumentParser(
        description="Test VL caption + embedding alignment"
    )
    parser.add_argument(
        "-i", "--image", required=True,
        help="Input image path"
    )
    parser.add_argument(
        "-o", "--output-dir", default="experiments/results/caption_embedding_test",
        help="Output directory"
    )
    parser.add_argument(
        "--caption-tokens", type=int, default=1024,
        help="Target caption length in tokens (actual output will be this + thinking overhead)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2500,
        help="Max tokens for generation (includes thinking block). Should be caption-tokens + ~1000 for thinking + buffer"
    )
    parser.add_argument(
        "--steps", type=int, default=9,
        help="Inference steps"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--vl-model-path", type=str, default=None,
        help="Path to Qwen3-VL model (auto-detects Thinking variant)"
    )
    parser.add_argument(
        "--z-image-path", type=str, default=None,
        help="Path to Z-Image model"
    )
    parser.add_argument(
        "--vl-hidden-layer", type=int, default=-6,
        help="VL hidden layer for extraction"
    )
    parser.add_argument(
        "--vl-alpha", type=float, default=0.5,
        help="VL alpha for blending"
    )
    parser.add_argument(
        "--config", type=str, default="config.toml",
        help="Config file for Z-Image settings"
    )
    parser.add_argument(
        "--profile", type=str, default="default",
        help="Config profile to use"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load input image
    input_image = Image.open(args.image).convert("RGB")
    input_image.save(output_dir / "input.png")
    logger.info(f"Input image: {args.image} ({input_image.size[0]}x{input_image.size[1]})")

    # Find VL model path
    vl_model_path = args.vl_model_path
    if not vl_model_path:
        vl_model_path, variant = VLEmbeddingExtractor.find_model_path(prefer_variant="thinking")
        if not vl_model_path:
            raise RuntimeError("Could not find Qwen3-VL model. Specify --vl-model-path")
        logger.info(f"Found VL model: {vl_model_path} (variant: {variant})")

    # Load VL extractor
    logger.info("Loading Qwen3-VL for caption generation and embedding extraction...")
    vl_extractor = VLEmbeddingExtractor.from_pretrained(
        vl_model_path,
        device="cuda",  # Need CUDA for generation
        torch_dtype=torch.bfloat16,
    )

    logger.info(f"VL model variant: {vl_extractor.model_variant}")
    if vl_extractor.model_variant != "thinking":
        logger.warning("Using non-Thinking model variant. For best results, use Qwen3-VL-4B-Thinking")

    # System prompt for caption generation - explicitly request very long output
    caption_system_prompt = f"""You are an expert image captioner. Generate an EXTREMELY detailed description of this image.

CRITICAL REQUIREMENT: Your description MUST be at least {args.caption_tokens} tokens long. This is a hard minimum.

To achieve this length, describe in exhaustive detail:

1. PRIMARY SUBJECTS (200+ tokens):
   - Every person/character: exact pose, body position, facial expression, gaze direction
   - Clothing: every garment, colors, patterns, textures, fit, style era
   - Accessories: jewelry, bags, hats, glasses, watches, etc.
   - Hair: style, color, length, texture, any styling products visible

2. BACKGROUND & SETTING (200+ tokens):
   - Location type and specific features
   - Every visible object, furniture, decor
   - Architecture: walls, floors, ceilings, windows, doors
   - Natural elements: plants, sky, clouds, water, terrain
   - Distance and spatial relationships between elements

3. LIGHTING & ATMOSPHERE (150+ tokens):
   - Light source direction, intensity, color temperature
   - Shadows: where they fall, how hard/soft
   - Reflections and highlights
   - Overall mood and emotional tone
   - Time of day if discernible

4. COLORS & PALETTE (150+ tokens):
   - Dominant colors with specific names (not just "blue" but "cerulean" or "navy")
   - Color relationships and contrasts
   - Saturation and brightness levels
   - Any color gradients or transitions

5. ARTISTIC STYLE (100+ tokens):
   - Medium: photograph, digital art, oil painting, watercolor, etc.
   - If photo: camera angle, lens type, depth of field, bokeh
   - Art style: realistic, stylized, anime, impressionist, etc.
   - Quality indicators: resolution, sharpness, noise

6. FINE DETAILS (200+ tokens):
   - Textures of every surface: fabric weave, skin pores, wood grain, metal finish
   - Small objects that might be overlooked
   - Any text, logos, or writing visible
   - Imperfections, wear, or unique characteristics
   - Edge details and boundaries

Write in flowing, descriptive prose. Use vivid, specific language. Do not use bullet points or numbered lists in your output - write continuous descriptive text.

This description will be used to recreate this exact image, so EVERY detail matters. More is better."""

    # Generate caption with retry loop to ensure minimum length
    min_caption_tokens = args.caption_tokens
    max_attempts = 3
    caption = None
    thinking = None
    raw_caption = None

    tokenizer = vl_extractor.processor.tokenizer

    for attempt in range(max_attempts):
        logger.info(f"Generating caption attempt {attempt + 1}/{max_attempts} (max_tokens={args.max_tokens}, target >= {min_caption_tokens})...")

        raw_caption = vl_extractor.generate(
            image=input_image,
            system_prompt=caption_system_prompt,
            max_new_tokens=args.max_tokens,
            temperature=0.6 + (attempt * 0.1),  # Slightly increase temperature on retries
            top_p=0.95,
            top_k=20,
            presence_penalty=0.0,
            do_sample=True,
        )

        # Parse out thinking content
        caption, thinking = parse_thinking_content(raw_caption)
        caption_token_count = count_tokens(caption, tokenizer)

        logger.info(f"  Caption length: {caption_token_count} tokens")

        if caption_token_count >= min_caption_tokens:
            logger.info(f"  Caption meets minimum length requirement!")
            break
        else:
            logger.warning(f"  Caption too short ({caption_token_count} < {min_caption_tokens}), retrying...")

    if caption_token_count < min_caption_tokens:
        logger.warning(f"Could not generate caption >= {min_caption_tokens} tokens after {max_attempts} attempts. Proceeding with {caption_token_count} tokens.")

    # Count tokens for reporting
    raw_token_count = count_tokens(raw_caption, tokenizer)
    thinking_token_count = count_tokens(thinking, tokenizer) if thinking else 0

    logger.info(f"Generated caption:")
    logger.info(f"  - Raw output: {raw_token_count} tokens")
    logger.info(f"  - Caption only: {caption_token_count} tokens")
    if thinking:
        logger.info(f"  - Thinking block: {thinking_token_count} tokens")

    # Save caption and thinking
    (output_dir / "caption.txt").write_text(caption)
    if thinking:
        (output_dir / "thinking.txt").write_text(thinking)
    (output_dir / "raw_output.txt").write_text(raw_caption)

    logger.info(f"Caption preview (first 500 chars):")
    logger.info(f"  {caption[:500]}...")

    # Now extract VL embeddings with the generated caption
    logger.info(f"\nExtracting VL embeddings with generated caption + image...")
    logger.info(f"  - Hidden layer: {args.vl_hidden_layer}")
    logger.info(f"  - Alpha: {args.vl_alpha}")

    vl_result = vl_extractor.extract(
        image=input_image,
        text=caption,  # Use the generated caption as text context
        hidden_layer=args.vl_hidden_layer,
        text_tokens_only=False,  # Include image tokens
        scale_to_text=True,
        normalization_mode="global",
    )

    logger.info(f"VL extraction result:")
    logger.info(f"  - Embeddings shape: {vl_result.embeddings.shape}")
    logger.info(f"  - Num tokens: {vl_result.num_tokens}")
    logger.info(f"  - Original std: {vl_result.original_std:.4f}")
    logger.info(f"  - Scaled std: {vl_result.scaled_std:.4f}")
    logger.info(f"  - Model variant: {vl_result.model_variant}")

    # Save VL embedding stats
    vl_emb = vl_result.embeddings

    # Also extract with a generic short prompt for comparison
    logger.info("\nExtracting VL embeddings with generic short prompt for comparison...")
    vl_result_generic = vl_extractor.extract(
        image=input_image,
        text="Describe this image",  # Generic prompt
        hidden_layer=args.vl_hidden_layer,
        text_tokens_only=False,
        scale_to_text=True,
        normalization_mode="global",
    )
    vl_emb_generic = vl_result_generic.embeddings

    logger.info(f"Generic VL result: {vl_result_generic.num_tokens} tokens")

    # Unload VL model before loading Z-Image
    logger.info("\nUnloading VL model to free VRAM...")
    vl_extractor.unload()
    del vl_extractor
    gc.collect()
    torch.cuda.empty_cache()

    # Load Z-Image pipeline
    logger.info("\nLoading Z-Image pipeline...")

    from llm_dit.cli import load_runtime_config
    from llm_dit.startup import PipelineLoader

    # Create args-like object for load_runtime_config
    class ConfigArgs:
        pass

    config_args = ConfigArgs()
    config_args.config = args.config
    config_args.profile = args.profile
    config_args.steps = args.steps

    # Set defaults for all expected attributes
    for attr in ['model_path', 'text_encoder_device', 'dit_device', 'vae_device',
                 'cpu_offload', 'flash_attn', 'compile', 'debug', 'verbose',
                 'attention_backend', 'use_custom_scheduler', 'tiled_vae',
                 'embedding_cache', 'long_prompt_mode', 'hidden_layer', 'shift',
                 'lora', 'api_url', 'api_model', 'local_encoder', 'templates_dir',
                 'torch_dtype', 'text_encoder_path', 'tile_size', 'tile_overlap',
                 'cache_size', 'rewriter_use_api', 'rewriter_api_url', 'rewriter_api_model',
                 'rewriter_temperature', 'rewriter_top_p', 'rewriter_top_k',
                 'rewriter_min_p', 'rewriter_presence_penalty', 'rewriter_max_tokens',
                 'width', 'height', 'guidance_scale', 'negative_prompt', 'seed',
                 'embeddings_file', 'template', 'system_prompt', 'thinking_content',
                 'assistant_content', 'enable_thinking',
                 'vl_model_path', 'vl_device', 'vl_hidden_layer', 'vl_alpha',
                 'vl_blend_mode', 'vl_auto_unload']:
        if not hasattr(config_args, attr):
            setattr(config_args, attr, None)

    # Override with our specific settings
    if args.z_image_path:
        config_args.model_path = args.z_image_path
    config_args.seed = args.seed

    runtime_config = load_runtime_config(config_args)

    z_image_path = runtime_config.model_path
    if not z_image_path:
        # Try common paths
        for path in [Path.home() / "Storage" / "z-image-turbo"]:
            if path.exists():
                z_image_path = str(path)
                runtime_config.model_path = z_image_path
                break

    if not z_image_path:
        raise RuntimeError("Could not find Z-Image model. Specify --z-image-path")

    logger.info(f"Using Z-Image from: {z_image_path}")

    # Load pipeline
    loader = PipelineLoader(runtime_config)
    pipeline_result = loader.load_pipeline()
    pipe = pipeline_result.pipeline
    text_encoder = pipeline_result.encoder
    logger.info("Z-Image pipeline loaded")

    # Get text embeddings for the caption
    logger.info("\nEncoding caption with Qwen3-4B text encoder...")
    text_result = text_encoder.encode(caption)
    text_emb = text_result.embeddings[0]  # Remove batch dim

    logger.info(f"Text embeddings: {text_emb.shape}")
    logger.info(f"  - Mean: {text_emb.mean():.4f}, Std: {text_emb.std():.4f}")

    # Also encode a generic prompt
    generic_result = text_encoder.encode("Describe this image")
    generic_text_emb = generic_result.embeddings[0]

    # Blend embeddings
    logger.info("\nBlending embeddings...")

    # Z-Image has a 1504 token limit for embeddings
    MAX_TOKENS = 1504

    # blend_interpolate expects 2D tensors (seq, dim), not 3D (batch, seq, dim)
    # Method 1: VL (caption-aligned) + Text (caption)
    blended_aligned = blend_interpolate(
        vl_emb.to("cuda"),  # (seq, dim)
        text_emb.to("cuda"),  # (seq, dim)
        alpha=args.vl_alpha,
    )

    # Interpolate to MAX_TOKENS if too long
    # Note: Pipeline expects 2D embeddings (seq_len, hidden_dim) without batch dimension
    if blended_aligned.shape[0] > MAX_TOKENS:
        logger.info(f"  Blended aligned: {blended_aligned.shape[0]} -> {MAX_TOKENS} tokens (interpolated)")
        blended_aligned = torch.nn.functional.interpolate(
            blended_aligned.T.unsqueeze(0),  # (1, dim, seq)
            size=MAX_TOKENS,
            mode="linear",
            align_corners=False,
        ).squeeze(0).T  # (seq, dim)

    logger.info(f"  Blended aligned shape: {blended_aligned.shape}")

    # Method 2: VL (generic) + Text (caption) - for comparison
    blended_generic_vl = blend_interpolate(
        vl_emb_generic.to("cuda"),
        text_emb.to("cuda"),
        alpha=args.vl_alpha,
    )

    if blended_generic_vl.shape[0] > MAX_TOKENS:
        logger.info(f"  Blended generic: {blended_generic_vl.shape[0]} -> {MAX_TOKENS} tokens (interpolated)")
        blended_generic_vl = torch.nn.functional.interpolate(
            blended_generic_vl.T.unsqueeze(0),
            size=MAX_TOKENS,
            mode="linear",
            align_corners=False,
        ).squeeze(0).T

    logger.info(f"  Blended generic shape: {blended_generic_vl.shape}")

    # Method 3: Pure text (caption only, no VL)
    pure_text_emb = text_emb.to("cuda")
    if pure_text_emb.shape[0] > MAX_TOKENS:
        logger.info(f"  Pure text: {pure_text_emb.shape[0]} -> {MAX_TOKENS} tokens (interpolated)")
        pure_text_emb = torch.nn.functional.interpolate(
            pure_text_emb.T.unsqueeze(0),
            size=MAX_TOKENS,
            mode="linear",
            align_corners=False,
        ).squeeze(0).T

    logger.info(f"  Pure text shape: {pure_text_emb.shape}")

    # Generate images
    logger.info("\nGenerating images...")
    generator = torch.Generator("cpu").manual_seed(args.seed)

    results = {}

    # Generate with aligned VL + caption
    # Note: Pipeline returns PIL.Image directly when batch_size=1
    logger.info("  1/3: VL (caption-aligned) + Text (caption)...")
    generator.manual_seed(args.seed)
    image_aligned = pipe(
        prompt_embeds=blended_aligned,
        width=1024,
        height=1024,
        num_inference_steps=args.steps,
        generator=generator,
    )
    image_aligned.save(output_dir / "result_vl_aligned.png")
    results["vl_aligned"] = {
        "description": "VL embeddings with caption-aligned text + blending",
        "vl_prompt": caption[:100] + "...",
        "alpha": args.vl_alpha,
    }

    # Generate with generic VL + caption
    logger.info("  2/3: VL (generic) + Text (caption)...")
    generator.manual_seed(args.seed)
    image_generic_vl = pipe(
        prompt_embeds=blended_generic_vl,
        width=1024,
        height=1024,
        num_inference_steps=args.steps,
        generator=generator,
    )
    image_generic_vl.save(output_dir / "result_vl_generic.png")
    results["vl_generic"] = {
        "description": "VL embeddings with generic text + caption blending",
        "vl_prompt": "Describe this image",
        "alpha": args.vl_alpha,
    }

    # Generate with pure text (caption only)
    logger.info("  3/3: Pure text (caption only, no VL)...")
    generator.manual_seed(args.seed)
    image_text_only = pipe(
        prompt_embeds=pure_text_emb,
        width=1024,
        height=1024,
        num_inference_steps=args.steps,
        generator=generator,
    )
    image_text_only.save(output_dir / "result_text_only.png")
    results["text_only"] = {
        "description": "Pure text embedding from caption",
        "prompt": caption[:100] + "...",
    }

    # Save metadata
    metadata = {
        "input_image": args.image,
        "timestamp": timestamp,
        "config": {
            "caption_tokens_target": args.caption_tokens,
            "max_tokens": args.max_tokens,
            "steps": args.steps,
            "seed": args.seed,
            "vl_hidden_layer": args.vl_hidden_layer,
            "vl_alpha": args.vl_alpha,
            "vl_model_path": vl_model_path,
            "z_image_path": z_image_path,
        },
        "caption": {
            "raw_tokens": raw_token_count,
            "caption_tokens": caption_token_count,
            "thinking_tokens": thinking_token_count,
            "full_text": caption,
        },
        "vl_extraction": {
            "aligned": {
                "num_tokens": vl_result.num_tokens,
                "original_std": vl_result.original_std,
                "scaled_std": vl_result.scaled_std,
            },
            "generic": {
                "num_tokens": vl_result_generic.num_tokens,
                "original_std": vl_result_generic.original_std,
                "scaled_std": vl_result_generic.scaled_std,
            },
        },
        "results": results,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Create comparison grid
    logger.info("\nCreating comparison grid...")

    from PIL import ImageDraw, ImageFont

    images = [
        input_image.resize((512, 512)),
        image_aligned.resize((512, 512)),
        image_generic_vl.resize((512, 512)),
        image_text_only.resize((512, 512)),
    ]
    labels = [
        "Input",
        f"VL+Caption (alpha={args.vl_alpha})",
        f"VL Generic (alpha={args.vl_alpha})",
        "Text Only",
    ]

    grid = Image.new("RGB", (512 * 4 + 30, 512 + 80), (255, 255, 255))

    for i, (img, label) in enumerate(zip(images, labels)):
        x = 10 + i * (512 + 5)
        grid.paste(img, (x, 60))
        draw = ImageDraw.Draw(grid)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
        draw.text((x + 5, 10), label, fill=(0, 0, 0), font=font)

    grid.save(output_dir / "comparison_grid.png")

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"  - comparison_grid.png: Side-by-side comparison")
    logger.info(f"  - caption.txt: Generated caption ({caption_token_count} tokens)")
    logger.info(f"  - metadata.json: Full experiment metadata")

    print(f"\n{'='*60}")
    print(f"Experiment complete!")
    print(f"Results: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
