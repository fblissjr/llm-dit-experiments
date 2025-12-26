#!/usr/bin/env python3
"""
Test image generation with Qwen3-Embedding-4B vs Qwen3-4B.

Last updated: 2025-12-14

This script compares images generated using embeddings from:
1. Qwen3-4B (Z-Image's default encoder)
2. Qwen3-Embedding-4B (alternative embedding model)

Based on embedding comparison analysis:
- Layer -2 produces 99% cosine similarity between models
- Apply 1.15x scaling to match Qwen3-4B magnitude distribution
- No instruction prefix (dilutes prompt)

Usage:
    uv run experiments/test_embedding_generation.py --quick
    uv run experiments/test_embedding_generation.py --prompts "A cat" "A mountain"
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Model paths from environment variables
ZIMAGE_PATH = os.environ.get("ZIMAGE_PATH")
QWEN3_4B_PATH = os.environ.get("QWEN3_PATH")
QWEN3_EMBEDDING_PATH = os.environ.get("QWEN3_EMBEDDING_PATH")

if not all([ZIMAGE_PATH, QWEN3_4B_PATH, QWEN3_EMBEDDING_PATH]):
    raise ValueError(
        "Set environment variables: ZIMAGE_PATH, QWEN3_PATH, QWEN3_EMBEDDING_PATH"
    )

# Test prompts
DEFAULT_PROMPTS = [
    "A cat sleeping in sunlight",
    "A mountain landscape at sunset",
]


def create_comparison_grid(images: list[Image.Image], labels: list[str], title: str = "") -> Image.Image:
    """Create a comparison grid from multiple images."""
    from PIL import ImageDraw, ImageFont

    if not images:
        raise ValueError("No images to create grid")

    n_images = len(images)
    img_width, img_height = images[0].size

    # Grid layout
    cols = min(n_images, 4)
    rows = (n_images + cols - 1) // cols

    # Add space for labels
    label_height = 40
    grid_width = cols * img_width
    grid_height = rows * (img_height + label_height)

    # Add title space
    title_height = 60 if title else 0

    grid = Image.new("RGB", (grid_width, grid_height + title_height), "white")
    draw = ImageDraw.Draw(grid)

    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
        title_font = font

    # Draw title
    if title:
        draw.text((grid_width // 2, 20), title, fill="black", font=title_font, anchor="mm")

    # Place images with labels
    for i, (img, label) in enumerate(zip(images, labels)):
        row = i // cols
        col = i % cols

        x = col * img_width
        y = title_height + row * (img_height + label_height)

        grid.paste(img, (x, y))

        # Draw label
        label_y = y + img_height + label_height // 2
        draw.text((x + img_width // 2, label_y), label, fill="black", font=font, anchor="mm")

    return grid


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-Embedding-4B for image generation")
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS, help="Prompts to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--steps", type=int, default=9, help="Inference steps")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/embedding_generation_test"))
    parser.add_argument("--quick", action="store_true", help="Quick test with one prompt")
    parser.add_argument("--scale-factors", nargs="+", type=float, default=[1.0, 1.15],
                       help="Scale factors to test for embedding model")
    args = parser.parse_args()

    if args.quick:
        args.prompts = [args.prompts[0]]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading models...")

    # Import here to avoid slow startup
    from llm_dit.embedding import EmbeddingExtractor
    from llm_dit.backends.transformers import TransformersBackend
    from llm_dit.pipelines.z_image import ZImagePipeline

    def format_prompt_chat(prompt: str) -> str:
        """Format prompt with Qwen3 chat template.

        Note: This produces 58% cosine similarity between models due to
        different handling of special tokens. Use for Qwen3-4B only.
        """
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    def format_prompt_raw(prompt: str) -> str:
        """Return raw prompt without chat template.

        Note: This produces 98% cosine similarity between models.
        Use for Qwen3-Embedding-4B to maximize compatibility.
        """
        return prompt

    # -------------------------------------------------------------------------
    # Phase 1: Extract embeddings from both models (sequentially to save VRAM)
    # -------------------------------------------------------------------------
    all_embeddings = {}  # prompt -> {"qwen3": tensor, "embedding_X": tensor, ...}

    # Load Qwen3-4B, encode, unload
    logger.info("Loading Qwen3-4B encoder...")
    qwen3_backend = TransformersBackend.from_pretrained(
        QWEN3_4B_PATH,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        model_subfolder="",  # No subfolder - model files at root
        tokenizer_subfolder="",  # No subfolder - tokenizer at root
    )

    logger.info("Encoding prompts with Qwen3-4B...")
    for prompt in args.prompts:
        # Use chat template for Qwen3-4B (Z-Image official format)
        formatted = format_prompt_chat(prompt)
        output = qwen3_backend.encode([formatted])  # Note: expects list of strings
        all_embeddings[prompt] = {"qwen3_chat": output.embeddings[0].cpu()}
        logger.info(f"  '{prompt}' (chat): {len(all_embeddings[prompt]['qwen3_chat'])} tokens")

        # Also encode raw prompt for comparison
        output_raw = qwen3_backend.encode([prompt])
        all_embeddings[prompt]["qwen3_raw"] = output_raw.embeddings[0].cpu()
        logger.info(f"  '{prompt}' (raw): {len(all_embeddings[prompt]['qwen3_raw'])} tokens")

    # Unload Qwen3-4B
    logger.info("Unloading Qwen3-4B...")
    del qwen3_backend
    torch.cuda.empty_cache()

    # Load Qwen3-Embedding-4B, encode, unload
    logger.info("Loading Qwen3-Embedding-4B...")
    embedding_extractor = EmbeddingExtractor.from_pretrained(
        QWEN3_EMBEDDING_PATH,
        device="cuda",
        torch_dtype=torch.bfloat16,
    )

    logger.info("Encoding prompts with Qwen3-Embedding-4B (raw prompt)...")
    for prompt in args.prompts:
        for scale in args.scale_factors:
            # Use RAW prompt - chat template breaks compatibility (58% vs 98% cosine sim)
            emb = embedding_extractor.encode_for_zimage(
                prompt, hidden_layer=-2, scale_factor=scale
            )
            all_embeddings[prompt][f"embedding_raw_{scale}"] = emb.cpu()
            logger.info(f"  '{prompt}' (raw, scale={scale}): {len(emb)} tokens, std={emb.std():.2f}")

    # Unload embedding model
    logger.info("Unloading Qwen3-Embedding-4B...")
    embedding_extractor.unload()
    del embedding_extractor
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Phase 2: Generate images with Z-Image
    # -------------------------------------------------------------------------
    logger.info("Loading Z-Image pipeline...")
    pipe = ZImagePipeline.from_pretrained(
        ZIMAGE_PATH,
        torch_dtype=torch.bfloat16,
        text_encoder_device="cpu",  # Save VRAM - we have pre-computed embeddings
        dit_device="cuda",
        vae_device="cuda",
    )

    # Process each prompt
    for prompt in args.prompts:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing prompt: {prompt}")
        logger.info(f"{'='*60}")

        images = []
        labels = []
        prompt_embeddings = all_embeddings[prompt]
        safe_prompt = prompt.replace(" ", "_")[:30]

        # Generate with Qwen3-4B (chat template - official Z-Image format)
        logger.info("Generating with Qwen3-4B (chat template)...")
        qwen3_chat_emb = prompt_embeddings["qwen3_chat"]
        logger.info(f"  Qwen3-4B chat: {len(qwen3_chat_emb)} tokens, std={qwen3_chat_emb.std():.2f}")

        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        result = pipe(
            prompt_embeds=qwen3_chat_emb.to("cuda", torch.bfloat16),
            num_inference_steps=args.steps,
            generator=generator,
        )
        img = result.images[0] if hasattr(result, 'images') else result
        images.append(img)
        labels.append(f"Qwen3-4B\n(chat)\nstd={qwen3_chat_emb.std():.1f}")
        img.save(args.output_dir / f"{safe_prompt}_qwen3_4b_chat.png")

        # Generate with Qwen3-4B (raw prompt - for fair comparison)
        logger.info("Generating with Qwen3-4B (raw prompt)...")
        qwen3_raw_emb = prompt_embeddings["qwen3_raw"]
        logger.info(f"  Qwen3-4B raw: {len(qwen3_raw_emb)} tokens, std={qwen3_raw_emb.std():.2f}")

        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        result = pipe(
            prompt_embeds=qwen3_raw_emb.to("cuda", torch.bfloat16),
            num_inference_steps=args.steps,
            generator=generator,
        )
        img = result.images[0] if hasattr(result, 'images') else result
        images.append(img)
        labels.append(f"Qwen3-4B\n(raw)\nstd={qwen3_raw_emb.std():.1f}")
        img.save(args.output_dir / f"{safe_prompt}_qwen3_4b_raw.png")

        # Generate with Qwen3-Embedding-4B (raw prompt) at different scale factors
        for scale in args.scale_factors:
            logger.info(f"Generating with Qwen3-Embedding-4B (raw, scale={scale})...")
            emb_embeddings = prompt_embeddings[f"embedding_raw_{scale}"]
            logger.info(f"  Embedding raw: {len(emb_embeddings)} tokens, std={emb_embeddings.std():.2f}")

            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            result = pipe(
                prompt_embeds=emb_embeddings.to("cuda", torch.bfloat16),
                num_inference_steps=args.steps,
                generator=generator,
            )
            img = result.images[0] if hasattr(result, 'images') else result
            images.append(img)
            labels.append(f"Qwen3-Emb\n(raw)\nscale={scale}")
            img.save(args.output_dir / f"{safe_prompt}_embedding_raw_scale{scale}.png")

        # Create comparison grid
        grid = create_comparison_grid(
            images,
            labels,
            title=f'"{prompt}" (seed={args.seed})'
        )
        grid.save(args.output_dir / f"{safe_prompt}_comparison.png")
        logger.info(f"Saved comparison grid to {args.output_dir / f'{safe_prompt}_comparison.png'}")

    # Cleanup
    logger.info("\nCleaning up...")
    del pipe
    torch.cuda.empty_cache()

    logger.info(f"\nResults saved to {args.output_dir}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
