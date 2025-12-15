#!/usr/bin/env python3
"""
Fix Qwen3-Embedding-4B dimensions based on per-dimension analysis.

Last updated: 2025-12-14

Applies dimension-level corrections to make Qwen3-Embedding-4B embeddings
compatible with Z-Image DiT training distribution.

Fixes applied:
1. Rescale outlier dimensions (std ratio > 3x or < 0.33x)
2. Zero out embedding-only dead dimensions
3. Clamp embedding-only hyperactive dimensions
4. Optional: Full distribution matching per dimension

Usage:
    # Generate images with dimension fixes
    uv run experiments/fix_embedding_dimensions.py \
        --prompt "A cat sleeping" \
        --output results/fixed_comparison/

    # Test different fix strategies
    uv run experiments/fix_embedding_dimensions.py \
        --prompt "A mountain landscape" \
        --fix-modes rescale mask clamp full \
        --output results/fix_comparison/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import torch.nn.functional as F
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


QWEN3_4B_PATH = "/home/fbliss/Storage/Qwen3-4B"
QWEN3_EMBEDDING_PATH = "/home/fbliss/Storage/Qwen3-Embedding-4B"
ZIMAGE_PATH = "/home/fbliss/Storage/Tongyi-MAI_Z-Image-Turbo"


def compute_dimension_stats(embeddings: torch.Tensor) -> dict:
    """Compute per-dimension statistics."""
    flat = embeddings.view(-1, embeddings.size(-1))
    return {
        "means": flat.mean(dim=0),
        "stds": flat.std(dim=0),
        "mins": flat.min(dim=0).values,
        "maxs": flat.max(dim=0).values,
    }


def identify_problematic_dimensions(
    emb_stats: dict,
    qwen3_stats: dict,
    outlier_threshold: float = 3.0,
    dead_threshold: float = 0.01,
    hyperactive_multiplier: float = 5.0,
) -> dict:
    """Identify dimensions that need fixing."""
    emb_stds = emb_stats["stds"]
    qwen3_stds = qwen3_stats["stds"]

    # Avoid division by zero
    safe_qwen3_stds = torch.where(qwen3_stds < 1e-6, torch.ones_like(qwen3_stds), qwen3_stds)
    std_ratios = emb_stds / safe_qwen3_stds

    # Outliers
    outlier_high = (std_ratios > outlier_threshold).nonzero(as_tuple=True)[0]
    outlier_low = (std_ratios < 1.0 / outlier_threshold).nonzero(as_tuple=True)[0]

    # Dead dimensions
    emb_dead = (emb_stds < dead_threshold).nonzero(as_tuple=True)[0]
    qwen3_dead = (qwen3_stds < dead_threshold).nonzero(as_tuple=True)[0]
    emb_only_dead = torch.tensor([d for d in emb_dead if d not in qwen3_dead])

    # Hyperactive dimensions
    emb_median = emb_stds.median()
    qwen3_median = qwen3_stds.median()
    emb_hyper = (emb_stds > emb_median * hyperactive_multiplier).nonzero(as_tuple=True)[0]
    qwen3_hyper = (qwen3_stds > qwen3_median * hyperactive_multiplier).nonzero(as_tuple=True)[0]
    emb_only_hyper = torch.tensor([d for d in emb_hyper if d not in qwen3_hyper])

    return {
        "outlier_high": outlier_high,
        "outlier_low": outlier_low,
        "std_ratios": std_ratios,
        "emb_only_dead": emb_only_dead,
        "emb_only_hyper": emb_only_hyper,
    }


def fix_rescale_outliers(embeddings: torch.Tensor, problematic: dict, emb_stats: dict, qwen3_stats: dict) -> torch.Tensor:
    """Rescale outlier dimensions to match Qwen3-4B distribution."""
    fixed = embeddings.clone()
    emb_stds = emb_stats["stds"]
    qwen3_stds = qwen3_stats["stds"]

    all_outliers = torch.cat([problematic["outlier_high"], problematic["outlier_low"]])

    for dim in all_outliers:
        if emb_stds[dim] > 1e-6:  # Avoid division by zero
            scale = qwen3_stds[dim] / emb_stds[dim]
            fixed[:, dim] *= scale

    logger.info(f"  Rescaled {len(all_outliers)} outlier dimensions")
    return fixed


def fix_mask_dead(embeddings: torch.Tensor, problematic: dict) -> torch.Tensor:
    """Zero out embedding-only dead dimensions."""
    fixed = embeddings.clone()
    emb_only_dead = problematic["emb_only_dead"]

    for dim in emb_only_dead:
        fixed[:, dim] = 0.0

    logger.info(f"  Masked {len(emb_only_dead)} embedding-only dead dimensions")
    return fixed


def fix_clamp_hyperactive(embeddings: torch.Tensor, problematic: dict, qwen3_stats: dict) -> torch.Tensor:
    """Clamp embedding-only hyperactive dimensions."""
    fixed = embeddings.clone()
    emb_only_hyper = problematic["emb_only_hyper"]
    qwen3_stds = qwen3_stats["stds"]

    for dim in emb_only_hyper:
        # Clamp to ±3 std of Qwen3-4B
        clamp_range = 3.0 * qwen3_stds[dim]
        fixed[:, dim] = torch.clamp(fixed[:, dim], -clamp_range, clamp_range)

    logger.info(f"  Clamped {len(emb_only_hyper)} embedding-only hyperactive dimensions")
    return fixed


def fix_full_distribution_matching(embeddings: torch.Tensor, emb_stats: dict, qwen3_stats: dict) -> torch.Tensor:
    """Match full per-dimension distribution to Qwen3-4B."""
    fixed = embeddings.clone()
    emb_means = emb_stats["means"]
    emb_stds = emb_stats["stds"]
    qwen3_means = qwen3_stats["means"]
    qwen3_stds = qwen3_stats["stds"]

    for dim in range(fixed.size(-1)):
        if emb_stds[dim] > 1e-6:  # Avoid division by zero
            # Standardize to zero mean, unit variance
            fixed[:, dim] = (fixed[:, dim] - emb_means[dim]) / emb_stds[dim]
            # Rescale to Qwen3-4B distribution
            fixed[:, dim] = fixed[:, dim] * qwen3_stds[dim] + qwen3_means[dim]

    logger.info(f"  Applied full distribution matching to all 2560 dimensions")
    return fixed


def apply_fixes(
    embeddings: torch.Tensor,
    emb_stats: dict,
    qwen3_stats: dict,
    mode: str = "rescale",
) -> torch.Tensor:
    """Apply dimension fixes based on mode."""
    logger.info(f"Applying fix mode: {mode}")

    # Identify problematic dimensions
    problematic = identify_problematic_dimensions(emb_stats, qwen3_stats)

    if mode == "rescale":
        return fix_rescale_outliers(embeddings, problematic, emb_stats, qwen3_stats)
    elif mode == "mask":
        fixed = fix_rescale_outliers(embeddings, problematic, emb_stats, qwen3_stats)
        fixed = fix_mask_dead(fixed, problematic)
        return fixed
    elif mode == "clamp":
        fixed = fix_rescale_outliers(embeddings, problematic, emb_stats, qwen3_stats)
        fixed = fix_mask_dead(fixed, problematic)
        fixed = fix_clamp_hyperactive(fixed, problematic, qwen3_stats)
        return fixed
    elif mode == "full":
        return fix_full_distribution_matching(embeddings, emb_stats, qwen3_stats)
    else:
        raise ValueError(f"Unknown fix mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description="Fix Qwen3-Embedding dimensions for Z-Image")
    parser.add_argument("--prompt", default="A cat sleeping in sunlight", help="Test prompt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--steps", type=int, default=9, help="Inference steps")
    parser.add_argument(
        "--fix-modes",
        nargs="+",
        default=["none", "rescale", "mask", "clamp", "full"],
        help="Fix modes to test"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/dimension_fix_test"),
        help="Output directory"
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("Dimension Fix Test")
    logger.info("="*60)
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Fix modes: {args.fix_modes}")
    logger.info("")

    from llm_dit.embedding import EmbeddingExtractor
    from llm_dit.backends.transformers import TransformersBackend
    from llm_dit.pipelines.z_image import ZImagePipeline

    # -------------------------------------------------------------------------
    # Phase 1: Extract embeddings
    # -------------------------------------------------------------------------
    logger.info("Phase 1: Extracting embeddings...")

    # Qwen3-4B (reference)
    logger.info("  Loading Qwen3-4B...")
    qwen3_backend = TransformersBackend.from_pretrained(
        QWEN3_4B_PATH,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        model_subfolder="",
        tokenizer_subfolder="",
    )

    output = qwen3_backend.encode([args.prompt])
    qwen3_emb = output.embeddings[0].cpu()
    qwen3_stats = compute_dimension_stats(qwen3_emb)
    logger.info(f"  Qwen3-4B: {qwen3_emb.shape}")

    del qwen3_backend
    torch.cuda.empty_cache()

    # Qwen3-Embedding-4B
    logger.info("  Loading Qwen3-Embedding-4B...")
    embedding_extractor = EmbeddingExtractor.from_pretrained(
        QWEN3_EMBEDDING_PATH,
        device="cuda",
        torch_dtype=torch.bfloat16,
    )

    emb_emb = embedding_extractor.encode_for_zimage(args.prompt, hidden_layer=-2, scale_factor=1.0)
    emb_stats = compute_dimension_stats(emb_emb)
    logger.info(f"  Qwen3-Embedding: {emb_emb.shape}")

    embedding_extractor.unload()
    del embedding_extractor
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Phase 2: Apply fixes and generate images
    # -------------------------------------------------------------------------
    logger.info("\nPhase 2: Generating images with fixes...")

    logger.info("  Loading Z-Image pipeline...")
    pipe = ZImagePipeline.from_pretrained(
        ZIMAGE_PATH,
        torch_dtype=torch.bfloat16,
        text_encoder_device="cpu",
        dit_device="cuda",
        vae_device="cuda",
    )

    images = []
    labels = []

    # Baseline: Qwen3-4B (reference)
    logger.info("\n  Generating with Qwen3-4B (reference)...")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    result = pipe(
        prompt_embeds=qwen3_emb.to("cuda", torch.bfloat16),
        num_inference_steps=args.steps,
        generator=generator,
    )
    img = result.images[0] if hasattr(result, 'images') else result
    images.append(img)
    labels.append("Qwen3-4B\n(reference)")
    img.save(args.output / "qwen3_4b_reference.png")

    # Test each fix mode
    for mode in args.fix_modes:
        logger.info(f"\n  Generating with fix mode: {mode}...")

        if mode == "none":
            # Raw embedding without fixes
            fixed_emb = emb_emb
        else:
            # Apply fixes
            fixed_emb = apply_fixes(emb_emb, emb_stats, qwen3_stats, mode=mode)

        # Compute similarity after fixing
        similarity = F.cosine_similarity(
            fixed_emb.flatten().unsqueeze(0),
            qwen3_emb.flatten().unsqueeze(0),
        ).item()

        logger.info(f"    Cosine similarity with Qwen3-4B: {similarity:.4f}")

        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        result = pipe(
            prompt_embeds=fixed_emb.to("cuda", torch.bfloat16),
            num_inference_steps=args.steps,
            generator=generator,
        )
        img = result.images[0] if hasattr(result, 'images') else result
        images.append(img)
        labels.append(f"Fix: {mode}\nsim={similarity:.3f}")
        img.save(args.output / f"embedding_fix_{mode}.png")

    # -------------------------------------------------------------------------
    # Phase 3: Create comparison grid
    # -------------------------------------------------------------------------
    logger.info("\nPhase 3: Creating comparison grid...")

    from PIL import ImageDraw, ImageFont

    # Grid layout
    cols = min(len(images), 3)
    rows = (len(images) + cols - 1) // cols

    img_width, img_height = images[0].size
    label_height = 60
    grid_width = cols * img_width
    grid_height = rows * (img_height + label_height)
    title_height = 80

    grid = Image.new("RGB", (grid_width, grid_height + title_height), "white")
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
        title_font = font

    # Title
    draw.text(
        (grid_width // 2, 20),
        f"Dimension Fix Comparison",
        fill="black",
        font=title_font,
        anchor="mm"
    )
    draw.text(
        (grid_width // 2, 50),
        f'Prompt: "{args.prompt}" (seed={args.seed})',
        fill="gray",
        font=font,
        anchor="mm"
    )

    # Place images
    for i, (img, label) in enumerate(zip(images, labels)):
        row = i // cols
        col = i % cols

        x = col * img_width
        y = title_height + row * (img_height + label_height)

        grid.paste(img, (x, y))

        label_y = y + img_height + label_height // 2
        draw.text((x + img_width // 2, label_y), label, fill="black", font=font, anchor="mm")

    grid.save(args.output / "comparison_grid.png")
    logger.info(f"  Saved: {args.output / 'comparison_grid.png'}")

    # -------------------------------------------------------------------------
    # Phase 4: Save analysis
    # -------------------------------------------------------------------------
    logger.info("\nPhase 4: Saving analysis...")

    problematic = identify_problematic_dimensions(emb_stats, qwen3_stats)

    analysis = {
        "prompt": args.prompt,
        "problematic_dimensions": {
            "outlier_high_count": len(problematic["outlier_high"]),
            "outlier_low_count": len(problematic["outlier_low"]),
            "emb_only_dead_count": len(problematic["emb_only_dead"]),
            "emb_only_hyper_count": len(problematic["emb_only_hyper"]),
            "outlier_high_dims": problematic["outlier_high"].tolist()[:20],
            "outlier_low_dims": problematic["outlier_low"].tolist()[:20],
            "emb_only_dead_dims": problematic["emb_only_dead"].tolist(),
            "emb_only_hyper_dims": problematic["emb_only_hyper"].tolist(),
        },
        "std_stats": {
            "emb_mean": emb_stats["stds"].mean().item(),
            "emb_median": emb_stats["stds"].median().item(),
            "qwen3_mean": qwen3_stats["stds"].mean().item(),
            "qwen3_median": qwen3_stats["stds"].median().item(),
        },
    }

    with open(args.output / "dimension_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)

    logger.info(f"  Saved: {args.output / 'dimension_analysis.json'}")

    # Cleanup
    del pipe
    torch.cuda.empty_cache()

    logger.info("\n" + "="*60)
    logger.info("Dimension fix test complete!")
    logger.info(f"Results saved to: {args.output}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
