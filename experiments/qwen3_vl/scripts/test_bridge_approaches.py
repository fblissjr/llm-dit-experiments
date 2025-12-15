#!/usr/bin/env python3
"""
Test novel approaches to bridge VL and text embeddings for Z-Image.

Last updated: 2025-12-14

Problem: Qwen3-VL can process images but produces artifacts when used with Z-Image.
Goal: Find a zero-shot method to inject image information into Qwen3-4B embedding space.

Approaches tested (priority order):
1. Gram Matrix Style Transfer - Second-order statistics capture style without position
2. Surgical Dimension Fixing - Replace outlier dimensions with text equivalents
3. Slerp Interpolation - Spherical interpolation preserves manifold structure

Usage:
    uv run experiments/qwen3_vl/scripts/test_bridge_approaches.py \
        --reference experiments/inputs/style_reference.png \
        --prompt "Homer Simpson eating a donut" \
        --output-dir experiments/results/bridge_test
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import torch.nn.functional as F
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Model paths
ZIMAGE_PATH = "/home/fbliss/Storage/Tongyi-MAI_Z-Image-Turbo"
QWEN3_4B_PATH = "/home/fbliss/Storage/Qwen3-4B"
QWEN3_VL_PATH = "/home/fbliss/Storage/Qwen3-VL-4B-Instruct"

# Outlier dimensions identified from per-dimension analysis
# High std ratio (VL has MORE variance than expected)
HIGH_VARIANCE_DIMS = [2341, 1573, 2083, 1065, 1323, 1761, 1142, 1320, 1331, 1459]
# Low std ratio (VL has LESS variance than expected)
LOW_VARIANCE_DIMS = [329, 2492, 473, 1672, 779, 411, 1783, 1834, 1815, 236]
# Combined outlier dimensions
OUTLIER_DIMS = HIGH_VARIANCE_DIMS + LOW_VARIANCE_DIMS


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """
    Compute Gram matrix for style transfer.

    The Gram matrix G = F^T @ F captures correlations between feature channels,
    which represents "style" (texture, patterns) without spatial/positional info.

    Args:
        features: (seq_len, hidden_dim) tensor

    Returns:
        Gram matrix (hidden_dim, hidden_dim)
    """
    # Normalize by number of elements for numerical stability
    n_elements = features.shape[0]
    gram = torch.mm(features.T, features) / n_elements
    return gram


def apply_gram_style(
    content_emb: torch.Tensor,
    style_gram: torch.Tensor,
    alpha: float = 0.3,
    iterations: int = 100,
) -> torch.Tensor:
    """
    Apply style from Gram matrix to content embeddings.

    This is a simplified version of neural style transfer at the embedding level.
    We optimize the content embeddings to match the style Gram matrix while
    preserving content structure.

    Args:
        content_emb: Content embeddings (seq_len, hidden_dim)
        style_gram: Style Gram matrix (hidden_dim, hidden_dim)
        alpha: Style weight (0=pure content, 1=pure style)
        iterations: Optimization iterations

    Returns:
        Styled embeddings
    """
    # Start from content embeddings
    styled = content_emb.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([styled], lr=0.1)

    content_gram = gram_matrix(content_emb.detach())

    for i in range(iterations):
        optimizer.zero_grad()

        # Content loss: preserve original structure
        content_loss = F.mse_loss(styled, content_emb.detach())

        # Style loss: match Gram matrix
        current_gram = gram_matrix(styled)
        style_loss = F.mse_loss(current_gram, style_gram)

        # Combined loss
        loss = (1 - alpha) * content_loss + alpha * style_loss

        loss.backward()
        optimizer.step()

    return styled.detach()


def gram_style_transfer(
    text_emb: torch.Tensor,
    vl_emb: torch.Tensor,
    alpha: float = 0.3,
) -> torch.Tensor:
    """
    Transfer style from VL embeddings to text embeddings using Gram matrices.

    Args:
        text_emb: Pure text embeddings from Qwen3-4B (seq_len, 2560)
        vl_emb: VL embeddings with image info (seq_len, 2560)
        alpha: Style transfer strength

    Returns:
        Text embeddings with VL style applied
    """
    # Compute style Gram matrix from VL embeddings
    style_gram = gram_matrix(vl_emb)

    # Apply style to text embeddings
    styled = apply_gram_style(text_emb, style_gram, alpha=alpha)

    return styled


def surgical_dimension_fix(
    vl_emb: torch.Tensor,
    text_emb: torch.Tensor,
    outlier_dims: list[int] = OUTLIER_DIMS,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Replace outlier dimensions in VL embeddings with text embedding values.

    Hypothesis: The 20 outlier dimensions (with >5x std ratio) cause artifacts.
    By replacing just those dimensions with text values, we might get VL's
    image understanding without the distributional artifacts.

    Args:
        vl_emb: VL embeddings (seq_len, 2560)
        text_emb: Text embeddings (seq_len, 2560)
        outlier_dims: List of dimension indices to fix
        alpha: Blend ratio for non-outlier dimensions

    Returns:
        Fixed embeddings
    """
    # Match sequence lengths
    min_len = min(len(vl_emb), len(text_emb))
    vl_truncated = vl_emb[:min_len].clone()
    text_truncated = text_emb[:min_len]

    # Replace outlier dimensions entirely with text values
    for dim in outlier_dims:
        if dim < vl_truncated.shape[1]:
            vl_truncated[:, dim] = text_truncated[:, dim]

    # Blend remaining dimensions
    mask = torch.ones(vl_truncated.shape[1], dtype=torch.bool)
    for dim in outlier_dims:
        if dim < len(mask):
            mask[dim] = False

    # Linear blend for non-outlier dimensions
    result = vl_truncated.clone()
    result[:, mask] = alpha * vl_truncated[:, mask] + (1 - alpha) * text_truncated[:, mask]

    return result


def slerp(
    a: torch.Tensor,
    b: torch.Tensor,
    t: float,
) -> torch.Tensor:
    """
    Spherical linear interpolation between two tensors.

    Unlike linear interpolation which cuts through the embedding space,
    slerp follows the hypersphere surface, potentially preserving
    the manifold structure better.

    Args:
        a: First tensor (any shape)
        b: Second tensor (same shape as a)
        t: Interpolation factor (0=a, 1=b)

    Returns:
        Interpolated tensor
    """
    # Normalize to unit vectors
    a_norm = F.normalize(a.flatten(), dim=0)
    b_norm = F.normalize(b.flatten(), dim=0)

    # Compute angle between vectors
    dot = torch.clamp(torch.dot(a_norm, b_norm), -1.0, 1.0)
    omega = torch.acos(dot)

    # Handle near-parallel vectors (fall back to linear)
    if omega.abs() < 1e-6:
        return (1 - t) * a + t * b

    # Slerp formula
    sin_omega = torch.sin(omega)
    coef_a = torch.sin((1 - t) * omega) / sin_omega
    coef_b = torch.sin(t * omega) / sin_omega

    # Interpolate normalized vectors
    result_norm = coef_a * a_norm + coef_b * b_norm

    # Restore magnitude (interpolate magnitudes linearly)
    mag_a = a.flatten().norm()
    mag_b = b.flatten().norm()
    mag_result = (1 - t) * mag_a + t * mag_b

    return (result_norm * mag_result).reshape(a.shape)


def slerp_blend(
    text_emb: torch.Tensor,
    vl_emb: torch.Tensor,
    alpha: float = 0.3,
) -> torch.Tensor:
    """
    Blend text and VL embeddings using spherical interpolation.

    Args:
        text_emb: Text embeddings (seq_len, 2560)
        vl_emb: VL embeddings (seq_len, 2560)
        alpha: VL influence (0=pure text, 1=pure VL)

    Returns:
        Slerp-blended embeddings
    """
    # Match sequence lengths
    min_len = min(len(text_emb), len(vl_emb))
    text_truncated = text_emb[:min_len]
    vl_truncated = vl_emb[:min_len]

    # Apply slerp per token
    result = torch.zeros_like(text_truncated)
    for i in range(min_len):
        result[i] = slerp(text_truncated[i], vl_truncated[i], alpha)

    return result


def linear_blend(
    text_emb: torch.Tensor,
    vl_emb: torch.Tensor,
    alpha: float = 0.3,
) -> torch.Tensor:
    """Standard linear interpolation (baseline)."""
    min_len = min(len(text_emb), len(vl_emb))
    return (1 - alpha) * text_emb[:min_len] + alpha * vl_emb[:min_len]


def create_comparison_grid(
    images: list[Image.Image],
    labels: list[str],
    title: str = "",
) -> Image.Image:
    """Create a comparison grid from multiple images."""
    from PIL import ImageDraw, ImageFont

    n_images = len(images)
    img_width, img_height = images[0].size

    cols = min(n_images, 4)
    rows = (n_images + cols - 1) // cols

    label_height = 60
    title_height = 80 if title else 0

    grid_width = cols * img_width
    grid_height = rows * (img_height + label_height) + title_height

    grid = Image.new("RGB", (grid_width, grid_height), "white")
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
        title_font = font

    if title:
        draw.text((grid_width // 2, 40), title, fill="black", font=title_font, anchor="mm")

    for i, (img, label) in enumerate(zip(images, labels)):
        row = i // cols
        col = i % cols

        x = col * img_width
        y = title_height + row * (img_height + label_height)

        grid.paste(img, (x, y))

        # Multi-line label support
        label_y = y + img_height + 10
        for line in label.split('\n'):
            draw.text((x + img_width // 2, label_y), line, fill="black", font=font, anchor="mt")
            label_y += 22

    return grid


def main():
    parser = argparse.ArgumentParser(description="Test bridge approaches for VL-to-text")
    parser.add_argument("--reference", type=Path, required=True, help="Reference image for style")
    parser.add_argument("--prompt", type=str, default="Homer Simpson eating a donut", help="Text prompt")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/bridge_test"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--alpha", type=float, default=0.3, help="Style/blend strength")
    parser.add_argument("--vl-layer", type=int, default=-6, help="VL hidden layer (default: -6)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load reference image
    if not args.reference.exists():
        logger.error(f"Reference image not found: {args.reference}")
        return

    reference_image = Image.open(args.reference).convert("RGB")
    logger.info(f"Loaded reference image: {args.reference} ({reference_image.size})")

    # Results tracking
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "reference": str(args.reference),
            "prompt": args.prompt,
            "seed": args.seed,
            "alpha": args.alpha,
            "vl_layer": args.vl_layer,
        },
        "approaches": {},
    }

    # =========================================================================
    # Phase 1: Extract embeddings
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("Phase 1: Extracting embeddings")
    logger.info("="*60)

    from llm_dit.backends.transformers import TransformersBackend
    from llm_dit.vl import VLEmbeddingExtractor
    from llm_dit.vl.blending import scale_embeddings

    # Extract text embeddings from Qwen3-4B
    logger.info("Loading Qwen3-4B...")
    text_backend = TransformersBackend.from_pretrained(
        QWEN3_4B_PATH,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        model_subfolder="",
        tokenizer_subfolder="",
    )

    text_output = text_backend.encode([args.prompt])
    text_emb = text_output.embeddings[0].cpu().float()
    logger.info(f"Text embeddings: {text_emb.shape}, std={text_emb.std():.2f}")

    del text_backend
    torch.cuda.empty_cache()

    # Extract VL embeddings
    logger.info("Loading Qwen3-VL...")
    vl_extractor = VLEmbeddingExtractor.from_pretrained(
        QWEN3_VL_PATH,
        device="cuda",
        torch_dtype=torch.bfloat16,
    )

    vl_result = vl_extractor.extract(
        reference_image,
        text=args.prompt,
        hidden_layer=args.vl_layer,
        text_tokens_only=False,  # Include image tokens
        scale_to_text=True,
    )
    vl_emb = vl_result.embeddings.cpu().float()
    logger.info(f"VL embeddings: {vl_emb.shape}, std={vl_emb.std():.2f}")

    vl_extractor.unload()
    del vl_extractor
    torch.cuda.empty_cache()

    # =========================================================================
    # Phase 2: Apply bridge approaches
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("Phase 2: Applying bridge approaches")
    logger.info("="*60)

    approaches = {}

    # Baseline: Pure text
    approaches["text_only"] = {
        "embeddings": text_emb,
        "label": "Text Only\n(Baseline)",
    }

    # Baseline: Linear blend
    logger.info("Applying linear blend...")
    linear_emb = linear_blend(text_emb, vl_emb, alpha=args.alpha)
    approaches["linear_blend"] = {
        "embeddings": linear_emb,
        "label": f"Linear Blend\nalpha={args.alpha}",
    }

    # Approach 1: Gram Matrix Style Transfer
    logger.info("Applying Gram matrix style transfer...")
    try:
        gram_emb = gram_style_transfer(text_emb, vl_emb, alpha=args.alpha)
        approaches["gram_style"] = {
            "embeddings": gram_emb,
            "label": f"Gram Style\nalpha={args.alpha}",
        }
        logger.info(f"  Gram style: std={gram_emb.std():.2f}")
    except Exception as e:
        logger.error(f"Gram style failed: {e}")
        results["approaches"]["gram_style"] = {"error": str(e)}

    # Approach 2: Surgical Dimension Fix
    logger.info("Applying surgical dimension fix...")
    try:
        surgical_emb = surgical_dimension_fix(vl_emb, text_emb, alpha=args.alpha)
        approaches["surgical_fix"] = {
            "embeddings": surgical_emb,
            "label": f"Surgical Fix\n{len(OUTLIER_DIMS)} dims",
        }
        logger.info(f"  Surgical fix: std={surgical_emb.std():.2f}")
    except Exception as e:
        logger.error(f"Surgical fix failed: {e}")
        results["approaches"]["surgical_fix"] = {"error": str(e)}

    # Approach 3: Slerp Interpolation
    logger.info("Applying slerp interpolation...")
    try:
        slerp_emb = slerp_blend(text_emb, vl_emb, alpha=args.alpha)
        approaches["slerp_blend"] = {
            "embeddings": slerp_emb,
            "label": f"Slerp Blend\nalpha={args.alpha}",
        }
        logger.info(f"  Slerp blend: std={slerp_emb.std():.2f}")
    except Exception as e:
        logger.error(f"Slerp failed: {e}")
        results["approaches"]["slerp_blend"] = {"error": str(e)}

    # =========================================================================
    # Phase 3: Generate images
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("Phase 3: Generating images")
    logger.info("="*60)

    from llm_dit.pipelines.z_image import ZImagePipeline

    logger.info("Loading Z-Image pipeline...")
    pipe = ZImagePipeline.from_pretrained(
        ZIMAGE_PATH,
        torch_dtype=torch.bfloat16,
        text_encoder_device="cpu",
        dit_device="cuda",
        vae_device="cuda",
    )

    images = []
    labels = []

    for name, data in approaches.items():
        if "error" in data:
            continue

        logger.info(f"Generating with {name}...")
        emb = data["embeddings"]

        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        result = pipe(
            prompt_embeds=emb.to("cuda", torch.bfloat16),
            num_inference_steps=args.steps,
            generator=generator,
        )
        img = result.images[0] if hasattr(result, 'images') else result

        # Save individual image
        img.save(args.output_dir / f"{name}.png")
        images.append(img)
        labels.append(data["label"])

        # Record results
        results["approaches"][name] = {
            "embedding_shape": list(emb.shape),
            "embedding_std": float(emb.std()),
            "image_path": f"{name}.png",
        }

        logger.info(f"  Saved {name}.png")

    # Also save reference image in grid
    reference_resized = reference_image.resize((1024, 1024))
    images.insert(0, reference_resized)
    labels.insert(0, "Reference\nImage")

    # Create comparison grid
    grid = create_comparison_grid(
        images, labels,
        title=f'Bridge Approaches: "{args.prompt}" (alpha={args.alpha})'
    )
    grid.save(args.output_dir / "comparison_grid.png")
    logger.info(f"Saved comparison grid to {args.output_dir / 'comparison_grid.png'}")

    # Cleanup
    del pipe
    torch.cuda.empty_cache()

    # Save results JSON
    with open(args.output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {args.output_dir}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
