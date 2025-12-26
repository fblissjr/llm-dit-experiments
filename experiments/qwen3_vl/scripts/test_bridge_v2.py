#!/usr/bin/env python3
"""
Test novel bridge approaches v2: Lower alpha Gram + Timestep-dependent + Multi-layer.

Last updated: 2025-12-14

New approaches based on Gram Matrix finding (the only artifact-free approach):
1. Lower alpha Gram (0.05, 0.1, 0.15) - May preserve content while avoiding artifacts
2. Timestep-dependent blending - High VL early (structure), low VL late (details)
3. Multi-layer extraction - Combine different layers for different information

Usage:
    uv run experiments/qwen3_vl/scripts/test_bridge_v2.py \
        --reference experiments/inputs/style_anime_girl.png \
        --prompt "Homer Simpson eating a donut"
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Callable

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

# Model paths from environment variables
ZIMAGE_PATH = os.environ.get("ZIMAGE_PATH")
QWEN3_4B_PATH = os.environ.get("QWEN3_PATH")
QWEN3_VL_PATH = os.environ.get("QWEN3_VL_PATH")

if not all([ZIMAGE_PATH, QWEN3_4B_PATH, QWEN3_VL_PATH]):
    raise ValueError(
        "Set environment variables: ZIMAGE_PATH, QWEN3_PATH, QWEN3_VL_PATH"
    )


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """Compute Gram matrix for style transfer."""
    n_elements = features.shape[0]
    gram = torch.mm(features.T, features) / n_elements
    return gram


def apply_gram_style(
    content_emb: torch.Tensor,
    style_gram: torch.Tensor,
    alpha: float = 0.3,
    iterations: int = 100,
) -> torch.Tensor:
    """Apply style from Gram matrix to content embeddings."""
    styled = content_emb.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([styled], lr=0.1)

    for i in range(iterations):
        optimizer.zero_grad()

        # Content loss
        content_loss = F.mse_loss(styled, content_emb.detach())

        # Style loss
        current_gram = gram_matrix(styled)
        style_loss = F.mse_loss(current_gram, style_gram)

        # Combined
        loss = (1 - alpha) * content_loss + alpha * style_loss
        loss.backward()
        optimizer.step()

    return styled.detach()


def gram_style_transfer(
    text_emb: torch.Tensor,
    vl_emb: torch.Tensor,
    alpha: float = 0.3,
) -> torch.Tensor:
    """Transfer style from VL embeddings to text using Gram matrices."""
    style_gram = gram_matrix(vl_emb)
    return apply_gram_style(text_emb, style_gram, alpha=alpha)


def linear_blend(
    text_emb: torch.Tensor,
    vl_emb: torch.Tensor,
    alpha: float = 0.3,
) -> torch.Tensor:
    """Standard linear interpolation."""
    min_len = min(len(text_emb), len(vl_emb))
    return (1 - alpha) * text_emb[:min_len] + alpha * vl_emb[:min_len]


def create_comparison_grid(
    images: list[Image.Image],
    labels: list[str],
    title: str = "",
    cols: int = 4,
) -> Image.Image:
    """Create a comparison grid from multiple images."""
    from PIL import ImageDraw, ImageFont

    n_images = len(images)
    img_width, img_height = images[0].size

    cols = min(n_images, cols)
    rows = (n_images + cols - 1) // cols

    label_height = 70
    title_height = 80 if title else 0

    grid_width = cols * img_width
    grid_height = rows * (img_height + label_height) + title_height

    grid = Image.new("RGB", (grid_width, grid_height), "white")
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
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

        label_y = y + img_height + 8
        for line in label.split('\n'):
            draw.text((x + img_width // 2, label_y), line, fill="black", font=font, anchor="mt")
            label_y += 20

    return grid


class TimestepBlendingCallback:
    """Callback to modify embeddings based on timestep during generation."""

    def __init__(
        self,
        text_emb: torch.Tensor,
        vl_emb: torch.Tensor,
        schedule: str = "linear_decay",
        max_alpha: float = 0.5,
        min_alpha: float = 0.0,
    ):
        """
        Args:
            text_emb: Pure text embeddings
            vl_emb: VL embeddings
            schedule: "linear_decay", "cosine_decay", "step"
            max_alpha: VL influence at start (t=1.0)
            min_alpha: VL influence at end (t=0.0)
        """
        self.text_emb = text_emb
        self.vl_emb = vl_emb
        self.schedule = schedule
        self.max_alpha = max_alpha
        self.min_alpha = min_alpha
        self.min_len = min(len(text_emb), len(vl_emb))

    def get_alpha(self, timestep: float, num_steps: int, current_step: int) -> float:
        """Get alpha value for current timestep."""
        # Normalize step progress (0=start, 1=end)
        progress = current_step / (num_steps - 1) if num_steps > 1 else 1.0

        if self.schedule == "linear_decay":
            # High VL at start, linear decay to low VL at end
            alpha = self.max_alpha * (1 - progress) + self.min_alpha * progress
        elif self.schedule == "cosine_decay":
            # Smoother decay using cosine
            import math
            alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * (1 + math.cos(progress * math.pi)) / 2
        elif self.schedule == "step":
            # High VL for first half, low VL for second half
            alpha = self.max_alpha if progress < 0.5 else self.min_alpha
        else:
            alpha = (self.max_alpha + self.min_alpha) / 2

        return alpha

    def get_embeddings(self, timestep: float, num_steps: int, current_step: int) -> torch.Tensor:
        """Get blended embeddings for current timestep."""
        alpha = self.get_alpha(timestep, num_steps, current_step)
        blended = (1 - alpha) * self.text_emb[:self.min_len] + alpha * self.vl_emb[:self.min_len]
        return blended


def generate_with_timestep_blending(
    pipe,
    text_emb: torch.Tensor,
    vl_emb: torch.Tensor,
    schedule: str,
    max_alpha: float,
    min_alpha: float,
    num_steps: int,
    seed: int,
) -> Image.Image:
    """Generate image with timestep-dependent VL blending."""
    from llm_dit.schedulers.flow_match import FlowMatchScheduler

    # Create blending callback
    callback = TimestepBlendingCallback(
        text_emb, vl_emb,
        schedule=schedule,
        max_alpha=max_alpha,
        min_alpha=min_alpha,
    )

    # Manual generation loop with blending
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(
        (1, 16, 128, 128),  # Z-Image latent shape for 1024x1024
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )

    # Get scheduler timesteps
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps

    for i, t in enumerate(timesteps):
        # Get timestep-specific embeddings
        emb = callback.get_embeddings(t.item(), num_steps, i)
        emb = emb.to("cuda", torch.bfloat16).unsqueeze(0)  # Add batch dim

        # Predict noise
        with torch.no_grad():
            noise_pred = pipe.transformer(
                latents,
                encoder_hidden_states=emb,
                timestep=t.unsqueeze(0).to("cuda"),
            )
            if hasattr(noise_pred, 'sample'):
                noise_pred = noise_pred.sample

        # Step
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Decode
    with torch.no_grad():
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype("uint8")

    return Image.fromarray(image)


def main():
    parser = argparse.ArgumentParser(description="Test bridge approaches v2")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="Homer Simpson eating a donut")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/bridge_v2"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--vl-layer", type=int, default=-6)
    parser.add_argument(
        "--test",
        choices=["gram_sweep", "timestep", "multilayer", "all"],
        default="gram_sweep",
        help="Which test to run"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference_image = Image.open(args.reference).convert("RGB")
    logger.info(f"Loaded reference: {args.reference}")

    # =========================================================================
    # Phase 1: Extract embeddings
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("Extracting embeddings...")
    logger.info("="*60)

    from llm_dit.backends.transformers import TransformersBackend
    from llm_dit.vl import VLEmbeddingExtractor

    # Text embeddings
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
    logger.info(f"Text: {text_emb.shape}, std={text_emb.std():.2f}")
    del text_backend
    torch.cuda.empty_cache()

    # VL embeddings (main layer)
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
        text_tokens_only=False,
        scale_to_text=True,
    )
    vl_emb = vl_result.embeddings.cpu().float()
    logger.info(f"VL (layer {args.vl_layer}): {vl_emb.shape}, std={vl_emb.std():.2f}")

    # Multi-layer extraction if needed
    vl_layers = {}
    if args.test in ["multilayer", "all"]:
        for layer in [-2, -6, -8, -15]:
            result = vl_extractor.extract(
                reference_image,
                text=args.prompt,
                hidden_layer=layer,
                text_tokens_only=False,
                scale_to_text=True,
            )
            vl_layers[layer] = result.embeddings.cpu().float()
            logger.info(f"  Layer {layer}: std={vl_layers[layer].std():.2f}")

    vl_extractor.unload()
    del vl_extractor
    torch.cuda.empty_cache()

    # =========================================================================
    # Phase 2: Load Z-Image pipeline
    # =========================================================================
    logger.info("\nLoading Z-Image pipeline...")
    from llm_dit.pipelines.z_image import ZImagePipeline

    pipe = ZImagePipeline.from_pretrained(
        ZIMAGE_PATH,
        torch_dtype=torch.bfloat16,
        text_encoder_device="cpu",
        dit_device="cuda",
        vae_device="cuda",
    )

    results = {"metadata": {"prompt": args.prompt, "seed": args.seed}, "tests": {}}
    all_images = []
    all_labels = []

    # =========================================================================
    # Test 1: Gram Matrix Alpha Sweep
    # =========================================================================
    if args.test in ["gram_sweep", "all"]:
        logger.info("\n" + "="*60)
        logger.info("Test 1: Gram Matrix Alpha Sweep")
        logger.info("="*60)

        # Baseline: text only
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        result = pipe(
            prompt_embeds=text_emb.to("cuda", torch.bfloat16),
            num_inference_steps=args.steps,
            generator=generator,
        )
        img = result.images[0] if hasattr(result, 'images') else result
        img.save(args.output_dir / "gram_text_only.png")
        all_images.append(img)
        all_labels.append("Text Only\n(Baseline)")

        # Test multiple alpha values
        alphas = [0.05, 0.1, 0.15, 0.2, 0.3]
        for alpha in alphas:
            logger.info(f"  Gram alpha={alpha}...")
            gram_emb = gram_style_transfer(text_emb, vl_emb, alpha=alpha)
            logger.info(f"    std={gram_emb.std():.2f}")

            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            result = pipe(
                prompt_embeds=gram_emb.to("cuda", torch.bfloat16),
                num_inference_steps=args.steps,
                generator=generator,
            )
            img = result.images[0] if hasattr(result, 'images') else result
            img.save(args.output_dir / f"gram_alpha_{alpha}.png")
            all_images.append(img)
            all_labels.append(f"Gram\nalpha={alpha}")

            results["tests"][f"gram_{alpha}"] = {"std": float(gram_emb.std())}

        # Create grid for gram sweep
        ref_resized = reference_image.resize((1024, 1024))
        gram_images = [ref_resized] + all_images
        gram_labels = ["Reference"] + all_labels
        grid = create_comparison_grid(
            gram_images, gram_labels,
            title=f'Gram Alpha Sweep: "{args.prompt}"',
            cols=4,
        )
        grid.save(args.output_dir / "gram_sweep_grid.png")
        logger.info(f"Saved gram_sweep_grid.png")

    # =========================================================================
    # Test 2: Timestep-Dependent Blending
    # =========================================================================
    if args.test in ["timestep", "all"]:
        logger.info("\n" + "="*60)
        logger.info("Test 2: Timestep-Dependent Blending")
        logger.info("="*60)

        timestep_images = []
        timestep_labels = []

        # Baseline
        if not all_images:  # If we haven't done gram test
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            result = pipe(
                prompt_embeds=text_emb.to("cuda", torch.bfloat16),
                num_inference_steps=args.steps,
                generator=generator,
            )
            img = result.images[0] if hasattr(result, 'images') else result
            timestep_images.append(img)
            timestep_labels.append("Text Only\n(Baseline)")

        # Different schedules
        schedules = [
            ("linear_decay", 0.5, 0.0, "Linear 0.5->0"),
            ("linear_decay", 0.3, 0.0, "Linear 0.3->0"),
            ("cosine_decay", 0.5, 0.0, "Cosine 0.5->0"),
            ("step", 0.4, 0.0, "Step 0.4/0"),
        ]

        for schedule, max_a, min_a, label in schedules:
            logger.info(f"  {label}...")
            try:
                img = generate_with_timestep_blending(
                    pipe, text_emb, vl_emb,
                    schedule=schedule,
                    max_alpha=max_a,
                    min_alpha=min_a,
                    num_steps=args.steps,
                    seed=args.seed,
                )
                img.save(args.output_dir / f"timestep_{schedule}_{max_a}_{min_a}.png")
                timestep_images.append(img)
                timestep_labels.append(f"Timestep\n{label}")
            except Exception as e:
                logger.error(f"    Failed: {e}")

        if timestep_images:
            ref_resized = reference_image.resize((1024, 1024))
            ts_grid = create_comparison_grid(
                [ref_resized] + timestep_images,
                ["Reference"] + timestep_labels,
                title=f'Timestep Blending: "{args.prompt}"',
                cols=3,
            )
            ts_grid.save(args.output_dir / "timestep_grid.png")
            logger.info("Saved timestep_grid.png")

    # =========================================================================
    # Test 3: Multi-Layer Blending
    # =========================================================================
    if args.test in ["multilayer", "all"] and vl_layers:
        logger.info("\n" + "="*60)
        logger.info("Test 3: Multi-Layer Blending")
        logger.info("="*60)

        ml_images = []
        ml_labels = []

        # Baseline
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        result = pipe(
            prompt_embeds=text_emb.to("cuda", torch.bfloat16),
            num_inference_steps=args.steps,
            generator=generator,
        )
        ml_images.append(result.images[0] if hasattr(result, 'images') else result)
        ml_labels.append("Text Only")

        # Single layer Gram at alpha=0.15 (our best from sweep)
        best_alpha = 0.15
        for layer in [-2, -6, -8, -15]:
            if layer in vl_layers:
                logger.info(f"  Gram layer {layer} alpha={best_alpha}...")
                gram_emb = gram_style_transfer(text_emb, vl_layers[layer], alpha=best_alpha)

                generator = torch.Generator(device="cpu").manual_seed(args.seed)
                result = pipe(
                    prompt_embeds=gram_emb.to("cuda", torch.bfloat16),
                    num_inference_steps=args.steps,
                    generator=generator,
                )
                ml_images.append(result.images[0] if hasattr(result, 'images') else result)
                ml_labels.append(f"Gram L{layer}\na={best_alpha}")

        # Multi-layer blend: average Gram matrices from multiple layers
        if len(vl_layers) >= 2:
            logger.info("  Multi-layer Gram blend...")

            # Average Gram matrices from different layers
            gram_matrices = [gram_matrix(vl_layers[l]) for l in vl_layers.keys()]
            avg_gram = torch.stack(gram_matrices).mean(dim=0)

            ml_gram_emb = apply_gram_style(text_emb, avg_gram, alpha=best_alpha)

            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            result = pipe(
                prompt_embeds=ml_gram_emb.to("cuda", torch.bfloat16),
                num_inference_steps=args.steps,
                generator=generator,
            )
            ml_images.append(result.images[0] if hasattr(result, 'images') else result)
            ml_labels.append(f"Multi-Layer\nGram Avg")

        ref_resized = reference_image.resize((1024, 1024))
        ml_grid = create_comparison_grid(
            [ref_resized] + ml_images,
            ["Reference"] + ml_labels,
            title=f'Multi-Layer Comparison: "{args.prompt}"',
            cols=4,
        )
        ml_grid.save(args.output_dir / "multilayer_grid.png")
        logger.info("Saved multilayer_grid.png")

    # Cleanup
    del pipe
    torch.cuda.empty_cache()

    # Save results
    with open(args.output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
