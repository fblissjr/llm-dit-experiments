#!/usr/bin/env python3
"""
Test VL + img2img combined: VL-conditioned embeddings + VAE latent start.

This combines both approaches:
- VL provides style/mood influence through blended embeddings
- img2img provides structural influence through VAE latents

IMPORTANT: uses text_tokens_only=False to include image tokens (required for VL to work)
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm_dit.vl import VLEmbeddingExtractor, blend_embeddings
from llm_dit.startup import PipelineLoader
from llm_dit.cli import load_runtime_config
from experiments.qwen3_vl.scripts.grid_utils import make_grid

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test VL + img2img combined")
    parser.add_argument("--image", "-i", type=str, required=True)
    parser.add_argument("--prompt", "-p", type=str, required=True)
    parser.add_argument("--vl-alphas", type=float, nargs="+", default=[0.0, 0.3, 0.5])
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.7, 0.8, 0.9])
    parser.add_argument("--output-dir", "-o", type=str, default="experiments/results/vl_img2img_test")
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--vl-model-path", type=str, default="/home/fbliss/Storage/Qwen3-VL-4B-Instruct")
    parser.add_argument("--hidden-layer", type=int, default=-6)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    image = Image.open(args.image).convert("RGB")
    logger.info(f"Loaded image: {args.image} ({image.size})")
    image.save(output_dir / "reference.png")

    # Load VL extractor
    logger.info("Loading Qwen3-VL...")
    vl_extractor = VLEmbeddingExtractor.from_pretrained(
        args.vl_model_path, device="cpu", torch_dtype=torch.bfloat16
    )

    # Extract VL embeddings - MUST use text_tokens_only=False to include image info
    logger.info(f"Extracting VL embeddings (layer {args.hidden_layer}, full tokens)...")
    vl_result = vl_extractor.extract(
        image, text=args.prompt, hidden_layer=args.hidden_layer,
        text_tokens_only=False,  # CRITICAL: must be False to include image tokens
        scale_to_text=True
    )
    vl_emb = vl_result.embeddings
    logger.info(f"  VL: shape={vl_emb.shape}, std={vl_emb.std():.2f}")

    vl_extractor.unload()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Load pipeline
    logger.info("Loading Z-Image pipeline...")
    class ConfigArgs:
        config = args.config
        profile = "default"
    config = load_runtime_config(ConfigArgs())
    loader = PipelineLoader(config)
    pipe = loader.load_pipeline().pipeline

    # Get text embeddings
    logger.info(f"Encoding: {args.prompt}")
    text_emb = pipe.encoder.encode(args.prompt).embeddings[0]
    logger.info(f"  Text: shape={text_emb.shape}, std={text_emb.std():.2f}")

    # Generate grid: vl_alpha x strength
    for vl_alpha in args.vl_alphas:
        # Blend embeddings
        if vl_alpha == 0.0:
            blended = text_emb
        else:
            blended = blend_embeddings(vl_emb, text_emb, alpha=vl_alpha)

        for strength in args.strengths:
            logger.info(f"Generating: vl_alpha={vl_alpha}, strength={strength}")

            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            generator.manual_seed(args.seed)

            result = pipe.img2img(
                prompt_embeds=blended,
                image=image,
                strength=strength,
                num_inference_steps=config.steps,
                guidance_scale=config.guidance_scale,
                generator=generator,
            )

            out_path = output_dir / f"a{int(vl_alpha*10)}_s{int(strength*10)}.png"
            result.save(out_path)
            logger.info(f"  Saved: {out_path}")

    # Generate comparison grid
    logger.info("Generating comparison grid...")
    images = []
    labels = []
    for vl_alpha in args.vl_alphas:
        for strength in args.strengths:
            images.append(output_dir / f"a{int(vl_alpha*10)}_s{int(strength*10)}.png")
            labels.append(f"a={vl_alpha} s={strength}")

    grid_path = make_grid(images, labels, len(args.strengths), output_dir / "grid.png")
    logger.info(f"Grid saved: {grid_path}")

    logger.info(f"\nDone! Results in {output_dir}/")
    logger.info(f"Grid: {len(args.vl_alphas)} rows (alpha) x {len(args.strengths)} cols (strength)")


if __name__ == "__main__":
    main()
