#!/usr/bin/env python3
"""
Test AdaIN-style blending for VL conditioning.

AdaIN transfers VL statistics (mean/std) to text embeddings while preserving
text content structure. Unlike style delta, this doesn't add VL values directly.

Usage:
    uv run experiments/qwen3_vl/scripts/test_adain.py \
        --image experiments/inputs/purple_hexagon.png \
        --prompt "Homer Simpson" \
        --alphas 0.3 0.5 0.7 1.0 \
        --output-dir experiments/results/adain_test
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm_dit.vl import VLEmbeddingExtractor, blend_adain, blend_adain_per_dim
from llm_dit.startup import PipelineLoader
from llm_dit.cli import load_runtime_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test AdaIN VL conditioning")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to style image")
    parser.add_argument("--prompt", "-p", type=str, required=True, help="Text prompt")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7, 1.0])
    parser.add_argument("--output-dir", "-o", type=str, default="experiments/results/adain_test")
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--vl-model-path", type=str, default="/home/fbliss/Storage/Qwen3-VL-4B-Instruct")
    parser.add_argument("--hidden-layer", type=int, default=-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", nargs="+", default=["per_token", "per_dim", "global"],
                        choices=["per_token", "per_dim", "global"])

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load style image
    style_image = Image.open(args.image).convert("RGB")
    logger.info(f"Loaded style image: {args.image} ({style_image.size})")

    # Save reference
    style_image.save(output_dir / "reference_style.png")

    # Load VL extractor
    logger.info("Loading Qwen3-VL...")
    vl_extractor = VLEmbeddingExtractor.from_pretrained(
        args.vl_model_path,
        device="cpu",
        torch_dtype=torch.bfloat16,
    )

    # Extract VL embeddings
    logger.info(f"Extracting VL embeddings (layer {args.hidden_layer})...")
    vl_result = vl_extractor.extract(
        style_image,
        text=args.prompt,
        hidden_layer=args.hidden_layer,
        text_tokens_only=False,
        scale_to_text=True,
    )
    vl_emb = vl_result.embeddings
    logger.info(f"  VL embeddings: shape={vl_emb.shape}, std={vl_emb.std():.2f}")

    # Unload VL
    vl_extractor.unload()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load Z-Image pipeline
    logger.info("Loading Z-Image pipeline...")

    class ConfigArgs:
        config = args.config
        profile = "default"

    config = load_runtime_config(ConfigArgs())
    loader = PipelineLoader(config)
    result = loader.load_pipeline()
    pipe = result.pipeline

    # Get text embeddings
    logger.info(f"Encoding text prompt: {args.prompt}")
    text_output = pipe.encoder.encode(args.prompt)
    text_emb = text_output.embeddings[0]
    logger.info(f"  Text embeddings: shape={text_emb.shape}, std={text_emb.std():.2f}")

    # Generate baseline
    logger.info("Generating baseline (no VL)...")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    baseline = pipe(
        prompt_embeds=text_emb,
        height=config.height,
        width=config.width,
        num_inference_steps=config.steps,
        guidance_scale=config.guidance_scale,
        generator=generator,
    )
    baseline.save(output_dir / "baseline_no_vl.png")

    # Test each mode
    for mode in args.modes:
        logger.info(f"\n=== Testing AdaIN mode: {mode} ===")

        for alpha in args.alphas:
            logger.info(f"  Alpha={alpha}...")
            generator = torch.Generator(device="cpu").manual_seed(args.seed)

            # Apply AdaIN
            if mode == "per_token":
                styled_emb = blend_adain(text_emb, vl_emb, alpha=alpha, per_token=True)
            elif mode == "global":
                styled_emb = blend_adain(text_emb, vl_emb, alpha=alpha, per_token=False)
            elif mode == "per_dim":
                styled_emb = blend_adain_per_dim(text_emb, vl_emb, alpha=alpha)

            logger.info(f"    Styled std: {styled_emb.std():.2f}")

            # Generate
            image = pipe(
                prompt_embeds=styled_emb,
                height=config.height,
                width=config.width,
                num_inference_steps=config.steps,
                guidance_scale=config.guidance_scale,
                generator=generator,
            )

            output_path = output_dir / f"adain_{mode}_alpha_{alpha}.png"
            image.save(output_path)
            logger.info(f"    Saved: {output_path}")

    logger.info(f"\nDone! Results in {output_dir}/")


if __name__ == "__main__":
    main()
