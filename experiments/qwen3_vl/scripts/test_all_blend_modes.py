#!/usr/bin/env python3
"""Test all blend modes with style transfer.

Generates a grid comparing: interpolate, adain_per_dim, adain, linear
"""

import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[3]))

from src.llm_dit.vl import (
    VLEmbeddingExtractor,
    blend_interpolate,
    blend_adain,
    blend_adain_per_dim,
    blend_embeddings,
)
from src.llm_dit.startup import PipelineLoader
from src.llm_dit.cli import load_runtime_config
from experiments.qwen3_vl.scripts.grid_utils import make_grid


BLEND_MODES = {
    "interpolate": lambda vl, text, alpha: blend_interpolate(vl, text, alpha),
    "adain_per_dim": lambda vl, text, alpha: blend_adain_per_dim(text, vl, alpha),
    "adain": lambda vl, text, alpha: blend_adain(text, vl, alpha),
    "linear": lambda vl, text, alpha: blend_embeddings(vl, text, alpha),
}

# Test parameters
ALPHA = 0.3
STRENGTH = 0.9
PROMPT = "Homer Simpson"
HIDDEN_LAYER = -6


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", default="experiments/inputs/style_anime_girl.png",
                        help="Reference image for style")
    parser.add_argument("-p", "--prompt", default=PROMPT, help="Subject prompt")
    parser.add_argument("-a", "--alpha", type=float, default=ALPHA, help="VL alpha")
    parser.add_argument("-s", "--strength", type=float, default=STRENGTH, help="img2img strength")
    parser.add_argument("-o", "--output", default="experiments/results/blend_mode_comparison",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reference image
    print(f"Reference: {args.image}")
    reference = Image.open(args.image).convert("RGB")
    reference.save(output_dir / "reference.png")

    # Load VL extractor
    print("Loading Qwen3-VL...")
    vl_extractor = VLEmbeddingExtractor.from_pretrained(
        "/home/fbliss/Storage/Qwen3-VL-4B-Instruct",
        device="cuda",
        torch_dtype=torch.bfloat16,
    )

    # Extract VL embeddings
    print(f"Extracting VL embeddings (layer {HIDDEN_LAYER})...")
    vl_result = vl_extractor.extract(
        reference,
        text=args.prompt,
        hidden_layer=HIDDEN_LAYER,
        text_tokens_only=False,
        scale_to_text=True,
    )
    vl_emb = vl_result.embeddings
    print(f"  VL shape: {vl_emb.shape}, std: {vl_emb.std():.2f}")

    # Unload VL to free VRAM
    vl_extractor.unload()
    torch.cuda.empty_cache()

    # Load Z-Image pipeline
    print("Loading Z-Image pipeline...")
    class ConfigArgs:
        config = "config.toml"
        profile = "default"
    config = load_runtime_config(ConfigArgs())
    loader = PipelineLoader(config)
    pipe = loader.load_pipeline().pipeline

    # Get text embeddings
    print(f"Encoding text: {args.prompt}")
    text_emb = pipe.encoder.encode(args.prompt).embeddings[0]
    print(f"  Text shape: {text_emb.shape}, std: {text_emb.std():.2f}")

    # Generate baseline (no VL)
    print("\n=== Generating baseline (no VL) ===")
    generator = torch.Generator(device="cuda").manual_seed(42)
    baseline = pipe.img2img(
        prompt_embeds=text_emb,
        image=reference,
        strength=args.strength,
        num_inference_steps=9,
        generator=generator,
    )
    baseline_img = baseline.images[0] if hasattr(baseline, 'images') else baseline
    baseline_img.save(output_dir / "baseline_no_vl.png")

    # Test each blend mode
    results = {"reference": output_dir / "reference.png", "no_vl": output_dir / "baseline_no_vl.png"}

    for mode_name, blend_fn in BLEND_MODES.items():
        print(f"\n=== Testing: {mode_name} ===")

        # Blend embeddings
        blended = blend_fn(vl_emb, text_emb, args.alpha)
        print(f"  Blended shape: {blended.shape}, std: {blended.std():.2f}")

        # Generate
        generator = torch.Generator(device="cuda").manual_seed(42)
        result = pipe.img2img(
            prompt_embeds=blended,
            image=reference,
            strength=args.strength,
            num_inference_steps=9,
            generator=generator,
        )

        result_img = result.images[0] if hasattr(result, 'images') else result
        out_path = output_dir / f"{mode_name}.png"
        result_img.save(out_path)
        print(f"  Saved: {out_path}")
        results[mode_name] = out_path

    # Create comparison grid
    print("\n=== Creating comparison grid ===")
    images = [
        results["reference"],
        results["no_vl"],
        results["interpolate"],
        results["adain_per_dim"],
        results["adain"],
        results["linear"],
    ]
    labels = ["Reference", "No VL", "interpolate", "adain_per_dim", "adain", "linear (truncate)"]

    make_grid(images, labels, cols=3, output_path=output_dir / "comparison_grid.png", cell_size=256)

    # Print summary
    print(f"\n=== Summary ===")
    print(f"Reference: {args.image}")
    print(f"Prompt: {args.prompt}")
    print(f"Alpha: {args.alpha}, Strength: {args.strength}")
    print(f"\nResults saved to: {output_dir}")
    print(f"Grid: {output_dir}/comparison_grid.png")
    print(f"\nExpected differences:")
    print(f"  - interpolate: Preserves all VL info via resampling")
    print(f"  - adain_per_dim: Strongest style transfer (per-dim statistics)")
    print(f"  - adain: Moderate style transfer (global statistics)")
    print(f"  - linear: Weakest - truncates to first 11 tokens (loses 99%)")


if __name__ == "__main__":
    main()
