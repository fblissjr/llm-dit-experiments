#!/usr/bin/env python3
"""
Quick experiment: Test different hidden layers and outlier masking on text-to-image.

Last updated: 2025-12-18

Tests:
1. Layer -2 (default) - baseline
2. Layer -6 (our "clean" layer)
3. Layer -1 (collapsed outliers)
4. Layer -2 + mask dims 4, 396, 100

Usage:
    uv run experiments/test_layer_and_masking.py --config config.toml --profile default
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_dit.cli import create_base_parser, load_runtime_config
from llm_dit.startup import PipelineLoader


# Outlier dimensions identified from analysis
OUTLIER_DIMS = [4, 100, 396]


def mask_outlier_dimensions(embeddings: torch.Tensor, dims: list[int]) -> torch.Tensor:
    """Zero out specific dimensions in embeddings."""
    masked = embeddings.clone()
    for dim in dims:
        masked[..., dim] = 0.0
    return masked


def main():
    parser = create_base_parser()
    parser.add_argument("--prompt", default="A tabby cat with green eyes sitting on a red cushion, detailed fur texture, soft lighting")
    parser.add_argument("--output-dir", default="experiments/results/layer_masking_test")
    args = parser.parse_args()

    # Use seed from config or default
    seed = args.seed if hasattr(args, 'seed') and args.seed else 42

    # Load config
    config = load_runtime_config(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    print("Loading pipeline...")
    loader = PipelineLoader(config)
    result = loader.load_pipeline()
    pipe = result.pipeline
    encoder = result.encoder

    device = pipe.device if hasattr(pipe, 'device') else 'cuda'

    # Test configurations
    tests = [
        {"name": "layer_-2_baseline", "layer": -2, "mask": False},
        {"name": "layer_-6", "layer": -6, "mask": False},
        {"name": "layer_-1_collapsed", "layer": -1, "mask": False},
        {"name": "layer_-2_masked", "layer": -2, "mask": True},
        {"name": "layer_-6_masked", "layer": -6, "mask": True},
    ]

    print(f"\nPrompt: {args.prompt}")
    print(f"Seed: {seed}")
    print(f"Output: {output_dir}\n")

    results_log = []

    for test in tests:
        name = test["name"]
        layer = test["layer"]
        mask = test["mask"]

        print(f"Testing {name}...")
        try:
            # Encode with specified layer
            output = encoder.encode(args.prompt, layer_index=layer)
            embeddings = output.embeddings  # This is a list of tensors

            # Pad to single tensor for pipeline
            # embeddings is a list with one tensor of shape (seq_len, 2560)
            embeddings = embeddings[0].unsqueeze(0)  # (1, seq_len, 2560)

            # Optionally mask outliers
            if mask:
                print(f"  Masking dims {OUTLIER_DIMS}")
                embeddings = mask_outlier_dimensions(embeddings, OUTLIER_DIMS)

            # Generate - use cpu generator (pipeline handles device placement)
            generator = torch.Generator(device="cpu").manual_seed(seed)

            image = pipe(
                prompt_embeds=embeddings,
                height=config.height,
                width=config.width,
                num_inference_steps=args.steps,
                guidance_scale=0.0,
                generator=generator,
            ).images[0]

            output_path = output_dir / f"{name}.png"
            image.save(output_path)
            print(f"  Saved: {output_path}")
            results_log.append({"name": name, "status": "success"})

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results_log.append({"name": name, "status": "failed", "error": str(e)})

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    for r in results_log:
        status = "OK" if r["status"] == "success" else "FAIL"
        print(f"  [{status}] {r['name']}")

    print(f"\nImages saved to: {output_dir}")
    print("\nNext: Compare these visually to see if masking or layer changes help prompt adherence.")


if __name__ == "__main__":
    main()
