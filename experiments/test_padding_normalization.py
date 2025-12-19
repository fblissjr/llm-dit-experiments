#!/usr/bin/env python3
"""
Test: Does padding prompts to consistent length normalize image generation?

Last updated: 2025-12-18

Hypothesis: Outlier dimensions encode sequence length/attention distribution.
Padding all prompts to the same length should produce more consistent results.

Test:
1. Generate with raw prompt (short)
2. Generate with prompt padded to 128 tokens
3. Generate with prompt padded to 256 tokens
4. Generate with prompt padded to 512 tokens
5. Compare: Do padded versions look more similar to each other?
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_dit.cli import create_base_parser, load_runtime_config
from llm_dit.startup import PipelineLoader

# Qwen3 special tokens
PAD_TOKEN_ID = 151643  # <|endoftext|>


def pad_embeddings_to_length(embeddings: torch.Tensor, target_length: int, pad_value: float = 0.0) -> torch.Tensor:
    """
    Pad embeddings to a target sequence length.

    Args:
        embeddings: Shape (seq_len, hidden_dim) or (batch, seq_len, hidden_dim)
        target_length: Target sequence length
        pad_value: Value to use for padding

    Returns:
        Padded embeddings of shape (..., target_length, hidden_dim)
    """
    if embeddings.dim() == 2:
        seq_len, hidden_dim = embeddings.shape
        if seq_len >= target_length:
            return embeddings[:target_length]

        padding = torch.full(
            (target_length - seq_len, hidden_dim),
            pad_value,
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        return torch.cat([embeddings, padding], dim=0)

    elif embeddings.dim() == 3:
        batch, seq_len, hidden_dim = embeddings.shape
        if seq_len >= target_length:
            return embeddings[:, :target_length, :]

        padding = torch.full(
            (batch, target_length - seq_len, hidden_dim),
            pad_value,
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        return torch.cat([embeddings, padding], dim=1)

    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {embeddings.dim()}D")


def get_outlier_stats(embeddings: torch.Tensor) -> dict:
    """Get stats for outlier dimensions."""
    OUTLIER_DIMS = [4, 100, 396]
    emb_np = embeddings.float().cpu().numpy()

    if emb_np.ndim == 3:
        emb_np = emb_np[0]  # Take first batch

    stats = {"seq_len": emb_np.shape[0]}
    for dim in OUTLIER_DIMS:
        stats[f"dim_{dim}_mean"] = float(emb_np[:, dim].mean())
        stats[f"dim_{dim}_std"] = float(emb_np[:, dim].std())

    return stats


def main():
    parser = create_base_parser()
    parser.add_argument("--prompt", default="A tabby cat with green eyes")
    parser.add_argument("--output-dir", default="experiments/results/padding_test")
    parser.add_argument("--target-lengths", nargs="+", type=int, default=[0, 64, 128, 256, 512])
    args = parser.parse_args()

    seed = args.seed if hasattr(args, 'seed') and args.seed else 42

    config = load_runtime_config(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pipeline...")
    loader = PipelineLoader(config)
    result = loader.load_pipeline()
    pipe = result.pipeline
    encoder = result.encoder

    print(f"\nPrompt: {args.prompt}")
    print(f"Seed: {seed}")
    print(f"Target lengths: {args.target_lengths}")
    print(f"Output: {output_dir}\n")

    results = []

    for target_len in args.target_lengths:
        name = f"len_{target_len}" if target_len > 0 else "raw"
        print(f"\nTesting {name}...")

        try:
            # Encode prompt
            output = encoder.encode(args.prompt, layer_index=-2)
            embeddings = output.embeddings[0]  # Shape: (seq_len, 2560)

            original_len = embeddings.shape[0]
            print(f"  Original length: {original_len} tokens")

            # Pad if requested
            if target_len > 0:
                embeddings = pad_embeddings_to_length(embeddings, target_len)
                print(f"  Padded to: {embeddings.shape[0]} tokens")

            # Get outlier stats
            stats = get_outlier_stats(embeddings)
            print(f"  Dim 4 mean: {stats['dim_4_mean']:.1f}, Dim 396 mean: {stats['dim_396_mean']:.1f}")

            # Add batch dimension
            embeddings = embeddings.unsqueeze(0)

            # Generate
            generator = torch.Generator(device="cpu").manual_seed(seed)

            image = pipe(
                prompt_embeds=embeddings,
                height=config.height,
                width=config.width,
                num_inference_steps=9,
                guidance_scale=0.0,
                generator=generator,
            ).images[0]

            output_path = output_dir / f"{name}.png"
            image.save(output_path)
            print(f"  Saved: {output_path}")

            results.append({
                "name": name,
                "target_len": target_len,
                "actual_len": embeddings.shape[1],
                "original_len": original_len,
                **stats,
                "status": "success",
            })

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({"name": name, "status": "failed", "error": str(e)})

    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Name':<10} {'Length':<8} {'Dim4 Mean':<12} {'Dim396 Mean':<12} {'Status':<8}")
    print("-"*60)

    for r in results:
        if r["status"] == "success":
            print(f"{r['name']:<10} {r['actual_len']:<8} {r['dim_4_mean']:<12.1f} {r['dim_396_mean']:<12.1f} OK")
        else:
            print(f"{r['name']:<10} {'N/A':<8} {'N/A':<12} {'N/A':<12} FAIL")

    print(f"\nImages saved to: {output_dir}")
    print("\nCompare the images:")
    print("- 'raw' = original short prompt")
    print("- 'len_X' = padded to X tokens")
    print("\nIf padding normalizes outliers, padded versions should look similar to each other")
    print("while 'raw' may look different (higher outlier influence).")


if __name__ == "__main__":
    main()
