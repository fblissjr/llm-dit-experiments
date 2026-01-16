#!/usr/bin/env python3
"""
LTX-2 Entropy-Guided Encoding Experiment

Last Updated: 2026-01-15

Zero-training technique: Downweight high-entropy (uncertain) token positions.

Concept:
    hidden, attention_maps = encoder(prompt, output_attentions=True)
    entropy = -sum(attn * log(attn))  # High entropy = model uncertain
    confidence = 1 / (entropy + 1)
    return hidden * confidence.unsqueeze(-1)

Hypothesis: DiT attention gets diluted by uninformative tokens.
High entropy positions represent model uncertainty - downweighting them
may improve signal quality to the DiT.

Usage:
    uv run python experiments/ltx2/entropy_guided_encoding.py
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import prompts from centralized module
# These match the official LTX-2 prompting guide format (100+ words, dialogue, etc.)
from experiments.ltx2.prompts import CATEGORY_PROMPTS

TEST_PROMPTS = [
    CATEGORY_PROMPTS["animal"],   # Animal motion (replaces "dog playing")
    CATEGORY_PROMPTS["nature"],   # Natural scene (replaces "mountain landscape")
    CATEGORY_PROMPTS["human"],    # Human activity (replaces "cooking")
]


def compute_frame_statistics(frames: list) -> dict:
    """Compute statistics on generated frames."""
    frame_arrays = [np.array(f) for f in frames]
    stacked = np.stack(frame_arrays, axis=0)  # [T, H, W, C]

    return {
        "mean": float(stacked.mean()),
        "std": float(stacked.std()),
        "min": float(stacked.min()),
        "max": float(stacked.max()),
        "temporal_variance": float(stacked.var(axis=0).mean()),
    }


def run_entropy_experiment(
    output_dir: str = "experiments/outputs/entropy_encoding",
    model_path: str = "models/LTX-2",
    save_videos: bool = True,
    entropy_weights: list = None,
):
    """
    Run entropy-guided encoding experiment.

    Tests whether downweighting high-entropy (uncertain) token positions
    improves generation quality.

    Args:
        output_dir: Output directory for results
        model_path: Path to LTX-2 model
        save_videos: Whether to save output videos
        entropy_weights: Weight values for entropy-based scaling.
                        0.0 = no weighting (baseline)
                        1.0 = full confidence weighting
    """
    from diffusers import LTX2Pipeline
    from diffusers.utils import export_to_video

    if entropy_weights is None:
        entropy_weights = [0.0, 0.5, 1.0]

    print("=" * 60)
    print("LTX-2 Entropy-Guided Encoding Experiment")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    print("\nLoading pipeline...")
    pipe = LTX2Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_sequential_cpu_offload()

    results = {}

    for weight in entropy_weights:
        print(f"\n{'='*50}")
        print(f"Entropy Weight: {weight}")
        print("=" * 50)

        weight_results = []

        for i, prompt in enumerate(TEST_PROMPTS):
            print(f"\n  [{i+1}/{len(TEST_PROMPTS)}] {prompt[:50]}...")

            start_time = time.time()

            try:
                # Hook to apply entropy weighting
                original_encode = pipe.encode_prompt

                def entropy_weighted_encode(*args, **kwargs):
                    result = original_encode(*args, **kwargs)
                    # result is tuple: (prompt_embeds, attention_mask, neg_embeds, neg_mask)
                    prompt_embeds = result[0]

                    if weight > 0.0:
                        # Compute token-wise entropy from embeddings
                        # (proxy for attention entropy since we can't access internal attention)
                        # Use embedding variance per position as uncertainty proxy
                        emb_variance = prompt_embeds.var(dim=-1)  # [B, T]

                        # Normalize to [0, 1]
                        emb_var_norm = (emb_variance - emb_variance.min()) / (
                            emb_variance.max() - emb_variance.min() + 1e-9
                        )

                        # Confidence = inverse of variance (high variance = uncertain)
                        confidence = 1.0 - emb_var_norm  # [B, T]

                        # Blend between no weighting and full weighting
                        scaling = 1.0 - weight * (1.0 - confidence)  # [B, T]
                        scaling = scaling.unsqueeze(-1)  # [B, T, 1]

                        # Apply weighting
                        weighted_embeds = prompt_embeds * scaling

                        return (weighted_embeds, result[1], result[2], result[3])

                    return result

                pipe.encode_prompt = entropy_weighted_encode

                generator = torch.Generator(device="cpu").manual_seed(42 + i)

                output = pipe(
                    prompt=prompt,
                    height=512,
                    width=768,
                    num_frames=33,
                    num_inference_steps=25,
                    guidance_scale=3.0,
                    generator=generator,
                )

                frames = output.frames[0]
                gen_time = time.time() - start_time

                # Restore original encode
                pipe.encode_prompt = original_encode

                # Compute statistics
                stats = compute_frame_statistics(frames)
                stats["generation_time"] = gen_time
                stats["prompt"] = prompt
                weight_results.append(stats)

                print(f"  Time: {gen_time:.1f}s | Mean: {stats['mean']:.1f} | Std: {stats['std']:.1f}")

                # Save video
                if save_videos:
                    video_path = output_path / f"entropy{weight:.1f}_sample{i}.mp4"
                    export_to_video(frames, str(video_path), fps=24)
                    print(f"  Saved: {video_path.name}")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                weight_results.append({"error": str(e), "prompt": prompt})
                pipe.encode_prompt = original_encode

            # Memory cleanup
            gc.collect()
            torch.cuda.empty_cache()

        results[f"weight_{weight}"] = weight_results

    # Analysis
    print("\n" + "=" * 60)
    print("ENTROPY ENCODING ANALYSIS")
    print("=" * 60)

    baseline_stats = results.get("weight_0.0", [])
    if baseline_stats and "mean" in baseline_stats[0]:
        baseline_mean = np.mean([s["mean"] for s in baseline_stats if "mean" in s])
        baseline_std = np.mean([s["std"] for s in baseline_stats if "std" in s])

        print(f"\nBaseline (weight=0): mean={baseline_mean:.1f}, std={baseline_std:.1f}")

        for weight_key, weight_results in results.items():
            if weight_key == "weight_0.0":
                continue

            valid_results = [s for s in weight_results if "mean" in s]
            if valid_results:
                config_mean = np.mean([s["mean"] for s in valid_results])
                config_std = np.mean([s["std"] for s in valid_results])
                mean_diff = (config_mean - baseline_mean) / baseline_mean * 100
                std_diff = (config_std - baseline_std) / baseline_std * 100

                weight_val = float(weight_key.replace("weight_", ""))
                print(f"\n{weight_key}:")
                print(f"  mean={config_mean:.1f} ({mean_diff:+.1f}% vs baseline)")
                print(f"  std={config_std:.1f} ({std_diff:+.1f}% vs baseline)")

    # Save results
    results_file = output_path / "entropy_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Cleanup
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="LTX-2 Entropy-Guided Encoding")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/outputs/entropy_encoding",
        help="Output directory",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0],
        help="Entropy weighting values to test",
    )
    parser.add_argument(
        "--no-save-videos",
        action="store_true",
        help="Don't save output videos",
    )
    args = parser.parse_args()

    run_entropy_experiment(
        output_dir=args.output_dir,
        entropy_weights=args.weights,
        save_videos=not args.no_save_videos,
    )


if __name__ == "__main__":
    main()
