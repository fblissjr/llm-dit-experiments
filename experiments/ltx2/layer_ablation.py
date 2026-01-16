#!/usr/bin/env python3
"""
LTX-2 Layer Ablation Study (Experiment 4.5 + 4.1)

Last Updated: 2026-01-15

Tests the impact of excluding layers on generation quality.
Primary goal: Validate that Layer 47 self-attenuates (no quality loss when excluded).
Secondary goal: Find minimum layer count for acceptable quality.

Usage:
    uv run python experiments/ltx2/layer_ablation.py
    uv run python experiments/ltx2/layer_ablation.py --num-samples 5
"""

import argparse
import gc
import time
from pathlib import Path

import torch
import numpy as np


# Import prompts from centralized module
# These match the official LTX-2 prompting guide format (100+ words, dialogue, etc.)
from experiments.ltx2.prompts import CATEGORY_PROMPTS

# Test prompts covering different categories
# Uses properly formatted prompts instead of short out-of-distribution ones
TEST_PROMPTS = [
    CATEGORY_PROMPTS["animal"],   # Animal motion
    CATEGORY_PROMPTS["nature"],   # Natural scene (replaces ocean waves)
    CATEGORY_PROMPTS["human"],    # Human activity (replaces laptop typing)
]


def create_layer_subset_projection(
    original_proj_weight: torch.Tensor,
    layer_indices: list,
    hidden_dim: int = 3840,
    total_layers: int = 48,
) -> torch.Tensor:
    """
    Create a projection matrix for a subset of layers.

    The original projection is [3840, 188160] where 188160 = 49 * 3840.
    We need to extract only the columns corresponding to selected layers.
    """
    # Original weight shape: [out_dim, in_dim] = [3840, 49*3840]
    out_dim = original_proj_weight.shape[0]

    # Extract columns for selected layers
    selected_cols = []
    for layer_idx in layer_indices:
        start = layer_idx * hidden_dim
        end = (layer_idx + 1) * hidden_dim
        selected_cols.append(original_proj_weight[:, start:end])

    # Concatenate selected layer projections
    subset_weight = torch.cat(selected_cols, dim=1)

    return subset_weight


def create_ablated_pipeline(
    model_path: str,
    layer_indices: list,
    hidden_dim: int = 3840,
    total_layers: int = 49,
):
    """
    Create a pipeline with ablated layer contributions.

    CRITICAL: Must modify weights BEFORE enable_sequential_cpu_offload()
    because offload uses 'meta' device tensors that discard direct modifications.
    """
    from diffusers import LTX2Pipeline

    # Load pipeline WITHOUT offload first
    pipe = LTX2Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )

    # Modify projection weights BEFORE enabling offload
    original_weight = pipe.connectors.text_proj_in.weight.data.clone()

    all_layers = set(range(total_layers))
    excluded_layers = all_layers - set(layer_indices)

    # Zero out excluded layer columns
    for layer_idx in excluded_layers:
        start = layer_idx * hidden_dim
        end = (layer_idx + 1) * hidden_dim
        original_weight[:, start:end] = 0

    # Replace weight with modified version
    pipe.connectors.text_proj_in.weight.data = original_weight

    # NOW enable offload (uses modified weights)
    pipe.enable_sequential_cpu_offload()

    return pipe


def generate_with_layer_subset(
    pipe,
    prompt: str,
    layer_indices: list,  # Not used when pipe is already ablated
    seed: int = 42,
    num_frames: int = 33,
    height: int = 512,
    width: int = 768,
    num_inference_steps: int = 25,
    guidance_scale: float = 3.0,
):
    """
    Generate video using the provided (potentially ablated) pipeline.

    Note: layer_indices parameter kept for API compatibility but the pipe
    should already have ablated weights from create_ablated_pipeline().
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)

    output = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    frames = output.frames[0]

    return frames


def compute_frame_statistics(frames: list) -> dict:
    """Compute statistics on generated frames."""
    # Convert PIL images to numpy
    frame_arrays = [np.array(f) for f in frames]
    stacked = np.stack(frame_arrays, axis=0)  # [T, H, W, C]

    return {
        "mean": stacked.mean(),
        "std": stacked.std(),
        "min": stacked.min(),
        "max": stacked.max(),
        "temporal_variance": stacked.var(axis=0).mean(),  # Variance over time
    }


def run_ablation(
    prompts: list,
    num_samples: int = 3,
    output_dir: str = "experiments/outputs/layer_ablation",
    save_videos: bool = True,
    model_path: str = "models/LTX-2",
):
    """Run layer ablation experiments.

    FIXED: Creates fresh pipeline for each config to ensure weight modifications
    take effect BEFORE enable_sequential_cpu_offload() is called.
    """
    from diffusers.utils import export_to_video

    print("=" * 60)
    print("LTX-2 Layer Ablation Study (CORRECTED)")
    print("=" * 60)

    # Define layer configurations to test
    # Layer indices are 0-48 (49 total: 48 decoder + 1 embedding)
    configs = {
        "all_49": list(range(49)),                    # Baseline (all layers)
        "exclude_L48": list(range(48)),               # Exclude final layer (L48/index 48)
        "exclude_L47_48": list(range(47)),            # Exclude last two
        "even_layers": list(range(0, 49, 2)),         # Every other layer (25 layers)
        "late_only": list(range(32, 49)),             # Late layers only (17 layers)
        "early_mid_late": [0, 8, 16, 24, 32, 40, 47], # Representative sample (7 layers)
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    for config_name, layer_indices in configs.items():
        print(f"\n{'='*40}")
        print(f"Config: {config_name} ({len(layer_indices)} layers)")
        print(f"Layers: {layer_indices[:5]}...{layer_indices[-3:]}" if len(layer_indices) > 8 else f"Layers: {layer_indices}")
        print("="*40)

        # Create fresh pipeline with ablated weights for this config
        # CRITICAL: Must do this per-config because weight mods must happen
        # BEFORE enable_sequential_cpu_offload()
        print(f"  Loading pipeline with {len(layer_indices)} active layers...")
        pipe = create_ablated_pipeline(model_path, layer_indices)

        config_results = []

        for i, prompt in enumerate(prompts[:num_samples]):
            print(f"\n  [{i+1}/{min(len(prompts), num_samples)}] {prompt[:40]}...")

            start_time = time.time()

            try:
                frames = generate_with_layer_subset(
                    pipe,
                    prompt=prompt,
                    layer_indices=layer_indices,
                    seed=42 + i,  # Different seed per prompt, same across configs
                    num_frames=33,
                    height=512,
                    width=768,
                    num_inference_steps=25,
                    guidance_scale=3.0,
                )

                gen_time = time.time() - start_time

                # Compute statistics
                stats = compute_frame_statistics(frames)
                stats["generation_time"] = gen_time
                stats["prompt"] = prompt
                config_results.append(stats)

                print(f"    Time: {gen_time:.1f}s, Mean: {stats['mean']:.1f}, Std: {stats['std']:.1f}")

                # Save video
                if save_videos:
                    video_path = output_path / f"{config_name}_sample{i}.mp4"
                    export_to_video(frames, str(video_path), fps=24)
                    print(f"    Saved: {video_path.name}")

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
                config_results.append({"error": str(e), "prompt": prompt})

            # Memory cleanup
            gc.collect()
            torch.cuda.empty_cache()

        results[config_name] = config_results

        # Cleanup pipeline before loading next config
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)

    baseline_stats = results.get("all_49", [])
    if baseline_stats and "mean" in baseline_stats[0]:
        baseline_mean = np.mean([s["mean"] for s in baseline_stats if "mean" in s])
        baseline_std = np.mean([s["std"] for s in baseline_stats if "std" in s])

        print(f"\nBaseline (all_49): mean={baseline_mean:.1f}, std={baseline_std:.1f}")

        for config_name, config_results in results.items():
            if config_name == "all_49":
                continue

            valid_results = [s for s in config_results if "mean" in s]
            if valid_results:
                config_mean = np.mean([s["mean"] for s in valid_results])
                config_std = np.mean([s["std"] for s in valid_results])
                mean_diff = abs(config_mean - baseline_mean) / baseline_mean * 100
                std_diff = abs(config_std - baseline_std) / baseline_std * 100

                num_layers = len(configs[config_name])
                savings = (49 - num_layers) / 49 * 100

                print(f"\n{config_name} ({num_layers} layers, {savings:.0f}% reduction):")
                print(f"  mean={config_mean:.1f} ({mean_diff:.1f}% diff)")
                print(f"  std={config_std:.1f} ({std_diff:.1f}% diff)")

    # Save results
    results_file = output_path / "ablation_results.npz"
    np.savez(results_file, **{k: str(v) for k, v in results.items()})
    print(f"\nResults saved to {results_file}")

    # Final cleanup (pipe already deleted per-config in loop)
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="LTX-2 Layer Ablation Study")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples per configuration",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/outputs/layer_ablation",
        help="Output directory",
    )
    parser.add_argument(
        "--no-save-videos",
        action="store_true",
        help="Don't save output videos",
    )
    args = parser.parse_args()

    run_ablation(
        prompts=TEST_PROMPTS,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        save_videos=not args.no_save_videos,
    )


if __name__ == "__main__":
    main()
