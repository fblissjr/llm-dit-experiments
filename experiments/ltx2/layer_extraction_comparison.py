#!/usr/bin/env python3
"""
LTX-2 Layer Extraction Comparison Experiment

Last Updated: 2026-01-15

Zero-training technique: Test which Gemma-3 layers matter for which visual attributes.

Hypothesis from LLM research:
- Early layers (0-16): Phonetic/syntactic information
- Middle layers (17-32): Semantic meaning
- Late layers (33-48): Abstract/high-level concepts

For DiT conditioning, this may translate to:
- Early: Text rendering, literal descriptions
- Middle: Object/scene semantics
- Late: Style, composition, abstract qualities

LTX-2 uses all 49 Gemma layers with uniform blending. This experiment tests
selective layer usage by zeroing out contributions from layer subsets.

Usage:
    uv run python experiments/ltx2/layer_extraction_comparison.py
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch


TEST_PROMPTS = [
    # Literal/concrete prompt - should benefit from early layers
    "The word HELLO written in red on a white background",
    # Semantic prompt - should benefit from middle layers
    "A golden retriever playing fetch with a tennis ball in a park",
    # Abstract/style prompt - should benefit from late layers
    "A dreamlike surreal landscape, ethereal atmosphere, abstract composition",
]

# Layer configurations to test
# LTX-2 uses 49 Gemma layers (0-48)
LAYER_CONFIGS = {
    "all_layers": list(range(49)),  # Baseline: all layers
    "early_only": list(range(17)),  # Layers 0-16
    "middle_only": list(range(17, 33)),  # Layers 17-32
    "late_only": list(range(33, 49)),  # Layers 33-48
    "early_middle": list(range(33)),  # Layers 0-32
    "middle_late": list(range(17, 49)),  # Layers 17-48
    "no_early": list(range(17, 49)),  # Skip early layers
    "no_late": list(range(33)),  # Skip late layers
}


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


def run_layer_extraction_experiment(
    output_dir: str = "experiments/outputs/layer_extraction",
    model_path: str = "models/LTX-2",
    save_videos: bool = True,
    configs_to_test: list = None,
):
    """
    Run layer extraction comparison experiment.

    Tests which layer subsets contribute most to generation quality
    by masking out contributions from excluded layers.
    """
    from diffusers import LTX2Pipeline
    from diffusers.utils import export_to_video

    if configs_to_test is None:
        configs_to_test = ["all_layers", "early_only", "middle_only", "late_only"]

    print("=" * 60)
    print("LTX-2 Layer Extraction Comparison")
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

    # Get the text encoder connector for layer masking
    # LTX-2's connector handles the layer combination
    connector = pipe.transformer.transformer.text_encoder_connector

    results = {}

    for config_name in configs_to_test:
        if config_name not in LAYER_CONFIGS:
            print(f"Unknown config: {config_name}, skipping")
            continue

        active_layers = LAYER_CONFIGS[config_name]
        print(f"\n{'='*50}")
        print(f"Config: {config_name}")
        print(f"Active layers: {len(active_layers)} ({min(active_layers)}-{max(active_layers)})")
        print("=" * 50)

        config_results = []

        # Store original projection weights
        original_proj = {}
        if hasattr(connector, "per_layer_proj"):
            for layer_idx in range(49):
                if layer_idx not in active_layers:
                    # Zero out excluded layers
                    proj_name = f"layer_{layer_idx}"
                    if hasattr(connector.per_layer_proj, proj_name):
                        proj = getattr(connector.per_layer_proj, proj_name)
                        original_proj[proj_name] = {
                            "weight": proj.weight.data.clone(),
                            "bias": proj.bias.data.clone() if proj.bias is not None else None,
                        }
                        proj.weight.data.zero_()
                        if proj.bias is not None:
                            proj.bias.data.zero_()

        for i, prompt in enumerate(TEST_PROMPTS):
            print(f"\n  [{i+1}/{len(TEST_PROMPTS)}] {prompt[:50]}...")

            start_time = time.time()

            try:
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

                # Compute statistics
                stats = compute_frame_statistics(frames)
                stats["generation_time"] = gen_time
                stats["prompt"] = prompt
                stats["prompt_type"] = ["literal", "semantic", "abstract"][i]
                config_results.append(stats)

                print(f"  Time: {gen_time:.1f}s | Mean: {stats['mean']:.1f} | Std: {stats['std']:.1f}")

                # Save video
                if save_videos:
                    video_path = output_path / f"{config_name}_sample{i}.mp4"
                    export_to_video(frames, str(video_path), fps=24)
                    print(f"  Saved: {video_path.name}")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                config_results.append({"error": str(e), "prompt": prompt})

            # Memory cleanup
            gc.collect()
            torch.cuda.empty_cache()

        # Restore original weights
        if hasattr(connector, "per_layer_proj"):
            for proj_name, data in original_proj.items():
                if hasattr(connector.per_layer_proj, proj_name):
                    proj = getattr(connector.per_layer_proj, proj_name)
                    proj.weight.data.copy_(data["weight"])
                    if data["bias"] is not None:
                        proj.bias.data.copy_(data["bias"])

        results[config_name] = config_results

    # Analysis
    print("\n" + "=" * 60)
    print("LAYER EXTRACTION ANALYSIS")
    print("=" * 60)

    # Summary table
    print("\n## Per-Config Statistics\n")
    print("| Config | Layers | Mean | Std | Temp Var |")
    print("|--------|--------|------|-----|----------|")

    for config_name, config_results in results.items():
        valid = [s for s in config_results if "mean" in s]
        if valid:
            mean_avg = np.mean([s["mean"] for s in valid])
            std_avg = np.mean([s["std"] for s in valid])
            temp_avg = np.mean([s["temporal_variance"] for s in valid])
            n_layers = len(LAYER_CONFIGS.get(config_name, []))
            print(f"| {config_name:14} | {n_layers:6} | {mean_avg:5.1f} | {std_avg:4.1f} | {temp_avg:8.1f} |")

    # Per-prompt-type analysis
    print("\n## Layer Effects by Prompt Type\n")
    prompt_types = ["literal", "semantic", "abstract"]

    for ptype in prompt_types:
        print(f"\n### {ptype.capitalize()} prompt")
        print("| Config | Mean | Std |")
        print("|--------|------|-----|")
        for config_name, config_results in results.items():
            matching = [s for s in config_results if s.get("prompt_type") == ptype and "mean" in s]
            if matching:
                s = matching[0]
                print(f"| {config_name:14} | {s['mean']:5.1f} | {s['std']:4.1f} |")

    # Save results
    results_file = output_path / "layer_extraction_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Cleanup
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="LTX-2 Layer Extraction Comparison")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/outputs/layer_extraction",
        help="Output directory",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=["all_layers", "early_only", "middle_only", "late_only"],
        help="Layer configurations to test",
    )
    parser.add_argument(
        "--no-save-videos",
        action="store_true",
        help="Don't save output videos",
    )
    args = parser.parse_args()

    run_layer_extraction_experiment(
        output_dir=args.output_dir,
        configs_to_test=args.configs,
        save_videos=not args.no_save_videos,
    )


if __name__ == "__main__":
    main()
