#!/usr/bin/env python3
"""
LTX-2 Prompt Structure Ablation Experiment

Last Updated: 2026-01-15

Zero-training technique: Test whether prompt structure affects output quality.
Nobody has published rigorous comparisons for DiT conditioning specifically.

Prompt Structures Tested:
1. Terse: "cat"
2. Expanded: "a fluffy orange cat with green eyes sitting"
3. Structured: "subject: cat, action: sitting, style: realistic"
4. Reasoning: "I want to generate a cat. It should be fluffy and orange. Final: cat sitting"
5. Conclusion-only: "final output: fluffy orange cat sitting"

Usage:
    uv run python experiments/ltx2/prompt_structure_ablation.py
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch


# Test subjects with different prompt structures
# Each subject has multiple prompt formats to test
TEST_SUBJECTS = [
    {
        "name": "cat",
        "prompts": {
            "terse": "cat",
            "basic": "a cat sitting",
            "expanded": "a fluffy orange tabby cat with green eyes sitting on a windowsill, soft natural lighting",
            "structured": "Subject: orange tabby cat. Action: sitting on windowsill. Lighting: soft natural. Style: realistic.",
            "reasoning": "I want a peaceful scene with a cat. The cat should be fluffy and orange, a tabby pattern. It's sitting on a windowsill with soft natural lighting coming through. Final scene: fluffy orange tabby cat sitting on windowsill.",
            "conclusion_only": "Final output: fluffy orange tabby cat with green eyes sitting on windowsill, soft natural lighting",
        }
    },
    {
        "name": "ocean",
        "prompts": {
            "terse": "ocean",
            "basic": "ocean waves crashing",
            "expanded": "dramatic ocean waves crashing against rocky cliffs at golden hour sunset, foam spraying, warm orange light",
            "structured": "Subject: ocean waves. Action: crashing on cliffs. Time: golden hour sunset. Style: dramatic cinematic.",
            "reasoning": "I need a dramatic ocean scene. Big waves hitting rocks. The time should be sunset for warm colors. Lots of spray and foam. Final scene: dramatic waves crashing on rocky cliffs at golden hour.",
            "conclusion_only": "Final output: dramatic ocean waves crashing against rocky cliffs at golden hour sunset with foam spray",
        }
    },
    {
        "name": "forest",
        "prompts": {
            "terse": "forest",
            "basic": "a forest path",
            "expanded": "sunlight streaming through tall pine trees in a misty morning forest, dappled light on a winding path, serene atmosphere",
            "structured": "Subject: pine forest path. Time: misty morning. Lighting: sunbeams through trees. Mood: serene peaceful.",
            "reasoning": "A peaceful forest scene is needed. Morning mist would add atmosphere. Sunlight streaming through pine trees creates dappled patterns. A winding path adds depth. Final scene: misty morning forest with sunbeams and path.",
            "conclusion_only": "Final output: misty morning pine forest with sunlight streaming through trees, dappled light on winding path",
        }
    },
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


def run_prompt_ablation(
    output_dir: str = "experiments/outputs/prompt_structure",
    model_path: str = "models/LTX-2",
    save_videos: bool = True,
    seed: int = 42,
):
    """
    Run prompt structure ablation experiment.

    Tests whether different prompt structures produce meaningfully different outputs.
    Uses same seed across structures to isolate prompt effect.
    """
    from diffusers import LTX2Pipeline
    from diffusers.utils import export_to_video

    print("=" * 60)
    print("LTX-2 Prompt Structure Ablation")
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

    for subject in TEST_SUBJECTS:
        subject_name = subject["name"]
        print(f"\n{'='*50}")
        print(f"Subject: {subject_name}")
        print("=" * 50)

        subject_results = {}

        for structure_name, prompt in subject["prompts"].items():
            print(f"\n  [{structure_name}]")
            print(f"  Prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")

            start_time = time.time()

            try:
                generator = torch.Generator(device="cpu").manual_seed(seed)

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
                stats["prompt_length"] = len(prompt)
                subject_results[structure_name] = stats

                print(f"  Time: {gen_time:.1f}s | Mean: {stats['mean']:.1f} | Std: {stats['std']:.1f}")

                # Save video
                if save_videos:
                    safe_name = structure_name.replace("_", "-")
                    video_path = output_path / f"{subject_name}_{safe_name}.mp4"
                    export_to_video(frames, str(video_path), fps=24)
                    print(f"  Saved: {video_path.name}")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                subject_results[structure_name] = {"error": str(e), "prompt": prompt}

            # Memory cleanup
            gc.collect()
            torch.cuda.empty_cache()

        results[subject_name] = subject_results

    # Analysis
    print("\n" + "=" * 60)
    print("PROMPT STRUCTURE ANALYSIS")
    print("=" * 60)

    print("\n## Per-Subject Statistics\n")
    print("| Subject | Structure | Mean | Std | Temp Var | Prompt Len |")
    print("|---------|-----------|------|-----|----------|------------|")

    for subject_name, subject_results in results.items():
        for structure, stats in subject_results.items():
            if "mean" in stats:
                print(f"| {subject_name:7} | {structure:9} | {stats['mean']:5.1f} | {stats['std']:4.1f} | {stats['temporal_variance']:.1f} | {stats['prompt_length']:3} |")

    # Cross-structure comparison
    print("\n## Structure Effects (Averaged Across Subjects)\n")
    structure_averages = {}
    for subject_name, subject_results in results.items():
        for structure, stats in subject_results.items():
            if "mean" not in stats:
                continue
            if structure not in structure_averages:
                structure_averages[structure] = {"means": [], "stds": [], "temp_vars": []}
            structure_averages[structure]["means"].append(stats["mean"])
            structure_averages[structure]["stds"].append(stats["std"])
            structure_averages[structure]["temp_vars"].append(stats["temporal_variance"])

    print("| Structure | Avg Mean | Avg Std | Avg Temp Var |")
    print("|-----------|----------|---------|--------------|")
    for structure, avgs in structure_averages.items():
        mean_avg = np.mean(avgs["means"])
        std_avg = np.mean(avgs["stds"])
        temp_avg = np.mean(avgs["temp_vars"])
        print(f"| {structure:9} | {mean_avg:8.1f} | {std_avg:7.1f} | {temp_avg:12.1f} |")

    # Calculate variance between structures
    all_means = [np.mean(avgs["means"]) for avgs in structure_averages.values()]
    structure_variance = np.var(all_means)
    print(f"\nVariance between structures: {structure_variance:.2f}")

    if structure_variance < 5:
        print("→ Low variance: Prompt structure has MINIMAL effect on output statistics")
    elif structure_variance < 20:
        print("→ Moderate variance: Prompt structure has SOME effect on output")
    else:
        print("→ High variance: Prompt structure SIGNIFICANTLY affects output")

    # Save results
    results_file = output_path / "prompt_ablation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Cleanup
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="LTX-2 Prompt Structure Ablation")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/outputs/prompt_structure",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (same across all structures)",
    )
    parser.add_argument(
        "--no-save-videos",
        action="store_true",
        help="Don't save output videos",
    )
    args = parser.parse_args()

    run_prompt_ablation(
        output_dir=args.output_dir,
        save_videos=not args.no_save_videos,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
