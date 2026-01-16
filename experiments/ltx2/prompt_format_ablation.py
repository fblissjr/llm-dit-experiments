#!/usr/bin/env python3
"""
LTX-2 Prompt Format Ablation Experiment

Last Updated: 2026-01-16

Tests whether structured prompt formats (markdown, JSON, XML, YAML) work as well
as prose for LTX-2 generation. Based on LTX-2 training data analysis showing that
prose captions are the native format.

Hypothesis: Prose format will outperform structured formats because:
1. LTX-2's training data uses prose video captions
2. Structured formats create out-of-distribution inputs
3. The model may not parse structured syntax correctly

Expected Results:
- Prose baseline: Best quality (in-distribution)
- JSON: Possibly degraded (syntax tokens take up sequence budget)
- XML: Possibly degraded (verbose tags reduce semantic density)
- Markdown: Intermediate (closer to natural text)
- YAML: Intermediate (somewhat readable)

Measured Metrics:
- SigLIP score (text-image alignment)
- ImageReward (human preference proxy)
- Basic frame statistics

Usage:
    # Quick test (2 formats x 1 seed)
    uv run python experiments/ltx2/prompt_format_ablation.py --quick

    # Full sweep (5 formats x 3 seeds)
    uv run python experiments/ltx2/prompt_format_ablation.py

    # Specific formats
    uv run python experiments/ltx2/prompt_format_ablation.py --formats prose_baseline json xml
"""

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import prompts from centralized module
from experiments.ltx2.prompts import STRUCTURED_PROMPTS, get_structured_prompts


def compute_metrics(frame: Image.Image, prompt: str) -> dict:
    """Compute quality metrics for a frame."""
    metrics = {}

    # Basic statistics
    frame_array = np.array(frame)
    metrics["mean_brightness"] = float(frame_array.mean())
    metrics["std"] = float(frame_array.std())
    metrics["min"] = float(frame_array.min())
    metrics["max"] = float(frame_array.max())

    # Try SigLIP score
    try:
        from experiments.metrics.siglip_score import compute_siglip_score
        # For SigLIP, use the prose version for fair comparison across formats
        # (structured formats shouldn't be penalized for format syntax)
        metrics["siglip_score"] = compute_siglip_score(
            STRUCTURED_PROMPTS["prose_baseline"], frame
        )
    except Exception as e:
        metrics["siglip_score"] = None
        metrics["siglip_error"] = str(e)

    # Try ImageReward
    try:
        from experiments.metrics.image_reward import compute_image_reward
        # Same logic - evaluate against prose semantic content
        metrics["image_reward"] = compute_image_reward(
            STRUCTURED_PROMPTS["prose_baseline"], frame
        )
    except Exception as e:
        metrics["image_reward"] = None
        metrics["image_reward_error"] = str(e)

    return metrics


def compute_frame_statistics(frames: list) -> dict:
    """Compute statistics on generated frames."""
    frame_arrays = [np.array(f) for f in frames]
    stacked = np.stack(frame_arrays, axis=0)  # [T, H, W, C]

    return {
        "mean_brightness": float(stacked.mean()),
        "std": float(stacked.std()),
        "min": float(stacked.min()),
        "max": float(stacked.max()),
        "temporal_variance": float(stacked.var(axis=0).mean()),
    }


def run_format_ablation(
    model_path: str = "models/LTX-2",
    output_dir: str = "experiments/results",
    formats: list[str] | None = None,
    seeds: list[int] | None = None,
    quick: bool = False,
    num_inference_steps: int = 25,
    guidance_scale: float = 3.0,
    height: int = 512,
    width: int = 768,
    num_frames: int = 33,
    save_videos: bool = True,
):
    """
    Run prompt format ablation experiment.

    Tests whether structured formats (markdown, JSON, XML, YAML) produce
    similar quality to prose format when used as LTX-2 prompts.

    Args:
        model_path: Path to LTX-2 model
        output_dir: Output directory for results
        formats: List of format names to test (default: all)
        seeds: List of random seeds for multiple runs
        quick: Quick mode (2 formats, 1 seed)
        num_inference_steps: Diffusion steps
        guidance_scale: CFG scale
        height: Video height
        width: Video width
        num_frames: Number of frames
        save_videos: Whether to save MP4 videos
    """
    from diffusers import LTX2Pipeline
    from diffusers.utils import export_to_video

    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()

    # Setup
    if formats is None:
        formats = list(STRUCTURED_PROMPTS.keys()) if not quick else ["prose_baseline", "json"]

    if seeds is None:
        seeds = [42] if quick else [42, 123, 456]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"ltx2_format_ablation_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "images").mkdir(exist_ok=True)
    if save_videos:
        (output_path / "videos").mkdir(exist_ok=True)
    (output_path / "metadata").mkdir(exist_ok=True)

    print("=" * 60)
    print("LTX-2 Prompt Format Ablation Experiment")
    print("=" * 60)
    print(f"Output: {output_path}")
    print(f"Formats: {formats}")
    print(f"Seeds: {seeds}")
    print(f"Quick mode: {quick}")

    total_gens = len(formats) * len(seeds)
    print(f"Total generations: {total_gens}")

    # Load pipeline
    print("\nLoading pipeline...")
    pipe = LTX2Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    # Results storage
    all_results = []
    format_summaries = {}

    gen_idx = 0
    for format_name in formats:
        if format_name not in STRUCTURED_PROMPTS:
            print(f"Unknown format: {format_name}, skipping")
            continue

        prompt_text = STRUCTURED_PROMPTS[format_name]
        print(f"\n{'='*60}")
        print(f"Format: {format_name}")
        print(f"Prompt length: {len(prompt_text)} chars, {len(prompt_text.split())} words")
        print("=" * 60)
        print(f"Preview: {prompt_text[:100]}...")

        format_results = []

        for seed in seeds:
            gen_idx += 1
            print(f"\n[{gen_idx}/{total_gens}] {format_name} (seed={seed})")

            start_time = time.time()

            try:
                generator = torch.Generator(device="cpu").manual_seed(seed)

                output = pipe(
                    prompt=prompt_text,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )

                frames = output.frames[0]
                gen_time = time.time() - start_time

                # Extract first frame
                first_frame = frames[0]

                # Compute metrics
                metrics = compute_metrics(first_frame, prompt_text)
                frame_stats = compute_frame_statistics(frames)

                # Save outputs
                sample_name = f"{format_name}_seed{seed}"
                image_path = output_path / "images" / f"{sample_name}.png"
                first_frame.save(image_path)

                video_path = None
                if save_videos:
                    video_path = output_path / "videos" / f"{sample_name}.mp4"
                    export_to_video(frames, str(video_path), fps=24)

                # Build result
                result = {
                    "config": {
                        "format": format_name,
                        "seed": seed,
                        "prompt_length_chars": len(prompt_text),
                        "prompt_length_words": len(prompt_text.split()),
                    },
                    "generation_time_seconds": gen_time,
                    "output_path": str(image_path.relative_to(output_path)),
                    "video_path": str(video_path.relative_to(output_path)) if video_path else None,
                    **metrics,
                    **frame_stats,
                }

                # Save metadata
                metadata_path = output_path / "metadata" / f"{sample_name}.json"
                with open(metadata_path, "w") as f:
                    json.dump(result, f, indent=2)

                all_results.append(result)
                format_results.append(result)

                print(f"  Time: {gen_time:.1f}s | Brightness: {metrics['mean_brightness']:.1f}")
                if metrics.get("siglip_score") is not None:
                    print(f"  SigLIP: {metrics['siglip_score']:.4f}", end="")
                if metrics.get("image_reward") is not None:
                    print(f" | ImageReward: {metrics['image_reward']:.4f}", end="")
                print()

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "config": {"format": format_name, "seed": seed},
                    "error": str(e),
                })

            # Cleanup
            gc.collect()
            torch.cuda.empty_cache()

        # Summarize format
        valid_results = [r for r in format_results if "error" not in r]
        if valid_results:
            format_summaries[format_name] = {
                "num_samples": len(valid_results),
                "mean_brightness": np.mean([r["mean_brightness"] for r in valid_results]),
                "mean_siglip": np.mean([
                    r["siglip_score"] for r in valid_results
                    if r.get("siglip_score") is not None
                ]) if any(r.get("siglip_score") is not None for r in valid_results) else None,
                "mean_image_reward": np.mean([
                    r["image_reward"] for r in valid_results
                    if r.get("image_reward") is not None
                ]) if any(r.get("image_reward") is not None for r in valid_results) else None,
                "prompt_length_words": len(STRUCTURED_PROMPTS[format_name].split()),
            }

    # Save summary
    summary = {
        "experiment": "ltx2_prompt_format_ablation",
        "timestamp": timestamp,
        "hypothesis": "Prose format outperforms structured formats (markdown, JSON, XML, YAML)",
        "parameters": {
            "formats_tested": formats,
            "seeds": seeds,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "resolution": f"{height}x{width}",
            "num_frames": num_frames,
        },
        "total_generations": len(all_results),
        "format_summaries": format_summaries,
        "results": all_results,
    }

    summary_path = output_path / "format_ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print analysis
    print("\n" + "=" * 60)
    print("FORMAT ABLATION RESULTS")
    print("=" * 60)

    print(f"\n{'Format':<18} {'Words':<8} {'Brightness':<12} {'SigLIP':<10} {'ImgReward':<10}")
    print("-" * 70)

    baseline_siglip = format_summaries.get("prose_baseline", {}).get("mean_siglip")
    baseline_reward = format_summaries.get("prose_baseline", {}).get("mean_image_reward")

    for format_name, stats in format_summaries.items():
        siglip_str = f"{stats['mean_siglip']:.4f}" if stats.get('mean_siglip') is not None else "N/A"
        reward_str = f"{stats['mean_image_reward']:.4f}" if stats.get('mean_image_reward') is not None else "N/A"

        # Add delta from baseline
        if format_name != "prose_baseline" and baseline_siglip and stats.get('mean_siglip'):
            delta = stats['mean_siglip'] - baseline_siglip
            siglip_str += f" ({delta:+.3f})"

        print(f"{format_name:<18} {stats['prompt_length_words']:<8} {stats['mean_brightness']:<12.1f} {siglip_str:<10} {reward_str:<10}")

    # Create visualization
    if len(format_summaries) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        format_names = list(format_summaries.keys())
        x = range(len(format_names))

        # Brightness
        ax1 = axes[0]
        brightness_values = [format_summaries[f]["mean_brightness"] for f in format_names]
        colors = ['green' if f == 'prose_baseline' else 'steelblue' for f in format_names]
        ax1.bar(x, brightness_values, color=colors)
        ax1.set_xticks(x)
        ax1.set_xticklabels(format_names, rotation=45, ha="right")
        ax1.set_ylabel("Mean Brightness")
        ax1.set_title("Brightness by Format")

        # SigLIP
        ax2 = axes[1]
        siglip_values = [format_summaries[f].get("mean_siglip") or 0 for f in format_names]
        ax2.bar(x, siglip_values, color=colors)
        ax2.set_xticks(x)
        ax2.set_xticklabels(format_names, rotation=45, ha="right")
        ax2.set_ylabel("SigLIP Score")
        ax2.set_title("Text-Image Alignment by Format")
        if baseline_siglip:
            ax2.axhline(y=baseline_siglip, color='red', linestyle='--', label='Prose baseline')
            ax2.legend()

        # ImageReward
        ax3 = axes[2]
        reward_values = [format_summaries[f].get("mean_image_reward") or 0 for f in format_names]
        ax3.bar(x, reward_values, color=colors)
        ax3.set_xticks(x)
        ax3.set_xticklabels(format_names, rotation=45, ha="right")
        ax3.set_ylabel("ImageReward Score")
        ax3.set_title("Human Preference by Format")
        if baseline_reward:
            ax3.axhline(y=baseline_reward, color='red', linestyle='--', label='Prose baseline')
            ax3.legend()

        plt.tight_layout()
        plot_path = output_path / "format_ablation_comparison.png"
        plt.savefig(plot_path, dpi=150)
        print(f"\nSaved plot to: {plot_path}")
        plt.close()

    # Cleanup
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nResults saved to: {output_path}")
    print("=" * 60)

    # Conclusion
    if len(format_summaries) > 1 and baseline_siglip:
        print("\n" + "=" * 60)
        print("HYPOTHESIS TEST")
        print("=" * 60)

        better_than_prose = []
        worse_than_prose = []

        for fmt, stats in format_summaries.items():
            if fmt == "prose_baseline":
                continue
            if stats.get("mean_siglip") is not None:
                if stats["mean_siglip"] > baseline_siglip:
                    better_than_prose.append((fmt, stats["mean_siglip"] - baseline_siglip))
                else:
                    worse_than_prose.append((fmt, baseline_siglip - stats["mean_siglip"]))

        if worse_than_prose and not better_than_prose:
            print("HYPOTHESIS SUPPORTED: Prose outperforms all structured formats")
        elif better_than_prose:
            print("HYPOTHESIS CHALLENGED: Some structured formats performed better")
            for fmt, delta in better_than_prose:
                print(f"  - {fmt}: +{delta:.4f} SigLIP")
        else:
            print("INCONCLUSIVE: Need more samples or metric data")

    return summary


def main():
    parser = argparse.ArgumentParser(description="LTX-2 Prompt Format Ablation")
    parser.add_argument("--model-path", default="models/LTX-2", help="Path to LTX-2 model")
    parser.add_argument("--output-dir", default="experiments/results", help="Output directory")
    parser.add_argument("--formats", nargs="+", help="Formats to test")
    parser.add_argument("--seeds", type=int, nargs="+", help="Random seeds")
    parser.add_argument("--quick", action="store_true", help="Quick test (2 formats, 1 seed)")
    parser.add_argument("--steps", type=int, default=25, help="Inference steps")
    parser.add_argument("--cfg", type=float, default=3.0, help="Guidance scale")
    parser.add_argument("--height", type=int, default=512, help="Video height")
    parser.add_argument("--width", type=int, default=768, help="Video width")
    parser.add_argument("--frames", type=int, default=33, help="Number of frames")
    parser.add_argument("--no-save-videos", action="store_true", help="Don't save MP4 videos")
    args = parser.parse_args()

    run_format_ablation(
        model_path=args.model_path,
        output_dir=args.output_dir,
        formats=args.formats,
        seeds=args.seeds,
        quick=args.quick,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        save_videos=not args.no_save_videos,
    )


if __name__ == "__main__":
    main()
