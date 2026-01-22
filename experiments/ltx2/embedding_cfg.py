#!/usr/bin/env python3
"""
LTX-2 Embedding-Space CFG Experiment

Last Updated: 2026-01-15

Apply classifier-free guidance at the encoder output level instead of DiT output.
This is computationally cheaper than standard CFG (one embedding manipulation vs
many DiT forward passes).

Concept:
    good_hidden = encoder("a majestic lion, golden hour lighting")
    bad_hidden = encoder("a lion")
    refined = good_hidden + alpha * (good_hidden - bad_hidden)

Key Question: Does embedding-space CFG complement or conflict with DiT CFG?

Usage:
    uv run python experiments/ltx2/embedding_cfg.py
    uv run python experiments/ltx2/embedding_cfg.py --embed-alpha 1.5
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch

# Prompt pairs for embedding CFG (good = detailed, bad = minimal)
CFG_PAIRS = [
    {
        "bad": "a dog",
        "good": "a golden retriever puppy playing fetch in a sunny park, soft bokeh background",
        "test": "a dog running",  # Neutral test prompt
    },
    {
        "bad": "mountains",
        "good": "majestic snow-capped mountains at sunrise, golden light, dramatic clouds",
        "test": "mountain landscape",
    },
    {
        "bad": "food",
        "good": "a gourmet pasta dish with fresh basil, steam rising, professional food photography",
        "test": "cooking pasta",
    },
]


def extract_embedding_via_hook(pipe, prompt: str, seed: int = 42):
    """
    Extract embedding by hooking into encode_prompt during minimal generation.

    Direct encode_prompt() OOMs on RTX 4090 with Gemma-3 12B.
    This workaround uses the pipeline's internal memory management.
    """
    captured = {"embeds": None}
    original_encode = pipe.encode_prompt

    def hooked_encode(*args, **kwargs):
        result = original_encode(*args, **kwargs)
        captured["embeds"] = result[0].cpu()
        captured["mask"] = result[1].cpu()
        return result

    pipe.encode_prompt = hooked_encode

    try:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        pipe(
            prompt=prompt,
            height=256,  # Minimal size
            width=256,
            num_frames=9,
            num_inference_steps=2,
            guidance_scale=1.0,
            generator=generator,
        )
    except Exception as e:
        print(f"    Error during extraction: {e}")
    finally:
        pipe.encode_prompt = original_encode

    gc.collect()
    torch.cuda.empty_cache()

    return captured.get("embeds"), captured.get("mask")


def generate_with_embedding_cfg(
    pipe,
    test_prompt: str,
    good_prompt: str,
    bad_prompt: str,
    embed_alpha: float,
    dit_cfg: float = 3.0,
    seed: int = 42,
    num_frames: int = 33,
    height: int = 512,
    width: int = 768,
    num_inference_steps: int = 25,
    device="cuda",
    precomputed_direction: torch.Tensor = None,
):
    """
    Generate video with embedding-space CFG.

    The refinement formula:
        refined = test_embeds + embed_alpha * (good_embeds - bad_embeds)

    This pushes the test embedding in the direction of "what makes good detailed".

    Uses hook-based approach to avoid OOM with Gemma-3 12B on RTX 4090.

    Args:
        pipe: LTX2Pipeline
        test_prompt: Neutral prompt to generate
        good_prompt: Detailed/high-quality version (only used if precomputed_direction is None)
        bad_prompt: Minimal/low-quality version (only used if precomputed_direction is None)
        embed_alpha: Embedding CFG strength (0 = no effect, higher = more enhancement)
        dit_cfg: Standard DiT CFG scale (guidance_scale parameter)
        seed: Random seed
        precomputed_direction: Pre-computed (good - bad) direction to avoid re-extraction

    Returns:
        frames: List of PIL images
    """
    # Compute direction if not provided
    if precomputed_direction is None and embed_alpha != 0.0:
        print("    Extracting embeddings for CFG direction...")
        good_embeds, _ = extract_embedding_via_hook(pipe, good_prompt, seed)
        bad_embeds, _ = extract_embedding_via_hook(pipe, bad_prompt, seed)

        if good_embeds is None or bad_embeds is None:
            print("    ERROR: Failed to extract embeddings for CFG")
            return None

        direction = good_embeds - bad_embeds
    else:
        direction = precomputed_direction

    # Hook into encode_prompt to apply CFG
    original_encode = pipe.encode_prompt

    def cfg_encode(*args, **kwargs):
        result = original_encode(*args, **kwargs)
        test_embeds = result[0]

        if embed_alpha != 0.0 and direction is not None:
            dir_on_device = direction.to(device=test_embeds.device, dtype=test_embeds.dtype)

            # Handle sequence length mismatch
            dir_seq = dir_on_device.shape[1]
            emb_seq = test_embeds.shape[1]

            if dir_seq != emb_seq:
                if dir_seq > emb_seq:
                    dir_on_device = dir_on_device[:, :emb_seq, :]
                else:
                    pad = torch.zeros(
                        1,
                        emb_seq - dir_seq,
                        dir_on_device.shape[2],
                        device=dir_on_device.device,
                        dtype=dir_on_device.dtype,
                    )
                    dir_on_device = torch.cat([dir_on_device, pad], dim=1)

            refined = test_embeds + embed_alpha * dir_on_device
            return (refined, result[1], result[2], result[3])

        return result

    pipe.encode_prompt = cfg_encode

    try:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        output = pipe(
            prompt=test_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=dit_cfg,
            generator=generator,
        )
        return output.frames[0]
    finally:
        pipe.encode_prompt = original_encode
        gc.collect()
        torch.cuda.empty_cache()


def compute_frame_statistics(frames: list) -> dict:
    """Compute statistics on generated frames."""
    frame_arrays = [np.array(f) for f in frames]
    stacked = np.stack(frame_arrays, axis=0)  # [T, H, W, C]

    return {
        "mean": stacked.mean(),
        "std": stacked.std(),
        "min": stacked.min(),
        "max": stacked.max(),
        "temporal_variance": stacked.var(axis=0).mean(),
    }


def run_embedding_cfg_experiment(
    embed_alphas: list = [0.0, 0.5, 1.0, 1.5, 2.0],
    dit_cfg: float = 3.0,
    output_dir: str = "experiments/results/ltx2",
    model_path: str = "models/LTX-2",
    save_videos: bool = True,
):
    """
    Run embedding-space CFG experiment with various alpha values.

    Compares effect of embedding CFG while keeping DiT CFG constant.
    """
    from diffusers import LTX2Pipeline
    from diffusers.utils import export_to_video

    print("=" * 60)
    print("LTX-2 Embedding-Space CFG Experiment")
    print("=" * 60)
    print(f"DiT CFG (constant): {dit_cfg}")
    print(f"Embed CFG alphas: {embed_alphas}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    print("\nLoading pipeline...")
    pipe = LTX2Pipeline.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
    )
    pipe.enable_sequential_cpu_offload()

    # Results storage
    results = {}

    # Test each alpha value
    for embed_alpha in embed_alphas:
        print(f"\n{'=' * 40}")
        print(f"Embed Alpha: {embed_alpha}")
        print("=" * 40)

        alpha_results = []

        for i, pair in enumerate(CFG_PAIRS):
            print(f"\n  [{i + 1}/{len(CFG_PAIRS)}] Test: {pair['test'][:30]}...")
            print(f"      Good: {pair['good'][:30]}...")
            print(f"      Bad: {pair['bad'][:30]}...")

            start_time = time.time()

            try:
                frames = generate_with_embedding_cfg(
                    pipe,
                    test_prompt=pair["test"],
                    good_prompt=pair["good"],
                    bad_prompt=pair["bad"],
                    embed_alpha=embed_alpha,
                    dit_cfg=dit_cfg,
                    seed=42 + i,
                    num_frames=33,
                    height=512,
                    width=768,
                    num_inference_steps=25,
                )

                gen_time = time.time() - start_time

                # Compute statistics
                stats = compute_frame_statistics(frames)
                stats["generation_time"] = gen_time
                stats["test_prompt"] = pair["test"]
                stats["good_prompt"] = pair["good"]
                stats["bad_prompt"] = pair["bad"]
                alpha_results.append(stats)

                print(
                    f"    Time: {gen_time:.1f}s, Mean: {stats['mean']:.1f}, Std: {stats['std']:.1f}"
                )

                # Save video
                if save_videos:
                    video_path = output_path / f"embed_cfg{embed_alpha:.1f}_sample{i}.mp4"
                    export_to_video(frames, str(video_path), fps=24)
                    print(f"    Saved: {video_path.name}")

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback

                traceback.print_exc()
                alpha_results.append({"error": str(e), "test_prompt": pair["test"]})

            # Memory cleanup
            gc.collect()
            torch.cuda.empty_cache()

        results[f"embed_alpha_{embed_alpha}"] = alpha_results

    # Summary
    print("\n" + "=" * 60)
    print("EMBEDDING CFG SUMMARY")
    print("=" * 60)

    baseline_stats = results.get("embed_alpha_0.0", [])
    if baseline_stats and "mean" in baseline_stats[0]:
        baseline_mean = np.mean([s["mean"] for s in baseline_stats if "mean" in s])
        baseline_std = np.mean([s["std"] for s in baseline_stats if "std" in s])

        print(f"\nBaseline (embed_alpha=0): mean={baseline_mean:.1f}, std={baseline_std:.1f}")
        print(f"(This is equivalent to standard DiT CFG @ {dit_cfg})")

        for alpha_key, alpha_results in results.items():
            if alpha_key == "embed_alpha_0.0":
                continue

            valid_results = [s for s in alpha_results if "mean" in s]
            if valid_results:
                config_mean = np.mean([s["mean"] for s in valid_results])
                config_std = np.mean([s["std"] for s in valid_results])
                mean_diff = (config_mean - baseline_mean) / baseline_mean * 100
                std_diff = (config_std - baseline_std) / baseline_std * 100

                print(f"\n{alpha_key}:")
                print(f"  mean={config_mean:.1f} ({mean_diff:+.1f}% vs baseline)")
                print(f"  std={config_std:.1f} ({std_diff:+.1f}% vs baseline)")

    # Save results
    results_file = output_path / "embedding_cfg_results.npz"
    np.savez(results_file, **{k: str(v) for k, v in results.items()})
    print(f"\nResults saved to {results_file}")

    # Save experiment config
    config = {
        "dit_cfg": dit_cfg,
        "embed_alphas": embed_alphas,
        "cfg_pairs": CFG_PAIRS,
    }
    torch.save(config, output_path / "experiment_config.pt")

    # Cleanup
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return results


def compare_cfg_methods(
    dit_cfg_values: list = [1.0, 3.0, 5.0],
    embed_alpha: float = 1.0,
    output_dir: str = "experiments/results/ltx2",
    model_path: str = "models/LTX-2",
    save_videos: bool = True,
):
    """
    Compare embedding CFG vs standard DiT CFG at different scales.

    For each DiT CFG value, generate:
    1. Standard: DiT CFG only
    2. Combined: DiT CFG + Embedding CFG
    """
    from diffusers import LTX2Pipeline
    from diffusers.utils import export_to_video

    print("=" * 60)
    print("CFG Method Comparison")
    print("=" * 60)
    print(f"DiT CFG values: {dit_cfg_values}")
    print(f"Embed alpha (when used): {embed_alpha}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    print("\nLoading pipeline...")
    pipe = LTX2Pipeline.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
    )
    pipe.enable_sequential_cpu_offload()

    results = {}
    test_pair = CFG_PAIRS[0]  # Use first pair for comparison

    for dit_cfg in dit_cfg_values:
        print(f"\n{'=' * 40}")
        print(f"DiT CFG: {dit_cfg}")
        print("=" * 40)

        # Standard DiT CFG only
        print("\n  Standard (DiT CFG only)...")
        start_time = time.time()

        frames_standard = generate_with_embedding_cfg(
            pipe,
            test_prompt=test_pair["test"],
            good_prompt=test_pair["good"],
            bad_prompt=test_pair["bad"],
            embed_alpha=0.0,  # No embedding CFG
            dit_cfg=dit_cfg,
            seed=42,
        )
        gen_time = time.time() - start_time
        stats_standard = compute_frame_statistics(frames_standard)
        stats_standard["method"] = "dit_only"
        stats_standard["dit_cfg"] = dit_cfg
        stats_standard["embed_alpha"] = 0.0

        print(
            f"    Time: {gen_time:.1f}s, Mean: {stats_standard['mean']:.1f}, Std: {stats_standard['std']:.1f}"
        )

        if save_videos:
            video_path = output_path / f"dit_cfg{dit_cfg:.1f}_standard.mp4"
            export_to_video(frames_standard, str(video_path), fps=24)

        # Combined: DiT CFG + Embedding CFG
        print(f"  Combined (DiT CFG + Embed Alpha {embed_alpha})...")
        start_time = time.time()

        frames_combined = generate_with_embedding_cfg(
            pipe,
            test_prompt=test_pair["test"],
            good_prompt=test_pair["good"],
            bad_prompt=test_pair["bad"],
            embed_alpha=embed_alpha,
            dit_cfg=dit_cfg,
            seed=42,
        )
        gen_time = time.time() - start_time
        stats_combined = compute_frame_statistics(frames_combined)
        stats_combined["method"] = "combined"
        stats_combined["dit_cfg"] = dit_cfg
        stats_combined["embed_alpha"] = embed_alpha

        print(
            f"    Time: {gen_time:.1f}s, Mean: {stats_combined['mean']:.1f}, Std: {stats_combined['std']:.1f}"
        )

        if save_videos:
            video_path = output_path / f"dit_cfg{dit_cfg:.1f}_combined.mp4"
            export_to_video(frames_combined, str(video_path), fps=24)

        results[f"dit_cfg_{dit_cfg}"] = {
            "standard": stats_standard,
            "combined": stats_combined,
        }

        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    for dit_cfg, cfg_results in results.items():
        std = cfg_results["standard"]
        comb = cfg_results["combined"]
        mean_diff = (comb["mean"] - std["mean"]) / std["mean"] * 100
        std_diff = (comb["std"] - std["std"]) / std["std"] * 100

        print(f"\n{dit_cfg}:")
        print(f"  Standard: mean={std['mean']:.1f}, std={std['std']:.1f}")
        print(f"  Combined: mean={comb['mean']:.1f}, std={comb['std']:.1f}")
        print(f"  Difference: mean {mean_diff:+.1f}%, std {std_diff:+.1f}%")

    # Save results
    results_file = output_path / "comparison_results.npz"
    np.savez(results_file, **{k: str(v) for k, v in results.items()})
    print(f"\nResults saved to {results_file}")

    # Cleanup
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="LTX-2 Embedding-Space CFG")
    parser.add_argument(
        "--embed-alpha",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 1.5, 2.0],
        help="Embedding CFG strength values to test",
    )
    parser.add_argument(
        "--dit-cfg",
        type=float,
        default=3.0,
        help="DiT CFG scale (constant across embed alphas)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison mode instead",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/ltx2",
        help="Output directory",
    )
    parser.add_argument(
        "--no-save-videos",
        action="store_true",
        help="Don't save output videos",
    )
    args = parser.parse_args()

    if args.compare:
        compare_cfg_methods(
            output_dir=args.output_dir + "/comparison",
            save_videos=not args.no_save_videos,
        )
    else:
        run_embedding_cfg_experiment(
            embed_alphas=args.embed_alpha,
            dit_cfg=args.dit_cfg,
            output_dir=args.output_dir,
            save_videos=not args.no_save_videos,
        )


if __name__ == "__main__":
    main()
