#!/usr/bin/env python3
"""
LTX-2 Layer Blend Sweep Experiment

Last Updated: 2026-01-16

Informed by projection matrix analysis (Session 27):
- Late layers (43-47) contribute ~25% of signal when accounting for activations
- Early layers (0-4) contribute <1% of signal
- Layer 48 (final) is paradoxically low (0.02%)
- Projection W is nearly uniform; activation magnitudes create differentiation

This experiment tests weighted layer combinations using proper blending
(not zeroing, which creates OOD inputs).

Blends tested:
1. baseline: All 49 layers, uniform weights
2. late_heavy: Upweight layers 40-47 (where contribution is highest)
3. early_excluded: Zero weight on layers 0-10 (minimal contribution)
4. top_contributors: Only layers 43-47 (top 5 by contribution)
5. u_shaped: Early (0-16) + Late (40-48), skip middle (traditional hypothesis)
6. anti_u: Middle only (15-35), exclude early and late

Memory-optimized for 24GB GPUs (RTX 4090):
- Text encoder: 8-bit quantized (~13GB instead of ~54GB)
- Pipeline: Group offloading for transformer blocks
- Sequential loading: Encode first, then offload, then generate

Usage:
    # Quick test (3 blends × 2 prompts)
    uv run python experiments/ltx2/layer_blend_sweep.py --quick

    # Full sweep
    uv run python experiments/ltx2/layer_blend_sweep.py

    # With specific seed
    uv run python experiments/ltx2/layer_blend_sweep.py --seed 42
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
# These match the official LTX-2 prompting guide format (100+ words, dialogue, etc.)
from experiments.ltx2.prompts import get_category_prompts, QUICK_CATEGORY

# Import memory-efficient utilities for 24GB GPUs
from experiments.ltx2.memory_utils import (
    load_text_encoder_8bit,
    encode_prompt_with_layer_weights,
    pack_text_embeds,
    load_pipeline_with_offloading,
    create_layer_weights,
    cleanup_memory,
    get_gpu_memory,
)

# Category prompts for layer blend experiments
TEST_PROMPTS = get_category_prompts(quick=False)

# Layer blend configurations
# Each config specifies layer weights (non-active layers get 0 weight)
# Weights are normalized to sum to 1 during generation
BLEND_CONFIGS = {
    "baseline": {
        "description": "All 49 layers, uniform weights",
        "active_layers": list(range(49)),
        "weights": None,  # None means uniform
    },
    "late_heavy": {
        "description": "Upweight layers 40-47 (2x), others normal",
        "active_layers": list(range(49)),
        "weights": {i: (2.0 if 40 <= i <= 47 else 1.0) for i in range(49)},
    },
    "early_excluded": {
        "description": "Exclude layers 0-10 (near-zero contribution)",
        "active_layers": list(range(11, 49)),
        "weights": None,
    },
    "top_contributors": {
        "description": "Only layers 43-47 (~25% of baseline contribution)",
        "active_layers": list(range(43, 48)),
        "weights": None,
    },
    "late_only": {
        "description": "Only layers 40-48",
        "active_layers": list(range(40, 49)),
        "weights": None,
    },
    "u_shaped": {
        "description": "Early (0-16) + Late (40-48), skip middle",
        "active_layers": list(range(0, 17)) + list(range(40, 49)),
        "weights": None,
    },
    "anti_u": {
        "description": "Middle only (15-35), exclude early and late",
        "active_layers": list(range(15, 36)),
        "weights": None,
    },
    "bottom_excluded": {
        "description": "Exclude bottom 25 layers (0-24)",
        "active_layers": list(range(25, 49)),
        "weights": None,
    },
    "gradual": {
        "description": "Linear weight increase 0→1 across layers",
        "active_layers": list(range(49)),
        "weights": {i: (i + 1) / 49 for i in range(49)},
    },
    "exponential": {
        "description": "Exponential weight increase toward late layers",
        "active_layers": list(range(49)),
        "weights": {i: np.exp(i / 10) for i in range(49)},
    },
}

QUICK_CONFIGS = ["baseline", "late_heavy", "top_contributors"]
QUICK_PROMPTS = QUICK_CATEGORY  # Use centralized quick category subset


def compute_metrics(frame: Image.Image, prompt: str) -> dict:
    """Compute quality metrics for a frame.

    Only computes SigLIP score - the only meaningful metric for understanding
    layer contributions to text-image alignment. Brightness/pixel statistics
    are meaningless for this analysis.
    """
    metrics = {}

    # SigLIP score - measures text-image alignment (the only metric that matters)
    try:
        from experiments.metrics.siglip_score import compute_siglip_score
        metrics["siglip_score"] = compute_siglip_score(prompt, frame)
    except Exception as e:
        metrics["siglip_score"] = None
        metrics["siglip_error"] = str(e)

    return metrics


def run_layer_blend_sweep(
    model_path: str = "models/LTX-2",
    output_dir: str = "experiments/results",
    seed: int = 42,
    quick: bool = False,
    num_inference_steps: int = 25,
    guidance_scale: float = 3.0,
    height: int = 512,
    width: int = 768,
    num_frames: int = 33,
    num_blocks_per_group: int = 1,
):
    """Run layer blend sweep experiment.

    Memory-optimized for 24GB GPUs using:
    - 8-bit quantized text encoder (~13GB)
    - Group offloading for transformer blocks
    - Sequential loading: encode → offload → generate
    """
    from diffusers.utils import export_to_video

    # Clear GPU memory before starting
    cleanup_memory()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"ltx2_layer_blend_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "images").mkdir(exist_ok=True)
    (output_path / "videos").mkdir(exist_ok=True)
    (output_path / "metadata").mkdir(exist_ok=True)

    print("=" * 60)
    print("LTX-2 Layer Blend Sweep (Memory-Optimized)")
    print("=" * 60)
    print(f"Output: {output_path}")
    print(f"Seed: {seed}")
    print(f"Quick mode: {quick}")

    # Select configs and prompts
    configs = QUICK_CONFIGS if quick else list(BLEND_CONFIGS.keys())
    prompts = QUICK_PROMPTS if quick else list(TEST_PROMPTS.keys())

    total_gens = len(configs) * len(prompts)
    print(f"Generations: {len(configs)} configs × {len(prompts)} prompts = {total_gens}")

    # ==========================================================================
    # PHASE 1: Text Encoding (8-bit quantized, ~13GB VRAM)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Text Encoding (8-bit quantized)")
    print("=" * 60)

    print(f"Loading text encoder... (GPU: {get_gpu_memory():.2f}GB)")
    text_encoder, tokenizer = load_text_encoder_8bit(model_path)
    print(f"Text encoder loaded (GPU: {get_gpu_memory():.2f}GB)")

    # Pre-encode all (config, prompt) combinations
    # Structure: embeddings_cache[(config_name, prompt_id)] = (prompt_embeds, attention_mask, seq_len)
    embeddings_cache = {}

    encode_idx = 0
    for config_name in configs:
        config = BLEND_CONFIGS[config_name]

        # Build layer weights for this config
        layer_weights = create_layer_weights(
            active_layers=config["active_layers"],
            weights=config["weights"],
            num_layers=49,
            normalize=True,
        )

        for prompt_id in prompts:
            encode_idx += 1
            prompt_text = TEST_PROMPTS[prompt_id]
            print(f"  [{encode_idx}/{total_gens}] Encoding: {config_name} × {prompt_id}")

            # Encode with layer weights applied
            hidden_states, attention_mask, seq_len = encode_prompt_with_layer_weights(
                text_encoder,
                tokenizer,
                prompt_text,
                layer_weights=layer_weights,
            )

            # Pack to create prompt_embeds (same as pipeline._pack_text_embeds)
            prompt_embeds = pack_text_embeds(
                hidden_states,
                seq_len,
                device=torch.device("cuda"),
            )

            # Cache on CPU to free VRAM
            embeddings_cache[(config_name, prompt_id)] = {
                "prompt_embeds": prompt_embeds.cpu(),
                "attention_mask": attention_mask.cpu(),
                "sequence_length": seq_len,
                "prompt": prompt_text,
            }

    print(f"\nEncoded {len(embeddings_cache)} prompt/config combinations")

    # ==========================================================================
    # PHASE 2: Offload Text Encoder
    # ==========================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Offloading Text Encoder")
    print("=" * 60)

    print(f"Before offload (GPU: {get_gpu_memory():.2f}GB)")
    del text_encoder, tokenizer
    cleanup_memory()
    print(f"After offload (GPU: {get_gpu_memory():.2f}GB)")

    # ==========================================================================
    # PHASE 3: Generation (Group Offloading, ~5GB VRAM)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Video Generation (Group Offloading)")
    print("=" * 60)

    print(f"Loading pipeline with group offloading...")
    pipe = load_pipeline_with_offloading(
        model_path,
        num_blocks_per_group=num_blocks_per_group,
        use_stream=True,
    )
    print(f"Pipeline loaded (GPU: {get_gpu_memory():.2f}GB)")

    # Results storage
    all_results = []
    config_summaries = {}

    gen_idx = 0
    for config_name in configs:
        config = BLEND_CONFIGS[config_name]
        print(f"\n{'='*60}")
        print(f"Config: {config_name}")
        print(f"  {config['description']}")
        print(f"  Active layers: {len(config['active_layers'])}")
        print("=" * 60)

        config_results = []

        for prompt_id in prompts:
            gen_idx += 1
            print(f"\n[{gen_idx}/{total_gens}] {config_name} × {prompt_id}")

            # Get cached embeddings
            cached = embeddings_cache[(config_name, prompt_id)]
            prompt_embeds = cached["prompt_embeds"].to("cuda")
            prompt_attention_mask = cached["attention_mask"].to("cuda")
            prompt_text = cached["prompt"]

            start_time = time.time()

            # Generate with pre-computed embeddings
            generator = torch.Generator(device="cpu").manual_seed(seed)

            output = pipe(
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
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

            # Save outputs
            sample_name = f"{config_name}_{prompt_id}"
            image_path = output_path / "images" / f"{sample_name}.png"
            video_path = output_path / "videos" / f"{sample_name}.mp4"

            first_frame.save(image_path)
            export_to_video(frames, str(video_path), fps=24)

            # Build result
            result = {
                "config": {
                    "blend_name": config_name,
                    "blend_description": config["description"],
                    "active_layers": config["active_layers"],
                    "num_active_layers": len(config["active_layers"]),
                    "prompt_id": prompt_id,
                    "seed": seed,
                },
                "generation_time_seconds": gen_time,
                "output_path": str(image_path.relative_to(output_path)),
                "video_path": str(video_path.relative_to(output_path)),
                **metrics,
            }

            # Save metadata
            metadata_path = output_path / "metadata" / f"{sample_name}.json"
            with open(metadata_path, "w") as f:
                json.dump(result, f, indent=2)

            all_results.append(result)
            config_results.append(result)

            siglip_str = f"{metrics['siglip_score']:.4f}" if metrics.get("siglip_score") is not None else "N/A"
            print(f"  Time: {gen_time:.1f}s | SigLIP: {siglip_str} | GPU: {get_gpu_memory():.1f}GB")

            # Cleanup between generations
            del frames, output, first_frame, prompt_embeds
            cleanup_memory()

        # Summarize config - only SigLIP matters
        siglip_scores = [r["siglip_score"] for r in config_results if r.get("siglip_score") is not None]
        config_summaries[config_name] = {
            "description": config["description"],
            "num_active_layers": len(config["active_layers"]),
            "mean_siglip": float(np.mean(siglip_scores)) if siglip_scores else None,
            "std_siglip": float(np.std(siglip_scores)) if len(siglip_scores) > 1 else None,
            "num_samples": len(siglip_scores),
        }

    # Save summary
    summary = {
        "experiment": "ltx2_layer_blend_sweep",
        "timestamp": timestamp,
        "parameters": {
            "seed": seed,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "resolution": f"{height}x{width}",
            "num_frames": num_frames,
        },
        "total_generations": len(all_results),
        "config_summaries": config_summaries,
        "results": all_results,
    }

    summary_path = output_path / "ltx2_layer_blend_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Generate comparison visualization
    print("\n" + "=" * 60)
    print("SUMMARY - SigLIP Scores by Layer Configuration")
    print("=" * 60)

    print(f"\n{'Config':<20} {'Layers':<8} {'SigLIP':<12} {'Std':<10} {'N':<5}")
    print("-" * 55)

    # Sort by SigLIP score (descending) for easier analysis
    sorted_configs = sorted(
        config_summaries.items(),
        key=lambda x: x[1].get('mean_siglip') or 0,
        reverse=True
    )

    for config_name, stats in sorted_configs:
        siglip_str = f"{stats['mean_siglip']:.4f}" if stats.get('mean_siglip') is not None else "N/A"
        std_str = f"{stats['std_siglip']:.4f}" if stats.get('std_siglip') is not None else "N/A"
        print(f"{config_name:<20} {stats['num_active_layers']:<8} {siglip_str:<12} {std_str:<10} {stats['num_samples']:<5}")

    # Create visualization - SigLIP only with error bars
    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort by SigLIP for visual clarity
    sorted_names = [c[0] for c in sorted_configs]
    x = range(len(sorted_names))

    siglip_values = [config_summaries[c].get("mean_siglip") or 0 for c in sorted_names]
    siglip_stds = [config_summaries[c].get("std_siglip") or 0 for c in sorted_names]
    layer_counts = [config_summaries[c]["num_active_layers"] for c in sorted_names]

    # Color bars by layer count (fewer = more compute efficient)
    colors = plt.cm.viridis([count / 49 for count in layer_counts])

    bars = ax.bar(x, siglip_values, yerr=siglip_stds, capsize=4, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({layer_counts[i]}L)" for i, n in enumerate(sorted_names)], rotation=45, ha="right")
    ax.set_ylabel("SigLIP Score (text-image alignment)")
    ax.set_xlabel("Layer Configuration (active layers)")
    ax.set_title("LTX-2 Layer Blend Sweep: Text-Image Alignment by Configuration")

    # Add baseline reference line
    if "baseline" in config_summaries and config_summaries["baseline"].get("mean_siglip"):
        ax.axhline(y=config_summaries["baseline"]["mean_siglip"], color='red', linestyle='--', label='Baseline', alpha=0.7)
        ax.legend()

    plt.tight_layout()
    plot_path = output_path / "layer_blend_comparison.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved plot to: {plot_path}")
    plt.close()

    # Cleanup
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nResults saved to: {output_path}")
    print("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="LTX-2 Layer Blend Sweep (Memory-Optimized)")
    parser.add_argument("--model-path", default="models/LTX-2", help="Path to LTX-2 model")
    parser.add_argument("--output-dir", default="experiments/results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Quick test (3 configs × 2 prompts)")
    parser.add_argument("--steps", type=int, default=25, help="Inference steps")
    parser.add_argument("--cfg", type=float, default=3.0, help="Guidance scale")
    parser.add_argument("--height", type=int, default=512, help="Video height")
    parser.add_argument("--width", type=int, default=768, help="Video width")
    parser.add_argument("--frames", type=int, default=33, help="Number of frames")
    parser.add_argument(
        "--blocks-per-group",
        type=int,
        default=1,
        help="Transformer blocks per offload group (1=min VRAM, higher=faster)",
    )
    args = parser.parse_args()

    run_layer_blend_sweep(
        model_path=args.model_path,
        output_dir=args.output_dir,
        seed=args.seed,
        quick=args.quick,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        num_blocks_per_group=args.blocks_per_group,
    )


if __name__ == "__main__":
    main()
