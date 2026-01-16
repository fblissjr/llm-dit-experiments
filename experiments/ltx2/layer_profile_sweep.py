#!/usr/bin/env python3
"""
LTX-2 Layer Profile Sweep Experiment

Last Updated: 2026-01-16

Generate videos using each of the 49 Gemma layers in isolation to understand
what each layer contributes to video generation. Outputs are viewer-compatible
for visual comparison and analysis.

Discovery Question: "What does each Gemma layer contribute to video generation?"

Method:
1. Generate video for each layer in isolation (zero others)
2. Create 49xN grid (layers x prompts)
3. Extract first frame as PNG for viewer
4. Compute SigLIP2 + ImageReward per sample

Memory-optimized for 24GB GPUs (RTX 4090):
- Text encoder: 8-bit quantized (~13GB instead of ~54GB)
- Pipeline: Group offloading for transformer blocks
- Sequential loading: Encode first, then offload, then generate

Output Structure (viewer-compatible):
    experiments/results/ltx2_layer_profile_{timestamp}/
    ├── images/           # First frames as PNG
    ├── videos/           # Full MP4s (optional)
    ├── metadata/         # Per-sample JSON with metrics
    │   └── layer_00_prompt_001.json
    └── ltx2_layer_profile_summary.json

Usage:
    # Full sweep (49 layers x 10 prompts = 490 generations)
    uv run python experiments/ltx2/layer_profile_sweep.py

    # Quick test (3 layers x 3 prompts)
    uv run python experiments/ltx2/layer_profile_sweep.py --quick

    # Custom layers
    uv run python experiments/ltx2/layer_profile_sweep.py --layers 0 1 23 47 48

    # Skip video saving (just first frames)
    uv run python experiments/ltx2/layer_profile_sweep.py --no-save-videos

    # Skip metric computation (faster)
    uv run python experiments/ltx2/layer_profile_sweep.py --skip-metrics
"""

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# LTX-2 Gemma configuration
NUM_GEMMA_LAYERS = 49  # Layers 0-48
GEMMA_HIDDEN_DIM = 3840

# Import prompts from centralized module
# These match the official LTX-2 prompting guide format (100+ words, dialogue, etc.)
from experiments.ltx2.prompts import (
    get_all_prompts,
    QUICK_OFFICIAL,
    QUICK_CATEGORY,
)

# Import memory-efficient utilities for 24GB GPUs
from experiments.ltx2.memory_utils import (
    load_text_encoder_8bit,
    encode_prompt_with_layer_masking,
    pack_text_embeds,
    load_pipeline_with_offloading,
    cleanup_memory,
    get_gpu_memory,
)

# Full test prompts (official + category)
TEST_PROMPTS = get_all_prompts(quick=False)

# Quick test subset (3 official + 2 category = 5 prompts)
QUICK_PROMPTS = get_all_prompts(quick=True)

QUICK_LAYERS = [0, 24, 48]  # Early, middle, late


# NOTE: create_layer_masking_hook was removed in favor of memory-efficient
# encode_prompt_with_layer_masking() from memory_utils.py which applies
# masking during encoding phase before offloading the text encoder.


def compute_frame_statistics(frames: list) -> dict:
    """Compute statistics on generated frames.

    NOTE: Brightness/pixel statistics are NOT useful for understanding
    layer contributions. This function exists for backwards compatibility
    but only returns an empty dict. Use SigLIP score for meaningful analysis.
    """
    # Brightness metrics removed - they tell us nothing about layer contribution
    # or text-image alignment. Keep function for API compatibility.
    return {}


def save_metadata(
    output_dir: Path,
    layer_idx: int,
    prompt_id: str,
    prompt_text: str,
    seed: int,
    image_path: Path,
    video_path: Path | None,
    generation_time: float,
    frame_stats: dict,
    siglip_score: float | None = None,
):
    """Save viewer-compatible metadata JSON."""
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    filename = f"layer_{layer_idx:02d}_{prompt_id}.json"
    filepath = metadata_dir / filename

    data = {
        "config": {
            "prompt_id": prompt_id,
            "variable_name": "layer_index",
            "variable_value": layer_idx,
            "seed": seed,
            "prompt_text": prompt_text,
        },
        "siglip_score": siglip_score,
        "generation_time_seconds": generation_time,
        "output_path": str(image_path.relative_to(output_dir.parent.parent.parent)),
        "video_path": str(video_path.relative_to(output_dir.parent.parent.parent)) if video_path else None,
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return filepath


def run_layer_profile_sweep(
    output_base: str = "experiments/results",
    model_path: str = "models/LTX-2",
    layers_to_test: list[int] | None = None,
    prompts: dict[str, str] | None = None,
    save_videos: bool = True,
    compute_metrics: bool = True,
    seed: int = 42,
    height: int = 512,
    width: int = 768,
    num_frames: int = 33,
    num_inference_steps: int = 25,
    guidance_scale: float = 3.0,
    masking_mode: str = "soft",
    num_blocks_per_group: int = 1,
):
    """
    Run the full layer profile sweep experiment.

    Memory-optimized for 24GB GPUs using:
    - 8-bit quantized text encoder (~13GB)
    - Group offloading for transformer blocks
    - Sequential loading: encode → offload → generate

    Args:
        output_base: Base directory for results
        model_path: Path to LTX-2 model
        layers_to_test: List of layer indices (0-48) to test. None = all 49 layers
        prompts: Dict of prompt_id -> prompt_text. None = use TEST_PROMPTS
        save_videos: Whether to save full MP4 videos
        compute_metrics: Whether to compute SigLIP2 and ImageReward scores
        seed: Random seed for reproducibility
        height: Video height
        width: Video width
        num_frames: Number of frames to generate
        num_inference_steps: Diffusion steps
        guidance_scale: CFG scale
        masking_mode: How to mask inactive layers: "soft", "zero", or "weighted"
            - "soft" (recommended): Replace inactive with per-layer mean
            - "zero": Zero out (creates OOD artifacts)
            - "weighted": Scale active layers to preserve norm
        num_blocks_per_group: Transformer blocks per offload group (1=min VRAM)

    Returns:
        Path to results directory
    """
    from diffusers.utils import export_to_video

    # Clear GPU memory before starting
    cleanup_memory()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Setup
    if layers_to_test is None:
        layers_to_test = list(range(NUM_GEMMA_LAYERS))  # All 49 layers

    if prompts is None:
        prompts = TEST_PROMPTS

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base) / f"ltx2_layer_profile_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    if save_videos:
        videos_dir = output_dir / "videos"
        videos_dir.mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("LTX-2 Layer Profile Sweep (Memory-Optimized)")
    logger.info("=" * 60)
    logger.info(f"Output: {output_dir}")
    logger.info(f"Layers: {len(layers_to_test)} ({min(layers_to_test)}-{max(layers_to_test)})")
    logger.info(f"Prompts: {len(prompts)}")
    logger.info(f"Masking mode: {masking_mode}")
    total = len(layers_to_test) * len(prompts)
    logger.info(f"Total generations: {total}")

    # ==========================================================================
    # PHASE 1: Text Encoding (8-bit quantized, ~13GB VRAM)
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: Text Encoding (8-bit quantized)")
    logger.info("=" * 60)

    logger.info(f"Loading text encoder... (GPU: {get_gpu_memory():.2f}GB)")
    text_encoder, tokenizer = load_text_encoder_8bit(model_path)
    logger.info(f"Text encoder loaded (GPU: {get_gpu_memory():.2f}GB)")

    # Pre-encode all (layer, prompt) combinations
    # Structure: embeddings_cache[(layer_idx, prompt_id)] = (prompt_embeds, prompt_text)
    embeddings_cache = {}

    # Encode negative prompt once (empty string for CFG)
    # This is required when using guidance_scale > 1.0 with pre-computed embeddings
    logger.info("\n  Encoding negative prompt (empty string for CFG)")
    neg_inputs = tokenizer("", return_tensors="pt", padding="max_length", max_length=512, truncation=True)
    neg_input_ids = neg_inputs["input_ids"].to(text_encoder.device)
    neg_attention_mask = neg_inputs["attention_mask"].to(text_encoder.device)

    with torch.no_grad():
        neg_outputs = text_encoder(
            input_ids=neg_input_ids,
            attention_mask=neg_attention_mask,
            output_hidden_states=True,
        )

    # Include ALL hidden states (embedding + 48 layers = 49 total)
    neg_hidden_states = torch.stack(neg_outputs.hidden_states[:49], dim=-1)
    neg_seq_len = neg_attention_mask.sum().item()
    negative_prompt_embeds = pack_text_embeds(
        neg_hidden_states,
        neg_seq_len,
        device=torch.device("cuda"),
    ).cpu()
    negative_attention_mask = neg_attention_mask.cpu()

    encode_idx = 0
    for layer_idx in layers_to_test:
        logger.info(f"\n  Layer {layer_idx} (single layer active)")

        for prompt_id, prompt_text in prompts.items():
            encode_idx += 1
            logger.info(f"    [{encode_idx}/{total}] Encoding: layer_{layer_idx:02d} × {prompt_id}")

            # Encode with layer masking applied
            hidden_states, attention_mask, seq_len = encode_prompt_with_layer_masking(
                text_encoder,
                tokenizer,
                prompt_text,
                active_layers=[layer_idx],
                masking_mode=masking_mode,
            )

            # Pack to create prompt_embeds
            prompt_embeds = pack_text_embeds(
                hidden_states,
                seq_len,
                device=torch.device("cuda"),
            )

            # Cache on CPU to free VRAM (include negative embeds for CFG)
            embeddings_cache[(layer_idx, prompt_id)] = {
                "prompt_embeds": prompt_embeds.cpu(),
                "attention_mask": attention_mask.cpu(),
                "negative_prompt_embeds": negative_prompt_embeds,
                "negative_prompt_attention_mask": negative_attention_mask,
                "prompt": prompt_text,
            }

    logger.info(f"\nEncoded {len(embeddings_cache)} layer/prompt combinations (+ negative prompt for CFG)")

    # ==========================================================================
    # PHASE 2: Offload Text Encoder
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Offloading Text Encoder")
    logger.info("=" * 60)

    logger.info(f"Before offload (GPU: {get_gpu_memory():.2f}GB)")
    del text_encoder, tokenizer
    cleanup_memory()
    logger.info(f"After offload (GPU: {get_gpu_memory():.2f}GB)")

    # ==========================================================================
    # PHASE 3: Generation (Group Offloading, ~5GB VRAM)
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: Video Generation (Group Offloading)")
    logger.info("=" * 60)

    logger.info("Loading pipeline with group offloading...")
    pipe = load_pipeline_with_offloading(
        model_path,
        num_blocks_per_group=num_blocks_per_group,
        use_stream=True,
    )
    logger.info(f"Pipeline loaded (GPU: {get_gpu_memory():.2f}GB)")

    # Load SigLIP scorer (the only meaningful metric)
    siglip_scorer = None

    if compute_metrics:
        try:
            from experiments.metrics.siglip_score import SigLIPScorer
            siglip_scorer = SigLIPScorer()
            logger.info("SigLIP2 scorer loaded (text-image alignment metric)")
        except ImportError as e:
            logger.warning(f"Could not load SigLIP scorer: {e}")

    # Results accumulator
    all_results = []

    # Main generation loop
    count = 0

    for layer_idx in layers_to_test:
        logger.info(f"\n{'='*50}")
        logger.info(f"Layer {layer_idx} (single layer active)")
        logger.info("=" * 50)

        for prompt_id, prompt_text in prompts.items():
            count += 1
            logger.info(f"\n  [{count}/{total}] layer_{layer_idx:02d} × {prompt_id}")

            # Get cached embeddings
            cached = embeddings_cache[(layer_idx, prompt_id)]
            prompt_embeds = cached["prompt_embeds"].to("cuda")
            prompt_attention_mask = cached["attention_mask"].to("cuda")
            negative_prompt_embeds = cached["negative_prompt_embeds"].to("cuda")
            negative_prompt_attention_mask = cached["negative_prompt_attention_mask"].to("cuda")

            start_time = time.time()

            try:
                # Generate video
                generator = torch.Generator(device="cpu").manual_seed(seed)

                output = pipe(
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_prompt_attention_mask=negative_prompt_attention_mask,
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
                image_filename = f"layer_{layer_idx:02d}_{prompt_id}.png"
                image_path = images_dir / image_filename
                first_frame.save(image_path)

                # Save video if requested
                video_path = None
                if save_videos:
                    video_filename = f"layer_{layer_idx:02d}_{prompt_id}.mp4"
                    video_path = videos_dir / video_filename
                    export_to_video(frames, str(video_path), fps=24)

                # Compute frame statistics (empty - kept for API compat)
                frame_stats = compute_frame_statistics(frames)

                # Compute SigLIP score (only meaningful metric)
                siglip_score = None

                if siglip_scorer:
                    try:
                        siglip_score = siglip_scorer.score(prompt_text, first_frame)
                    except Exception as e:
                        logger.warning(f"SigLIP scoring failed: {e}")

                # Save metadata
                meta_path = save_metadata(
                    output_dir=output_dir,
                    layer_idx=layer_idx,
                    prompt_id=prompt_id,
                    prompt_text=prompt_text,
                    seed=seed,
                    image_path=image_path,
                    video_path=video_path,
                    generation_time=gen_time,
                    frame_stats=frame_stats,
                    siglip_score=siglip_score,
                )

                result = {
                    "layer_idx": layer_idx,
                    "prompt_id": prompt_id,
                    "generation_time": gen_time,
                    "siglip_score": siglip_score,
                }
                all_results.append(result)

                # Log progress
                siglip_str = f"{siglip_score:.4f}" if siglip_score is not None else "N/A"
                logger.info(f"  Time: {gen_time:.1f}s | SigLIP: {siglip_str} | GPU: {get_gpu_memory():.1f}GB")

            except Exception as e:
                logger.error(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "layer_idx": layer_idx,
                    "prompt_id": prompt_id,
                    "error": str(e),
                })

            # Memory cleanup
            del prompt_embeds, prompt_attention_mask
            del negative_prompt_embeds, negative_prompt_attention_mask
            cleanup_memory()

    # Save summary
    summary = {
        "experiment_type": "ltx2_layer_profile",
        "timestamp": timestamp,
        "config": {
            "layers_tested": layers_to_test,
            "prompts": prompts,
            "seed": seed,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "model_path": model_path,
            "masking_mode": masking_mode,
        },
        "results": all_results,
        "statistics": compute_sweep_statistics(all_results),
    }

    summary_path = output_dir / "ltx2_layer_profile_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("SWEEP COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Total generations: {count}")
    logger.info(f"Successful: {len([r for r in all_results if 'error' not in r])}")
    logger.info(f"Errors: {len([r for r in all_results if 'error' in r])}")

    # Print analysis hints
    print_analysis_summary(all_results, layers_to_test, prompts)

    # Cleanup
    del pipe
    if siglip_scorer:
        del siglip_scorer
    cleanup_memory()

    return output_dir


def compute_sweep_statistics(results: list[dict]) -> dict:
    """Compute aggregate statistics from sweep results.

    Only computes SigLIP statistics - the only meaningful metric for
    understanding layer contributions to text-image alignment.
    """
    valid = [r for r in results if "error" not in r]

    if not valid:
        return {"error": "No valid results"}

    # Per-layer statistics - SigLIP only
    layer_stats = {}
    layers = sorted(set(r["layer_idx"] for r in valid))

    for layer_idx in layers:
        layer_results = [r for r in valid if r["layer_idx"] == layer_idx]
        siglip_scores = [r["siglip_score"] for r in layer_results if r.get("siglip_score") is not None]

        layer_stats[f"layer_{layer_idx:02d}"] = {
            "mean_gen_time": float(np.mean([r["generation_time"] for r in layer_results])),
            "mean_siglip": float(np.mean(siglip_scores)) if siglip_scores else None,
            "std_siglip": float(np.std(siglip_scores)) if len(siglip_scores) > 1 else None,
            "num_samples": len(siglip_scores),
        }

    # Per-prompt statistics - SigLIP only
    prompt_stats = {}
    prompt_ids = sorted(set(r["prompt_id"] for r in valid))

    for prompt_id in prompt_ids:
        prompt_results = [r for r in valid if r["prompt_id"] == prompt_id]
        siglip_scores = [r["siglip_score"] for r in prompt_results if r.get("siglip_score") is not None]
        prompt_stats[prompt_id] = {
            "mean_siglip": float(np.mean(siglip_scores)) if siglip_scores else None,
            "std_siglip": float(np.std(siglip_scores)) if len(siglip_scores) > 1 else None,
        }

    return {
        "per_layer": layer_stats,
        "per_prompt": prompt_stats,
        "total_valid": len(valid),
        "total_errors": len(results) - len(valid),
    }


def print_analysis_summary(results: list[dict], layers: list[int], prompts: dict):
    """Print a quick analysis summary - SigLIP scores only."""
    valid = [r for r in results if "error" not in r]

    if not valid:
        return

    print("\n" + "=" * 60)
    print("LAYER CONTRIBUTION ANALYSIS (SigLIP)")
    print("=" * 60)

    # Calculate SigLIP scores per layer
    layer_siglip = {}
    for layer_idx in layers:
        layer_results = [r for r in valid if r["layer_idx"] == layer_idx and r.get("siglip_score")]
        if layer_results:
            scores = [r["siglip_score"] for r in layer_results]
            layer_siglip[layer_idx] = {
                "mean": np.mean(scores),
                "std": np.std(scores) if len(scores) > 1 else 0,
                "n": len(scores),
            }

    if layer_siglip:
        sorted_siglip = sorted(layer_siglip.items(), key=lambda x: x[1]["mean"], reverse=True)

        print("\nTop 10 layers by SigLIP score (best text-image alignment):")
        print(f"  {'Layer':<8} {'SigLIP':<10} {'Std':<10}")
        print("  " + "-" * 28)
        for layer_idx, stats in sorted_siglip[:10]:
            print(f"  Layer {layer_idx:2d}  {stats['mean']:.4f}    {stats['std']:.4f}")

        print("\nBottom 10 layers:")
        print(f"  {'Layer':<8} {'SigLIP':<10} {'Std':<10}")
        print("  " + "-" * 28)
        for layer_idx, stats in sorted_siglip[-10:]:
            print(f"  Layer {layer_idx:2d}  {stats['mean']:.4f}    {stats['std']:.4f}")

        # Layer regions analysis
        early = [layer_siglip[i]["mean"] for i in range(0, 17) if i in layer_siglip]
        middle = [layer_siglip[i]["mean"] for i in range(17, 35) if i in layer_siglip]
        late = [layer_siglip[i]["mean"] for i in range(35, 49) if i in layer_siglip]

        print("\nRegion averages:")
        if early:
            print(f"  Early (0-16):   {np.mean(early):.4f}")
        if middle:
            print(f"  Middle (17-34): {np.mean(middle):.4f}")
        if late:
            print(f"  Late (35-48):   {np.mean(late):.4f}")

    print("\nView results in the experiment viewer:")
    print("  uv run experiments/viewer/server.py")
    print("  # Navigate to http://localhost:7861")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="LTX-2 Layer Profile Sweep Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output-base",
        default="experiments/results",
        help="Base directory for results",
    )
    parser.add_argument(
        "--model-path",
        default="models/LTX-2",
        help="Path to LTX-2 model",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        help="Layer indices to test (default: all 49)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 3 layers x 3 prompts",
    )
    parser.add_argument(
        "--no-save-videos",
        action="store_true",
        help="Don't save full MP4 videos (only first frames)",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Skip SigLIP2 and ImageReward computation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Video height (default: 512)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Video width (default: 768)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=33,
        help="Number of frames (default: 33)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="Inference steps (default: 25)",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=3.0,
        help="Guidance scale (default: 3.0)",
    )
    parser.add_argument(
        "--masking-mode",
        choices=["soft", "zero", "weighted"],
        default="soft",
        help="How to mask inactive layers (default: soft). "
             "soft=replace with mean (recommended), "
             "zero=zero out (creates OOD artifacts), "
             "weighted=scale to preserve norm",
    )
    parser.add_argument(
        "--blocks-per-group",
        type=int,
        default=1,
        help="Transformer blocks per offload group (1=min VRAM, higher=faster)",
    )

    args = parser.parse_args()

    # Determine layers and prompts
    if args.quick:
        layers = QUICK_LAYERS
        prompts = QUICK_PROMPTS
    else:
        layers = args.layers  # None = all
        prompts = None  # None = all TEST_PROMPTS

    run_layer_profile_sweep(
        output_base=args.output_base,
        model_path=args.model_path,
        layers_to_test=layers,
        prompts=prompts,
        save_videos=not args.no_save_videos,
        compute_metrics=not args.skip_metrics,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        masking_mode=args.masking_mode,
        num_blocks_per_group=args.blocks_per_group,
    )


if __name__ == "__main__":
    main()
