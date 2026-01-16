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
from typing import Callable

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

# Test prompts covering different visual attributes
# LTX-2 requires detailed, paragraph-style prompts (4-8 sentences)
# See experiments/ltx2/prompting_guide.md for details
TEST_PROMPTS = {
    "text_render": (
        "A close-up shot of a white wall with the word HELLO painted in bold red letters. "
        "The camera slowly zooms in as warm afternoon sunlight casts soft shadows across the textured surface. "
        "The letters appear hand-painted with visible brush strokes. "
        "Dust particles float gently through the beam of light."
    ),
    "simple_obj": (
        "A medium shot of a bright red rubber ball resting on a pristine white surface. "
        "The lighting is soft and even, creating gentle shadows beneath the ball. "
        "The camera holds steady as dust particles float through a beam of sunlight. "
        "The ball's glossy surface reflects the ambient light with a subtle sheen."
    ),
    "animal": (
        "A golden retriever runs joyfully through a sun-dappled park, its fur gleaming in the warm afternoon light. "
        "The camera tracks alongside as the dog bounds across lush green grass, tongue out and tail wagging. "
        "Birds chirp in the background as leaves rustle in a gentle breeze. "
        "The scene captures the pure happiness of a dog at play."
    ),
    "complex_scene": (
        "A bustling city street at night comes alive with neon signs reflecting off rain-slicked pavement. "
        "Crowds of people in dark coats hurry past storefronts while taxis honk in the distance. "
        "The camera slowly pans across the scene capturing the vibrant energy of urban nightlife. "
        "Steam rises from a nearby food cart as streetlights cast long shadows."
    ),
    "abstract": (
        "A dreamlike surreal landscape unfolds with floating islands suspended in a pink and purple sky. "
        "Ethereal mist swirls around ancient stone structures as bioluminescent plants pulse with soft light. "
        "The camera drifts slowly through this otherworldly realm. "
        "Crystalline formations catch and refract light in rainbow patterns."
    ),
    "spatial": (
        "A tabby cat sits regally on top of a weathered wooden crate in front of a cozy cottage. "
        "Warm golden hour light bathes the scene as the cat surveys its domain with half-closed eyes. "
        "The camera holds a static medium shot capturing the peaceful rural atmosphere. "
        "Ivy climbs the cottage walls while flower pots line the entrance."
    ),
    "lighting": (
        "A person's face is dramatically illuminated by a single flickering candle in otherwise complete darkness. "
        "The warm orange glow dances across their features creating deep shadows and highlights. "
        "The camera holds a close-up shot as the flame gently sways. "
        "Wisps of smoke curl upward catching the light."
    ),
    "color": (
        "Three apples arranged in a row on a marble countertop, each a different vibrant color. "
        "A red apple on the left gleams with moisture, a green Granny Smith sits in the middle, and a golden yellow apple rests on the right. "
        "Soft diffused lighting from above creates subtle reflections on the polished surface. "
        "The camera slowly dollies across the arrangement."
    ),
    "motion": (
        "A sleek sports car races down an empty highway at high speed, its red paint catching the sunlight. "
        "Motion blur streaks the background as the car cuts through the frame. "
        "The camera tracks alongside in a dynamic side shot capturing the sense of velocity. "
        "Heat waves shimmer off the asphalt as the engine roars."
    ),
    "person": (
        "A lone figure walks slowly through heavy rain on a city sidewalk, holding a bright red umbrella. "
        "Raindrops splash against the pavement creating small ripples in puddles. "
        "The camera follows from behind as streetlights create golden halos in the downpour. "
        "Their silhouette is reflected in the wet pavement below."
    ),
}

# Quick test subset
QUICK_PROMPTS = {
    "text_render": TEST_PROMPTS["text_render"],
    "animal": TEST_PROMPTS["animal"],
    "abstract": TEST_PROMPTS["abstract"],
}

QUICK_LAYERS = [0, 24, 48]  # Early, middle, late


def create_layer_masking_hook(
    pipe,
    active_layers: list[int],
) -> Callable:
    """
    Create a hook that masks inactive layers in Gemma embeddings.

    The hook replaces _get_gemma_prompt_embeds to zero out contributions
    from layers not in active_layers.

    Args:
        pipe: LTX2Pipeline instance
        active_layers: List of layer indices (0-48) to keep active

    Returns:
        Hook function to install via pipe._get_gemma_prompt_embeds = hook
    """
    active_set = set(active_layers)

    def masked_get_embeds(*args, **kwargs):
        """Hook that masks specific layers after text encoder."""
        prompt = kwargs.get("prompt", args[0] if args else None)
        num_videos_per_prompt = kwargs.get("num_videos_per_prompt", 1)
        max_sequence_length = kwargs.get("max_sequence_length", 1024)
        scale_factor = kwargs.get("scale_factor", 8)
        device = kwargs.get("device", pipe._execution_device)
        dtype = kwargs.get("dtype", pipe.text_encoder.dtype)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        # Tokenize
        text_inputs = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_attention_mask = text_inputs.attention_mask.to(device)

        # Get hidden states from all layers
        text_encoder_outputs = pipe.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        )
        text_encoder_hidden_states = text_encoder_outputs.hidden_states
        text_encoder_hidden_states = torch.stack(text_encoder_hidden_states, dim=-1)
        # Shape: [batch, seq, hidden_dim, num_layers] = [B, T, 3840, 49]

        # LAYER MASKING: Zero out excluded layers
        for layer_idx in range(NUM_GEMMA_LAYERS):
            if layer_idx not in active_set:
                text_encoder_hidden_states[:, :, :, layer_idx] = 0.0

        sequence_lengths = prompt_attention_mask.sum(dim=-1)

        # Pack text embeds (normalize and flatten)
        prompt_embeds = pipe._pack_text_embeds(
            text_encoder_hidden_states,
            sequence_lengths,
            device=device,
            padding_side=pipe.tokenizer.padding_side,
            scale_factor=scale_factor,
        )
        prompt_embeds = prompt_embeds.to(dtype=dtype)

        # Duplicate for multiple videos per prompt
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_videos_per_prompt, 1)

        return prompt_embeds, prompt_attention_mask

    return masked_get_embeds


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
    image_reward: float | None = None,
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
        "image_reward": image_reward,
        "generation_time_seconds": generation_time,
        "output_path": str(image_path.relative_to(output_dir.parent.parent.parent)),
        "video_path": str(video_path.relative_to(output_dir.parent.parent.parent)) if video_path else None,
        "frame_stats": frame_stats,
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
):
    """
    Run the full layer profile sweep experiment.

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

    Returns:
        Path to results directory
    """
    from diffusers import LTX2Pipeline
    from diffusers.utils import export_to_video

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
    logger.info("LTX-2 Layer Profile Sweep")
    logger.info("=" * 60)
    logger.info(f"Output: {output_dir}")
    logger.info(f"Layers: {len(layers_to_test)} ({min(layers_to_test)}-{max(layers_to_test)})")
    logger.info(f"Prompts: {len(prompts)}")
    logger.info(f"Total generations: {len(layers_to_test) * len(prompts)}")

    # Load pipeline
    logger.info("\nLoading LTX-2 pipeline...")
    pipe = LTX2Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_sequential_cpu_offload()

    # Store original method
    original_get_embeds = pipe._get_gemma_prompt_embeds

    # Load metric scorers if needed
    siglip_scorer = None
    imgreward_scorer = None

    if compute_metrics:
        try:
            from experiments.metrics.siglip_score import SigLIPScorer
            siglip_scorer = SigLIPScorer()
            logger.info("SigLIP2 scorer loaded")
        except ImportError as e:
            logger.warning(f"Could not load SigLIP scorer: {e}")

        try:
            from experiments.metrics.image_reward import ImageRewardScorer
            imgreward_scorer = ImageRewardScorer()
            logger.info("ImageReward scorer loaded")
        except ImportError as e:
            logger.warning(f"Could not load ImageReward scorer: {e}")

    # Results accumulator
    all_results = []

    # Main sweep loop
    total = len(layers_to_test) * len(prompts)
    count = 0

    for layer_idx in layers_to_test:
        logger.info(f"\n{'='*50}")
        logger.info(f"Layer {layer_idx} (single layer active)")
        logger.info("=" * 50)

        # Install layer masking hook for this single layer
        hook = create_layer_masking_hook(pipe, active_layers=[layer_idx])
        pipe._get_gemma_prompt_embeds = hook

        for prompt_id, prompt_text in prompts.items():
            count += 1
            logger.info(f"\n  [{count}/{total}] {prompt_id}: {prompt_text[:40]}...")

            start_time = time.time()

            try:
                # Generate video
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
                image_filename = f"layer_{layer_idx:02d}_{prompt_id}.png"
                image_path = images_dir / image_filename
                first_frame.save(image_path)

                # Save video if requested
                video_path = None
                if save_videos:
                    video_filename = f"layer_{layer_idx:02d}_{prompt_id}.mp4"
                    video_path = videos_dir / video_filename
                    export_to_video(frames, str(video_path), fps=24)

                # Compute frame statistics
                frame_stats = compute_frame_statistics(frames)

                # Compute metrics
                siglip_score = None
                image_reward = None

                if siglip_scorer:
                    try:
                        siglip_score = siglip_scorer.score(prompt_text, first_frame)
                    except Exception as e:
                        logger.warning(f"SigLIP scoring failed: {e}")

                if imgreward_scorer:
                    try:
                        image_reward = imgreward_scorer.score(prompt_text, first_frame)
                    except Exception as e:
                        logger.warning(f"ImageReward scoring failed: {e}")

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
                    image_reward=image_reward,
                )

                result = {
                    "layer_idx": layer_idx,
                    "prompt_id": prompt_id,
                    "generation_time": gen_time,
                    "siglip_score": siglip_score,
                    "image_reward": image_reward,
                    **frame_stats,
                }
                all_results.append(result)

                # Log progress
                metrics_str = ""
                if siglip_score is not None:
                    metrics_str += f" SL:{siglip_score:.3f}"
                if image_reward is not None:
                    metrics_str += f" IR:{image_reward:.2f}"

                logger.info(
                    f"  Time: {gen_time:.1f}s | "
                    f"Mean: {frame_stats['mean_brightness']:.1f} | "
                    f"Std: {frame_stats['std']:.1f}"
                    f"{metrics_str}"
                )

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
            gc.collect()
            torch.cuda.empty_cache()

        # Restore original method between layers for clean state
        pipe._get_gemma_prompt_embeds = original_get_embeds

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
    if imgreward_scorer:
        del imgreward_scorer
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir


def compute_sweep_statistics(results: list[dict]) -> dict:
    """Compute aggregate statistics from sweep results."""
    valid = [r for r in results if "error" not in r]

    if not valid:
        return {"error": "No valid results"}

    # Per-layer statistics
    layer_stats = {}
    layers = sorted(set(r["layer_idx"] for r in valid))

    for layer_idx in layers:
        layer_results = [r for r in valid if r["layer_idx"] == layer_idx]
        layer_stats[f"layer_{layer_idx:02d}"] = {
            "mean_brightness": np.mean([r["mean_brightness"] for r in layer_results]),
            "mean_std": np.mean([r["std"] for r in layer_results]),
            "mean_temporal_var": np.mean([r["temporal_variance"] for r in layer_results]),
            "mean_gen_time": np.mean([r["generation_time"] for r in layer_results]),
        }

        # Metrics if available
        siglip_scores = [r["siglip_score"] for r in layer_results if r.get("siglip_score") is not None]
        if siglip_scores:
            layer_stats[f"layer_{layer_idx:02d}"]["mean_siglip"] = np.mean(siglip_scores)

        imgreward_scores = [r["image_reward"] for r in layer_results if r.get("image_reward") is not None]
        if imgreward_scores:
            layer_stats[f"layer_{layer_idx:02d}"]["mean_imgreward"] = np.mean(imgreward_scores)

    # Per-prompt statistics
    prompt_stats = {}
    prompt_ids = sorted(set(r["prompt_id"] for r in valid))

    for prompt_id in prompt_ids:
        prompt_results = [r for r in valid if r["prompt_id"] == prompt_id]
        prompt_stats[prompt_id] = {
            "mean_brightness": np.mean([r["mean_brightness"] for r in prompt_results]),
            "std_across_layers": np.std([r["mean_brightness"] for r in prompt_results]),
        }

    return {
        "per_layer": layer_stats,
        "per_prompt": prompt_stats,
        "total_valid": len(valid),
        "total_errors": len(results) - len(valid),
    }


def print_analysis_summary(results: list[dict], layers: list[int], prompts: dict):
    """Print a quick analysis summary."""
    valid = [r for r in results if "error" not in r]

    if not valid:
        return

    print("\n" + "=" * 60)
    print("QUICK ANALYSIS")
    print("=" * 60)

    # Find layers with highest/lowest mean brightness
    layer_brightness = {}
    for layer_idx in layers:
        layer_results = [r for r in valid if r["layer_idx"] == layer_idx]
        if layer_results:
            layer_brightness[layer_idx] = np.mean([r["mean_brightness"] for r in layer_results])

    if layer_brightness:
        sorted_layers = sorted(layer_brightness.items(), key=lambda x: x[1])
        print("\nLayers by mean brightness (low to high):")
        for layer_idx, brightness in sorted_layers[:5]:
            print(f"  Layer {layer_idx:2d}: {brightness:.1f}")
        print("  ...")
        for layer_idx, brightness in sorted_layers[-5:]:
            print(f"  Layer {layer_idx:2d}: {brightness:.1f}")

    # Find layers with highest SigLIP if available
    layer_siglip = {}
    for layer_idx in layers:
        layer_results = [r for r in valid if r["layer_idx"] == layer_idx and r.get("siglip_score")]
        if layer_results:
            layer_siglip[layer_idx] = np.mean([r["siglip_score"] for r in layer_results])

    if layer_siglip:
        sorted_siglip = sorted(layer_siglip.items(), key=lambda x: x[1], reverse=True)
        print("\nTop 10 layers by SigLIP score:")
        for layer_idx, score in sorted_siglip[:10]:
            print(f"  Layer {layer_idx:2d}: {score:.3f}")

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
    )


if __name__ == "__main__":
    main()
