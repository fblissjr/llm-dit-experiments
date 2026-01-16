#!/usr/bin/env python3
"""
LTX-2 Chunk Boundary Analysis Experiment

Last Updated: 2026-01-16

Test the hypothesis that LTX-2's 8-frame temporal compression creates
"chunk boundaries" where state transitions happen more cleanly than mid-chunk.

Hypothesis:
- VAE compresses 8 pixel frames → 1 latent frame
- Transitions at frame 8→9 (boundary) may be sharper than frame 4→5 (mid-chunk)
- Position embeddings might behave differently near boundaries

Test Strategy:
1. Generate videos with continuous motion (metronome, walking, rolling ball)
2. Compare frame-to-frame consistency at boundaries vs mid-chunk
3. Visual inspection for motion hitches at chunk boundaries
4. Quantitative: LPIPS perceptual distance, optical flow discontinuity

Frame Count Constraints:
- Must satisfy: (num_frames - 1) % 8 == 0
- Valid counts: 9, 17, 25, 33, 41, 49...
- Latent frames: 1 + (pixel_frames - 1) / 8

Memory-optimized for 24GB GPUs (RTX 4090):
- Text encoder: 8-bit quantized (~13GB)
- Pipeline: Group offloading for transformer blocks
- Sequential loading: Encode → Offload → Generate

Output Structure (viewer-compatible):
    experiments/results/ltx2_chunk_boundary_{timestamp}/
    ├── images/           # First frames as PNG
    ├── videos/           # Full MP4s
    ├── frames/           # All frames extracted for analysis
    ├── metadata/         # Per-sample JSON with metrics
    └── ltx2_chunk_boundary_summary.json

Usage:
    # Quick test (2 frame counts x 2 prompts)
    uv run python experiments/ltx2/chunk_boundary_analysis.py --quick

    # Full sweep (4 frame counts x 5 prompts)
    uv run python experiments/ltx2/chunk_boundary_analysis.py

    # Specific frame counts
    uv run python experiments/ltx2/chunk_boundary_analysis.py --frame-counts 17 25
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

# Import memory-efficient utilities for 24GB GPUs
from experiments.ltx2.memory_utils import (
    load_text_encoder_8bit,
    pack_text_embeds,
    load_pipeline_with_offloading,
    cleanup_memory,
    get_gpu_memory,
)


# =============================================================================
# CONTINUOUS MOTION PROMPTS - Designed to reveal temporal discontinuities
# =============================================================================
# These follow LTX-2 format (100+ words, scene headings, dialogue, camera)

CONTINUOUS_MOTION_PROMPTS = {
    "metronome": (
        "INT. MUSIC STUDIO – AFTERNOON. A classic wooden metronome sits on a grand piano, "
        "its golden arm swinging steadily from left to right. The camera holds a static "
        "medium shot, capturing the rhythmic pendulum motion against the dark wood of "
        "the piano's surface.\n"
        "The metronome clicks softly with each swing – tick, tick, tick – marking perfect "
        "time. A pianist's hand enters frame from the bottom, fingers hovering over the "
        "keys. The pianist, an older gentleman, speaks softly: \"Sixty beats per minute. "
        "The heartbeat of music.\"\n"
        "The arm swings left. Click. Right. Click. Left. Click. The reflection of "
        "afternoon sunlight glints off the brass weight as it passes center. Dust motes "
        "float lazily in the shaft of light behind the metronome, undisturbed by the "
        "precise mechanical motion.\n"
        "The camera slowly pushes in as the swinging continues, filling the frame with "
        "the hypnotic back-and-forth rhythm. documentary style, warm lighting"
    ),
    "ball_rolling": (
        "INT. PHYSICS LAB – DAY. A white billiard ball rests at the left edge of a "
        "polished black table marked with evenly spaced vertical white lines creating "
        "a measurement grid. The camera holds a static overhead shot looking straight "
        "down at the smooth surface.\n"
        "A researcher's hand enters frame, wearing a white lab coat sleeve. She gives "
        "the ball a gentle push. \"Uniform velocity test, trial seven,\" she announces "
        "calmly. The ball begins rolling smoothly from left to right across the grid "
        "lines, each line passing beneath it at regular intervals.\n"
        "The ball's shadow follows beneath it as it travels – one line, two lines, three "
        "lines – maintaining steady speed across the dark surface. The grid provides "
        "perfect reference marks for tracking its position. A wall clock ticks softly "
        "in the background.\n"
        "The camera remains perfectly still as the ball crosses the frame, white against "
        "black, motion against stillness. \"Mark the velocity,\" the researcher notes "
        "off-screen. scientific documentary style"
    ),
    "pendulum": (
        "INT. CLOCKMAKER'S WORKSHOP – EVENING. A brass pendulum hangs from an ornate "
        "grandfather clock, swinging in a slow, graceful arc. Golden lamplight catches "
        "the polished metal as it passes through center, creating soft reflections on "
        "the wooden clock case behind it.\n"
        "The camera frames a tight shot on the pendulum itself, tracking its hypnotic "
        "left-to-right motion. An elderly clockmaker watches from the background, his "
        "weathered face partially visible. \"Sixty years I've been listening to this "
        "swing,\" he murmurs. \"Never missed a beat.\"\n"
        "The pendulum reaches the apex of its leftward swing, pauses for a heartbeat, "
        "then falls back through center with silent grace. Right it goes, slowing as "
        "gravity pulls it back. The arc is perfect, mechanical, eternal.\n"
        "Dust catches the lamplight as the pendulum displaces air with each pass. The "
        "clockmaker reaches for his tools. \"Some things never need fixing.\" The swinging "
        "continues, left to right to left, marking time itself. period drama lighting"
    ),
    "walking_person": (
        "EXT. BEACH BOARDWALK – SUNSET. A young woman in a flowing white sundress walks "
        "along a wooden boardwalk, her bare feet padding softly against the weathered "
        "planks. The camera tracks alongside her in a smooth lateral dolly shot, keeping "
        "pace with her steady stride.\n"
        "The golden hour sun casts long shadows from the boardwalk railings, creating "
        "parallel lines that she crosses with each step. Her dress ripples gently in "
        "the ocean breeze. She glances toward the camera and smiles. \"This is my "
        "favorite time of day,\" she says softly.\n"
        "One step, two steps, three steps – her gait is unhurried and rhythmic. Behind "
        "her, the waves roll in with distant white foam. Seagulls call overhead. Her "
        "sandals dangle from her right hand, swinging slightly with her motion.\n"
        "The camera maintains its parallel tracking as she continues walking, the "
        "boardwalk planks providing natural markers of her progress. Each footfall "
        "lands with quiet precision. cinematic warm tones, shallow depth of field"
    ),
    "clock_control": (
        "INT. ANTIQUE SHOP – AFTERNOON. An ornate grandfather clock stands against a "
        "velvet-draped wall, its brass face gleaming in the soft lamplight. The camera "
        "holds a medium shot on the clock face, capturing the slow rotation of its "
        "black hands against the ivory dial.\n"
        "The second hand sweeps smoothly around the numbered face – past the twelve, "
        "the three, the six – marking time with mechanical precision. An antique dealer "
        "watches from nearby, adjusting his spectacles. \"This piece is from 1847,\" he "
        "explains. \"Still keeps perfect time.\"\n"
        "The minute hand inches imperceptibly toward the next marker. The second hand "
        "continues its eternal circle, passing each Roman numeral in turn. The shop "
        "is quiet except for the soft tick-tick-tick of the mechanism within.\n"
        "Dust motes float in the shaft of afternoon sunlight beside the clock. The dealer "
        "leans closer to examine the face. \"Notice how the hands move – perfectly "
        "calibrated.\" The camera slowly pushes in on the sweeping second hand. "
        "period drama cinematography, warm amber tones"
    ),
}

# Quick mode prompts
QUICK_MOTION_PROMPTS = ["metronome", "walking_person"]

# Frame count configurations
# Each count = (pixel_frames - 1) / 8 + 1 latent frames
# Boundaries occur between latent frames
FRAME_COUNT_CONFIGS = {
    9: {"latent_frames": 2, "boundaries": 1, "description": "1 boundary at frame 8→9"},
    17: {"latent_frames": 3, "boundaries": 2, "description": "2 boundaries at frames 8→9, 16→17"},
    25: {"latent_frames": 4, "boundaries": 3, "description": "3 boundaries at frames 8→9, 16→17, 24→25"},
    33: {"latent_frames": 5, "boundaries": 4, "description": "4 boundaries (standard generation)"},
}

QUICK_FRAME_COUNTS = [17, 25]  # 2-3 boundaries
FULL_FRAME_COUNTS = [9, 17, 25, 33]  # 1-4 boundaries


def get_boundary_frames(num_frames: int) -> list[int]:
    """Get the frame indices where chunk boundaries occur.

    Boundaries occur at frame 8, 16, 24, 32... (every 8th frame after the first).
    The transition is from frame N to frame N+1.
    """
    boundaries = []
    for i in range(8, num_frames, 8):
        boundaries.append(i)
    return boundaries


def extract_all_frames(frames: list, output_dir: Path, prefix: str) -> list[Path]:
    """Extract and save all frames from a video for analysis."""
    frame_paths = []
    for i, frame in enumerate(frames):
        frame_path = output_dir / f"{prefix}_frame_{i:03d}.png"
        frame.save(frame_path)
        frame_paths.append(frame_path)
    return frame_paths


def compute_frame_differences(frames: list) -> dict:
    """Compute frame-to-frame differences for boundary analysis.

    Returns dict with:
    - all_diffs: list of mean absolute differences between consecutive frames
    - boundary_diffs: differences at chunk boundaries (frames 7→8, 15→16, etc.)
    - mid_chunk_diffs: differences at mid-chunk positions
    """
    import numpy as np

    all_diffs = []
    for i in range(len(frames) - 1):
        arr1 = np.array(frames[i]).astype(float)
        arr2 = np.array(frames[i + 1]).astype(float)
        diff = np.mean(np.abs(arr1 - arr2))
        all_diffs.append(float(diff))

    # Identify boundary and mid-chunk positions
    # Boundary transitions: 7→8, 15→16, 23→24 (indices 7, 15, 23 in diff array)
    boundary_indices = [i for i in range(7, len(all_diffs), 8)]
    # Mid-chunk: 3→4, 11→12, 19→20 (indices 3, 11, 19)
    mid_chunk_indices = [i for i in range(3, len(all_diffs), 8)]

    boundary_diffs = [all_diffs[i] for i in boundary_indices if i < len(all_diffs)]
    mid_chunk_diffs = [all_diffs[i] for i in mid_chunk_indices if i < len(all_diffs)]

    return {
        "all_diffs": all_diffs,
        "boundary_diffs": boundary_diffs,
        "mid_chunk_diffs": mid_chunk_diffs,
        "boundary_indices": boundary_indices,
        "mid_chunk_indices": mid_chunk_indices,
        "mean_boundary_diff": float(np.mean(boundary_diffs)) if boundary_diffs else None,
        "mean_mid_chunk_diff": float(np.mean(mid_chunk_diffs)) if mid_chunk_diffs else None,
    }


def save_metadata(
    output_dir: Path,
    num_frames: int,
    prompt_id: str,
    prompt_text: str,
    seed: int,
    image_path: Path,
    video_path: Path | None,
    generation_time: float,
    frame_diffs: dict,
    boundary_frames: list[int],
    siglip_score: float | None = None,
):
    """Save viewer-compatible metadata JSON."""
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    config = FRAME_COUNT_CONFIGS.get(num_frames, {})
    filename = f"frames_{num_frames:02d}_{prompt_id}.json"
    filepath = metadata_dir / filename

    data = {
        "config": {
            "prompt_id": prompt_id,
            "variable_name": "num_frames",
            "variable_value": num_frames,
            "seed": seed,
            "prompt_text": prompt_text,
            "latent_frames": config.get("latent_frames"),
            "num_boundaries": config.get("boundaries"),
            "boundary_frame_indices": boundary_frames,
        },
        "analysis": {
            "frame_diffs": frame_diffs,
            "hypothesis_test": {
                "description": "Compare frame differences at boundaries vs mid-chunk",
                "boundary_diff_mean": frame_diffs.get("mean_boundary_diff"),
                "mid_chunk_diff_mean": frame_diffs.get("mean_mid_chunk_diff"),
            },
        },
        "siglip_score": siglip_score,
        "generation_time_seconds": generation_time,
        "output_path": str(image_path.relative_to(output_dir.parent.parent.parent)),
        "video_path": str(video_path.relative_to(output_dir.parent.parent.parent)) if video_path else None,
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return filepath


def run_chunk_boundary_analysis(
    output_base: str = "experiments/results",
    model_path: str = "models/LTX-2",
    frame_counts: list[int] | None = None,
    prompts: dict[str, str] | None = None,
    save_videos: bool = True,
    save_all_frames: bool = True,
    compute_metrics: bool = True,
    seed: int = 42,
    height: int = 512,
    width: int = 512,  # Square for cleaner motion analysis
    num_inference_steps: int = 25,
    guidance_scale: float = 3.0,
    num_blocks_per_group: int = 1,
):
    """
    Run the chunk boundary analysis experiment.

    Memory-optimized for 24GB GPUs using:
    - 8-bit quantized text encoder (~13GB)
    - Group offloading for transformer blocks
    - Sequential loading: encode → offload → generate

    Args:
        output_base: Base directory for results
        model_path: Path to LTX-2 model
        frame_counts: List of frame counts to test. None = FULL_FRAME_COUNTS
        prompts: Dict of prompt_id -> prompt_text. None = CONTINUOUS_MOTION_PROMPTS
        save_videos: Whether to save full MP4 videos
        save_all_frames: Whether to extract and save all frames
        compute_metrics: Whether to compute SigLIP2 scores
        seed: Random seed for reproducibility
        height: Video height
        width: Video width
        num_inference_steps: Diffusion steps
        guidance_scale: CFG scale
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
    if frame_counts is None:
        frame_counts = FULL_FRAME_COUNTS

    if prompts is None:
        prompts = CONTINUOUS_MOTION_PROMPTS

    # Validate frame counts
    for fc in frame_counts:
        if (fc - 1) % 8 != 0:
            raise ValueError(f"Invalid frame count {fc}. Must satisfy (n-1) % 8 == 0")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base) / f"ltx2_chunk_boundary_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    if save_videos:
        videos_dir = output_dir / "videos"
        videos_dir.mkdir(exist_ok=True)

    if save_all_frames:
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("LTX-2 Chunk Boundary Analysis (Memory-Optimized)")
    logger.info("=" * 60)
    logger.info(f"Output: {output_dir}")
    logger.info(f"Frame counts: {frame_counts}")
    for fc in frame_counts:
        config = FRAME_COUNT_CONFIGS.get(fc, {})
        logger.info(f"  {fc} frames: {config.get('description', 'N/A')}")
    logger.info(f"Prompts: {len(prompts)}")
    total = len(frame_counts) * len(prompts)
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

    # Pre-encode all prompts (same embedding works for all frame counts)
    # Structure: embeddings_cache[prompt_id] = (prompt_embeds, prompt_text)
    embeddings_cache = {}

    # First encode the negative prompt (empty string for CFG)
    logger.info("  Encoding: negative prompt (empty)")
    neg_inputs = tokenizer(
        "",
        return_tensors="pt",
        padding="max_length",
        max_length=256,
        truncation=True,
    )
    neg_input_ids = neg_inputs["input_ids"].to(text_encoder.device)
    neg_attention_mask = neg_inputs["attention_mask"].to(text_encoder.device)

    with torch.no_grad():
        neg_outputs = text_encoder(
            input_ids=neg_input_ids,
            attention_mask=neg_attention_mask,
            output_hidden_states=True,
        )

    # Stack ALL 49 hidden states (embedding + 48 transformer layers)
    # The projection matrix expects 49 layers (188160 = 49 × 3840)
    neg_hidden_states = torch.stack(neg_outputs.hidden_states[:49], dim=-1)
    neg_seq_len = neg_attention_mask.sum().item()
    negative_prompt_embeds = pack_text_embeds(
        neg_hidden_states,
        neg_seq_len,
        device=torch.device("cuda"),
    ).cpu()
    negative_attention_mask = neg_attention_mask.cpu()

    for prompt_id, prompt_text in prompts.items():
        logger.info(f"  Encoding: {prompt_id}")

        # Standard encoding (no layer masking for this experiment)
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            padding="max_length",
            max_length=256,
            truncation=True,
        )

        input_ids = inputs["input_ids"].to(text_encoder.device)
        attention_mask = inputs["attention_mask"].to(text_encoder.device)

        with torch.no_grad():
            outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Stack ALL 49 hidden states (embedding + 48 transformer layers)
        # The projection matrix expects 49 layers (188160 = 49 × 3840)
        hidden_states = torch.stack(outputs.hidden_states[:49], dim=-1)  # [B, T, D, 49]
        seq_len = attention_mask.sum().item()

        # Pack to create prompt_embeds
        prompt_embeds = pack_text_embeds(
            hidden_states,
            seq_len,
            device=torch.device("cuda"),
        )

        # Cache on CPU (include negative embeds for CFG)
        embeddings_cache[prompt_id] = {
            "prompt_embeds": prompt_embeds.cpu(),
            "prompt_attention_mask": attention_mask.cpu(),
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_attention_mask": negative_attention_mask,
            "prompt": prompt_text,
        }

    logger.info(f"\nEncoded {len(embeddings_cache)} prompts (+ negative prompt for CFG)")

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

    # Load SigLIP scorer
    siglip_scorer = None
    if compute_metrics:
        try:
            from experiments.metrics.siglip_score import SigLIPScorer
            siglip_scorer = SigLIPScorer()
            logger.info("SigLIP2 scorer loaded")
        except ImportError as e:
            logger.warning(f"Could not load SigLIP scorer: {e}")

    # Results accumulator
    all_results = []

    # Main generation loop
    count = 0

    for num_frames in frame_counts:
        config = FRAME_COUNT_CONFIGS.get(num_frames, {})
        boundary_frames = get_boundary_frames(num_frames)

        logger.info(f"\n{'='*50}")
        logger.info(f"{num_frames} frames ({config.get('description', 'N/A')})")
        logger.info(f"Boundary frames: {boundary_frames}")
        logger.info("=" * 50)

        for prompt_id, prompt_text in prompts.items():
            count += 1
            logger.info(f"\n  [{count}/{total}] {num_frames} frames × {prompt_id}")

            # Get cached embeddings
            cached = embeddings_cache[prompt_id]
            prompt_embeds = cached["prompt_embeds"].to("cuda")
            prompt_attention_mask = cached["prompt_attention_mask"].to("cuda")
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
                image_filename = f"frames_{num_frames:02d}_{prompt_id}.png"
                image_path = images_dir / image_filename
                first_frame.save(image_path)

                # Save video if requested
                video_path = None
                if save_videos:
                    video_filename = f"frames_{num_frames:02d}_{prompt_id}.mp4"
                    video_path = videos_dir / video_filename
                    export_to_video(frames, str(video_path), fps=24)

                # Extract all frames for analysis
                if save_all_frames:
                    frame_prefix = f"frames_{num_frames:02d}_{prompt_id}"
                    extract_all_frames(frames, frames_dir, frame_prefix)

                # Compute frame-to-frame differences for boundary analysis
                frame_diffs = compute_frame_differences(frames)

                # Compute SigLIP score
                siglip_score = None
                if siglip_scorer:
                    try:
                        siglip_score = siglip_scorer.score(prompt_text, first_frame)
                    except Exception as e:
                        logger.warning(f"SigLIP scoring failed: {e}")

                # Save metadata
                meta_path = save_metadata(
                    output_dir=output_dir,
                    num_frames=num_frames,
                    prompt_id=prompt_id,
                    prompt_text=prompt_text,
                    seed=seed,
                    image_path=image_path,
                    video_path=video_path,
                    generation_time=gen_time,
                    frame_diffs=frame_diffs,
                    boundary_frames=boundary_frames,
                    siglip_score=siglip_score,
                )

                result = {
                    "num_frames": num_frames,
                    "prompt_id": prompt_id,
                    "generation_time": gen_time,
                    "siglip_score": siglip_score,
                    "boundary_diff_mean": frame_diffs.get("mean_boundary_diff"),
                    "mid_chunk_diff_mean": frame_diffs.get("mean_mid_chunk_diff"),
                    "num_boundaries": len(boundary_frames),
                }
                all_results.append(result)

                # Log progress with boundary analysis
                bd_mean = frame_diffs.get("mean_boundary_diff")
                mc_mean = frame_diffs.get("mean_mid_chunk_diff")
                logger.info(f"  Time: {gen_time:.1f}s | GPU: {get_gpu_memory():.1f}GB")
                if bd_mean is not None and mc_mean is not None:
                    ratio = bd_mean / mc_mean if mc_mean > 0 else 0
                    logger.info(f"  Boundary diff: {bd_mean:.2f} | Mid-chunk diff: {mc_mean:.2f} | Ratio: {ratio:.2f}")

            except Exception as e:
                logger.error(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "num_frames": num_frames,
                    "prompt_id": prompt_id,
                    "error": str(e),
                })

            # Memory cleanup
            del prompt_embeds, prompt_attention_mask
            del negative_prompt_embeds, negative_prompt_attention_mask
            cleanup_memory()

    # Save summary
    summary = {
        "experiment_type": "ltx2_chunk_boundary_analysis",
        "timestamp": timestamp,
        "hypothesis": (
            "State transitions at 8-frame chunk boundaries (frames 8, 16, 24...) "
            "are sharper/cleaner than mid-chunk transitions due to VAE temporal compression."
        ),
        "config": {
            "frame_counts": frame_counts,
            "prompts": list(prompts.keys()),
            "seed": seed,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "model_path": model_path,
        },
        "results": all_results,
        "analysis": compute_boundary_statistics(all_results),
    }

    summary_path = output_dir / "ltx2_chunk_boundary_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Total generations: {count}")

    # Print analysis
    print_boundary_analysis(all_results)

    # Cleanup
    del pipe
    if siglip_scorer:
        del siglip_scorer
    cleanup_memory()

    return output_dir


def compute_boundary_statistics(results: list[dict]) -> dict:
    """Compute aggregate statistics for boundary analysis."""
    valid = [r for r in results if "error" not in r]

    if not valid:
        return {"error": "No valid results"}

    # Aggregate boundary vs mid-chunk differences
    all_boundary_diffs = [r["boundary_diff_mean"] for r in valid if r.get("boundary_diff_mean") is not None]
    all_mid_chunk_diffs = [r["mid_chunk_diff_mean"] for r in valid if r.get("mid_chunk_diff_mean") is not None]

    boundary_mean = float(np.mean(all_boundary_diffs)) if all_boundary_diffs else None
    mid_chunk_mean = float(np.mean(all_mid_chunk_diffs)) if all_mid_chunk_diffs else None

    ratio = None
    if boundary_mean is not None and mid_chunk_mean is not None and mid_chunk_mean > 0:
        ratio = boundary_mean / mid_chunk_mean

    return {
        "overall": {
            "mean_boundary_diff": boundary_mean,
            "mean_mid_chunk_diff": mid_chunk_mean,
            "boundary_to_midchunk_ratio": ratio,
            "hypothesis_supported": ratio > 1.1 if ratio else None,  # >10% higher = supported
        },
        "per_frame_count": {
            fc: {
                "mean_boundary": float(np.mean([r["boundary_diff_mean"] for r in valid
                    if r["num_frames"] == fc and r.get("boundary_diff_mean")]))
                    if any(r["num_frames"] == fc and r.get("boundary_diff_mean") for r in valid) else None,
                "mean_midchunk": float(np.mean([r["mid_chunk_diff_mean"] for r in valid
                    if r["num_frames"] == fc and r.get("mid_chunk_diff_mean")]))
                    if any(r["num_frames"] == fc and r.get("mid_chunk_diff_mean") for r in valid) else None,
            }
            for fc in sorted(set(r["num_frames"] for r in valid))
        },
        "per_prompt": {
            pid: {
                "mean_boundary": float(np.mean([r["boundary_diff_mean"] for r in valid
                    if r["prompt_id"] == pid and r.get("boundary_diff_mean")]))
                    if any(r["prompt_id"] == pid and r.get("boundary_diff_mean") for r in valid) else None,
                "mean_midchunk": float(np.mean([r["mid_chunk_diff_mean"] for r in valid
                    if r["prompt_id"] == pid and r.get("mid_chunk_diff_mean")]))
                    if any(r["prompt_id"] == pid and r.get("mid_chunk_diff_mean") for r in valid) else None,
            }
            for pid in sorted(set(r["prompt_id"] for r in valid))
        },
        "total_valid": len(valid),
        "total_errors": len(results) - len(valid),
    }


def print_boundary_analysis(results: list[dict]):
    """Print boundary analysis summary."""
    valid = [r for r in results if "error" not in r]

    if not valid:
        return

    print("\n" + "=" * 60)
    print("CHUNK BOUNDARY HYPOTHESIS TEST")
    print("=" * 60)
    print("\nHypothesis: Transitions at 8-frame boundaries show larger")
    print("frame-to-frame differences than mid-chunk transitions.")
    print("\n" + "-" * 60)

    # Per-prompt analysis
    print("\nPer-Prompt Results:")
    print(f"  {'Prompt':<20} {'Boundary':<12} {'Mid-Chunk':<12} {'Ratio':<10}")
    print("  " + "-" * 54)

    for prompt_id in sorted(set(r["prompt_id"] for r in valid)):
        prompt_results = [r for r in valid if r["prompt_id"] == prompt_id]
        bd_vals = [r["boundary_diff_mean"] for r in prompt_results if r.get("boundary_diff_mean")]
        mc_vals = [r["mid_chunk_diff_mean"] for r in prompt_results if r.get("mid_chunk_diff_mean")]

        if bd_vals and mc_vals:
            bd_mean = np.mean(bd_vals)
            mc_mean = np.mean(mc_vals)
            ratio = bd_mean / mc_mean if mc_mean > 0 else 0
            indicator = "✓" if ratio > 1.1 else "✗"
            print(f"  {prompt_id:<20} {bd_mean:<12.2f} {mc_mean:<12.2f} {ratio:<8.2f} {indicator}")

    # Overall
    all_bd = [r["boundary_diff_mean"] for r in valid if r.get("boundary_diff_mean")]
    all_mc = [r["mid_chunk_diff_mean"] for r in valid if r.get("mid_chunk_diff_mean")]

    if all_bd and all_mc:
        overall_bd = np.mean(all_bd)
        overall_mc = np.mean(all_mc)
        overall_ratio = overall_bd / overall_mc if overall_mc > 0 else 0

        print("\n" + "-" * 60)
        print(f"\nOVERALL:")
        print(f"  Mean boundary diff:   {overall_bd:.2f}")
        print(f"  Mean mid-chunk diff:  {overall_mc:.2f}")
        print(f"  Ratio (boundary/mid): {overall_ratio:.2f}")

        if overall_ratio > 1.1:
            print(f"\n  ✓ HYPOTHESIS SUPPORTED: Boundary diffs are {(overall_ratio-1)*100:.1f}% higher")
        elif overall_ratio > 1.0:
            print(f"\n  ~ INCONCLUSIVE: Boundary diffs are only {(overall_ratio-1)*100:.1f}% higher")
        else:
            print(f"\n  ✗ HYPOTHESIS NOT SUPPORTED: Mid-chunk diffs are higher")

    print("\n" + "=" * 60)
    print("Visual inspection of videos is recommended to confirm findings.")
    print("Look for motion 'hitches' at frame boundaries (8, 16, 24...).")
    print("\nView results:")
    print("  uv run experiments/viewer/server.py")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="LTX-2 Chunk Boundary Analysis Experiment",
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
        "--frame-counts",
        type=int,
        nargs="+",
        help="Frame counts to test (must satisfy (n-1) %% 8 == 0)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 2 frame counts x 2 prompts",
    )
    parser.add_argument(
        "--no-save-videos",
        action="store_true",
        help="Don't save full MP4 videos",
    )
    parser.add_argument(
        "--no-save-frames",
        action="store_true",
        help="Don't extract individual frames",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Skip SigLIP2 computation",
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
        default=512,
        help="Video width (default: 512, square for motion analysis)",
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
        "--blocks-per-group",
        type=int,
        default=1,
        help="Transformer blocks per offload group (1=min VRAM, higher=faster)",
    )

    args = parser.parse_args()

    # Determine frame counts and prompts
    if args.quick:
        frame_counts = QUICK_FRAME_COUNTS
        prompts = {k: CONTINUOUS_MOTION_PROMPTS[k] for k in QUICK_MOTION_PROMPTS}
    else:
        frame_counts = args.frame_counts or FULL_FRAME_COUNTS
        prompts = None  # Use all CONTINUOUS_MOTION_PROMPTS

    # Validate frame counts
    for fc in frame_counts:
        if (fc - 1) % 8 != 0:
            parser.error(f"Invalid frame count {fc}. Valid counts: 9, 17, 25, 33, 41...")

    run_chunk_boundary_analysis(
        output_base=args.output_base,
        model_path=args.model_path,
        frame_counts=frame_counts,
        prompts=prompts,
        save_videos=not args.no_save_videos,
        save_all_frames=not args.no_save_frames,
        compute_metrics=not args.skip_metrics,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        num_blocks_per_group=args.blocks_per_group,
    )


if __name__ == "__main__":
    main()
