#!/usr/bin/env python3
"""
LTX-2 Activation Steering Experiment

Last Updated: 2026-01-16

Zero-training technique: steer encoder hidden states toward "detailed description"
directions without any model training.

Concept:
    detailed_acts = mean([encoder(p) for p in detailed_prompts])
    vague_acts = mean([encoder(p) for p in vague_prompts])
    detail_direction = detailed_acts - vague_acts

    # At inference
    steered = encoder(user_prompt) + alpha * detail_direction

Usage:
    uv run python experiments/ltx2/activation_steering.py
    uv run python experiments/ltx2/activation_steering.py --alpha 0.5
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.ltx2.prompts import CATEGORY_PROMPTS

# Contrastive prompt pairs for direction extraction
# Semantic pairs: "detailed" vs "vague" descriptions (original)
SEMANTIC_DIRECTION_PAIRS = [
    {
        "vague": "a cat",
        "detailed": "a fluffy orange tabby cat with green eyes, soft fur catching sunlight",
    },
    {
        "vague": "ocean",
        "detailed": "dramatic ocean waves crashing on rocky cliffs at golden hour sunset",
    },
    {
        "vague": "a person",
        "detailed": "a young woman with curly brown hair wearing a red dress, smiling warmly",
    },
]

# Visual pairs: explicit visual properties (brightness-focused)
VISUAL_BRIGHTNESS_PAIRS = [
    {
        "vague": "a dark moody night scene, low light, shadows, dimly lit",
        "detailed": "a bright sunny day, well-lit scene, vibrant lighting, sunlight",
    },
    {
        "vague": "dark rainy evening, overcast sky, gloomy atmosphere",
        "detailed": "bright clear morning, blue sky, cheerful lighting, sunshine",
    },
    {
        "vague": "nighttime cityscape, dark streets, neon lights in darkness",
        "detailed": "daytime cityscape, bright streets, sunlit buildings, clear day",
    },
]

# Visual pairs: sharpness-focused
VISUAL_SHARPNESS_PAIRS = [
    {
        "vague": "blurry soft image, out of focus, low resolution, hazy",
        "detailed": "sharp crisp image, high detail, crystal clear, 4K quality",
    },
    {
        "vague": "motion blur, unfocused, grainy footage, soft edges",
        "detailed": "perfectly focused, sharp edges, high definition, detailed",
    },
    {
        "vague": "dreamy soft focus, gaussian blur, misty, ethereal",
        "detailed": "razor sharp focus, tack sharp, professional clarity, HD",
    },
]

# Default (legacy compatibility)
DIRECTION_PAIRS = SEMANTIC_DIRECTION_PAIRS

# Test prompts for generation (using standardized LTX-2 format prompts)
TEST_PROMPTS = list(CATEGORY_PROMPTS.values())[:3]  # animal, urban, nature prompts


def extract_embeddings_via_generation(pipe, prompt: str, device="cuda"):
    """
    Extract embeddings by doing a minimal generation (1 step) and intercepting.

    This uses the pipeline's internal memory management which works with
    sequential_cpu_offload, unlike direct encode_prompt calls.
    """
    captured_embeds = {}

    # Hook to capture embeddings
    original_call = pipe.__class__.__call__

    def capture_hook(self, *args, **kwargs):
        # Do the encoding step manually with minimal generation
        prompt_embeds, prompt_mask, negative_embeds, negative_mask = self.encode_prompt(
            prompt=kwargs.get("prompt", args[0] if args else None),
            negative_prompt=kwargs.get("negative_prompt", ""),
            do_classifier_free_guidance=True,
            num_videos_per_prompt=1,
            max_sequence_length=128,
            device=device,
            dtype=torch.bfloat16,
        )
        # Store and return early
        captured_embeds["prompt_embeds"] = prompt_embeds.cpu()
        captured_embeds["prompt_mask"] = prompt_mask.cpu()
        # Raise to abort generation
        raise InterruptedError("Captured embeddings")

    try:
        pipe.__class__.__call__ = capture_hook
        pipe(prompt=prompt, num_inference_steps=1)
    except InterruptedError:
        pass
    finally:
        pipe.__class__.__call__ = original_call

    gc.collect()
    torch.cuda.empty_cache()

    return captured_embeds.get("prompt_embeds"), captured_embeds.get("prompt_mask")


def extract_steering_direction(pipe, direction_pairs: list, device="cuda"):
    """
    Extract steering direction from contrastive prompt pairs.

    Uses a memory-efficient approach that works with sequential_cpu_offload:
    - Delete and reload pipeline between encodings
    - Aggressive cache clearing

    Returns:
        direction: Tensor of shape [seq_len, hidden_dim] representing
                   the "detailed description" direction in embedding space.
        magnitude: Scalar indicating the magnitude of the direction.
    """
    detailed_embeds = []
    vague_embeds = []

    for i, pair in enumerate(direction_pairs):
        print(f"  Encoding pair {i+1}/{len(direction_pairs)}...")
        print(f"  Encoding pair {i + 1}/{len(direction_pairs)}...")

        # Encode detailed prompt
        d_embeds, d_mask = pipe.encode_prompt(
            prompt=pair["detailed"],
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=128,
            device=device,
            dtype=torch.bfloat16,
        )
        # Move to CPU immediately to free VRAM
        detailed_embeds.append(d_embeds.cpu())

        # Clear CUDA cache between encodings
        gc.collect()
        torch.cuda.empty_cache()

        # Encode vague prompt
        v_embeds, v_mask = pipe.encode_prompt(
            prompt=pair["vague"],
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=128,
            device=device,
            dtype=torch.bfloat16,
        )
        vague_embeds.append(v_embeds.cpu())

        # Clear CUDA cache between encodings
        gc.collect()
        torch.cuda.empty_cache()

    # Stack and compute means on CPU
    detailed_stack = torch.cat(detailed_embeds, dim=0)  # [N, seq, hidden]
    vague_stack = torch.cat(vague_embeds, dim=0)  # [N, seq, hidden]

    detailed_mean = detailed_stack.mean(dim=0, keepdim=True)  # [1, seq, hidden]
    vague_mean = vague_stack.mean(dim=0, keepdim=True)  # [1, seq, hidden]

    # Compute direction (keep on CPU, will move to device when needed)
    direction = detailed_mean - vague_mean  # [1, seq, hidden]

    # Compute magnitude for diagnostics
    magnitude = torch.norm(direction).item()

    print(f"  Direction magnitude: {magnitude:.4f}")
    print(f"  Detailed mean norm: {torch.norm(detailed_mean):.4f}")
    print(f"  Vague mean norm: {torch.norm(vague_mean):.4f}")

    return direction, magnitude


def extract_embedding_via_generation(pipe, prompt: str, seed: int = 42):
    """
    Extract embedding by running minimal generation and hooking encode_prompt.

    CRITICAL: Direct encode_prompt() OOMs on RTX 4090 with Gemma-3 12B.
    But pipe(prompt=...) works because the pipeline manages memory internally.
    We hook encode_prompt to capture embeddings during the normal flow.

    Returns embeddings BEFORE connector projection [B, T, 188160].
    """
    captured = {"embeds": None, "mask": None}

    # Hook into encode_prompt to capture outputs
    original_encode = pipe.encode_prompt

    def hooked_encode(*args, **kwargs):
        result = original_encode(*args, **kwargs)
        # result is tuple: (prompt_embeds, attention_mask, neg_embeds, neg_mask)
        captured["embeds"] = result[0].cpu()
        captured["mask"] = result[1].cpu()
        return result

    pipe.encode_prompt = hooked_encode

    try:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        output = pipe(
            prompt=prompt,
            height=256,  # Minimal size
            width=256,
            num_frames=9,  # Minimal frames
            num_inference_steps=2,
            guidance_scale=1.0,  # No CFG to keep embeds simple
            generator=generator,
        )
    except Exception as e:
        print(f"    Error during generation: {e}")
    finally:
        # Restore original encode_prompt
        pipe.encode_prompt = original_encode

    gc.collect()
    torch.cuda.empty_cache()

    return captured.get("embeds"), captured.get("mask")


def extract_direction_via_generation(pipe, direction_pairs: list):
    """
    Extract steering direction by capturing embeddings during minimal generations.

    This works around the OOM issue with direct encode_prompt calls.
    """
    detailed_embeds = []
    vague_embeds = []

    for i, pair in enumerate(direction_pairs):
        print(f"  Pair {i + 1}/{len(direction_pairs)}...")
        print(f"    Detailed: {pair['detailed'][:40]}...")

        d_embeds, _ = extract_embedding_via_generation(pipe, pair["detailed"], seed=42 + i)
        if d_embeds is not None:
            detailed_embeds.append(d_embeds)
            print(f"    Got detailed embeds: {d_embeds.shape}")
        else:
            print(f"    Failed to get detailed embeds!")

        print(f"    Vague: {pair['vague'][:40]}...")
        v_embeds, _ = extract_embedding_via_generation(pipe, pair["vague"], seed=42 + i)
        if v_embeds is not None:
            vague_embeds.append(v_embeds)
            print(f"    Got vague embeds: {v_embeds.shape}")
        else:
            print(f"    Failed to get vague embeds!")

    if not detailed_embeds or not vague_embeds:
        print("  ERROR: Failed to extract embeddings!")
        return None, 0.0

    # Compute direction on CPU
    detailed_stack = torch.cat(detailed_embeds, dim=0)
    vague_stack = torch.cat(vague_embeds, dim=0)

    detailed_mean = detailed_stack.mean(dim=0, keepdim=True)
    vague_mean = vague_stack.mean(dim=0, keepdim=True)

    direction = detailed_mean - vague_mean
    magnitude = torch.norm(direction).item()

    print(f"  Direction magnitude: {magnitude:.4f}")

    return direction, magnitude


def extract_direction_with_reload(model_path: str, direction_pairs: list):
    """
    Extract steering direction by reloading the pipeline for each prompt.

    DEPRECATED: This approach still OOMs because encode_prompt itself fails.
    Use extract_direction_via_generation instead.
    """
    from diffusers import LTX2Pipeline

    print("  WARNING: extract_direction_with_reload is deprecated due to OOM issues")
    print("  Using extract_direction_via_generation instead...")

    # Load pipeline once
    pipe = LTX2Pipeline.from_pretrained(model_path, dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()

    direction, magnitude = extract_direction_via_generation(pipe, direction_pairs)

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return direction, magnitude


def generate_with_steering(
    pipe,
    prompt: str,
    steering_direction: torch.Tensor,
    alpha: float,
    seed: int = 42,
    num_frames: int = 33,
    height: int = 512,
    width: int = 768,
    num_inference_steps: int = 25,
    guidance_scale: float = 3.0,
    device="cuda",
):
    """
    Generate video with steered embeddings.

    Uses a hook into encode_prompt to modify embeddings on-the-fly.
    This works around the OOM issue with direct encode_prompt calls.

    Args:
        pipe: LTX2Pipeline
        prompt: Text prompt
        steering_direction: Direction tensor [1, T, 188160] from extract_direction_via_generation
        alpha: Steering strength (0 = no steering, 1 = full steering)
        seed: Random seed

    Returns:
        frames: List of PIL images
    """
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()

    # Hook into encode_prompt to apply steering
    original_encode = pipe.encode_prompt

    def steered_encode(*args, **kwargs):
        # Get original embeddings
        result = original_encode(*args, **kwargs)
        # result is tuple: (prompt_embeds, attention_mask, neg_embeds, neg_mask)
        prompt_embeds = result[0]

        # Apply steering
        if alpha != 0.0:
            # Move direction to same device/dtype
            direction_on_device = steering_direction.to(
                device=prompt_embeds.device, dtype=prompt_embeds.dtype
            )

            # Handle sequence length mismatch (pad/truncate direction to match)
            dir_seq_len = direction_on_device.shape[1]
            emb_seq_len = prompt_embeds.shape[1]

            if dir_seq_len != emb_seq_len:
                if dir_seq_len > emb_seq_len:
                    # Truncate direction
                    direction_on_device = direction_on_device[:, :emb_seq_len, :]
                else:
                    # Pad direction with zeros
                    pad = torch.zeros(
                        1,
                        emb_seq_len - dir_seq_len,
                        direction_on_device.shape[2],
                        device=direction_on_device.device,
                        dtype=direction_on_device.dtype,
                    )
                    direction_on_device = torch.cat([direction_on_device, pad], dim=1)

            # Apply steering
            steered_embeds = prompt_embeds + alpha * direction_on_device

            # Return with steered embeddings
            return (steered_embeds, result[1], result[2], result[3])

        return result

    pipe.encode_prompt = steered_encode

    try:
        # Generate with steered encoding
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

        return output.frames[0]

    finally:
        # Restore original encode_prompt
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


def run_steering_experiment(
    alphas: list = [0.0, 0.1, 0.3, 0.5, 1.0],
    output_dir: str = "experiments/results/ltx2",
    model_path: str = "models/LTX-2",
    save_videos: bool = True,
    direction_pairs: list = None,
):
    """
    Run activation steering experiment with various alpha values.

    Args:
        alphas: List of steering strength values to test
        output_dir: Directory for output files
        model_path: Path to LTX-2 model
        save_videos: Whether to save output videos
        direction_pairs: Custom direction pairs for steering. If None, uses default semantic pairs.
    """
    from diffusers import LTX2Pipeline
    from diffusers.utils import export_to_video

    # Use provided direction pairs or default
    pairs = direction_pairs if direction_pairs is not None else DIRECTION_PAIRS

    print("=" * 60)
    print("LTX-2 Activation Steering Experiment")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if pre-computed direction exists
    direction_file = Path(output_dir) / "steering_direction.pt"

    # Load pipeline
    print("\nLoading pipeline...")
    pipe = LTX2Pipeline.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
    )
    pipe.enable_sequential_cpu_offload()

    if direction_file.exists():
        print(f"\nLoading pre-computed steering direction from {direction_file}...")
        saved = torch.load(direction_file, weights_only=False)
        steering_direction = saved["direction"]
        magnitude = saved["magnitude"]
        print(f"  Direction magnitude: {magnitude:.4f}")
    else:
        # Extract steering direction using generation hook method
        print("\nExtracting steering direction from contrastive pairs...")
        print("  Using generation hook method (runs minimal generations to extract embeddings)")

        steering_direction, magnitude = extract_direction_via_generation(pipe, pairs)

        if steering_direction is None:
            print("  ERROR: Failed to extract steering direction!")
            return {}

        # Save for future runs
        torch.save(
            {
                "direction": steering_direction.cpu()
                if steering_direction.is_cuda
                else steering_direction,
                "magnitude": magnitude,
                "direction_pairs": pairs,
            },
            direction_file,
        )
        print(f"  Saved steering direction to {direction_file}")

    # Results storage
    results = {}

    # Test each alpha value
    for alpha in alphas:
        print(f"\n{'=' * 40}")
        print(f"Alpha: {alpha}")
        print("=" * 40)

        alpha_results = []

        for i, prompt in enumerate(TEST_PROMPTS):
            print(f"\n  [{i + 1}/{len(TEST_PROMPTS)}] {prompt[:40]}...")

            start_time = time.time()

            try:
                frames = generate_with_steering(
                    pipe,
                    prompt=prompt,
                    steering_direction=steering_direction,
                    alpha=alpha,
                    seed=42 + i,  # Same seed per prompt, different across prompts
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
                alpha_results.append(stats)

                print(
                    f"    Time: {gen_time:.1f}s, Mean: {stats['mean']:.1f}, Std: {stats['std']:.1f}"
                )

                # Save video
                if save_videos:
                    video_path = output_path / f"alpha{alpha:.1f}_sample{i}.mp4"
                    export_to_video(frames, str(video_path), fps=24)
                    print(f"    Saved: {video_path.name}")

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback

                traceback.print_exc()
                alpha_results.append({"error": str(e), "prompt": prompt})

            # Memory cleanup
            gc.collect()
            torch.cuda.empty_cache()

        results[f"alpha_{alpha}"] = alpha_results

    # Summary
    print("\n" + "=" * 60)
    print("STEERING SUMMARY")
    print("=" * 60)

    baseline_stats = results.get("alpha_0.0", [])
    if baseline_stats and "mean" in baseline_stats[0]:
        baseline_mean = np.mean([s["mean"] for s in baseline_stats if "mean" in s])
        baseline_std = np.mean([s["std"] for s in baseline_stats if "std" in s])

        print(f"\nBaseline (alpha=0): mean={baseline_mean:.1f}, std={baseline_std:.1f}")

        for alpha_key, alpha_results in results.items():
            if alpha_key == "alpha_0.0":
                continue

            valid_results = [s for s in alpha_results if "mean" in s]
            if valid_results:
                config_mean = np.mean([s["mean"] for s in valid_results])
                config_std = np.mean([s["std"] for s in valid_results])
                mean_diff = (config_mean - baseline_mean) / baseline_mean * 100
                std_diff = (config_std - baseline_std) / baseline_std * 100

                alpha_val = float(alpha_key.replace("alpha_", ""))
                print(f"\n{alpha_key}:")
                print(f"  mean={config_mean:.1f} ({mean_diff:+.1f}% vs baseline)")
                print(f"  std={config_std:.1f} ({std_diff:+.1f}% vs baseline)")

    # Save results
    results_file = output_path / "steering_results.npz"
    np.savez(results_file, **{k: str(v) for k, v in results.items()})
    print(f"\nResults saved to {results_file}")

    # Save steering direction for analysis
    torch.save(
        {
            "direction": steering_direction.cpu(),
            "magnitude": magnitude,
            "direction_pairs": pairs,
        },
        output_path / "steering_direction.pt",
    )
    # Cleanup
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="LTX-2 Activation Steering")
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.3, 0.5, 1.0],
        help="Steering strength values to test",
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
    parser.add_argument(
        "--direction-type",
        type=str,
        choices=["semantic", "brightness", "sharpness"],
        default="semantic",
        help="Type of contrastive pairs: semantic (detailed vs vague), brightness (bright vs dark), sharpness (sharp vs blurry)",
    )
    args = parser.parse_args()

    # Select direction pairs based on type
    direction_pairs_map = {
        "semantic": SEMANTIC_DIRECTION_PAIRS,
        "brightness": VISUAL_BRIGHTNESS_PAIRS,
        "sharpness": VISUAL_SHARPNESS_PAIRS,
    }
    direction_pairs = direction_pairs_map[args.direction_type]
    print(f"Using {args.direction_type} direction pairs")

    run_steering_experiment(
        alphas=args.alpha,
        output_dir=args.output_dir,
        save_videos=not args.no_save_videos,
        direction_pairs=direction_pairs,
    )


if __name__ == "__main__":
    main()
