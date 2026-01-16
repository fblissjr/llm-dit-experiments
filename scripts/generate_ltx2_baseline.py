#!/usr/bin/env python3
"""
LTX-2 Baseline Generation Test

Last Updated: 2026-01-16

Tests baseline LTX-2 video generation using diffusers 0.37.0.dev0.
This script validates that:
1. LTX2Pipeline loads correctly from pretrained directory
2. Text encoder (Gemma3) and connectors work properly
3. Video generation completes without OOM on RTX 4090 (24GB)

Now includes optional latent normalization (ported from ComfyUI-LTXVideo)
to prevent CFG-induced overbaking artifacts.

Usage:
    uv run python scripts/generate_ltx2_baseline.py
    uv run python scripts/generate_ltx2_baseline.py --lora-scale 0.75
    uv run python scripts/generate_ltx2_baseline.py --steps 8 --height 384 --width 512

    # With latent normalization (prevents overbaking)
    uv run python scripts/generate_ltx2_baseline.py --normalize
    uv run python scripts/generate_ltx2_baseline.py --normalize --norm-factors "0.95,0.8,0.6,0.4,0.2,0.0"
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.ltx2.prompts import CATEGORY_PROMPTS


def main():
    parser = argparse.ArgumentParser(description="LTX-2 Baseline Generation Test")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/LTX-2",
        help="Path to LTX-2 model directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=list(CATEGORY_PROMPTS.values())[0],  # animal prompt (143 words)
        help="Text prompt for video generation (default: standardized LTX-2 format prompt)",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt",
    )
    parser.add_argument("--height", type=int, default=512, help="Video height")
    parser.add_argument("--width", type=int, default=768, help="Video width")
    parser.add_argument("--num-frames", type=int, default=33, help="Number of frames")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument("--steps", type=int, default=12, help="Inference steps")
    parser.add_argument("--guidance-scale", type=float, default=3.5, help="CFG scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA weights (e.g., ltx-2-19b-distilled-lora-384.safetensors)",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=0.75,
        help="LoRA scale (0.75-1.0 recommended)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/ltx2_baseline.mp4",
        help="Output video path",
    )
    # Latent normalization arguments (ComfyUI-style)
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Enable latent normalization to prevent CFG overbaking",
    )
    parser.add_argument(
        "--norm-factors",
        type=str,
        default="0.9,0.75,0.5,0.25,0.0",
        help="Comma-separated normalization factors per step (default: 0.9,0.75,0.5,0.25,0.0)",
    )
    parser.add_argument(
        "--norm-target-mean",
        type=float,
        default=0.0,
        help="Target mean for normalization",
    )
    parser.add_argument(
        "--norm-target-std",
        type=float,
        default=1.0,
        help="Target std for normalization",
    )
    parser.add_argument(
        "--norm-percentile",
        type=float,
        default=95.0,
        help="Percentile for robust statistics (0-100)",
    )
    parser.add_argument(
        "--use-wrapper",
        action="store_true",
        help="Use our LTX2Pipeline wrapper instead of raw diffusers",
    )
    # Audio normalization arguments
    parser.add_argument(
        "--audio-normalize",
        action="store_true",
        help="Enable audio latent normalization",
    )
    parser.add_argument(
        "--audio-norm-factors",
        type=str,
        default="1,1,0.25,1,1,0.25",
        help="Per-step audio normalization factors (default: 1,1,0.25,1,1,0.25)",
    )
    # FFN chunking for memory efficiency
    parser.add_argument(
        "--ffn-chunk",
        action="store_true",
        help="Enable FFN chunking for memory efficiency",
    )
    parser.add_argument(
        "--ffn-chunks",
        type=int,
        default=4,
        help="Number of FFN chunks (default: 4)",
    )
    parser.add_argument(
        "--ffn-threshold",
        type=int,
        default=4096,
        help="FFN chunking sequence length threshold (default: 4096)",
    )
    # Preset flags for common combinations
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Enable quality preset: latent normalization",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Enable memory preset: FFN chunking",
    )
    parser.add_argument(
        "--all-enhancements",
        action="store_true",
        help="Enable all enhancement techniques",
    )
    args = parser.parse_args()

    # Apply presets
    if args.quality:
        args.normalize = True
    if args.memory:
        args.ffn_chunk = True
    if args.all_enhancements:
        args.normalize = True
        args.audio_normalize = True
        args.ffn_chunk = True

    print("=" * 60)
    print("LTX-2 Baseline Generation Test")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Prompt: {args.prompt}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Frames: {args.num_frames} @ {args.fps} fps")
    print(f"Steps: {args.steps}, Guidance: {args.guidance_scale}")
    if args.lora_path:
        print(f"LoRA: {args.lora_path} @ scale {args.lora_scale}")

    # Print enabled enhancements
    enhancements = []
    if args.normalize:
        enhancements.append(f"latent_norm (factors={args.norm_factors})")
    if args.audio_normalize:
        enhancements.append(f"audio_norm (factors={args.audio_norm_factors})")
    if args.ffn_chunk:
        enhancements.append(f"ffn_chunk (chunks={args.ffn_chunks})")

    if enhancements:
        print(f"Enhancements: {', '.join(enhancements)}")
    else:
        print("Enhancements: none")

    if args.use_wrapper or enhancements:
        print("Using: llm_dit.pipelines.LTX2Pipeline wrapper")
    else:
        print("Using: diffusers.LTX2Pipeline (raw)")
    print("=" * 60)

    # Clear CUDA cache before loading
    gc.collect()
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        print(f"Available VRAM: {free_mem / 1e9:.1f} GB")

    # Determine whether to use our wrapper (required for enhancements)
    use_wrapper = args.use_wrapper or args.normalize or args.audio_normalize or args.ffn_chunk

    if use_wrapper:
        from llm_dit.pipelines.ltx2 import LTX2Pipeline as WrappedLTX2Pipeline
        from diffusers.utils import export_to_video

        # Load pipeline using our wrapper
        print("\nLoading LTX2Pipeline (llm_dit wrapper)...")
        start_load = time.time()

        pipe = WrappedLTX2Pipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            enable_cpu_offload=True,
        )

        load_time = time.time() - start_load
        print(f"Pipeline loaded in {load_time:.1f}s")

        # Load LoRA if specified
        if args.lora_path:
            lora_path = Path(args.lora_path)
            if not lora_path.is_absolute():
                lora_path = Path(args.model_path) / args.lora_path

            if lora_path.exists():
                print(f"\nLoading LoRA weights from {lora_path}...")
                pipe.load_lora(str(lora_path), scale=args.lora_scale)
                print(f"LoRA loaded with scale {args.lora_scale}")
            else:
                print(f"Warning: LoRA path not found: {lora_path}")

        # Set up generator for reproducibility
        generator = None
        if args.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(args.seed)
            print(f"\nUsing seed: {args.seed}")

        # Generate video
        print("\nGenerating video...")
        start_gen = time.time()

        output = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt if args.negative_prompt else "worst quality, blurry, distorted",
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            # Latent normalization settings
            enable_latent_normalization=args.normalize,
            normalization_factors=args.norm_factors,
            normalization_target_mean=args.norm_target_mean,
            normalization_target_std=args.norm_target_std,
            normalization_percentile=args.norm_percentile,
            # Audio normalization settings
            enable_audio_normalization=args.audio_normalize,
            audio_normalization_factors=args.audio_norm_factors,
            # FFN chunking settings
            enable_ffn_chunking=args.ffn_chunk,
            ffn_chunk_count=args.ffn_chunks,
            ffn_dim_threshold=args.ffn_threshold,
            return_dict=False,
        )
        video_frames, audio = output

    else:
        # Use raw diffusers pipeline
        from diffusers import LTX2Pipeline
        from diffusers.utils import export_to_video

        # Load pipeline
        print("\nLoading LTX2Pipeline (diffusers)...")
        start_load = time.time()

        pipe = LTX2Pipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
        )

        # Enable CPU offload (required for 24GB)
        print("Enabling sequential CPU offload...")
        pipe.enable_sequential_cpu_offload()

        load_time = time.time() - start_load
        print(f"Pipeline loaded in {load_time:.1f}s")

        # Load LoRA if specified
        if args.lora_path:
            lora_path = Path(args.lora_path)
            if not lora_path.is_absolute():
                lora_path = Path(args.model_path) / args.lora_path

            if lora_path.exists():
                print(f"\nLoading LoRA weights from {lora_path}...")
                pipe.load_lora_weights(str(lora_path), adapter_name="distilled")
                pipe.set_adapters(["distilled"], [args.lora_scale])
                print(f"LoRA loaded with scale {args.lora_scale}")
            else:
                print(f"Warning: LoRA path not found: {lora_path}")

        # Set up generator for reproducibility
        generator = None
        if args.seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            print(f"\nUsing seed: {args.seed}")

        # Generate video
        print("\nGenerating video...")
        start_gen = time.time()

        output = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt if args.negative_prompt else None,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )
        video_frames = output.frames[0]  # First (and only) batch

    gen_time = time.time() - start_gen
    print(f"\nGeneration complete in {gen_time:.1f}s ({gen_time / args.steps:.2f}s/step)")

    # Save video
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle different output formats
    if hasattr(video_frames, 'shape'):
        # numpy array from wrapper - may have batch dim
        if video_frames.ndim == 5:
            video_frames = video_frames[0]  # Remove batch dim
        print(f"Video shape: {video_frames.shape[0]} frames")
    else:
        # List of PIL images from diffusers
        print(f"Video shape: {len(video_frames)} frames")

    from diffusers.utils import export_to_video
    export_to_video(video_frames, str(output_path), fps=args.fps)
    print(f"\nSaved to: {output_path}")

    # Cleanup
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
