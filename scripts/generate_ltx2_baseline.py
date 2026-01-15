#!/usr/bin/env python3
"""
LTX-2 Baseline Generation Test

Last Updated: 2026-01-15

Tests baseline LTX-2 video generation using diffusers 0.37.0.dev0.
This script validates that:
1. LTX2Pipeline loads correctly from pretrained directory
2. Text encoder (Gemma3) and connectors work properly
3. Video generation completes without OOM on RTX 4090 (24GB)

Usage:
    uv run python scripts/generate_ltx2_baseline.py
    uv run python scripts/generate_ltx2_baseline.py --lora-scale 0.75
    uv run python scripts/generate_ltx2_baseline.py --steps 8 --height 384 --width 512
"""

import argparse
import gc
import time
from pathlib import Path

import torch


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
        default="A cat walking through a sunny garden",
        help="Text prompt for video generation",
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
    args = parser.parse_args()

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
    print("=" * 60)

    # Clear CUDA cache before loading
    gc.collect()
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        print(f"Available VRAM: {free_mem / 1e9:.1f} GB")

    # Import diffusers
    from diffusers import LTX2Pipeline
    from diffusers.utils import export_to_video

    # Load pipeline
    print("\nLoading LTX2Pipeline...")
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

    gen_time = time.time() - start_gen
    print(f"\nGeneration complete in {gen_time:.1f}s ({gen_time / args.steps:.2f}s/step)")

    # Save video
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_frames = output.frames[0]  # First (and only) batch
    print(f"Video shape: {len(video_frames)} frames")

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
