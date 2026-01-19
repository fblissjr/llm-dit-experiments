#!/usr/bin/env python3
"""
Generate videos with enhanced prompts using LTX-2's T2V system prompt style.

Last Updated: 2026-01-19

This script demonstrates prompt enhancement for video generation.
The enhanced prompts follow LTX-2's official prompt engineering guidelines.

Usage:
    python scripts/generate_with_enhanced_prompts.py "A cat walking"
    python scripts/generate_with_enhanced_prompts.py --list  # Use preset prompts
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Pre-enhanced prompts (simulating what Gemma would produce with the T2V system prompt)
# These follow the guidelines: Style prefix, present progressive verbs, integrated audio,
# temporal flow, restrained language, no camera motion unless requested
ENHANCED_PROMPTS = {
    "A cat walking": (
        "Style: realistic with soft natural lighting. A fluffy orange tabby cat with green eyes "
        "walks gracefully across a wooden floor, its paws making soft padding sounds with each step. "
        "The cat's tail sways gently side to side as it moves, whiskers twitching slightly. "
        "Ambient room sounds include a distant clock ticking and the quiet hum of an air conditioner. "
        "The cat pauses briefly, ears perking forward, then continues walking with deliberate, "
        "measured steps, claws making faint clicks on the hardwood."
    ),

    "A woman drinking coffee": (
        "Style: cinematic with warm interior lighting. A woman in her late 20s with dark wavy hair "
        "sits at a kitchen counter, both hands wrapped around a white ceramic mug. Steam rises gently "
        "from the coffee, catching the morning light from a nearby window. She lifts the mug slowly "
        "to her lips, taking a careful sip, eyes closing briefly in appreciation. The soft clink of "
        "ceramic against the counter sounds as she sets it down. A refrigerator hums quietly in the "
        "background. She wraps her fingers tighter around the warm mug, a slight smile forming on her face."
    ),

    "Ocean waves at sunset": (
        "Style: cinematic nature documentary. Golden sunlight bathes the ocean surface as gentle waves "
        "roll toward a sandy shore, creating soft white foam patterns. The rhythmic sound of waves "
        "crashing mingles with the distant cry of seagulls. Water recedes with a hissing sound over "
        "wet sand, revealing small shells and pebbles. Another wave builds in the distance, its crest "
        "catching the warm orange light before curling and breaking with a satisfying splash. "
        "The constant ambient roar of the ocean provides a soothing backdrop."
    ),

    "A chef cooking": (
        "Style: realistic kitchen documentary. A chef in a white coat stands at a stainless steel "
        "stove, tossing vegetables in a large pan with practiced wrist movements. The sizzle of "
        "onions and peppers fills the air as they hit the hot surface. Steam rises from the pan "
        "while the chef reaches for a wooden spoon, stirring the contents with smooth circular "
        "motions. The clatter of utensils and the gentle bubbling of a nearby pot create a busy "
        "kitchen atmosphere. The chef sprinkles salt from a small bowl, the grains catching light "
        "as they fall into the pan."
    ),

    "Rain on a window": (
        "Style: intimate close-up, moody lighting. Raindrops strike a glass window pane, each drop "
        "creating a small splash before trickling downward in winding paths. The steady patter of "
        "rain against glass provides a constant ambient rhythm. Water droplets merge and separate "
        "as they travel down the surface, distorting the blurred lights visible outside. "
        "Occasional heavier drops land with a louder tap, while thunder rumbles softly in the "
        "distance. The glass fogs slightly from the temperature difference between inside and out."
    ),
}


def get_enhanced_prompt(original: str) -> str:
    """
    Get an enhanced prompt for video generation.

    In the official LTX-2 pipeline, this would call Gemma3's enhance_t2v() method
    which uses the T2V system prompt to expand the user's input.

    For this demo, we use pre-written enhanced prompts that follow the guidelines.
    """
    # Check if we have a pre-enhanced version
    if original in ENHANCED_PROMPTS:
        return ENHANCED_PROMPTS[original]

    # For unknown prompts, return a basic enhancement
    # In production, this would call Gemma3 with the T2V system prompt
    return f"Style: cinematic realistic. {original}"


def main():
    parser = argparse.ArgumentParser(description="Generate videos with enhanced prompts")
    parser.add_argument("prompt", nargs="?", help="User prompt to enhance and generate")
    parser.add_argument("--list", action="store_true", help="List available preset prompts")
    parser.add_argument("--original-only", action="store_true", help="Only generate with original prompt")
    parser.add_argument("--enhanced-only", action="store_true", help="Only generate with enhanced prompt")
    parser.add_argument("--frames", type=int, default=33, help="Number of frames (default: 33, ~1.3sec)")
    parser.add_argument("--height", type=int, default=256, help="Video height (default: 256)")
    parser.add_argument("--width", type=int, default=384, help="Video width (default: 384)")
    parser.add_argument("--steps", type=int, default=15, help="Inference steps (default: 15)")
    parser.add_argument("--cfg", type=float, default=1.0,
                       help="CFG guidance scale (default: 1.0=disabled, >1.0 needs >24GB VRAM)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default="outputs/enhanced_prompts",
                       help="Output directory")

    args = parser.parse_args()

    if args.list:
        print("Available preset prompts:")
        print("-" * 50)
        for i, (original, enhanced) in enumerate(ENHANCED_PROMPTS.items(), 1):
            print(f"\n{i}. Original: {original}")
            print(f"   Enhanced: {enhanced[:100]}...")
        return

    if not args.prompt:
        # Default to first preset
        args.prompt = list(ENHANCED_PROMPTS.keys())[0]
        print(f"No prompt provided, using default: '{args.prompt}'")

    original_prompt = args.prompt
    enhanced_prompt = get_enhanced_prompt(original_prompt)

    print("=" * 60)
    print("PROMPT ENHANCEMENT")
    print("=" * 60)
    print(f"\nOriginal ({len(original_prompt)} chars):")
    print(f"  {original_prompt}")
    print(f"\nEnhanced ({len(enhanced_prompt)} chars):")
    print(f"  {enhanced_prompt}")
    print("=" * 60)

    # Import generation function
    from llm_dit.pipelines.generate import generate_video_with_offloading, GenerationConfig

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save prompt info
    prompt_info = {
        "original_prompt": original_prompt,
        "enhanced_prompt": enhanced_prompt,
        "config": {
            "frames": args.frames,
            "height": args.height,
            "width": args.width,
            "steps": args.steps,
            "seed": args.seed,
        }
    }
    with open(output_dir / "prompts.json", "w") as f:
        json.dump(prompt_info, f, indent=2)

    # Generation config
    # Note: CFG (guidance_scale > 1.0) doubles memory due to two forward passes
    # For 24GB GPU with FP8: 17 frames @ 256x384 with CFG=3.0 should fit
    config = GenerationConfig(
        num_frames=args.frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        seed=args.seed,
    )

    model_path = Path("models/LTX-2")

    prompts_to_generate = []
    if not args.enhanced_only:
        prompts_to_generate.append(("original", original_prompt))
    if not args.original_only:
        prompts_to_generate.append(("enhanced", enhanced_prompt))

    for name, prompt in prompts_to_generate:
        print(f"\n{'=' * 60}")
        print(f"GENERATING: {name.upper()}")
        print(f"{'=' * 60}")
        print(f"Prompt: {prompt[:80]}...")

        video = generate_video_with_offloading(
            prompt=prompt,
            model_path=model_path,
            config=config,
        )

        # Save video
        output_path = output_dir / f"video_{name}.mp4"

        import torch
        if isinstance(video, torch.Tensor):
            # Convert to numpy for saving
            video_np = video.cpu().numpy()
        else:
            video_np = video

        # Save using imageio
        import imageio.v3 as iio
        iio.imwrite(str(output_path), video_np, fps=24)

        print(f"Saved: {output_path}")

    print(f"\n{'=' * 60}")
    print(f"OUTPUT DIRECTORY: {output_dir}")
    print(f"{'=' * 60}")
    print("\nFiles created:")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
