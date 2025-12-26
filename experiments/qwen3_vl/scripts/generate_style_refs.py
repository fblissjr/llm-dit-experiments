#!/usr/bin/env python3
"""Generate style reference images for VL style transfer experiments.

Creates distinct visual styles that Homer Simpson would look unusual in,
to clearly demonstrate style transfer effects.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[3]))

from src.llm_dit import ZImagePipeline

# Paths relative to experiments/ directory
EXPERIMENTS_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = EXPERIMENTS_DIR / "inputs"

# Style reference prompts - scenes/subjects that have very distinct visual styles
STYLE_REFS = {
    "anime_girl": "A cute anime girl with big eyes and pink hair, cherry blossoms falling, soft pastel colors, anime art style, studio ghibli aesthetic",
    "noir_detective": "A black and white noir scene, detective in trench coat under streetlamp, rain, high contrast shadows, film noir cinematography, 1940s style",
    "oil_painting": "A beautiful oil painting of a vase of sunflowers, thick brushstrokes, impressionist style, warm golden colors, Vincent van Gogh inspired",
    "cyberpunk_city": "A neon-lit cyberpunk cityscape at night, holographic advertisements, rain-slicked streets, blade runner aesthetic, purple and cyan lighting",
    "watercolor_forest": "A serene forest scene in watercolor style, soft edges, bleeding colors, delicate washes, nature illustration, peaceful atmosphere",
    "pixel_art": "A retro pixel art scene of a castle on a hill, 16-bit video game style, limited color palette, nostalgic gaming aesthetic",
    "ukiyo_wave": "The great wave off kanagawa style, japanese woodblock print, traditional ukiyo-e art, blue and white, ocean waves, mount fuji",
    "art_deco": "An art deco poster design, geometric shapes, gold and black, 1920s glamour, elegant typography space, vintage luxury aesthetic",
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    import os
    model_path = os.environ.get("ZIMAGE_PATH")
    if not model_path:
        raise ValueError("Set ZIMAGE_PATH environment variable")

    print("Loading Z-Image pipeline...")
    pipe = ZImagePipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )

    for name, prompt in STYLE_REFS.items():
        print(f"\nGenerating: {name}")
        print(f"  Prompt: {prompt[:60]}...")

        generator = torch.Generator(device="cpu").manual_seed(42)
        result = pipe(
            prompt=prompt,
            num_inference_steps=9,
            generator=generator,
        )
        image = result.images[0] if hasattr(result, 'images') else result

        output_path = OUTPUT_DIR / f"style_{name}.png"
        image.save(output_path)
        print(f"  Saved: {output_path}")

    print(f"\nDone! Generated {len(STYLE_REFS)} style reference images in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
