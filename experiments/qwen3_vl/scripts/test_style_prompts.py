#!/usr/bin/env python3
"""Test VL conditioning with style transfer prompts.

Tests whether VL can transfer style from reference images using prompts like:
- "Create Homer Simpson in the style of this image"
- "Homer Simpson rendered in the artistic style shown in the image"
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[3]))

from src.llm_dit.vl import VLEmbeddingExtractor, blend_embeddings
from src.llm_dit import ZImagePipeline
from experiments.qwen3_vl.scripts.grid_utils import make_grid


# Paths relative to experiments/ directory
EXPERIMENTS_DIR = Path(__file__).parent.parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Test style prompts with VL")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to Z-Image model",
    )
    parser.add_argument(
        "--vl-model-path",
        type=str,
        required=True,
        help="Path to Qwen3-VL model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(EXPERIMENTS_DIR / "results/vl_style_prompts"),
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


# Reference images for style transfer
REFERENCE_IMAGES = {
    "anime": str(EXPERIMENTS_DIR / "inputs/style_anime_girl.png"),
    "noir": str(EXPERIMENTS_DIR / "inputs/style_noir_detective.png"),
    "oil": str(EXPERIMENTS_DIR / "inputs/style_oil_painting.png"),
    "cyberpunk": str(EXPERIMENTS_DIR / "inputs/style_cyberpunk_city.png"),
    "watercolor": str(EXPERIMENTS_DIR / "inputs/style_watercolor_forest.png"),
    "pixel": str(EXPERIMENTS_DIR / "inputs/style_pixel_art.png"),
    "ukiyo": str(EXPERIMENTS_DIR / "inputs/style_ukiyo_wave.png"),
    "deco": str(EXPERIMENTS_DIR / "inputs/style_art_deco.png"),
}

# Style transfer prompts (explicitly ask for style from the image)
STYLE_PROMPTS = [
    "Create Homer Simpson in the style of this image",
    "Homer Simpson rendered in the artistic style shown in the image",
]

# Alpha values to test
ALPHAS = [0.3, 0.5, 0.7]


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # PHASE 1: Load VL and extract all embeddings
    print("\n=== PHASE 1: Extracting VL Embeddings ===")
    print("Loading VL extractor...")
    vl_extractor = VLEmbeddingExtractor.from_pretrained(
        args.vl_model_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
    )

    vl_cache = {}  # (ref_name, prompt_idx) -> embeddings

    for ref_name, ref_path in REFERENCE_IMAGES.items():
        print(f"\nProcessing style: {ref_name}")
        reference = Image.open(ref_path).convert("RGB")
        reference.save(output_dir / f"reference_{ref_name}.png")

        for i, prompt in enumerate(STYLE_PROMPTS):
            print(f"  Prompt {i+1}: {prompt[:40]}...")
            vl_result = vl_extractor.extract(
                reference,
                text=prompt,
                hidden_layer=-6,
                text_tokens_only=False,
                scale_to_text=True,
            )
            vl_cache[(ref_name, i)] = vl_result.embeddings.cpu()

    # Unload VL
    print("\nUnloading VL model...")
    vl_extractor.unload()
    del vl_extractor
    torch.cuda.empty_cache()

    # PHASE 2: Load Z-Image and generate
    print("\n=== PHASE 2: Loading Z-Image Pipeline ===")
    pipe = ZImagePipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )

    # Generate baseline
    print("\n=== Generating Baseline ===")
    baseline_prompt = "Homer Simpson"
    text_emb = pipe.encode_prompt(baseline_prompt)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    result = pipe(
        prompt_embeds=text_emb,
        num_inference_steps=9,
        generator=generator,
    )
    image = result.images[0] if hasattr(result, 'images') else result
    image.save(output_dir / "baseline_homer.png")

    # PHASE 3: Generate styled images
    print("\n=== PHASE 3: Generating Styled Images ===")
    for ref_name in REFERENCE_IMAGES.keys():
        print(f"\n--- Style: {ref_name} ---")

        for i, prompt in enumerate(STYLE_PROMPTS):
            print(f"Prompt {i+1}: {prompt[:40]}...")
            vl_emb = vl_cache[(ref_name, i)].to("cuda")

            for alpha in ALPHAS:
                print(f"  Alpha {alpha}...")

                text_emb = pipe.encode_prompt(prompt)
                blended = blend_embeddings(vl_emb, text_emb, alpha)

                generator = torch.Generator(device="cpu").manual_seed(args.seed)
                result = pipe(
                    prompt_embeds=blended,
                    num_inference_steps=9,
                    generator=generator,
                )
                image = result.images[0] if hasattr(result, 'images') else result

                filename = f"style_{ref_name}_p{i+1}_a{int(alpha*10)}.png"
                image.save(output_dir / filename)

    # Create grids
    print("\n=== Creating Grids ===")

    # Main overview grid: all references
    print("Creating references grid...")
    overview_images = [Image.open(output_dir / "baseline_homer.png")]
    overview_labels = ["Baseline"]

    for ref_name in REFERENCE_IMAGES.keys():
        overview_images.append(Image.open(output_dir / f"reference_{ref_name}.png"))
        overview_labels.append(f"Ref: {ref_name}")

    make_grid(
        overview_images, overview_labels,
        cols=3,
        output_path=output_dir / "grid_references.png",
        cell_size=256,
    )

    # Results grid: reference -> alpha 0.5 result for each style
    print("Creating results grid...")
    results_images = []
    results_labels = []

    for ref_name in REFERENCE_IMAGES.keys():
        results_images.append(Image.open(output_dir / f"reference_{ref_name}.png"))
        results_labels.append(f"{ref_name} ref")
        results_images.append(Image.open(output_dir / f"style_{ref_name}_p1_a5.png"))
        results_labels.append(f"{ref_name} a=0.5")

    make_grid(
        results_images, results_labels,
        cols=4,
        output_path=output_dir / "grid_results.png",
        cell_size=256,
    )

    # Alpha comparison for select styles
    print("Creating alpha comparison grid...")
    select_styles = ["anime", "noir", "cyberpunk", "oil"]
    alpha_images = []
    alpha_labels = []

    for style in select_styles:
        alpha_images.append(Image.open(output_dir / f"reference_{style}.png"))
        alpha_labels.append(f"{style} ref")
        for alpha in ALPHAS:
            alpha_images.append(Image.open(output_dir / f"style_{style}_p1_a{int(alpha*10)}.png"))
            alpha_labels.append(f"a={alpha}")

    make_grid(
        alpha_images, alpha_labels,
        cols=4,
        output_path=output_dir / "grid_alpha_comparison.png",
        cell_size=256,
    )

    # Save prompts
    with open(output_dir / "prompts.txt", "w") as f:
        f.write("=== Style Transfer Prompts ===\n")
        for i, p in enumerate(STYLE_PROMPTS):
            f.write(f"P{i+1}: {p}\n")
        f.write("\n=== Reference Images ===\n")
        for name, path in REFERENCE_IMAGES.items():
            f.write(f"{name}: {path}\n")

    print(f"\nResults saved to {output_dir}")
    print("Grids created:")
    print("  - grid_references.png (all reference styles)")
    print("  - grid_results.png (ref -> result pairs)")
    print("  - grid_alpha_comparison.png (alpha sweep for 4 styles)")


if __name__ == "__main__":
    main()
