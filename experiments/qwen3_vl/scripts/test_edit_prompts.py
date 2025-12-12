#!/usr/bin/env python3
"""Test VL conditioning with image editing style prompts.

Tests whether VL responds differently to:
1. Edit-style instructions: "Add Homer Simpson to this image"
2. Descriptive prompts: "Homer Simpson in a sunset landscape"
3. Hybrid prompts: "Homer Simpson standing in this scenic landscape"
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


def parse_args():
    parser = argparse.ArgumentParser(description="Test edit-style prompts with VL")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/fbliss/Storage/Tongyi-MAI_Z-Image-Turbo",
        help="Path to Z-Image model",
    )
    parser.add_argument(
        "--vl-model-path",
        type=str,
        default="/home/fbliss/Storage/Qwen3-VL-4B-Instruct",
        help="Path to Qwen3-VL model",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="/home/fbliss/workspace/llm-dit-experiments/experiments/inputs/sunset.png",
        help="Reference image path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/fbliss/workspace/llm-dit-experiments/experiments/results/vl_edit_prompts",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


# 1. Edit-style prompts (instruct the model to modify the reference)
EDIT_PROMPTS = [
    "Add Homer Simpson to this image",
    "Put Homer Simpson in this scene",
    "Place a cartoon Homer Simpson in the foreground",
]

# 2. Descriptive prompts (just describe what should be there, no reference to image)
DESCRIPTIVE_PROMPTS = [
    "Homer Simpson in a sunset landscape",
    "Homer Simpson standing in a scenic outdoor setting at golden hour",
    "A cartoon Homer Simpson with a beautiful colorful sky behind him",
]

# 3. Hybrid prompts (describe what you want + explicitly reference "the image")
HYBRID_PROMPTS = [
    "Homer Simpson standing in the scenic landscape from the image",
    "Homer Simpson in the foreground of the landscape as shown in the image",
    "Homer Simpson standing in the outdoor scene as depicted in the image",
]

# Alpha values to test
ALPHAS = [0.3, 0.5, 0.7]


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reference image
    print(f"Loading reference image: {args.reference}")
    reference = Image.open(args.reference).convert("RGB")
    reference.save(output_dir / "reference.png")

    # Load models
    print("Loading Z-Image pipeline...")
    pipe = ZImagePipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )

    print("Loading VL extractor...")
    vl_extractor = VLEmbeddingExtractor.from_pretrained(
        args.vl_model_path,
        device="cpu",  # Save VRAM
        torch_dtype=torch.bfloat16,
    )

    all_prompt_sets = [
        ("edit", EDIT_PROMPTS),
        ("desc", DESCRIPTIVE_PROMPTS),
        ("hybrid", HYBRID_PROMPTS),
    ]

    # Test each prompt set
    for set_name, prompts in all_prompt_sets:
        print(f"\n=== Testing {set_name.upper()} Prompts ===")

        for i, prompt in enumerate(prompts):
            print(f"\nPrompt {i+1}: {prompt}")

            for alpha in ALPHAS:
                print(f"  Alpha {alpha}...")

                # Extract VL embeddings with this prompt
                vl_result = vl_extractor.extract(
                    reference,
                    text=prompt,
                    hidden_layer=-6,
                    text_tokens_only=False,
                    scale_to_text=True,
                )
                vl_emb = vl_result.embeddings

                # Get text embeddings
                text_emb = pipe.encode_prompt(prompt)

                # Blend using the utility function (handles length mismatch)
                blended = blend_embeddings(vl_emb, text_emb, alpha)

                # Generate
                generator = torch.Generator(device="cpu").manual_seed(args.seed)
                result = pipe(
                    prompt_embeds=blended,
                    num_inference_steps=9,
                    generator=generator,
                )
                image = result.images[0] if hasattr(result, 'images') else result

                # Save
                filename = f"{set_name}_{i+1}_a{int(alpha*10)}.png"
                image.save(output_dir / filename)

    # Generate baselines (pure text, no VL)
    print("\n=== Pure Text Baselines (no VL) ===")
    baseline_prompts = [
        ("edit", EDIT_PROMPTS[0]),
        ("desc", DESCRIPTIVE_PROMPTS[0]),
        ("hybrid", HYBRID_PROMPTS[0]),
    ]
    for name, prompt in baseline_prompts:
        print(f"Baseline ({name}): {prompt}")
        text_emb = pipe.encode_prompt(prompt)
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        result = pipe(
            prompt_embeds=text_emb,
            num_inference_steps=9,
            generator=generator,
        )
        image = result.images[0] if hasattr(result, 'images') else result
        image.save(output_dir / f"baseline_{name}.png")

    # Unload VL model
    vl_extractor.unload()

    # Create comparison grids
    print("\n=== Creating Grids ===")

    # Grid for each prompt type
    for set_name, prompts in all_prompt_sets:
        images = []
        labels = []
        for i in range(len(prompts)):
            for alpha in ALPHAS:
                filename = f"{set_name}_{i+1}_a{int(alpha*10)}.png"
                images.append(Image.open(output_dir / filename))
                labels.append(f"P{i+1} a={alpha}")

        make_grid(
            images, labels,
            cols=len(ALPHAS),
            output_path=output_dir / f"grid_{set_name}.png",
            cell_size=256,
        )

    # Comparison grid: all 3 types at alpha=0.5
    print("Creating comparison grid (all types at a=0.5)...")
    comparison_images = [Image.open(output_dir / "reference.png")]
    comparison_labels = ["Reference"]

    # Add baselines
    for name, _ in baseline_prompts:
        comparison_images.append(Image.open(output_dir / f"baseline_{name}.png"))
        comparison_labels.append(f"{name.title()} (no VL)")

    # Add VL versions at alpha=0.5
    for set_name, prompts in all_prompt_sets:
        comparison_images.append(Image.open(output_dir / f"{set_name}_1_a5.png"))
        comparison_labels.append(f"{set_name.title()} a=0.5")

    make_grid(
        comparison_images, comparison_labels,
        cols=4,
        output_path=output_dir / "grid_comparison.png",
        cell_size=256,
    )

    # Alpha progression for each type (prompt 1 across all alphas)
    print("Creating alpha progression grid...")
    prog_images = []
    prog_labels = []
    for set_name, _ in all_prompt_sets:
        for alpha in ALPHAS:
            prog_images.append(Image.open(output_dir / f"{set_name}_1_a{int(alpha*10)}.png"))
            prog_labels.append(f"{set_name} a={alpha}")

    make_grid(
        prog_images, prog_labels,
        cols=len(ALPHAS),
        output_path=output_dir / "grid_alpha_progression.png",
        cell_size=256,
    )

    # Save prompts to file for reference
    with open(output_dir / "prompts.txt", "w") as f:
        f.write("=== Edit-Style Prompts ===\n")
        for i, p in enumerate(EDIT_PROMPTS):
            f.write(f"E{i+1}: {p}\n")
        f.write("\n=== Descriptive Prompts ===\n")
        for i, p in enumerate(DESCRIPTIVE_PROMPTS):
            f.write(f"D{i+1}: {p}\n")
        f.write("\n=== Hybrid Prompts ===\n")
        for i, p in enumerate(HYBRID_PROMPTS):
            f.write(f"H{i+1}: {p}\n")

    print(f"\nResults saved to {output_dir}")
    print("Grids created:")
    print("  - grid_edit.png (edit prompts x alphas)")
    print("  - grid_desc.png (descriptive prompts x alphas)")
    print("  - grid_hybrid.png (hybrid prompts x alphas)")
    print("  - grid_comparison.png (all types side-by-side)")
    print("  - grid_alpha_progression.png (alpha sweep for each type)")


if __name__ == "__main__":
    main()
