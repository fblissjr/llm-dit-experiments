#!/usr/bin/env python3
"""Test VL + img2img combined for style transfer.

This uses all 3 input sources:
- VL embeddings: style/mood from reference
- img2img (VAE encode): structure from reference
- text embeddings: semantic content (Homer Simpson)
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[3]))

from src.llm_dit.vl import VLEmbeddingExtractor, blend_embeddings
from src.llm_dit import ZImagePipeline
from experiments.qwen3_vl.scripts.grid_utils import make_grid


def parse_args():
    parser = argparse.ArgumentParser(description="Test VL + img2img style transfer")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/fbliss/Storage/Tongyi-MAI_Z-Image-Turbo",
    )
    parser.add_argument(
        "--vl-model-path",
        type=str,
        default="/home/fbliss/Storage/Qwen3-VL-4B-Instruct",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/fbliss/workspace/llm-dit-experiments/experiments/results/vl_img2img_styles",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# Style reference images
STYLES = {
    "anime": "/home/fbliss/workspace/llm-dit-experiments/experiments/inputs/style_anime_girl.png",
    "noir": "/home/fbliss/workspace/llm-dit-experiments/experiments/inputs/style_noir_detective.png",
    "oil": "/home/fbliss/workspace/llm-dit-experiments/experiments/inputs/style_oil_painting.png",
    "cyberpunk": "/home/fbliss/workspace/llm-dit-experiments/experiments/inputs/style_cyberpunk_city.png",
}

# Test prompt
PROMPT = "Homer Simpson"

# Parameters to test
ALPHAS = [0.1, 0.2, 0.3]  # Lower alphas to preserve Homer
STRENGTHS = [0.5, 0.7]    # img2img strength


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # PHASE 1: Extract VL embeddings
    print("\n=== PHASE 1: Extracting VL Embeddings ===")
    vl_extractor = VLEmbeddingExtractor.from_pretrained(
        args.vl_model_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
    )

    vl_cache = {}
    for style_name, style_path in STYLES.items():
        print(f"Extracting VL for {style_name}...")
        ref_img = Image.open(style_path).convert("RGB")
        ref_img.save(output_dir / f"ref_{style_name}.png")

        vl_result = vl_extractor.extract(
            ref_img,
            text=PROMPT,
            hidden_layer=-6,
            text_tokens_only=False,
            scale_to_text=True,
        )
        vl_cache[style_name] = vl_result.embeddings.cpu()

    vl_extractor.unload()
    del vl_extractor
    torch.cuda.empty_cache()

    # PHASE 2: Load pipeline and generate
    print("\n=== PHASE 2: Loading Pipeline ===")
    pipe = ZImagePipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )

    # Generate baseline (pure text)
    print("\nGenerating baseline...")
    text_emb = pipe.encode_prompt(PROMPT)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    result = pipe(prompt_embeds=text_emb, num_inference_steps=9, generator=generator)
    baseline = result.images[0] if hasattr(result, 'images') else result
    baseline.save(output_dir / "baseline.png")

    # PHASE 3: Generate with VL + img2img
    print("\n=== PHASE 3: Generating VL + img2img ===")

    for style_name, style_path in STYLES.items():
        print(f"\n--- Style: {style_name} ---")
        ref_img = Image.open(style_path).convert("RGB")
        vl_emb = vl_cache[style_name].to("cuda")
        text_emb = pipe.encode_prompt(PROMPT)

        for alpha in ALPHAS:
            for strength in STRENGTHS:
                print(f"  alpha={alpha}, strength={strength}...")

                # Blend VL + text
                blended = blend_embeddings(vl_emb, text_emb, alpha)

                # Generate with img2img
                generator = torch.Generator(device="cuda").manual_seed(args.seed)
                result = pipe.img2img(
                    prompt_embeds=blended,
                    image=ref_img,
                    strength=strength,
                    num_inference_steps=9,
                    generator=generator,
                )
                image = result.images[0] if hasattr(result, 'images') else result

                filename = f"{style_name}_a{int(alpha*10)}_s{int(strength*10)}.png"
                image.save(output_dir / filename)

    # Also test txt2img with VL only (for comparison)
    print("\n=== Generating txt2img with VL (no img2img) ===")
    for style_name in STYLES.keys():
        vl_emb = vl_cache[style_name].to("cuda")
        text_emb = pipe.encode_prompt(PROMPT)
        blended = blend_embeddings(vl_emb, text_emb, 0.2)  # Low alpha

        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        result = pipe(prompt_embeds=blended, num_inference_steps=9, generator=generator)
        image = result.images[0] if hasattr(result, 'images') else result
        image.save(output_dir / f"{style_name}_txt2img_a2.png")

    # Create grids
    print("\n=== Creating Grids ===")

    # Grid: txt2img vs img2img comparison for each style
    for style_name in STYLES.keys():
        images = [
            output_dir / f"ref_{style_name}.png",
            output_dir / "baseline.png",
            output_dir / f"{style_name}_txt2img_a2.png",
        ]
        labels = ["Reference", "Baseline", "txt2img a=0.2"]

        for strength in STRENGTHS:
            for alpha in ALPHAS:
                images.append(output_dir / f"{style_name}_a{int(alpha*10)}_s{int(strength*10)}.png")
                labels.append(f"a={alpha} s={strength}")

        make_grid(images, labels, cols=3, output_path=output_dir / f"grid_{style_name}.png", cell_size=256)

    # Overview grid: all styles at alpha=0.2, strength=0.7
    overview_images = [output_dir / "baseline.png"]
    overview_labels = ["Baseline"]
    for style_name in STYLES.keys():
        overview_images.append(output_dir / f"ref_{style_name}.png")
        overview_labels.append(f"{style_name} ref")
        overview_images.append(output_dir / f"{style_name}_a2_s7.png")
        overview_labels.append(f"{style_name} a=0.2 s=0.7")

    make_grid(overview_images, overview_labels, cols=3, output_path=output_dir / "grid_overview.png", cell_size=256)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
