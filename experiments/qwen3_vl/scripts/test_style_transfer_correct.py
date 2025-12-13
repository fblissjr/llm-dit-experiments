#!/usr/bin/env python3
"""Correct style transfer test: VL for style, img2img for structure.

The key insight: use DIFFERENT images for VL and img2img:
- VL embeddings from STYLE reference (anime, noir, etc.) → provides style
- img2img from CONTENT image (Homer baseline) → provides structure
- Text prompt → provides semantics
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/fbliss/Storage/Tongyi-MAI_Z-Image-Turbo")
    parser.add_argument("--vl-model-path", default="/home/fbliss/Storage/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output-dir", default="/home/fbliss/workspace/llm-dit-experiments/experiments/results/vl_style_transfer_correct")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


STYLES = {
    "anime": "/home/fbliss/workspace/llm-dit-experiments/experiments/inputs/style_anime_girl.png",
    "noir": "/home/fbliss/workspace/llm-dit-experiments/experiments/inputs/style_noir_detective.png",
    "oil": "/home/fbliss/workspace/llm-dit-experiments/experiments/inputs/style_oil_painting.png",
    "cyberpunk": "/home/fbliss/workspace/llm-dit-experiments/experiments/inputs/style_cyberpunk_city.png",
}

PROMPT = "Homer Simpson"
ALPHAS = [0.1, 0.2, 0.3]
STRENGTHS = [0.3, 0.5, 0.7]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # PHASE 1: Generate Homer baseline (this will be our CONTENT image)
    print("\n=== PHASE 1: Generate Homer Baseline ===")
    pipe = ZImagePipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )

    text_emb = pipe.encode_prompt(PROMPT)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    result = pipe(prompt_embeds=text_emb, num_inference_steps=9, generator=generator)
    homer_baseline = result.images[0] if hasattr(result, 'images') else result
    homer_baseline.save(output_dir / "homer_baseline.png")
    print("Saved homer_baseline.png")

    # Unload pipeline to make room for VL
    del pipe
    torch.cuda.empty_cache()

    # PHASE 2: Extract VL embeddings from style references
    print("\n=== PHASE 2: Extract VL Embeddings ===")
    vl_extractor = VLEmbeddingExtractor.from_pretrained(
        args.vl_model_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
    )

    vl_cache = {}
    for style_name, style_path in STYLES.items():
        print(f"Extracting VL for {style_name}...")
        style_img = Image.open(style_path).convert("RGB")
        style_img.save(output_dir / f"style_{style_name}.png")

        vl_result = vl_extractor.extract(
            style_img,
            text=PROMPT,  # Use same prompt for VL extraction
            hidden_layer=-6,
            text_tokens_only=False,
            scale_to_text=True,
        )
        vl_cache[style_name] = vl_result.embeddings.cpu()

    vl_extractor.unload()
    del vl_extractor
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # PHASE 3: Reload pipeline and generate styled Homer
    print("\n=== PHASE 3: Generate Styled Homer ===")
    pipe = ZImagePipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )

    # Load Homer baseline as content image for img2img
    homer_img = Image.open(output_dir / "homer_baseline.png").convert("RGB")

    for style_name in STYLES.keys():
        print(f"\n--- Style: {style_name} ---")
        vl_emb = vl_cache[style_name].to("cuda")
        text_emb = pipe.encode_prompt(PROMPT)

        for alpha in ALPHAS:
            for strength in STRENGTHS:
                print(f"  alpha={alpha}, strength={strength}...")

                # Blend VL (style) + text (semantics)
                blended = blend_embeddings(vl_emb, text_emb, alpha)

                # img2img from Homer baseline (structure)
                generator = torch.Generator(device="cuda").manual_seed(args.seed)
                result = pipe.img2img(
                    prompt_embeds=blended,
                    image=homer_img,  # Homer baseline as structure source!
                    strength=strength,
                    num_inference_steps=9,
                    generator=generator,
                )
                image = result.images[0] if hasattr(result, 'images') else result

                filename = f"{style_name}_a{int(alpha*10)}_s{int(strength*10)}.png"
                image.save(output_dir / filename)

    # Create grids
    print("\n=== Creating Grids ===")

    # Per-style grids
    for style_name in STYLES.keys():
        images = [
            output_dir / f"style_{style_name}.png",
            output_dir / "homer_baseline.png",
        ]
        labels = [f"{style_name} style", "Homer content"]

        for strength in STRENGTHS:
            for alpha in ALPHAS:
                images.append(output_dir / f"{style_name}_a{int(alpha*10)}_s{int(strength*10)}.png")
                labels.append(f"a={alpha} s={strength}")

        make_grid(images, labels, cols=5, output_path=output_dir / f"grid_{style_name}.png", cell_size=256)

    # Overview: best results (alpha=0.2, strength=0.5)
    overview = [output_dir / "homer_baseline.png"]
    labels = ["Homer"]
    for style_name in STYLES.keys():
        overview.append(output_dir / f"style_{style_name}.png")
        labels.append(f"{style_name} ref")
        overview.append(output_dir / f"{style_name}_a2_s5.png")
        labels.append(f"{style_name} result")

    make_grid(overview, labels, cols=3, output_path=output_dir / "grid_overview.png", cell_size=256)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
