#!/usr/bin/env python3
"""Compare different blending methods for VL + img2img.

Tests:
1. truncate (current) - takes first N tokens
2. interpolate - linear interpolation to compress VL
3. adain - transfer VL statistics to text structure
4. adain_per_dim - per-dimension AdaIN
"""

import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[3]))

from src.llm_dit.vl import VLEmbeddingExtractor
from src.llm_dit.vl.blending import (
    blend_embeddings,
    blend_adain,
    blend_adain_per_dim,
    _interpolate_sequence,
)
from src.llm_dit.startup import PipelineLoader
from src.llm_dit.cli import load_runtime_config
from experiments.qwen3_vl.scripts.grid_utils import make_grid


def blend_with_interpolation(vl_emb, text_emb, alpha):
    """Interpolate VL to text length, then blend."""
    vl_interp = _interpolate_sequence(vl_emb, text_emb.shape[0])
    vl_interp = vl_interp.to(device=text_emb.device, dtype=text_emb.dtype)
    return alpha * vl_interp + (1 - alpha) * text_emb


BLEND_METHODS = {
    "truncate": blend_embeddings,  # Current: takes first N tokens
    "interpolate": blend_with_interpolation,  # Compress all tokens
    "adain": blend_adain,  # Transfer VL stats to text structure
    "adain_per_dim": blend_adain_per_dim,  # Per-dimension AdaIN
}

# Test parameters
ALPHA = 0.3
STRENGTH = 0.9
PROMPT = "Homer Simpson"


def main():
    output_dir = Path("experiments/results/vl_blend_methods")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reference image
    ref_path = "experiments/inputs/style_anime_girl.png"
    reference = Image.open(ref_path).convert("RGB")
    reference.save(output_dir / "reference.png")

    # Load VL extractor
    import os
    vl_model_path = os.environ.get("QWEN3_VL_PATH")
    if not vl_model_path:
        raise ValueError("Set QWEN3_VL_PATH environment variable")
    print("Loading Qwen3-VL...")
    vl_extractor = VLEmbeddingExtractor.from_pretrained(
        vl_model_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
    )

    # Extract VL embeddings
    print(f"Extracting VL embeddings...")
    vl_result = vl_extractor.extract(
        reference,
        text=PROMPT,
        hidden_layer=-6,
        text_tokens_only=False,
        scale_to_text=True,
    )
    vl_emb = vl_result.embeddings
    print(f"  VL shape: {vl_emb.shape}, std: {vl_emb.std():.2f}")

    # Unload VL
    vl_extractor.unload()
    torch.cuda.empty_cache()

    # Load Z-Image pipeline
    print("Loading Z-Image pipeline...")
    class ConfigArgs:
        config = "config.toml"
        profile = "default"
    config = load_runtime_config(ConfigArgs())
    loader = PipelineLoader(config)
    pipe = loader.load_pipeline().pipeline

    # Get text embeddings
    print(f"Encoding text: {PROMPT}")
    text_emb = pipe.encoder.encode(PROMPT).embeddings[0]
    print(f"  Text shape: {text_emb.shape}, std: {text_emb.std():.2f}")

    # Generate baseline (no VL)
    print("\nGenerating baseline...")
    generator = torch.Generator(device="cuda").manual_seed(42)
    result = pipe.img2img(
        prompt_embeds=text_emb,
        image=reference,
        strength=STRENGTH,
        num_inference_steps=9,
        generator=generator,
    )
    result.save(output_dir / "baseline_no_vl.png")

    # Test each blend method
    results = {}
    for method_name, blend_fn in BLEND_METHODS.items():
        print(f"\n=== Testing: {method_name} ===")

        # Blend embeddings
        if method_name in ["adain", "adain_per_dim"]:
            # AdaIN methods take (text, vl, alpha) not (vl, text, alpha)
            blended = blend_fn(text_emb, vl_emb, alpha=ALPHA)
        else:
            blended = blend_fn(vl_emb, text_emb, alpha=ALPHA)

        print(f"  Blended shape: {blended.shape}, std: {blended.std():.2f}")

        # Generate
        generator = torch.Generator(device="cuda").manual_seed(42)
        result = pipe.img2img(
            prompt_embeds=blended,
            image=reference,
            strength=STRENGTH,
            num_inference_steps=9,
            generator=generator,
        )

        out_path = output_dir / f"{method_name}.png"
        result.save(out_path)
        print(f"  Saved: {out_path}")
        results[method_name] = out_path

    # Create comparison grid
    print("\n=== Creating comparison grid ===")
    images = [
        output_dir / "reference.png",
        output_dir / "baseline_no_vl.png",
    ]
    labels = ["Reference", "No VL"]

    for method_name in BLEND_METHODS.keys():
        images.append(output_dir / f"{method_name}.png")
        labels.append(method_name)

    make_grid(images, labels, cols=3, output_path=output_dir / "grid_comparison.png", cell_size=256)

    # Print summary
    print(f"\n=== Summary ===")
    print(f"Reference: {ref_path}")
    print(f"Prompt: {PROMPT}")
    print(f"Alpha: {ALPHA}, Strength: {STRENGTH}")
    print(f"VL shape: {vl_emb.shape} -> blended to: {text_emb.shape[0]} tokens")
    print(f"\nResults saved to: {output_dir}")
    print(f"Grid: {output_dir}/grid_comparison.png")


if __name__ == "__main__":
    main()
