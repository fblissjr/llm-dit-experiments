#!/usr/bin/env python3
"""
Test different image vs text conditioning strengths.

We vary the text provided to VL:
- No text (just image) -> pure image understanding
- Short text -> minimal guidance
- Full descriptive text -> strong text guidance

This helps understand how much the VL model's understanding of the image
vs the text influences the output.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import torch
from PIL import Image


def main():
    from llm_dit.vl import VLEmbeddingExtractor
    from llm_dit.cli import load_runtime_config
    from llm_dit.startup import PipelineLoader

    # Load config
    class ConfigArgs:
        pass
    config_args = ConfigArgs()
    config_args.config = "config.toml"
    config_args.profile = "rtx4090"
    for attr in ['model_path', 'text_encoder_device', 'dit_device', 'vae_device',
                 'cpu_offload', 'flash_attn', 'compile', 'debug', 'verbose',
                 'attention_backend', 'use_custom_scheduler', 'tiled_vae',
                 'embedding_cache', 'long_prompt_mode', 'hidden_layer', 'shift',
                 'lora', 'api_url', 'api_model', 'local_encoder', 'templates_dir',
                 'torch_dtype', 'text_encoder_path', 'tile_size', 'tile_overlap',
                 'cache_size', 'steps', 'rewriter_use_api', 'rewriter_api_url',
                 'rewriter_api_model', 'rewriter_temperature', 'rewriter_top_p',
                 'rewriter_top_k', 'rewriter_min_p', 'rewriter_presence_penalty',
                 'rewriter_max_tokens', 'width', 'height', 'guidance_scale',
                 'negative_prompt', 'seed', 'embeddings_file', 'template',
                 'system_prompt', 'thinking_content', 'assistant_content',
                 'enable_thinking', 'vl_model_path', 'vl_device', 'vl_hidden_layer',
                 'vl_alpha', 'vl_blend_mode', 'vl_auto_unload']:
        if not hasattr(config_args, attr):
            setattr(config_args, attr, None)
    z_config = load_runtime_config(config_args)

    output_dir = Path("experiments/results/vl_conditioning_strength")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reference image
    ref_image = Image.open("experiments/inputs/test_scene.png").convert("RGB")
    print(f"Reference image: {ref_image.size}")
    ref_image.save(output_dir / "reference.png")

    # Different text conditioning levels
    text_conditions = [
        ("no_text", None),  # Just image, no text
        ("minimal", "image"),  # Single word
        ("short", "a house"),  # Basic description
        ("medium", "a cartoon house with red roof"),  # Medium detail
        ("full", "A cartoon house with a red roof on a green hill under a blue sky with yellow sun"),  # Full detail
        ("modify", "Transform this into a winter scene with snow"),  # Transformation request
        ("character", "Homer Simpson standing in front of this house"),  # Add character
        ("style", "This scene in watercolor painting style"),  # Style transfer
    ]

    # Load VL extractor
    print("\nLoading VL extractor...")
    extractor = VLEmbeddingExtractor.from_pretrained(
        str(Path.home() / "Storage" / "Qwen3-VL-4B-Instruct"),
        device="cuda",
        torch_dtype=torch.bfloat16,
    )

    # Extract embeddings for each condition
    embeddings = {}
    for name, text in text_conditions:
        print(f"\nExtracting: {name} -> {text}")
        result = extractor.extract(
            image=ref_image,
            text=text,
            hidden_layer=-8,  # Use layer -8 based on experiments (cleaner than -2)
            text_tokens_only=True,
            scale_to_text=True,
        )
        embeddings[name] = {
            "emb": result.embeddings,
            "text": text,
            "shape": result.embeddings.shape,
            "std": result.scaled_std,
            "scale_factor": result.scale_factor,
        }
        print(f"  Shape: {result.embeddings.shape}, std: {result.scaled_std:.2f}, scale: {result.scale_factor:.2f}")

    # Unload VL
    extractor.unload()
    torch.cuda.empty_cache()

    # Load pipeline
    print("\nLoading Z-Image pipeline...")
    loader = PipelineLoader(z_config)
    pipeline_result = loader.load_pipeline()
    pipe = pipeline_result.pipeline

    # Generate images
    results = []
    for name, data in embeddings.items():
        print(f"\nGenerating: {name}...")
        generator = torch.Generator().manual_seed(42)

        result = pipe(
            prompt_embeds=data["emb"],
            height=1024,
            width=1024,
            num_inference_steps=9,
            guidance_scale=0.0,
            generator=generator,
        )

        output_path = output_dir / f"{name}.png"
        result.save(output_path)
        print(f"  Saved to {output_path}")

        results.append({
            "name": name,
            "text": data["text"],
            "shape": list(data["shape"]),
            "std": data["std"],
            "scale_factor": data["scale_factor"],
        })

    # Also generate pure text versions for comparison
    print("\nGenerating pure text comparisons...")
    text_comparisons = [
        ("pure_text_full", "A cartoon house with a red roof on a green hill under a blue sky with yellow sun"),
        ("pure_text_homer", "Homer Simpson standing in front of a cartoon house"),
        ("pure_text_winter", "A winter scene with a house covered in snow"),
    ]

    for name, prompt in text_comparisons:
        print(f"  Generating: {name}...")
        generator = torch.Generator().manual_seed(42)
        result = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=9,
            guidance_scale=0.0,
            generator=generator,
        )
        result.save(output_dir / f"{name}.png")

    # Save results
    import json
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
