#!/usr/bin/env python3
"""
Test VL conditioning WITHOUT scaling.

The hypothesis is that the large scale factors (5-10x) needed to match
text embedding std are destroying the embedding structure.

This test generates images with:
1. No scaling at all (raw VL embeddings)
2. Different scale factors to find the sweet spot
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import torch
from PIL import Image
from dataclasses import dataclass

@dataclass
class ScalingConfig:
    name: str
    scale_factor: float | None  # None = no scaling, float = explicit factor


def main():
    from llm_dit.vl import VLEmbeddingExtractor
    from llm_dit.cli import load_runtime_config
    from llm_dit.startup import PipelineLoader

    prompt = "Homer Simpson eating spaghetti"
    image_path = Path(__file__).parent.parent.parent / "inputs" / "test_scene.png"
    output_dir = Path("experiments/results/scaling_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    image = Image.open(image_path).convert("RGB")
    print(f"Loaded image: {image.size}")

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

    # Find VL model
    vl_model_path = None
    for candidate in [
        Path.home() / "Storage" / "Qwen3-VL-4B-Instruct",
        Path.home() / "models" / "Qwen3-VL-4B-Instruct",
    ]:
        if candidate.exists():
            vl_model_path = str(candidate)
            break

    # Load VL extractor
    print(f"Loading VL extractor from {vl_model_path}...")
    extractor = VLEmbeddingExtractor.from_pretrained(
        vl_model_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
    )

    # Extract raw embeddings (no scaling)
    print("Extracting VL embeddings (no scaling)...")
    vl_result = extractor.extract(
        image=image,
        text=prompt,
        hidden_layer=-2,
        scale_to_text=False,  # NO SCALING
    )
    vl_emb_raw = vl_result.embeddings
    print(f"Raw VL: shape={vl_emb_raw.shape}, std={vl_emb_raw.std():.2f}, mean={vl_emb_raw.mean():.4f}")

    # Unload VL to free memory
    extractor.unload()
    torch.cuda.empty_cache()

    # Load pipeline
    print("Loading Z-Image pipeline...")
    loader = PipelineLoader(z_config)
    pipeline_result = loader.load_pipeline()
    pipe = pipeline_result.pipeline

    # Test different scaling approaches
    configs = [
        ScalingConfig("no_scaling", None),
        ScalingConfig("scale_1.5x", 1.5),
        ScalingConfig("scale_2x", 2.0),
        ScalingConfig("scale_3x", 3.0),
        ScalingConfig("scale_5x", 5.0),
    ]

    generator = torch.Generator().manual_seed(42)

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config.name}")
        print(f"{'='*60}")

        if config.scale_factor is None:
            emb = vl_emb_raw.clone()
        else:
            emb = vl_emb_raw * config.scale_factor

        print(f"  Embedding std: {emb.std():.2f}")
        print(f"  Embedding mean: {emb.mean():.4f}")
        print(f"  Embedding range: [{emb.min():.2f}, {emb.max():.2f}]")

        generator.manual_seed(42)
        result = pipe(
            prompt_embeds=emb,
            height=z_config.height,
            width=z_config.width,
            num_inference_steps=z_config.steps,
            guidance_scale=z_config.guidance_scale,
            generator=generator,
        )

        output_path = output_dir / f"{config.name}.png"
        result.save(output_path)
        print(f"  Saved to {output_path}")

    print(f"\nResults saved to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
