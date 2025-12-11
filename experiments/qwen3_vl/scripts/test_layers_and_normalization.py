#!/usr/bin/env python3
"""
Test different hidden layers and normalization strategies for VL text tokens.

Hypothesis 1: Earlier layers may have less VL-specific artifacts
Hypothesis 2: Per-dimension normalization may align distributions better than global scaling

Memory management: Extract all VL embeddings first, unload VL, then generate.
"""

import sys
from pathlib import Path
import gc

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import torch
from PIL import Image
from dataclasses import dataclass
import json


@dataclass
class TestConfig:
    name: str
    hidden_layer: int
    normalization: str  # "global", "per_dim", "none"


def normalize_global(emb: torch.Tensor, target_std: float) -> torch.Tensor:
    """Scale to match target std (current approach)."""
    return emb * (target_std / emb.std())


def normalize_per_dim(emb: torch.Tensor, target_mean: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
    """Normalize each dimension independently to match target distribution."""
    emb_mean = emb.mean(dim=0, keepdim=True)
    emb_std = emb.std(dim=0, keepdim=True) + 1e-8

    normalized = (emb - emb_mean) / emb_std
    scaled = normalized * target_std.unsqueeze(0) + target_mean.unsqueeze(0)

    return scaled


def main():
    from llm_dit.vl import VLEmbeddingExtractor
    from llm_dit.cli import load_runtime_config
    from llm_dit.startup import PipelineLoader
    from llm_dit.conversation import Qwen3Formatter, Conversation, Message, Role

    prompt = "A cartoon house with a red roof on a green hill under a blue sky with yellow sun"

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

    output_dir = Path("experiments/results/vl_layers_and_norm")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reference image
    ref_image = Image.open("experiments/inputs/test_scene.png").convert("RGB")
    print(f"Reference image: {ref_image.size}")
    ref_image.save(output_dir / "reference.png")

    # =========================================================================
    # PHASE 1: Get Qwen3-4B text embedding stats (for normalization targets)
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Getting Qwen3-4B text embedding statistics...")
    print("=" * 60)

    loader = PipelineLoader(z_config)
    encoder_result = loader.load_encoder()
    encoder = encoder_result.encoder

    formatter = Qwen3Formatter()
    conv = Conversation(
        messages=[Message(role=Role.USER, content=prompt)],
        enable_thinking=True,
        is_final=True,
    )
    formatted = formatter.format(conv)

    text_result = encoder.encode(formatted)
    text_emb = text_result.embeddings[0].cpu()  # Move to CPU to save GPU memory

    # Get per-dimension stats
    text_per_dim_mean = text_emb.mean(dim=0)
    text_per_dim_std = text_emb.std(dim=0)
    text_global_std = text_emb.std().item()

    print(f"Text embedding shape: {text_emb.shape}")
    print(f"Text global std: {text_global_std:.2f}")
    print(f"Text per-dim std range: [{text_per_dim_std.min():.2f}, {text_per_dim_std.max():.2f}]")

    # Cleanup
    del encoder, encoder_result, loader, text_result
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # PHASE 2: Extract VL embeddings at different layers
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Extracting VL embeddings at different layers...")
    print("=" * 60)

    extractor = VLEmbeddingExtractor.from_pretrained(
        str(Path.home() / "Storage" / "Qwen3-VL-4B-Instruct"),
        device="cuda",
        torch_dtype=torch.bfloat16,
    )

    # Layers to test
    layers_to_test = [-2, -4, -8, -16, -24]

    raw_embeddings = {}
    for layer in layers_to_test:
        print(f"\nExtracting layer {layer}...")
        result = extractor.extract(
            image=ref_image,
            text=prompt,
            hidden_layer=layer,
            text_tokens_only=True,
            scale_to_text=False,  # We'll normalize ourselves
        )
        raw_embeddings[layer] = result.embeddings.cpu()  # Store on CPU
        print(f"  Shape: {result.embeddings.shape}, std: {result.embeddings.std():.2f}")

    # Cleanup VL
    extractor.unload()
    del extractor
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # PHASE 3: Load pipeline and generate images
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Generating images...")
    print("=" * 60)

    loader = PipelineLoader(z_config)
    pipeline_result = loader.load_pipeline()
    pipe = pipeline_result.pipeline

    # Generate pure text baseline first
    print("\nGenerating pure text baseline...")
    generator = torch.Generator().manual_seed(42)
    baseline = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=9,
        guidance_scale=0.0,
        generator=generator,
    )
    baseline.save(output_dir / "baseline_pure_text.png")

    # Test configurations
    configs = [
        # Different layers with global normalization
        TestConfig("layer_-2_global", -2, "global"),
        TestConfig("layer_-4_global", -4, "global"),
        TestConfig("layer_-8_global", -8, "global"),
        TestConfig("layer_-16_global", -16, "global"),
        TestConfig("layer_-24_global", -24, "global"),
        # Per-dimension normalization at key layers
        TestConfig("layer_-2_per_dim", -2, "per_dim"),
        TestConfig("layer_-8_per_dim", -8, "per_dim"),
        TestConfig("layer_-16_per_dim", -16, "per_dim"),
    ]

    results = []

    for config in configs:
        print(f"\n{'=' * 60}")
        print(f"Testing: {config.name}")
        print(f"  Layer: {config.hidden_layer}, Normalization: {config.normalization}")
        print("=" * 60)

        vl_emb = raw_embeddings[config.hidden_layer]
        print(f"  Raw VL: shape={vl_emb.shape}, std={vl_emb.std():.2f}, mean={vl_emb.mean():.4f}")

        # Apply normalization
        if config.normalization == "global":
            emb = normalize_global(vl_emb, target_std=text_global_std)
        elif config.normalization == "per_dim":
            emb = normalize_per_dim(vl_emb, text_per_dim_mean, text_per_dim_std)
        else:  # none
            emb = vl_emb

        print(f"  After norm: std={emb.std():.2f}, mean={emb.mean():.4f}")
        print(f"  Range: [{emb.min():.2f}, {emb.max():.2f}]")

        # Generate
        generator = torch.Generator().manual_seed(42)
        result = pipe(
            prompt_embeds=emb,
            height=1024,
            width=1024,
            num_inference_steps=9,
            guidance_scale=0.0,
            generator=generator,
        )

        output_path = output_dir / f"{config.name}.png"
        result.save(output_path)
        print(f"  Saved to {output_path}")

        results.append({
            "name": config.name,
            "layer": config.hidden_layer,
            "normalization": config.normalization,
            "raw_std": vl_emb.std().item(),
            "final_std": emb.std().item(),
            "final_mean": emb.mean().item(),
        })

    # Save results summary
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {output_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
