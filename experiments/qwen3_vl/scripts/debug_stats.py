#!/usr/bin/env python3
"""
Debug script to compare embedding statistics between Qwen3-4B and Qwen3-VL.

This helps diagnose why VL embeddings produce noisy/corrupted images.
"""

import sys
from pathlib import Path

import torch
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from llm_dit.vl import VLEmbeddingExtractor
from llm_dit.cli import load_runtime_config


def main():
    # Load a test image
    image_path = Path(__file__).parent.parent.parent / "inputs" / "test_scene.png"
    if not image_path.exists():
        print(f"Test image not found: {image_path}")
        return 1

    image = Image.open(image_path).convert("RGB")
    print(f"Loaded image: {image.size}")

    # Test prompt
    prompt = "A red barn in a sunny field"

    # Find VL model
    vl_model_path = None
    for candidate in [
        Path.home() / "Storage" / "Qwen3-VL-4B-Instruct",
        Path.home() / "models" / "Qwen3-VL-4B-Instruct",
    ]:
        if candidate.exists():
            vl_model_path = str(candidate)
            break

    if not vl_model_path:
        print("Could not find Qwen3-VL model")
        return 1

    print(f"\nLoading VL extractor from {vl_model_path}...")
    extractor = VLEmbeddingExtractor.from_pretrained(
        vl_model_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
    )

    print("\n" + "=" * 60)
    print("Extracting VL embeddings (WITH scaling)")
    print("=" * 60)

    result_scaled = extractor.extract(
        image=image,
        text=prompt,
        hidden_layer=-2,
        scale_to_text=True,
    )

    print(f"  Token selection: {result_scaled.token_selection}")
    print(f"  Num tokens: {result_scaled.num_tokens}")
    print(f"  Hidden layer: {result_scaled.hidden_layer}")
    print(f"  ORIGINAL std: {result_scaled.original_std:.4f}")
    print(f"  SCALED std: {result_scaled.scaled_std:.4f}")
    print(f"  Scale factor: {result_scaled.scale_factor:.4f}")
    print(f"  Chat template: {result_scaled.chat_template_format}")

    emb = result_scaled.embeddings
    print(f"\n  Embedding stats (after scaling):")
    print(f"    Shape: {emb.shape}")
    print(f"    Mean: {emb.mean():.4f}")
    print(f"    Std: {emb.std():.4f}")
    print(f"    Min: {emb.min():.4f}")
    print(f"    Max: {emb.max():.4f}")

    print("\n" + "=" * 60)
    print("Extracting VL embeddings (WITHOUT scaling)")
    print("=" * 60)

    result_unscaled = extractor.extract(
        image=image,
        text=prompt,
        hidden_layer=-2,
        scale_to_text=False,
    )

    emb_raw = result_unscaled.embeddings
    print(f"  Embedding stats (raw, no scaling):")
    print(f"    Shape: {emb_raw.shape}")
    print(f"    Mean: {emb_raw.mean():.4f}")
    print(f"    Std: {emb_raw.std():.4f}")
    print(f"    Min: {emb_raw.min():.4f}")
    print(f"    Max: {emb_raw.max():.4f}")

    print("\n" + "=" * 60)
    print("Loading Qwen3-4B for text embedding comparison")
    print("=" * 60)

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

    # Encode text with Qwen3-4B
    from llm_dit.startup import PipelineLoader
    loader = PipelineLoader(z_config)

    # Just load text encoder
    print("Loading text encoder...")
    text_result = loader.load_text_encoder()
    encoder = text_result.encoder

    # Encode the prompt
    print(f"Encoding prompt: {prompt}")
    from llm_dit.conversation import Qwen3Formatter
    formatter = Qwen3Formatter()
    formatted = formatter.format(prompt)

    encode_result = encoder.encode(formatted)
    text_emb = encode_result.embeddings[0]  # Remove batch dim

    print(f"\n  Qwen3-4B text embedding stats:")
    print(f"    Shape: {text_emb.shape}")
    print(f"    Mean: {text_emb.mean():.4f}")
    print(f"    Std: {text_emb.std():.4f}")
    print(f"    Min: {text_emb.min():.4f}")
    print(f"    Max: {text_emb.max():.4f}")

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  Qwen3-4B text std:    {text_emb.std():.4f}")
    print(f"  Qwen3-VL raw std:     {emb_raw.std():.4f}")
    print(f"  Qwen3-VL scaled std:  {emb.std():.4f}")
    print(f"  Scale factor applied: {result_scaled.scale_factor:.4f}")

    print(f"\n  Qwen3-4B text mean:   {text_emb.mean():.4f}")
    print(f"  Qwen3-VL raw mean:    {emb_raw.mean():.4f}")
    print(f"  Qwen3-VL scaled mean: {emb.mean():.4f}")

    # Check if mean is also significantly different
    mean_ratio = abs(emb.mean() / (text_emb.mean() + 1e-8))
    print(f"\n  Mean ratio (VL/text): {mean_ratio:.4f}")

    if abs(emb.mean()) > 1.0 and abs(text_emb.mean()) < 0.1:
        print("\n  WARNING: VL embeddings have significantly different mean!")
        print("  This could cause issues - consider mean-centering.")

    # Cleanup
    extractor.unload()
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    sys.exit(main())
