#!/usr/bin/env python3
"""
Compare embedding statistics between Qwen3-4B text and Qwen3-VL.

This diagnoses why VL embeddings produce noisy/corrupted images.
We compare: mean, std, min, max, and distribution shape.

Runs sequentially to avoid OOM - loads one model at a time.
"""

import sys
from pathlib import Path
import gc

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import torch
from PIL import Image


def get_text_embedding_stats(prompt: str) -> dict:
    """Get Qwen3-4B text embedding statistics."""
    from llm_dit.cli import load_runtime_config
    from llm_dit.startup import PipelineLoader
    from llm_dit.conversation import Qwen3Formatter, Conversation, Message, Role

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

    # Load encoder
    print("Loading Qwen3-4B text encoder...")
    loader = PipelineLoader(z_config)
    text_result = loader.load_encoder()
    encoder = text_result.encoder

    # Format using Conversation (proper API)
    formatter = Qwen3Formatter()
    conv = Conversation(
        messages=[Message(role=Role.USER, content=prompt)],
        enable_thinking=True,  # Add empty think block
        is_final=True,
    )
    formatted = formatter.format(conv)
    print(f"Formatted prompt: {repr(formatted[:100])}...")

    encode_result = encoder.encode(formatted)
    text_emb = encode_result.embeddings[0]  # Remove batch dim

    stats = {
        "shape": list(text_emb.shape),
        "dtype": str(text_emb.dtype),
        "mean": text_emb.mean().item(),
        "std": text_emb.std().item(),
        "min": text_emb.min().item(),
        "max": text_emb.max().item(),
        "per_dim_mean_range": [text_emb.mean(dim=0).min().item(), text_emb.mean(dim=0).max().item()],
        "per_dim_std_range": [text_emb.std(dim=0).min().item(), text_emb.std(dim=0).max().item()],
    }

    # Cleanup
    del encoder
    del text_result
    del loader
    gc.collect()
    torch.cuda.empty_cache()

    return stats


def get_vl_embedding_stats(prompt: str) -> dict:
    """Get Qwen3-VL embedding statistics."""
    from llm_dit.vl import VLEmbeddingExtractor

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
        raise FileNotFoundError("Could not find Qwen3-VL model")

    # Load test image
    image_path = Path(__file__).parent.parent.parent / "inputs" / "test_scene.png"
    if not image_path.exists():
        image = Image.new("RGB", (512, 512), color="red")
    else:
        image = Image.open(image_path).convert("RGB")

    print(f"Loading VL extractor from {vl_model_path}...")
    extractor = VLEmbeddingExtractor.from_pretrained(
        vl_model_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
    )

    # Extract WITHOUT scaling
    print("Extracting VL embeddings (NO scaling)...")
    vl_result_raw = extractor.extract(
        image=image,
        text=prompt,
        hidden_layer=-2,
        scale_to_text=False,
    )
    vl_emb_raw = vl_result_raw.embeddings

    # Extract WITH scaling
    print("Extracting VL embeddings (WITH scaling)...")
    vl_result_scaled = extractor.extract(
        image=image,
        text=prompt,
        hidden_layer=-2,
        scale_to_text=True,
    )
    vl_emb_scaled = vl_result_scaled.embeddings

    stats = {
        "raw": {
            "shape": list(vl_emb_raw.shape),
            "dtype": str(vl_emb_raw.dtype),
            "mean": vl_emb_raw.mean().item(),
            "std": vl_emb_raw.std().item(),
            "min": vl_emb_raw.min().item(),
            "max": vl_emb_raw.max().item(),
            "per_dim_mean_range": [vl_emb_raw.mean(dim=0).min().item(), vl_emb_raw.mean(dim=0).max().item()],
            "per_dim_std_range": [vl_emb_raw.std(dim=0).min().item(), vl_emb_raw.std(dim=0).max().item()],
        },
        "scaled": {
            "shape": list(vl_emb_scaled.shape),
            "mean": vl_emb_scaled.mean().item(),
            "std": vl_emb_scaled.std().item(),
            "min": vl_emb_scaled.min().item(),
            "max": vl_emb_scaled.max().item(),
            "scale_factor": vl_result_scaled.scale_factor,
        },
        "full_prompt": vl_result_raw.full_prompt_with_tokens[:200] + "...",
    }

    # Cleanup
    extractor.unload()
    gc.collect()
    torch.cuda.empty_cache()

    return stats, vl_emb_raw


def main():
    prompt = "Homer Simpson eating spaghetti"

    print("=" * 70)
    print("STEP 1: Get Qwen3-4B text embeddings")
    print("=" * 70)
    text_stats = get_text_embedding_stats(prompt)

    print(f"\nQwen3-4B Text Embeddings:")
    print(f"  Shape: {text_stats['shape']}")
    print(f"  Mean:  {text_stats['mean']:.6f}")
    print(f"  Std:   {text_stats['std']:.6f}")
    print(f"  Min:   {text_stats['min']:.6f}")
    print(f"  Max:   {text_stats['max']:.6f}")

    print("\n" + "=" * 70)
    print("STEP 2: Get Qwen3-VL embeddings")
    print("=" * 70)
    vl_stats, vl_emb_raw = get_vl_embedding_stats(prompt)

    print(f"\nQwen3-VL Raw Embeddings (no scaling):")
    print(f"  Shape: {vl_stats['raw']['shape']}")
    print(f"  Mean:  {vl_stats['raw']['mean']:.6f}")
    print(f"  Std:   {vl_stats['raw']['std']:.6f}")
    print(f"  Min:   {vl_stats['raw']['min']:.6f}")
    print(f"  Max:   {vl_stats['raw']['max']:.6f}")

    print(f"\nQwen3-VL Scaled Embeddings:")
    print(f"  Mean:  {vl_stats['scaled']['mean']:.6f}")
    print(f"  Std:   {vl_stats['scaled']['std']:.6f}")
    print(f"  Min:   {vl_stats['scaled']['min']:.6f}")
    print(f"  Max:   {vl_stats['scaled']['max']:.6f}")
    print(f"  Scale factor: {vl_stats['scaled']['scale_factor']:.4f}")

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    text_mean = text_stats['mean']
    text_std = text_stats['std']
    vl_raw_mean = vl_stats['raw']['mean']
    vl_raw_std = vl_stats['raw']['std']
    vl_scaled_mean = vl_stats['scaled']['mean']
    vl_scaled_std = vl_stats['scaled']['std']

    print(f"\n                    Qwen3-4B    VL Raw      VL Scaled")
    print("-" * 60)
    print(f"Mean:               {text_mean:10.4f}  {vl_raw_mean:10.4f}  {vl_scaled_mean:10.4f}")
    print(f"Std:                {text_std:10.4f}  {vl_raw_std:10.4f}  {vl_scaled_std:10.4f}")
    print(f"Min:                {text_stats['min']:10.4f}  {vl_stats['raw']['min']:10.4f}  {vl_stats['scaled']['min']:10.4f}")
    print(f"Max:                {text_stats['max']:10.4f}  {vl_stats['raw']['max']:10.4f}  {vl_stats['scaled']['max']:10.4f}")

    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    mean_diff = abs(text_mean - vl_scaled_mean)
    std_diff = abs(text_std - vl_scaled_std)

    print(f"\nAfter scaling:")
    print(f"  Mean difference: {mean_diff:.4f}")
    print(f"  Std difference:  {std_diff:.4f}")

    if mean_diff > 0.5:
        print(f"\n*** ISSUE: Mean mismatch! ***")
        print(f"  Current scaling only adjusts STD, not MEAN.")
        print(f"  Text mean:      {text_mean:.4f}")
        print(f"  VL scaled mean: {vl_scaled_mean:.4f}")

        # Show what proper normalization would produce
        print(f"\n  RECOMMENDED FIX: Normalize mean AND std")
        vl_normalized = (vl_emb_raw - vl_emb_raw.mean()) / vl_emb_raw.std() * text_std + text_mean
        print(f"  After proper normalization:")
        print(f"    Mean: {vl_normalized.mean().item():.4f} (target: {text_mean:.4f})")
        print(f"    Std:  {vl_normalized.std().item():.4f} (target: {text_std:.4f})")

    if std_diff > 1.0:
        print(f"\n*** ISSUE: Std mismatch after scaling! ***")
        print(f"  This should not happen - check scaling logic.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
