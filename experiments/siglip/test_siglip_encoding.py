#!/usr/bin/env python3
"""
last updated: 2025-12-19

Test SigLIP2 encoding to understand the embedding space.
Prepares us for Z-Image Omni integration when weights are released.

Usage:
    uv run python experiments/siglip/test_siglip_encoding.py
"""

import torch
from pathlib import Path
from PIL import Image
import numpy as np

# SigLIP2 model path
SIGLIP_PATH = Path.home() / "Storage/google_siglip2-base-patch16-224"


def load_siglip():
    """Load SigLIP2 vision model."""
    from transformers import AutoModel, AutoProcessor

    print(f"Loading SigLIP2 from {SIGLIP_PATH}")
    model = AutoModel.from_pretrained(SIGLIP_PATH, device_map="cpu")
    processor = AutoProcessor.from_pretrained(SIGLIP_PATH)

    print(f"Vision config: {model.config.vision_config}")
    print(f"  Hidden size: {model.config.vision_config.hidden_size}")
    print(f"  Patch size: {model.config.vision_config.patch_size}")
    print(f"  Image size: {model.config.vision_config.image_size}")
    print(f"  Num layers: {model.config.vision_config.num_hidden_layers}")

    return model, processor


def extract_vision_features(model, processor, image: Image.Image):
    """Extract vision features from an image."""
    # Process image
    inputs = processor(images=[image], return_tensors="pt")

    print(f"\nProcessor output keys: {inputs.keys()}")
    print(f"  pixel_values shape: {inputs['pixel_values'].shape}")

    # Get vision features
    with torch.no_grad():
        # Use vision model directly
        vision_outputs = model.vision_model(
            pixel_values=inputs['pixel_values'],
            output_hidden_states=True,
        )

    # Last hidden state
    last_hidden = vision_outputs.last_hidden_state
    print(f"\nVision output:")
    print(f"  last_hidden_state shape: {last_hidden.shape}")
    print(f"  (batch, num_patches, hidden_dim)")

    # For 224x224 with 16x16 patches: 14x14 = 196 patches (no CLS token in SigLIP)

    # All hidden states (may be None if not returned)
    all_hidden = vision_outputs.hidden_states
    if all_hidden is not None:
        print(f"\nAll hidden states: {len(all_hidden)} layers")
        for i, h in enumerate(all_hidden):
            print(f"  Layer {i}: {h.shape}")
    else:
        print("\nHidden states not returned (need output_hidden_states=True)")
        all_hidden = [last_hidden]  # Just use last hidden state

    return last_hidden, all_hidden


def analyze_embeddings(embeddings: torch.Tensor, name: str = "embeddings"):
    """Analyze embedding statistics."""
    print(f"\n=== {name} Statistics ===")

    # Flatten to (num_tokens, hidden_dim)
    if embeddings.dim() == 3:
        embeddings = embeddings.squeeze(0)

    print(f"Shape: {embeddings.shape}")

    # Global stats
    print(f"Global mean: {embeddings.mean().item():.4f}")
    print(f"Global std: {embeddings.std().item():.4f}")
    print(f"Global min: {embeddings.min().item():.4f}")
    print(f"Global max: {embeddings.max().item():.4f}")

    # Per-dimension stats
    dim_means = embeddings.mean(dim=0)
    dim_stds = embeddings.std(dim=0)

    print(f"\nPer-dimension stats:")
    print(f"  Mean of means: {dim_means.mean().item():.4f}")
    print(f"  Std of means: {dim_means.std().item():.4f}")
    print(f"  Mean of stds: {dim_stds.mean().item():.4f}")
    print(f"  Std of stds: {dim_stds.std().item():.4f}")

    # Find outlier dimensions (high std ratio)
    mean_std = dim_stds.mean()
    std_ratios = dim_stds / mean_std

    outliers = []
    for dim_idx in range(len(std_ratios)):
        ratio = std_ratios[dim_idx].item()
        if ratio > 3.0:  # More than 3x average std
            outliers.append((dim_idx, ratio))

    if outliers:
        outliers.sort(key=lambda x: x[1], reverse=True)
        print(f"\nOutlier dimensions (>3x avg std):")
        for dim_idx, ratio in outliers[:10]:
            print(f"  Dim {dim_idx}: {ratio:.2f}x")
    else:
        print(f"\nNo outlier dimensions found (all within 3x avg std)")

    return {
        "mean": embeddings.mean().item(),
        "std": embeddings.std().item(),
        "dim_stds": dim_stds,
        "outliers": outliers,
    }


def compare_to_qwen3(siglip_stats: dict):
    """Compare SigLIP stats to Qwen3-4B reference (different embedding spaces)."""
    print("\n=== Comparison to Qwen3-4B (for reference only) ===")

    # Qwen3-4B text embedding stats (from our experiments)
    qwen3_std = 70.0  # Approximate std of Qwen3-4B text embeddings (2560 dim)

    print(f"Qwen3-4B text embedding std: ~{qwen3_std} (2560 dim)")
    print(f"SigLIP2 vision embedding std: {siglip_stats['std']:.2f} (768 dim)")
    print(f"Ratio (Qwen3/SigLIP): {qwen3_std / siglip_stats['std']:.2f}x")

    # Note: These are DIFFERENT embedding spaces
    # SigLIP goes through siglip_embedder (768->3840) with trained weights
    # Qwen3-4B goes through cap_embedder (2560->3840) with trained weights
    # The projection layers handle the normalization


def test_different_images():
    """Test with various image types."""
    model, processor = load_siglip()

    # Create test images
    test_images = {
        "white": Image.new("RGB", (512, 512), (255, 255, 255)),
        "black": Image.new("RGB", (512, 512), (0, 0, 0)),
        "red": Image.new("RGB", (512, 512), (255, 0, 0)),
        "gradient": None,  # Will create below
    }

    # Create gradient image
    gradient = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in range(512):
        gradient[i, :, 0] = int(255 * i / 511)  # Red gradient
        gradient[:, i, 1] = int(255 * i / 511)  # Green gradient
    test_images["gradient"] = Image.fromarray(gradient)

    print("\n" + "=" * 60)
    print("Testing different image types")
    print("=" * 60)

    all_stats = {}
    for name, img in test_images.items():
        print(f"\n--- {name.upper()} IMAGE ---")
        last_hidden, _ = extract_vision_features(model, processor, img)
        stats = analyze_embeddings(last_hidden, f"SigLIP2 {name}")
        all_stats[name] = stats

    # Compare stats across images
    print("\n=== Cross-Image Comparison ===")
    print(f"{'Image':<12} {'Mean':>10} {'Std':>10} {'Outliers':>10}")
    print("-" * 45)
    for name, stats in all_stats.items():
        print(f"{name:<12} {stats['mean']:>10.4f} {stats['std']:>10.4f} {len(stats['outliers']):>10}")

    return all_stats


def test_real_image(image_path: str = None):
    """Test with a real image if provided."""
    if image_path is None:
        # Try to find a test image
        test_paths = [
            Path("experiments/inputs"),
            Path("experiments/qwen3_vl/inputs"),
        ]
        for p in test_paths:
            if p.exists():
                images = list(p.glob("*.png")) + list(p.glob("*.jpg"))
                if images:
                    image_path = str(images[0])
                    break

    if image_path is None:
        print("\nNo real image found for testing")
        return None

    print(f"\n=== Testing Real Image: {image_path} ===")

    model, processor = load_siglip()
    img = Image.open(image_path).convert("RGB")
    print(f"Image size: {img.size}")

    last_hidden, all_hidden = extract_vision_features(model, processor, img)
    stats = analyze_embeddings(last_hidden, "SigLIP2 real image")
    compare_to_qwen3(stats)

    # Also check different layers
    print("\n=== Per-Layer Statistics ===")
    for i, hidden in enumerate(all_hidden):
        layer_stats = analyze_embeddings(hidden, f"Layer {i}")

    return stats


def main():
    print("=" * 60)
    print("SigLIP2 Embedding Analysis")
    print("Preparing for Z-Image Omni integration")
    print("=" * 60)

    # Test with synthetic images
    all_stats = test_different_images()

    # Test with real image
    real_stats = test_real_image()

    print("\n" + "=" * 60)
    print("KEY FINDINGS FOR OMNI INTEGRATION")
    print("=" * 60)

    avg_std = np.mean([s["std"] for s in all_stats.values()])
    print(f"""
SigLIP2-base-patch16-224:
  - Hidden dim: 768
  - Patches: 14x14 = 196 tokens (for 224x224 input)
  - Average embedding std: {avg_std:.2f}

Z-Image Omni architecture (from diffusers PR 12857):
  - Uses SAME Z-Image Turbo DiT (transformer unchanged)
  - Adds: siglip_embedder: RMSNorm + Linear(768 -> 3840)
  - Adds: siglip_refiner: 2-layer transformer blocks
  - Adds: siglip_pad_token: learnable parameter
  - Sequence: [caption_tokens] + [latent_tokens] + [siglip_tokens]

Status:
  - Omni model weights not yet released by Alibaba
  - siglip_feat_dim config already in transformer (set to None currently)
  - When released, we can load directly or adapt our code

Note: Our Qwen3-VL experiments were speculation - Z-Image team used SigLIP2.
""")


if __name__ == "__main__":
    main()
