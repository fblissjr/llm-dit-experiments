#!/usr/bin/env python3
"""
last updated: 2025-12-19

Test Z-Image Omni pipeline using diffusers PR 12857 code.

This script tests the omni mode by:
1. Loading Z-Image Turbo transformer
2. Loading SigLIP2 encoder
3. Running generation with image conditioning

Note: siglip_embedder/refiner are randomly initialized (not trained).
Results will be garbage but validate the architecture works.

Usage:
    uv run python experiments/siglip/test_omni_pipeline.py
"""

import sys
from pathlib import Path

# Add diffusers PR to path
DIFFUSERS_PR_PATH = Path(__file__).parent.parent.parent / "coderef/diffusers/src"
sys.path.insert(0, str(DIFFUSERS_PR_PATH))

import torch
from PIL import Image


def test_siglip_encoder():
    """Test that SigLIP encoder works."""
    print("\n=== Testing SigLIP Encoder ===")

    from siglip_encoder import SigLIPEncoder

    encoder = SigLIPEncoder(device="cpu")

    # Test with sample image
    img = Image.new("RGB", (512, 512), (128, 64, 192))
    embeddings = encoder.encode(img)

    print(f"Embedding shape: {embeddings[0].shape}")
    print(f"Hidden size: {encoder.hidden_size}")

    encoder.unload()
    return encoder.hidden_size


def test_transformer_with_siglip(siglip_feat_dim: int):
    """Test loading transformer with siglip support."""
    print("\n=== Testing Transformer with SigLIP ===")

    from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel

    # Load with siglip_feat_dim set
    print(f"Creating transformer with siglip_feat_dim={siglip_feat_dim}")

    # Create a minimal transformer config
    transformer = ZImageTransformer2DModel(
        in_channels=16,
        dim=3840,
        n_layers=2,  # Minimal for testing
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        cap_feat_dim=2560,
        siglip_feat_dim=siglip_feat_dim,  # Enable SigLIP
    )

    print(f"Created transformer")
    print(f"  siglip_embedder: {transformer.siglip_embedder}")
    print(f"  siglip_refiner: {transformer.siglip_refiner}")
    print(f"  siglip_pad_token: {transformer.siglip_pad_token.shape if transformer.siglip_pad_token is not None else None}")

    return transformer


def test_omni_forward(transformer, siglip_feat_dim: int):
    """Test omni mode forward pass."""
    print("\n=== Testing Omni Forward ===")

    device = "cpu"
    dtype = torch.float32
    bsz = 1

    # Expected format from diffusers PR:
    # x_item: (C, F, H, W) where C=channels, F=frames, H=height, W=width
    # For latents: C=16, F=1, H=W=64 (for 1024x1024 image with 8x VAE scale and 2x patch)

    # Target latent: (C, F, H, W) = (16, 1, 64, 64)
    target_latent = torch.randn(16, 1, 64, 64, device=device, dtype=dtype)

    # Condition latent: same shape
    cond_latent = torch.randn(16, 1, 64, 64, device=device, dtype=dtype)

    # Text embeddings: list of list of tensors
    # Each batch has a list of text segments (for vision token separators)
    # Each segment is (seq_len, 2560)
    cap_feats = [
        [
            torch.randn(50, 2560, device=device, dtype=dtype),  # Before vision
            torch.randn(10, 2560, device=device, dtype=dtype),  # After vision
        ]
    ]

    # SigLIP embeddings: list of list of tensors (H, W, hidden_size)
    # One per condition image, plus None for target
    siglip_feats = [
        [
            torch.randn(14, 14, siglip_feat_dim, device=device, dtype=dtype),  # Cond image
            None,  # Target image placeholder
        ]
    ]

    # Condition latents: list of list of tensors (C, F, H, W)
    cond_latents = [
        [cond_latent]  # One condition image latent
    ]

    # Timestep
    t = torch.tensor([0.5], device=device, dtype=dtype)

    # Target latent as list (one per batch)
    x = [target_latent]

    print(f"Inputs:")
    print(f"  x: {[xi.shape for xi in x]}")
    print(f"  t: {t.shape}")
    print(f"  cap_feats: {[[cf.shape for cf in batch] for batch in cap_feats]}")
    print(f"  cond_latents: {[[cl.shape for cl in batch] for batch in cond_latents]}")
    print(f"  siglip_feats: {[[sf.shape if sf is not None else None for sf in batch] for batch in siglip_feats]}")

    # Forward pass
    try:
        output = transformer(
            x=x,
            t=t,
            cap_feats=cap_feats,
            cond_latents=cond_latents,
            siglip_feats=siglip_feats,
            return_dict=False,
        )
        print(f"\nOutput: {type(output)}")
        if output and output[0]:
            print(f"  Shape: {output[0][0].shape}")
        print("\nOMNI FORWARD PASS SUCCEEDED!")
        return True
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test full omni pipeline if model is available."""
    print("\n=== Testing Full Pipeline ===")

    # Check if model path exists
    model_paths = [
        Path.home() / "Storage/Z-Image-Turbo",
        Path.home() / "Storage/z-image-turbo",
        Path.home() / "Storage/Tongyi-MAI_Z-Image-Turbo",
    ]

    model_path = None
    for p in model_paths:
        if p.exists():
            model_path = p
            break

    if model_path is None:
        print("No Z-Image model found. Skipping full pipeline test.")
        print("Checked paths:")
        for p in model_paths:
            print(f"  {p}")
        return False

    print(f"Found model at: {model_path}")

    try:
        from diffusers.pipelines.z_image.pipeline_z_image_omni import ZImageOmniPipeline

        # This would require all components (VAE, text encoder, etc.)
        # For now just verify imports work
        print("ZImageOmniPipeline imported successfully")
        return True
    except Exception as e:
        print(f"Could not import pipeline: {e}")
        return False


def main():
    print("=" * 60)
    print("Z-Image Omni Pipeline Test")
    print("Testing diffusers PR 12857 integration")
    print("=" * 60)

    # Test 1: SigLIP encoder
    try:
        siglip_feat_dim = test_siglip_encoder()
    except Exception as e:
        print(f"SigLIP encoder test failed: {e}")
        siglip_feat_dim = 768  # Fallback

    # Test 2: Transformer with SigLIP components
    try:
        transformer = test_transformer_with_siglip(siglip_feat_dim)
    except Exception as e:
        print(f"Transformer test failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 3: Omni forward pass
    try:
        success = test_omni_forward(transformer, siglip_feat_dim)
    except Exception as e:
        print(f"Omni forward test failed: {e}")
        import traceback
        traceback.print_exc()
        success = False

    # Test 4: Full pipeline (optional)
    test_full_pipeline()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"SigLIP hidden dim: {siglip_feat_dim}")
    print(f"Omni forward pass: {'PASSED' if success else 'FAILED'}")
    print("""
Next steps:
1. Get Z-Image Turbo model loaded
2. Initialize siglip_embedder/refiner (random or from weights if available)
3. Run full generation with image conditioning
""")


if __name__ == "__main__":
    main()
