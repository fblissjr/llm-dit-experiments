#!/usr/bin/env python3
"""
last updated: 2025-12-19

Test using Bagel's trained connector weights for Z-Image Omni.

The hail mary: Bagel uses the same 1152-dim SigLIP encoder.
Their connector projects 1152 -> 3584.
Z-Image needs 1152 -> 3840.

We pad Bagel's weights to the target dimension.
"""

import sys
from pathlib import Path

# Add diffusers PR to path
DIFFUSERS_PR_PATH = Path(__file__).parent.parent.parent / "coderef/diffusers/src"
sys.path.insert(0, str(DIFFUSERS_PR_PATH))

import torch
from safetensors import safe_open
from PIL import Image


# Paths
BAGEL_PATH = Path.home() / "Storage/ByteDance-Seed_BAGEL-7B-MoT/ema.safetensors"
ZIMAGE_PATH = Path.home() / "Storage/Tongyi-MAI_Z-Image-Turbo"
SIGLIP_PATH = Path.home() / "Storage/google_siglip2-so400m-patch14-384"
QWEN3_PATH = Path.home() / "Storage/Qwen3-4B"


def load_bagel_connector():
    """Load Bagel's trained connector weights."""
    print("Loading Bagel connector weights...")

    with safe_open(str(BAGEL_PATH), framework='pt', device='cpu') as f:
        fc1_weight = f.get_tensor('connector.fc1.weight')  # (3584, 1152)
        fc1_bias = f.get_tensor('connector.fc1.bias')      # (3584,)
        fc2_weight = f.get_tensor('connector.fc2.weight')  # (3584, 3584)
        fc2_bias = f.get_tensor('connector.fc2.bias')      # (3584,)

    print(f"  fc1_weight: {fc1_weight.shape}")
    print(f"  fc1_bias: {fc1_bias.shape}")
    print(f"  fc2_weight: {fc2_weight.shape}")
    print(f"  fc2_bias: {fc2_bias.shape}")

    return {
        'fc1_weight': fc1_weight,
        'fc1_bias': fc1_bias,
        'fc2_weight': fc2_weight,
        'fc2_bias': fc2_bias,
    }


def adapt_to_zimage(bagel_weights, target_dim=3840):
    """
    Adapt Bagel's connector (1152->3584) to Z-Image's (1152->3840).

    Strategy: Pad the output dimension with learned extension.
    """
    fc1_weight = bagel_weights['fc1_weight']  # (3584, 1152)
    fc1_bias = bagel_weights['fc1_bias']      # (3584,)

    current_out = fc1_weight.shape[0]  # 3584
    pad_size = target_dim - current_out  # 256

    print(f"\nAdapting {current_out} -> {target_dim} (padding {pad_size} dims)")

    # Strategy 1: Zero padding
    pad_weight = torch.zeros(pad_size, fc1_weight.shape[1], dtype=fc1_weight.dtype)
    pad_bias = torch.zeros(pad_size, dtype=fc1_bias.dtype)

    # Strategy 2: Small random init for padded dims (to avoid dead neurons)
    # Use similar std to existing weights
    std = fc1_weight.float().std().item()
    pad_weight = torch.randn(pad_size, fc1_weight.shape[1], dtype=fc1_weight.dtype) * (std * 0.1)
    pad_bias = torch.zeros(pad_size, dtype=fc1_bias.dtype)

    # Concatenate
    adapted_weight = torch.cat([fc1_weight, pad_weight], dim=0)  # (3840, 1152)
    adapted_bias = torch.cat([fc1_bias, pad_bias], dim=0)        # (3840,)

    print(f"  Adapted weight: {adapted_weight.shape}")
    print(f"  Adapted bias: {adapted_bias.shape}")

    return adapted_weight, adapted_bias


def test_siglip_projection(adapted_weight, adapted_bias):
    """Test projecting SigLIP embeddings through the adapted connector."""
    print("\nTesting SigLIP projection...")

    # Load SigLIP
    from transformers import AutoModel, AutoProcessor

    siglip_full = AutoModel.from_pretrained(str(SIGLIP_PATH))
    siglip = siglip_full.vision_model
    processor = AutoProcessor.from_pretrained(str(SIGLIP_PATH))

    # Create test image
    test_image = Image.new("RGB", (384, 384), (100, 150, 200))

    # Get SigLIP embeddings
    inputs = processor(images=[test_image], return_tensors="pt")
    with torch.no_grad():
        outputs = siglip(pixel_values=inputs['pixel_values'])
        siglip_emb = outputs.last_hidden_state  # (1, num_patches, 1152)

    print(f"  SigLIP output: {siglip_emb.shape}")
    print(f"  SigLIP stats: mean={siglip_emb.mean():.4f}, std={siglip_emb.std():.4f}")

    # Project through adapted connector
    siglip_flat = siglip_emb.squeeze(0)  # (num_patches, 1152)

    # Apply projection: out = x @ W.T + b
    projected = torch.nn.functional.linear(
        siglip_flat.float(),
        adapted_weight.float(),
        adapted_bias.float()
    )

    print(f"  Projected shape: {projected.shape}")  # Should be (num_patches, 3840)
    print(f"  Projected stats: mean={projected.mean():.4f}, std={projected.std():.4f}")

    return projected


def compare_to_text_embeddings(projected_emb):
    """Compare projected SigLIP to Qwen3 text embeddings for compatibility check."""
    print("\nComparing to Qwen3 text embeddings...")

    from transformers import AutoModel, AutoTokenizer

    # Load Qwen3
    tokenizer = AutoTokenizer.from_pretrained(str(QWEN3_PATH), trust_remote_code=True)
    text_encoder = AutoModel.from_pretrained(str(QWEN3_PATH), trust_remote_code=True)

    # Encode a test prompt
    test_prompt = "A beautiful sunset over mountains"
    inputs = tokenizer(test_prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = text_encoder(**inputs, output_hidden_states=True)
        text_emb = outputs.hidden_states[-2]  # (1, seq_len, 2560)

    print(f"  Text embedding shape: {text_emb.shape}")
    print(f"  Text embedding stats: mean={text_emb.mean():.4f}, std={text_emb.std():.4f}")

    # Compare statistics
    proj_mean = projected_emb.mean().item()
    proj_std = projected_emb.std().item()
    text_mean = text_emb.mean().item()
    text_std = text_emb.std().item()

    print(f"\n  Comparison:")
    print(f"    Projected SigLIP: mean={proj_mean:.4f}, std={proj_std:.4f}")
    print(f"    Qwen3 text:       mean={text_mean:.4f}, std={text_std:.4f}")
    print(f"    Std ratio: {proj_std/text_std:.2f}x")

    return text_emb


def test_full_generation():
    """Test full Z-Image generation with Bagel's adapted connector."""
    print("\n" + "="*60)
    print("Testing full Z-Image generation with Bagel connector")
    print("="*60)

    # Load Bagel connector
    bagel_weights = load_bagel_connector()

    # Adapt to Z-Image dimensions
    adapted_weight, adapted_bias = adapt_to_zimage(bagel_weights, target_dim=3840)

    # Test projection
    projected = test_siglip_projection(adapted_weight, adapted_bias)

    # Compare to text embeddings
    compare_to_text_embeddings(projected)

    # Now try loading into Z-Image transformer
    print("\n" + "="*60)
    print("Loading Z-Image transformer with Bagel connector...")
    print("="*60)

    from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel

    # Load transformer with siglip support
    transformer = ZImageTransformer2DModel.from_pretrained(
        str(ZIMAGE_PATH),
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    # Check if siglip components exist, if not create them
    if transformer.siglip_embedder is None:
        print("Adding SigLIP components to transformer...")

        config = transformer.config
        new_config = dict(config)
        new_config["siglip_feat_dim"] = 1152

        new_transformer = ZImageTransformer2DModel(**new_config)

        # Copy existing weights
        missing, unexpected = new_transformer.load_state_dict(
            transformer.state_dict(), strict=False
        )
        print(f"  Missing keys (siglip components): {len(missing)}")

        transformer = new_transformer

    # NOW THE KEY PART: Load Bagel's adapted weights into siglip_embedder
    print("\nInjecting Bagel connector weights...")

    # siglip_embedder is nn.Sequential(RMSNorm, Linear)
    # We need to set the Linear layer's weights
    siglip_linear = transformer.siglip_embedder[1]  # The Linear layer

    print(f"  siglip_embedder Linear: {siglip_linear.weight.shape}")
    print(f"  Adapted weights: {adapted_weight.shape}")

    # Set the weights
    with torch.no_grad():
        siglip_linear.weight.copy_(adapted_weight.to(siglip_linear.weight.dtype))
        siglip_linear.bias.copy_(adapted_bias.to(siglip_linear.bias.dtype))

    print("  Weights injected!")

    # Verify
    print(f"  New weight stats: mean={siglip_linear.weight.float().mean():.4f}, std={siglip_linear.weight.float().std():.4f}")

    # Save the adapted weights for later use
    save_path = Path(__file__).parent / "bagel_adapted_siglip_embedder.pt"
    torch.save({
        'weight': adapted_weight,
        'bias': adapted_bias,
        'source': 'bagel_connector_fc1',
        'original_dim': 3584,
        'target_dim': 3840,
    }, save_path)
    print(f"\nSaved adapted weights to {save_path}")

    return transformer


def main():
    print("="*60)
    print("Bagel Connector -> Z-Image Omni Test")
    print("="*60)

    # Check paths
    if not BAGEL_PATH.exists():
        print(f"Error: Bagel weights not found at {BAGEL_PATH}")
        return

    # Run tests
    transformer = test_full_generation()

    print("\n" + "="*60)
    print("SUCCESS - Bagel connector weights loaded into Z-Image!")
    print("="*60)
    print("""
Next steps:
1. Run full generation with a reference image
2. Check if output is meaningful (not noise)
3. If promising, tune the siglip_refiner initialization
""")


if __name__ == "__main__":
    main()
