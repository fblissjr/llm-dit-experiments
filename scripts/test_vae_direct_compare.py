"""
Direct VAE comparison - load same latents, compare layer-by-layer outputs.

This script imports the reference VideoDecoder class directly and compares
intermediate values with our implementation using identical inputs.

Last Updated: 2026-01-20
"""

import sys
from pathlib import Path

import torch

# Add coderef to path
coderef_root = Path(__file__).parent.parent / "coderef" / "LTX-2" / "packages"
sys.path.insert(0, str(coderef_root / "ltx-core" / "src"))


def print_stats(name: str, tensor: torch.Tensor) -> None:
    """Print tensor statistics."""
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}, dtype: {tensor.dtype}")
    print(f"  Mean: {tensor.mean():.4f}, Std: {tensor.std():.4f}")
    print(f"  Range: [{tensor.min():.4f}, {tensor.max():.4f}]")


def load_reference_decoder(checkpoint_path: str, dtype: torch.dtype = torch.bfloat16):
    """Load reference decoder with manual config (matching checkpoint architecture)."""
    from ltx_core.model.video_vae.video_vae import VideoDecoder
    from ltx_core.model.video_vae.enums import NormLayerType, PaddingModeType
    from safetensors.torch import load_file

    # Architecture from checkpoint (not from config.json which is diffusers format)
    # Based on weight shapes: conv_in has [1024, 128, 3,3,3] so first block is 1024 channels
    decoder = VideoDecoder(
        convolution_dimensions=3,
        in_channels=128,
        out_channels=3,
        decoder_blocks=[
            # (channels, num_layers, upsample_stride, residual)
            (1024, 1, None, False),  # UNetMidBlock3D
            (512, 1, (2, 2, 2), True),  # DepthToSpaceUpsample with residual
            (512, 1, None, False),  # UNetMidBlock3D
            (256, 1, (2, 2, 2), True),  # DepthToSpaceUpsample with residual
            (256, 1, None, False),  # UNetMidBlock3D
            (128, 1, (2, 2, 2), True),  # DepthToSpaceUpsample with residual
            (128, 5, None, False),  # UNetMidBlock3D with 5 layers
        ],
        patch_size=4,
        norm_layer=NormLayerType.PIXEL_NORM,
        causal=False,
        timestep_conditioning=False,
        decoder_spatial_padding_mode=PaddingModeType.REFLECT,
    )

    # Load checkpoint weights
    state_dict = load_file(checkpoint_path)

    # Map keys
    mapped = {}
    for k, v in state_dict.items():
        if k == "latents_mean":
            mapped["per_channel_statistics.mean-of-means"] = v.to(dtype)
        elif k == "latents_std":
            mapped["per_channel_statistics.std-of-means"] = v.to(dtype)
        elif k.startswith("decoder."):
            mapped[k[8:]] = v.to(dtype)

    # Load with strict=False to see what's missing/unexpected
    result = decoder.load_state_dict(mapped, strict=False)
    print(f"Reference decoder load result:")
    print(f"  Missing keys: {len(result.missing_keys)}")
    print(f"  Unexpected keys: {len(result.unexpected_keys)}")
    if result.missing_keys:
        print(f"  First 5 missing: {result.missing_keys[:5]}")

    return decoder.to(dtype)


def load_our_decoder(vae_path: str, dtype: torch.dtype = torch.bfloat16):
    """Load our decoder."""
    from llm_dit.models.ltx2.vae import load_ltx2_vae_decoder
    return load_ltx2_vae_decoder(Path(vae_path), dtype=dtype, device="cpu")


def main():
    print("=" * 80)
    print("DIRECT VAE COMPARISON (same latents, both implementations)")
    print("=" * 80)

    device = "cuda"
    dtype = torch.bfloat16
    checkpoint_path = "models/LTX-2/vae/diffusion_pytorch_model.safetensors"
    vae_path = "models/LTX-2/vae"

    # Fixed random latents for reproducibility
    torch.manual_seed(42)
    latents = torch.randn(1, 128, 2, 8, 12, device="cpu", dtype=dtype)
    print_stats("\nTest latents", latents)

    # Load OUR decoder
    print("\n" + "=" * 40)
    print("LOADING OUR DECODER")
    print("=" * 40)
    our_decoder = load_our_decoder(vae_path, dtype).to(device)

    # Check architecture
    print(f"\nOur decoder up_blocks: {len(our_decoder.up_blocks)}")
    for i, block in enumerate(our_decoder.up_blocks):
        print(f"  [{i}] {type(block).__name__}")

    # Run our decoder
    print("\n--- OUR OUTPUT ---")
    with torch.no_grad():
        our_output = our_decoder(latents.clone().to(device))
    print_stats("Our output", our_output)

    # Clean up
    del our_decoder
    torch.cuda.empty_cache()

    # Load REFERENCE decoder
    print("\n" + "=" * 40)
    print("LOADING REFERENCE DECODER")
    print("=" * 40)
    try:
        ref_decoder = load_reference_decoder(checkpoint_path, dtype).to(device)

        print(f"\nReference decoder up_blocks: {len(ref_decoder.up_blocks)}")
        for i, block in enumerate(ref_decoder.up_blocks):
            print(f"  [{i}] {type(block).__name__}")

        # Run reference decoder
        print("\n--- REFERENCE OUTPUT ---")
        with torch.no_grad():
            ref_output = ref_decoder(latents.clone().to(device))
        print_stats("Reference output", ref_output)

        # Compare
        print("\n" + "=" * 40)
        print("COMPARISON")
        print("=" * 40)
        diff = (our_output - ref_output).abs()
        print_stats("Absolute difference", diff)
        print(f"Max diff: {diff.max():.6f}")
        print(f"Mean diff: {diff.mean():.6f}")

        if diff.max() < 0.01:
            print("\n✓ Outputs match within tolerance!")
        else:
            print("\n✗ Outputs differ significantly - investigating...")

    except Exception as e:
        print(f"Error with reference decoder: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
