"""
Compare VAE decode output between our implementation and reference.

This script:
1. Generates latents using our pipeline (or loads pre-saved latents)
2. Decodes using BOTH our VAE and the reference VAE
3. Compares outputs to identify differences

Last Updated: 2026-01-20
"""

import sys
from pathlib import Path

import torch

# Add coderef to path for reference VAE
coderef_root = Path(__file__).parent.parent / "coderef" / "LTX-2" / "packages"
sys.path.insert(0, str(coderef_root / "ltx-core" / "src"))


def print_stats(name: str, tensor: torch.Tensor) -> None:
    """Print tensor statistics."""
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}, dtype: {tensor.dtype}")
    print(f"  Mean: {tensor.mean():.4f}, Std: {tensor.std():.4f}")
    print(f"  Range: [{tensor.min():.4f}, {tensor.max():.4f}]")


def load_reference_vae(model_path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Load reference VideoDecoder from coderef."""
    import json
    from ltx_core.model.video_vae.model_configurator import VideoDecoderConfigurator
    from safetensors.torch import load_file

    vae_path = Path(model_path) / "vae"

    # Load config JSON
    with open(vae_path / "config.json", "r") as f:
        config = json.load(f)

    # Build decoder using reference configurator
    decoder = VideoDecoderConfigurator.from_config(config)

    # Load weights
    state_dict = load_file(vae_path / "diffusion_pytorch_model.safetensors")

    # Map keys (checkpoint uses decoder. prefix, reference wants direct keys)
    mapped_state_dict = {}
    for k, v in state_dict.items():
        # Handle latents_mean/std -> per_channel_statistics
        if k == "latents_mean":
            mapped_state_dict["per_channel_statistics.mean-of-means"] = v.to(dtype)
            continue
        if k == "latents_std":
            mapped_state_dict["per_channel_statistics.std-of-means"] = v.to(dtype)
            continue
        if k.startswith("encoder."):
            continue
        if k.startswith("decoder."):
            # Remove decoder. prefix
            new_k = k[8:]  # len("decoder.") = 8
            mapped_state_dict[new_k] = v.to(dtype)

    decoder.load_state_dict(mapped_state_dict, strict=False)
    return decoder.to(device).to(dtype)


def load_our_vae(model_path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Load our VideoDecoder."""
    from llm_dit.models.ltx2.vae import load_ltx2_vae_decoder
    vae = load_ltx2_vae_decoder(Path(model_path) / "vae", dtype=dtype, device="cpu")
    return vae.to(device)


def main():
    print("=" * 80)
    print("VAE DECODE COMPARISON TEST")
    print("=" * 80)

    device = "cuda"
    dtype = torch.bfloat16
    model_path = "models/LTX-2"

    # Generate test latents (or use fixed random for reproducibility)
    print("\n--- Generating test latents ---")
    torch.manual_seed(42)
    # Shape: [B, C, T_lat, H_lat, W_lat] = [1, 128, 2, 8, 12] for 9 frames @ 256x384
    latents = torch.randn(1, 128, 2, 8, 12, device=device, dtype=dtype)
    print_stats("Test latents", latents)

    # Test 1: Load and run OUR VAE
    print("\n" + "=" * 40)
    print("OUR VAE DECODER")
    print("=" * 40)
    our_vae = load_our_vae(model_path, device, dtype)

    # Check buffers
    our_std = our_vae.per_channel_statistics.get_buffer("std-of-means")
    our_mean = our_vae.per_channel_statistics.get_buffer("mean-of-means")
    print(f"\nOur PerChannelStatistics:")
    print(f"  std-of-means: mean={our_std.mean():.4f}, range=[{our_std.min():.4f}, {our_std.max():.4f}]")
    print(f"  mean-of-means: mean={our_mean.mean():.4f}, range=[{our_mean.min():.4f}, {our_mean.max():.4f}]")

    with torch.no_grad():
        our_output = our_vae(latents.clone())
    print_stats("\nOur VAE output", our_output)

    # Clean up our VAE
    del our_vae
    torch.cuda.empty_cache()

    # Test 2: Load and run REFERENCE VAE
    print("\n" + "=" * 40)
    print("REFERENCE VAE DECODER")
    print("=" * 40)

    try:
        ref_vae = load_reference_vae(model_path, device, dtype)

        # Check buffers
        ref_std = ref_vae.per_channel_statistics.get_buffer("std-of-means")
        ref_mean = ref_vae.per_channel_statistics.get_buffer("mean-of-means")
        print(f"\nReference PerChannelStatistics:")
        print(f"  std-of-means: mean={ref_std.mean():.4f}, range=[{ref_std.min():.4f}, {ref_std.max():.4f}]")
        print(f"  mean-of-means: mean={ref_mean.mean():.4f}, range=[{ref_mean.min():.4f}, {ref_mean.max():.4f}]")

        with torch.no_grad():
            ref_output = ref_vae(latents.clone())
        print_stats("\nReference VAE output", ref_output)

        # Compare outputs
        print("\n" + "=" * 40)
        print("COMPARISON")
        print("=" * 40)
        diff = (our_output - ref_output).abs()
        print_stats("Absolute difference", diff)
        print(f"Max diff: {diff.max():.6f}")
        print(f"Mean diff: {diff.mean():.6f}")

        # Check if buffers match
        std_diff = (our_std - ref_std).abs().max()
        mean_diff = (our_mean - ref_mean).abs().max()
        print(f"\nBuffer differences:")
        print(f"  std-of-means max diff: {std_diff:.6f}")
        print(f"  mean-of-means max diff: {mean_diff:.6f}")

    except Exception as e:
        print(f"Error loading reference VAE: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
