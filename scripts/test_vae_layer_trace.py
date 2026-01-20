"""
Layer-by-layer trace of our VAE forward pass vs reference operations.

This script compares individual operations (PixelNorm, un_normalize, conv)
between our implementation and the reference to find divergence.

Last Updated: 2026-01-20
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

# Add coderef to path
coderef_root = Path(__file__).parent.parent / "coderef" / "LTX-2" / "packages"
sys.path.insert(0, str(coderef_root / "ltx-core" / "src"))


def print_stats(name: str, tensor: torch.Tensor) -> None:
    """Print tensor statistics."""
    print(f"  {name}: mean={tensor.mean():.4f}, std={tensor.std():.4f}, "
          f"range=[{tensor.min():.4f}, {tensor.max():.4f}]")


def run_reference_operations_comparison():
    """Compare reference VAE operations."""
    from ltx_core.model.video_vae.ops import PerChannelStatistics
    from ltx_core.model.video_vae.normalization import PixelNorm

    print("\n=== REFERENCE OPERATIONS TEST ===")

    # Test PixelNorm
    print("\n--- PixelNorm Comparison ---")
    from llm_dit.models.ltx2.vae.normalization import PixelNorm as OurPixelNorm

    torch.manual_seed(42)
    test_input = torch.randn(1, 128, 2, 8, 12, device="cuda", dtype=torch.bfloat16)

    ref_pn = PixelNorm().cuda()
    our_pn = OurPixelNorm().cuda()

    ref_out = ref_pn(test_input.clone())
    our_out = our_pn(test_input.clone())

    print_stats("Input", test_input)
    print_stats("Reference PixelNorm", ref_out)
    print_stats("Our PixelNorm", our_out)
    print(f"PixelNorm diff: {(ref_out - our_out).abs().max():.8f}")

    # Test PerChannelStatistics
    print("\n--- PerChannelStatistics Comparison ---")
    from llm_dit.models.ltx2.vae.ops import PerChannelStatistics as OurPCS

    ckpt = load_file("models/LTX-2/vae/diffusion_pytorch_model.safetensors")

    ref_pcs = PerChannelStatistics(latent_channels=128).cuda()
    ref_pcs.register_buffer("mean-of-means", ckpt["latents_mean"].cuda().to(torch.bfloat16))
    ref_pcs.register_buffer("std-of-means", ckpt["latents_std"].cuda().to(torch.bfloat16))

    our_pcs = OurPCS(latent_channels=128).cuda()
    our_pcs.register_buffer("mean-of-means", ckpt["latents_mean"].cuda().to(torch.bfloat16))
    our_pcs.register_buffer("std-of-means", ckpt["latents_std"].cuda().to(torch.bfloat16))

    torch.manual_seed(42)
    test_input = torch.randn(1, 128, 2, 8, 12, device="cuda", dtype=torch.bfloat16)

    ref_unnorm = ref_pcs.un_normalize(test_input.clone())
    our_unnorm = our_pcs.un_normalize(test_input.clone())

    print_stats("Input", test_input)
    print_stats("Reference un_normalize", ref_unnorm)
    print_stats("Our un_normalize", our_unnorm)
    print(f"un_normalize diff: {(ref_unnorm - our_unnorm).abs().max():.8f}")

    return ref_unnorm, our_unnorm


def compare_conv_operations():
    """Compare convolution operations."""
    from ltx_core.model.video_vae.convolution import make_conv_nd as ref_make_conv_nd
    from llm_dit.models.ltx2.vae.convolution import make_conv_nd as our_make_conv_nd
    from ltx_core.model.video_vae.enums import PaddingModeType as RefPaddingMode
    from llm_dit.models.ltx2.vae.enums import PaddingModeType as OurPaddingMode

    print("\n=== CONVOLUTION COMPARISON ===")

    # Create identical convs with same seed
    torch.manual_seed(123)
    ref_conv = ref_make_conv_nd(
        dims=3, in_channels=128, out_channels=1024,
        kernel_size=3, stride=1, padding=1, causal=True,
        spatial_padding_mode=RefPaddingMode.REFLECT
    ).cuda().to(torch.bfloat16)

    torch.manual_seed(123)
    our_conv = our_make_conv_nd(
        dims=3, in_channels=128, out_channels=1024,
        kernel_size=3, stride=1, padding=1, causal=True,
        spatial_padding_mode=OurPaddingMode.REFLECT
    ).cuda().to(torch.bfloat16)

    # Compare weights
    print(f"Weight diff: {(ref_conv.conv.weight - our_conv.conv.weight).abs().max():.8f}")
    print(f"Bias diff: {(ref_conv.conv.bias - our_conv.conv.bias).abs().max():.8f}")

    # Test forward
    torch.manual_seed(42)
    test_input = torch.randn(1, 128, 2, 8, 12, device="cuda", dtype=torch.bfloat16)

    ref_out = ref_conv(test_input.clone(), causal=True)
    our_out = our_conv(test_input.clone(), causal=True)

    print_stats("Input", test_input)
    print_stats("Reference conv", ref_out)
    print_stats("Our conv", our_out)
    print(f"Conv output diff: {(ref_out - our_out).abs().max():.8f}")


def run_our_vae_with_trace():
    """Run our VAE and trace output."""
    from llm_dit.models.ltx2.vae import load_ltx2_vae_decoder

    print("\n=== OUR VAE FORWARD TRACE ===")

    # Load decoder
    decoder = load_ltx2_vae_decoder("models/LTX-2/vae", dtype=torch.bfloat16, device="cpu")
    decoder = decoder.cuda()

    # Generate test latents
    torch.manual_seed(42)
    latents = torch.randn(1, 128, 2, 8, 12, device="cuda", dtype=torch.bfloat16)
    print("\nTest latents:")
    print_stats("Input", latents)

    with torch.no_grad():
        output = decoder(latents)

    print("\nOutput:")
    print_stats("Final", output)

    return output


def main():
    # First run basic operation comparisons
    run_reference_operations_comparison()
    compare_conv_operations()

    # Then run our VAE
    run_our_vae_with_trace()

    print("\n" + "=" * 80)
    print("TRACE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
