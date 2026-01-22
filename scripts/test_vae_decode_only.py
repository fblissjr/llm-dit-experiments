#!/usr/bin/env python3
"""
Test VAE decode only using reference latents.

Last Updated: 2026-01-21

This script:
1. Uses ModelLedger to load the reference VAE
2. Creates normalized test latents (mean≈0, std≈1)
3. Decodes with reference VAE
4. Decodes with our VAE
5. Compares outputs

This isolates VAE decode without needing the full text encoder.

Usage:
    uv run python scripts/test_vae_decode_only.py
"""

import sys
from pathlib import Path

import torch
from PIL import Image

# Add reference code paths
sys.path.insert(0, str(Path(__file__).parent.parent / "coderef/LTX-2/packages/ltx-core/src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "coderef/LTX-2/packages/ltx-pipelines/src"))


def print_stats(name: str, tensor: torch.Tensor) -> None:
    """Print tensor statistics."""
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}, dtype: {tensor.dtype}")
    print(f"  Mean: {tensor.float().mean():.4f}, Std: {tensor.float().std():.4f}")
    print(f"  Range: [{tensor.float().min():.4f}, {tensor.float().max():.4f}]")


def main():
    print("=" * 80)
    print("VAE DECODE COMPARISON TEST")
    print("=" * 80)

    device = "cuda"
    dtype = torch.bfloat16
    checkpoint_path = "models/LTX-2/ltx-2-19b-dev-fp8.safetensors"

    # Test latent shape: [B, C, T, H, W] = [1, 128, 2, 8, 12]
    # This corresponds to 9 frames at 256x384 resolution
    # T_latent = (9-1)/8 + 1 = 2
    # H_latent = 256/32 = 8
    # W_latent = 384/32 = 12
    latent_shape = (1, 128, 2, 8, 12)

    # Create normalized test latents (what the model should output after denoising)
    print("\n--- Creating normalized test latents ---")
    torch.manual_seed(42)
    latents = torch.randn(latent_shape, device=device, dtype=dtype)
    print_stats("Test latents (normalized)", latents)

    # =========================================================================
    # REFERENCE VAE DECODE
    # =========================================================================
    print("\n" + "=" * 40)
    print("REFERENCE VAE DECODE (via ModelLedger)")
    print("=" * 40)

    try:
        from ltx_pipelines.utils import ModelLedger

        model_ledger = ModelLedger(
            dtype=dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=None,  # Don't load Gemma
            fp8transformer=True,
        )

        # Get reference decoder
        ref_decoder = model_ledger.video_decoder()
        print(f"Reference decoder type: {type(ref_decoder).__name__}")

        # Check per_channel_statistics buffers
        ref_std = ref_decoder.per_channel_statistics.get_buffer("std-of-means")
        ref_mean = ref_decoder.per_channel_statistics.get_buffer("mean-of-means")
        print(f"\nReference PerChannelStatistics:")
        print(f"  std-of-means: mean={ref_std.mean():.4f}, range=[{ref_std.min():.4f}, {ref_std.max():.4f}]")
        print(f"  mean-of-means: mean={ref_mean.mean():.4f}, range=[{ref_mean.min():.4f}, {ref_mean.max():.4f}]")

        # Decode with reference
        with torch.no_grad():
            ref_output = ref_decoder(latents.clone())
        print_stats("\nReference VAE output", ref_output)

        # Free memory
        del ref_decoder
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Reference VAE failed: {e}")
        import traceback
        traceback.print_exc()
        ref_output = None

    # =========================================================================
    # OUR VAE DECODE
    # =========================================================================
    print("\n" + "=" * 40)
    print("OUR VAE DECODE")
    print("=" * 40)

    try:
        from llm_dit.models.ltx2.vae import load_ltx2_vae_decoder

        our_decoder = load_ltx2_vae_decoder(
            "models/LTX-2/vae",
            dtype=dtype,
            device="cpu",
        ).to(device)
        print(f"Our decoder type: {type(our_decoder).__name__}")

        # Check per_channel_statistics buffers
        our_std = our_decoder.per_channel_statistics.get_buffer("std-of-means")
        our_mean = our_decoder.per_channel_statistics.get_buffer("mean-of-means")
        print(f"\nOur PerChannelStatistics:")
        print(f"  std-of-means: mean={our_std.mean():.4f}, range=[{our_std.min():.4f}, {our_std.max():.4f}]")
        print(f"  mean-of-means: mean={our_mean.mean():.4f}, range=[{our_mean.min():.4f}, {our_mean.max():.4f}]")

        # Decode with our VAE
        with torch.no_grad():
            our_output = our_decoder(latents.clone())
        print_stats("\nOur VAE output", our_output)

        # Free memory
        del our_decoder
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Our VAE failed: {e}")
        import traceback
        traceback.print_exc()
        our_output = None

    # =========================================================================
    # COMPARISON
    # =========================================================================
    print("\n" + "=" * 40)
    print("COMPARISON")
    print("=" * 40)

    if ref_output is not None and our_output is not None:
        diff = (ref_output.float() - our_output.float()).abs()
        print_stats("Absolute difference", diff)
        print(f"\nMax absolute difference: {diff.max():.6f}")
        print(f"Mean absolute difference: {diff.mean():.6f}")

        # Check if outputs match
        if diff.max() < 0.01:
            print("\n✅ OUTPUTS MATCH CLOSELY!")
        elif diff.max() < 0.1:
            print("\n⚠️ OUTPUTS SIMILAR but not exact")
        else:
            print("\n❌ OUTPUTS DIFFER SIGNIFICANTLY")

        # Save comparison frames
        import os
        os.makedirs("/tmp/claude", exist_ok=True)

        def save_frame(tensor: torch.Tensor, path: str):
            # tensor is [B, C, T, H, W] in [-1, 1]
            frame = tensor[0, :, 0]  # First frame
            frame = ((frame + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
            frame = frame.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
            Image.fromarray(frame).save(path)

        save_frame(ref_output, "/tmp/claude/vae_ref_decode.png")
        save_frame(our_output, "/tmp/claude/vae_our_decode.png")
        print("\nSaved comparison frames to /tmp/claude/")
    else:
        print("Cannot compare - one or both decoders failed")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
