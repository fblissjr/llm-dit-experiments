#!/usr/bin/env python3
"""
VAE Reconstruction Test - Round-trip encode/decode to verify VAE isolation.

Last Updated: 2026-01-21

This test validates VAE correctness by:
1. Loading a real image (512x512)
2. Encoding to latent space
3. Measuring latent statistics (expect std ≈ 1.0 if normalized correctly)
4. Decoding back to pixel space
5. Comparing reconstruction to original (PSNR > 30dB expected)

Decision tree:
- If reconstruction looks good → Issue is in generation pipeline
- If reconstruction is blurry → Issue is in VAE code itself

Usage:
    uv run python scripts/test_vae_reconstruction.py
"""

import json
import logging
import sys
from pathlib import Path

import torch
from PIL import Image
from einops import rearrange

# Add coderef paths for reference VAE/encoder
coderef_root = Path(__file__).parent.parent / "coderef" / "LTX-2" / "packages"
sys.path.insert(0, str(coderef_root / "ltx-core" / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_test_image(size: tuple[int, int] = (512, 512)) -> torch.Tensor:
    """
    Load or create a test image for VAE reconstruction test.

    Returns:
        Tensor of shape [1, 3, 1, H, W] in range [-1, 1] (video format with 1 frame)
    """
    # Try to find a real test image
    test_image_paths = [
        Path("coderef/LightX2V/assets/inputs/imgs/img_0.jpg"),
        Path("coderef/DiffSynth-Engine/tests/data/input/capybara.jpg"),
        Path(".venv/lib/python3.13/site-packages/matplotlib/mpl-data/sample_data/grace_hopper.jpg"),
    ]

    image = None
    for path in test_image_paths:
        if path.exists():
            logger.info(f"Loading test image: {path}")
            image = Image.open(path).convert("RGB")
            break

    if image is None:
        # Create a synthetic test image with gradients and patterns
        logger.info("Creating synthetic test image")
        import numpy as np
        h, w = size
        # Create gradient with some high-frequency content
        y, x = np.mgrid[0:h, 0:w]
        r = (x / w * 255).astype(np.uint8)
        g = (y / h * 255).astype(np.uint8)
        b = ((np.sin(x / 10) * 0.5 + 0.5) * 255).astype(np.uint8)
        image = Image.fromarray(np.stack([r, g, b], axis=-1))

    # Resize to target size
    image = image.resize(size, Image.Resampling.LANCZOS)

    # Convert to tensor [1, 3, 1, H, W] - video format with 1 frame
    import numpy as np
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # [3, H, W]
    image_tensor = (image_tensor * 2) - 1  # Scale to [-1, 1]
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]

    return image_tensor


def calculate_psnr(original: torch.Tensor, reconstruction: torch.Tensor) -> float:
    """Calculate Peak Signal-to-Noise Ratio between two images."""
    # Both should be in [-1, 1] range
    mse = ((original - reconstruction) ** 2).mean()
    if mse == 0:
        return float('inf')
    # For [-1, 1] range, max value is 2
    psnr = 10 * torch.log10(4.0 / mse)
    return psnr.item()


def load_reference_encoder_decoder(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16
):
    """Load reference VideoEncoder and VideoDecoder from coderef."""
    from ltx_core.model.video_vae.model_configurator import (
        VideoEncoderConfigurator,
        VideoDecoderConfigurator,
    )
    from safetensors.torch import load_file

    vae_path = Path(model_path) / "vae"

    # Load config
    with open(vae_path / "config.json", "r") as f:
        config = json.load(f)

    # Build encoder and decoder
    encoder = VideoEncoderConfigurator.from_config(config)
    decoder = VideoDecoderConfigurator.from_config(config)

    # Load weights
    state_dict = load_file(vae_path / "diffusion_pytorch_model.safetensors")

    # Map keys for encoder
    encoder_state = {}
    for k, v in state_dict.items():
        if k == "latents_mean":
            encoder_state["per_channel_statistics.mean-of-means"] = v.to(dtype)
        elif k == "latents_std":
            encoder_state["per_channel_statistics.std-of-means"] = v.to(dtype)
        elif k.startswith("encoder."):
            encoder_state[k[8:]] = v.to(dtype)  # Remove "encoder." prefix

    # Map keys for decoder
    decoder_state = {}
    for k, v in state_dict.items():
        if k == "latents_mean":
            decoder_state["per_channel_statistics.mean-of-means"] = v.to(dtype)
        elif k == "latents_std":
            decoder_state["per_channel_statistics.std-of-means"] = v.to(dtype)
        elif k.startswith("decoder."):
            decoder_state[k[8:]] = v.to(dtype)  # Remove "decoder." prefix

    encoder.load_state_dict(encoder_state, strict=False)
    decoder.load_state_dict(decoder_state, strict=False)

    return encoder.to(device).to(dtype), decoder.to(device).to(dtype)


def load_our_decoder(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16
):
    """Load our VideoDecoder implementation."""
    from llm_dit.models.ltx2.vae import load_ltx2_vae_decoder
    decoder = load_ltx2_vae_decoder(Path(model_path) / "vae", dtype=dtype, device="cpu")
    return decoder.to(device)


def print_tensor_stats(name: str, tensor: torch.Tensor):
    """Print comprehensive tensor statistics."""
    print(f"\n{name}:")
    print(f"  Shape: {tensor.shape}, dtype: {tensor.dtype}")
    print(f"  Mean: {tensor.float().mean():.4f}")
    print(f"  Std: {tensor.float().std():.4f}")
    print(f"  Range: [{tensor.float().min():.4f}, {tensor.float().max():.4f}]")

    # Per-channel stats for latents
    if len(tensor.shape) == 5 and tensor.shape[1] == 128:
        per_channel_std = tensor.float().std(dim=(0, 2, 3, 4))
        print(f"  Per-channel std (first 8): {per_channel_std[:8].tolist()}")
        print(f"  Per-channel std mean: {per_channel_std.mean():.4f}")


def main():
    print("=" * 80)
    print("VAE RECONSTRUCTION TEST")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    model_path = "models/LTX-2"

    # Check model path exists
    if not Path(model_path).exists():
        logger.error(f"Model path not found: {model_path}")
        return

    # Load test image
    print("\n" + "=" * 40)
    print("LOADING TEST IMAGE")
    print("=" * 40)
    image = load_test_image((512, 512))
    print_tensor_stats("Input image", image)

    # Save original for visual comparison
    original_pil = Image.fromarray(
        ((image[0, :, 0].permute(1, 2, 0).numpy() + 1) / 2 * 255).astype('uint8')
    )
    original_pil.save("/tmp/claude/vae_test_original.png")
    logger.info("Saved original to /tmp/claude/vae_test_original.png")

    # Move to device
    image = image.to(device, dtype)

    # =========================================================================
    # TEST 1: Reference VAE Round-Trip
    # =========================================================================
    print("\n" + "=" * 40)
    print("TEST 1: REFERENCE VAE ROUND-TRIP")
    print("=" * 40)

    try:
        ref_encoder, ref_decoder = load_reference_encoder_decoder(model_path, device, dtype)

        # Check buffer values
        ref_std = ref_encoder.per_channel_statistics.get_buffer("std-of-means")
        ref_mean = ref_encoder.per_channel_statistics.get_buffer("mean-of-means")
        print(f"\nReference PerChannelStatistics:")
        print(f"  std-of-means: mean={ref_std.mean():.4f}, range=[{ref_std.min():.4f}, {ref_std.max():.4f}]")
        print(f"  mean-of-means: mean={ref_mean.mean():.4f}, range=[{ref_mean.min():.4f}, {ref_mean.max():.4f}]")

        with torch.no_grad():
            # Encode
            ref_latents = ref_encoder(image)
            print_tensor_stats("Reference latents (after encode)", ref_latents)

            # Decode
            ref_recon = ref_decoder(ref_latents)
            print_tensor_stats("Reference reconstruction", ref_recon)

        # Calculate PSNR
        ref_psnr = calculate_psnr(image, ref_recon)
        print(f"\nReference VAE PSNR: {ref_psnr:.2f} dB")

        # Save reference reconstruction
        ref_recon_pil = Image.fromarray(
            ((ref_recon[0, :, 0].permute(1, 2, 0).float().cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype('uint8')
        )
        ref_recon_pil.save("/tmp/claude/vae_test_ref_recon.png")
        logger.info("Saved reference reconstruction to /tmp/claude/vae_test_ref_recon.png")

        # Store latents for our VAE test
        test_latents = ref_latents.clone()

        del ref_encoder, ref_decoder
        torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Reference VAE test failed: {e}")
        import traceback
        traceback.print_exc()
        # Create synthetic latents for our VAE test
        test_latents = torch.randn(1, 128, 1, 16, 16, device=device, dtype=dtype)

    # =========================================================================
    # TEST 2: Our VAE Decoder (with reference-encoded latents)
    # =========================================================================
    print("\n" + "=" * 40)
    print("TEST 2: OUR VAE DECODER")
    print("=" * 40)

    try:
        our_decoder = load_our_decoder(model_path, device, dtype)

        # Check buffer values
        our_std = our_decoder.per_channel_statistics.get_buffer("std-of-means")
        our_mean = our_decoder.per_channel_statistics.get_buffer("mean-of-means")
        print(f"\nOur PerChannelStatistics:")
        print(f"  std-of-means: mean={our_std.mean():.4f}, range=[{our_std.min():.4f}, {our_std.max():.4f}]")
        print(f"  mean-of-means: mean={our_mean.mean():.4f}, range=[{our_mean.min():.4f}, {our_mean.max():.4f}]")

        # Verify buffers match reference
        if 'ref_std' in dir():
            std_match = torch.allclose(our_std.float(), ref_std.float(), atol=1e-4)
            mean_match = torch.allclose(our_mean.float(), ref_mean.float(), atol=1e-4)
            print(f"\n  Buffers match reference: std={std_match}, mean={mean_match}")

        with torch.no_grad():
            print_tensor_stats("Input latents", test_latents)
            our_recon = our_decoder(test_latents.clone())
            print_tensor_stats("Our reconstruction", our_recon)

        # Calculate PSNR
        our_psnr = calculate_psnr(image, our_recon)
        print(f"\nOur VAE PSNR: {our_psnr:.2f} dB")

        # Save our reconstruction
        our_recon_pil = Image.fromarray(
            ((our_recon[0, :, 0].permute(1, 2, 0).float().cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype('uint8')
        )
        our_recon_pil.save("/tmp/claude/vae_test_our_recon.png")
        logger.info("Saved our reconstruction to /tmp/claude/vae_test_our_recon.png")

        del our_decoder
        torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Our VAE test failed: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nOutput files saved to /tmp/claude/:")
    print(f"  - vae_test_original.png")
    print(f"  - vae_test_ref_recon.png")
    print(f"  - vae_test_our_recon.png")
    print()
    print("Decision tree:")
    print("  - If our reconstruction matches reference → VAE code is correct")
    print("  - If our reconstruction is blurry but reference is sharp → Bug in our VAE")
    print("  - If both are blurry → Check encoder or test image")
    print()
    print("Expected latent statistics after encode:")
    print("  - Mean ≈ 0 (due to normalization)")
    print("  - Std ≈ 1.0 (due to normalization)")


if __name__ == "__main__":
    main()
