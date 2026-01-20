#!/usr/bin/env python3
"""
Test FP8 transformer forward pass with dummy/precomputed inputs.

Goal: Verify the transformer works correctly independent of text encoder.
"""

import logging
import sys
from pathlib import Path

# Add reference code paths
sys.path.insert(0, str(Path(__file__).parent.parent / "coderef/LTX-2/packages/ltx-core/src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "coderef/LTX-2/packages/ltx-pipelines/src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "coderef/LTX-2/packages/ltx-trainer/src"))

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 70)
    logger.info("FP8 TRANSFORMER FORWARD PASS TEST")
    logger.info("=" * 70)

    checkpoint_path = "models/LTX-2/ltx-2-19b-dev-fp8.safetensors"
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Import after path setup
    from ltx_pipelines.utils import ModelLedger
    from ltx_core.model.transformer.modality import Modality

    # Load transformer via ModelLedger
    logger.info("\nLoading FP8 transformer via ModelLedger...")
    model_ledger = ModelLedger(
        dtype=dtype,
        device=device,
        checkpoint_path=checkpoint_path,
        gemma_root_path=None,
        fp8transformer=True,
    )
    transformer = model_ledger.transformer()

    # Memory status
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    # Create dummy inputs matching expected shapes
    # Video: small test case - 1 batch, ~8 latent frames (17 pixel frames -> 3 latent)
    # The patchifier creates sequence of spatial patches
    batch_size = 1
    num_latent_frames = 3  # (17-1)/8 + 1 = 3 latent frames for 17 pixel frames
    latent_h = 8  # 256 / 32
    latent_w = 12  # 384 / 32
    video_seq_len = num_latent_frames * latent_h * latent_w  # 3 * 8 * 12 = 288
    video_latent_dim = 128  # LTX-2 video latent dimension

    # Context (text embedding) - Gemma 12B hidden_size is 3840
    # The caption_projection maps 3840 -> 4096 (DiT hidden size)
    context_seq_len = 1504
    context_dim = 3840  # Gemma 12B hidden dimension

    logger.info(f"\nTest configuration:")
    logger.info(f"  Video seq_len: {video_seq_len} (from {num_latent_frames}x{latent_h}x{latent_w})")
    logger.info(f"  Context seq_len: {context_seq_len}")
    logger.info(f"  Latent dim: {video_latent_dim}, Context dim: {context_dim}")

    # Create dummy tensors
    video_latent = torch.randn(batch_size, video_seq_len, video_latent_dim, dtype=dtype, device=device)
    video_timesteps = torch.full((batch_size, video_seq_len), 1.0, dtype=torch.float32, device=device)

    # Positions: [B, 3, seq_len, 2] for video (3 dims: time, height, width)
    video_positions = torch.zeros(batch_size, 3, video_seq_len, 2, dtype=torch.float32, device=device)

    # Context: [1, 1504, 4096]
    video_context = torch.randn(batch_size, context_seq_len, context_dim, dtype=dtype, device=device)

    # Audio disabled for this test
    audio_latent = torch.zeros(batch_size, 1, 128, dtype=dtype, device=device)
    audio_timesteps = torch.zeros(batch_size, 1, dtype=torch.float32, device=device)
    audio_positions = torch.zeros(batch_size, 1, 1, 2, dtype=torch.float32, device=device)
    audio_context = torch.zeros(batch_size, context_seq_len, context_dim, dtype=dtype, device=device)

    # Create Modality objects
    video = Modality(
        enabled=True,
        latent=video_latent,
        timesteps=video_timesteps,
        positions=video_positions,
        context=video_context,
        context_mask=None,
    )
    audio = Modality(
        enabled=False,
        latent=audio_latent,
        timesteps=audio_timesteps,
        positions=audio_positions,
        context=audio_context,
        context_mask=None,
    )

    # Forward pass - just test the velocity model first
    logger.info("\nRunning forward pass through velocity_model only...")
    with torch.no_grad():
        # Test just the velocity model to see what shapes come out
        vx, ax = transformer.velocity_model(video=video, audio=audio, perturbations=None)
        logger.info(f"Velocity model output (vx) shape: {vx.shape}")
        logger.info(f"Video latent shape: {video.latent.shape}")
        logger.info(f"Video timesteps shape: {video.timesteps.shape}")

        # Now test to_denoised manually
        from ltx_core.utils import to_denoised
        sigma = video.timesteps  # [1, 288]
        logger.info(f"Sigma shape for to_denoised: {sigma.shape}")

        # The issue: to_denoised does velocity * sigma
        # sigma needs to broadcast to [1, 288, 128]
        # Let's test the broadcast
        sigma_expanded = sigma.unsqueeze(-1)  # [1, 288, 1]
        logger.info(f"Sigma expanded shape: {sigma_expanded.shape}")

        # Manual denoised calculation
        video_pred = (video.latent.float() - vx.float() * sigma_expanded).to(video.latent.dtype)

    logger.info(f"\nOutput shapes:")
    logger.info(f"  Video pred: {video_pred.shape}")
    logger.info(f"  Audio velocity (ax): {ax.shape if ax is not None else 'None'}")

    # Analyze output
    logger.info(f"\nVideo prediction stats:")
    logger.info(f"  Mean: {video_pred.float().mean():.4f}")
    logger.info(f"  Std: {video_pred.float().std():.4f}")
    logger.info(f"  Min: {video_pred.float().min():.4f}")
    logger.info(f"  Max: {video_pred.float().max():.4f}")
    logger.info(f"  Has NaN: {video_pred.isnan().any().item()}")
    logger.info(f"  Has Inf: {video_pred.isinf().any().item()}")

    # Check for signal death
    zero_fraction = (video_pred.abs() < 1e-6).float().mean().item()
    logger.info(f"  Zero fraction: {zero_fraction*100:.2f}%")

    logger.info("\n" + "=" * 70)
    logger.info("FORWARD PASS TEST COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
