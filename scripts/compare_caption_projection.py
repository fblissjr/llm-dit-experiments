#!/usr/bin/env python3
"""
Compare caption_projection behavior between our implementation and reference.

Last Updated: 2026-01-20

This script tests the CRITICAL question:
- Does the reference implementation also produce -9.5 mean at caption_projection.linear_1?
- If yes: -9.5 is expected behavior and we're done
- If no: our connector has a bug even with correct Gemma weights

Strategy:
1. Load our encoder with fixed Gemma weights
2. Get connector output (what goes INTO caption_projection)
3. Load caption_projection weights from DiT checkpoint
4. Compute Wx + b decomposition
5. Compare against expected behavior
"""

import logging
import sys
from pathlib import Path

import torch
from safetensors import safe_open

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_caption_projection_weights(checkpoint_path: str) -> dict:
    """Load caption_projection weights from DiT checkpoint."""
    with safe_open(checkpoint_path, framework="pt") as f:
        weights = {}
        for key in f.keys():
            if "caption_projection" in key:
                weights[key] = f.get_tensor(key)
                logger.info(f"Loaded {key}: {weights[key].shape}")
    return weights


def analyze_wx_decomposition(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> dict:
    """Analyze the Wx + b decomposition to understand where -9.5 comes from."""
    # x: [B, T, D_in]
    # W: [D_out, D_in]
    # b: [D_out]

    B, T, D_in = x.shape
    D_out = W.shape[0]

    # Flatten for analysis
    x_flat = x.reshape(-1, D_in)  # [B*T, D_in]

    # Compute Wx
    Wx = x_flat @ W.T  # [B*T, D_out]

    # Full output
    out = Wx + b

    # Per-dimension analysis of input
    x_dim_means = x_flat.mean(dim=0)  # [D_in]
    x_dim_stds = x_flat.std(dim=0)

    # Weight column analysis
    W_col_sums = W.sum(dim=0)  # [D_in] - sum of each column

    # Per-dimension contribution to Wx mean
    # Wx_mean ≈ sum(x_dim_mean[i] * W_col_sum[i]) / D_out
    dim_contributions = x_dim_means * W_col_sums

    return {
        "x_mean": x_flat.mean().item(),
        "x_std": x_flat.std().item(),
        "x_dim_mean_range": (x_dim_means.min().item(), x_dim_means.max().item()),
        "x_dim_std_range": (x_dim_stds.min().item(), x_dim_stds.max().item()),
        "Wx_mean": Wx.mean().item(),
        "Wx_std": Wx.std().item(),
        "b_mean": b.mean().item(),
        "out_mean": out.mean().item(),
        "out_std": out.std().item(),
        "total_dim_contribution": dim_contributions.sum().item(),
        "top_negative_dims": torch.topk(-dim_contributions, 5),
        "top_positive_dims": torch.topk(dim_contributions, 5),
    }


def main():
    logger.info("=" * 70)
    logger.info("CAPTION_PROJECTION ANALYSIS (Post Gemma Fix)")
    logger.info("=" * 70)

    # Load our encoder with fixed Gemma weights
    logger.info("\nStep 1: Loading encoder with fixed Gemma weights...")
    from llm_dit.encoders.gemma3 import Gemma3Encoder

    encoder = Gemma3Encoder.from_pretrained(
        device="auto",
        max_sequence_length=256,
        max_memory={0: "20GiB", "cpu": "32GiB"},
    )

    # Encode test prompt
    test_prompt = "A fluffy orange cat sleeping peacefully on a soft red couch."
    logger.info(f"\nStep 2: Encoding prompt: '{test_prompt}'")
    output = encoder.encode(test_prompt, return_padded=True)
    connector_output = output.padded_embeddings  # [1, 256, 3840]

    logger.info(f"Connector output shape: {connector_output.shape}")
    logger.info(f"Connector output mean: {connector_output.float().mean():.4f}")
    logger.info(f"Connector output std: {connector_output.float().std():.4f}")

    # Per-dimension analysis
    co_flat = connector_output[0].float()  # [256, 3840]
    dim_means = co_flat.mean(dim=0)
    dim_stds = co_flat.std(dim=0)
    logger.info(f"Per-dim mean range: [{dim_means.min():.2f}, {dim_means.max():.2f}]")
    logger.info(f"Per-dim std range: [{dim_stds.min():.2f}, {dim_stds.max():.2f}]")
    logger.info(f"Dims with |mean| > 5: {(dim_means.abs() > 5).sum().item()}")

    # Load caption_projection weights
    logger.info("\nStep 3: Loading caption_projection weights from DiT...")
    dit_path = "models/LTX-2/transformer/diffusion_pytorch_model-00001-of-00008.safetensors"
    if not Path(dit_path).exists():
        logger.error(f"DiT checkpoint not found: {dit_path}")
        return False

    with safe_open(dit_path, framework="pt") as f:
        linear1_weight = f.get_tensor("caption_projection.linear_1.weight").to(torch.float32)
        linear1_bias = f.get_tensor("caption_projection.linear_1.bias").to(torch.float32)
        linear2_weight = f.get_tensor("caption_projection.linear_2.weight").to(torch.float32)
        linear2_bias = f.get_tensor("caption_projection.linear_2.bias").to(torch.float32)

    logger.info(f"linear_1 weight: {linear1_weight.shape}")
    logger.info(f"linear_1 bias: {linear1_bias.shape}")

    # Analyze Wx + b decomposition
    logger.info("\nStep 4: Analyzing Wx + b decomposition...")
    x = connector_output.float().to("cpu")
    analysis = analyze_wx_decomposition(x, linear1_weight, linear1_bias)

    logger.info(f"\n--- Input (Connector Output) ---")
    logger.info(f"  mean: {analysis['x_mean']:.4f}")
    logger.info(f"  std: {analysis['x_std']:.4f}")
    logger.info(f"  per-dim mean range: {analysis['x_dim_mean_range']}")
    logger.info(f"  per-dim std range: {analysis['x_dim_std_range']}")

    logger.info(f"\n--- Wx Term ---")
    logger.info(f"  mean: {analysis['Wx_mean']:.4f}")
    logger.info(f"  std: {analysis['Wx_std']:.4f}")

    logger.info(f"\n--- Bias Term ---")
    logger.info(f"  mean: {analysis['b_mean']:.4f}")

    logger.info(f"\n--- Output (Wx + b) ---")
    logger.info(f"  mean: {analysis['out_mean']:.4f}")
    logger.info(f"  std: {analysis['out_std']:.4f}")

    logger.info(f"\n--- Dimension Contribution Analysis ---")
    logger.info(f"  Total contribution: {analysis['total_dim_contribution']:.2f}")

    top_neg = analysis['top_negative_dims']
    logger.info(f"  Top 5 negative contributors:")
    for i in range(5):
        idx = top_neg.indices[i].item()
        val = -top_neg.values[i].item()  # Negated back
        logger.info(f"    Dim {idx}: contribution = {val:.2f}")

    # GELU analysis
    logger.info("\nStep 5: GELU activation analysis...")
    Wx = x.reshape(-1, x.shape[-1]) @ linear1_weight.T + linear1_bias
    gelu = torch.nn.functional.gelu(Wx, approximate='tanh')

    zero_mask = gelu.abs() < 1e-6
    near_zero_mask = gelu.abs() < 0.01

    logger.info(f"  Wx output mean: {Wx.mean():.2f}")
    logger.info(f"  GELU output mean: {gelu.mean():.4f}")
    logger.info(f"  GELU output std: {gelu.std():.4f}")
    logger.info(f"  Zero activations: {zero_mask.float().mean() * 100:.1f}%")
    logger.info(f"  Near-zero activations: {near_zero_mask.float().mean() * 100:.1f}%")

    # Verdict
    logger.info("\n" + "=" * 70)
    if analysis['Wx_mean'] < -5:
        logger.info("🔴 VERDICT: Wx mean is highly negative ({:.2f})".format(analysis['Wx_mean']))
        logger.info("   The connector output correlates negatively with caption_projection weights.")
        logger.info("   This causes 90%+ of GELU activations to die.")
        logger.info("")
        logger.info("   CRITICAL QUESTION: Does the reference implementation behave the same way?")
        logger.info("   If yes → expected behavior (model uses sparse activations)")
        logger.info("   If no → our connector still has a bug in forward() logic or RoPE")
    else:
        logger.info("🟢 VERDICT: Wx mean is healthy ({:.2f})".format(analysis['Wx_mean']))
        logger.info("   The fix worked! Signal should flow through GELU properly.")
    logger.info("=" * 70)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
