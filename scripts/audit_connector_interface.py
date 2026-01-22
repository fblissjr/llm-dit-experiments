#!/usr/bin/env python3
"""
Audit the Connector-to-DiT interface.

Last Updated: 2026-01-20

GEMINI'S MATHEMATICAL INSIGHT:
  output = Wx + b
  -9.4   = Wx + (-0.007)
  ∴ Wx   ≈ -9.4

The connector output x has structure that correlates negatively with
caption_projection weights W. This script traces exactly where that
structure comes from.

TEST:
1. Get healthy feature extractor output (verified: mean≈0, std≈0.14)
2. Pass through connector
3. Measure connector output structure
4. Pass through caption_projection.linear_1
5. See if we reproduce the -9.4

If we can reproduce -9.4, we can then compare against reference connector.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_caption_projection_weights(dit_path: str) -> dict:
    """Load caption_projection weights from DiT checkpoint."""
    weights = {}
    with safe_open(dit_path, framework="pt") as f:
        for key in f.keys():
            if "caption_projection" in key and "audio" not in key:
                weights[key] = f.get_tensor(key)
    return weights


def analyze_wx_contribution(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> dict:
    """
    Analyze the Wx + b decomposition.

    Args:
        x: Input tensor [B, T, D_in]
        W: Weight matrix [D_out, D_in]
        b: Bias vector [D_out]

    Returns:
        Dict with Wx stats, b stats, and total output stats
    """
    # Compute Wx
    Wx = torch.matmul(x.float(), W.float().t())  # [B, T, D_out]

    # Compute output
    output = Wx + b.float()

    return {
        "Wx_mean": Wx.mean().item(),
        "Wx_std": Wx.std().item(),
        "b_mean": b.float().mean().item(),
        "output_mean": output.mean().item(),
        "output_std": output.std().item(),
        "x_mean": x.float().mean().item(),
        "x_std": x.float().std().item(),
    }


def run_full_pipeline_audit(
    connectors_path: str,
    dit_path: str,
    model_id: str = "models/LTX-2/text_encoder",
    test_prompt: str = "A fluffy orange cat sleeping peacefully on a soft red couch.",
):
    """
    Run the full pipeline and trace where -9.4 comes from.
    """
    logger.info("=" * 70)
    logger.info("CONNECTOR-TO-DIT INTERFACE AUDIT")
    logger.info("=" * 70)

    # Load our encoder with memory limits to allow CPU offloading
    logger.info("\nLoading Gemma3Encoder with memory limits...")
    from llm_dit.encoders.gemma3 import Gemma3Encoder

    encoder = Gemma3Encoder.from_pretrained(
        model_path=model_id,
        device="auto",  # Allow accelerate to manage memory
        dtype="bfloat16",
        connectors_path=connectors_path,
        use_connector=True,  # Full pipeline with connector
        max_memory={0: "20GiB", "cpu": "32GiB"},  # Limit GPU, allow CPU offload
    )

    # Encode the prompt
    logger.info(f"\nEncoding: '{test_prompt}'")
    output = encoder.encode(test_prompt, return_padded=True)

    # Get the padded embeddings (this is what goes to DiT)
    connector_output = output.padded_embeddings  # [B, T, 3840]
    attention_mask = output.padded_mask  # [B, T]

    logger.info(f"\n--- Connector Output (Input to DiT) ---")
    logger.info(f"Shape: {connector_output.shape}")

    # Compute stats on valid tokens only
    valid_mask = attention_mask.bool()
    valid_embeddings = connector_output[valid_mask]

    logger.info(f"Valid tokens: {valid_embeddings.shape[0]}")
    logger.info(f"Overall mean: {valid_embeddings.float().mean():.6f}")
    logger.info(f"Overall std: {valid_embeddings.float().std():.6f}")

    # Per-dimension analysis (this is key!)
    per_dim_mean = valid_embeddings.float().mean(dim=0)  # [3840]
    per_dim_std = valid_embeddings.float().std(dim=0)  # [3840]

    logger.info(f"\nPer-dimension analysis:")
    logger.info(f"  Mean range: [{per_dim_mean.min():.4f}, {per_dim_mean.max():.4f}]")
    logger.info(f"  Std range: [{per_dim_std.min():.4f}, {per_dim_std.max():.4f}]")
    logger.info(f"  Dims with |mean| > 1.0: {(per_dim_mean.abs() > 1.0).sum().item()}")
    logger.info(f"  Dims with std < 0.5: {(per_dim_std < 0.5).sum().item()}")

    # Free GPU memory (skip offload if using accelerate device_map)
    try:
        encoder.offload()
    except RuntimeError as e:
        if "accelerate" in str(e).lower() or "offloaded" in str(e).lower():
            logger.info("Skipping offload (model uses accelerate hooks)")
        else:
            raise
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    # Move to CPU for caption_projection analysis
    connector_output_cpu = connector_output.cpu().float()

    # Load caption_projection weights
    logger.info(f"\n--- Caption Projection Analysis ---")
    logger.info(f"Loading weights from {dit_path}...")

    cp_weights = load_caption_projection_weights(dit_path)

    W1 = cp_weights["caption_projection.linear_1.weight"]  # [4096, 3840]
    b1 = cp_weights["caption_projection.linear_1.bias"]  # [4096]

    logger.info(f"linear_1 weight shape: {W1.shape}")
    logger.info(f"linear_1 bias shape: {b1.shape}")

    # THE KEY TEST: Decompose output = Wx + b
    logger.info(f"\n--- Wx + b Decomposition ---")

    # Use valid tokens only
    valid_input = connector_output_cpu[attention_mask.cpu().bool()]  # [N, 3840]

    decomp = analyze_wx_contribution(valid_input.unsqueeze(0), W1, b1)

    logger.info(f"Input x:  mean={decomp['x_mean']:.6f}, std={decomp['x_std']:.6f}")
    logger.info(f"Wx term:  mean={decomp['Wx_mean']:.6f}, std={decomp['Wx_std']:.6f}")
    logger.info(f"b term:   mean={decomp['b_mean']:.6f}")
    logger.info(f"Output:   mean={decomp['output_mean']:.6f}, std={decomp['output_std']:.6f}")

    # Check if we reproduced -9.4
    if decomp["output_mean"] < -5.0:
        logger.info(f"\n🔴 REPRODUCED: Output mean is {decomp['output_mean']:.2f} (expected ~-9.4)")
        logger.info(
            f"   The connector output correlates negatively with caption_projection weights!"
        )
    else:
        logger.info(f"\n🟢 Output mean is {decomp['output_mean']:.2f} (NOT in -9.4 range)")
        logger.info(f"   Need to check if this matches reference behavior...")

    # Detailed correlation analysis
    logger.info(f"\n--- Weight-Input Correlation Analysis ---")

    # Compute correlation between input dimensions and weight rows
    # High negative correlation means input structure opposes weight structure
    x_centered = valid_input - valid_input.mean(dim=0, keepdim=True)
    W_centered = W1.float() - W1.float().mean(dim=0, keepdim=True)

    # Compute mean contribution per input dimension
    # Each input dim contributes: sum over output dims of (x_i * W_ij)
    per_dim_contribution = valid_input.float().mean(dim=0) * W1.float().sum(dim=0)

    logger.info(f"Per-dimension contribution to Wx:")
    logger.info(f"  Sum: {per_dim_contribution.sum():.4f}")
    logger.info(f"  Top 5 positive: {per_dim_contribution.topk(5).values.tolist()}")
    logger.info(f"  Top 5 negative: {(-per_dim_contribution).topk(5).values.tolist()}")

    # Find which dimensions contribute most to the negative shift
    neg_contributors = (per_dim_contribution < -0.1).nonzero().squeeze()
    if neg_contributors.numel() > 0:
        logger.info(f"  Dims with large negative contribution: {neg_contributors.numel()}")

    return {
        "connector_output": connector_output_cpu,
        "decomposition": decomp,
        "per_dim_mean": per_dim_mean,
        "per_dim_std": per_dim_std,
    }


def compare_with_reference_connector():
    """
    TODO: Load and run reference connector for comparison.

    This would require:
    1. Loading the reference VideoGemmaTextEncoderModel
    2. Running same input through it
    3. Comparing output structure
    """
    logger.info("\n--- Reference Comparison (TODO) ---")
    logger.info("To implement: Load reference connector and compare outputs")
    logger.info("This will determine if -9.4 is expected or a bug in our connector")


def main():
    parser = argparse.ArgumentParser(description="Audit connector-to-DiT interface")
    parser.add_argument(
        "--connectors-path",
        type=str,
        default="models/LTX-2/connectors/diffusion_pytorch_model.safetensors",
        help="Path to connectors checkpoint",
    )
    parser.add_argument(
        "--dit-path",
        type=str,
        default="models/LTX-2/transformer/diffusion_pytorch_model-00001-of-00008.safetensors",
        help="Path to DiT transformer checkpoint (first shard)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="models/LTX-2/text_encoder",
        help="Gemma model ID or path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A fluffy orange cat sleeping peacefully on a soft red couch.",
        help="Test prompt",
    )
    args = parser.parse_args()

    results = run_full_pipeline_audit(
        connectors_path=args.connectors_path,
        dit_path=args.dit_path,
        model_id=args.model_id,
        test_prompt=args.prompt,
    )

    compare_with_reference_connector()


if __name__ == "__main__":
    main()
