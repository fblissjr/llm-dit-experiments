#!/usr/bin/env python3
"""
Compare connector outputs between our weights and reference weights.

Last Updated: 2026-01-20

Critical test: Do both connector weight sets produce the same extreme per-dim means?

If yes → expected behavior of this connector architecture
If no → bug in our connector weight loading or forward pass
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


def load_connector_weights(path: str, prefix: str = "video_connector.") -> dict:
    """Load connector weights from safetensors file."""
    weights = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            if key.startswith(prefix):
                clean_key = key[len(prefix):]  # Remove prefix
                weights[clean_key] = f.get_tensor(key)
    logger.info(f"Loaded {len(weights)} weights from {Path(path).name}")
    return weights


def test_connector_with_weights(connector_weights: dict, input_tensor: torch.Tensor, name: str):
    """Test connector output statistics with given weights."""
    from llm_dit.encoders.embeddings_connector import Embeddings1DConnector
    import json

    # Load connector config
    config_path = "models/LTX-2/connectors/config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Create connector
    connector = Embeddings1DConnector.from_config(config)
    connector = connector.to(dtype=torch.bfloat16, device="cuda")

    # Load weights
    connector.load_state_dict(connector_weights, strict=False)

    # Run forward pass
    with torch.no_grad():
        # Create dummy additive mask (all valid)
        mask = torch.zeros(1, 1, 1, input_tensor.shape[1], device="cuda", dtype=torch.bfloat16)
        output, out_mask = connector(input_tensor, mask)

    # Analyze output
    out_flat = output[0].float().cpu()
    dim_means = out_flat.mean(dim=0)
    dim_stds = out_flat.std(dim=0)

    logger.info(f"\n{name}:")
    logger.info(f"  Output shape: {output.shape}")
    logger.info(f"  Overall mean: {out_flat.mean():.4f}, std: {out_flat.std():.4f}")
    logger.info(f"  Per-dim mean range: [{dim_means.min():.2f}, {dim_means.max():.2f}]")
    logger.info(f"  Per-dim std range: [{dim_stds.min():.2f}, {dim_stds.max():.2f}]")
    logger.info(f"  Dims with |mean| > 5: {(dim_means.abs() > 5).sum().item()}")

    return output, dim_means


def main():
    logger.info("=" * 70)
    logger.info("COMPARING CONNECTOR OUTPUTS WITH DIFFERENT WEIGHTS")
    logger.info("=" * 70)

    # Load connector weights from two sources
    our_weights_path = "models/LTX-2/text_encoder/diffusion_pytorch_model-00011-of-00012.safetensors"
    standalone_weights_path = "models/LTX-2/connectors/model.safetensors"

    if not Path(our_weights_path).exists():
        logger.error(f"Not found: {our_weights_path}")
        return False

    our_weights = load_connector_weights(our_weights_path)

    standalone_exists = Path(standalone_weights_path).exists()
    if standalone_exists:
        standalone_weights = load_connector_weights(standalone_weights_path)
    else:
        logger.warning(f"Standalone connector not found: {standalone_weights_path}")
        standalone_weights = None

    # Create a test input that matches the feature_extractor output distribution
    # Based on tracing: mean=0, std=0.03, per-dim means near zero
    logger.info("\nCreating test input matching feature_extractor distribution...")
    test_input = torch.randn(1, 256, 3840, dtype=torch.bfloat16, device="cuda") * 0.03
    logger.info(f"  Input mean: {test_input.float().mean():.4f}, std: {test_input.float().std():.4f}")

    # Test with our weights
    out1, means1 = test_connector_with_weights(our_weights, test_input, "Our weights (text_encoder shard)")

    # Test with standalone weights if available
    if standalone_weights is not None:
        out2, means2 = test_connector_with_weights(standalone_weights, test_input, "Standalone weights (connectors/)")

        # Compare
        logger.info("\n" + "=" * 70)
        logger.info("COMPARISON")
        logger.info("=" * 70)

        mean_diff = (means1 - means2).abs().mean().item()
        max_diff = (means1 - means2).abs().max().item()
        logger.info(f"Mean absolute difference in per-dim means: {mean_diff:.4f}")
        logger.info(f"Max absolute difference in per-dim means: {max_diff:.4f}")

    # Verdict
    logger.info("\n" + "=" * 70)
    logger.info("VERDICT")
    logger.info("=" * 70)

    if means1.abs().max() > 5:
        logger.info("The connector architecture creates extreme per-dim offsets.")
        logger.info("")
        logger.info("NEXT STEP: Check if reference LTX-2 also has these offsets.")
        logger.info("If yes -> this is expected behavior")
        logger.info("If no -> our connector forward pass has a bug")
    else:
        logger.info("Per-dim means are reasonable after connector")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
