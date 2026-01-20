#!/usr/bin/env python3
"""
Trace where the extreme per-dimension means originate in the text encoder pipeline.

Last Updated: 2026-01-20

Pipeline stages:
1. Gemma hidden states extraction (49 layers)
2. Layer stacking + normalization (_norm_and_concat_layers)
3. Feature extraction (text_proj_in: 188160 -> 3840)
4. Embeddings connector (2 transformer layers)
5. Final output

Track per-dim means at each stage to find the source.
"""

import logging
import sys

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def analyze_tensor(name: str, tensor: torch.Tensor):
    """Analyze per-dimension statistics of a tensor."""
    if tensor.dim() == 2:
        # [T, D]
        flat = tensor.float()
    elif tensor.dim() == 3:
        # [B, T, D]
        flat = tensor[0].float()
    else:
        logger.warning(f"{name}: Unexpected shape {tensor.shape}")
        return

    dim_means = flat.mean(dim=0)
    dim_stds = flat.std(dim=0)

    logger.info(f"\n{name}:")
    logger.info(f"  Shape: {tensor.shape}")
    logger.info(f"  Overall mean: {flat.mean():.4f}, std: {flat.std():.4f}")
    logger.info(f"  Per-dim mean range: [{dim_means.min():.2f}, {dim_means.max():.2f}]")
    logger.info(f"  Per-dim std range: [{dim_stds.min():.2f}, {dim_stds.max():.2f}]")
    logger.info(f"  Dims with |mean| > 5: {(dim_means.abs() > 5).sum().item()}")
    logger.info(f"  Dims with |mean| > 1: {(dim_means.abs() > 1).sum().item()}")


def main():
    logger.info("=" * 70)
    logger.info("TRACING PER-DIMENSION MEANS THROUGH PIPELINE")
    logger.info("=" * 70)

    # We need to hook into the encoder to capture intermediate activations
    from llm_dit.encoders.gemma3 import Gemma3Encoder

    logger.info("\nLoading encoder...")
    encoder = Gemma3Encoder.from_pretrained(
        device="auto",
        max_sequence_length=256,
        max_memory={0: "20GiB", "cpu": "32GiB"},
    )

    # Register hooks to capture intermediate values
    captured = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            if isinstance(out, torch.Tensor):
                captured[name] = out.detach().cpu()
        return hook

    # Hook feature extractor
    encoder._feature_extractor.register_forward_hook(make_hook("feature_extractor"))

    # Hook connector output (after connector)
    if encoder._embeddings_connector is not None:
        # Hook the final layer of connector
        encoder._embeddings_connector.register_forward_hook(make_hook("connector_output"))
        # Hook connector input
        encoder._embeddings_connector.register_forward_pre_hook(
            lambda m, inp: captured.update({"connector_input": inp[0].detach().cpu()})
        )

    # Encode test prompt
    test_prompt = "A fluffy orange cat sleeping peacefully on a soft red couch."
    logger.info(f"\nEncoding: '{test_prompt}'")
    output = encoder.encode(test_prompt, return_padded=True)

    # Analyze captured stages
    logger.info("\n" + "=" * 70)
    logger.info("PER-DIMENSION ANALYSIS AT EACH STAGE")
    logger.info("=" * 70)

    if "feature_extractor" in captured:
        analyze_tensor("After feature_extractor (text_proj_in)", captured["feature_extractor"])

    if "connector_input" in captured:
        analyze_tensor("Connector input (before transformer blocks)", captured["connector_input"])

    if "connector_output" in captured:
        if isinstance(captured["connector_output"], tuple):
            analyze_tensor("Connector output (after transformer blocks)", captured["connector_output"][0])
        else:
            analyze_tensor("Connector output (after transformer blocks)", captured["connector_output"])

    # Final output
    analyze_tensor("Final encoder output", output.padded_embeddings)

    # Analysis summary
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 70)

    # Check if extreme means appear after feature_extractor or connector
    if "feature_extractor" in captured:
        fe_means = captured["feature_extractor"][0].float().mean(dim=0)
        if fe_means.abs().max() > 5:
            logger.info("🔴 Extreme per-dim means appear AFTER feature_extractor")
            logger.info("   → Problem is in text_proj_in projection")
        else:
            logger.info("✅ feature_extractor output has reasonable per-dim means")

    if "connector_output" in captured:
        co = captured["connector_output"]
        if isinstance(co, tuple):
            co = co[0]
        co_means = co[0].float().mean(dim=0)
        if co_means.abs().max() > 5:
            logger.info("🔴 Extreme per-dim means appear AFTER connector")
            if "feature_extractor" in captured:
                fe_means = captured["feature_extractor"][0].float().mean(dim=0)
                if fe_means.abs().max() <= 5:
                    logger.info("   → Problem is in connector transformer blocks")
        else:
            logger.info("✅ connector output has reasonable per-dim means")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
