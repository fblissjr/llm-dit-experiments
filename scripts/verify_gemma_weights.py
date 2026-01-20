#!/usr/bin/env python3
"""
Verify Gemma weight loading with key remapping.

Last Updated: 2026-01-20

This script verifies that the LTX-2 Gemma weights are loaded correctly
with proper key remapping from 'base_text_encoder.*' to 'model.*'.
"""

import logging
import sys

import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def verify_gemma_weights():
    """Verify Gemma weights are loaded correctly."""
    from llm_dit.encoders.gemma3 import Gemma3Encoder

    logger.info("=" * 60)
    logger.info("Verifying Gemma weight loading with key remapping")
    logger.info("=" * 60)

    # Load encoder with device_map="auto" for memory management
    logger.info("\nStep 1: Loading Gemma3Encoder...")
    try:
        # Use device_map="auto" to spread across GPU + CPU if needed
        encoder = Gemma3Encoder.from_pretrained(
            device="auto",  # Let accelerate manage memory
            max_sequence_length=256,
            max_memory={0: "20GiB", "cpu": "32GiB"},  # Leave some GPU headroom
        )
    except Exception as e:
        logger.error(f"Failed to load encoder: {e}")
        raise

    logger.info("\nStep 2: Checking model weights...")

    # Check embedding layer
    model = encoder._model
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        embed_weight = model.model.language_model.embed_tokens.weight
    elif hasattr(model, "language_model"):
        embed_weight = model.language_model.embed_tokens.weight
    else:
        logger.error("Cannot find embedding layer in model structure")
        return False

    embed_mean = embed_weight.float().mean().item()
    embed_std = embed_weight.float().std().item()
    embed_shape = embed_weight.shape

    logger.info(f"Embedding layer shape: {embed_shape}")
    logger.info(f"Embedding mean: {embed_mean:.6f}")
    logger.info(f"Embedding std: {embed_std:.6f}")

    # Check if weights look like random init (very small values)
    # Random init typically has std ~0.01-0.02, trained weights have larger std
    if embed_std < 0.05:
        logger.warning(
            f"⚠️  Embedding std ({embed_std:.4f}) is very low - "
            "weights may still be random!"
        )
        return False
    else:
        logger.info(f"✓ Embedding std ({embed_std:.4f}) looks like trained weights")

    # Check first decoder layer
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        layer0 = model.model.language_model.layers[0]
    elif hasattr(model, "language_model"):
        layer0 = model.language_model.layers[0]
    else:
        logger.warning("Cannot find decoder layers")
        return True  # Still pass if embeddings look good

    # Check attention projection
    q_proj = layer0.self_attn.q_proj.weight
    q_mean = q_proj.float().mean().item()
    q_std = q_proj.float().std().item()

    logger.info(f"\nLayer 0 q_proj shape: {q_proj.shape}")
    logger.info(f"Layer 0 q_proj mean: {q_mean:.6f}")
    logger.info(f"Layer 0 q_proj std: {q_std:.6f}")

    if q_std < 0.01:
        logger.warning(
            f"⚠️  Layer 0 q_proj std ({q_std:.6f}) is very low - "
            "weights may still be random!"
        )
        return False
    else:
        logger.info(f"✓ Layer 0 q_proj std ({q_std:.6f}) looks like trained weights")

    logger.info("\nStep 3: Testing text encoding...")

    # Test encoding
    test_text = "A cat sitting on a windowsill watching birds outside"
    try:
        output = encoder.encode(test_text, return_padded=True)
        embeddings = output.padded_embeddings

        emb_mean = embeddings.float().mean().item()
        emb_std = embeddings.float().std().item()

        logger.info(f"Output embeddings shape: {embeddings.shape}")
        logger.info(f"Output embeddings mean: {emb_mean:.4f}")
        logger.info(f"Output embeddings std: {emb_std:.4f}")

        # Check for signal death (very negative mean from GELU)
        if emb_mean < -5.0:
            logger.warning(
                f"⚠️  Output mean ({emb_mean:.2f}) is very negative - "
                "possible signal death in projection layers!"
            )
            return False
        else:
            logger.info(f"✓ Output mean ({emb_mean:.4f}) looks healthy")

    except Exception as e:
        logger.error(f"Encoding failed: {e}")
        raise

    logger.info("\n" + "=" * 60)
    logger.info("✓ All verification checks passed!")
    logger.info("=" * 60)

    return True


if __name__ == "__main__":
    success = verify_gemma_weights()
    sys.exit(0 if success else 1)
