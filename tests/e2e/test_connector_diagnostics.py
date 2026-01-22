"""
Connector diagnostics test for debugging Block 0 explosion.

Last Updated: 2026-01-20

This test implements Gemini's recommended diagnostic checks to find the cause
of the 3000x per-dimension range explosion in the Embeddings1DConnector.

Key Checks:
1. Weight magnitudes - std > 1.0 indicates massive weights bug
2. RoPE values - should be in [-1, 1] range
3. Token segmentation - text vs register tokens analyzed separately
4. Per-stage per-dim range tracking

Usage:
    pytest tests/e2e/test_connector_diagnostics.py -v --runslow

Output:
    outputs/tests/runs/{backend}_test_connector_diagnostics_{timestamp}/
    ├── connector_diagnostics.json   # All statistics
    ├── generation.log               # Full log
    └── debug.log                    # Debug trace
"""

import logging
from pathlib import Path

import pytest
import torch

logger = logging.getLogger(__name__)


# Skip if backends not available
pytest.importorskip("tests.backends")


@pytest.fixture
def llm_dit_backend_only():
    """Force llm_dit backend for this test (diagnostics only available there)."""
    from tests.backends import is_llm_dit_available
    from tests.backends.llm_dit_backend import LLMDitBackend

    if not is_llm_dit_available():
        pytest.skip("llm_dit backend not available")

    return LLMDitBackend()


@pytest.mark.e2e
@pytest.mark.slow
def test_connector_diagnostics(llm_dit_backend_only, output_dir, smoke_prompt):
    """
    Run Gemini's recommended checks to find the 3000x explosion cause.

    This test doesn't generate a full video - it just encodes the prompt
    and captures connector diagnostics.
    """
    backend = llm_dit_backend_only

    logger.info("=" * 60)
    logger.info("CONNECTOR DIAGNOSTICS TEST")
    logger.info("=" * 60)
    logger.info(f"Prompt: {smoke_prompt}")
    logger.info(f"Output dir: {output_dir}")

    # Run text encoding with diagnostics enabled
    embeddings = backend.encode_text(
        prompt=smoke_prompt,
        output_dir=output_dir,
        debug_trace=True,
    )

    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(f"Embeddings dtype: {embeddings.dtype}")

    # Load diagnostics saved during encoding
    from tests.backends.diagnostics import load_diagnostics

    diagnostics_path = output_dir / "connector_diagnostics.json"
    assert diagnostics_path.exists(), f"Diagnostics file not found at {diagnostics_path}"

    diagnostics = load_diagnostics(diagnostics_path)

    # Print summary for human review
    print("\n" + diagnostics.summary())

    # Check for anomalies (automated Gemini checks)
    warnings = diagnostics.check_for_anomalies()
    for w in warnings:
        print(f"⚠️  {w}")
        logger.warning(f"ANOMALY: {w}")

    # =========================================================================
    # GEMINI CHECK 1: Weight magnitudes
    # =========================================================================
    logger.info("=" * 40)
    logger.info("CHECK 1: Weight Magnitudes")
    logger.info("=" * 40)

    for layer_name, stats in diagnostics.weight_stats.items():
        std = stats.get('std', 0)
        abs_max = stats.get('abs_max', 0)
        logger.info(f"  {layer_name}: std={std:.4f}, abs_max={abs_max:.4f}")

        # Soft assertion - log warning but don't fail
        if std > 2.0:
            logger.error(f"  MASSIVE WEIGHT: {layer_name} has std={std:.4f} (expected < 1.0)")

    # =========================================================================
    # GEMINI CHECK 2: RoPE ranges
    # =========================================================================
    logger.info("=" * 40)
    logger.info("CHECK 2: RoPE Values")
    logger.info("=" * 40)

    cos_min, cos_max = diagnostics.rope_cos_range
    sin_min, sin_max = diagnostics.rope_sin_range

    logger.info(f"  cos range: [{cos_min:.4f}, {cos_max:.4f}]")
    logger.info(f"  sin range: [{sin_min:.4f}, {sin_max:.4f}]")

    # Soft assertion
    if abs(cos_min) > 1.5 or abs(cos_max) > 1.5:
        logger.error(f"  BROKEN ROPE: cos out of expected [-1, 1] range")
    if abs(sin_min) > 1.5 or abs(sin_max) > 1.5:
        logger.error(f"  BROKEN ROPE: sin out of expected [-1, 1] range")

    # =========================================================================
    # GEMINI CHECK 3: Token segmentation
    # =========================================================================
    logger.info("=" * 40)
    logger.info("CHECK 3: Token Segmentation")
    logger.info("=" * 40)

    logger.info(f"  Text tokens: {diagnostics.num_text_tokens}")
    logger.info(f"  Register tokens: {diagnostics.num_register_tokens}")

    if diagnostics.text_tokens_stats:
        text_per_dim = diagnostics.text_tokens_stats.get('per_dim_range', 0)
        logger.info(f"  Text per-dim range: {text_per_dim:.4f}")
        if text_per_dim > 5.0:
            logger.error(f"  TEXT TOKEN ISSUE: per_dim_range={text_per_dim:.4f} > 5.0")

    if diagnostics.register_tokens_stats:
        reg_per_dim = diagnostics.register_tokens_stats.get('per_dim_range', 0)
        reg_mean = diagnostics.register_tokens_stats.get('mean', 0)
        logger.info(f"  Register per-dim range: {reg_per_dim:.4f}")
        logger.info(f"  Register mean: {reg_mean:.4f}")

    # =========================================================================
    # KEY METRIC: Per-stage explosion tracking
    # =========================================================================
    logger.info("=" * 40)
    logger.info("KEY METRIC: Per-Stage Per-Dim Range")
    logger.info("=" * 40)

    explosion_detected = False
    explosion_stage = None

    for stage, value in sorted(diagnostics.per_dim_range_by_stage.items()):
        flag = ""
        if value > 100:
            flag = " <-- EXPLOSION!"
            if not explosion_detected:
                explosion_detected = True
                explosion_stage = stage
        logger.info(f"  {stage}: {value:.2f}{flag}")

    # =========================================================================
    # Final Analysis
    # =========================================================================
    logger.info("=" * 40)
    logger.info("ANALYSIS")
    logger.info("=" * 40)

    if explosion_detected:
        logger.error(f"EXPLOSION DETECTED at stage: {explosion_stage}")
        logger.error(
            "If all weight/RoPE checks passed, the issue is in the forward pass logic."
        )
        logger.error(
            "Next steps: Add intermediate logging to the connector's forward() method."
        )

        # Identify the most likely cause
        has_weight_issue = any(
            stats.get('std', 0) > 1.0
            for stats in diagnostics.weight_stats.values()
        )
        has_rope_issue = (
            abs(cos_min) > 1.5 or abs(cos_max) > 1.5 or
            abs(sin_min) > 1.5 or abs(sin_max) > 1.5
        )

        if has_weight_issue:
            logger.error("CAUSE: Massive weight magnitudes detected")
        elif has_rope_issue:
            logger.error("CAUSE: Broken RoPE frequency computation")
        else:
            logger.error("CAUSE: Forward pass logic issue (weights and RoPE look OK)")

        # Don't fail the test - this is a diagnostic tool
        print("\n" + "=" * 60)
        print("DIAGNOSTIC COMPLETE - SEE LOG FOR ANALYSIS")
        print("=" * 60)
    else:
        logger.info("No explosion detected - connector appears healthy!")
        print("\n✅ Connector diagnostics passed - no explosion detected")

    # Always pass - this is a diagnostic test, not a validation test
    # The purpose is to gather information, not to fail CI


@pytest.mark.e2e
@pytest.mark.slow
def test_connector_block_breakdown(llm_dit_backend_only, output_dir, smoke_prompt):
    """
    Detailed block-by-block analysis of the connector.

    Captures statistics after each transformer block to pinpoint
    exactly where the explosion occurs.
    """
    backend = llm_dit_backend_only

    logger.info("=" * 60)
    logger.info("CONNECTOR BLOCK BREAKDOWN TEST")
    logger.info("=" * 60)

    # Run text encoding with diagnostics enabled
    embeddings = backend.encode_text(
        prompt=smoke_prompt,
        output_dir=output_dir,
        debug_trace=True,
    )

    # Load diagnostics
    from tests.backends.diagnostics import load_diagnostics

    diagnostics = load_diagnostics(output_dir / "connector_diagnostics.json")

    # Print block-by-block breakdown
    print("\n" + "=" * 60)
    print("BLOCK-BY-BLOCK BREAKDOWN")
    print("=" * 60)

    for block_name, stats in sorted(diagnostics.block_stats.items()):
        per_dim_range = stats.get('per_dim_range', 0)
        mean = stats.get('mean', 0)
        std = stats.get('std', 0)

        flag = " ⚠️" if per_dim_range > 100 else ""
        print(f"{block_name}:{flag}")
        print(f"  per_dim_range: {per_dim_range:.2f}")
        print(f"  mean: {mean:.4f}")
        print(f"  std: {std:.4f}")
        print()

    print("=" * 60)


@pytest.mark.e2e
def test_connector_diagnostics_quick(llm_dit_backend_only, tmp_output_dir, smoke_prompt):
    """
    Quick version of connector diagnostics (no --runslow required).

    Just verifies the diagnostic collection works without deep analysis.
    """
    backend = llm_dit_backend_only

    # Run encoding with diagnostics
    embeddings = backend.encode_text(
        prompt=smoke_prompt,
        output_dir=tmp_output_dir,
        debug_trace=True,
    )

    # Verify diagnostics file was created
    diagnostics_path = tmp_output_dir / "connector_diagnostics.json"
    assert diagnostics_path.exists()

    # Load and do basic validation
    from tests.backends.diagnostics import load_diagnostics

    diagnostics = load_diagnostics(diagnostics_path)

    # Basic sanity checks
    assert len(diagnostics.weight_stats) > 0, "No weight stats collected"
    assert len(diagnostics.per_dim_range_by_stage) > 0, "No per-stage stats collected"

    # Verify some expected stages are present
    assert "input" in diagnostics.per_dim_range_by_stage
    assert "final_output" in diagnostics.per_dim_range_by_stage

    logger.info("Quick diagnostics test passed")
    logger.info(f"Collected {len(diagnostics.weight_stats)} weight stats")
    logger.info(f"Collected {len(diagnostics.per_dim_range_by_stage)} stage stats")
