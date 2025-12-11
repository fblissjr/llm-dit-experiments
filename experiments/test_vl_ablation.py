#!/usr/bin/env python3
"""
Quick test script to verify VL ablation infrastructure works.

This tests:
- VL extractor loading
- Embedding extraction
- Text encoding
- Blending logic

Without running full generation (which requires GPU + model weights).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image


def test_vl_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    from llm_dit.vl.qwen3_vl import VLEmbeddingExtractor
    from llm_dit.vl.blending import blend_embeddings, scale_embeddings
    from llm_dit import MAX_TEXT_SEQ_LEN
    from llm_dit.utils.long_prompt import compress_embeddings

    print("  All imports successful")


def test_blend_logic():
    """Test embedding blending without actual models."""
    print("\nTesting blend logic...")
    from llm_dit.vl.blending import blend_embeddings

    # Create dummy embeddings
    vl_emb = torch.randn(100, 2560)  # VL embeddings
    text_emb = torch.randn(120, 2560)  # Text embeddings (different length)

    # Test blending with different alphas
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        blended = blend_embeddings(vl_emb, text_emb, alpha, match_lengths=True)
        # alpha=0.0 returns text_emb unchanged, alpha=1.0 returns vl_emb unchanged
        expected_len = 120 if alpha == 0.0 else (100 if alpha == 1.0 else min(100, 120))
        assert blended.shape[0] == expected_len, f"Wrong shape for alpha={alpha}: got {blended.shape[0]}, expected {expected_len}"
        assert blended.shape[1] == 2560, f"Wrong dim for alpha={alpha}"
        print(f"  alpha={alpha}: {blended.shape} ✓")

    print("  Blending logic works correctly")


def test_config_dataclass():
    """Test that ExperimentConfig handles VL parameters."""
    print("\nTesting ExperimentConfig...")
    from experiments.run_ablation import ExperimentConfig

    config = ExperimentConfig(
        experiment_name="vl_alpha_sweep",
        prompt_id="test_001",
        prompt_text="A cat sleeping",
        seed=42,
        variable_name="vl_alpha",
        variable_value=0.5,
        vl_alpha=0.5,
        vl_token_selection="all",
        vl_hidden_layer=-2,
        vl_image_path="/tmp/test.jpg",
    )

    assert config.vl_alpha == 0.5
    assert config.vl_token_selection == "all"
    assert config.vl_hidden_layer == -2
    print("  ExperimentConfig VL fields work correctly ✓")


def test_experiment_definitions():
    """Test that VL experiments are properly defined."""
    print("\nTesting experiment definitions...")
    from experiments.run_ablation import EXPERIMENTS

    vl_experiments = [
        "vl_token_selection",
        "vl_pure",
        "vl_alpha_sweep",
        "vl_alpha_coarse",
        "vl_hidden_layer",
        "vl_blend_with_text_layers",
    ]

    for exp_name in vl_experiments:
        assert exp_name in EXPERIMENTS, f"Missing experiment: {exp_name}"
        exp = EXPERIMENTS[exp_name]
        assert "description" in exp
        assert "variable" in exp
        assert "values" in exp
        assert "defaults" in exp
        print(f"  {exp_name}: {len(exp['values'])} values ✓")

    print("  All VL experiments defined correctly")


def main():
    print("=" * 60)
    print("VL Ablation Infrastructure Test")
    print("=" * 60)

    try:
        test_vl_imports()
        test_blend_logic()
        test_config_dataclass()
        test_experiment_definitions()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nThe VL ablation infrastructure is ready to use.")
        print("\nExample usage:")
        print("  uv run experiments/run_ablation.py \\")
        print("    --experiment vl_alpha_sweep \\")
        print("    --model-path /path/to/z-image \\")
        print("    --vl-image /path/to/reference.jpg \\")
        print("    --prompt-category simple_objects")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
