#!/usr/bin/env python3
"""
LTX-2 Port Validation Script

Last Updated: 2026-01-15

Tests that the LTX-2 port correctly:
1. Extracts hidden states from all Gemma layers
2. Projects them through the feature extractor
3. Returns valid EncodingOutput structure
4. Exposes layer_stack for routing experiments
5. Can generate video (basic smoke test)

Usage:
    uv run python scripts/validate_ltx2_port.py
    uv run python scripts/validate_ltx2_port.py --skip-pipeline  # Skip slow pipeline test
"""

import argparse
import sys
from pathlib import Path

import torch


def test_encoding_output_structure():
    """Test that EncodingOutput has correct fields."""
    print("=" * 60)
    print("Test 0: EncodingOutput Structure")
    print("=" * 60)

    from llm_dit.backends.protocol import EncodingOutput

    # Check required fields exist
    required_fields = [
        "embeddings",
        "attention_masks",
        "padded_embeddings",
        "padded_mask",
        "formatted_prompts",
        "token_counts",
        "layer_stack",  # New field for routing experiments
    ]

    for field in required_fields:
        assert hasattr(EncodingOutput, "__dataclass_fields__"), (
            "EncodingOutput should be a dataclass"
        )
        assert field in EncodingOutput.__dataclass_fields__, f"Missing field: {field}"
        print(f"  [OK] Field '{field}' present")

    print("  [PASS] EncodingOutput structure correct")
    return True


def test_encoder_shapes(model_id: str = "models/LTX-2/text_encoder"):
    """Test that encoder produces correct shapes."""
    print("\n" + "=" * 60)
    print("Test 1: Encoder Shapes")
    print("=" * 60)

    from llm_dit.encoders.gemma3 import (
        GEMMA3_HIDDEN_DIM,
        GEMMA3_NUM_LAYERS,
        Gemma3Encoder,
    )

    print(f"  Loading encoder: {model_id}")
    print("  Using 8-bit quantization with CPU offload for memory efficiency...")

    # Limit GPU memory to force CPU offloading for larger models
    # This ensures the model fits in 24GB VRAM with headroom for activations
    max_memory = {0: "18GiB", "cpu": "32GiB"}

    encoder = Gemma3Encoder.from_pretrained(
        model_id,
        device="auto",  # Let accelerate handle device placement
        dtype="bfloat16",
        max_sequence_length=128,
        quantization="8bit",  # Use 8-bit to fit in 24GB VRAM
        max_memory=max_memory,  # Force CPU offloading for overflow
    )

    # Test encode()
    print("  Testing encode()...")
    output = encoder.encode(["A cat sleeping on a sunny windowsill"])

    assert isinstance(output.embeddings, list), "embeddings should be list"
    assert len(output.embeddings) == 1, "should have 1 embedding"
    assert output.embeddings[0].shape[-1] == GEMMA3_HIDDEN_DIM, (
        f"hidden dim should be {GEMMA3_HIDDEN_DIM}, got {output.embeddings[0].shape[-1]}"
    )
    assert len(output.attention_masks) == 1, "should have 1 mask"
    assert output.token_counts is not None, "should have token_counts"
    assert len(output.token_counts) == 1, "should have 1 token count"

    print(f"  embeddings[0] shape: {output.embeddings[0].shape}")
    print(f"  attention_masks[0] shape: {output.attention_masks[0].shape}")
    print(f"  token_counts: {output.token_counts}")
    print("  [PASS] Encoder shapes correct")

    return encoder


def test_multilayer_extraction(encoder):
    """Test multi-layer hidden state extraction."""
    print("\n" + "=" * 60)
    print("Test 2: Multi-Layer Extraction")
    print("=" * 60)

    from llm_dit.encoders.gemma3 import GEMMA3_HIDDEN_DIM, GEMMA3_NUM_LAYERS

    # Test full layer extraction
    print("  Testing encode_multilayer() with all layers...")
    result = encoder.encode_multilayer(
        ["A detailed portrait of an elderly woman"],
        return_projected=True,
    )

    assert "layer_stack" in result, "should have layer_stack"
    assert "attention_mask" in result, "should have attention_mask"
    assert "projected" in result, "should have projected"
    assert "seq_lengths" in result, "should have seq_lengths"

    layer_stack = result["layer_stack"]
    # Note: QAT model has 48 layers, full model has 49 (48 decoder + embedding)
    # Accept either as valid
    actual_layers = layer_stack.shape[-1]
    assert actual_layers >= 36, f"should have at least 36 layers, got {actual_layers}"
    assert layer_stack.shape[2] == GEMMA3_HIDDEN_DIM, (
        f"hidden dim should be {GEMMA3_HIDDEN_DIM}, got {layer_stack.shape[2]}"
    )

    print(f"  layer_stack shape: {layer_stack.shape}")
    print(f"  Actual layers: {actual_layers} (QAT has 48, full has 49)")
    print("  [PASS] Full layer extraction correct")

    # Test selective layer extraction
    print("\n  Testing encode_multilayer() with selected layers...")
    # Use layers that exist in both 48-layer QAT and 49-layer full model
    selected_layers = [10, 20, 30, 40, 47]  # Max index 47 for 48-layer model
    result_selective = encoder.encode_multilayer(
        ["A cat"],
        layer_indices=selected_layers,
        return_projected=True,
    )

    assert result_selective["layer_stack"].shape[-1] == len(selected_layers), (
        f"should have {len(selected_layers)} selected layers"
    )

    print(f"  Selective layer_stack shape: {result_selective['layer_stack'].shape}")
    print(f"  Selected layers: {selected_layers}")
    print("  [PASS] Selective layer extraction correct")

    return result


def test_layer_variance(result):
    """Test that different layers have different representations."""
    print("\n" + "=" * 60)
    print("Test 3: Layer Variance")
    print("=" * 60)

    layer_stack = result["layer_stack"]  # [B, T, D, L]

    # Compare early vs late layers
    early = layer_stack[..., :10].mean(dim=-1)  # Average of first 10 layers
    late = layer_stack[..., -10:].mean(dim=-1)  # Average of last 10 layers

    diff = (early - late).abs().mean()
    cosine = torch.nn.functional.cosine_similarity(early.flatten(), late.flatten(), dim=0)

    print(f"  Early vs Late layer mean diff: {diff:.4f}")
    print(f"  Cosine similarity: {cosine:.4f}")

    if cosine > 0.99:
        print("  [WARN] Layers very similar - check hidden states extraction")
        return False
    else:
        print("  [PASS] Layers show expected variance")
        return True


def test_projection_shapes(encoder):
    """Test that projection produces correct output shapes."""
    print("\n" + "=" * 60)
    print("Test 4: Projection Shapes")
    print("=" * 60)

    from llm_dit.encoders.gemma3 import GEMMA3_OUTPUT_DIM

    result = encoder.encode_multilayer(
        ["Test prompt"],
        return_projected=True,
    )

    projected = result["projected"]
    assert projected is not None, "projected should not be None"
    assert projected.shape[-1] == GEMMA3_OUTPUT_DIM, (
        f"projected dim should be {GEMMA3_OUTPUT_DIM}, got {projected.shape[-1]}"
    )

    print(f"  projected shape: {projected.shape}")
    print(f"  Expected: [1, seq_len, {GEMMA3_OUTPUT_DIM}]")
    print("  [PASS] Projection shapes correct")
    return True


def test_pipeline_smoke(model_path: str = "models/LTX-2"):
    """Basic smoke test for full pipeline."""
    print("\n" + "=" * 60)
    print("Test 5: Pipeline Smoke Test")
    print("=" * 60)

    model_path = Path(model_path).expanduser()
    if not model_path.exists():
        print(f"  [SKIP] Model path not found: {model_path}")
        return None

    try:
        from llm_dit.pipelines.ltx2 import LTX2Pipeline

        print("  Loading pipeline (this may take a while)...")
        pipe = LTX2Pipeline.from_pretrained(
            str(model_path),
            dtype=torch.bfloat16,
            enable_cpu_offload=True,
        )

        # Minimal generation test
        print("  Running minimal generation test...")
        output = pipe(
            prompt="A cat",
            num_inference_steps=2,
            height=256,
            width=256,
            num_frames=5,
        )

        assert output.frames is not None, "should produce frames"
        print(f"  Generated frames shape: {output.frames.shape}")
        print("  [PASS] Pipeline smoke test passed")
        return True

    except Exception as e:
        print(f"  [FAIL] Pipeline test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_generate_with_embeddings_api():
    """Test that generate_with_embeddings method exists and has correct signature."""
    print("\n" + "=" * 60)
    print("Test 6: generate_with_embeddings API")
    print("=" * 60)

    import inspect

    from llm_dit.pipelines.ltx2 import LTX2Pipeline

    # Check method exists
    assert hasattr(LTX2Pipeline, "generate_with_embeddings"), (
        "LTX2Pipeline should have generate_with_embeddings method"
    )

    # Check signature
    sig = inspect.signature(LTX2Pipeline.generate_with_embeddings)
    params = list(sig.parameters.keys())

    required_params = ["prompt_embeds", "attention_mask"]
    for param in required_params:
        assert param in params, f"Missing parameter: {param}"
        print(f"  [OK] Parameter '{param}' present")

    print("  [PASS] generate_with_embeddings API correct")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate LTX-2 port")
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Skip slow pipeline smoke test",
    )
    parser.add_argument(
        "--model-path",
        default="models/LTX-2",
        help="Path to LTX-2 model files",
    )
    parser.add_argument(
        "--encoder-model",
        default="models/LTX-2/text_encoder",
        help="Gemma model for encoder tests. Default: LTX-2 compatible 12B model with CPU offloading.",
    )
    args = parser.parse_args()

    print("\nLTX-2 Port Validation")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    results = {}

    # Test 0: EncodingOutput structure
    try:
        results["encoding_output"] = test_encoding_output_structure()
    except Exception as e:
        print(f"  [FAIL] {e}")
        results["encoding_output"] = False

    # Test 1: Encoder shapes
    encoder = None
    try:
        encoder = test_encoder_shapes(model_id=args.encoder_model)
        results["encoder_shapes"] = True
    except Exception as e:
        print(f"  [FAIL] {e}")
        results["encoder_shapes"] = False
        import traceback

        traceback.print_exc()

    if encoder is not None:
        # Test 2: Multi-layer extraction
        try:
            result = test_multilayer_extraction(encoder)
            results["multilayer"] = True
        except Exception as e:
            print(f"  [FAIL] {e}")
            results["multilayer"] = False
            result = None

        # Test 3: Layer variance
        if result is not None:
            try:
                results["layer_variance"] = test_layer_variance(result)
            except Exception as e:
                print(f"  [FAIL] {e}")
                results["layer_variance"] = False

        # Test 4: Projection shapes
        try:
            results["projection"] = test_projection_shapes(encoder)
        except Exception as e:
            print(f"  [FAIL] {e}")
            results["projection"] = False

    # Test 5: Pipeline smoke (optional)
    if not args.skip_pipeline:
        try:
            results["pipeline"] = test_pipeline_smoke(args.model_path)
        except Exception as e:
            print(f"  [FAIL] {e}")
            results["pipeline"] = False
    else:
        print("\n[SKIP] Pipeline smoke test (--skip-pipeline)")
        results["pipeline"] = None

    # Test 6: API check
    try:
        results["api"] = test_generate_with_embeddings_api()
    except Exception as e:
        print(f"  [FAIL] {e}")
        results["api"] = False
        results["api"] = False

    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)

    passed = 0
    failed = 0
    skipped = 0

    for test_name, result in results.items():
        if result is True:
            status = "[PASS]"
            passed += 1
        elif result is False:
            status = "[FAIL]"
            failed += 1
        else:
            status = "[SKIP]"
            skipped += 1
        print(f"  {status} {test_name}")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        print("\nSome tests failed. Check output above for details.")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
