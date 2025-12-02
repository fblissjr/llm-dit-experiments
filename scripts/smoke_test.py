#!/usr/bin/env python3
"""
Smoke test to validate the llm-dit-experiments setup.

This script tests:
1. Package imports work
2. Template loading works
3. Conversation formatting works
4. (Optional) Backend loading works if model path provided

Usage:
    uv run scripts/smoke_test.py
    uv run scripts/smoke_test.py --model-path /path/to/z-image
"""

import argparse
import sys
from pathlib import Path


def test_imports():
    """Test that all package imports work."""
    print("Testing imports...")

    try:
        from llm_dit.backends import TextEncoderBackend, BackendConfig, EncodingOutput
        from llm_dit.conversation import Message, Conversation, Role, Qwen3Formatter
        from llm_dit.templates import Template, TemplateRegistry, load_template

        print("  [OK] All imports successful")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_templates():
    """Test template loading."""
    print("Testing template loading...")

    from llm_dit.templates import TemplateRegistry

    templates_dir = Path(__file__).parent.parent / "templates" / "z_image"
    if not templates_dir.exists():
        print(f"  [SKIP] Templates directory not found: {templates_dir}")
        return True

    registry = TemplateRegistry.from_directory(templates_dir)
    count = len(registry)
    print(f"  [OK] Loaded {count} templates")

    # Test a specific template
    if "photorealistic" in registry:
        template = registry["photorealistic"]
        print(f"  [OK] 'photorealistic' template loaded")
        print(f"       - add_think_block: {template.add_think_block}")
        print(f"       - category: {template.category}")
        if template.thinking_content:
            preview = template.thinking_content[:50].replace("\n", " ")
            print(f"       - thinking preview: {preview}...")
    else:
        print("  [WARN] 'photorealistic' template not found")

    return count > 0


def test_conversation():
    """Test conversation formatting."""
    print("Testing conversation formatting...")

    from llm_dit.conversation import Conversation, Qwen3Formatter

    # Create a simple conversation
    conv = Conversation.simple(
        user_prompt="A cat sleeping on a windowsill, afternoon sunlight",
        system_prompt="Generate photorealistic images with natural lighting.",
        thinking_content="I should focus on soft shadows and warm golden hour tones.",
        enable_thinking=True,
    )

    formatter = Qwen3Formatter()
    formatted = formatter.format(conv)

    # Verify key tokens are present
    checks = [
        ("<|im_start|>system", "system start"),
        ("<|im_end|>", "im_end token"),
        ("<|im_start|>user", "user start"),
        ("<|im_start|>assistant", "assistant start"),
        ("<think>", "think start"),
        ("</think>", "think end"),
        ("A cat sleeping", "user prompt content"),
        ("soft shadows", "thinking content"),
    ]

    all_ok = True
    for token, desc in checks:
        if token in formatted:
            print(f"  [OK] Found {desc}")
        else:
            print(f"  [FAIL] Missing {desc}")
            all_ok = False

    # Print formatted output for inspection
    print("\n  Formatted output preview:")
    for line in formatted.split("\n")[:15]:
        print(f"    {line}")
    if formatted.count("\n") > 15:
        print(f"    ... ({formatted.count(chr(10)) - 15} more lines)")

    return all_ok


def test_backend(model_path: str | None):
    """Test backend loading (optional, requires model)."""
    if not model_path:
        print("Testing backend... [SKIP] No model path provided")
        return True

    print(f"Testing backend with model: {model_path}")

    try:
        from llm_dit.backends.transformers import TransformersBackend

        print("  Loading model (this may take a moment)...")
        backend = TransformersBackend.from_pretrained(model_path)
        print(f"  [OK] Model loaded")
        print(f"       - embedding_dim: {backend.embedding_dim}")
        print(f"       - max_length: {backend.max_sequence_length}")
        print(f"       - device: {backend.device}")
        print(f"       - dtype: {backend.dtype}")

        # Test encoding
        from llm_dit.conversation import format_prompt

        formatted = format_prompt(
            user_prompt="A cat",
            system_prompt="Generate images.",
            enable_thinking=True,
        )

        print("  Testing encoding...")
        output = backend.encode([formatted])
        embeds = output.embeddings[0]
        print(f"  [OK] Encoding successful")
        print(f"       - sequence length: {embeds.shape[0]}")
        print(f"       - embedding shape: {embeds.shape}")

        return True

    except Exception as e:
        print(f"  [FAIL] Backend error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Smoke test for llm-dit-experiments")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to Z-Image model for backend testing",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("llm-dit-experiments Smoke Test")
    print("=" * 60)
    print()

    results = []
    results.append(("Imports", test_imports()))
    results.append(("Templates", test_templates()))
    results.append(("Conversation", test_conversation()))
    results.append(("Backend", test_backend(args.model_path)))

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
