#!/usr/bin/env python3
"""Test Qwen3-4B think token behavior matching DiffSynth-Studio.

This test verifies:
1. How apply_chat_template formats prompts with enable_thinking=True/False
2. Whether <think> tokens appear in generated output
3. How to properly parse thinking content from generation
"""

import argparse
import sys
from pathlib import Path

MODEL_PATH: Path = None  # Set by parse_args()


def test_chat_template_formatting():
    """Test how apply_chat_template formats prompts."""
    from transformers import AutoTokenizer

    print("=" * 60)
    print("TEST: Chat Template Formatting")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))

    messages = [{"role": "user", "content": "A cat sleeping in sunlight"}]

    # Test enable_thinking=True (default, what DiffSynth uses)
    print("\n1. enable_thinking=True (DiffSynth default):")
    print("-" * 40)
    text_thinking = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    print(repr(text_thinking))

    # Test enable_thinking=False
    print("\n2. enable_thinking=False:")
    print("-" * 40)
    text_no_thinking = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    print(repr(text_no_thinking))

    # Check for think tokens
    print("\n3. Analysis:")
    print("-" * 40)
    print(f"  enable_thinking=True has <think>: {'<think>' in text_thinking}")
    print(f"  enable_thinking=False has <think>: {'<think>' in text_no_thinking}")

    # Show the special tokens
    print("\n4. Key token IDs:")
    print("-" * 40)
    think_start = tokenizer.convert_tokens_to_ids("<think>")
    think_end = tokenizer.convert_tokens_to_ids("</think>")
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    print(f"  <think>: {think_start}")
    print(f"  </think>: {think_end}")
    print(f"  <|im_start|>: {im_start}")
    print(f"  <|im_end|>: {im_end}")

    return tokenizer


def test_generation_with_thinking(tokenizer):
    """Test actual generation to see if think tokens appear in output."""
    import torch
    from transformers import AutoModelForCausalLM

    print("\n" + "=" * 60)
    print("TEST: Generation with Thinking")
    print("=" * 60)

    print("\nLoading model (this may take a moment)...")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Simple prompt
    prompt = "What is 2+2?"
    messages = [{"role": "user", "content": prompt}]

    # Format with enable_thinking=True
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    print(f"\nInput prompt: {prompt}")
    print(f"Formatted (truncated): {text[:200]}...")

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    print(f"\nGenerating (max 256 tokens)...")

    # Generate with recommended settings for thinking mode
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        do_sample=True,
    )

    # Get only generated tokens
    generated_ids = outputs[0][input_length:].tolist()

    # Decode WITHOUT skipping special tokens to see <think> tags
    generated_with_special = tokenizer.decode(generated_ids, skip_special_tokens=False)
    generated_without_special = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("\n1. Generated output (WITH special tokens):")
    print("-" * 40)
    print(generated_with_special[:500])
    if len(generated_with_special) > 500:
        print("... (truncated)")

    print("\n2. Generated output (WITHOUT special tokens):")
    print("-" * 40)
    print(generated_without_special[:500])
    if len(generated_without_special) > 500:
        print("... (truncated)")

    # Check for think tokens
    print("\n3. Analysis:")
    print("-" * 40)
    has_think_start = "<think>" in generated_with_special
    has_think_end = "</think>" in generated_with_special
    print(f"  Has <think> in output: {has_think_start}")
    print(f"  Has </think> in output: {has_think_end}")

    # Parse thinking content using Qwen's recommended method
    think_end_token = 151668  # </think>
    try:
        # Find </think> token from the end
        index = len(generated_ids) - generated_ids[::-1].index(think_end_token)
        thinking_content = tokenizer.decode(generated_ids[:index], skip_special_tokens=True).strip()
        final_content = tokenizer.decode(generated_ids[index:], skip_special_tokens=True).strip()
        print(f"\n4. Parsed content (Qwen method):")
        print("-" * 40)
        print(f"  Thinking ({len(thinking_content)} chars): {thinking_content[:200]}...")
        print(f"  Final ({len(final_content)} chars): {final_content[:200]}...")
    except ValueError:
        print(f"\n4. No </think> token (151668) found in output")
        print(f"   Token IDs: {generated_ids[:20]}...")

    return model


def test_diffsynth_style_encoding(tokenizer):
    """Test encoding exactly like DiffSynth does for image generation."""
    print("\n" + "=" * 60)
    print("TEST: DiffSynth-Style Encoding (for embedding extraction)")
    print("=" * 60)

    prompt = "A cat sleeping in warm sunlight, soft fur, peaceful expression"
    messages = [{"role": "user", "content": prompt}]

    # This is exactly what DiffSynth does
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    print(f"\nPrompt: {prompt}")
    print(f"\nFormatted for embedding extraction:")
    print("-" * 40)
    print(formatted)
    print("-" * 40)

    # Tokenize like DiffSynth
    text_inputs = tokenizer(
        [formatted],
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )

    print(f"\nToken count: {text_inputs.attention_mask.sum().item()} / 512")
    print(f"Input IDs shape: {text_inputs.input_ids.shape}")

    # Show the actual tokens
    tokens = tokenizer.convert_ids_to_tokens(text_inputs.input_ids[0][:50])
    print(f"\nFirst 50 tokens:")
    print(tokens)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test Qwen3-4B think token behavior matching DiffSynth-Studio"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to Qwen3-4B model directory",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Run generation test (slower, requires GPU)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set global model path
    MODEL_PATH = args.model_path.expanduser()
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        sys.exit(1)

    print("Qwen3-4B Think Token Test")
    print("Model path:", MODEL_PATH)
    print()

    # Test 1: Template formatting
    tokenizer = test_chat_template_formatting()

    # Test 2: DiffSynth-style encoding
    test_diffsynth_style_encoding(tokenizer)

    # Test 3: Actual generation (optional, slower)
    if args.generate:
        test_generation_with_thinking(tokenizer)
    else:
        print("\n" + "=" * 60)
        print("Skipping generation test (use --generate to enable)")
        print("=" * 60)

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
