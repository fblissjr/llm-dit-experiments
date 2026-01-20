#!/usr/bin/env python3
"""
Verify tokenizer fix for LTX-2 signal death issue.

Last Updated: 2026-01-20

This script compares:
1. Token IDs between HuggingFace and local LTX-2 tokenizer
2. Vocabulary sizes and special tokens
3. Validates the fix addresses the tokenizer mismatch hypothesis

Expected behavior:
- If token IDs DIFFER: The fix is critical (wrong tokens caused signal death)
- If token IDs MATCH: Tokenizer wasn't the issue, investigate elsewhere
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def compare_tokenizers():
    """Compare HuggingFace vs local LTX-2 tokenizer."""
    from transformers import AutoTokenizer

    print("=" * 60)
    print("TOKENIZER COMPARISON: HuggingFace vs Local LTX-2")
    print("=" * 60)

    # Load both tokenizers
    print("\n[1] Loading tokenizers...")

    try:
        hf_tok = AutoTokenizer.from_pretrained(
            "google/gemma-3-12b-it-qat-q4_0-unquantized"
        )
        print(f"  HuggingFace tokenizer loaded: vocab_size={hf_tok.vocab_size}")
    except Exception as e:
        print(f"  ERROR loading HuggingFace tokenizer: {e}")
        hf_tok = None

    try:
        local_tok = AutoTokenizer.from_pretrained(
            "models/LTX-2/text_encoder",
            local_files_only=True,
        )
        print(f"  Local LTX-2 tokenizer loaded: vocab_size={local_tok.vocab_size}")
    except Exception as e:
        print(f"  ERROR loading local tokenizer: {e}")
        local_tok = None

    if hf_tok is None or local_tok is None:
        print("\nCannot compare - one or both tokenizers failed to load")
        return

    # Compare vocabulary sizes
    print("\n[2] Vocabulary comparison...")
    print(f"  HuggingFace vocab_size: {hf_tok.vocab_size}")
    print(f"  Local vocab_size: {local_tok.vocab_size}")
    if hf_tok.vocab_size != local_tok.vocab_size:
        print("  ⚠️  VOCAB SIZE MISMATCH - tokenizers are different!")
    else:
        print("  ✓ Vocab sizes match")

    # Compare special tokens
    print("\n[3] Special tokens comparison...")
    special_tokens = ["bos_token", "eos_token", "pad_token", "unk_token"]
    for tok_name in special_tokens:
        hf_val = getattr(hf_tok, tok_name, None)
        local_val = getattr(local_tok, tok_name, None)
        match = "✓" if hf_val == local_val else "⚠️ DIFF"
        print(f"  {tok_name}: HF={repr(hf_val)} vs Local={repr(local_val)} [{match}]")

    # Test prompts
    test_prompts = [
        "A fluffy orange cat sleeping peacefully on a soft red couch.",
        "A woman walking her dog in Central Park on a sunny autumn day.",
        "Cinematic shot of a rocket launching into space with dramatic lighting.",
    ]

    print("\n[4] Token ID comparison on test prompts...")
    print("-" * 60)

    all_match = True
    for prompt in test_prompts:
        hf_ids = hf_tok(prompt).input_ids
        local_ids = local_tok(prompt).input_ids

        match = hf_ids == local_ids
        all_match = all_match and match

        status = "✓ MATCH" if match else "⚠️ MISMATCH"
        print(f"\nPrompt: \"{prompt[:50]}...\"")
        print(f"  Status: {status}")
        print(f"  Length: HF={len(hf_ids)}, Local={len(local_ids)}")

        if not match:
            # Show first 10 tokens for comparison
            print(f"  HF first 10: {hf_ids[:10]}")
            print(f"  Local first 10: {local_ids[:10]}")

            # Count differing positions
            min_len = min(len(hf_ids), len(local_ids))
            diffs = sum(1 for i in range(min_len) if hf_ids[i] != local_ids[i])
            print(f"  Differing positions: {diffs}/{min_len} ({100*diffs/min_len:.1f}%)")

    print("\n" + "=" * 60)
    if all_match:
        print("RESULT: Token IDs MATCH - tokenizer was NOT the issue")
        print("        Signal death must have another cause")
        print("        (But local tokenizer is still more correct to use)")
    else:
        print("RESULT: Token IDs DIFFER - tokenizer WAS the issue!")
        print("        This fix should address the signal death problem")
        print("        Re-run audit_connector_interface.py to verify")
    print("=" * 60)


def verify_weight_paths():
    """Verify the weight loading paths are correct."""
    print("\n" + "=" * 60)
    print("WEIGHT PATH VERIFICATION")
    print("=" * 60)

    from llm_dit.encoders.gemma3 import (
        DEFAULT_TOKENIZER_PATH,
        DEFAULT_CONNECTOR_WEIGHTS_SHARD,
        DEFAULT_CONNECTORS_CONFIG,
    )

    print("\n[1] Configured paths:")
    print(f"  Tokenizer: {DEFAULT_TOKENIZER_PATH}")
    print(f"  Connector weights: {DEFAULT_CONNECTOR_WEIGHTS_SHARD}")
    print(f"  Connector config: {DEFAULT_CONNECTORS_CONFIG}")

    print("\n[2] Path existence check:")
    paths = {
        "Tokenizer dir": Path(DEFAULT_TOKENIZER_PATH),
        "Tokenizer model": Path(DEFAULT_TOKENIZER_PATH) / "tokenizer.model",
        "Connector weights": Path(DEFAULT_CONNECTOR_WEIGHTS_SHARD),
        "Connector config": Path(DEFAULT_CONNECTORS_CONFIG),
    }

    for name, path in paths.items():
        exists = path.exists()
        status = "✓ exists" if exists else "✗ MISSING"
        print(f"  {name}: {status}")
        if not exists:
            print(f"    Expected at: {path}")


if __name__ == "__main__":
    print("\nLTX-2 Tokenizer Fix Verification")
    print("Purpose: Verify the tokenizer mismatch hypothesis\n")

    verify_weight_paths()
    compare_tokenizers()
