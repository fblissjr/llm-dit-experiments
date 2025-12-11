#!/usr/bin/env python3
"""Verify the token IDs for think block injection."""

from pathlib import Path
from transformers import AutoTokenizer

# Find Qwen3-4B
qwen_path = None
for candidate in [
    Path.home() / "Storage" / "Qwen3-4B",
    Path.home() / "models" / "Qwen3-4B",
]:
    if candidate.exists():
        qwen_path = str(candidate)
        break

if not qwen_path:
    print("Could not find Qwen3-4B")
    exit(1)

tokenizer = AutoTokenizer.from_pretrained(qwen_path)

# Check the think block tokens
think_block = "<think>\n\n</think>\n\n"
tokens = tokenizer.encode(think_block, add_special_tokens=False)
print(f"Think block: {repr(think_block)}")
print(f"Token IDs: {tokens}")

# Decode each token
for tid in tokens:
    decoded = tokenizer.decode([tid])
    print(f"  {tid} -> {repr(decoded)}")

# Check individual tokens
print("\nIndividual token checks:")
print(f"  '<think>' -> {tokenizer.encode('<think>', add_special_tokens=False)}")
print(f"  '</think>' -> {tokenizer.encode('</think>', add_special_tokens=False)}")
print(f"  '\\n' -> {tokenizer.encode(chr(10), add_special_tokens=False)}")
print(f"  '\\n\\n' -> {tokenizer.encode(chr(10) + chr(10), add_special_tokens=False)}")

# What does our expected sequence look like?
expected = [151667, 198, 198, 151668, 198, 198]
print(f"\nExpected sequence: {expected}")
print(f"Decoded: {repr(tokenizer.decode(expected))}")
