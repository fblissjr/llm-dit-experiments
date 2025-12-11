#!/usr/bin/env python3
"""
Check what chat template Qwen3-VL actually produces.
Qwen3-VL is NOT a thinking model - enable_thinking is ignored.

This also tests that our manual think block injection is working.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from transformers import AutoProcessor

# Find Qwen3-VL
vl_model_path = None
for candidate in [
    Path.home() / "Storage" / "Qwen3-VL-4B-Instruct",
    Path.home() / "models" / "Qwen3-VL-4B-Instruct",
]:
    if candidate.exists():
        vl_model_path = str(candidate)
        break

if not vl_model_path:
    print("Could not find Qwen3-VL model")
    exit(1)

print(f"Loading processor from {vl_model_path}...")
processor = AutoProcessor.from_pretrained(vl_model_path)

# Create a test message (no actual image needed for template check)
from PIL import Image

# Create a tiny dummy image
dummy_img = Image.new("RGB", (64, 64), color="red")

messages = [
    {"role": "user", "content": [
        {"type": "image", "image": dummy_img},
        {"type": "text", "text": "A red barn in a field"}
    ]}
]

print("\n" + "=" * 60)
print("RAW Qwen3-VL template (no think block):")
print("=" * 60)

result_raw = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
print(repr(result_raw))

print("\n" + "=" * 60)
print("Now testing our MANUAL think block injection:")
print("=" * 60)

import torch

# This is what our qwen3_vl.py does:
THINK_START_TOKEN_ID = 151667  # <think>
THINK_END_TOKEN_ID = 151668    # </think>
DOUBLE_NEWLINE_TOKEN_ID = 271  # \n\n

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

print(f"Before injection: {inputs['input_ids'].shape[1]} tokens")
print(f"Token IDs: {inputs['input_ids'][0].tolist()}")

# Inject think block
think_block_tokens = torch.tensor([[
    THINK_START_TOKEN_ID,
    DOUBLE_NEWLINE_TOKEN_ID,
    THINK_END_TOKEN_ID,
    DOUBLE_NEWLINE_TOKEN_ID,
]], dtype=inputs["input_ids"].dtype)

inputs["input_ids"] = torch.cat([inputs["input_ids"], think_block_tokens], dim=1)

print(f"\nAfter injection: {inputs['input_ids'].shape[1]} tokens")
print(f"Token IDs: {inputs['input_ids'][0].tolist()}")

# Decode to see the full prompt
full_prompt = processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
print(f"\nFull prompt with all tokens:")
print(repr(full_prompt))

# Check what Qwen3-4B produces for comparison
print("\n" + "=" * 60)
print("Qwen3-4B reference (with think block):")
print("=" * 60)

# Find Qwen3-4B
qwen_path = None
for candidate in [
    Path.home() / "Storage" / "Qwen3-4B",
    Path.home() / "models" / "Qwen3-4B",
]:
    if candidate.exists():
        qwen_path = str(candidate)
        break

if qwen_path:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(qwen_path)

    messages_text = [{"role": "user", "content": "A red barn in a field"}]

    text_result = tokenizer.apply_chat_template(
        messages_text,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # ADD think block
    )
    print(repr(text_result))

    print("\n" + "=" * 60)
    print("COMPARISON - Do the endings match?")
    print("=" * 60)

    # Extract just the assistant portion
    vl_assistant_part = full_prompt.split("<|im_start|>assistant")[-1]
    qwen_assistant_part = text_result.split("<|im_start|>assistant")[-1]

    print(f"VL assistant portion:    {repr(vl_assistant_part)}")
    print(f"Qwen3 assistant portion: {repr(qwen_assistant_part)}")
    print(f"Match: {vl_assistant_part == qwen_assistant_part}")

else:
    print("Could not find Qwen3-4B for comparison")
