#!/usr/bin/env python3
"""
Test: Pad prompts with filler text to normalize outlier dimensions.

Last updated: 2025-12-18

Instead of padding embeddings (which breaks the transformer), pad the
actual prompt text to achieve longer sequences through the full pipeline.
"""

import argparse
import sys
from pathlib import Path
import subprocess

# Filler text to pad prompts (neutral, non-semantic)
FILLER = " . " * 100  # 300 characters of periods

def make_padded_prompt(base_prompt: str, target_tokens: int, current_tokens: int) -> str:
    """
    Create a padded prompt by appending filler text.

    Rough estimate: ~3-4 chars per token for English text.
    """
    if target_tokens <= current_tokens:
        return base_prompt

    tokens_needed = target_tokens - current_tokens
    # Each " . " is roughly 1-2 tokens
    filler_amount = tokens_needed * 2  # " . " pairs
    filler = " ." * filler_amount

    return base_prompt + filler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="A tabby cat with green eyes")
    parser.add_argument("--output-dir", default="experiments/results/prompt_padding_test")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_prompt = args.prompt

    # Test cases: raw, and padded to different lengths
    test_cases = [
        ("raw", base_prompt),
        ("padded_64", base_prompt + " ." * 50),
        ("padded_128", base_prompt + " ." * 100),
        ("padded_256", base_prompt + " ." * 200),
    ]

    print(f"Base prompt: {base_prompt}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_dir}\n")

    for name, prompt in test_cases:
        print(f"\nGenerating {name}...")
        print(f"  Prompt length: {len(prompt)} chars")

        output_path = output_dir / f"{name}.png"

        cmd = [
            "uv", "run", "scripts/generate.py",
            "--config", args.config,
            "--profile", args.profile,
            "--seed", str(args.seed),
            "--output", str(output_path),
            prompt,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                print(f"  Saved: {output_path}")
            else:
                print(f"  FAILED: {result.stderr[-500:]}")

        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n\nImages saved to: {output_dir}")
    print("\nCompare visually:")
    print("- If padding normalizes outliers, padded versions should look more similar")
    print("- Raw (short) may have different characteristics due to higher outlier influence")


if __name__ == "__main__":
    main()
