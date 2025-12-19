#!/usr/bin/env python3
"""
Test: Does prompt LENGTH or SEMANTIC CONTENT drive outlier dimension activations?

Last updated: 2025-12-18

Hypothesis 1 (Length): Softmax distributes attention. Short sequences concentrate
attention into fewer tokens, causing higher outlier activations.

Hypothesis 2 (Semantics): The outliers encode "semantic density" or "information
content", not sequence length.

Test: Same semantic content, different token counts.
"""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer

OUTLIER_DIMS = [4, 100, 396]

# Test prompts: Same semantics, different lengths
TEST_CASES = {
    # Simple prompt
    "simple_short": "A cat",

    # Same semantics, artificially lengthened by repetition
    "simple_repeated": "A cat. A cat. A cat. A cat. A cat. A cat. A cat. A cat. A cat. A cat.",

    # Same semantics, lengthened with synonyms/elaboration
    "simple_expanded": "A feline creature, specifically a domestic cat, which is a small carnivorous mammal",

    # Complex prompt
    "complex_short": "A tabby cat with green eyes on a red cushion",

    # Same complex prompt repeated
    "complex_repeated": "A tabby cat with green eyes on a red cushion. A tabby cat with green eyes on a red cushion. A tabby cat with green eyes on a red cushion.",

    # Padding with neutral tokens (periods, spaces)
    "simple_padded": "A cat . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .",

    # High semantic density (many concepts)
    "high_density": "Cat dog bird fish tree house car boat plane train mountain river ocean sunset sunrise",

    # Low semantic density (few concepts, many words)
    "low_density": "The cat is a cat that is very much a cat and it is definitely a cat because it is a cat",
}


def analyze_prompt(model, tokenizer, prompt: str, layer: int = -2) -> dict:
    """Get outlier dimension stats for a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[layer].squeeze(0).float().cpu().numpy()

    result = {
        "num_tokens": hidden.shape[0],
        "prompt_chars": len(prompt),
    }

    for dim in OUTLIER_DIMS:
        dim_vals = hidden[:, dim]
        result[f"dim_{dim}_mean"] = float(dim_vals.mean())
        result[f"dim_{dim}_std"] = float(dim_vals.std())
        result[f"dim_{dim}_sum"] = float(dim_vals.sum())  # Total activation
        result[f"dim_{dim}_per_token"] = float(dim_vals.sum() / hidden.shape[0])  # Normalized

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/fbliss/Storage/Qwen3-4B")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()

    print("\n" + "="*80)
    print("LENGTH vs SEMANTICS TEST")
    print("="*80)

    results = {}
    for name, prompt in TEST_CASES.items():
        results[name] = analyze_prompt(model, tokenizer, prompt)
        results[name]["prompt"] = prompt[:50] + "..." if len(prompt) > 50 else prompt

    # Print results table
    print(f"\n{'Name':<20} {'Tokens':<8} {'Dim4 Mean':<12} {'Dim4/Token':<12} {'Dim396 Mean':<12} {'Dim396/Token':<12}")
    print("-"*80)

    for name, r in results.items():
        print(f"{name:<20} {r['num_tokens']:<8} {r['dim_4_mean']:<12.1f} {r['dim_4_per_token']:<12.1f} {r['dim_396_mean']:<12.1f} {r['dim_396_per_token']:<12.1f}")

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    # Compare simple_short vs simple_repeated (same semantics, different length)
    short = results["simple_short"]
    repeated = results["simple_repeated"]

    print(f"\n1. REPETITION TEST (same semantics, more tokens)")
    print(f"   'A cat' ({short['num_tokens']} tokens) vs repeated ({repeated['num_tokens']} tokens)")
    print(f"   Dim 4 mean:  {short['dim_4_mean']:.1f} vs {repeated['dim_4_mean']:.1f} (ratio: {repeated['dim_4_mean']/short['dim_4_mean']:.2f}x)")
    print(f"   Dim 4/token: {short['dim_4_per_token']:.1f} vs {repeated['dim_4_per_token']:.1f}")

    # Compare simple_short vs simple_expanded (same semantics, different phrasing)
    expanded = results["simple_expanded"]
    print(f"\n2. EXPANSION TEST (same semantics, more descriptive)")
    print(f"   'A cat' ({short['num_tokens']} tokens) vs expanded ({expanded['num_tokens']} tokens)")
    print(f"   Dim 4 mean:  {short['dim_4_mean']:.1f} vs {expanded['dim_4_mean']:.1f} (ratio: {expanded['dim_4_mean']/short['dim_4_mean']:.2f}x)")

    # Compare high vs low semantic density
    high = results["high_density"]
    low = results["low_density"]
    print(f"\n3. SEMANTIC DENSITY TEST")
    print(f"   High density ({high['num_tokens']} tokens): Dim 4 mean = {high['dim_4_mean']:.1f}")
    print(f"   Low density ({low['num_tokens']} tokens):  Dim 4 mean = {low['dim_4_mean']:.1f}")
    print(f"   Ratio: {low['dim_4_mean']/high['dim_4_mean']:.2f}x")

    # Conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("""
If LENGTH matters (attention distribution):
- Repeated prompts should have LOWER outlier means (attention spread across more tokens)
- Per-token values should be similar across lengths

If SEMANTICS matters:
- Repeated prompts should have SIMILAR outlier means (same information)
- High-density prompts should differ from low-density
""")


if __name__ == "__main__":
    main()
