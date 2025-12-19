#!/usr/bin/env python3
"""
Analyze whether outlier dimensions correlate with reasoning/thinking patterns.

Last updated: 2025-12-18

Hypothesis: Dimensions 4, 396, 100 encode chain-of-thought reasoning state,
not visual semantics. If true, these dimensions should:
1. Have higher activation for complex/uncertain prompts
2. Show different patterns for "think" vs "non-think" formatted prompts
3. Correlate with prompt complexity, not visual content

Usage:
    uv run experiments/analyze_thinking_dimensions.py --model-path /path/to/Qwen3-4B
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer

# Outlier dimensions from our analysis
OUTLIER_DIMS = [4, 100, 396]

# Test prompts: Simple vs Complex
SIMPLE_PROMPTS = [
    "A cat",
    "A red ball",
    "A tree",
    "Blue sky",
    "A house",
]

COMPLEX_PROMPTS = [
    "A tabby cat with green eyes sitting on a velvet cushion in a Victorian parlor with afternoon light streaming through lace curtains",
    "A complex mathematical equation visualized as geometric shapes floating in an abstract space with interconnected nodes",
    "Multiple people of different ages engaged in various activities at a busy farmers market on a sunny morning",
    "A surreal dreamscape combining elements of ocean, desert, and forest in impossible geometric arrangements",
    "A highly detailed cross-section of a futuristic city showing underground transit, surface buildings, and aerial vehicles",
]

# Uncertain vs Confident
UNCERTAIN_PROMPTS = [
    "Maybe a cat or possibly a dog",
    "Something like a forest, perhaps with trees",
    "Could be a person, hard to tell",
    "An abstract concept, possibly freedom",
    "Some kind of building structure",
]

CONFIDENT_PROMPTS = [
    "Definitely a tabby cat",
    "A dense pine forest with tall trees",
    "A professional woman in a business suit",
    "The statue of liberty at sunset",
    "A red brick Victorian mansion",
]

# With and without think block
def format_with_think(prompt: str) -> str:
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

def format_without_think(prompt: str) -> str:
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def get_outlier_activations(
    model,
    tokenizer,
    prompt: str,
    layer: int = -2,
    device: str = "cuda",
) -> dict:
    """Get activation statistics for outlier dimensions."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Get specified layer
    hidden = outputs.hidden_states[layer]  # (1, seq, 2560)
    hidden = hidden.squeeze(0).float().cpu().numpy()  # (seq, 2560)

    # Compute per-dimension stats
    results = {
        "prompt_length": hidden.shape[0],
        "global_mean": float(hidden.mean()),
        "global_std": float(hidden.std()),
    }

    # Outlier dimension stats
    for dim in OUTLIER_DIMS:
        dim_vals = hidden[:, dim]
        results[f"dim_{dim}_mean"] = float(dim_vals.mean())
        results[f"dim_{dim}_std"] = float(dim_vals.std())
        results[f"dim_{dim}_max"] = float(dim_vals.max())
        results[f"dim_{dim}_min"] = float(dim_vals.min())
        results[f"dim_{dim}_range"] = float(dim_vals.max() - dim_vals.min())

    return results


def analyze_prompt_category(
    model,
    tokenizer,
    prompts: list[str],
    category_name: str,
    device: str = "cuda",
) -> dict:
    """Analyze a category of prompts."""
    all_results = []

    for prompt in prompts:
        result = get_outlier_activations(model, tokenizer, prompt, device=device)
        all_results.append(result)

    # Aggregate
    summary = {"category": category_name, "n_prompts": len(prompts)}

    for dim in OUTLIER_DIMS:
        means = [r[f"dim_{dim}_mean"] for r in all_results]
        stds = [r[f"dim_{dim}_std"] for r in all_results]
        ranges = [r[f"dim_{dim}_range"] for r in all_results]

        summary[f"dim_{dim}_avg_mean"] = float(np.mean(means))
        summary[f"dim_{dim}_avg_std"] = float(np.mean(stds))
        summary[f"dim_{dim}_avg_range"] = float(np.mean(ranges))

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
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

    print("\n" + "="*70)
    print("TESTING: Simple vs Complex Prompts")
    print("="*70)

    simple_results = analyze_prompt_category(model, tokenizer, SIMPLE_PROMPTS, "simple", args.device)
    complex_results = analyze_prompt_category(model, tokenizer, COMPLEX_PROMPTS, "complex", args.device)

    print(f"\n{'Dimension':<12} {'Simple (mean)':<15} {'Complex (mean)':<15} {'Ratio':<10}")
    print("-"*55)
    for dim in OUTLIER_DIMS:
        simple_val = simple_results[f"dim_{dim}_avg_mean"]
        complex_val = complex_results[f"dim_{dim}_avg_mean"]
        ratio = complex_val / simple_val if simple_val != 0 else float('inf')
        print(f"Dim {dim:<7} {simple_val:<15.2f} {complex_val:<15.2f} {ratio:<10.2f}x")

    print("\n" + "="*70)
    print("TESTING: Uncertain vs Confident Prompts")
    print("="*70)

    uncertain_results = analyze_prompt_category(model, tokenizer, UNCERTAIN_PROMPTS, "uncertain", args.device)
    confident_results = analyze_prompt_category(model, tokenizer, CONFIDENT_PROMPTS, "confident", args.device)

    print(f"\n{'Dimension':<12} {'Uncertain (mean)':<17} {'Confident (mean)':<17} {'Ratio':<10}")
    print("-"*60)
    for dim in OUTLIER_DIMS:
        uncertain_val = uncertain_results[f"dim_{dim}_avg_mean"]
        confident_val = confident_results[f"dim_{dim}_avg_mean"]
        ratio = uncertain_val / confident_val if confident_val != 0 else float('inf')
        print(f"Dim {dim:<7} {uncertain_val:<17.2f} {confident_val:<17.2f} {ratio:<10.2f}x")

    print("\n" + "="*70)
    print("TESTING: With vs Without Think Block")
    print("="*70)

    # Use same prompts, different formatting
    base_prompts = SIMPLE_PROMPTS[:3]

    with_think = [format_with_think(p) for p in base_prompts]
    without_think = [format_without_think(p) for p in base_prompts]

    think_results = analyze_prompt_category(model, tokenizer, with_think, "with_think", args.device)
    nothink_results = analyze_prompt_category(model, tokenizer, without_think, "without_think", args.device)

    print(f"\n{'Dimension':<12} {'With Think (mean)':<18} {'No Think (mean)':<18} {'Ratio':<10}")
    print("-"*65)
    for dim in OUTLIER_DIMS:
        think_val = think_results[f"dim_{dim}_avg_mean"]
        nothink_val = nothink_results[f"dim_{dim}_avg_mean"]
        ratio = think_val / nothink_val if nothink_val != 0 else float('inf')
        print(f"Dim {dim:<7} {think_val:<18.2f} {nothink_val:<18.2f} {ratio:<10.2f}x")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("""
If the 'Thinking Model Hypothesis' is correct:
- Complex prompts should have HIGHER outlier activations (more reasoning)
- Uncertain prompts should have HIGHER outlier activations (more deliberation)
- Think block prompts should have DIFFERENT patterns (thinking state initialized)

If outlier dimensions are just 'attention sinks':
- Activations should be relatively constant regardless of prompt type
- No clear correlation with complexity or uncertainty
""")


if __name__ == "__main__":
    main()
