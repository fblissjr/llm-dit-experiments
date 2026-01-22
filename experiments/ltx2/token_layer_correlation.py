#!/usr/bin/env python3
"""
LTX-2 Token-Type Layer Correlation (Experiment 1.2)

Last Updated: 2026-01-15

Analyzes whether different token types (nouns, verbs, adjectives) have different
optimal layers in Gemma3 hidden states.

Hypothesis: Nouns may prefer later (semantic) layers, while verbs may prefer
middle (syntactic) layers, and adjectives may prefer early-middle layers.

Usage:
    uv run python experiments/ltx2/token_layer_correlation.py
    uv run python experiments/ltx2/token_layer_correlation.py --save-plots
"""

import argparse
import gc
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np


# Prompts designed to have clear POS patterns
DEFAULT_PROMPTS = [
    # Noun-heavy
    "a cat sitting on a chair in the kitchen",
    "mountains rivers forests valleys",
    "the dog the ball the tree the house",
    "sunset ocean beach sand waves",

    # Verb-heavy
    "running jumping climbing swimming",
    "the man walks talks eats drinks",
    "birds flying fish swimming dogs running",
    "spinning turning twisting falling",

    # Adjective-heavy
    "a big red fluffy soft cat",
    "beautiful golden shimmering sunset",
    "tiny ancient weathered stone wall",
    "bright colorful vibrant flowers",

    # Mixed/balanced
    "a large dog quickly runs through the green field",
    "the old man slowly walks down the dusty road",
    "bright stars silently shine over dark mountains",
    "soft rain gently falls on fresh leaves",

    # Complex scenes
    "a cat chasing a red ball under a tree in a sunny park",
    "an old man sitting on a bench reading a newspaper",
    "children playing in a fountain on a hot summer day",
    "a sailboat on calm water at sunset",

    # Action sequences
    "a bird takes flight from a tree branch",
    "water droplets fall and splash",
    "a dancer spins gracefully",
    "leaves falling from a tree in autumn",

    # Spatial relationships
    "a lamp above a table next to a window",
    "stairs leading up to a red door",
    "a bridge over a river through mountains",
    "a cat behind a dog in front of a house",
]


def get_spacy_pos_tags(text: str):
    """Get POS tags for each token using spaCy."""
    import spacy

    # Load English model (small is sufficient for POS)
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)

    # Map tokens to their POS categories
    pos_map = {}
    for token in doc:
        # Simplified POS categories
        if token.pos_ in ("NOUN", "PROPN"):
            pos_map[token.text.lower()] = "NOUN"
        elif token.pos_ == "VERB":
            pos_map[token.text.lower()] = "VERB"
        elif token.pos_ == "ADJ":
            pos_map[token.text.lower()] = "ADJ"
        elif token.pos_ == "ADV":
            pos_map[token.text.lower()] = "ADV"
        elif token.pos_ in ("ADP", "SCONJ", "CCONJ"):
            pos_map[token.text.lower()] = "FUNC"  # Function words
        elif token.pos_ == "DET":
            pos_map[token.text.lower()] = "DET"
        else:
            pos_map[token.text.lower()] = "OTHER"

    return pos_map


def align_tokens_to_subwords(text: str, tokenizer, pos_map: dict):
    """
    Align spaCy tokens to subword tokens.

    Returns a list of (subword_idx, pos_tag) pairs.
    """
    # Tokenize with the model's tokenizer
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = encoded.get("offset_mapping", [])
    input_ids = encoded["input_ids"]

    # Decode each token to get its text
    subword_pos = []
    for idx, token_id in enumerate(input_ids):
        token_text = tokenizer.decode([token_id]).strip().lower()

        # Try to find matching POS
        if token_text in pos_map:
            subword_pos.append((idx, pos_map[token_text]))
        else:
            # Try partial matching (for subwords)
            matched = False
            for word, pos in pos_map.items():
                if token_text in word or word in token_text:
                    subword_pos.append((idx, pos))
                    matched = True
                    break
            if not matched:
                subword_pos.append((idx, "OTHER"))

    return subword_pos


def compute_token_layer_stats(
    layer_stack: torch.Tensor,
    attention_mask: torch.Tensor,
    subword_pos: list,
) -> dict:
    """
    Compute per-token-type layer statistics.

    Args:
        layer_stack: [1, T, D, L] tensor
        attention_mask: [1, T] tensor
        subword_pos: list of (token_idx, pos_tag)

    Returns:
        Dict mapping POS -> per-layer statistics
    """
    B, T, D, L = layer_stack.shape

    # Calculate padding offset (left-padded)
    # The valid tokens are at the end of the sequence
    num_valid = attention_mask.sum().item()
    num_tokens = len(subword_pos)
    padding_offset = T - num_valid

    # Group tokens by POS
    pos_stats = defaultdict(lambda: {
        "layer_norms": [[] for _ in range(L)],
        "layer_vars": [[] for _ in range(L)],
        "count": 0,
    })

    for token_idx, pos_tag in subword_pos:
        # Adjust for left padding
        adjusted_idx = padding_offset + token_idx

        if adjusted_idx >= T:
            continue
        if attention_mask[0, adjusted_idx] == 0:
            continue

        # Get this token's representation across layers
        token_repr = layer_stack[0, adjusted_idx, :, :]  # [D, L]

        pos_stats[pos_tag]["count"] += 1

        for l in range(L):
            layer_repr = token_repr[:, l]  # [D]
            norm = layer_repr.float().norm().item()
            var = layer_repr.float().var().item()

            pos_stats[pos_tag]["layer_norms"][l].append(norm)
            pos_stats[pos_tag]["layer_vars"][l].append(var)

    return dict(pos_stats)


def run_experiment(
    prompts: list,
    model_id: str = "google/gemma-3-12b-it-qat-q4_0-unquantized",
    max_sequence_length: int = 128,
    save_plots: bool = False,
    output_dir: str = "experiments/results/ltx2",
):
    """Run token-type layer correlation experiment."""
    from llm_dit.encoders.gemma3 import Gemma3Encoder

    print("=" * 60)
    print("LTX-2 Token-Type Layer Correlation (Experiment 1.2)")
    print("=" * 60)
    print(f"Number of prompts: {len(prompts)}")
    print(f"Model: {model_id}")

    # Load encoder
    print("\nLoading encoder...")
    max_memory = {0: "18GiB", "cpu": "32GiB"}

    encoder = Gemma3Encoder.from_pretrained(
        model_id,
        device="auto",
        dtype="bfloat16",
        max_sequence_length=max_sequence_length,
        quantization="8bit",
        max_memory=max_memory,
    )

    # Aggregate stats across all prompts
    # Use a regular dict with explicit initialization to avoid closure issues
    all_pos_stats = {}

    print(f"\nProcessing {len(prompts)} prompts...")

    for i, prompt in enumerate(prompts):
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(prompts)}] {prompt[:40]}...")

        # Get POS tags
        pos_map = get_spacy_pos_tags(prompt)

        # Get subword alignment (access private tokenizer)
        subword_pos = align_tokens_to_subwords(prompt, encoder._tokenizer, pos_map)

        # Get hidden states
        result = encoder.encode_multilayer(prompt, return_projected=False)
        layer_stack = result['layer_stack']  # [1, T, D, L]
        attention_mask = result['attention_mask']  # [1, T]

        # Compute per-token-type stats
        pos_stats = compute_token_layer_stats(layer_stack, attention_mask, subword_pos)

        # Aggregate
        for pos_tag, stats in pos_stats.items():
            if pos_tag not in all_pos_stats:
                all_pos_stats[pos_tag] = {
                    "layer_norms": [[] for _ in range(49)],
                    "layer_vars": [[] for _ in range(49)],
                    "count": 0,
                }
            all_pos_stats[pos_tag]["count"] += stats["count"]
            for l in range(len(stats["layer_norms"])):
                all_pos_stats[pos_tag]["layer_norms"][l].extend(stats["layer_norms"][l])
                all_pos_stats[pos_tag]["layer_vars"][l].extend(stats["layer_vars"][l])

        # Memory cleanup
        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Compute aggregate statistics
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    num_layers = 49  # Gemma3 has 48 layers + embedding

    # Debug: show raw counts
    print("\nRaw token counts by POS:")
    for pos_tag, stats in sorted(all_pos_stats.items()):
        print(f"  {pos_tag}: {stats['count']} tokens")

    results = {}
    for pos_tag, stats in all_pos_stats.items():
        if stats["count"] < 3:  # Skip very rare categories
            continue

        avg_norms = []
        avg_vars = []
        for l in range(num_layers):
            if stats["layer_norms"][l]:
                avg_norms.append(np.mean(stats["layer_norms"][l]))
                avg_vars.append(np.mean(stats["layer_vars"][l]))
            else:
                avg_norms.append(0)
                avg_vars.append(0)

        results[pos_tag] = {
            "count": stats["count"],
            "avg_norms": np.array(avg_norms),
            "avg_vars": np.array(avg_vars),
        }

        # Find peak layer (excluding Layer 47 which is anomalous)
        norms_excl47 = np.array(avg_norms[:47])
        peak_layer = np.argmax(norms_excl47)

        print(f"\n{pos_tag} ({stats['count']} tokens):")
        print(f"  Peak norm layer: {peak_layer}")
        print(f"  Early (0-15) avg norm:  {np.mean(avg_norms[:16]):.1f}")
        print(f"  Middle (16-31) avg norm: {np.mean(avg_norms[16:32]):.1f}")
        print(f"  Late (32-46) avg norm:  {np.mean(avg_norms[32:47]):.1f}")

    # Compare categories
    print("\n" + "=" * 60)
    print("Cross-Category Analysis")
    print("=" * 60)

    if "NOUN" in results and "VERB" in results:
        noun_peak = np.argmax(results["NOUN"]["avg_norms"][:47])
        verb_peak = np.argmax(results["VERB"]["avg_norms"][:47])
        print(f"\nNOUN peak layer: {noun_peak}")
        print(f"VERB peak layer: {verb_peak}")

        # Compute relative preference
        noun_late_ratio = np.mean(results["NOUN"]["avg_norms"][32:47]) / np.mean(results["NOUN"]["avg_norms"][:16])
        verb_late_ratio = np.mean(results["VERB"]["avg_norms"][32:47]) / np.mean(results["VERB"]["avg_norms"][:16])
        print(f"\nNOUN late/early ratio: {noun_late_ratio:.2f}")
        print(f"VERB late/early ratio: {verb_late_ratio:.2f}")

    if "ADJ" in results:
        adj_peak = np.argmax(results["ADJ"]["avg_norms"][:47])
        adj_late_ratio = np.mean(results["ADJ"]["avg_norms"][32:47]) / np.mean(results["ADJ"]["avg_norms"][:16])
        print(f"\nADJ peak layer: {adj_peak}")
        print(f"ADJ late/early ratio: {adj_late_ratio:.2f}")

    # Save results
    if save_plots:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save numpy data
        np.savez(
            output_path / "token_correlation_results.npz",
            **{f"{k}_norms": v["avg_norms"] for k, v in results.items()},
            **{f"{k}_vars": v["avg_vars"] for k, v in results.items()},
            **{f"{k}_count": v["count"] for k, v in results.items()},
        )
        print(f"\nResults saved to {output_path}/token_correlation_results.npz")

        try:
            import matplotlib.pyplot as plt

            # Plot 1: Layer norms by POS
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = {"NOUN": "blue", "VERB": "red", "ADJ": "green", "ADV": "orange", "FUNC": "gray", "DET": "purple"}

            for pos_tag, data in results.items():
                if pos_tag in colors:
                    ax.plot(range(num_layers), data["avg_norms"],
                            label=f"{pos_tag} (n={data['count']})",
                            color=colors.get(pos_tag, "black"),
                            alpha=0.7)

            ax.axvline(x=16, color='gray', linestyle='--', alpha=0.3, label='Early/Middle')
            ax.axvline(x=32, color='gray', linestyle='--', alpha=0.3, label='Middle/Late')
            ax.axvline(x=47, color='red', linestyle='--', alpha=0.5, label='Layer 47')

            ax.set_xlabel('Layer')
            ax.set_ylabel('Average L2 Norm')
            ax.set_title('Token Representation Norm by Part-of-Speech')
            ax.legend(loc='upper left')
            fig.savefig(output_path / "pos_layer_norms.png", dpi=150)
            plt.close()

            # Plot 2: Normalized comparison (to remove magnitude effect)
            fig, ax = plt.subplots(figsize=(12, 6))

            for pos_tag, data in results.items():
                if pos_tag in colors:
                    # Normalize to [0, 1] range for comparison
                    norms = data["avg_norms"][:47]  # Exclude Layer 47
                    normalized = (norms - norms.min()) / (norms.max() - norms.min() + 1e-8)
                    ax.plot(range(47), normalized,
                            label=f"{pos_tag}",
                            color=colors.get(pos_tag, "black"),
                            alpha=0.7)

            ax.axvline(x=16, color='gray', linestyle='--', alpha=0.3)
            ax.axvline(x=32, color='gray', linestyle='--', alpha=0.3)

            ax.set_xlabel('Layer')
            ax.set_ylabel('Normalized Norm (0-1)')
            ax.set_title('Normalized Layer Preference by POS (excluding Layer 47)')
            ax.legend(loc='upper left')
            fig.savefig(output_path / "pos_normalized_comparison.png", dpi=150)
            plt.close()

            print(f"Plots saved to {output_path}/")

        except ImportError:
            print("\nNote: matplotlib not available for plotting")

    # Cleanup
    del encoder
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Token-Type Layer Correlation")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Number of prompts to use",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/gemma-3-12b-it-qat-q4_0-unquantized",
        help="Gemma model ID",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots and results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/ltx2",
        help="Output directory",
    )
    args = parser.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.num_prompts:
        prompts = prompts[:args.num_prompts]

    run_experiment(
        prompts=prompts,
        model_id=args.model_id,
        save_plots=args.save_plots,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
