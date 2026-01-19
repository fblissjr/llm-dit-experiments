#!/usr/bin/env python3
"""
Deeper analysis of LTX-2 projection matrix.

Last Updated: 2026-01-16

The initial analysis showed nearly uniform Frobenius norms across all 49 layer
blocks. This script investigates whether that's meaningful or an artifact.

Questions to answer:
1. Is the uniform norm from initialization or learned?
2. What do the actual weight patterns look like (not just norms)?
3. How do layer activations interact with W?
4. What's the effective rank of each layer block?

Usage:
    uv run python experiments/ltx2/analyze_projection_deeper.py
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def analyze_projection_structure(
    model_path: str = "models/LTX-2",
    output_dir: str = "experiments/results/projection_analysis",
):
    """Deeper structural analysis of projection W."""
    from diffusers import LTX2Pipeline

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LTX-2 Projection Matrix - Deeper Analysis")
    print("=" * 60)

    # Load pipeline
    print("\nLoading model...")
    pipe = LTX2Pipeline.from_pretrained(model_path, dtype=torch.float32)

    # Get projection W
    W = pipe.connectors.text_proj_in.weight.data
    print(f"Projection W shape: {W.shape}")  # [3840, 188160]

    num_layers = 49
    hidden_dim = 3840

    # Reshape to [out_dim, num_layers, hidden_dim]
    W_reshaped = W.reshape(hidden_dim, num_layers, hidden_dim)
    print(f"Reshaped to: {W_reshaped.shape}")

    # Analysis 1: Check if weights look initialized or learned
    print("\n" + "=" * 60)
    print("ANALYSIS 1: Initialization vs Learned")
    print("=" * 60)

    # Xavier uniform init for [3840, 188160] would have bounds ±sqrt(6/(3840+188160)) ≈ ±0.0056
    xavier_bound = np.sqrt(6 / (hidden_dim + num_layers * hidden_dim))
    kaiming_std = np.sqrt(2 / (num_layers * hidden_dim))

    actual_std = W.std().item()
    actual_mean = W.mean().item()
    actual_min = W.min().item()
    actual_max = W.max().item()

    print(f"Expected Xavier uniform bounds: ±{xavier_bound:.6f}")
    print(f"Expected Kaiming std: {kaiming_std:.6f}")
    print(f"Actual W stats:")
    print(f"  mean: {actual_mean:.6f}")
    print(f"  std:  {actual_std:.6f}")
    print(f"  min:  {actual_min:.6f}")
    print(f"  max:  {actual_max:.6f}")

    # If actual stats are close to initialization, weights weren't trained much
    if abs(actual_std - kaiming_std) < 0.001:
        print("\n⚠️  W std is very close to Kaiming init - may not be well-trained")
    else:
        print(f"\n✓ W std differs from init by {abs(actual_std - kaiming_std):.6f}")

    # Analysis 2: Per-layer weight statistics (not just norm)
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Per-Layer Weight Structure")
    print("=" * 60)

    layer_stats = []
    for i in range(num_layers):
        block = W_reshaped[:, i, :]  # [3840, 3840]
        stats = {
            "layer": i,
            "mean": block.mean().item(),
            "std": block.std().item(),
            "min": block.min().item(),
            "max": block.max().item(),
            "frobenius": torch.norm(block).item(),
            "spectral": torch.linalg.norm(block, ord=2).item(),  # Largest singular value
        }
        layer_stats.append(stats)

    # Print comparison
    print(f"\n{'Layer':<8} {'Frob Norm':<12} {'Spectral':<12} {'Std':<12} {'Mean':<12}")
    print("-" * 56)
    for s in layer_stats[:5]:
        print(
            f"{s['layer']:<8} {s['frobenius']:<12.4f} {s['spectral']:<12.4f} {s['std']:<12.6f} {s['mean']:<12.6f}"
        )
    print("...")
    for s in layer_stats[-5:]:
        print(
            f"{s['layer']:<8} {s['frobenius']:<12.4f} {s['spectral']:<12.4f} {s['std']:<12.6f} {s['mean']:<12.6f}"
        )

    # Check variance in spectral norms (more informative than Frobenius)
    spectral_norms = [s["spectral"] for s in layer_stats]
    spectral_range = max(spectral_norms) - min(spectral_norms)
    spectral_cv = np.std(spectral_norms) / np.mean(spectral_norms)  # Coefficient of variation

    print(f"\nSpectral norm range: {min(spectral_norms):.4f} - {max(spectral_norms):.4f}")
    print(f"Spectral coefficient of variation: {spectral_cv:.4f}")

    if spectral_cv < 0.05:
        print("⚠️  Very uniform spectral norms - layers may be treated similarly")
    else:
        print("✓ Spectral norms show variation between layers")

    # Analysis 3: Effective rank of each layer block
    print("\n" + "=" * 60)
    print("ANALYSIS 3: Effective Rank per Layer")
    print("=" * 60)

    effective_ranks = []
    for i in range(num_layers):
        block = W_reshaped[:, i, :]
        # SVD
        U, S, Vh = torch.linalg.svd(block)
        # Effective rank: sum(S)^2 / sum(S^2) (how many singular values contribute)
        eff_rank = (S.sum() ** 2 / (S**2).sum()).item()
        effective_ranks.append(eff_rank)

    print(f"Effective rank range: {min(effective_ranks):.1f} - {max(effective_ranks):.1f}")
    print(f"Mean effective rank: {np.mean(effective_ranks):.1f}")

    # If all blocks have similar effective rank, they have similar structure
    rank_cv = np.std(effective_ranks) / np.mean(effective_ranks)
    print(f"Effective rank CV: {rank_cv:.4f}")

    # Analysis 4: Cross-layer weight correlation
    print("\n" + "=" * 60)
    print("ANALYSIS 4: Cross-Layer Weight Similarity")
    print("=" * 60)

    # Flatten each layer's block and compute cosine similarity
    flattened = (
        W_reshaped.reshape(hidden_dim, num_layers, hidden_dim)
        .permute(1, 0, 2)
        .reshape(num_layers, -1)
    )
    # Normalize
    flattened_norm = flattened / (torch.norm(flattened, dim=1, keepdim=True) + 1e-8)
    # Cosine similarity matrix
    similarity = torch.mm(flattened_norm, flattened_norm.T).numpy()

    # Average similarity (excluding diagonal)
    mask = ~np.eye(num_layers, dtype=bool)
    avg_similarity = similarity[mask].mean()
    max_similarity = similarity[mask].max()

    print(f"Average cross-layer cosine similarity: {avg_similarity:.4f}")
    print(f"Max cross-layer similarity: {max_similarity:.4f}")

    if avg_similarity > 0.5:
        print("⚠️  High cross-layer similarity - weight blocks may be near-copies")
    elif avg_similarity > 0.2:
        print("ℹ️  Moderate cross-layer similarity")
    else:
        print("✓ Layer blocks are fairly independent")

    # Find most similar pairs
    np.fill_diagonal(similarity, -1)
    flat_idx = np.argsort(similarity.flatten())[::-1]
    print("\nMost similar layer pairs:")
    seen = set()
    count = 0
    for idx in flat_idx:
        i, j = divmod(idx, num_layers)
        if (j, i) not in seen and i != j:
            print(f"  Layer {i} ↔ Layer {j}: {similarity[i, j]:.4f}")
            seen.add((i, j))
            count += 1
            if count >= 5:
                break

    # Analysis 5: Test with a sample prompt
    print("\n" + "=" * 60)
    print("ANALYSIS 5: Activation-Weighted Contribution")
    print("=" * 60)
    print("(Computing per-layer contribution with actual activations)")

    # Get hidden states from a test prompt
    test_prompt = (
        "A golden retriever runs through a sun-dappled park, its fur gleaming in warm afternoon light. "
        "The camera tracks alongside as the dog bounds across lush green grass."
    )

    # Tokenize and get hidden states
    text_inputs = pipe.tokenizer(
        test_prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        text_outputs = pipe.text_encoder(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            output_hidden_states=True,
        )

    hidden_states = torch.stack(text_outputs.hidden_states, dim=-1)
    # Shape: [1, seq_len, 3840, 49]

    # Get actual sequence length (non-padding)
    seq_len = text_inputs.attention_mask.sum().item()
    print(f"Test prompt tokens: {seq_len}")

    # Compute per-layer contribution: ||W_layer @ h_layer||
    layer_contributions = []
    h_stacked = hidden_states[0, :seq_len, :, :]  # [seq_len, 3840, 49]

    for i in range(num_layers):
        h_layer = h_stacked[:, :, i]  # [seq_len, 3840]
        W_layer = W_reshaped[:, i, :]  # [3840, 3840]
        output = h_layer @ W_layer.T  # [seq_len, 3840]
        contribution = torch.norm(output).item()
        layer_contributions.append(contribution)

    # Normalize to percentages
    total = sum(layer_contributions)
    layer_pcts = [c / total * 100 for c in layer_contributions]

    print(f"\nTop 10 layers by actual contribution:")
    sorted_layers = sorted(enumerate(layer_pcts), key=lambda x: x[1], reverse=True)
    for rank, (layer_idx, pct) in enumerate(sorted_layers[:10], 1):
        print(f"  {rank}. Layer {layer_idx}: {pct:.2f}%")

    print(f"\nBottom 5 layers:")
    for rank, (layer_idx, pct) in enumerate(sorted_layers[-5:], 1):
        print(f"  {rank}. Layer {layer_idx}: {pct:.2f}%")

    # Check if contribution distribution is more varied than weights
    contrib_cv = np.std(layer_pcts) / np.mean(layer_pcts)
    print(f"\nContribution CV: {contrib_cv:.4f} (weight CV was {spectral_cv:.4f})")

    if contrib_cv > spectral_cv * 2:
        print("✓ Activations create more layer differentiation than weights alone")
    else:
        print("ℹ️  Activation contribution mirrors weight distribution")

    # Save results
    results = {
        "weight_stats": {
            "mean": actual_mean,
            "std": actual_std,
            "expected_kaiming_std": kaiming_std,
        },
        "layer_stats": layer_stats,
        "effective_ranks": effective_ranks,
        "cross_layer_similarity": {
            "average": float(avg_similarity),
            "max": float(max_similarity),
        },
        "activation_contribution": {
            "layer_percentages": layer_pcts,
            "contribution_cv": float(contrib_cv),
            "weight_cv": float(spectral_cv),
        },
    }

    results_path = output_path / "projection_deep_analysis.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {results_path}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Frobenius vs Spectral norms
    ax1 = axes[0, 0]
    ax1.bar(range(num_layers), [s["frobenius"] for s in layer_stats], alpha=0.7, label="Frobenius")
    ax1.bar(range(num_layers), [s["spectral"] for s in layer_stats], alpha=0.7, label="Spectral")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Norm")
    ax1.set_title("Frobenius vs Spectral Norm per Layer")
    ax1.legend()

    # Plot 2: Effective ranks
    ax2 = axes[0, 1]
    ax2.bar(range(num_layers), effective_ranks)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Effective Rank")
    ax2.set_title("Effective Rank per Layer Block")
    ax2.axhline(
        y=np.mean(effective_ranks),
        color="r",
        linestyle="--",
        label=f"Mean: {np.mean(effective_ranks):.1f}",
    )
    ax2.legend()

    # Plot 3: Cross-layer similarity heatmap
    ax3 = axes[1, 0]
    np.fill_diagonal(similarity, 1)
    im = ax3.imshow(similarity, cmap="RdBu_r", vmin=-0.5, vmax=1)
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("Layer")
    ax3.set_title("Cross-Layer Weight Similarity")
    plt.colorbar(im, ax=ax3)

    # Plot 4: Activation-weighted contribution
    ax4 = axes[1, 1]
    ax4.bar(range(num_layers), layer_pcts)
    ax4.set_xlabel("Layer")
    ax4.set_ylabel("Contribution (%)")
    ax4.set_title("Per-Layer Contribution (with actual activations)")
    ax4.axhline(
        y=100 / num_layers, color="r", linestyle="--", label=f"Uniform: {100 / num_layers:.2f}%"
    )
    ax4.legend()

    plt.tight_layout()
    plot_path = output_path / "projection_deep_analysis.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to: {plot_path}")
    plt.close()

    del pipe
    return results


def main():
    parser = argparse.ArgumentParser(description="Deeper projection matrix analysis")
    parser.add_argument("--model-path", default="models/LTX-2")
    parser.add_argument("--output-dir", default="experiments/results/projection_analysis")
    args = parser.parse_args()

    analyze_projection_structure(args.model_path, args.output_dir)


if __name__ == "__main__":
    main()
