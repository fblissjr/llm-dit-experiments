#!/usr/bin/env python3
"""
LTX-2 Thinking Token Analysis

Last Updated: 2026-01-16

Novel research: Analyze the 128 learnable "thinking tokens" (registers) that
LTX-2 uses to replace padding in the text sequence.

From the LTX-2 paper:
- Thinking tokens replace padded positions in the text sequence
- They serve as "global information carriers" with bidirectional attention
- 128 learnable tokens that can aggregate information across the sequence

Discovery Questions:
1. What do thinking tokens encode?
2. How do they differ from text tokens in attention patterns?
3. What happens to generation quality WITHOUT thinking tokens?

Usage:
    # Analyze thinking token statistics
    uv run python experiments/ltx2/thinking_token_analysis.py

    # Compare generation with/without thinking tokens
    uv run python experiments/ltx2/thinking_token_analysis.py --compare-generation
"""

import argparse
import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def analyze_thinking_tokens(
    model_path: str = "models/LTX-2",
    output_dir: str = "experiments/results/thinking_tokens",
):
    """
    Analyze the learned thinking token embeddings in LTX-2.

    The text encoder uses 128 learnable "thinking tokens" as registers
    that replace padding and serve as global information aggregators.
    """
    from diffusers import LTX2Pipeline

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LTX-2 Thinking Token Analysis")
    print("=" * 60)

    # Load pipeline (CPU for analysis)
    print("\nLoading model...")
    pipe = LTX2Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    )

    # Find thinking tokens
    print("\nSearching for thinking tokens...")

    thinking_tokens = None
    token_name = None

    # Search common locations
    locations_to_check = [
        ("text_encoder", "thinking_tokens"),
        ("text_encoder.model", "thinking_tokens"),
        ("transformer", "thinking_tokens"),
        ("transformer.connector", "thinking_tokens"),
    ]

    for parent_name, attr_name in locations_to_check:
        try:
            parent = pipe
            for part in parent_name.split("."):
                parent = getattr(parent, part)
            if hasattr(parent, attr_name):
                thinking_tokens = getattr(parent, attr_name)
                token_name = f"{parent_name}.{attr_name}"
                print(f"Found: {token_name}")
                break
        except AttributeError:
            continue

    # Search in state dict
    if thinking_tokens is None:
        print("\nSearching state dict for thinking/register tokens...")
        for component_name, component in [("text_encoder", pipe.text_encoder), ("transformer", pipe.transformer)]:
            if component is None:
                continue
            for name, param in component.named_parameters():
                name_lower = name.lower()
                if any(kw in name_lower for kw in ["thinking", "register", "learnable"]):
                    print(f"  Found: {component_name}.{name} - shape {param.shape}")
                    if thinking_tokens is None:
                        thinking_tokens = param.data
                        token_name = f"{component_name}.{name}"

    if thinking_tokens is None:
        print("\nCould not find thinking tokens directly.")
        print("Listing potentially relevant parameters:")
        for name, param in pipe.text_encoder.named_parameters():
            if param.dim() >= 2 and param.shape[0] < 256:  # Look for small embeddings
                print(f"  {name}: {param.shape}")

        # Try to find them by shape (128 tokens × hidden_dim)
        for name, param in pipe.text_encoder.named_parameters():
            if param.dim() == 2 and param.shape[0] == 128:
                print(f"\nPossible thinking tokens: {name} - shape {param.shape}")
                thinking_tokens = param.data
                token_name = name
                break

        if thinking_tokens is None:
            print("\nNo thinking tokens found. The model may not use this architecture.")
            return None

    print(f"\nThinking tokens: {token_name}")
    print(f"Shape: {thinking_tokens.shape}")  # Expected: [128, hidden_dim]

    if thinking_tokens.dim() != 2:
        print(f"Unexpected shape. Expected 2D tensor.")
        return None

    num_tokens, hidden_dim = thinking_tokens.shape

    analysis = {
        "model_path": model_path,
        "token_name": token_name,
        "num_tokens": num_tokens,
        "hidden_dim": hidden_dim,
        "statistics": {},
        "per_token": {},
    }

    # Global statistics
    print("\n" + "=" * 60)
    print("GLOBAL STATISTICS")
    print("=" * 60)

    global_mean = thinking_tokens.mean().item()
    global_std = thinking_tokens.std().item()
    global_min = thinking_tokens.min().item()
    global_max = thinking_tokens.max().item()
    global_norm = torch.norm(thinking_tokens).item()

    print(f"Mean: {global_mean:.6f}")
    print(f"Std:  {global_std:.6f}")
    print(f"Min:  {global_min:.6f}")
    print(f"Max:  {global_max:.6f}")
    print(f"Frobenius norm: {global_norm:.4f}")

    analysis["statistics"] = {
        "mean": global_mean,
        "std": global_std,
        "min": global_min,
        "max": global_max,
        "frobenius_norm": global_norm,
    }

    # Per-token statistics
    print("\n" + "=" * 60)
    print("PER-TOKEN STATISTICS")
    print("=" * 60)

    token_norms = torch.norm(thinking_tokens, dim=1).numpy()
    token_means = thinking_tokens.mean(dim=1).numpy()
    token_stds = thinking_tokens.std(dim=1).numpy()

    print(f"\n{'Token':<8} {'Norm':<12} {'Mean':<12} {'Std':<12}")
    print("-" * 44)

    # Print first 10 and last 10
    for i in range(min(10, num_tokens)):
        print(f"{i:<8} {token_norms[i]:<12.4f} {token_means[i]:<12.6f} {token_stds[i]:<12.6f}")

    if num_tokens > 20:
        print("...")
        for i in range(num_tokens - 10, num_tokens):
            print(f"{i:<8} {token_norms[i]:<12.4f} {token_means[i]:<12.6f} {token_stds[i]:<12.6f}")

    # Store all token stats
    for i in range(num_tokens):
        analysis["per_token"][f"token_{i:03d}"] = {
            "norm": float(token_norms[i]),
            "mean": float(token_means[i]),
            "std": float(token_stds[i]),
        }

    # Token similarity analysis
    print("\n" + "=" * 60)
    print("TOKEN SIMILARITY ANALYSIS")
    print("=" * 60)

    # Cosine similarity matrix
    normed = thinking_tokens / (torch.norm(thinking_tokens, dim=1, keepdim=True) + 1e-8)
    similarity = torch.mm(normed, normed.T).numpy()

    # Average pairwise similarity (excluding diagonal)
    mask = ~np.eye(num_tokens, dtype=bool)
    avg_similarity = similarity[mask].mean()
    max_similarity = similarity[mask].max()
    min_similarity = similarity[mask].min()

    print(f"Average pairwise cosine similarity: {avg_similarity:.4f}")
    print(f"Max pairwise similarity: {max_similarity:.4f}")
    print(f"Min pairwise similarity: {min_similarity:.4f}")

    analysis["statistics"]["avg_pairwise_similarity"] = float(avg_similarity)
    analysis["statistics"]["max_pairwise_similarity"] = float(max_similarity)
    analysis["statistics"]["min_pairwise_similarity"] = float(min_similarity)

    # Find most similar pairs
    np.fill_diagonal(similarity, -1)  # Exclude self-similarity
    flat_idx = np.argsort(similarity.flatten())[::-1]
    top_pairs = []
    for idx in flat_idx[:5]:
        i, j = divmod(idx, num_tokens)
        if i < j:  # Avoid duplicates
            top_pairs.append((i, j, similarity[i, j]))

    print("\nMost similar token pairs:")
    for i, j, sim in top_pairs:
        print(f"  Token {i} ↔ Token {j}: {sim:.4f}")

    analysis["statistics"]["most_similar_pairs"] = [[int(i), int(j), float(s)] for i, j, s in top_pairs]

    # PCA analysis to understand structure
    print("\n" + "=" * 60)
    print("DIMENSIONALITY ANALYSIS")
    print("=" * 60)

    # Compute SVD to understand variance structure
    U, S, Vh = torch.linalg.svd(thinking_tokens)
    singular_values = S.numpy()

    # Cumulative variance explained
    var_explained = singular_values ** 2
    var_explained = var_explained / var_explained.sum() * 100
    cumulative_var = np.cumsum(var_explained)

    # How many components for 80%, 90%, 95%?
    dims_for_80 = np.argmax(cumulative_var >= 80) + 1
    dims_for_90 = np.argmax(cumulative_var >= 90) + 1
    dims_for_95 = np.argmax(cumulative_var >= 95) + 1

    print(f"Dimensions for 80% variance: {dims_for_80}")
    print(f"Dimensions for 90% variance: {dims_for_90}")
    print(f"Dimensions for 95% variance: {dims_for_95}")

    analysis["statistics"]["dims_for_80pct_var"] = int(dims_for_80)
    analysis["statistics"]["dims_for_90pct_var"] = int(dims_for_90)
    analysis["statistics"]["dims_for_95pct_var"] = int(dims_for_95)

    # Create visualizations
    print("\nGenerating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Token norms
    ax1 = axes[0, 0]
    ax1.bar(range(num_tokens), token_norms)
    ax1.set_xlabel("Token Index")
    ax1.set_ylabel("L2 Norm")
    ax1.set_title("Per-Token Norm Distribution")
    ax1.axhline(y=token_norms.mean(), color="r", linestyle="--", label=f"Mean: {token_norms.mean():.2f}")
    ax1.legend()

    # Plot 2: Similarity heatmap
    ax2 = axes[0, 1]
    np.fill_diagonal(similarity, 1)  # Restore diagonal
    im = ax2.imshow(similarity, cmap="RdBu_r", vmin=-1, vmax=1)
    ax2.set_xlabel("Token Index")
    ax2.set_ylabel("Token Index")
    ax2.set_title("Pairwise Cosine Similarity")
    plt.colorbar(im, ax=ax2)

    # Plot 3: Singular value spectrum
    ax3 = axes[1, 0]
    ax3.plot(range(1, len(singular_values) + 1), singular_values, "b-", linewidth=2)
    ax3.set_xlabel("Component")
    ax3.set_ylabel("Singular Value")
    ax3.set_title("Singular Value Spectrum")
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Cumulative variance
    ax4 = axes[1, 1]
    ax4.plot(range(1, len(cumulative_var) + 1), cumulative_var, "b-", linewidth=2)
    ax4.axhline(y=80, color="r", linestyle="--", alpha=0.5, label="80%")
    ax4.axhline(y=90, color="orange", linestyle="--", alpha=0.5, label="90%")
    ax4.axhline(y=95, color="g", linestyle="--", alpha=0.5, label="95%")
    ax4.axvline(x=dims_for_80, color="r", linestyle=":", alpha=0.5)
    ax4.axvline(x=dims_for_90, color="orange", linestyle=":", alpha=0.5)
    ax4.axvline(x=dims_for_95, color="g", linestyle=":", alpha=0.5)
    ax4.set_xlabel("Number of Components")
    ax4.set_ylabel("Cumulative Variance Explained (%)")
    ax4.set_title("Cumulative Variance Explained by PCA")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_path / "thinking_tokens_analysis.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to: {plot_path}")
    plt.close()

    # Save analysis
    analysis_path = output_path / "thinking_tokens_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved analysis to: {analysis_path}")

    # Save raw data
    tokens_path = output_path / "thinking_tokens.npy"
    np.save(tokens_path, thinking_tokens.numpy())
    print(f"Saved tokens to: {tokens_path}")

    # Cleanup
    del pipe

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    return analysis


def compare_generation_with_without_thinking(
    model_path: str = "models/LTX-2",
    output_dir: str = "experiments/results/thinking_tokens",
):
    """
    Compare video generation with and without thinking tokens.

    This is a more expensive experiment that requires inference.
    """
    import time

    from diffusers import LTX2Pipeline
    from diffusers.utils import export_to_video

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generation Comparison: With/Without Thinking Tokens")
    print("=" * 60)

    # Test prompt
    test_prompt = (
        "A golden retriever runs through a sun-dappled park, its fur gleaming in warm afternoon light. "
        "The camera tracks alongside as the dog bounds across lush green grass, tongue out and tail wagging energetically. "
        "Birds chirp softly in the background as leaves rustle in a gentle breeze."
    )

    # Load pipeline
    print("\nLoading pipeline...")
    pipe = LTX2Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    results = {}

    # Generate WITH thinking tokens (baseline)
    print("\n[1/2] Generating WITH thinking tokens (baseline)...")
    start_time = time.time()

    generator = torch.Generator(device="cpu").manual_seed(42)
    output_with = pipe(
        prompt=test_prompt,
        height=512,
        width=768,
        num_frames=33,
        num_inference_steps=25,
        guidance_scale=3.0,
        generator=generator,
    )
    frames_with = output_with.frames[0]
    gen_time_with = time.time() - start_time

    video_path_with = output_path / "generation_with_thinking.mp4"
    export_to_video(frames_with, str(video_path_with), fps=24)
    print(f"  Time: {gen_time_with:.1f}s | Saved: {video_path_with.name}")

    # Compute statistics
    frames_array = np.stack([np.array(f) for f in frames_with])
    results["with_thinking"] = {
        "generation_time": gen_time_with,
        "mean_brightness": float(frames_array.mean()),
        "std": float(frames_array.std()),
    }

    # Generate WITHOUT thinking tokens
    # This requires hooking into the text encoding process
    print("\n[2/2] Generating WITHOUT thinking tokens...")

    # We need to modify the text encoding to skip thinking tokens
    # This is model-specific and may not work with all architectures
    # For now, we'll try masking out thinking token positions in attention

    # TODO: Implement proper thinking token removal
    # This requires understanding the specific architecture

    print("  NOTE: Full implementation requires model-specific hooks.")
    print("  Skipping generation without thinking tokens for now.")

    # Save results
    results_path = output_path / "generation_comparison.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Cleanup
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze LTX-2 thinking tokens"
    )
    parser.add_argument(
        "--model-path",
        default="models/LTX-2",
        help="Path to LTX-2 model",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results/thinking_tokens",
        help="Output directory",
    )
    parser.add_argument(
        "--compare-generation",
        action="store_true",
        help="Also compare generation with/without thinking tokens (expensive)",
    )
    args = parser.parse_args()

    # Always run analysis
    analysis = analyze_thinking_tokens(
        model_path=args.model_path,
        output_dir=args.output_dir,
    )

    # Optionally run generation comparison
    if args.compare_generation:
        compare_generation_with_without_thinking(
            model_path=args.model_path,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
