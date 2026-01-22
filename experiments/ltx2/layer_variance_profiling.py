#!/usr/bin/env python3
"""
LTX-2 Layer Variance Profiling (Experiment 1.1)

Last Updated: 2026-01-15

Characterizes how hidden states differ across layers for DiT-relevant prompts.

Hypothesis: Early layers (1-15) encode phonetic/positional information,
middle layers (16-35) encode syntactic structure, late layers (36-48)
encode semantic/abstract concepts.

Usage:
    uv run python experiments/ltx2/layer_variance_profiling.py
    uv run python experiments/ltx2/layer_variance_profiling.py --num-prompts 50
    uv run python experiments/ltx2/layer_variance_profiling.py --save-plots
"""

import argparse
import gc
from pathlib import Path

import torch
import numpy as np


# Default diverse prompt set covering different categories
DEFAULT_PROMPTS = [
    # Simple Objects (baseline)
    "a cat",
    "a red ball",
    "a tree",
    "a dog running",
    "a blue car",

    # Complex Scenes
    "a cat chasing a red ball under a tree in a sunny park",
    "an old man sitting on a bench reading a newspaper",
    "children playing in a fountain on a hot summer day",
    "a busy city street at night with neon signs",
    "a sailboat on calm water at sunset",

    # Abstract Concepts
    "happiness",
    "the concept of time",
    "entropy",
    "loneliness",
    "freedom",

    # Action Sequences
    "a man walks, picks up a phone, answers it",
    "a bird takes flight from a tree branch",
    "water droplets fall and splash",
    "a dancer spins gracefully",
    "leaves falling from a tree in autumn",

    # Spatial Relationships
    "a cat behind a dog in front of a house",
    "three apples arranged in a triangle",
    "a lamp above a table next to a window",
    "stairs leading up to a red door",
    "a bridge over a river through mountains",

    # Attributes
    "a large blue elephant",
    "three small red apples",
    "an ancient weathered stone wall",
    "a shimmering golden sunset",
    "a tiny delicate flower",

    # Phonetic Focus
    "buzz whirr click bang",
    "shh whisper sigh",
    "boom crash thunder",
    "drip drop splash",
    "crunch munch chomp",

    # Text Rendering
    "a sign that says HELLO",
    "graffiti reading PEACE",
    "a book with WISDOM on the cover",
    "a billboard showing SALE",
    "a neon sign spelling OPEN",

    # Video-specific (motion/temporal)
    "A cat walking through a sunny garden",
    "waves crashing on a rocky shore",
    "clouds drifting across a blue sky",
    "flames dancing in a fireplace",
    "snow falling gently on a forest",
]


def compute_layer_statistics(layer_stack: torch.Tensor, attention_mask: torch.Tensor) -> dict:
    """
    Compute statistics for each layer in the stack.

    Args:
        layer_stack: [B, T, D, L] tensor of hidden states
        attention_mask: [B, T] tensor of valid token mask

    Returns:
        Dict with per-layer statistics
    """
    B, T, D, L = layer_stack.shape
    device = layer_stack.device

    # Mask out padding tokens for statistics
    mask = attention_mask.unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]
    masked_stack = layer_stack * mask

    # Valid token count per batch
    valid_tokens = attention_mask.sum(dim=1)  # [B]

    stats = {
        'per_layer_mean': [],
        'per_layer_std': [],
        'per_layer_l2_norm': [],
        'per_layer_dim_variance': [],
        'inter_layer_cosine': torch.zeros(L, L, device=device),
    }

    # Per-layer statistics
    for l in range(L):
        layer = masked_stack[..., l]  # [B, T, D]

        # Mean across all dimensions (masked)
        layer_sum = layer.sum()
        total_valid = valid_tokens.sum()
        layer_mean = layer_sum / (total_valid * D)
        stats['per_layer_mean'].append(layer_mean.item())

        # Std across tokens
        layer_std = layer.std()
        stats['per_layer_std'].append(layer_std.item())

        # L2 norm (average per token)
        token_norms = layer.norm(dim=-1)  # [B, T]
        masked_norms = token_norms * attention_mask
        avg_norm = masked_norms.sum() / valid_tokens.sum()
        stats['per_layer_l2_norm'].append(avg_norm.item())

        # Dimension variance (how much each dimension varies)
        dim_var = layer.var(dim=(0, 1))  # [D]
        stats['per_layer_dim_variance'].append(dim_var.mean().item())

    # Inter-layer cosine similarity matrix
    # Average across all valid tokens first
    layer_means = []
    for l in range(L):
        layer = masked_stack[..., l]  # [B, T, D]
        # Get mean representation for this layer
        layer_sum = layer.sum(dim=(0, 1))
        valid_count = valid_tokens.sum()
        layer_mean = layer_sum / valid_count
        layer_means.append(layer_mean)

    layer_means = torch.stack(layer_means, dim=0)  # [L, D]

    # Normalize for cosine similarity
    layer_means_norm = layer_means / (layer_means.norm(dim=-1, keepdim=True) + 1e-8)

    # Compute cosine similarity matrix
    stats['inter_layer_cosine'] = torch.mm(layer_means_norm, layer_means_norm.t())

    return stats


def compute_token_layer_variance(layer_stack: torch.Tensor, attention_mask: torch.Tensor) -> dict:
    """
    Compute per-token variance across layers.

    Args:
        layer_stack: [B, T, D, L] tensor
        attention_mask: [B, T] tensor

    Returns:
        Dict with token-level statistics
    """
    B, T, D, L = layer_stack.shape

    # Per-token variance across layers (how much does representation change?)
    token_layer_var = layer_stack.var(dim=-1)  # [B, T, D]
    token_layer_var_mean = token_layer_var.mean(dim=-1)  # [B, T]

    # Mask invalid tokens
    token_layer_var_mean = token_layer_var_mean * attention_mask

    return {
        'token_variance': token_layer_var_mean,  # [B, T]
        'mean_token_variance': token_layer_var_mean.sum() / attention_mask.sum(),
    }


def run_profiling(
    prompts: list,
    model_id: str = "google/gemma-3-12b-it-qat-q4_0-unquantized",
    max_sequence_length: int = 128,
    save_plots: bool = False,
    output_dir: str = "experiments/results/ltx2",
):
    """
    Run layer variance profiling on a set of prompts.
    """
    from llm_dit.encoders.gemma3 import Gemma3Encoder

    print("=" * 60)
    print("LTX-2 Layer Variance Profiling")
    print("=" * 60)
    print(f"Number of prompts: {len(prompts)}")
    print(f"Model: {model_id}")
    print(f"Max sequence length: {max_sequence_length}")

    # Load encoder with memory constraints for RTX 4090
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

    # Aggregate statistics across all prompts
    all_layer_means = []
    all_layer_stds = []
    all_layer_norms = []
    all_cosine_matrices = []
    all_token_variances = []

    print(f"\nProcessing {len(prompts)} prompts...")

    for i, prompt in enumerate(prompts):
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(prompts)}] {prompt[:50]}...")

        # Get multi-layer hidden states
        result = encoder.encode_multilayer(
            prompt,
            return_projected=False,
        )

        layer_stack = result['layer_stack']  # [1, T, 3840, L]
        attention_mask = result['attention_mask']  # [1, T]

        # Compute statistics
        stats = compute_layer_statistics(layer_stack, attention_mask)
        token_stats = compute_token_layer_variance(layer_stack, attention_mask)

        all_layer_means.append(stats['per_layer_mean'])
        all_layer_stds.append(stats['per_layer_std'])
        all_layer_norms.append(stats['per_layer_l2_norm'])
        all_cosine_matrices.append(stats['inter_layer_cosine'].float().cpu().numpy())
        all_token_variances.append(token_stats['mean_token_variance'].item())

        # Clear cache periodically
        if (i + 1) % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Aggregate results
    num_layers = len(all_layer_means[0])

    results = {
        'num_prompts': len(prompts),
        'num_layers': num_layers,
        'layer_means': np.array(all_layer_means),  # [P, L]
        'layer_stds': np.array(all_layer_stds),  # [P, L]
        'layer_norms': np.array(all_layer_norms),  # [P, L]
        'cosine_matrices': np.array(all_cosine_matrices),  # [P, L, L]
        'token_variances': np.array(all_token_variances),  # [P]
    }

    # Compute aggregate metrics
    avg_cosine = results['cosine_matrices'].mean(axis=0)  # [L, L]
    avg_norms = results['layer_norms'].mean(axis=0)  # [L]
    avg_stds = results['layer_stds'].mean(axis=0)  # [L]

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    # Layer group analysis
    early = avg_norms[:16].mean()
    middle = avg_norms[16:32].mean()
    late = avg_norms[32:].mean()
    print(f"\nAverage L2 Norm by Layer Group:")
    print(f"  Early (0-15):  {early:.4f}")
    print(f"  Middle (16-31): {middle:.4f}")
    print(f"  Late (32-47):   {late:.4f}")

    # Cosine similarity analysis
    early_early = avg_cosine[:16, :16].mean()
    late_late = avg_cosine[32:, 32:].mean()
    early_late = avg_cosine[:16, 32:].mean()
    print(f"\nCosine Similarity Between Layer Groups:")
    print(f"  Early-Early:  {early_early:.4f}")
    print(f"  Late-Late:    {late_late:.4f}")
    print(f"  Early-Late:   {early_late:.4f}")

    # Adjacent layer similarity
    adjacent_sim = [avg_cosine[i, i+1] for i in range(num_layers-1)]
    print(f"\nAdjacent Layer Similarity:")
    print(f"  Mean: {np.mean(adjacent_sim):.4f}")
    print(f"  Min:  {np.min(adjacent_sim):.4f} (layers {np.argmin(adjacent_sim)}-{np.argmin(adjacent_sim)+1})")
    print(f"  Max:  {np.max(adjacent_sim):.4f} (layers {np.argmax(adjacent_sim)}-{np.argmax(adjacent_sim)+1})")

    # Token variance
    print(f"\nToken-Level Layer Variance:")
    print(f"  Mean: {results['token_variances'].mean():.4f}")
    print(f"  Std:  {results['token_variances'].std():.4f}")

    # Success criteria check
    print("\n" + "=" * 60)
    print("Success Criteria Check")
    print("=" * 60)

    # Check if layers show meaningful differentiation
    if early_late < 0.9 * early_early:
        print("[PASS] Early and late layers show meaningful differentiation")
    else:
        print("[WARN] Early and late layers too similar - routing may not help")

    if np.std(avg_norms) > 0.1 * np.mean(avg_norms):
        print("[PASS] Layers show variance in representation norms")
    else:
        print("[WARN] Layers have similar norms - limited routing potential")

    # Save results
    if save_plots:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save numpy arrays
        np.savez(
            output_path / "layer_profiling_results.npz",
            **results,
            avg_cosine=avg_cosine,
            avg_norms=avg_norms,
            avg_stds=avg_stds,
        )
        print(f"\nResults saved to {output_path}/layer_profiling_results.npz")

        # Try to create plots if matplotlib available
        try:
            import matplotlib.pyplot as plt

            # Plot 1: Inter-layer cosine similarity heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(avg_cosine, cmap='viridis', aspect='auto')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Layer')
            ax.set_title('Inter-Layer Cosine Similarity')
            plt.colorbar(im, ax=ax)
            fig.savefig(output_path / "cosine_similarity_matrix.png", dpi=150)
            plt.close()

            # Plot 2: Layer norms
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(num_layers), avg_norms)
            ax.axvline(x=16, color='r', linestyle='--', alpha=0.5, label='Early/Middle')
            ax.axvline(x=32, color='r', linestyle='--', alpha=0.5, label='Middle/Late')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Average L2 Norm')
            ax.set_title('Layer-wise Representation Norm')
            ax.legend()
            fig.savefig(output_path / "layer_norms.png", dpi=150)
            plt.close()

            # Plot 3: Adjacent layer similarity
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(len(adjacent_sim)), adjacent_sim)
            ax.set_xlabel('Layer Transition')
            ax.set_ylabel('Cosine Similarity')
            ax.set_title('Adjacent Layer Similarity')
            fig.savefig(output_path / "adjacent_similarity.png", dpi=150)
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
    parser = argparse.ArgumentParser(description="LTX-2 Layer Variance Profiling")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Number of prompts to use (default: all default prompts)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/gemma-3-12b-it-qat-q4_0-unquantized",
        help="Gemma model ID",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots and results to disk",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/ltx2",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.num_prompts:
        prompts = prompts[:args.num_prompts]

    run_profiling(
        prompts=prompts,
        model_id=args.model_id,
        max_sequence_length=args.max_seq_len,
        save_plots=args.save_plots,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
