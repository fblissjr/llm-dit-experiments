#!/usr/bin/env python3
"""
Embedding analysis and experimentation CLI.

Commands:
  extract     Extract and save embeddings from prompts
  visualize   Generate t-SNE/UMAP visualization
  compare     Compare embeddings (cosine similarity, MSE)
  steer       Compute steering vectors from prompt pairs
  layers      Layer-wise analysis (compare -1, -2, -3)
  validate    Validate model compatibility with Z-Image

Usage:
    # Extract embeddings from prompts
    uv run scripts/embeddings.py extract \\
      --model-path /path/to/z-image \\
      --prompts "A cat" "A dog" "A sunset" \\
      --output embeddings.safetensors

    # Visualize embeddings (requires: uv sync --extra analysis)
    uv run scripts/embeddings.py visualize \\
      --input embeddings.safetensors \\
      --method tsne \\
      --output visualization.png

    # Compare two embedding files
    uv run scripts/embeddings.py compare \\
      --embeddings-a cat.safetensors \\
      --embeddings-b dog.safetensors

    # Extract steering vector
    uv run scripts/embeddings.py steer \\
      --positive "A photorealistic cat" \\
      --negative "A cartoon cat" \\
      --model-path /path/to/z-image \\
      --output style_vector.safetensors

    # Compare layers
    uv run scripts/embeddings.py layers \\
      --model-path /path/to/z-image \\
      --prompt "A cat sleeping" \\
      --layers -1 -2 -3 -4

    # Validate model compatibility
    uv run scripts/embeddings.py validate \\
      --model-path Qwen/Qwen3-4B-Instruct-2507
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_analysis_deps():
    """Check if analysis dependencies are installed."""
    try:
        import matplotlib  # noqa: F401
        import sklearn  # noqa: F401

        return True
    except ImportError:
        return False


def get_encoder(model_path: str, device: str = "auto"):
    """Load the text encoder."""
    from llm_dit.encoders import ZImageTextEncoder

    logger.info(f"Loading encoder from {model_path}...")

    # Resolve device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    encoder = ZImageTextEncoder.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
    logger.info(f"Encoder loaded on {device}")
    return encoder


def cmd_extract(args):
    """Extract embeddings from prompts."""
    from llm_dit.utils.embeddings import save_embeddings, compute_stats

    encoder = get_encoder(args.model_path, args.device)

    embeddings_list = []
    for i, prompt in enumerate(args.prompts):
        logger.info(f"[{i + 1}/{len(args.prompts)}] Encoding: {prompt[:50]}...")

        start = time.time()
        output = encoder.encode(
            prompt,
            enable_thinking=args.enable_thinking,
            layer_index=args.layer,
        )
        elapsed = time.time() - start

        emb = output.embeddings[0]
        stats = compute_stats(emb)
        logger.info(f"  Shape: {stats.shape}, Time: {elapsed:.2f}s")
        logger.info(f"  Stats: mean={stats.mean:.4f}, std={stats.std:.4f}")

        embeddings_list.append(emb.cpu())

    # Save
    metadata = {
        "model_path": args.model_path,
        "layer": args.layer,
        "enable_thinking": args.enable_thinking,
    }

    output_path = save_embeddings(
        embeddings_list,
        args.output,
        metadata=metadata,
        prompts=args.prompts,
    )
    logger.info(f"Saved {len(embeddings_list)} embeddings to {output_path}")


def cmd_visualize(args):
    """Visualize embeddings with t-SNE or UMAP."""
    if not check_analysis_deps():
        logger.error("Visualization requires analysis dependencies.")
        logger.error("Install with: uv sync --extra analysis")
        return 1

    import matplotlib.pyplot as plt
    import numpy as np
    from llm_dit.utils.embeddings import load_embeddings, prepare_for_visualization

    # Load embeddings
    tensors, metadata = load_embeddings(args.input)
    prompts = metadata.get("prompts", [])

    # Collect all embeddings
    embeddings_list = []
    if "embeddings" in tensors:
        embeddings_list.append(tensors["embeddings"])
    else:
        i = 0
        while f"embeddings_{i}" in tensors:
            embeddings_list.append(tensors[f"embeddings_{i}"])
            i += 1

    if not embeddings_list:
        logger.error("No embeddings found in file")
        return 1

    logger.info(f"Loaded {len(embeddings_list)} embeddings")

    # Prepare for visualization (reduce to single vectors)
    X = prepare_for_visualization(embeddings_list, reduction=args.reduction)
    X_np = X.float().numpy()

    logger.info(f"Data shape for visualization: {X_np.shape}")

    # Apply dimensionality reduction
    if args.method == "tsne":
        from sklearn.manifold import TSNE

        logger.info("Running t-SNE...")
        reducer = TSNE(
            n_components=2,
            perplexity=min(30, len(X_np) - 1),
            random_state=args.seed,
        )
        X_2d = reducer.fit_transform(X_np)
    elif args.method == "umap":
        try:
            import umap

            logger.info("Running UMAP...")
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(15, len(X_np) - 1),
                random_state=args.seed,
            )
            X_2d = reducer.fit_transform(X_np)
        except ImportError:
            logger.error("UMAP not installed. Use --method tsne or install umap-learn")
            return 1
    elif args.method == "pca":
        from sklearn.decomposition import PCA

        logger.info("Running PCA...")
        reducer = PCA(n_components=2, random_state=args.seed)
        X_2d = reducer.fit_transform(X_np)
    else:
        logger.error(f"Unknown method: {args.method}")
        return 1

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.7)

    # Add labels if we have prompts
    if prompts and len(prompts) == len(X_2d):
        for i, (x, y) in enumerate(X_2d):
            label = prompts[i][:30] + "..." if len(prompts[i]) > 30 else prompts[i]
            ax.annotate(
                label,
                (x, y),
                fontsize=8,
                alpha=0.8,
                xytext=(5, 5),
                textcoords="offset points",
            )

    ax.set_title(f"Embedding Visualization ({args.method.upper()})")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    logger.info(f"Visualization saved to {args.output}")


def cmd_compare(args):
    """Compare two embedding files."""
    from llm_dit.utils.embeddings import (
        load_embeddings,
        compute_cosine_similarity,
        compute_mse,
        compute_stats,
    )

    # Load both files
    tensors_a, meta_a = load_embeddings(args.embeddings_a)
    tensors_b, meta_b = load_embeddings(args.embeddings_b)

    # Get first embedding from each
    emb_a = tensors_a.get("embeddings", tensors_a.get("embeddings_0"))
    emb_b = tensors_b.get("embeddings", tensors_b.get("embeddings_0"))

    if emb_a is None or emb_b is None:
        logger.error("Could not load embeddings from files")
        return 1

    # Stats
    stats_a = compute_stats(emb_a)
    stats_b = compute_stats(emb_b)

    print("\n=== Embedding Comparison ===\n")
    print(f"File A: {args.embeddings_a}")
    print(f"  Shape: {stats_a.shape}")
    print(f"  Stats: mean={stats_a.mean:.4f}, std={stats_a.std:.4f}")
    print(f"  Range: [{stats_a.min:.4f}, {stats_a.max:.4f}]")

    print(f"\nFile B: {args.embeddings_b}")
    print(f"  Shape: {stats_b.shape}")
    print(f"  Stats: mean={stats_b.mean:.4f}, std={stats_b.std:.4f}")
    print(f"  Range: [{stats_b.min:.4f}, {stats_b.max:.4f}]")

    # Similarity metrics
    print("\n=== Similarity Metrics ===\n")

    for reduction in ["mean", "last"]:
        cosine = compute_cosine_similarity(emb_a, emb_b, reduce=reduction)
        mse = compute_mse(emb_a, emb_b, reduce=reduction)
        print(f"Reduction: {reduction}")
        print(f"  Cosine similarity: {cosine:.4f}")
        print(f"  MSE: {mse:.6f}")

    # Prompts if available
    prompts_a = meta_a.get("prompts", [])
    prompts_b = meta_b.get("prompts", [])
    if prompts_a:
        print(f"\nPrompt A: {prompts_a[0]}")
    if prompts_b:
        print(f"Prompt B: {prompts_b[0]}")


def cmd_steer(args):
    """Extract steering vector from prompt pair."""
    from llm_dit.utils.embeddings import extract_steering_vector, save_embeddings, compute_stats

    encoder = get_encoder(args.model_path, args.device)

    # Encode both prompts
    logger.info(f"Encoding positive: {args.positive[:50]}...")
    pos_output = encoder.encode(args.positive, enable_thinking=args.enable_thinking)
    pos_emb = pos_output.embeddings[0]

    logger.info(f"Encoding negative: {args.negative[:50]}...")
    neg_output = encoder.encode(args.negative, enable_thinking=args.enable_thinking)
    neg_emb = neg_output.embeddings[0]

    # Extract steering vector
    logger.info("Computing steering vector...")
    steering_vec = extract_steering_vector(pos_emb, neg_emb, normalize=args.normalize)

    stats = compute_stats(steering_vec.unsqueeze(0))
    print(f"\n=== Steering Vector ===")
    print(f"Shape: {steering_vec.shape}")
    print(f"Norm: {steering_vec.norm().item():.4f}")
    print(f"Mean: {stats.mean:.4f}")
    print(f"Std: {stats.std:.4f}")

    # Save
    metadata = {
        "positive_prompt": args.positive,
        "negative_prompt": args.negative,
        "normalized": args.normalize,
        "model_path": args.model_path,
    }

    output_path = save_embeddings(
        steering_vec.cpu(),
        args.output,
        metadata=metadata,
    )
    logger.info(f"Steering vector saved to {output_path}")


def cmd_layers(args):
    """Compare embeddings from different layers."""
    from llm_dit.backends import TransformersBackend
    from llm_dit.conversation import Qwen3Formatter
    from llm_dit.utils.embeddings import compute_stats, compute_cosine_similarity

    # Load backend directly for layer access
    logger.info(f"Loading model from {args.model_path}...")

    backend = TransformersBackend.from_pretrained(
        args.model_path,
        model_subfolder="text_encoder",
        tokenizer_subfolder="tokenizer",
    )

    # Format prompt
    formatter = Qwen3Formatter()
    formatted = formatter.format(
        user_content=args.prompt,
        enable_thinking=args.enable_thinking,
    )

    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Formatted length: {len(formatted)} chars")

    # Extract embeddings from each layer
    results = []
    for layer in args.layers:
        logger.info(f"Extracting layer {layer}...")
        output = backend.encode([formatted], layer_index=layer)
        emb = output.embeddings[0]
        stats = compute_stats(emb)
        results.append((layer, emb, stats))

    # Print comparison
    print("\n=== Layer Comparison ===\n")
    print(f"{'Layer':<8} {'Shape':<15} {'Mean':<10} {'Std':<10} {'Norm':<10}")
    print("-" * 55)
    for layer, emb, stats in results:
        print(f"{layer:<8} {str(stats.shape):<15} {stats.mean:<10.4f} {stats.std:<10.4f} {stats.norm:<10.4f}")

    # Cosine similarity matrix
    if len(results) > 1:
        print("\n=== Cosine Similarity Between Layers ===\n")
        print(f"{'':>8}", end="")
        for layer, _, _ in results:
            print(f"{layer:>8}", end="")
        print()

        for i, (layer_i, emb_i, _) in enumerate(results):
            print(f"{layer_i:>8}", end="")
            for j, (layer_j, emb_j, _) in enumerate(results):
                sim = compute_cosine_similarity(emb_i, emb_j)
                print(f"{sim:>8.4f}", end="")
            print()

    # Optional visualization
    if args.output and check_analysis_deps():
        import matplotlib.pyplot as plt
        import numpy as np
        from llm_dit.utils.embeddings import reduce_embeddings

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Stats by layer
        layers = [r[0] for r in results]
        means = [r[2].mean for r in results]
        stds = [r[2].std for r in results]
        norms = [r[2].norm for r in results]

        ax1 = axes[0]
        x = np.arange(len(layers))
        width = 0.25

        ax1.bar(x - width, means, width, label="Mean", alpha=0.8)
        ax1.bar(x, stds, width, label="Std", alpha=0.8)
        ax1.bar(x + width, [n / 100 for n in norms], width, label="Norm/100", alpha=0.8)

        ax1.set_xlabel("Layer Index")
        ax1.set_ylabel("Value")
        ax1.set_title("Embedding Statistics by Layer")
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(l) for l in layers])
        ax1.legend()

        # Plot 2: Cosine similarity heatmap
        ax2 = axes[1]
        sim_matrix = np.zeros((len(results), len(results)))
        for i, (_, emb_i, _) in enumerate(results):
            for j, (_, emb_j, _) in enumerate(results):
                sim_matrix[i, j] = compute_cosine_similarity(emb_i, emb_j)

        im = ax2.imshow(sim_matrix, cmap="viridis", vmin=0.5, vmax=1.0)
        ax2.set_xticks(range(len(layers)))
        ax2.set_yticks(range(len(layers)))
        ax2.set_xticklabels([str(l) for l in layers])
        ax2.set_yticklabels([str(l) for l in layers])
        ax2.set_title("Cosine Similarity Between Layers")
        plt.colorbar(im, ax=ax2)

        plt.tight_layout()
        plt.savefig(args.output, dpi=150)
        logger.info(f"Layer analysis saved to {args.output}")


def cmd_validate(args):
    """Validate model compatibility with Z-Image."""
    from llm_dit.utils.model_compat import validate_model_path

    logger.info(f"Validating: {args.model_path}")

    result = validate_model_path(args.model_path, subfolder=args.subfolder)

    print("\n" + str(result))

    if result.compatible:
        logger.info("Model is compatible with Z-Image")
        return 0
    else:
        logger.error("Model is NOT compatible with Z-Image")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Embedding analysis and experimentation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # === extract ===
    p_extract = subparsers.add_parser("extract", help="Extract embeddings from prompts")
    p_extract.add_argument("--model-path", required=True, help="Path to Z-Image model")
    p_extract.add_argument("--prompts", nargs="+", required=True, help="Prompts to encode")
    p_extract.add_argument("--output", required=True, help="Output safetensors file")
    p_extract.add_argument("--device", default="auto", help="Device (auto/cpu/cuda/mps)")
    p_extract.add_argument("--layer", type=int, default=-2, help="Layer to extract (default: -2)")
    p_extract.add_argument("--enable-thinking", action="store_true", help="Add thinking block")
    p_extract.set_defaults(func=cmd_extract)

    # === visualize ===
    p_viz = subparsers.add_parser("visualize", help="Visualize embeddings")
    p_viz.add_argument("--input", required=True, help="Input safetensors file")
    p_viz.add_argument("--output", default="visualization.png", help="Output image")
    p_viz.add_argument("--method", default="tsne", choices=["tsne", "umap", "pca"], help="Reduction method")
    p_viz.add_argument("--reduction", default="mean", choices=["mean", "last", "max"], help="Sequence reduction")
    p_viz.add_argument("--seed", type=int, default=42, help="Random seed")
    p_viz.set_defaults(func=cmd_visualize)

    # === compare ===
    p_compare = subparsers.add_parser("compare", help="Compare embedding files")
    p_compare.add_argument("--embeddings-a", required=True, help="First embeddings file")
    p_compare.add_argument("--embeddings-b", required=True, help="Second embeddings file")
    p_compare.set_defaults(func=cmd_compare)

    # === steer ===
    p_steer = subparsers.add_parser("steer", help="Extract steering vector")
    p_steer.add_argument("--model-path", required=True, help="Path to Z-Image model")
    p_steer.add_argument("--positive", required=True, help="Positive concept prompt")
    p_steer.add_argument("--negative", required=True, help="Negative concept prompt")
    p_steer.add_argument("--output", required=True, help="Output safetensors file")
    p_steer.add_argument("--device", default="auto", help="Device (auto/cpu/cuda/mps)")
    p_steer.add_argument("--normalize", action="store_true", help="L2-normalize the vector")
    p_steer.add_argument("--enable-thinking", action="store_true", help="Add thinking block")
    p_steer.set_defaults(func=cmd_steer)

    # === layers ===
    p_layers = subparsers.add_parser("layers", help="Layer-wise analysis")
    p_layers.add_argument("--model-path", required=True, help="Path to Z-Image model")
    p_layers.add_argument("--prompt", required=True, help="Prompt to analyze")
    p_layers.add_argument("--layers", nargs="+", type=int, default=[-1, -2, -3, -4], help="Layers to compare")
    p_layers.add_argument("--output", default=None, help="Output visualization (optional)")
    p_layers.add_argument("--enable-thinking", action="store_true", help="Add thinking block")
    p_layers.set_defaults(func=cmd_layers)

    # === validate ===
    p_validate = subparsers.add_parser("validate", help="Validate model compatibility")
    p_validate.add_argument("--model-path", required=True, help="Path or HuggingFace ID")
    p_validate.add_argument("--subfolder", default="text_encoder", help="Model subfolder")
    p_validate.set_defaults(func=cmd_validate)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
