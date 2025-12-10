#!/usr/bin/env python3
"""
Analyze hidden layer representations in Qwen3-4B.

Measures cosine similarity between layers to identify:
- Where semantic content forms (low similarity to neighbors = high info gain)
- The "sweet spot" for prompt-adhering embeddings
- Layer clustering patterns

Usage:
    uv run experiments/analyze_layers.py --model-path /path/to/model "A cat sleeping"
    uv run experiments/analyze_layers.py --model-path /path/to/model --all-layers "prompt"
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path


def compute_layer_similarities(
    model_path: str,
    prompt: str,
    layers_to_analyze: list[int] | None = None,
    device: str = "cuda",
) -> dict:
    """
    Compute pairwise cosine similarities between hidden layer outputs.

    Returns dict with:
    - similarities: NxN matrix of cosine similarities
    - layer_indices: list of layer indices analyzed
    - token_count: number of tokens in prompt
    - layer_norms: L2 norm of each layer's output (indicator of "energy")
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    token_count = inputs.input_ids.shape[1]
    print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print(f"Token count: {token_count}")

    # Get all hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # Tuple of (batch, seq, hidden_dim)
    num_layers = len(hidden_states)
    print(f"Total layers: {num_layers} (including embedding layer)")

    # Default: analyze every 3rd layer
    if layers_to_analyze is None:
        layers_to_analyze = list(range(0, num_layers, 3))

    # Extract and flatten hidden states for analysis
    layer_embeddings = []
    for layer_idx in layers_to_analyze:
        # Mean pool across sequence dimension
        hidden = hidden_states[layer_idx][0]  # (seq, hidden_dim)
        pooled = hidden.mean(dim=0)  # (hidden_dim,)
        layer_embeddings.append(pooled)

    layer_embeddings = torch.stack(layer_embeddings)  # (num_layers, hidden_dim)

    # Compute pairwise cosine similarities
    normalized = F.normalize(layer_embeddings, p=2, dim=1)
    similarities = torch.mm(normalized, normalized.t())  # (num_layers, num_layers)

    # Compute L2 norms (embedding "energy")
    norms = torch.norm(layer_embeddings, p=2, dim=1)

    # Compute neighbor similarities (how much each layer differs from previous)
    neighbor_sims = []
    for i in range(1, len(layers_to_analyze)):
        sim = F.cosine_similarity(
            layer_embeddings[i : i + 1], layer_embeddings[i - 1 : i], dim=1
        )
        neighbor_sims.append(sim.item())

    return {
        "similarities": similarities.cpu().numpy(),
        "layer_indices": layers_to_analyze,
        "token_count": token_count,
        "layer_norms": norms.cpu().numpy(),
        "neighbor_similarities": neighbor_sims,
        "num_total_layers": num_layers,
    }


def print_analysis(results: dict) -> None:
    """Print analysis results in a readable format."""
    layer_indices = results["layer_indices"]
    similarities = results["similarities"]
    norms = results["layer_norms"]
    neighbor_sims = results["neighbor_similarities"]

    print("\n" + "=" * 70)
    print("LAYER ANALYSIS RESULTS")
    print("=" * 70)

    # Layer norms (embedding energy)
    print("\nLayer Norms (embedding 'energy' - higher = more activated):")
    print("-" * 50)
    max_norm = max(norms)
    for i, (layer_idx, norm) in enumerate(zip(layer_indices, norms)):
        bar = "#" * int(30 * norm / max_norm)
        print(f"  Layer {layer_idx:3d}: {norm:8.2f} |{bar}")

    # Neighbor similarities (information gain)
    print("\nNeighbor Similarities (lower = more information change):")
    print("-" * 50)
    for i, sim in enumerate(neighbor_sims):
        layer_from = layer_indices[i]
        layer_to = layer_indices[i + 1]
        info_change = 1 - sim
        bar = "#" * int(30 * info_change)
        print(f"  Layer {layer_from:3d} -> {layer_to:3d}: sim={sim:.4f} change={info_change:.4f} |{bar}")

    # Find interesting transition points
    print("\nKey Findings:")
    print("-" * 50)

    # Biggest information jumps
    if neighbor_sims:
        min_sim_idx = neighbor_sims.index(min(neighbor_sims))
        print(
            f"  Biggest change: Layer {layer_indices[min_sim_idx]} -> {layer_indices[min_sim_idx + 1]} "
            f"(sim={neighbor_sims[min_sim_idx]:.4f})"
        )

        max_sim_idx = neighbor_sims.index(max(neighbor_sims))
        print(
            f"  Most stable: Layer {layer_indices[max_sim_idx]} -> {layer_indices[max_sim_idx + 1]} "
            f"(sim={neighbor_sims[max_sim_idx]:.4f})"
        )

    # Similarity to last layer (task-specific abstraction)
    print("\nSimilarity to Last Layer (higher = more task-abstracted):")
    print("-" * 50)
    last_layer_sims = similarities[-1, :]
    for i, (layer_idx, sim) in enumerate(zip(layer_indices, last_layer_sims)):
        bar = "#" * int(30 * sim)
        print(f"  Layer {layer_idx:3d}: {sim:.4f} |{bar}")

    # Recommendations
    print("\nRecommendations for Image Generation:")
    print("-" * 50)

    # Find layers with moderate similarity to last (not too abstracted, not too raw)
    sweet_spot = []
    for i, sim in enumerate(last_layer_sims[:-1]):  # Exclude last layer itself
        if 0.7 < sim < 0.95:
            sweet_spot.append((layer_indices[i], sim))

    if sweet_spot:
        print("  Potential sweet spot layers (0.7 < sim_to_last < 0.95):")
        for layer_idx, sim in sweet_spot:
            print(f"    Layer {layer_idx} (sim={sim:.4f})")
    else:
        # Find the layers with maximum information content
        mid_idx = len(layer_indices) // 2
        print(f"  Try middle layers around: {layer_indices[mid_idx]}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze Qwen3-4B hidden layers")
    parser.add_argument("prompt", help="Prompt to analyze")
    parser.add_argument("--model-path", required=True, help="Path to model")
    parser.add_argument(
        "--all-layers", action="store_true", help="Analyze all layers (slow)"
    )
    parser.add_argument(
        "--layers",
        type=str,
        help="Comma-separated layer indices to analyze (e.g., '0,6,12,18,24,30,35')",
    )
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--output", "-o", help="Save results to JSON file")

    args = parser.parse_args()

    # Determine layers to analyze
    layers = None
    if args.all_layers:
        layers = None  # Will be set after loading model
    elif args.layers:
        layers = [int(x.strip()) for x in args.layers.split(",")]

    results = compute_layer_similarities(
        model_path=args.model_path,
        prompt=args.prompt,
        layers_to_analyze=layers if not args.all_layers else list(range(37)),
        device=args.device,
    )

    print_analysis(results)

    if args.output:
        import json
        import numpy as np

        # Convert numpy arrays to lists for JSON serialization
        output_data = {
            "prompt": args.prompt,
            "token_count": results["token_count"],
            "layer_indices": results["layer_indices"],
            "layer_norms": results["layer_norms"].tolist(),
            "neighbor_similarities": results["neighbor_similarities"],
            "similarity_matrix": results["similarities"].tolist(),
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
