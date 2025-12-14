#!/usr/bin/env python3
"""
Compare Qwen3-4B vs Qwen3-Embedding-4B for Z-Image text conditioning.

This script isolates and tests different embedding approaches:
1. Qwen3-4B (baseline - what Z-Image was trained with)
2. Qwen3-Embedding-4B (optimized for embedding quality)
3. Instruction-aware Qwen3-Embedding-4B

Usage:
    # Quick comparison (stats only)
    uv run experiments/compare_embedding_models.py \
        --qwen3-path /path/to/Qwen3-4B \
        --embedding-path /path/to/Qwen3-Embedding-4B \
        --prompts "A cat" "A mountain landscape"

    # Full comparison with image generation
    uv run experiments/compare_embedding_models.py \
        --config config.toml \
        --qwen3-path /path/to/Qwen3-4B \
        --embedding-path /path/to/Qwen3-Embedding-4B \
        --generate-images \
        --output-dir experiments/results/embedding_comparison

last updated: 2025-12-14
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingStats:
    """Statistics for a set of embeddings."""

    model: str
    layer: int
    prompt: str
    num_tokens: int
    mean: float
    std: float
    min: float
    max: float
    # Per-dimension statistics
    dim_mean_mean: float  # Mean of per-dim means
    dim_mean_std: float   # Std of per-dim means
    dim_std_mean: float   # Mean of per-dim stds
    dim_std_std: float    # Std of per-dim stds


@dataclass
class ComparisonResult:
    """Comparison between two embedding sets."""

    prompt: str
    qwen3_stats: EmbeddingStats
    embedding_stats: EmbeddingStats
    # Similarity metrics
    global_cosine_sim: float
    per_dim_mean_correlation: float
    per_dim_std_correlation: float
    # Distribution alignment
    mean_diff: float
    std_ratio: float  # embedding_std / qwen3_std


def compute_stats(embeddings: torch.Tensor, model: str, layer: int, prompt: str) -> EmbeddingStats:
    """Compute comprehensive statistics for embeddings."""
    # Global stats
    mean = embeddings.mean().item()
    std = embeddings.std().item()
    min_val = embeddings.min().item()
    max_val = embeddings.max().item()

    # Per-dimension stats (across sequence)
    dim_means = embeddings.mean(dim=0)  # (hidden_dim,)
    dim_stds = embeddings.std(dim=0)    # (hidden_dim,)

    return EmbeddingStats(
        model=model,
        layer=layer,
        prompt=prompt,
        num_tokens=embeddings.shape[0],
        mean=mean,
        std=std,
        min=min_val,
        max=max_val,
        dim_mean_mean=dim_means.mean().item(),
        dim_mean_std=dim_means.std().item(),
        dim_std_mean=dim_stds.mean().item(),
        dim_std_std=dim_stds.std().item(),
    )


def compare_embeddings(
    qwen3_emb: torch.Tensor,
    embedding_emb: torch.Tensor,
    prompt: str,
    qwen3_layer: int,
    embedding_layer: int,
) -> ComparisonResult:
    """Compare two embedding sets comprehensively."""

    # Compute individual stats
    qwen3_stats = compute_stats(qwen3_emb, "qwen3-4b", qwen3_layer, prompt)
    embedding_stats = compute_stats(embedding_emb, "qwen3-embedding-4b", embedding_layer, prompt)

    # Handle length differences by truncating to shorter
    min_len = min(len(qwen3_emb), len(embedding_emb))
    q_trunc = qwen3_emb[:min_len]
    e_trunc = embedding_emb[:min_len]

    # Global cosine similarity (flatten and compare)
    global_cosine = F.cosine_similarity(
        q_trunc.flatten().unsqueeze(0),
        e_trunc.flatten().unsqueeze(0),
    ).item()

    # Per-dimension correlations
    q_dim_means = q_trunc.mean(dim=0)
    e_dim_means = e_trunc.mean(dim=0)
    q_dim_stds = q_trunc.std(dim=0)
    e_dim_stds = e_trunc.std(dim=0)

    # Pearson correlation of per-dim means
    q_centered = q_dim_means - q_dim_means.mean()
    e_centered = e_dim_means - e_dim_means.mean()
    mean_corr = (
        (q_centered * e_centered).sum() / (q_centered.norm() * e_centered.norm() + 1e-8)
    ).item()

    # Pearson correlation of per-dim stds
    q_std_centered = q_dim_stds - q_dim_stds.mean()
    e_std_centered = e_dim_stds - e_dim_stds.mean()
    std_corr = (
        (q_std_centered * e_std_centered).sum() / (q_std_centered.norm() * e_std_centered.norm() + 1e-8)
    ).item()

    return ComparisonResult(
        prompt=prompt,
        qwen3_stats=qwen3_stats,
        embedding_stats=embedding_stats,
        global_cosine_sim=global_cosine,
        per_dim_mean_correlation=mean_corr,
        per_dim_std_correlation=std_corr,
        mean_diff=embedding_stats.mean - qwen3_stats.mean,
        std_ratio=embedding_stats.std / (qwen3_stats.std + 1e-8),
    )


def extract_qwen3_embeddings(
    model_path: str,
    prompts: list[str],
    hidden_layer: int = -2,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Extract embeddings from Qwen3-4B."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading Qwen3-4B from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    results = {}
    for prompt in prompts:
        logger.info(f"  Encoding: {prompt[:50]}...")
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        hidden_states = outputs.hidden_states[hidden_layer]
        mask = inputs["attention_mask"][0].bool()
        embeddings = hidden_states[0, mask].cpu()
        results[prompt] = embeddings

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return results


def extract_embedding_model_embeddings(
    model_path: str,
    prompts: list[str],
    hidden_layer: int = -1,
    instruction: str | None = None,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Extract embeddings from Qwen3-Embedding-4B."""
    from transformers import AutoModel, AutoTokenizer

    logger.info(f"Loading Qwen3-Embedding-4B from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    results = {}
    for prompt in prompts:
        # Apply instruction prefix if provided
        if instruction:
            input_text = f"Instruct: {instruction}\nQuery: {prompt}"
        else:
            input_text = prompt

        logger.info(f"  Encoding: {prompt[:50]}...")
        inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        hidden_states = outputs.hidden_states[hidden_layer]
        mask = inputs["attention_mask"][0].bool()
        embeddings = hidden_states[0, mask].cpu()
        results[prompt] = embeddings

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return results


def print_comparison_table(results: list[ComparisonResult]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print("EMBEDDING COMPARISON RESULTS")
    print("=" * 100)

    for r in results:
        print(f"\nPrompt: {r.prompt[:60]}...")
        print("-" * 80)
        print(f"{'Metric':<30} {'Qwen3-4B':<20} {'Qwen3-Embedding':<20} {'Diff/Ratio':<15}")
        print("-" * 80)
        print(f"{'Num tokens':<30} {r.qwen3_stats.num_tokens:<20} {r.embedding_stats.num_tokens:<20}")
        print(f"{'Mean':<30} {r.qwen3_stats.mean:<20.4f} {r.embedding_stats.mean:<20.4f} {r.mean_diff:+.4f}")
        print(f"{'Std':<30} {r.qwen3_stats.std:<20.4f} {r.embedding_stats.std:<20.4f} {r.std_ratio:.4f}x")
        print(f"{'Min':<30} {r.qwen3_stats.min:<20.4f} {r.embedding_stats.min:<20.4f}")
        print(f"{'Max':<30} {r.qwen3_stats.max:<20.4f} {r.embedding_stats.max:<20.4f}")
        print("-" * 80)
        print(f"{'Global cosine similarity':<30} {r.global_cosine_sim:.4f}")
        print(f"{'Per-dim mean correlation':<30} {r.per_dim_mean_correlation:.4f}")
        print(f"{'Per-dim std correlation':<30} {r.per_dim_std_correlation:.4f}")

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    avg_cosine = sum(r.global_cosine_sim for r in results) / len(results)
    avg_mean_corr = sum(r.per_dim_mean_correlation for r in results) / len(results)
    avg_std_ratio = sum(r.std_ratio for r in results) / len(results)

    print(f"Average cosine similarity: {avg_cosine:.4f}")
    print(f"Average per-dim mean correlation: {avg_mean_corr:.4f}")
    print(f"Average std ratio (embedding/qwen3): {avg_std_ratio:.4f}")

    if avg_std_ratio < 0.5 or avg_std_ratio > 2.0:
        print(f"\nWARNING: Large std ratio ({avg_std_ratio:.2f}x) suggests embeddings need scaling")
        print(f"Recommended scale factor: {1.0 / avg_std_ratio:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Qwen3-4B vs Qwen3-Embedding-4B embeddings"
    )
    parser.add_argument(
        "--qwen3-path",
        type=str,
        required=True,
        help="Path to Qwen3-4B model",
    )
    parser.add_argument(
        "--embedding-path",
        type=str,
        required=True,
        help="Path to Qwen3-Embedding-4B model",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "A cat sleeping in sunlight",
            "A mountain landscape at sunset",
            "A woman with red hair in a blue dress",
            "An astronaut riding a horse on mars, photorealistic",
        ],
        help="Prompts to compare",
    )
    parser.add_argument(
        "--qwen3-layer",
        type=int,
        default=-2,
        help="Hidden layer for Qwen3-4B (default: -2)",
    )
    parser.add_argument(
        "--embedding-layer",
        type=int,
        default=-2,
        help="Hidden layer for Qwen3-Embedding-4B (default: -2)",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Instruction prefix for Qwen3-Embedding (optional)",
    )
    parser.add_argument(
        "--test-instructions",
        action="store_true",
        help="Test multiple instruction variants",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    # Extract Qwen3-4B embeddings
    logger.info("\n=== Extracting Qwen3-4B embeddings ===")
    qwen3_embeddings = extract_qwen3_embeddings(
        args.qwen3_path,
        args.prompts,
        hidden_layer=args.qwen3_layer,
        device=args.device,
    )

    # Test configurations
    configs = [
        {"layer": args.embedding_layer, "instruction": args.instruction, "name": "default"},
    ]

    if args.test_instructions:
        configs.extend([
            {"layer": -1, "instruction": None, "name": "layer-1_no-inst"},
            {"layer": -2, "instruction": None, "name": "layer-2_no-inst"},
            {
                "layer": -2,
                "instruction": "Generate a semantic embedding for text-to-image synthesis",
                "name": "layer-2_image-inst",
            },
            {
                "layer": -2,
                "instruction": "Encode this text capturing all visual and descriptive details",
                "name": "layer-2_visual-inst",
            },
        ])

    all_results = {}

    for config in configs:
        logger.info(f"\n=== Testing config: {config['name']} ===")

        embedding_embeddings = extract_embedding_model_embeddings(
            args.embedding_path,
            args.prompts,
            hidden_layer=config["layer"],
            instruction=config["instruction"],
            device=args.device,
        )

        # Compare
        results = []
        for prompt in args.prompts:
            result = compare_embeddings(
                qwen3_embeddings[prompt],
                embedding_embeddings[prompt],
                prompt,
                args.qwen3_layer,
                config["layer"],
            )
            results.append(result)

        print_comparison_table(results)
        all_results[config["name"]] = [asdict(r) for r in results]

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
