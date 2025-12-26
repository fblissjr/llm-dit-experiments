#!/usr/bin/env python3
"""
Per-dimension analysis of Qwen3-Embedding-4B vs Qwen3-4B embeddings.

Last updated: 2025-12-14

Despite 98% cosine similarity, Qwen3-Embedding-4B produces severe visual
artifacts (industrial/data-center backgrounds) when used with Z-Image DiT.

This script performs deep per-dimension analysis to identify the root cause:
1. Per-dimension mean/std correlation between models
2. Outlier dimension detection (extreme std ratios)
3. Dead dimensions (near-zero variance)
4. Hyperactive dimensions (extreme variance)
5. Distribution shape differences (kurtosis, skewness)
6. Attention weight sensitivity (which dims might the DiT focus on)

Usage:
    uv run experiments/analyze_embedding_perdim.py --quick
    uv run experiments/analyze_embedding_perdim.py --prompts "A cat" "A mountain"
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import torch.nn.functional as F
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Model paths from environment variables
QWEN3_4B_PATH = os.environ.get("QWEN3_PATH")
QWEN3_EMBEDDING_PATH = os.environ.get("QWEN3_EMBEDDING_PATH")

if not all([QWEN3_4B_PATH, QWEN3_EMBEDDING_PATH]):
    raise ValueError(
        "Set environment variables: QWEN3_PATH, QWEN3_EMBEDDING_PATH"
    )

# Test prompts - diverse set to capture various semantic patterns
DEFAULT_PROMPTS = [
    "A cat sleeping in sunlight",
    "A mountain landscape at sunset",
    "A futuristic city with neon lights",
    "An old man reading a book",
    "A bowl of fresh fruit",
]


def compute_per_dimension_stats(embeddings: torch.Tensor) -> dict:
    """
    Compute per-dimension statistics for embedding tensor.

    Args:
        embeddings: (seq_len, hidden_dim) tensor

    Returns:
        Dictionary with per-dimension statistics
    """
    # Flatten to (n_tokens, 2560)
    flat = embeddings.view(-1, embeddings.size(-1))

    # Per-dimension statistics across all tokens
    dim_means = flat.mean(dim=0)  # (2560,)
    dim_stds = flat.std(dim=0)    # (2560,)
    dim_mins = flat.min(dim=0).values
    dim_maxs = flat.max(dim=0).values
    dim_ranges = dim_maxs - dim_mins

    # Distribution shape (manual implementation to avoid scipy dependency)
    # Kurtosis = E[(X - mean)^4] / std^4 - 3
    flat_np = flat.cpu().numpy()
    dim_kurtosis = np.zeros(flat_np.shape[1])
    dim_skewness = np.zeros(flat_np.shape[1])

    for i in range(flat_np.shape[1]):
        col = flat_np[:, i]
        if dim_stds[i].item() > 1e-6:  # Avoid division by zero
            centered = col - dim_means[i].item()
            dim_kurtosis[i] = np.mean(centered**4) / (dim_stds[i].item()**4) - 3
            dim_skewness[i] = np.mean(centered**3) / (dim_stds[i].item()**3)
        else:
            dim_kurtosis[i] = 0.0
            dim_skewness[i] = 0.0

    # Identify special dimensions
    dead_dims = (dim_stds < 0.01).nonzero(as_tuple=True)[0].tolist()
    hyperactive_dims = (dim_stds > dim_stds.median() * 5).nonzero(as_tuple=True)[0].tolist()

    return {
        "means": dim_means.cpu().numpy(),
        "stds": dim_stds.cpu().numpy(),
        "mins": dim_mins.cpu().numpy(),
        "maxs": dim_maxs.cpu().numpy(),
        "ranges": dim_ranges.cpu().numpy(),
        "kurtosis": dim_kurtosis,
        "skewness": dim_skewness,
        "dead_dimensions": dead_dims,
        "hyperactive_dimensions": hyperactive_dims,
        "median_std": dim_stds.median().item(),
        "mean_std": dim_stds.mean().item(),
    }


def compare_dimension_distributions(
    emb_stats: dict,
    qwen3_stats: dict,
) -> dict:
    """
    Compare per-dimension distributions between two models.

    Args:
        emb_stats: Stats from Qwen3-Embedding-4B
        qwen3_stats: Stats from Qwen3-4B

    Returns:
        Dictionary with comparison metrics
    """
    emb_means = torch.from_numpy(emb_stats["means"])
    qwen3_means = torch.from_numpy(qwen3_stats["means"])
    emb_stds = torch.from_numpy(emb_stats["stds"])
    qwen3_stds = torch.from_numpy(qwen3_stats["stds"])

    # Mean correlation
    mean_correlation = F.cosine_similarity(
        emb_means.unsqueeze(0),
        qwen3_means.unsqueeze(0),
    ).item()

    # Std correlation
    std_correlation = F.cosine_similarity(
        emb_stds.unsqueeze(0),
        qwen3_stds.unsqueeze(0),
    ).item()

    # Per-dimension std ratios
    # Avoid division by zero
    safe_qwen3_stds = torch.where(qwen3_stds < 1e-6, torch.ones_like(qwen3_stds), qwen3_stds)
    std_ratios = emb_stds / safe_qwen3_stds

    # Outlier dimensions (extreme std ratios)
    outlier_threshold = 3.0  # 3x difference is significant
    outlier_high = (std_ratios > outlier_threshold).nonzero(as_tuple=True)[0].tolist()
    outlier_low = (std_ratios < 1.0 / outlier_threshold).nonzero(as_tuple=True)[0].tolist()

    # Top 10 most different dimensions by std ratio
    top_high_dims = torch.argsort(std_ratios, descending=True)[:10].tolist()
    top_low_dims = torch.argsort(std_ratios, descending=False)[:10].tolist()

    # Distribution shape differences
    kurtosis_diff = np.abs(emb_stats["kurtosis"] - qwen3_stats["kurtosis"])
    skewness_diff = np.abs(emb_stats["skewness"] - qwen3_stats["skewness"])

    # Dimensions with extreme shape differences
    extreme_kurtosis_dims = np.argsort(kurtosis_diff)[-10:].tolist()
    extreme_skewness_dims = np.argsort(skewness_diff)[-10:].tolist()

    return {
        "mean_correlation": mean_correlation,
        "std_correlation": std_correlation,
        "std_ratios": {
            "median": std_ratios.median().item(),
            "mean": std_ratios.mean().item(),
            "min": std_ratios.min().item(),
            "max": std_ratios.max().item(),
        },
        "outlier_dimensions": {
            "high_variance": [
                {"dim": int(d), "ratio": float(std_ratios[d])}
                for d in outlier_high
            ],
            "low_variance": [
                {"dim": int(d), "ratio": float(std_ratios[d])}
                for d in outlier_low
            ],
        },
        "top_different_dimensions": {
            "highest_std_ratio": [
                {
                    "dim": int(d),
                    "ratio": float(std_ratios[d]),
                    "emb_std": float(emb_stds[d]),
                    "qwen3_std": float(qwen3_stds[d]),
                }
                for d in top_high_dims
            ],
            "lowest_std_ratio": [
                {
                    "dim": int(d),
                    "ratio": float(std_ratios[d]),
                    "emb_std": float(emb_stds[d]),
                    "qwen3_std": float(qwen3_stds[d]),
                }
                for d in top_low_dims
            ],
        },
        "shape_differences": {
            "extreme_kurtosis_dims": [
                {
                    "dim": int(d),
                    "emb_kurtosis": float(emb_stats["kurtosis"][d]),
                    "qwen3_kurtosis": float(qwen3_stats["kurtosis"][d]),
                    "diff": float(kurtosis_diff[d]),
                }
                for d in extreme_kurtosis_dims
            ],
            "extreme_skewness_dims": [
                {
                    "dim": int(d),
                    "emb_skewness": float(emb_stats["skewness"][d]),
                    "qwen3_skewness": float(qwen3_stats["skewness"][d]),
                    "diff": float(skewness_diff[d]),
                }
                for d in extreme_skewness_dims
            ],
        },
        "dead_dimensions": {
            "emb_only": list(set(emb_stats["dead_dimensions"]) - set(qwen3_stats["dead_dimensions"])),
            "qwen3_only": list(set(qwen3_stats["dead_dimensions"]) - set(emb_stats["dead_dimensions"])),
            "common": list(set(emb_stats["dead_dimensions"]) & set(qwen3_stats["dead_dimensions"])),
        },
        "hyperactive_dimensions": {
            "emb_only": list(set(emb_stats["hyperactive_dimensions"]) - set(qwen3_stats["hyperactive_dimensions"])),
            "qwen3_only": list(set(qwen3_stats["hyperactive_dimensions"]) - set(emb_stats["hyperactive_dimensions"])),
            "common": list(set(emb_stats["hyperactive_dimensions"]) & set(qwen3_stats["hyperactive_dimensions"])),
        },
    }


def analyze_attention_sensitivity(
    emb_embeddings: torch.Tensor,
    qwen3_embeddings: torch.Tensor,
) -> dict:
    """
    Analyze which dimensions might be most sensitive to attention mechanisms.

    The DiT uses attention over the text embeddings. Dimensions with large
    differences in variance or extreme values could be over-weighted by
    attention, causing artifacts.

    Args:
        emb_embeddings: Qwen3-Embedding-4B embeddings
        qwen3_embeddings: Qwen3-4B embeddings

    Returns:
        Dictionary with attention sensitivity analysis
    """
    # Flatten to (n_tokens, 2560)
    emb_flat = emb_embeddings.view(-1, emb_embeddings.size(-1))
    qwen3_flat = qwen3_embeddings.view(-1, qwen3_embeddings.size(-1))

    # Compute attention weights (simplified: softmax over L2 norm)
    # Real DiT attention is more complex, but this gives intuition
    emb_norms = emb_flat.norm(dim=1, keepdim=True)
    qwen3_norms = qwen3_flat.norm(dim=1, keepdim=True)

    emb_weights = F.softmax(emb_norms.squeeze(), dim=0)
    qwen3_weights = F.softmax(qwen3_norms.squeeze(), dim=0)

    # Weighted means and stds
    emb_weighted_mean = (emb_flat * emb_weights.unsqueeze(1)).sum(dim=0)
    qwen3_weighted_mean = (qwen3_flat * qwen3_weights.unsqueeze(1)).sum(dim=0)

    # Difference in attention-weighted means
    weighted_mean_diff = (emb_weighted_mean - qwen3_weighted_mean).abs()

    # Top dimensions by attention-weighted difference
    top_weighted_diff_dims = torch.argsort(weighted_mean_diff, descending=True)[:20].tolist()

    return {
        "attention_weight_correlation": F.cosine_similarity(
            emb_weights.unsqueeze(0),
            qwen3_weights.unsqueeze(0),
        ).item(),
        "top_weighted_diff_dimensions": [
            {
                "dim": int(d),
                "emb_weighted_mean": float(emb_weighted_mean[d]),
                "qwen3_weighted_mean": float(qwen3_weighted_mean[d]),
                "diff": float(weighted_mean_diff[d]),
            }
            for d in top_weighted_diff_dims
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-dimension differences between Qwen3-Embedding-4B and Qwen3-4B"
    )
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS, help="Prompts to analyze")
    parser.add_argument("--output", type=Path, default=Path("experiments/results/embedding_perdim_analysis.json"))
    parser.add_argument("--quick", action="store_true", help="Quick test with one prompt")
    parser.add_argument("--hidden-layer", type=int, default=-2, help="Hidden layer to extract")
    args = parser.parse_args()

    if args.quick:
        args.prompts = [args.prompts[0]]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading models...")

    from llm_dit.embedding import EmbeddingExtractor
    from llm_dit.backends.transformers import TransformersBackend

    # Results storage
    all_results = {
        "metadata": {
            "hidden_layer": args.hidden_layer,
            "num_prompts": len(args.prompts),
            "prompts": args.prompts,
        },
        "per_prompt_analysis": {},
        "aggregated_analysis": None,
    }

    # -------------------------------------------------------------------------
    # Phase 1: Extract embeddings from both models for all prompts
    # -------------------------------------------------------------------------
    prompt_embeddings = {}  # prompt -> {"qwen3": tensor, "embedding": tensor}

    # Load Qwen3-4B
    logger.info("Loading Qwen3-4B...")
    qwen3_backend = TransformersBackend.from_pretrained(
        QWEN3_4B_PATH,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        model_subfolder="",
        tokenizer_subfolder="",
    )

    logger.info("Encoding prompts with Qwen3-4B...")
    for prompt in args.prompts:
        # Use raw prompt for fair comparison (chat template adds noise)
        output = qwen3_backend.encode([prompt])
        prompt_embeddings[prompt] = {"qwen3": output.embeddings[0].cpu()}
        logger.info(f"  '{prompt}': {len(prompt_embeddings[prompt]['qwen3'])} tokens")

    # Unload Qwen3-4B
    logger.info("Unloading Qwen3-4B...")
    del qwen3_backend
    torch.cuda.empty_cache()

    # Load Qwen3-Embedding-4B
    logger.info("Loading Qwen3-Embedding-4B...")
    embedding_extractor = EmbeddingExtractor.from_pretrained(
        QWEN3_EMBEDDING_PATH,
        device="cuda",
        torch_dtype=torch.bfloat16,
    )

    logger.info("Encoding prompts with Qwen3-Embedding-4B...")
    for prompt in args.prompts:
        emb = embedding_extractor.encode_for_zimage(
            prompt,
            hidden_layer=args.hidden_layer,
            scale_factor=1.0,  # No scaling - analyze raw differences
        )
        prompt_embeddings[prompt]["embedding"] = emb.cpu()
        logger.info(f"  '{prompt}': {len(emb)} tokens")

    # Unload embedding model
    logger.info("Unloading Qwen3-Embedding-4B...")
    embedding_extractor.unload()
    del embedding_extractor
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Phase 2: Per-prompt analysis
    # -------------------------------------------------------------------------
    logger.info("\n" + "="*60)
    logger.info("Per-Prompt Analysis")
    logger.info("="*60)

    for prompt in args.prompts:
        logger.info(f"\nAnalyzing: {prompt}")

        qwen3_emb = prompt_embeddings[prompt]["qwen3"]
        emb_emb = prompt_embeddings[prompt]["embedding"]

        # Truncate to same length
        min_len = min(len(qwen3_emb), len(emb_emb))
        qwen3_emb = qwen3_emb[:min_len]
        emb_emb = emb_emb[:min_len]

        # Compute per-dimension stats
        logger.info("  Computing per-dimension statistics...")
        qwen3_stats = compute_per_dimension_stats(qwen3_emb)
        emb_stats = compute_per_dimension_stats(emb_emb)

        # Compare distributions
        logger.info("  Comparing distributions...")
        comparison = compare_dimension_distributions(emb_stats, qwen3_stats)

        # Attention sensitivity
        logger.info("  Analyzing attention sensitivity...")
        attention = analyze_attention_sensitivity(emb_emb, qwen3_emb)

        # Global cosine similarity (for reference)
        global_cosine = F.cosine_similarity(
            emb_emb.flatten().unsqueeze(0),
            qwen3_emb.flatten().unsqueeze(0),
        ).item()

        all_results["per_prompt_analysis"][prompt] = {
            "global_cosine_similarity": global_cosine,
            "num_tokens": min_len,
            "qwen3_stats": {
                "median_std": qwen3_stats["median_std"],
                "mean_std": qwen3_stats["mean_std"],
                "dead_dimensions": qwen3_stats["dead_dimensions"],
                "hyperactive_dimensions": qwen3_stats["hyperactive_dimensions"],
            },
            "embedding_stats": {
                "median_std": emb_stats["median_std"],
                "mean_std": emb_stats["mean_std"],
                "dead_dimensions": emb_stats["dead_dimensions"],
                "hyperactive_dimensions": emb_stats["hyperactive_dimensions"],
            },
            "comparison": comparison,
            "attention_sensitivity": attention,
        }

        # Log key findings
        logger.info(f"    Global cosine similarity: {global_cosine:.4f}")
        logger.info(f"    Mean correlation: {comparison['mean_correlation']:.4f}")
        logger.info(f"    Std correlation: {comparison['std_correlation']:.4f}")
        logger.info(f"    Std ratio range: {comparison['std_ratios']['min']:.2f} - {comparison['std_ratios']['max']:.2f}")
        logger.info(f"    High variance outliers: {len(comparison['outlier_dimensions']['high_variance'])}")
        logger.info(f"    Low variance outliers: {len(comparison['outlier_dimensions']['low_variance'])}")

        if comparison['outlier_dimensions']['high_variance']:
            logger.info("    Top high-variance outliers:")
            for item in comparison['outlier_dimensions']['high_variance'][:5]:
                logger.info(f"      Dim {item['dim']}: {item['ratio']:.2f}x")

        if comparison['outlier_dimensions']['low_variance']:
            logger.info("    Top low-variance outliers:")
            for item in comparison['outlier_dimensions']['low_variance'][:5]:
                logger.info(f"      Dim {item['dim']}: {item['ratio']:.2f}x")

    # -------------------------------------------------------------------------
    # Phase 3: Aggregated analysis across all prompts
    # -------------------------------------------------------------------------
    logger.info("\n" + "="*60)
    logger.info("Aggregated Analysis (All Prompts)")
    logger.info("="*60)

    # Concatenate all embeddings
    all_qwen3 = torch.cat([prompt_embeddings[p]["qwen3"] for p in args.prompts])
    all_emb = torch.cat([prompt_embeddings[p]["embedding"] for p in args.prompts])

    # Truncate to same length
    min_len = min(len(all_qwen3), len(all_emb))
    all_qwen3 = all_qwen3[:min_len]
    all_emb = all_emb[:min_len]

    logger.info(f"Total tokens analyzed: {min_len}")

    # Compute aggregated stats
    logger.info("Computing aggregated statistics...")
    agg_qwen3_stats = compute_per_dimension_stats(all_qwen3)
    agg_emb_stats = compute_per_dimension_stats(all_emb)

    logger.info("Comparing aggregated distributions...")
    agg_comparison = compare_dimension_distributions(agg_emb_stats, agg_qwen3_stats)

    logger.info("Analyzing aggregated attention sensitivity...")
    agg_attention = analyze_attention_sensitivity(all_emb, all_qwen3)

    all_results["aggregated_analysis"] = {
        "total_tokens": min_len,
        "qwen3_stats": {
            "median_std": agg_qwen3_stats["median_std"],
            "mean_std": agg_qwen3_stats["mean_std"],
            "dead_dimensions": agg_qwen3_stats["dead_dimensions"],
            "hyperactive_dimensions": agg_qwen3_stats["hyperactive_dimensions"],
        },
        "embedding_stats": {
            "median_std": agg_emb_stats["median_std"],
            "mean_std": agg_emb_stats["mean_std"],
            "dead_dimensions": agg_emb_stats["dead_dimensions"],
            "hyperactive_dimensions": agg_emb_stats["hyperactive_dimensions"],
        },
        "comparison": agg_comparison,
        "attention_sensitivity": agg_attention,
    }

    # Log aggregated findings
    logger.info(f"\nAggregated Results:")
    logger.info(f"  Mean correlation: {agg_comparison['mean_correlation']:.4f}")
    logger.info(f"  Std correlation: {agg_comparison['std_correlation']:.4f}")
    logger.info(f"  Std ratio range: {agg_comparison['std_ratios']['min']:.2f} - {agg_comparison['std_ratios']['max']:.2f}")
    logger.info(f"  High variance outliers: {len(agg_comparison['outlier_dimensions']['high_variance'])}")
    logger.info(f"  Low variance outliers: {len(agg_comparison['outlier_dimensions']['low_variance'])}")

    logger.info("\n  Top 10 dimensions by std ratio (high):")
    for item in agg_comparison['top_different_dimensions']['highest_std_ratio']:
        logger.info(f"    Dim {item['dim']}: {item['ratio']:.3f}x (emb={item['emb_std']:.2f}, qwen3={item['qwen3_std']:.2f})")

    logger.info("\n  Top 10 dimensions by std ratio (low):")
    for item in agg_comparison['top_different_dimensions']['lowest_std_ratio']:
        logger.info(f"    Dim {item['dim']}: {item['ratio']:.3f}x (emb={item['emb_std']:.2f}, qwen3={item['qwen3_std']:.2f})")

    logger.info("\n  Dead dimensions:")
    logger.info(f"    Embedding-only: {len(agg_comparison['dead_dimensions']['emb_only'])}")
    logger.info(f"    Qwen3-only: {len(agg_comparison['dead_dimensions']['qwen3_only'])}")
    logger.info(f"    Common: {len(agg_comparison['dead_dimensions']['common'])}")

    logger.info("\n  Hyperactive dimensions:")
    logger.info(f"    Embedding-only: {len(agg_comparison['hyperactive_dimensions']['emb_only'])}")
    logger.info(f"    Qwen3-only: {len(agg_comparison['hyperactive_dimensions']['qwen3_only'])}")
    logger.info(f"    Common: {len(agg_comparison['hyperactive_dimensions']['common'])}")

    logger.info("\n  Top 10 attention-weighted differences:")
    for item in agg_attention['top_weighted_diff_dimensions'][:10]:
        logger.info(f"    Dim {item['dim']}: diff={item['diff']:.3f}")

    # -------------------------------------------------------------------------
    # Phase 4: Save results
    # -------------------------------------------------------------------------
    logger.info(f"\nSaving results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info("Done!")
    logger.info(f"\nKey findings saved to {args.output}")
    logger.info("Use this data to identify problematic dimensions for masking/correction.")


if __name__ == "__main__":
    main()
