#!/usr/bin/env python3
"""
LTX-2 Gemma Dimension Analysis

Last Updated: 2026-01-16

Analyze per-dimension statistics across ALL 49 layers of Gemma-3 used in LTX-2.
Adapted from experiments/analyze_layer_dimensions.py (Qwen3-4B version).

Discovery Question: "Which dimensions in 3840-dim embeddings cause artifacts?"

Key differences from Qwen3 analysis:
- Gemma-3 has 49 layers (not 36)
- Hidden dimension is 3840 (not 2560)
- Accessed through LTX-2's text_encoder

Method:
1. Extract hidden states from all 49 layers across diverse prompts
2. Compute per-dimension: mean, std, kurtosis, skewness
3. Identify outliers at 10x, 50x, 100x median std
4. Generate heatmaps and analysis visualizations

Output Structure:
    experiments/results/ltx2_dimension_analysis_{timestamp}/
    ├── layer_stats/          # NPZ per layer
    │   └── layer_00_stats.npz
    ├── visualizations/
    │   ├── heatmap_per_dim_std.png     # 49x3840
    │   ├── outlier_count_vs_layer.png
    │   └── tracked_dim_*.png
    ├── summary.json
    └── analysis_report.md

Usage:
    # Full analysis with default prompts
    uv run python experiments/ltx2/dimension_analysis.py

    # Quick test
    uv run python experiments/ltx2/dimension_analysis.py --quick

    # Custom tracked dimensions
    uv run python experiments/ltx2/dimension_analysis.py --track-dims 0 4 396 1000 1920 3839
"""

import argparse
import gc
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.ltx2.prompts import CATEGORY_PROMPTS, LEGACY_SHORT_PROMPTS, QUICK_CATEGORY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# LTX-2 Gemma configuration
NUM_GEMMA_LAYERS = 49
GEMMA_HIDDEN_DIM = 3840

# Tracked dimensions (candidates for causing artifacts)
# Based on prior analysis and hypothesis
TRACKED_DIMENSIONS = [0, 4, 396, 1000, 1920, 3839]

# Outlier thresholds (multiples of median std)
OUTLIER_THRESHOLDS = [10, 50, 100]

# Default diverse prompts for statistical stability
# Using standardized LTX-2 format prompts (100-300 words each)
DEFAULT_PROMPTS = list(CATEGORY_PROMPTS.values())

# Quick mode uses fewer prompts
QUICK_PROMPTS = [CATEGORY_PROMPTS[k] for k in QUICK_CATEGORY]


@dataclass
class LayerDimensionStats:
    """Statistics for a single layer's embeddings."""

    layer_index: int  # 0-48

    # Per-dimension arrays (shape: 3840,)
    per_dim_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    per_dim_std: np.ndarray = field(default_factory=lambda: np.array([]))
    per_dim_min: np.ndarray = field(default_factory=lambda: np.array([]))
    per_dim_max: np.ndarray = field(default_factory=lambda: np.array([]))
    per_dim_kurtosis: np.ndarray = field(default_factory=lambda: np.array([]))
    per_dim_skewness: np.ndarray = field(default_factory=lambda: np.array([]))

    # Outlier lists (dimension indices)
    outliers_10x: list[int] = field(default_factory=list)
    outliers_50x: list[int] = field(default_factory=list)
    outliers_100x: list[int] = field(default_factory=list)

    # Global statistics
    global_mean: float = 0.0
    global_std: float = 0.0
    median_dim_std: float = 0.0
    max_dim_std: float = 0.0
    max_kurtosis: float = 0.0
    max_kurtosis_dim: int = 0

    # Tracked dimensions
    tracked_dim_stats: dict = field(default_factory=dict)

    # Token counts
    num_tokens_analyzed: int = 0
    num_prompts: int = 0


class DimensionAccumulator:
    """Accumulate statistics across multiple batches using Welford's algorithm."""

    def __init__(self, hidden_dim: int = GEMMA_HIDDEN_DIM):
        self.hidden_dim = hidden_dim
        self.n = 0
        self.mean = np.zeros(hidden_dim)
        self.M2 = np.zeros(hidden_dim)  # Sum of squared differences from mean
        self.M3 = np.zeros(hidden_dim)  # For skewness
        self.M4 = np.zeros(hidden_dim)  # For kurtosis
        self.min_vals = np.full(hidden_dim, np.inf)
        self.max_vals = np.full(hidden_dim, -np.inf)

    def update_batch(self, values: np.ndarray):
        """
        Update statistics with a batch of values.

        Args:
            values: Array of shape (n_tokens, hidden_dim)
        """
        if len(values) == 0:
            return

        # Compute batch stats and combine
        batch_n = len(values)
        batch_mean = values.mean(axis=0)
        batch_var = values.var(axis=0)

        # Combine with running statistics (parallel algorithm)
        if self.n == 0:
            self.n = batch_n
            self.mean = batch_mean
            self.M2 = batch_var * batch_n
        else:
            n_total = self.n + batch_n
            delta = batch_mean - self.mean
            self.mean = (self.n * self.mean + batch_n * batch_mean) / n_total
            self.M2 = self.M2 + batch_var * batch_n + delta**2 * self.n * batch_n / n_total
            self.n = n_total

        # Update min/max
        self.min_vals = np.minimum(self.min_vals, values.min(axis=0))
        self.max_vals = np.maximum(self.max_vals, values.max(axis=0))

    def finalize(
        self,
        layer_index: int,
        num_prompts: int,
        tracked_dims: list[int],
    ) -> LayerDimensionStats:
        """Compute final statistics and return LayerDimensionStats."""
        if self.n == 0:
            raise ValueError("No data accumulated")

        # Variance and std
        variance = self.M2 / self.n
        per_dim_std = np.sqrt(np.maximum(variance, 0))

        # Kurtosis and skewness (computed from variance)
        # For simplicity, using excess kurtosis approximation
        var_safe = np.where(variance > 1e-10, variance, 1e-10)
        kurtosis = (
            (self.M4 / self.n) / (var_safe**2) - 3
            if hasattr(self, "M4")
            else np.zeros(self.hidden_dim)
        )
        skewness = (
            (self.M3 / self.n) / (var_safe**1.5)
            if hasattr(self, "M3")
            else np.zeros(self.hidden_dim)
        )

        # Handle edge cases
        kurtosis = np.where(variance > 1e-10, kurtosis, 0)
        skewness = np.where(variance > 1e-10, skewness, 0)

        # Compute outliers
        median_std = np.median(per_dim_std)
        outliers_10x = np.where(per_dim_std > 10 * median_std)[0].tolist()
        outliers_50x = np.where(per_dim_std > 50 * median_std)[0].tolist()
        outliers_100x = np.where(per_dim_std > 100 * median_std)[0].tolist()

        # Tracked dimension stats
        tracked_stats = {}
        for dim in tracked_dims:
            if dim < self.hidden_dim:
                tracked_stats[dim] = {
                    "mean": float(self.mean[dim]),
                    "std": float(per_dim_std[dim]),
                    "std_ratio": float(per_dim_std[dim] / median_std) if median_std > 0 else 0,
                    "min": float(self.min_vals[dim]),
                    "max": float(self.max_vals[dim]),
                }

        # Max kurtosis info
        max_kurtosis_dim = int(np.argmax(np.abs(kurtosis)))
        max_kurtosis = float(kurtosis[max_kurtosis_dim])

        return LayerDimensionStats(
            layer_index=layer_index,
            per_dim_mean=self.mean,
            per_dim_std=per_dim_std,
            per_dim_min=self.min_vals,
            per_dim_max=self.max_vals,
            per_dim_kurtosis=kurtosis,
            per_dim_skewness=skewness,
            outliers_10x=outliers_10x,
            outliers_50x=outliers_50x,
            outliers_100x=outliers_100x,
            global_mean=float(self.mean.mean()),
            global_std=float(per_dim_std.mean()),
            median_dim_std=float(median_std),
            max_dim_std=float(per_dim_std.max()),
            max_kurtosis=max_kurtosis,
            max_kurtosis_dim=max_kurtosis_dim,
            tracked_dim_stats=tracked_stats,
            num_tokens_analyzed=self.n,
            num_prompts=num_prompts,
        )


def analyze_gemma_layers(
    model_path: str,
    prompts: list[str],
    device: str = "cuda",
    tracked_dimensions: list[int] = TRACKED_DIMENSIONS,
    max_sequence_length: int = 1024,
) -> dict[int, LayerDimensionStats]:
    """
    Analyze all 49 Gemma layers for per-dimension statistics.

    Uses the LTX-2 text_encoder directly to access Gemma hidden states.

    Args:
        model_path: Path to LTX-2 model
        prompts: List of prompts to analyze
        device: Device to use
        tracked_dimensions: Dimensions to track specifically
        max_sequence_length: Maximum sequence length for tokenization

    Returns:
        Dictionary mapping layer_index (0-48) to LayerDimensionStats
    """
    from diffusers import LTX2Pipeline

    logger.info(f"Loading LTX-2 pipeline from {model_path}...")
    pipe = LTX2Pipeline.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
    )

    # Move only text encoder to device (don't need full pipeline)
    text_encoder = pipe.text_encoder.to(device)
    tokenizer = pipe.tokenizer
    text_encoder.set_output_embeddings(None)  # Disable any output layer

    logger.info(
        f"Gemma config: hidden_size={text_encoder.config.hidden_size}, "
        f"num_hidden_layers={text_encoder.config.num_hidden_layers}"
    )

    # Verify dimensions match expectations
    actual_hidden = text_encoder.config.hidden_size
    actual_layers = text_encoder.config.num_hidden_layers + 1  # +1 for embedding

    if actual_hidden != GEMMA_HIDDEN_DIM:
        logger.warning(f"Hidden dim mismatch: expected {GEMMA_HIDDEN_DIM}, got {actual_hidden}")
    if actual_layers != NUM_GEMMA_LAYERS:
        logger.warning(f"Layer count mismatch: expected {NUM_GEMMA_LAYERS}, got {actual_layers}")

    # Initialize accumulators for all layers
    accumulators = {i: DimensionAccumulator(actual_hidden) for i in range(actual_layers)}

    logger.info(f"Analyzing {len(prompts)} prompts across {actual_layers} layers...")

    # Process each prompt
    for prompt_idx, prompt in enumerate(prompts):
        logger.info(f"Processing prompt {prompt_idx + 1}/{len(prompts)}: {prompt[:50]}...")

        # Tokenize
        inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        attention_mask = inputs.attention_mask.bool()

        # Forward pass with all hidden states
        with torch.no_grad():
            outputs = text_encoder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True,
            )

        # Extract hidden states for each layer
        # hidden_states is tuple of (batch, seq, hidden_dim)
        # hidden_states[0] = embeddings, hidden_states[1] = after layer 1, etc.
        for layer_idx, hidden in enumerate(outputs.hidden_states):
            # Extract valid tokens (filter by attention mask)
            mask = attention_mask[0]
            # Convert to float32 first (numpy doesn't support bfloat16)
            valid_tokens = hidden[0][mask].float().cpu().numpy().astype(np.float64)
            accumulators[layer_idx].update_batch(valid_tokens)

        # Memory cleanup
        del outputs
        torch.cuda.empty_cache()

    # Finalize statistics
    logger.info("Computing final statistics...")
    results = {}
    for layer_idx, acc in accumulators.items():
        stats = acc.finalize(
            layer_index=layer_idx,
            num_prompts=len(prompts),
            tracked_dims=tracked_dimensions,
        )
        results[layer_idx] = stats

        # Log summary
        logger.info(
            f"Layer {layer_idx:2d}: "
            f"median_std={stats.median_dim_std:.2f}, "
            f"max_std={stats.max_dim_std:.2f}, "
            f"outliers(10x)={len(stats.outliers_10x)}"
        )

    # Cleanup
    del text_encoder
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return results


def save_layer_stats(results: dict[int, LayerDimensionStats], output_dir: Path):
    """Save per-layer statistics to NPZ files."""
    stats_dir = output_dir / "layer_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx, stats in results.items():
        filename = f"layer_{layer_idx:02d}_stats.npz"
        filepath = stats_dir / filename

        np.savez(
            filepath,
            layer_index=stats.layer_index,
            per_dim_mean=stats.per_dim_mean,
            per_dim_std=stats.per_dim_std,
            per_dim_min=stats.per_dim_min,
            per_dim_max=stats.per_dim_max,
            per_dim_kurtosis=stats.per_dim_kurtosis,
            per_dim_skewness=stats.per_dim_skewness,
            global_mean=stats.global_mean,
            global_std=stats.global_std,
            median_dim_std=stats.median_dim_std,
            max_dim_std=stats.max_dim_std,
            num_tokens_analyzed=stats.num_tokens_analyzed,
            num_prompts=stats.num_prompts,
        )

    logger.info(f"Saved statistics for {len(results)} layers to {stats_dir}")


def save_summary_json(
    results: dict[int, LayerDimensionStats],
    output_dir: Path,
    prompts: list[str],
    tracked_dimensions: list[int],
):
    """Save summary JSON with aggregate statistics."""
    summary = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_prompts": len(prompts),
            "num_layers": len(results),
            "tracked_dimensions": tracked_dimensions,
            "hidden_dim": GEMMA_HIDDEN_DIM,
            "prompts": prompts,
        },
        "layer_summary": {},
        "outlier_analysis": {
            "threshold_10x": {},
            "threshold_50x": {},
            "threshold_100x": {},
        },
        "tracked_dimension_analysis": {str(dim): {} for dim in tracked_dimensions},
    }

    for layer_idx, stats in sorted(results.items()):
        layer_key = f"layer_{layer_idx:02d}"

        # Basic summary
        summary["layer_summary"][layer_key] = {
            "layer_index": stats.layer_index,
            "global_mean": stats.global_mean,
            "global_std": stats.global_std,
            "median_dim_std": stats.median_dim_std,
            "max_dim_std": stats.max_dim_std,
            "max_kurtosis": stats.max_kurtosis,
            "max_kurtosis_dim": stats.max_kurtosis_dim,
            "num_tokens_analyzed": stats.num_tokens_analyzed,
        }

        # Outlier counts
        summary["outlier_analysis"]["threshold_10x"][layer_key] = {
            "count": len(stats.outliers_10x),
            "dimensions": stats.outliers_10x[:20],  # Limit to first 20
        }
        summary["outlier_analysis"]["threshold_50x"][layer_key] = {
            "count": len(stats.outliers_50x),
            "dimensions": stats.outliers_50x,
        }
        summary["outlier_analysis"]["threshold_100x"][layer_key] = {
            "count": len(stats.outliers_100x),
            "dimensions": stats.outliers_100x,
        }

        # Tracked dimensions
        for dim in tracked_dimensions:
            summary["tracked_dimension_analysis"][str(dim)][layer_key] = (
                stats.tracked_dim_stats.get(dim, {})
            )

    # Save
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved summary to {summary_path}")


def plot_visualizations(
    results: dict[int, LayerDimensionStats],
    output_dir: Path,
    tracked_dimensions: list[int] = TRACKED_DIMENSIONS,
):
    """Generate visualization plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping visualizations")
        return

    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Sort results by layer index
    sorted_results = sorted(results.items(), key=lambda x: x[0])

    # 1. Per-dimension std heatmap
    logger.info("Generating per-dimension std heatmap...")
    hidden_dim = len(sorted_results[0][1].per_dim_std)
    num_layers = len(sorted_results)

    data = np.zeros((num_layers, hidden_dim))
    for i, (_, stats) in enumerate(sorted_results):
        data[i] = stats.per_dim_std

    fig, ax = plt.subplots(figsize=(20, 12))
    # Use log scale for better visibility
    im = ax.imshow(
        np.log10(data + 1),
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="log10(std + 1)")

    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Layer (0=embedding, 48=last)")
    ax.set_yticks(range(0, num_layers, 4))
    ax.set_yticklabels([sorted_results[i][0] for i in range(0, num_layers, 4)])
    ax.set_title(f"Per-Dimension Standard Deviation Across {num_layers} Gemma Layers")

    # Mark tracked dimensions
    for dim in tracked_dimensions:
        if dim < hidden_dim:
            ax.axvline(x=dim, color="red", linestyle="--", alpha=0.5, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(viz_dir / "heatmap_per_dim_std.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Outlier count vs layer depth
    logger.info("Generating outlier count plot...")
    fig, ax = plt.subplots(figsize=(14, 6))

    layers = [layer_idx for layer_idx, _ in sorted_results]

    for threshold, color, label in [
        (10, "blue", "10x"),
        (50, "orange", "50x"),
        (100, "red", "100x"),
    ]:
        if threshold == 10:
            counts = [len(stats.outliers_10x) for _, stats in sorted_results]
        elif threshold == 50:
            counts = [len(stats.outliers_50x) for _, stats in sorted_results]
        else:
            counts = [len(stats.outliers_100x) for _, stats in sorted_results]

        ax.plot(layers, counts, color=color, marker="o", label=f"{label} threshold", linewidth=2)

    ax.set_xlabel("Layer Index (0=embedding, 48=last transformer)")
    ax.set_ylabel("Number of Outlier Dimensions")
    ax.set_title("Outlier Dimension Count by Layer Depth (Gemma in LTX-2)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(viz_dir / "outlier_count_vs_layer.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Median std vs layer depth
    logger.info("Generating median std plot...")
    fig, ax = plt.subplots(figsize=(14, 6))

    median_stds = [stats.median_dim_std for _, stats in sorted_results]
    max_stds = [stats.max_dim_std for _, stats in sorted_results]

    ax.plot(layers, median_stds, color="blue", marker="o", linewidth=2, label="Median dim std")
    ax.plot(layers, max_stds, color="red", marker="s", linewidth=2, alpha=0.7, label="Max dim std")

    ax.set_xlabel("Layer Index (0=embedding, 48=last transformer)")
    ax.set_ylabel("Standard Deviation")
    ax.set_title("Embedding Statistics by Layer Depth (Gemma in LTX-2)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(viz_dir / "std_vs_layer.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Tracked dimension plots
    hidden_dim = len(sorted_results[0][1].per_dim_std)
    for dim in tracked_dimensions:
        if dim >= hidden_dim:
            continue

        logger.info(f"Generating dimension {dim} tracking plot...")
        fig, ax = plt.subplots(figsize=(14, 6))

        stds = [stats.per_dim_std[dim] for _, stats in sorted_results]

        ax.plot(layers, stds, color="blue", marker="o", linewidth=2)
        median_stds = [stats.median_dim_std for _, stats in sorted_results]
        ax.plot(
            layers,
            [m * 10 for m in median_stds],
            color="orange",
            linestyle="--",
            label="10x median (per layer)",
            alpha=0.7,
        )
        ax.plot(
            layers,
            [m * 100 for m in median_stds],
            color="red",
            linestyle="--",
            label="100x median (per layer)",
            alpha=0.7,
        )

        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Standard Deviation")
        ax.set_title(f"Dimension {dim} - Standard Deviation Across Layers")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(viz_dir / f"tracked_dim_{dim}_across_layers.png", dpi=150, bbox_inches="tight")
        plt.close()

    logger.info(f"Saved visualizations to {viz_dir}")


def generate_analysis_report(
    results: dict[int, LayerDimensionStats],
    output_dir: Path,
    tracked_dimensions: list[int] = TRACKED_DIMENSIONS,
):
    """Generate human-readable analysis report."""
    report_lines = [
        "# LTX-2 Gemma Dimension Analysis Report",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Executive Summary",
        "",
        f"- **Model:** Gemma-3 (via LTX-2 text_encoder)",
        f"- **Hidden dimension:** {GEMMA_HIDDEN_DIM}",
        f"- **Layers analyzed:** {len(results)}",
        "",
    ]

    # Sort by layer index
    sorted_results = sorted(results.items(), key=lambda x: x[0])

    # Find layers with fewest/most outliers at 10x threshold
    outlier_counts = [(layer_idx, len(stats.outliers_10x)) for layer_idx, stats in sorted_results]
    min_outliers = min(outlier_counts, key=lambda x: x[1])
    max_outliers = max(outlier_counts, key=lambda x: x[1])

    report_lines.extend(
        [
            f"- **Cleanest layer (10x threshold):** Layer {min_outliers[0]} ({min_outliers[1]} outliers)",
            f"- **Most outliers (10x threshold):** Layer {max_outliers[0]} ({max_outliers[1]} outliers)",
            "",
        ]
    )

    # Tracked dimensions summary
    report_lines.extend(
        [
            "## Tracked Dimensions",
            "",
        ]
    )

    for dim in tracked_dimensions:
        report_lines.append(f"### Dimension {dim}")
        report_lines.append("")
        report_lines.append("| Layer | Std | Std Ratio | Min | Max |")
        report_lines.append("|-------|-----|-----------|-----|-----|")

        for layer_idx, stats in sorted_results:
            dim_stats = stats.tracked_dim_stats.get(dim, {})
            std = dim_stats.get("std", 0)
            ratio = dim_stats.get("std_ratio", 0)
            min_val = dim_stats.get("min", 0)
            max_val = dim_stats.get("max", 0)
            report_lines.append(
                f"| {layer_idx} | {std:.2f} | {ratio:.1f}x | {min_val:.2f} | {max_val:.2f} |"
            )

        report_lines.append("")

    # Layer-by-layer summary table
    report_lines.extend(
        [
            "## Layer-by-Layer Summary",
            "",
            "| Layer | Median Std | Max Std | Outliers (10x) | Outliers (50x) | Outliers (100x) |",
            "|-------|-----------|---------|----------------|----------------|-----------------|",
        ]
    )

    for layer_idx, stats in sorted_results:
        report_lines.append(
            f"| {layer_idx} | {stats.median_dim_std:.2f} | {stats.max_dim_std:.2f} | "
            f"{len(stats.outliers_10x)} | {len(stats.outliers_50x)} | {len(stats.outliers_100x)} |"
        )

    report_lines.extend(
        [
            "",
            "## Recommendations",
            "",
            "Based on the outlier analysis:",
            "",
        ]
    )

    # Find clean layers (0 outliers at 10x)
    clean_layers = [
        layer_idx for layer_idx, stats in sorted_results if len(stats.outliers_10x) == 0
    ]
    if clean_layers:
        report_lines.append(f"- **Clean layers (no 10x outliers):** {clean_layers}")
    else:
        # Find layers with minimum outliers
        min_count = min(len(stats.outliers_10x) for _, stats in sorted_results)
        min_layers = [
            layer_idx for layer_idx, stats in sorted_results if len(stats.outliers_10x) == min_count
        ]
        report_lines.append(f"- **Cleanest layers ({min_count} outliers):** {min_layers}")

    # Find problematic layers (>5 outliers at 10x)
    problematic_layers = [
        layer_idx for layer_idx, stats in sorted_results if len(stats.outliers_10x) > 5
    ]
    if problematic_layers:
        report_lines.append(f"- **Problematic layers (>5 outliers):** {problematic_layers}")

    # Most common outlier dimensions across layers
    all_outlier_dims = {}
    for _, stats in sorted_results:
        for dim in stats.outliers_10x:
            all_outlier_dims[dim] = all_outlier_dims.get(dim, 0) + 1

    if all_outlier_dims:
        sorted_outliers = sorted(all_outlier_dims.items(), key=lambda x: x[1], reverse=True)[:10]
        report_lines.append("")
        report_lines.append("- **Most common outlier dimensions (across all layers):**")
        for dim, count in sorted_outliers:
            report_lines.append(f"  - Dim {dim}: outlier in {count}/{len(results)} layers")

    report_lines.extend(
        [
            "",
            "## Files Generated",
            "",
            f"- `layer_stats/layer_XX_stats.npz` - Per-layer statistics ({len(results)} files)",
            "- `summary.json` - Aggregate statistics in JSON format",
            "- `visualizations/` - Analysis plots",
            "  - `heatmap_per_dim_std.png` - 49x3840 heatmap of std",
            "  - `outlier_count_vs_layer.png` - Outliers by depth",
            "  - `std_vs_layer.png` - Std statistics by depth",
            "  - `tracked_dim_*_across_layers.png` - Tracked dimension plots",
            "",
        ]
    )

    # Write report
    report_path = output_dir / "analysis_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Generated report: {report_path}")


def run_dimension_analysis(
    output_base: str = "experiments/results",
    model_path: str = "models/LTX-2",
    prompts: list[str] | None = None,
    tracked_dimensions: list[int] = TRACKED_DIMENSIONS,
    skip_viz: bool = False,
    device: str = "cuda",
):
    """
    Run the full dimension analysis experiment.

    Args:
        output_base: Base directory for results
        model_path: Path to LTX-2 model
        prompts: List of prompts to analyze. None = use DEFAULT_PROMPTS
        tracked_dimensions: Dimensions to track specifically
        skip_viz: Skip visualization generation
        device: Device to use

    Returns:
        Path to results directory
    """
    if prompts is None:
        prompts = DEFAULT_PROMPTS

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base) / f"ltx2_dimension_analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("LTX-2 Gemma Dimension Analysis")
    logger.info("=" * 60)
    logger.info(f"Output: {output_dir}")
    logger.info(f"Prompts: {len(prompts)}")
    logger.info(f"Tracked dimensions: {tracked_dimensions}")

    # Run analysis
    results = analyze_gemma_layers(
        model_path=model_path,
        prompts=prompts,
        device=device,
        tracked_dimensions=tracked_dimensions,
    )

    # Save results
    save_layer_stats(results, output_dir)
    save_summary_json(results, output_dir, prompts, tracked_dimensions)

    if not skip_viz:
        plot_visualizations(results, output_dir, tracked_dimensions)

    generate_analysis_report(results, output_dir, tracked_dimensions)

    logger.info(f"\n{'=' * 60}")
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="LTX-2 Gemma Dimension Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output-base",
        default="experiments/results",
        help="Base directory for results",
    )
    parser.add_argument(
        "--model-path",
        default="models/LTX-2",
        help="Path to LTX-2 model",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        help="Custom prompts to analyze (default: built-in diverse set)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with fewer prompts",
    )
    parser.add_argument(
        "--track-dims",
        nargs="+",
        type=int,
        default=TRACKED_DIMENSIONS,
        help=f"Dimensions to track specifically (default: {TRACKED_DIMENSIONS})",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip visualization generation",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (default: cuda)",
    )

    args = parser.parse_args()

    # Determine prompts
    if args.prompts:
        prompts = args.prompts
    elif args.quick:
        prompts = QUICK_PROMPTS
    else:
        prompts = DEFAULT_PROMPTS

    run_dimension_analysis(
        output_base=args.output_base,
        model_path=args.model_path,
        prompts=prompts,
        tracked_dimensions=args.track_dims,
        skip_viz=args.skip_viz,
        device=args.device,
    )


if __name__ == "__main__":
    main()
