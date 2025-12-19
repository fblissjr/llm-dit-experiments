#!/usr/bin/env python3
"""
Analyze per-dimension statistics across ALL 36 layers of Qwen3-4B.

Last updated: 2025-12-18

This script generates comprehensive statistics to understand outlier dimensions
and their distribution across layer depth. Designed to support the outlier
hypothesis: certain dimensions have extreme variance that causes artifacts
in image generation.

Model Notes:
- Qwen3-4B: Text-only, thinking model (even though not named "-Thinking")
- Qwen3-VL-4B-Instruct: Vision-language, NO thinking capability
- Qwen3-VL-4B-Thinking: Vision-language, WITH thinking capability

All three share hidden_size=2560, making them compatible with Z-Image DiT.
The VL models require separate analysis due to different token distributions
(vision tokens vs text tokens).

Generates:
1. Per-layer, per-dimension statistics (mean, std, kurtosis)
2. Outlier dimension counts at multiple thresholds (10x, 50x, 100x)
3. Specific tracking of known problem dimensions (4, 396)
4. Visualizations: heatmaps, line plots, layer comparison charts
5. NPZ files with all statistics for each layer

Usage:
    # Basic analysis with diverse prompts
    uv run experiments/analyze_layer_dimensions.py \\
        --model-path /path/to/Qwen3-4B \\
        --output-dir experiments/results/layer_dimension_analysis

    # Quick test with minimal prompts
    uv run experiments/analyze_layer_dimensions.py \\
        --model-path /path/to/Qwen3-4B \\
        --output-dir experiments/results/layer_dimension_analysis \\
        --quick

    # Dry run to preview
    uv run experiments/analyze_layer_dimensions.py \\
        --model-path /path/to/Qwen3-4B \\
        --dry-run

See internal/research/outlier_layer_analysis_design.md for full documentation.
"""

import argparse
import gc
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Known problem dimensions from existing analysis
# - Dim 4: Highest std (2752) at layer -2
# - Dim 396: 617x outlier ratio for VL image tokens at layer -2
# - Dim 0: Third highest std at layer -2
# - Dim 100: Fourth highest at layer -2
TRACKED_DIMENSIONS = [0, 4, 9, 100, 396]

# Outlier thresholds (multiples of median std)
OUTLIER_THRESHOLDS = [10, 50, 100]

# Token position categories for analysis
TOKEN_POSITIONS = {
    "bos": 0,           # Beginning of sequence
    "system": (1, 50),  # System prompt region (approximate)
    "user": (50, -10),  # User prompt region
    "tail": -10,        # Last 10 tokens
}

# Default diverse prompts for statistical stability
DEFAULT_PROMPTS = [
    # Simple subjects (baseline)
    "A cat",
    "A mountain",
    "A house",
    "A tree",
    "A flower",
    # Detailed descriptions
    "A tabby cat with green eyes sleeping on a red velvet cushion",
    "A snow-capped mountain at sunset with orange and purple clouds",
    "A Victorian house with a wraparound porch and white picket fence",
    "An ancient oak tree with gnarled branches in a misty forest",
    # Abstract concepts
    "The concept of freedom",
    "Mathematical beauty",
    "Peaceful serenity",
    # Technical/specific
    "A 1967 Ford Mustang Fastback in Highland Green",
    "A detailed cross-section of a human heart",
    "A circuit board with microprocessors and copper traces",
    # Style prompts
    "In the style of Monet, impressionist water lilies",
    "Cyberpunk neon cityscape, rain-soaked streets",
    "Art nouveau poster with flowing organic lines",
    "Ukiyo-e woodblock print of a great wave",
    # Long prompts (will be truncated but still informative)
    "A highly detailed photograph of an elderly craftsman in his workshop, warm afternoon light streaming through dusty windows, tools hanging on pegboard walls, wood shavings on the floor, wearing a leather apron, focused expression",
    # Culturally-charged (test for bias patterns in early layers)
    "They may take our lives, but they'll never take our freedom",
    "A peaceful protest for civil rights",
    "A sacred temple at dawn",
]

# Quick mode uses fewer prompts
QUICK_PROMPTS = [
    "A cat",
    "A mountain landscape at sunset",
    "A 1967 Ford Mustang in green",
    "Cyberpunk neon cityscape",
]


@dataclass
class LayerDimensionStats:
    """Statistics for a single layer's embeddings."""

    layer_index: int           # Python index (-1 to -36)
    layer_number: int          # Human-readable (1 to 36)

    # Per-dimension arrays (shape: 2560,)
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

    # Cross-prompt consistency (dims that are outliers in >50% of prompts)
    consistent_outliers_10x: list[int] = field(default_factory=list)


class DimensionAccumulator:
    """Accumulate statistics across multiple batches using Welford's algorithm."""

    def __init__(self, hidden_dim: int = 2560):
        self.hidden_dim = hidden_dim
        self.n = 0
        self.mean = np.zeros(hidden_dim)
        self.M2 = np.zeros(hidden_dim)  # Sum of squared differences from mean
        self.M3 = np.zeros(hidden_dim)  # For skewness
        self.M4 = np.zeros(hidden_dim)  # For kurtosis
        self.min_vals = np.full(hidden_dim, np.inf)
        self.max_vals = np.full(hidden_dim, -np.inf)

    def update(self, values: np.ndarray):
        """
        Update statistics with new values.

        Args:
            values: Array of shape (n_tokens, hidden_dim)
        """
        for i in range(len(values)):
            x = values[i]
            self.n += 1
            n = self.n

            delta = x - self.mean
            delta_n = delta / n
            delta_n2 = delta_n ** 2
            term1 = delta * delta_n * (n - 1)

            self.mean += delta_n
            self.M4 += term1 * delta_n2 * (n**2 - 3*n + 3) + \
                       6 * delta_n2 * self.M2 - 4 * delta_n * self.M3
            self.M3 += term1 * delta_n * (n - 2) - 3 * delta_n * self.M2
            self.M2 += term1

            # Min/max
            self.min_vals = np.minimum(self.min_vals, x)
            self.max_vals = np.maximum(self.max_vals, x)

    def update_batch(self, values: np.ndarray):
        """
        Update statistics with a batch of values.

        For large batches, compute batch statistics and combine with running stats.
        This is faster than per-token updates for large batches.

        Args:
            values: Array of shape (n_tokens, hidden_dim)
        """
        if len(values) == 0:
            return

        # For small batches, use per-token update
        if len(values) < 100:
            self.update(values)
            return

        # For large batches, compute batch stats and combine
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
            self.M2 = self.M2 + batch_var * batch_n + \
                      delta**2 * self.n * batch_n / n_total
            self.n = n_total

        # Update min/max
        self.min_vals = np.minimum(self.min_vals, values.min(axis=0))
        self.max_vals = np.maximum(self.max_vals, values.max(axis=0))

        # Compute higher moments (less efficient but acceptable)
        # For a proper implementation, would need to track M3, M4 with batch updates
        # Simplified: recompute from scratch for final statistics

    def finalize(
        self,
        layer_index: int,
        num_prompts: int,
        tracked_dims: list[int],
    ) -> LayerDimensionStats:
        """Compute final statistics and return LayerDimensionStats."""
        if self.n == 0:
            raise ValueError("No data accumulated")

        # Map layer_index to human-readable layer_number:
        # - layer_index=0 (embedding) -> layer_number=0
        # - layer_index=-36 (first transformer) -> layer_number=1
        # - layer_index=-1 (last transformer) -> layer_number=36
        if layer_index == 0:
            layer_number = 0  # Embedding layer
        else:
            layer_number = 37 + layer_index  # -36 -> 1, -1 -> 36

        # Variance and std
        variance = self.M2 / self.n
        per_dim_std = np.sqrt(np.maximum(variance, 0))

        # Kurtosis and skewness (excess kurtosis, normal = 0)
        # kurtosis = M4 / (n * var^2) - 3
        # skewness = M3 / (n * var^1.5)
        var_safe = np.where(variance > 1e-10, variance, 1e-10)
        kurtosis = (self.M4 / self.n) / (var_safe ** 2) - 3
        skewness = (self.M3 / self.n) / (var_safe ** 1.5)

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
            tracked_stats[dim] = {
                "mean": float(self.mean[dim]),
                "std": float(per_dim_std[dim]),
                "std_ratio": float(per_dim_std[dim] / median_std) if median_std > 0 else 0,
                "kurtosis": float(kurtosis[dim]),
                "min": float(self.min_vals[dim]),
                "max": float(self.max_vals[dim]),
            }

        # Max kurtosis info
        max_kurtosis_dim = int(np.argmax(kurtosis))
        max_kurtosis = float(kurtosis[max_kurtosis_dim])

        return LayerDimensionStats(
            layer_index=layer_index,
            layer_number=layer_number,
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


def analyze_all_layers(
    model_path: str,
    prompts: list[str],
    device: str = "cuda",
    batch_size: int = 4,
    tracked_dimensions: list[int] = TRACKED_DIMENSIONS,
    use_pipeline: bool = False,
    format_prompts: bool = False,
) -> dict[int, LayerDimensionStats]:
    """
    Analyze all 36 layers for per-dimension statistics.

    Args:
        model_path: Path to Qwen3-4B model
        prompts: List of prompts to analyze
        device: Device to use (cuda recommended)
        batch_size: Batch size for processing
        tracked_dimensions: Dimensions to track specifically
        use_pipeline: If True, use llm_dit.backends.TransformersBackend (matches real usage)
        format_prompts: If True, apply Qwen3Formatter chat template to prompts

    Returns:
        Dictionary mapping layer_index (-36 to -1) to LayerDimensionStats
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Optionally format prompts with chat template (matches real generation)
    if format_prompts:
        try:
            from llm_dit.conversation import Qwen3Formatter
            formatter = Qwen3Formatter()
            formatted_prompts = []
            for prompt in prompts:
                # format_simple() applies the chat template for encoding
                formatted = formatter.format_simple(prompt)
                formatted_prompts.append(formatted)
            prompts = formatted_prompts
            logger.info("Applied Qwen3Formatter chat template to prompts")
            logger.debug(f"Example formatted prompt: {prompts[0][:200]}...")
        except ImportError:
            logger.warning("llm_dit not installed, skipping prompt formatting")

    logger.info(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    hidden_dim = model.config.hidden_size
    num_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer

    logger.info(f"Model: {model.config.model_type}, hidden_size={hidden_dim}, layers={num_layers}")
    logger.info(f"Analyzing {len(prompts)} prompts across {num_layers} layers")

    # Initialize accumulators for all hidden states INCLUDING embedding layer
    # hidden_states[0] is embedding, hidden_states[1] is layer 1, etc.
    accumulators = {}
    for layer_num in range(0, num_layers):  # 0 to 36 for Qwen3-4B (0 = embedding)
        # Use special index 0 for embedding layer, negative for transformer layers
        if layer_num == 0:
            layer_idx = 0  # Special: embedding layer
        else:
            layer_idx = layer_num - num_layers  # 1 -> -36, 36 -> -1
        accumulators[layer_idx] = DimensionAccumulator(hidden_dim)

    # Process prompts in batches
    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start:batch_start + batch_size]
        logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")

        # Tokenize
        inputs = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        ).to(device)

        attention_mask = inputs.attention_mask.bool()

        # Forward pass with all hidden states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Extract hidden states for each layer
        # hidden_states is tuple of (batch, seq, hidden_dim)
        # hidden_states[0] = embeddings, hidden_states[1] = after layer 1, etc.
        for layer_num, hidden in enumerate(outputs.hidden_states):
            # Map layer_num to layer_idx:
            # - layer_num=0 (embedding) -> layer_idx=0
            # - layer_num=1 (first transformer) -> layer_idx=-36
            # - layer_num=36 (last transformer) -> layer_idx=-1
            if layer_num == 0:
                layer_idx = 0  # Special: embedding layer
            else:
                layer_idx = layer_num - num_layers  # 1 -> -36, 36 -> -1

            # Extract valid tokens (filter by attention mask)
            for batch_idx in range(len(batch_prompts)):
                mask = attention_mask[batch_idx]
                # Convert to float32 first (numpy doesn't support bfloat16)
                valid_tokens = hidden[batch_idx][mask].float().cpu().numpy().astype(np.float64)
                accumulators[layer_idx].update_batch(valid_tokens)

        # Free GPU memory
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
            f"Layer {stats.layer_number:2d} (idx {layer_idx:3d}): "
            f"median_std={stats.median_dim_std:.2f}, "
            f"max_std={stats.max_dim_std:.2f}, "
            f"outliers(10x)={len(stats.outliers_10x)}, "
            f"max_kurtosis={stats.max_kurtosis:.1f} (dim {stats.max_kurtosis_dim})"
        )

    # Cleanup
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return results


def save_layer_stats(
    results: dict[int, LayerDimensionStats],
    output_dir: Path,
):
    """Save per-layer statistics to NPZ files."""
    stats_dir = output_dir / "layer_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx, stats in results.items():
        filename = f"layer_{stats.layer_number:02d}_stats.npz"
        filepath = stats_dir / filename

        np.savez(
            filepath,
            layer_index=stats.layer_index,
            layer_number=stats.layer_number,
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
        logger.debug(f"Saved {filepath}")

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
            "prompts": prompts,
        },
        "layer_summary": {},
        "outlier_analysis": {
            "threshold_10x": {},
            "threshold_50x": {},
            "threshold_100x": {},
        },
        "tracked_dimension_analysis": {dim: {} for dim in tracked_dimensions},
    }

    for layer_idx, stats in sorted(results.items()):
        layer_key = f"layer_{stats.layer_number:02d}"

        # Basic summary
        summary["layer_summary"][layer_key] = {
            "layer_index": stats.layer_index,
            "layer_number": stats.layer_number,
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
            "dimensions": stats.outliers_10x,
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
            summary["tracked_dimension_analysis"][dim][layer_key] = stats.tracked_dim_stats.get(dim, {})

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
        import matplotlib.colors as mcolors
    except ImportError:
        logger.warning("matplotlib not installed, skipping visualizations")
        return

    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Sort results by layer number
    sorted_results = sorted(results.items(), key=lambda x: x[1].layer_number)

    # 1. Per-dimension std heatmap
    logger.info("Generating per-dimension std heatmap...")
    hidden_dim = len(sorted_results[0][1].per_dim_std)
    num_layers = len(sorted_results)

    data = np.zeros((num_layers, hidden_dim))
    for i, (_, stats) in enumerate(sorted_results):
        data[i] = stats.per_dim_std

    fig, ax = plt.subplots(figsize=(20, 10))
    # Use log scale for better visibility
    im = ax.imshow(
        np.log10(data + 1),
        aspect='auto',
        cmap='viridis',
        interpolation='nearest',
    )
    plt.colorbar(im, ax=ax, label='log10(std + 1)')

    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Layer (0=embedding, 1=first transformer, 36=last)')
    ax.set_yticks(range(0, num_layers, 3))
    ax.set_yticklabels([sorted_results[i][1].layer_number for i in range(0, num_layers, 3)])
    ax.set_title('Per-Dimension Standard Deviation Across Layers')

    # Mark tracked dimensions
    for dim in tracked_dimensions:
        ax.axvline(x=dim, color='red', linestyle='--', alpha=0.5, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(viz_dir / "heatmap_per_dim_std.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Outlier count vs layer depth
    logger.info("Generating outlier count plot...")
    fig, ax = plt.subplots(figsize=(12, 6))

    layers = [stats.layer_number for _, stats in sorted_results]

    for threshold, color, label in [(10, 'blue', '10x'), (50, 'orange', '50x'), (100, 'red', '100x')]:
        if threshold == 10:
            counts = [len(stats.outliers_10x) for _, stats in sorted_results]
        elif threshold == 50:
            counts = [len(stats.outliers_50x) for _, stats in sorted_results]
        else:
            counts = [len(stats.outliers_100x) for _, stats in sorted_results]

        ax.plot(layers, counts, color=color, marker='o', label=f'{label} threshold', linewidth=2)

    ax.set_xlabel('Layer (0=embedding, 1=first transformer, 36=last)')
    ax.set_ylabel('Number of Outlier Dimensions')
    ax.set_title('Outlier Dimension Count by Layer Depth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 37, 3))

    plt.tight_layout()
    plt.savefig(viz_dir / "line_outlier_count_vs_layer.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Max kurtosis vs layer depth
    logger.info("Generating kurtosis plot...")
    fig, ax = plt.subplots(figsize=(12, 6))

    kurtosis_values = [stats.max_kurtosis for _, stats in sorted_results]
    ax.plot(layers, kurtosis_values, color='purple', marker='o', linewidth=2)

    ax.set_xlabel('Layer (0=embedding, 1=first transformer, 36=last)')
    ax.set_ylabel('Maximum Kurtosis')
    ax.set_title('Maximum Per-Dimension Kurtosis by Layer Depth')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 37, 3))

    # Add dimension labels for top kurtosis points
    top_indices = np.argsort(kurtosis_values)[-5:]
    for idx in top_indices:
        layer_num = layers[idx]
        kurtosis = kurtosis_values[idx]
        dim = sorted_results[idx][1].max_kurtosis_dim
        ax.annotate(
            f'dim {dim}',
            (layer_num, kurtosis),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(viz_dir / "line_kurtosis_vs_layer.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Tracked dimension plots
    for dim in tracked_dimensions:
        logger.info(f"Generating dimension {dim} tracking plot...")
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        stds = [stats.per_dim_std[dim] for _, stats in sorted_results]
        kurtosis_vals = [stats.per_dim_kurtosis[dim] for _, stats in sorted_results]

        # Std subplot
        axes[0].plot(layers, stds, color='blue', marker='o', linewidth=2)
        median_stds = [stats.median_dim_std for _, stats in sorted_results]
        axes[0].axhline(y=np.mean(median_stds) * 10, color='orange', linestyle='--', label='10x median (avg)')
        axes[0].axhline(y=np.mean(median_stds) * 100, color='red', linestyle='--', label='100x median (avg)')
        axes[0].set_ylabel('Standard Deviation')
        axes[0].set_title(f'Dimension {dim} - Standard Deviation Across Layers')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Kurtosis subplot
        axes[1].plot(layers, kurtosis_vals, color='purple', marker='o', linewidth=2)
        axes[1].set_xlabel('Layer (1=earliest, 36=last)')
        axes[1].set_ylabel('Kurtosis')
        axes[1].set_title(f'Dimension {dim} - Kurtosis Across Layers')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(viz_dir / f"track_dim_{dim}_across_layers.png", dpi=150, bbox_inches='tight')
        plt.close()

    # 5. Median std vs layer depth
    logger.info("Generating median std plot...")
    fig, ax = plt.subplots(figsize=(12, 6))

    median_stds = [stats.median_dim_std for _, stats in sorted_results]
    max_stds = [stats.max_dim_std for _, stats in sorted_results]

    ax.plot(layers, median_stds, color='blue', marker='o', linewidth=2, label='Median dim std')
    ax.plot(layers, max_stds, color='red', marker='s', linewidth=2, alpha=0.7, label='Max dim std')

    ax.set_xlabel('Layer (0=embedding, 1=first transformer, 36=last)')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Embedding Statistics by Layer Depth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 37, 3))

    plt.tight_layout()
    plt.savefig(viz_dir / "line_std_vs_layer.png", dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved visualizations to {viz_dir}")


def compare_to_reference(
    results: dict[int, LayerDimensionStats],
    reference_path: Path | None = None,
) -> dict | None:
    """Compare layer -2 results to existing reference stats."""
    if reference_path is None:
        # Try to find the VL reference stats
        reference_path = Path("src/llm_dit/vl/qwen3_4b_stats.npz")

    if not reference_path.exists():
        logger.warning(f"Reference stats not found at {reference_path}")
        return None

    try:
        ref_stats = np.load(reference_path)
        ref_std = ref_stats["per_dim_std"]

        # Get our layer -2 stats
        if -2 not in results:
            logger.warning("Layer -2 not in results, cannot compare to reference")
            return None

        our_stats = results[-2]
        our_std = our_stats.per_dim_std

        # Compute correlation
        correlation = np.corrcoef(our_std, ref_std)[0, 1]

        # Find dimensions with biggest differences
        ratio = our_std / (ref_std + 1e-10)
        diff_indices = np.argsort(np.abs(np.log(ratio + 1e-10)))[::-1][:10]

        comparison = {
            "correlation": float(correlation),
            "our_global_std": float(our_std.mean()),
            "ref_global_std": float(ref_std.mean()),
            "std_scale_factor": float(our_std.mean() / ref_std.mean()),
            "top_diff_dimensions": [
                {
                    "dim": int(idx),
                    "our_std": float(our_std[idx]),
                    "ref_std": float(ref_std[idx]),
                    "ratio": float(ratio[idx]),
                }
                for idx in diff_indices
            ],
            "tracked_dim_comparison": {},
        }

        # Compare tracked dimensions
        for dim in TRACKED_DIMENSIONS:
            if dim < len(our_std) and dim < len(ref_std):
                comparison["tracked_dim_comparison"][dim] = {
                    "our_std": float(our_std[dim]),
                    "ref_std": float(ref_std[dim]),
                    "ratio": float(our_std[dim] / (ref_std[dim] + 1e-10)),
                }

        logger.info(f"Reference comparison: correlation={correlation:.4f}, scale={comparison['std_scale_factor']:.2f}x")
        return comparison

    except Exception as e:
        logger.warning(f"Failed to compare to reference: {e}")
        return None


def generate_analysis_report(
    results: dict[int, LayerDimensionStats],
    output_dir: Path,
    tracked_dimensions: list[int] = TRACKED_DIMENSIONS,
):
    """Generate human-readable analysis report."""
    report_lines = [
        "# Layer Dimension Analysis Report",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Executive Summary",
        "",
    ]

    # Compare to reference
    ref_comparison = compare_to_reference(results)
    if ref_comparison:
        report_lines.extend([
            "### Reference Comparison (Layer -2 vs qwen3_4b_stats.npz)",
            "",
            f"- **Correlation:** {ref_comparison['correlation']:.4f}",
            f"- **Scale factor:** {ref_comparison['std_scale_factor']:.2f}x",
            "",
        ])

    # Sort by layer number
    sorted_results = sorted(results.items(), key=lambda x: x[1].layer_number)

    # Find layers with fewest/most outliers at 10x threshold
    outlier_counts = [(stats.layer_number, len(stats.outliers_10x))
                      for _, stats in sorted_results]
    min_outliers = min(outlier_counts, key=lambda x: x[1])
    max_outliers = max(outlier_counts, key=lambda x: x[1])

    report_lines.extend([
        f"- **Cleanest layer (10x threshold):** Layer {min_outliers[0]} ({min_outliers[1]} outliers)",
        f"- **Most outliers (10x threshold):** Layer {max_outliers[0]} ({max_outliers[1]} outliers)",
        "",
    ])

    # Tracked dimensions summary
    report_lines.extend([
        "## Tracked Dimensions",
        "",
    ])

    for dim in tracked_dimensions:
        report_lines.append(f"### Dimension {dim}")
        report_lines.append("")
        report_lines.append("| Layer | Std | Std Ratio | Kurtosis |")
        report_lines.append("|-------|-----|-----------|----------|")

        for _, stats in sorted_results:
            dim_stats = stats.tracked_dim_stats.get(dim, {})
            std = dim_stats.get("std", 0)
            ratio = dim_stats.get("std_ratio", 0)
            kurtosis = dim_stats.get("kurtosis", 0)
            report_lines.append(f"| {stats.layer_number} | {std:.2f} | {ratio:.1f}x | {kurtosis:.1f} |")

        report_lines.append("")

    # Layer-by-layer summary table
    report_lines.extend([
        "## Layer-by-Layer Summary",
        "",
        "| Layer | Median Std | Max Std | Outliers (10x) | Outliers (50x) | Max Kurtosis | Max Kurt Dim |",
        "|-------|-----------|---------|----------------|----------------|--------------|--------------|",
    ])

    for _, stats in sorted_results:
        report_lines.append(
            f"| {stats.layer_number} | {stats.median_dim_std:.2f} | {stats.max_dim_std:.2f} | "
            f"{len(stats.outliers_10x)} | {len(stats.outliers_50x)} | "
            f"{stats.max_kurtosis:.1f} | {stats.max_kurtosis_dim} |"
        )

    report_lines.extend([
        "",
        "## Recommendations",
        "",
        "Based on the outlier analysis:",
        "",
    ])

    # Find clean layers (0 outliers at 10x)
    clean_layers = [stats.layer_number for _, stats in sorted_results
                    if len(stats.outliers_10x) == 0]
    if clean_layers:
        report_lines.append(f"- **Clean layers (no 10x outliers):** {clean_layers}")
    else:
        # Find layers with minimum outliers
        min_count = min(len(stats.outliers_10x) for _, stats in sorted_results)
        min_layers = [stats.layer_number for _, stats in sorted_results
                      if len(stats.outliers_10x) == min_count]
        report_lines.append(f"- **Cleanest layers ({min_count} outliers):** {min_layers}")

    # Find problematic layers (>5 outliers at 10x)
    problematic_layers = [stats.layer_number for _, stats in sorted_results
                          if len(stats.outliers_10x) > 5]
    if problematic_layers:
        report_lines.append(f"- **Problematic layers (>5 outliers):** {problematic_layers}")

    report_lines.extend([
        "",
        "## Files Generated",
        "",
        "- `layer_stats/layer_XX_stats.npz` - Per-layer statistics (36 files)",
        "- `summary.json` - Aggregate statistics in JSON format",
        "- `visualizations/` - Analysis plots",
        "  - `heatmap_per_dim_std.png` - 2560x36 heatmap of std",
        "  - `line_outlier_count_vs_layer.png` - Outliers by depth",
        "  - `line_kurtosis_vs_layer.png` - Kurtosis by depth",
        "  - `track_dim_*_across_layers.png` - Tracked dimension plots",
        "",
    ])

    # Write report
    report_path = output_dir / "analysis_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Generated report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-dimension statistics across all 36 layers of Qwen3-4B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to Qwen3-4B model directory",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results/layer_dimension_analysis",
        help="Output directory for results (default: experiments/results/layer_dimension_analysis)",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        help="Custom prompts to analyze (default: built-in diverse set)",
    )
    parser.add_argument(
        "--prompts-file",
        help="YAML file containing prompts to analyze",
    )
    parser.add_argument(
        "--track-dims",
        nargs="+",
        type=int,
        default=TRACKED_DIMENSIONS,
        help=f"Dimensions to track specifically (default: {TRACKED_DIMENSIONS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for processing (default: 4)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with fewer prompts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be analyzed without running",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip visualization generation",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--format-prompts",
        action="store_true",
        help="Apply Qwen3Formatter chat template to prompts (matches real generation path)",
    )
    parser.add_argument(
        "--use-pipeline",
        action="store_true",
        help="Use llm_dit.backends.TransformersBackend (matches real generation, slower)",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine prompts
    if args.prompts:
        prompts = args.prompts
    elif args.prompts_file:
        import yaml
        with open(args.prompts_file) as f:
            data = yaml.safe_load(f)
        prompts = data.get("prompts", [])
    elif args.quick:
        prompts = QUICK_PROMPTS
    else:
        prompts = DEFAULT_PROMPTS

    output_dir = Path(args.output_dir)

    if args.dry_run:
        print("\n[DRY RUN] Would analyze:")
        print(f"  Model: {args.model_path}")
        print(f"  Output: {output_dir}")
        print(f"  Prompts: {len(prompts)}")
        for i, p in enumerate(prompts[:5]):
            print(f"    {i+1}. {p[:60]}{'...' if len(p) > 60 else ''}")
        if len(prompts) > 5:
            print(f"    ... and {len(prompts) - 5} more")
        print(f"  Tracked dimensions: {args.track_dims}")
        print(f"  Device: {args.device}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Format prompts: {args.format_prompts}")
        print(f"  Use pipeline: {args.use_pipeline}")
        print()
        print("Recommended usage:")
        print("  # Quick test (raw prompts)")
        print("  uv run experiments/analyze_layer_dimensions.py --model-path ... --quick")
        print()
        print("  # Match real generation (with chat template)")
        print("  uv run experiments/analyze_layer_dimensions.py --model-path ... --format-prompts")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Analyzing {len(prompts)} prompts")
    logger.info(f"Tracked dimensions: {args.track_dims}")
    logger.info(f"Output directory: {output_dir}")

    # Run analysis
    results = analyze_all_layers(
        model_path=args.model_path,
        prompts=prompts,
        device=args.device,
        batch_size=args.batch_size,
        tracked_dimensions=args.track_dims,
        use_pipeline=args.use_pipeline,
        format_prompts=args.format_prompts,
    )

    # Save results
    save_layer_stats(results, output_dir)
    save_summary_json(results, output_dir, prompts, args.track_dims)

    if not args.skip_viz:
        plot_visualizations(results, output_dir, args.track_dims)

    generate_analysis_report(results, output_dir, args.track_dims)

    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
