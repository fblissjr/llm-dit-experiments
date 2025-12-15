#!/usr/bin/env python3
"""
Visualize per-dimension analysis results.

Last updated: 2025-12-14

Creates plots to visualize:
1. Std ratio distribution across all dimensions
2. Top outlier dimensions
3. Dead vs hyperactive dimensions comparison
4. Attention-weighted differences

Usage:
    uv run experiments/visualize_perdim_results.py
    uv run experiments/visualize_perdim_results.py --input results/embedding_perdim_analysis.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_std_ratio_distribution(comparison_data, output_dir):
    """Plot histogram of std ratios across all dimensions."""
    # Extract std ratios from top dimensions
    top_high = comparison_data['top_different_dimensions']['highest_std_ratio']
    top_low = comparison_data['top_different_dimensions']['lowest_std_ratio']

    # Get summary stats
    std_stats = comparison_data['std_ratios']

    fig, ax = plt.subplots(figsize=(12, 6))

    # Show summary stats as text
    stats_text = f"""Std Ratio Statistics:
Mean: {std_stats['mean']:.3f}
Median: {std_stats['median']:.3f}
Range: {std_stats['min']:.3f} - {std_stats['max']:.3f}

Outliers (>3x or <0.33x):
High variance: {len(comparison_data['outlier_dimensions']['high_variance'])}
Low variance: {len(comparison_data['outlier_dimensions']['low_variance'])}
"""

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.axis('off')
    ax.set_title('Std Ratio Distribution Summary', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'std_ratio_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_dir / 'std_ratio_summary.png'}")


def plot_top_outliers(comparison_data, output_dir, top_n=20):
    """Plot top outlier dimensions by std ratio."""
    top_high = comparison_data['top_different_dimensions']['highest_std_ratio'][:top_n]
    top_low = comparison_data['top_different_dimensions']['lowest_std_ratio'][:top_n]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # High variance outliers
    dims_high = [d['dim'] for d in top_high]
    ratios_high = [d['ratio'] for d in top_high]

    ax1.barh(range(len(dims_high)), ratios_high, color='red', alpha=0.7)
    ax1.set_yticks(range(len(dims_high)))
    ax1.set_yticklabels([f"Dim {d}" for d in dims_high])
    ax1.set_xlabel('Std Ratio (emb_std / qwen3_std)', fontsize=12)
    ax1.set_title(f'Top {top_n} High Variance Dimensions', fontsize=14, fontweight='bold')
    ax1.axvline(x=3.0, color='black', linestyle='--', linewidth=1, label='3x threshold')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)

    # Annotate with actual std values
    for i, d in enumerate(top_high):
        ax1.text(d['ratio'], i, f"  {d['emb_std']:.1f}/{d['qwen3_std']:.1f}",
                va='center', fontsize=8)

    # Low variance outliers
    dims_low = [d['dim'] for d in top_low]
    ratios_low = [d['ratio'] for d in top_low]

    ax2.barh(range(len(dims_low)), ratios_low, color='blue', alpha=0.7)
    ax2.set_yticks(range(len(dims_low)))
    ax2.set_yticklabels([f"Dim {d}" for d in dims_low])
    ax2.set_xlabel('Std Ratio (emb_std / qwen3_std)', fontsize=12)
    ax2.set_title(f'Top {top_n} Low Variance Dimensions', fontsize=14, fontweight='bold')
    ax2.axvline(x=0.33, color='black', linestyle='--', linewidth=1, label='0.33x threshold')
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)

    # Annotate
    for i, d in enumerate(top_low):
        ax2.text(d['ratio'], i, f"  {d['emb_std']:.1f}/{d['qwen3_std']:.1f}",
                va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'top_outliers.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_dir / 'top_outliers.png'}")


def plot_dimension_categories(comparison_data, qwen3_stats, emb_stats, output_dir):
    """Plot comparison of dead/hyperactive dimensions."""
    categories = ['Dead\n(std < 0.01)', 'Hyperactive\n(std > 5x median)']

    qwen3_dead = len(qwen3_stats['dead_dimensions'])
    emb_dead = len(emb_stats['dead_dimensions'])
    qwen3_hyper = len(qwen3_stats['hyperactive_dimensions'])
    emb_hyper = len(emb_stats['hyperactive_dimensions'])

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width/2, [qwen3_dead, qwen3_hyper], width, label='Qwen3-4B', color='green', alpha=0.7)
    ax.bar(x + width/2, [emb_dead, emb_hyper], width, label='Qwen3-Embedding', color='orange', alpha=0.7)

    ax.set_ylabel('Number of Dimensions', fontsize=12)
    ax.set_title('Dead and Hyperactive Dimensions Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate([qwen3_dead, qwen3_hyper]):
        ax.text(i - width/2, v + 1, str(v), ha='center', va='bottom', fontweight='bold')
    for i, v in enumerate([emb_dead, emb_hyper]):
        ax.text(i + width/2, v + 1, str(v), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'dimension_categories.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_dir / 'dimension_categories.png'}")


def plot_correlation_summary(comparison_data, per_prompt_data, output_dir):
    """Plot correlation metrics across prompts."""
    prompts = list(per_prompt_data.keys())
    global_sims = [per_prompt_data[p]['global_cosine_similarity'] for p in prompts]
    mean_corrs = [per_prompt_data[p]['comparison']['mean_correlation'] for p in prompts]
    std_corrs = [per_prompt_data[p]['comparison']['std_correlation'] for p in prompts]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(prompts))
    width = 0.25

    ax.bar(x - width, global_sims, width, label='Global Cosine Sim', color='purple', alpha=0.7)
    ax.bar(x, mean_corrs, width, label='Mean Correlation', color='blue', alpha=0.7)
    ax.bar(x + width, std_corrs, width, label='Std Correlation', color='red', alpha=0.7)

    ax.set_ylabel('Correlation / Similarity', fontsize=12)
    ax.set_title('Embedding Similarity Across Prompts', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p[:30] + '...' if len(p) > 30 else p for p in prompts],
                       rotation=15, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (gs, mc, sc) in enumerate(zip(global_sims, mean_corrs, std_corrs)):
        ax.text(i - width, gs + 0.01, f'{gs:.3f}', ha='center', va='bottom', fontsize=7)
        ax.text(i, mc + 0.01, f'{mc:.3f}', ha='center', va='bottom', fontsize=7)
        ax.text(i + width, sc + 0.01, f'{sc:.3f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_dir / 'correlation_summary.png'}")


def generate_report(data, output_path):
    """Generate markdown report with key findings."""
    report = f"""# Per-Dimension Analysis Report

Generated: 2025-12-14

## Summary

- **Prompts analyzed:** {data['metadata']['num_prompts']}
- **Hidden layer:** {data['metadata']['hidden_layer']}

## Aggregated Results

### Global Statistics

"""

    agg = data['aggregated_analysis']
    comp = agg['comparison']

    report += f"""
- **Mean correlation:** {comp['mean_correlation']:.4f}
- **Std correlation:** {comp['std_correlation']:.4f}
- **Std ratio range:** {comp['std_ratios']['min']:.3f} - {comp['std_ratios']['max']:.3f}
- **Std ratio median:** {comp['std_ratios']['median']:.3f}

### Outlier Dimensions

- **High variance outliers (>3x):** {len(comp['outlier_dimensions']['high_variance'])}
- **Low variance outliers (<0.33x):** {len(comp['outlier_dimensions']['low_variance'])}

#### Top 10 High Variance Outliers

| Dimension | Std Ratio | Emb Std | Qwen3 Std |
|-----------|-----------|---------|-----------|
"""

    for d in comp['top_different_dimensions']['highest_std_ratio'][:10]:
        report += f"| {d['dim']} | {d['ratio']:.3f}x | {d['emb_std']:.2f} | {d['qwen3_std']:.2f} |\n"

    report += f"""
#### Top 10 Low Variance Outliers

| Dimension | Std Ratio | Emb Std | Qwen3 Std |
|-----------|-----------|---------|-----------|
"""

    for d in comp['top_different_dimensions']['lowest_std_ratio'][:10]:
        report += f"| {d['dim']} | {d['ratio']:.3f}x | {d['emb_std']:.2f} | {d['qwen3_std']:.2f} |\n"

    report += f"""
### Dead and Hyperactive Dimensions

#### Qwen3-4B
- Dead dimensions: {len(agg['qwen3_stats']['dead_dimensions'])}
- Hyperactive dimensions: {len(agg['qwen3_stats']['hyperactive_dimensions'])}

#### Qwen3-Embedding-4B
- Dead dimensions: {len(agg['embedding_stats']['dead_dimensions'])}
- Hyperactive dimensions: {len(agg['embedding_stats']['hyperactive_dimensions'])}

#### Differences
- Dead (Embedding-only): {len(comp['dead_dimensions']['emb_only'])}
- Dead (Qwen3-only): {len(comp['dead_dimensions']['qwen3_only'])}
- Hyperactive (Embedding-only): {len(comp['hyperactive_dimensions']['emb_only'])}
- Hyperactive (Qwen3-only): {len(comp['hyperactive_dimensions']['qwen3_only'])}

### Attention Sensitivity

- **Attention weight correlation:** {agg['attention_sensitivity']['attention_weight_correlation']:.4f}

Top 5 dimensions by attention-weighted difference:

| Dimension | Diff | Emb Weighted Mean | Qwen3 Weighted Mean |
|-----------|------|-------------------|---------------------|
"""

    for d in agg['attention_sensitivity']['top_weighted_diff_dimensions'][:5]:
        report += f"| {d['dim']} | {d['diff']:.3f} | {d['emb_weighted_mean']:.3f} | {d['qwen3_weighted_mean']:.3f} |\n"

    report += """
## Key Findings

### Hypothesis Validation

"""

    # Analyze results and provide findings
    high_outliers = len(comp['outlier_dimensions']['high_variance'])
    low_outliers = len(comp['outlier_dimensions']['low_variance'])

    if high_outliers > 10 or low_outliers > 10:
        report += f"""
**✓ Hypothesis 1 (Outlier Dimensions): CONFIRMED**
- Found {high_outliers} high-variance and {low_outliers} low-variance outliers
- These dimensions likely cause visual artifacts
- **Recommendation:** Apply per-dimension rescaling to these outliers
"""

    emb_dead = len(agg['embedding_stats']['dead_dimensions'])
    qwen3_dead = len(agg['qwen3_stats']['dead_dimensions'])

    if emb_dead > qwen3_dead + 10:
        report += f"""
**✓ Hypothesis 2 (Dead Dimensions): CONFIRMED**
- Embedding model has {emb_dead - qwen3_dead} more dead dimensions
- May create semantic gaps filled with default imagery
- **Recommendation:** Inject small noise or use Qwen3-4B values for dead dimensions
"""

    emb_hyper = len(agg['embedding_stats']['hyperactive_dimensions'])
    qwen3_hyper = len(agg['qwen3_stats']['hyperactive_dimensions'])
    emb_only_hyper = len(comp['hyperactive_dimensions']['emb_only'])

    if emb_only_hyper > 10:
        report += f"""
**✓ Hypothesis 3 (Hyperactive Dimensions): CONFIRMED**
- Found {emb_only_hyper} hyperactive dimensions unique to embedding model
- Likely encode discriminative features irrelevant for generation
- **Recommendation:** Clamp these dimensions to match Qwen3-4B distribution
"""

    if comp['std_correlation'] < 0.9:
        report += f"""
**✓ Hypothesis 4 (Distribution Mismatch): CONFIRMED**
- Std correlation is only {comp['std_correlation']:.3f} (< 0.9 threshold)
- Distribution shapes differ significantly
- **Recommendation:** Apply distribution matching or rank normalization
"""

    report += """
## Recommended Fixes

### Priority 1: Rescale Outlier Dimensions

Apply per-dimension rescaling to the identified outliers:

```python
def fix_outlier_dimensions(embeddings, std_ratios, threshold=3.0):
    outliers = (std_ratios > threshold) | (std_ratios < 1/threshold)
    for dim in outliers.nonzero():
        embeddings[:, dim] *= (qwen3_stds[dim] / emb_stds[dim])
    return embeddings
```

### Priority 2: Handle Dead Dimensions

For embedding-only dead dimensions, either:
1. Zero them out completely
2. Copy values from Qwen3-4B for same prompt
3. Inject small noise matching Qwen3-4B distribution

### Priority 3: Clamp Hyperactive Dimensions

For embedding-only hyperactive dimensions:
```python
for dim in emb_only_hyperactive:
    embeddings[:, dim] = torch.clamp(
        embeddings[:, dim],
        min=-3 * qwen3_stds[dim],
        max=3 * qwen3_stds[dim]
    )
```

## Next Steps

1. Generate test images with dimension fixes applied
2. Validate that artifacts disappear
3. Measure quality vs Qwen3-4B baseline
4. Document dimension masks for production use
5. Add `--fix-embedding-dimensions` flag to pipeline
"""

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize per-dimension analysis results")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiments/results/embedding_perdim_analysis.json"),
        help="Path to analysis JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/perdim_visualizations"),
        help="Output directory for plots"
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        print("Run analyze_embedding_perdim.py first to generate analysis data.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading analysis results from {args.input}...")
    with open(args.input) as f:
        data = json.load(f)

    print(f"\nGenerating visualizations in {args.output_dir}...")

    agg = data['aggregated_analysis']

    print("\n1. Plotting std ratio distribution...")
    plot_std_ratio_distribution(agg['comparison'], args.output_dir)

    print("\n2. Plotting top outliers...")
    plot_top_outliers(agg['comparison'], args.output_dir, top_n=20)

    print("\n3. Plotting dimension categories...")
    plot_dimension_categories(
        agg['comparison'],
        agg['qwen3_stats'],
        agg['embedding_stats'],
        args.output_dir
    )

    if data['metadata']['num_prompts'] > 1:
        print("\n4. Plotting correlation summary...")
        plot_correlation_summary(
            agg['comparison'],
            data['per_prompt_analysis'],
            args.output_dir
        )

    print("\n5. Generating markdown report...")
    generate_report(data, args.output_dir / 'analysis_report.md')

    print("\n" + "="*60)
    print("Visualization complete!")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
