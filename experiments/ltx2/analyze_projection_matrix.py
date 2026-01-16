#!/usr/bin/env python3
"""
LTX-2 Projection Matrix Analysis

Last Updated: 2026-01-16

Zero-inference analysis of the learned text projection weights.

Architecture insight (from dit-specialist):
- Gemma outputs 49 layers × 3840 hidden dims = 188160 total dims
- _pack_text_embeds just normalizes and flattens (no learned params)
- The learned projection is in connectors.text_proj_in: nn.Linear(188160, 3840)
- This means W is [3840, 188160] and can be reshaped to [3840, 49, 3840]

By analyzing W, we can discover which layers LTX-2 learned to weight most heavily
WITHOUT running any inference.

Discovery Question: "What layer weighting did LTX-2 learn?"

Usage:
    uv run python experiments/ltx2/analyze_projection_matrix.py

    # With custom model path
    uv run python experiments/ltx2/analyze_projection_matrix.py --model-path models/LTX-2
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def analyze_projection_matrix(
    model_path: str = "models/LTX-2",
    output_dir: str = "experiments/results/projection_analysis",
):
    """
    Analyze the learned projection matrix W from LTX-2's text connector.

    The projection W maps from 188160 (49 layers × 3840 dims) to 3840.
    By computing norms per "layer block", we can see which layers LTX-2
    learned to weight most heavily.
    """
    from diffusers import LTX2Pipeline

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LTX-2 Projection Matrix Analysis")
    print("=" * 60)

    # Load pipeline (CPU only - no GPU needed for this analysis)
    print("\nLoading model weights...")
    pipe = LTX2Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # Full precision for analysis
    )

    # Find the projection matrix
    # The text_proj_in is in pipe.connectors, NOT pipe.transformer
    # pipe.transformer = DiT model
    # pipe.connectors = LTX2TextConnectors with text_proj_in

    print("\nExploring model structure...")

    W = None
    W_name = None

    # Primary location: pipe.connectors.text_proj_in
    if hasattr(pipe, "connectors") and pipe.connectors is not None:
        if hasattr(pipe.connectors, "text_proj_in"):
            W = pipe.connectors.text_proj_in.weight.data
            W_name = "connectors.text_proj_in"
            print(f"Found: {W_name}")

    # Fallback: Search in connectors state dict
    if W is None and hasattr(pipe, "connectors") and pipe.connectors is not None:
        print("\nSearching connectors state dict...")
        for name, param in pipe.connectors.named_parameters():
            print(f"  {name}: {param.shape}")
            if "text_proj_in" in name or ("text" in name.lower() and "proj" in name.lower()):
                if param.dim() == 2 and 188160 in param.shape:
                    W = param.data
                    W_name = f"connectors.{name}"
                    print(f"  Selected: {W_name}")
                    break

    # Fallback: Try transformer (older diffusers versions)
    if W is None:
        transformer = pipe.transformer
        if hasattr(transformer, "text_proj_in"):
            W = transformer.text_proj_in.weight.data
            W_name = "transformer.text_proj_in"
            print(f"Found: {W_name}")

    # Last resort: Search entire pipeline
    if W is None:
        print("\nSearching entire pipeline for text projection...")
        for component_name in ["connectors", "transformer", "text_encoder"]:
            component = getattr(pipe, component_name, None)
            if component is None:
                continue
            for name, param in component.named_parameters():
                if param.dim() == 2 and 188160 in param.shape:
                    W = param.data
                    W_name = f"{component_name}.{name}"
                    print(f"  Found 188160-dim projection: {W_name}")
                    break
            if W is not None:
                break

    if W is None:
        # Debug: List all available components and their large params
        print("\nCould not find text_proj_in (188160 → 3840). Listing pipeline structure:")
        print(f"  pipe.connectors exists: {hasattr(pipe, 'connectors') and pipe.connectors is not None}")
        if hasattr(pipe, "connectors") and pipe.connectors is not None:
            print(f"  connectors type: {type(pipe.connectors)}")
            print("  connectors parameters:")
            for name, param in pipe.connectors.named_parameters():
                print(f"    {name}: {param.shape}")

        print("\nPlease specify the correct parameter name manually.")
        return None

    print(f"\nProjection matrix: {W_name}")
    print(f"Shape: {W.shape}")  # Expect [3840, 188160] or [188160, 3840]

    # Ensure W is [out_features, in_features] = [3840, 188160]
    if W.shape[0] == 188160:
        W = W.T
        print(f"Transposed to: {W.shape}")

    # Verify dimensions
    num_layers = 49
    hidden_dim = 3840
    expected_in = num_layers * hidden_dim

    if W.shape[1] != expected_in:
        print(f"WARNING: Expected in_features={expected_in}, got {W.shape[1]}")
        print("The model may have a different architecture than expected.")

    # Reshape to [out_dim, num_layers, hidden_dim]
    out_dim = W.shape[0]
    W_reshaped = W.reshape(out_dim, num_layers, hidden_dim)
    print(f"Reshaped to: {W_reshaped.shape}")

    # Compute per-layer statistics
    print("\nComputing per-layer statistics...")

    # Frobenius norm per layer block: ||W[:, i, :]||_F
    layer_norms = torch.norm(W_reshaped, dim=(0, 2)).numpy()  # [num_layers]

    # Mean absolute value per layer
    layer_mean_abs = W_reshaped.abs().mean(dim=(0, 2)).numpy()

    # Max absolute value per layer
    layer_max_abs = W_reshaped.abs().amax(dim=(0, 2)).numpy()

    # Variance per layer
    layer_var = W_reshaped.var(dim=(0, 2)).numpy()

    # Create analysis summary
    analysis = {
        "model_path": model_path,
        "projection_name": W_name,
        "shape": list(W.shape),
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "per_layer": {},
    }

    print("\n" + "=" * 60)
    print("PER-LAYER WEIGHT ANALYSIS")
    print("=" * 60)
    print(f"\n{'Layer':<8} {'Norm':<12} {'Mean|W|':<12} {'Max|W|':<12} {'Var':<12}")
    print("-" * 56)

    for i in range(num_layers):
        analysis["per_layer"][f"layer_{i:02d}"] = {
            "frobenius_norm": float(layer_norms[i]),
            "mean_abs": float(layer_mean_abs[i]),
            "max_abs": float(layer_max_abs[i]),
            "variance": float(layer_var[i]),
        }
        print(f"{i:<8} {layer_norms[i]:<12.4f} {layer_mean_abs[i]:<12.6f} {layer_max_abs[i]:<12.4f} {layer_var[i]:<12.6f}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Identify top layers
    sorted_by_norm = np.argsort(layer_norms)[::-1]
    print("\nTop 10 layers by Frobenius norm (highest weight):")
    for rank, idx in enumerate(sorted_by_norm[:10], 1):
        print(f"  {rank}. Layer {idx}: {layer_norms[idx]:.4f}")

    print("\nBottom 10 layers by Frobenius norm (lowest weight):")
    for rank, idx in enumerate(sorted_by_norm[-10:], 1):
        print(f"  {rank}. Layer {idx}: {layer_norms[idx]:.4f}")

    # Check for U-shaped pattern (early + late > middle)
    early_norm = layer_norms[:17].mean()
    middle_norm = layer_norms[17:33].mean()
    late_norm = layer_norms[33:].mean()

    print(f"\nLayer group averages:")
    print(f"  Early (0-16):  {early_norm:.4f}")
    print(f"  Middle (17-32): {middle_norm:.4f}")
    print(f"  Late (33-48):  {late_norm:.4f}")

    analysis["summary"] = {
        "top_10_layers": sorted_by_norm[:10].tolist(),
        "bottom_10_layers": sorted_by_norm[-10:].tolist(),
        "early_mean_norm": float(early_norm),
        "middle_mean_norm": float(middle_norm),
        "late_mean_norm": float(late_norm),
    }

    # Check hypothesis: U-shaped curve (high for early + late, low for middle)
    is_u_shaped = (early_norm > middle_norm) and (late_norm > middle_norm)
    analysis["summary"]["is_u_shaped"] = bool(is_u_shaped)  # Convert numpy bool to Python bool
    print(f"\nU-shaped pattern detected: {is_u_shaped}")

    # Create visualizations
    print("\nGenerating visualizations...")

    # Plot 1: Bar chart of layer norms
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    ax1.bar(range(num_layers), layer_norms)
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Frobenius Norm")
    ax1.set_title("Per-Layer Weight Magnitude (Frobenius Norm)")
    ax1.axvline(x=16.5, color="r", linestyle="--", alpha=0.5, label="Early/Middle")
    ax1.axvline(x=32.5, color="r", linestyle="--", alpha=0.5, label="Middle/Late")
    ax1.legend()

    # Plot 2: Heatmap of weight distribution
    ax2 = axes[0, 1]
    # Reduce dimensions for visualization: average over hidden_dim
    W_viz = W_reshaped.mean(dim=2).numpy()  # [out_dim, num_layers]
    im = ax2.imshow(W_viz, aspect="auto", cmap="RdBu_r")
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("Output Dimension")
    ax2.set_title("Weight Distribution Heatmap (mean over hidden)")
    plt.colorbar(im, ax=ax2)

    # Plot 3: Layer group comparison
    ax3 = axes[1, 0]
    groups = ["Early\n(0-16)", "Middle\n(17-32)", "Late\n(33-48)"]
    group_norms = [early_norm, middle_norm, late_norm]
    bars = ax3.bar(groups, group_norms, color=["#4CAF50", "#2196F3", "#FF9800"])
    ax3.set_ylabel("Mean Frobenius Norm")
    ax3.set_title("Layer Group Weight Comparison")
    for bar, val in zip(bars, group_norms):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f"{val:.4f}", ha="center", va="bottom")

    # Plot 4: Cumulative contribution
    ax4 = axes[1, 1]
    sorted_norms = np.sort(layer_norms)[::-1]
    cumulative = np.cumsum(sorted_norms) / np.sum(sorted_norms) * 100
    ax4.plot(range(1, num_layers + 1), cumulative, "b-", linewidth=2)
    ax4.axhline(y=80, color="r", linestyle="--", alpha=0.5, label="80% threshold")
    ax4.axhline(y=90, color="orange", linestyle="--", alpha=0.5, label="90% threshold")
    ax4.set_xlabel("Number of Top Layers")
    ax4.set_ylabel("Cumulative Contribution (%)")
    ax4.set_title("Cumulative Weight Contribution")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Find how many layers needed for 80%, 90%
    layers_for_80 = np.argmax(cumulative >= 80) + 1
    layers_for_90 = np.argmax(cumulative >= 90) + 1
    ax4.axvline(x=layers_for_80, color="r", linestyle=":", alpha=0.5)
    ax4.axvline(x=layers_for_90, color="orange", linestyle=":", alpha=0.5)

    analysis["summary"]["layers_for_80pct"] = int(layers_for_80)
    analysis["summary"]["layers_for_90pct"] = int(layers_for_90)
    print(f"Layers needed for 80% contribution: {layers_for_80}")
    print(f"Layers needed for 90% contribution: {layers_for_90}")

    plt.tight_layout()
    plot_path = output_path / "projection_analysis.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to: {plot_path}")
    plt.close()

    # Save analysis
    analysis_path = output_path / "projection_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved analysis to: {analysis_path}")

    # Save raw norms for further analysis
    norms_path = output_path / "layer_norms.npy"
    np.save(norms_path, layer_norms)
    print(f"Saved norms to: {norms_path}")

    # Cleanup
    del pipe, W, W_reshaped

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Analyze LTX-2 projection matrix (zero inference cost)"
    )
    parser.add_argument(
        "--model-path",
        default="models/LTX-2",
        help="Path to LTX-2 model",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results/projection_analysis",
        help="Output directory for analysis",
    )
    args = parser.parse_args()

    analyze_projection_matrix(
        model_path=args.model_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
