"""
Verify feature extractor weight loading and output.

Last Updated: 2026-01-20

Compares our feature extractor output with expected distribution.
"""

import torch
import json
from pathlib import Path
from safetensors import safe_open


def main():
    print("=" * 80)
    print("FEATURE EXTRACTOR VERIFICATION")
    print("=" * 80)

    # Load connector weights directly
    connectors_path = Path("models/LTX-2/connectors/diffusion_pytorch_model.safetensors")

    print(f"\nLoading: {connectors_path}")
    with safe_open(connectors_path, framework="pt") as f:
        keys = list(f.keys())
        print(f"Keys in checkpoint: {keys[:10]}...")

        # Load text_proj_in (feature extractor)
        if "text_proj_in.weight" in keys:
            fe_weight = f.get_tensor("text_proj_in.weight")
            print(f"\nFeature extractor (text_proj_in.weight):")
            print(f"  Shape: {fe_weight.shape}")  # Should be [3840, 188160]
            print(f"  Dtype: {fe_weight.dtype}")
            fe_f = fe_weight.to(torch.float32)
            print(f"  Mean: {fe_f.mean():.6f}")
            print(f"  Std: {fe_f.std():.6f}")
            print(f"  Range: [{fe_f.min():.6f}, {fe_f.max():.6f}]")

            # Per-row statistics (each row corresponds to one output dimension)
            row_norms = fe_f.norm(dim=1)
            print(f"\n  Per-output-dim norms:")
            print(f"    Mean: {row_norms.mean():.4f}")
            print(f"    Std: {row_norms.std():.4f}")
            print(f"    Range: [{row_norms.min():.4f}, {row_norms.max():.4f}]")

            # Per-column statistics (each column corresponds to one input dimension)
            col_norms = fe_f.norm(dim=0)
            print(f"\n  Per-input-dim norms:")
            print(f"    Mean: {col_norms.mean():.4f}")
            print(f"    Std: {col_norms.std():.4f}")
            print(f"    Range: [{col_norms.min():.4f}, {col_norms.max():.4f}]")

    # Now test what happens when we apply this to normalized hidden states
    print("\n" + "=" * 80)
    print("SIMULATED FEATURE EXTRACTION")
    print("=" * 80)

    # Simulate normalized hidden states (8 * (x - mean) / range)
    # After normalization, values should be in roughly [-8, 8] range
    batch_size = 1
    seq_len = 256
    feature_dim = 188160  # 3840 * 49

    # Simulate normalized input (centered, scaled by 8)
    x_normalized = torch.randn(batch_size, seq_len, feature_dim) * 2  # std ~2 is typical after 8x scaling
    print(f"\nSimulated normalized input:")
    print(f"  Shape: {x_normalized.shape}")
    print(f"  Mean: {x_normalized.mean():.4f}")
    print(f"  Std: {x_normalized.std():.4f}")

    # Apply feature extractor
    W = fe_weight.to(torch.float32)
    output = torch.nn.functional.linear(x_normalized, W)
    print(f"\nAfter feature extractor:")
    print(f"  Shape: {output.shape}")
    print(f"  Mean: {output.mean():.4f}")
    print(f"  Std: {output.std():.4f}")
    print(f"  Range: [{output.min():.4f}, {output.max():.4f}]")

    # Per-dimension statistics of output
    per_dim_mean = output.mean(dim=(0, 1))
    per_dim_std = output.std(dim=(0, 1))
    print(f"\n  Per-dim mean range: [{per_dim_mean.min():.4f}, {per_dim_mean.max():.4f}]")
    print(f"  Per-dim std range: [{per_dim_std.min():.4f}, {per_dim_std.max():.4f}]")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
