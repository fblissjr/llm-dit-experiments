"""
Check learnable registers statistics.

Last Updated: 2026-01-20
"""

import torch
from safetensors import safe_open
from pathlib import Path


def main():
    print("=" * 80)
    print("LEARNABLE REGISTERS ANALYSIS")
    print("=" * 80)

    connectors_path = Path("models/LTX-2/connectors/diffusion_pytorch_model.safetensors")

    with safe_open(connectors_path, framework="pt") as f:
        keys = [k for k in f.keys() if "video_connector" in k]
        print(f"\nVideo connector keys: {keys[:5]}...")

        # Load learnable registers
        if "video_connector.learnable_registers" in f.keys():
            registers = f.get_tensor("video_connector.learnable_registers")
            print(f"\n--- Learnable Registers ---")
            print(f"Shape: {registers.shape}")  # Should be [128, 3840]
            print(f"Dtype: {registers.dtype}")

            reg_f = registers.to(torch.float32)
            print(f"\nOverall statistics:")
            print(f"  Mean: {reg_f.mean():.6f}")
            print(f"  Std: {reg_f.std():.6f}")
            print(f"  Range: [{reg_f.min():.6f}, {reg_f.max():.6f}]")

            # Per-token statistics (each register is one token)
            token_means = reg_f.mean(dim=1)  # [128]
            token_stds = reg_f.std(dim=1)   # [128]
            print(f"\nPer-token statistics:")
            print(f"  Token means range: [{token_means.min():.4f}, {token_means.max():.4f}]")
            print(f"  Token stds range: [{token_stds.min():.4f}, {token_stds.max():.4f}]")

            # Per-dimension statistics (each dimension across all 128 registers)
            dim_means = reg_f.mean(dim=0)  # [3840]
            dim_stds = reg_f.std(dim=0)    # [3840]
            print(f"\nPer-dimension statistics (across 128 registers):")
            print(f"  Dim means range: [{dim_means.min():.4f}, {dim_means.max():.4f}]")
            print(f"  Dim stds range: [{dim_stds.min():.4f}, {dim_stds.max():.4f}]")

            # Check how many dimensions have large absolute means
            large_mean_dims = (dim_means.abs() > 1.0).sum().item()
            print(f"\n  Dims with |mean| > 1.0: {large_mean_dims} out of 3840")

            very_large_mean_dims = (dim_means.abs() > 5.0).sum().item()
            print(f"  Dims with |mean| > 5.0: {very_large_mean_dims} out of 3840")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
