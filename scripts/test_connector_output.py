"""
Test connector output structure vs caption_projection expectations.

Last Updated: 2026-01-20

Key finding: caption_projection crushes actual embeddings (gain 0.14) but amplifies random (gain 2.0).
This suggests embeddings lie in a low-gain subspace of the weight matrix.
"""

import torch
import json
from pathlib import Path
from safetensors.torch import load_file


def load_weights():
    """Load caption_projection weights."""
    ckpt_path = Path("models/LTX-2/transformer")
    index_file = ckpt_path / "diffusion_pytorch_model.safetensors.index.json"

    with open(index_file) as f:
        index = json.load(f)

    # Load caption_projection weights
    weight_map = index["weight_map"]
    weights = {}
    loaded_shards = set()

    for key in weight_map:
        if "caption_projection" in key and "audio" not in key:
            shard = weight_map[key]
            if shard not in loaded_shards:
                shard_path = ckpt_path / shard
                tensors = load_file(str(shard_path))
                weights.update(tensors)
                loaded_shards.add(shard)

    return {
        "linear_1_weight": weights["caption_projection.linear_1.weight"].to(torch.float32),
        "linear_1_bias": weights["caption_projection.linear_1.bias"].to(torch.float32),
        "linear_2_weight": weights["caption_projection.linear_2.weight"].to(torch.float32),
        "linear_2_bias": weights["caption_projection.linear_2.bias"].to(torch.float32),
    }


def analyze_forward_pass(weights, embeddings, name=""):
    """Trace forward pass through caption_projection."""
    print(f"\n=== Forward Pass: {name} ===")

    x = embeddings
    print(f"Input: shape={x.shape}, mean={x.mean():.4f}, std={x.std():.4f}")

    # Linear 1
    W1 = weights["linear_1_weight"]  # [4096, 3840]
    b1 = weights["linear_1_bias"]     # [4096]
    x1 = torch.nn.functional.linear(x, W1, b1)
    print(f"After linear_1: shape={x1.shape}, mean={x1.mean():.4f}, std={x1.std():.4f}")

    # GELU
    x2 = torch.nn.functional.gelu(x1, approximate="tanh")
    print(f"After GELU: shape={x2.shape}, mean={x2.mean():.4f}, std={x2.std():.4f}")

    # Linear 2
    W2 = weights["linear_2_weight"]  # [4096, 4096]
    b2 = weights["linear_2_bias"]     # [4096]
    x3 = torch.nn.functional.linear(x2, W2, b2)
    print(f"After linear_2: shape={x3.shape}, mean={x3.mean():.4f}, std={x3.std():.4f}")

    return x3


def test_different_inputs(weights):
    """Test caption_projection with different input distributions."""

    print("=" * 80)
    print("TESTING DIFFERENT INPUT DISTRIBUTIONS")
    print("=" * 80)

    # Input shape: [1, 256, 3840] (typical for LTX-2)
    B, T, D = 1, 256, 3840

    # Test 1: Standard normal
    x_normal = torch.randn(B, T, D)
    analyze_forward_pass(weights, x_normal, "Standard Normal (mean=0, std=1)")

    # Test 2: Unit norm per token (like RMSNorm output)
    x_unit = torch.randn(B, T, D)
    x_unit = x_unit / x_unit.norm(dim=-1, keepdim=True)  # Each token has unit norm
    analyze_forward_pass(weights, x_unit, "Unit norm per token")

    # Test 3: Simulated RMSNorm output (sqrt(D) * unit norm)
    x_rms = torch.randn(B, T, D)
    x_rms = x_rms / x_rms.pow(2).mean(dim=-1, keepdim=True).sqrt()  # RMSNorm
    analyze_forward_pass(weights, x_rms, "RMSNorm normalized (std ~1)")

    # Test 4: Very low variance
    x_low = torch.randn(B, T, D) * 0.1
    analyze_forward_pass(weights, x_low, "Low variance (std=0.1)")

    # Test 5: Very high variance
    x_high = torch.randn(B, T, D) * 10.0
    analyze_forward_pass(weights, x_high, "High variance (std=10)")

    # Test 6: Biased (mean != 0)
    x_biased = torch.randn(B, T, D) + 5.0
    analyze_forward_pass(weights, x_biased, "Biased (mean=5)")

    # Test 7: Sparse (many near-zero dimensions)
    x_sparse = torch.randn(B, T, D)
    x_sparse[:, :, ::2] = 0  # Zero out every other dimension
    analyze_forward_pass(weights, x_sparse, "Sparse (50% zeros)")


def analyze_weight_alignment():
    """Analyze how different input structures align with weight matrix."""
    weights = load_weights()
    W1 = weights["linear_1_weight"]

    print("\n" + "=" * 80)
    print("WEIGHT MATRIX ALIGNMENT ANALYSIS")
    print("=" * 80)

    # SVD of W1
    U, S, Vh = torch.linalg.svd(W1, full_matrices=False)
    print(f"\nW1 shape: {W1.shape}")
    print(f"Top 5 singular values: {S[:5].tolist()}")
    print(f"Bottom 5 singular values: {S[-5:].tolist()}")

    # For RMSNorm output: each token has fixed RMS (sqrt of mean squared value)
    # The key insight: if embeddings are constrained to have low projection
    # onto high-S directions, the gain will be low

    # Compute expected gain for unit-norm inputs in each right singular vector direction
    print("\nGain for input along each right singular vector:")
    gains = []
    for i in range(min(10, len(S))):
        # Input along i-th right singular vector (unit norm)
        v_i = Vh[i].unsqueeze(0).unsqueeze(0)  # [1, 1, 3840]
        out = torch.nn.functional.linear(v_i, W1, weights["linear_1_bias"].unsqueeze(0).unsqueeze(0))
        out_norm = out.norm().item()
        gains.append((i, S[i].item(), out_norm))
        if i < 5:
            print(f"  Direction {i}: singular value={S[i]:.4f}, output norm={out_norm:.4f}")

    # The key question: what is the average gain for a random unit vector?
    # This depends on how the input aligns with different singular directions

    # For random unit vector: expected gain = sqrt(sum(S^2) / D)
    expected_random = (S**2).sum().sqrt() / (W1.shape[0]**0.5)
    print(f"\nExpected gain for random unit vector: {expected_random.item():.4f}")


def test_dimension_specific_inputs(weights):
    """Test inputs that differ in per-dimension structure."""
    print("\n" + "=" * 80)
    print("PER-DIMENSION STRUCTURE ANALYSIS")
    print("=" * 80)

    B, T, D = 1, 256, 3840

    # Test: All tokens have same values (no inter-token variance)
    x_same = torch.randn(1, 1, D).expand(B, T, D)
    analyze_forward_pass(weights, x_same, "Same token repeated (no inter-token variance)")

    # Test: Different dimensions have different variances
    x_hetero = torch.randn(B, T, D)
    x_hetero[:, :, :D//2] *= 0.01  # First half low variance
    x_hetero[:, :, D//2:] *= 10.0  # Second half high variance
    # Renormalize to std=1
    x_hetero = x_hetero / x_hetero.std()
    analyze_forward_pass(weights, x_hetero, "Heterogeneous per-dim variance (std=1 overall)")

    # Test: Per-dim mean offset
    x_dimshift = torch.randn(B, T, D)
    x_dimshift = x_dimshift + torch.randn(1, 1, D) * 0.5  # Add random per-dim offset
    x_dimshift = x_dimshift - x_dimshift.mean()  # Re-center
    x_dimshift = x_dimshift / x_dimshift.std()  # Re-scale
    analyze_forward_pass(weights, x_dimshift, "Per-dim mean shifts (std=1)")

    # Test: Very few active dimensions (sparse in dimension space)
    x_dim_sparse = torch.zeros(B, T, D)
    active_dims = torch.randperm(D)[:100]  # Only 100 active dimensions
    x_dim_sparse[:, :, active_dims] = torch.randn(B, T, 100) * (D / 100) ** 0.5
    analyze_forward_pass(weights, x_dim_sparse, "Sparse dims (100 of 3840 active)")


def main():
    print("Loading weights...")
    weights = load_weights()

    analyze_weight_alignment()
    test_different_inputs(weights)
    test_dimension_specific_inputs(weights)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
