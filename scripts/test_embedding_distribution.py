"""
Test embedding distribution to understand caption_projection behavior.

Last Updated: 2026-01-20

FINDING: caption_projection crushes actual embeddings (std 1.0 → 0.14)
but amplifies random input (std 1.0 → 2.7). This suggests actual embeddings
lie in a low-gain subspace of the weight matrix.
"""

import torch
import numpy as np
from pathlib import Path


def analyze_caption_projection_directly():
    """Analyze caption_projection weights to understand gain behavior."""
    from safetensors.torch import load_file
    import json

    print("=" * 80)
    print("CAPTION PROJECTION WEIGHT ANALYSIS")
    print("=" * 80)

    # Load transformer checkpoint to get caption_projection weights
    ckpt_path = Path("models/LTX-2/transformer")
    index_file = ckpt_path / "diffusion_pytorch_model.safetensors.index.json"

    with open(index_file) as f:
        index = json.load(f)

    # Find caption_projection weights
    weight_map = index["weight_map"]
    caption_proj_keys = [k for k in weight_map.keys() if "caption_projection" in k]
    print(f"\nCaption projection keys: {caption_proj_keys}")

    # Load the shard containing caption_projection
    shard_files = set(weight_map[k] for k in caption_proj_keys)
    weights = {}
    for shard_file in shard_files:
        shard_path = ckpt_path / shard_file
        tensors = load_file(str(shard_path))
        for k, v in tensors.items():
            if "caption_projection" in k:
                weights[k] = v

    print("\nCaption projection weights loaded:")
    for k, v in weights.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    # Analyze the linear_1 weight (first projection)
    linear1_weight = weights.get("caption_projection.linear_1.weight")
    if linear1_weight is None:
        print("WARNING: caption_projection.linear_1.weight not found!")
        return

    W = linear1_weight.to(torch.float32)
    print(f"\n--- Linear 1 Weight Analysis ---")
    print(f"Shape: {W.shape}")  # [out_features, in_features]
    print(f"Frobenius norm: {W.norm():.4f}")
    print(f"Mean: {W.mean():.6f}")
    print(f"Std: {W.std():.6f}")
    print(f"Max singular value (approx): {W.norm(dim=1).max():.4f}")

    # SVD analysis - find principal components
    print(f"\n--- SVD Analysis ---")
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    print(f"Top 10 singular values: {S[:10].tolist()}")
    print(f"Singular value decay (S[10]/S[0]): {S[10]/S[0]:.4f}")
    print(f"Effective rank (sum(S)^2 / sum(S^2)): {(S.sum()**2 / (S**2).sum()):.1f}")

    # The key question: what's the gain for different input directions?
    # For random Gaussian input: expected output std = sqrt(sum(S^2)/out_dim)
    expected_gain_random = torch.sqrt((S**2).sum() / W.shape[0]).item()
    print(f"\nExpected gain for isotropic random input: {expected_gain_random:.4f}")

    return W, S, Vh


def analyze_gemma_output_distribution():
    """
    Load Gemma and analyze what distribution the text encoder actually outputs.
    This requires loading the full encoder pipeline.
    """
    from llm_dit.encoders.gemma3 import Gemma3Encoder

    print("\n" + "=" * 80)
    print("GEMMA OUTPUT DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # Load encoder with connector
    print("\nLoading Gemma3Encoder...")
    encoder = Gemma3Encoder(
        model_id="models/LTX-2/text_encoder",  # Local path
        device="cuda",
        dtype=torch.bfloat16,
        max_sequence_length=256,
        connectors_path="models/LTX-2/connectors",
        use_connector=True,
    )

    # Test prompts
    test_prompts = [
        "A fluffy orange cat walking through a sunny garden",
        "cinematic video of a person walking down a busy street",
        "abstract colorful patterns flowing across the screen",
    ]

    print("\n--- Encoding Test Prompts ---")
    for prompt in test_prompts:
        with torch.no_grad():
            output = encoder.encode(prompt)

        embeddings = output.embeddings[0]  # Get first batch element
        print(f"\nPrompt: '{prompt[:50]}...'")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Mean: {embeddings.mean():.4f}")
        print(f"  Std: {embeddings.std():.4f}")
        print(f"  Range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")

        # Per-dimension statistics
        per_dim_mean = embeddings.mean(dim=0)  # [dim]
        per_dim_std = embeddings.std(dim=0)
        print(f"  Per-dim mean range: [{per_dim_mean.min():.4f}, {per_dim_mean.max():.4f}]")
        print(f"  Per-dim std range: [{per_dim_std.min():.4f}, {per_dim_std.max():.4f}]")

    return encoder


def analyze_projection_subspace(W, encoder=None):
    """
    Analyze if Gemma outputs lie in a specific subspace of the projection input space.
    """
    if encoder is None:
        return

    print("\n" + "=" * 80)
    print("SUBSPACE ANALYSIS")
    print("=" * 80)

    # Get embeddings from encoder
    with torch.no_grad():
        output = encoder.encode("A fluffy orange cat walking through a sunny garden")

    # Flatten embeddings to vectors
    x = output.embeddings[0].to(torch.float32)  # [T, dim]
    print(f"Embedding shape: {x.shape}")

    # Compute the projection y = W @ x.T
    # W is [out_features, in_features], x is [T, in_features]
    y = (W @ x.T).T  # [T, out_features]

    print(f"\n--- Direct Projection Analysis ---")
    print(f"Input x: mean={x.mean():.4f}, std={x.std():.4f}")
    print(f"Output y: mean={y.mean():.4f}, std={y.std():.4f}")
    print(f"Gain (output_std / input_std): {y.std() / x.std():.4f}")

    # Compare to random input
    x_random = torch.randn_like(x)
    y_random = (W @ x_random.T).T
    print(f"\nRandom input: mean={x_random.mean():.4f}, std={x_random.std():.4f}")
    print(f"Random output: mean={y_random.mean():.4f}, std={y_random.std():.4f}")
    print(f"Random gain: {y_random.std() / x_random.std():.4f}")

    # PCA of actual embeddings
    print(f"\n--- Embedding PCA ---")
    x_centered = x - x.mean(dim=0, keepdim=True)
    _, S_x, Vh_x = torch.linalg.svd(x_centered, full_matrices=False)
    print(f"Embedding singular values (top 10): {S_x[:10].tolist()}")
    print(f"Embedding effective rank: {(S_x.sum()**2 / (S_x**2).sum()):.1f}")

    # Key question: how does the embedding subspace overlap with high-gain directions of W?
    # Get top principal components of embeddings (rows of Vh_x)
    # Get high-gain directions of W (top right singular vectors)
    _, S_W, Vh_W = torch.linalg.svd(W, full_matrices=False)

    # Compute overlap: for each embedding PC, what's its projection onto W's top singular vectors?
    n_compare = min(20, S_x.shape[0], Vh_W.shape[0])
    overlap_matrix = torch.abs(Vh_x[:n_compare] @ Vh_W[:n_compare].T)

    print(f"\n--- Subspace Overlap (Embedding PCs vs W's top singular vectors) ---")
    print(f"Max overlap values (should be high if they align):")
    print(f"  Embedding PC 0 with W SVs: {overlap_matrix[0].max():.4f}")
    print(f"  Embedding PC 1 with W SVs: {overlap_matrix[1].max():.4f}")
    print(f"  Embedding PC 2 with W SVs: {overlap_matrix[2].max():.4f}")

    # Average overlap - if embeddings align with W's high-gain directions, this should be high
    avg_overlap = overlap_matrix.mean().item()
    print(f"\n  Average overlap: {avg_overlap:.4f}")
    print(f"  (Random expectation: ~{1/np.sqrt(n_compare):.4f})")


def main():
    # Step 1: Analyze projection weights directly
    result = analyze_caption_projection_directly()
    if result is None:
        return
    W, S, Vh = result

    # Step 2: Analyze Gemma output distribution (requires GPU)
    try:
        encoder = analyze_gemma_output_distribution()
    except Exception as e:
        print(f"\nCould not load encoder: {e}")
        encoder = None

    # Step 3: Subspace analysis
    if encoder is not None:
        analyze_projection_subspace(W, encoder)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
