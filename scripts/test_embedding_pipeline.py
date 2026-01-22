"""
Test actual embedding pipeline to diagnose variance crush.

Last Updated: 2026-01-20

This traces embeddings from Gemma through connector to caption_projection
with detailed per-stage statistics.
"""

import torch
import json
from pathlib import Path
from safetensors.torch import load_file


def load_caption_projection_weights():
    """Load caption_projection weights."""
    ckpt_path = Path("models/LTX-2/transformer")
    index_file = ckpt_path / "diffusion_pytorch_model.safetensors.index.json"

    with open(index_file) as f:
        index = json.load(f)

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
        "linear_1_weight": weights["caption_projection.linear_1.weight"],
        "linear_1_bias": weights["caption_projection.linear_1.bias"],
        "linear_2_weight": weights["caption_projection.linear_2.weight"],
        "linear_2_bias": weights["caption_projection.linear_2.bias"],
    }


def detailed_projection_analysis(weights, x, name=""):
    """Detailed forward pass analysis."""
    x = x.to(torch.float32).cpu()  # Move to CPU for analysis
    W1 = weights["linear_1_weight"].to(torch.float32).cpu()
    b1 = weights["linear_1_bias"].to(torch.float32).cpu()
    W2 = weights["linear_2_weight"].to(torch.float32).cpu()
    b2 = weights["linear_2_bias"].to(torch.float32).cpu()

    print(f"\n=== {name} ===")
    print(f"Input: shape={x.shape}, mean={x.mean():.6f}, std={x.std():.6f}")
    print(f"  per-dim std range: [{x.std(dim=(0,1)).min():.6f}, {x.std(dim=(0,1)).max():.6f}]")
    print(f"  per-dim mean range: [{x.mean(dim=(0,1)).min():.6f}, {x.mean(dim=(0,1)).max():.6f}]")

    # Check alignment with weight matrix
    # Compute x @ W1^T to see which directions are activated
    U, S, Vh = torch.linalg.svd(W1, full_matrices=False)

    # Project x onto right singular vectors
    x_flat = x.reshape(-1, x.shape[-1])  # [B*T, 3840]
    projections = x_flat @ Vh.T  # [B*T, min(3840, 4096)]

    # Variance explained by each singular direction
    proj_var = projections.var(dim=0)  # [min_dim]
    print(f"\n  Projection onto W1 singular vectors:")
    print(f"    Top 5 singular directions variance: {proj_var[:5].tolist()}")
    print(f"    Top singular value: {S[0]:.4f}, bottom: {S[-1]:.6f}")

    # Weighted importance: variance * (singular_value^2)
    # High singular values should have high variance for good signal propagation
    importance = proj_var[:len(S)] * (S ** 2)
    print(f"    Importance-weighted top 5: {importance[:5].tolist()}")

    # Linear 1
    x1 = torch.nn.functional.linear(x, W1, b1)
    print(f"\nAfter linear_1: mean={x1.mean():.6f}, std={x1.std():.6f}")

    # GELU
    x2 = torch.nn.functional.gelu(x1, approximate="tanh")
    print(f"After GELU: mean={x2.mean():.6f}, std={x2.std():.6f}")

    # Linear 2
    x3 = torch.nn.functional.linear(x2, W2, b2)
    print(f"After linear_2: mean={x3.mean():.6f}, std={x3.std():.6f}")

    return x3


def main():
    print("=" * 80)
    print("EMBEDDING PIPELINE DEBUG")
    print("=" * 80)

    # Load encoder with connector
    print("\nLoading Gemma3Encoder with connector...")
    from llm_dit.encoders.gemma3 import Gemma3Encoder

    encoder = Gemma3Encoder(
        model_id="models/LTX-2/text_encoder",
        device="cuda",
        dtype=torch.bfloat16,
        max_sequence_length=256,
        connectors_path="models/LTX-2/connectors/diffusion_pytorch_model.safetensors",
        use_connector=False,  # Disable connector to see raw feature extractor output
        load_in_8bit=True,  # Use 8-bit for memory efficiency
    )

    # Encode test prompt
    print("\nEncoding prompt (WITHOUT CONNECTOR)...")
    prompt = "A fluffy orange cat walking through a sunny garden"
    output = encoder.encode(prompt)

    print("\n--- Raw Feature Extractor Output (no connector) ---")
    embeddings_raw = output.embeddings[0]
    print(f"Shape: {embeddings_raw.shape}")
    print(f"Mean: {embeddings_raw.mean():.6f}")
    print(f"Std: {embeddings_raw.std():.6f}")
    embeddings_raw_f = embeddings_raw.to(torch.float32)
    per_dim_mean_raw = embeddings_raw_f.mean(dim=0)
    per_dim_std_raw = embeddings_raw_f.std(dim=0)
    print(f"Per-dim mean range: [{per_dim_mean_raw.min():.4f}, {per_dim_mean_raw.max():.4f}]")
    print(f"Per-dim std range: [{per_dim_std_raw.min():.4f}, {per_dim_std_raw.max():.4f}]")

    del encoder
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Now load WITH connector
    print("\n\nLoading Gemma3Encoder WITH connector...")
    encoder = Gemma3Encoder(
        model_id="models/LTX-2/text_encoder",
        device="cuda",
        dtype=torch.bfloat16,
        max_sequence_length=256,
        connectors_path="models/LTX-2/connectors/diffusion_pytorch_model.safetensors",
        use_connector=True,
        load_in_8bit=True,
    )

    print("\nEncoding prompt (WITH CONNECTOR)...")
    output = encoder.encode(prompt)

    embeddings = output.embeddings[0]  # [seq_len, 3840]
    print(f"\n--- Encoded Embeddings ---")
    print(f"Shape: {embeddings.shape}")
    print(f"Dtype: {embeddings.dtype}")
    print(f"Mean: {embeddings.mean():.6f}")
    print(f"Std: {embeddings.std():.6f}")
    print(f"Range: [{embeddings.min():.6f}, {embeddings.max():.6f}]")

    # Per-dimension analysis
    embeddings_f = embeddings.to(torch.float32)
    per_dim_std = embeddings_f.std(dim=0)  # [3840]
    per_dim_mean = embeddings_f.mean(dim=0)  # [3840]
    print(f"\nPer-dimension statistics:")
    print(f"  std range: [{per_dim_std.min():.6f}, {per_dim_std.max():.6f}]")
    print(f"  mean range: [{per_dim_mean.min():.6f}, {per_dim_mean.max():.6f}]")
    print(f"  Dims with std < 0.1: {(per_dim_std < 0.1).sum().item()}")
    print(f"  Dims with std < 0.5: {(per_dim_std < 0.5).sum().item()}")

    # Load caption_projection weights
    print("\nLoading caption_projection weights...")
    weights = load_caption_projection_weights()

    # Analyze actual embeddings through projection
    embeddings_batched = embeddings.unsqueeze(0)  # [1, seq_len, 3840]
    detailed_projection_analysis(weights, embeddings_batched, "Actual Gemma Embeddings (raw)")

    # Test with centering fix
    embeddings_centered = embeddings_batched - embeddings_batched.mean(dim=1, keepdim=True)
    detailed_projection_analysis(weights, embeddings_centered, "Actual Gemma Embeddings (CENTERED)")

    # Compare to random baseline
    random_input = torch.randn_like(embeddings_batched)
    detailed_projection_analysis(weights, random_input, "Random Baseline (same shape)")

    # Test: Match the variance structure of actual embeddings but random values
    matched_input = torch.randn_like(embeddings_batched)
    # Scale each dimension to match actual per-dim std
    matched_input = matched_input * per_dim_std.unsqueeze(0).unsqueeze(0)
    matched_input = matched_input + per_dim_mean.unsqueeze(0).unsqueeze(0)
    detailed_projection_analysis(weights, matched_input, "Per-dim matched random")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
