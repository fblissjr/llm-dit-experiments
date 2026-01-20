"""
Test cross-attention vs self-attention weight statistics.

Last Updated: 2026-01-20

Checks if FP8 quantization is crushing cross-attention weights.
"""

import torch
from llm_dit.models.ltx2 import load_ltx2_transformer, load_ltx2_transformer_fp8_native


def main():
    print("=" * 80)
    print("CROSS-ATTENTION WEIGHT ANALYSIS")
    print("=" * 80)

    # Load model with FP8 quantization (as used in generation)
    print("\nLoading transformer with FP8 quantization...")
    model_fp8 = load_ltx2_transformer_fp8_native(
        "models/LTX-2/transformer",
        dtype=torch.bfloat16,
        device="cpu",
        video_only=True,
    )

    print("\n--- Block 0 Weight Statistics ---")
    block = model_fp8.transformer_blocks[0]

    # Self-attention weights (attn1 uses fused qkv)
    attn1_to_q = block.attn1.to_q.weight
    print(f"Self-attn (attn1) to_q:")
    print(f"  dtype: {attn1_to_q.dtype}")
    print(f"  shape: {attn1_to_q.shape}")
    # Upcast FP8 for stats
    attn1_to_q_f = attn1_to_q.to(torch.float32)
    print(f"  mean: {attn1_to_q_f.mean():.6f}, std: {attn1_to_q_f.std():.6f}")
    print(f"  range: [{attn1_to_q_f.min():.6f}, {attn1_to_q_f.max():.6f}]")

    # Cross-attention weights
    attn2 = block.attn2
    print(f"\nCross-attn (attn2) to_q:")
    to_q = attn2.to_q.weight
    print(f"  dtype: {to_q.dtype}")
    to_q_f = to_q.to(torch.float32)
    print(f"  mean: {to_q_f.mean():.6f}, std: {to_q_f.std():.6f}")
    print(f"  range: [{to_q_f.min():.6f}, {to_q_f.max():.6f}]")

    print(f"\nCross-attn (attn2) to_k:")
    to_k = attn2.to_k.weight
    print(f"  dtype: {to_k.dtype}")
    to_k_f = to_k.to(torch.float32)
    print(f"  mean: {to_k_f.mean():.6f}, std: {to_k_f.std():.6f}")
    print(f"  range: [{to_k_f.min():.6f}, {to_k_f.max():.6f}]")

    print(f"\nCross-attn (attn2) to_v:")
    to_v = attn2.to_v.weight
    print(f"  dtype: {to_v.dtype}")
    to_v_f = to_v.to(torch.float32)
    print(f"  mean: {to_v_f.mean():.6f}, std: {to_v_f.std():.6f}")
    print(f"  range: [{to_v_f.min():.6f}, {to_v_f.max():.6f}]")

    # Caption projection (should NOT be FP8)
    print(f"\n--- Caption Projection ---")
    cap_proj = model_fp8.caption_projection.linear_1.weight
    print(f"caption_projection.linear_1:")
    print(f"  dtype: {cap_proj.dtype}")
    cap_proj_f = cap_proj.to(torch.float32)
    print(f"  mean: {cap_proj_f.mean():.6f}, std: {cap_proj_f.std():.6f}")
    print(f"  range: [{cap_proj_f.min():.6f}, {cap_proj_f.max():.6f}]")

    # Compare to non-quantized model
    print("\n" + "=" * 80)
    print("LOADING NON-QUANTIZED MODEL FOR COMPARISON...")
    print("=" * 80)

    model_bf16 = load_ltx2_transformer(
        "models/LTX-2/transformer",
        dtype=torch.bfloat16,
        device="cpu",
        video_only=True,
    )

    block_bf16 = model_bf16.transformer_blocks[0]

    print("\n--- Non-Quantized Block 0 Weight Statistics ---")
    attn2_bf16 = block_bf16.attn2
    print(f"\nCross-attn (attn2) to_k (bf16):")
    to_k_bf16 = attn2_bf16.to_k.weight
    print(f"  dtype: {to_k_bf16.dtype}")
    to_k_bf16_f = to_k_bf16.to(torch.float32)
    print(f"  mean: {to_k_bf16_f.mean():.6f}, std: {to_k_bf16_f.std():.6f}")
    print(f"  range: [{to_k_bf16_f.min():.6f}, {to_k_bf16_f.max():.6f}]")

    print(f"\nCross-attn (attn2) to_v (bf16):")
    to_v_bf16 = attn2_bf16.to_v.weight
    print(f"  dtype: {to_v_bf16.dtype}")
    to_v_bf16_f = to_v_bf16.to(torch.float32)
    print(f"  mean: {to_v_bf16_f.mean():.6f}, std: {to_v_bf16_f.std():.6f}")
    print(f"  range: [{to_v_bf16_f.min():.6f}, {to_v_bf16_f.max():.6f}]")

    # Compute quantization error
    print("\n--- Quantization Error ---")
    to_k_error = (to_k_f - to_k_bf16_f).abs()
    print(f"to_k FP8 vs bf16 error: mean={to_k_error.mean():.6f}, max={to_k_error.max():.6f}")
    to_v_error = (to_v_f - to_v_bf16_f).abs()
    print(f"to_v FP8 vs bf16 error: mean={to_v_error.mean():.6f}, max={to_v_error.max():.6f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
