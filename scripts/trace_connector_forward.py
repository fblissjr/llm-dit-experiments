"""
Trace through connector forward pass to find where per-dim offsets are created.

Last Updated: 2026-01-20

We know:
- Feature extractor output has small per-dim offsets (0.52 range)
- Connector output has huge per-dim offsets (39.7 range)
- Weights are loaded correctly

This script traces through each step of the connector to pinpoint
where the offsets are created.
"""

import torch
from safetensors import safe_open
from pathlib import Path


def rms_norm(x, weight=None, eps=1e-6):
    """Our RMSNorm implementation."""
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)


def analyze(x, name):
    """Print per-dim statistics."""
    x = x.float()
    per_dim_mean = x.mean(dim=(0, 1)) if x.ndim == 3 else x.mean(dim=0)
    per_dim_std = x.std(dim=(0, 1)) if x.ndim == 3 else x.std(dim=0)
    mean_range = (per_dim_mean.max() - per_dim_mean.min()).item()
    print(f"{name}:")
    print(f"  Mean: {x.mean():.4f}, Std: {x.std():.4f}")
    print(f"  Per-dim mean range: [{per_dim_mean.min():.4f}, {per_dim_mean.max():.4f}] ({mean_range:.4f})")
    return mean_range


def main():
    print("=" * 80)
    print("CONNECTOR FORWARD PASS TRACING")
    print("=" * 80)

    # Load connector weights
    connectors_path = Path("models/LTX-2/connectors/diffusion_pytorch_model.safetensors")
    weights = {}
    with safe_open(connectors_path, framework="pt") as f:
        for k in f.keys():
            if "video_connector" in k:
                # Remove prefix
                new_key = k.replace("video_connector.", "")
                weights[new_key] = f.get_tensor(k)

    # Create synthetic input that matches feature extractor output statistics
    # Pre-connector: per-dim mean range ~0.5, overall std ~0.16
    B, T, D = 1, 256, 3840
    torch.manual_seed(42)

    # Start with centered random data
    x = torch.randn(B, T, D) * 0.16  # Match observed std
    # Add small per-dim offsets
    per_dim_offset = torch.randn(D) * 0.13  # Mean range ~0.5
    x = x + per_dim_offset

    print("\nInput (synthetic feature extractor output):")
    input_range = analyze(x, "  Input")

    # Get learnable registers
    registers = weights["learnable_registers"].float()
    print(f"\nLearnable registers: shape={registers.shape}")
    analyze(registers.unsqueeze(0), "  Registers")

    # Simulate _replace_padded_with_learnable_registers
    # Put 10 text tokens at the end, 246 registers at the start
    num_text = 10
    num_registers = 246
    num_reg_duplications = num_registers // 128  # 1
    extra = num_registers % 128  # 118

    # Tile registers to fill padding positions
    tiled_registers = torch.cat([
        registers.repeat(num_reg_duplications, 1),
        registers[:extra] if extra > 0 else torch.zeros(0, D)
    ], dim=0)  # [246, 3840]

    # Text tokens at the end (simulate left-padding then flip)
    text_tokens = x[0, -num_text:, :]  # Last 10 tokens as "valid"
    combined = torch.cat([tiled_registers, text_tokens], dim=0)  # [256, 3840]
    x = combined.unsqueeze(0)  # [1, 256, 3840]

    print("\nAfter register insertion:")
    after_reg_range = analyze(x, "  Combined (registers + text)")

    # Now trace through transformer blocks
    # Block 0
    print("\n" + "=" * 40)
    print("TRANSFORMER BLOCK 0")
    print("=" * 40)

    # Pre-norm before attention
    x_norm = rms_norm(x)
    print("\nAfter RMSNorm (pre-attention):")
    analyze(x_norm, "  Normalized")

    # Attention (simplified - just checking projection outputs)
    W_q = weights["transformer_blocks.0.attn1.to_q.weight"].float()
    b_q = weights["transformer_blocks.0.attn1.to_q.bias"].float()
    W_k = weights["transformer_blocks.0.attn1.to_k.weight"].float()
    b_k = weights["transformer_blocks.0.attn1.to_k.bias"].float()
    W_v = weights["transformer_blocks.0.attn1.to_v.weight"].float()
    b_v = weights["transformer_blocks.0.attn1.to_v.bias"].float()

    # QKV projections
    q = torch.nn.functional.linear(x_norm, W_q, b_q)
    k = torch.nn.functional.linear(x_norm, W_k, b_k)
    v = torch.nn.functional.linear(x_norm, W_v, b_v)

    print("\nQKV projections (before norm):")
    analyze(q, "  Q")
    analyze(k, "  K")
    analyze(v, "  V")

    # QK normalization
    norm_q_w = weights["transformer_blocks.0.attn1.norm_q.weight"].float()
    norm_k_w = weights["transformer_blocks.0.attn1.norm_k.weight"].float()

    q_normed = rms_norm(q, norm_q_w)
    k_normed = rms_norm(k, norm_k_w)

    print("\nAfter QK normalization:")
    analyze(q_normed, "  Q (normed)")
    analyze(k_normed, "  K (normed)")

    # Skip actual attention computation for simplicity
    # Just check the projection biases

    # Output projection
    W_out = weights["transformer_blocks.0.attn1.to_out.0.weight"].float()
    b_out = weights["transformer_blocks.0.attn1.to_out.0.bias"].float()

    print("\nOutput projection bias statistics:")
    print(f"  to_out bias: mean={b_out.mean():.4f}, range=[{b_out.min():.4f}, {b_out.max():.4f}]")

    # FFN
    W_ff1 = weights["transformer_blocks.0.ff.net.0.proj.weight"].float()
    b_ff1 = weights["transformer_blocks.0.ff.net.0.proj.bias"].float()
    W_ff2 = weights["transformer_blocks.0.ff.net.2.weight"].float()
    b_ff2 = weights["transformer_blocks.0.ff.net.2.bias"].float()

    print("\nFFN bias statistics:")
    print(f"  ff.net.0 bias: mean={b_ff1.mean():.4f}, range=[{b_ff1.min():.4f}, {b_ff1.max():.4f}]")
    print(f"  ff.net.2 bias: mean={b_ff2.mean():.4f}, range=[{b_ff2.min():.4f}, {b_ff2.max():.4f}]")

    # Check if FFN bias alone could create the offset
    print(f"\nPer-dim offset from ff.net.2 bias:")
    print(f"  Bias mean range: {(b_ff2.max() - b_ff2.min()).item():.4f}")

    # Block 1 biases
    print("\n" + "=" * 40)
    print("TRANSFORMER BLOCK 1 BIASES")
    print("=" * 40)

    b_out_1 = weights["transformer_blocks.1.attn1.to_out.0.bias"].float()
    b_ff2_1 = weights["transformer_blocks.1.ff.net.2.bias"].float()

    print(f"  to_out bias: mean={b_out_1.mean():.4f}, range=[{b_out_1.min():.4f}, {b_out_1.max():.4f}]")
    print(f"  ff.net.2 bias: mean={b_ff2_1.mean():.4f}, range=[{b_ff2_1.min():.4f}, {b_ff2_1.max():.4f}]")

    # Run actual forward pass through one transformer block
    print("\n" + "=" * 40)
    print("FULL BLOCK FORWARD (manual)")
    print("=" * 40)

    from llm_dit.encoders.embeddings_connector import BasicTransformerBlock1D, RopeType, precompute_freqs_cis

    # Create block with loaded weights
    block = BasicTransformerBlock1D(
        dim=3840,
        heads=30,
        dim_head=128,
        rope_type=RopeType.SPLIT,
    )

    # Load weights into block
    block.attn1.to_q.weight.data = weights["transformer_blocks.0.attn1.to_q.weight"]
    block.attn1.to_q.bias.data = weights["transformer_blocks.0.attn1.to_q.bias"]
    block.attn1.to_k.weight.data = weights["transformer_blocks.0.attn1.to_k.weight"]
    block.attn1.to_k.bias.data = weights["transformer_blocks.0.attn1.to_k.bias"]
    block.attn1.to_v.weight.data = weights["transformer_blocks.0.attn1.to_v.weight"]
    block.attn1.to_v.bias.data = weights["transformer_blocks.0.attn1.to_v.bias"]
    block.attn1.to_out[0].weight.data = weights["transformer_blocks.0.attn1.to_out.0.weight"]
    block.attn1.to_out[0].bias.data = weights["transformer_blocks.0.attn1.to_out.0.bias"]
    block.attn1.norm_q.weight.data = weights["transformer_blocks.0.attn1.norm_q.weight"]
    block.attn1.norm_k.weight.data = weights["transformer_blocks.0.attn1.norm_k.weight"]
    block.ff.net[0].proj.weight.data = weights["transformer_blocks.0.ff.net.0.proj.weight"]
    block.ff.net[0].proj.bias.data = weights["transformer_blocks.0.ff.net.0.proj.bias"]
    block.ff.net[2].weight.data = weights["transformer_blocks.0.ff.net.2.weight"]
    block.ff.net[2].bias.data = weights["transformer_blocks.0.ff.net.2.bias"]

    block = block.float()

    # Compute RoPE
    indices_grid = torch.arange(256, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 256]
    pe = precompute_freqs_cis(
        indices_grid=indices_grid,
        dim=3840,
        out_dtype=torch.float32,
        theta=10000.0,
        max_pos=[4096],
        num_attention_heads=30,
        rope_type=RopeType.SPLIT,
    )

    # Forward
    x_float = x.float()
    with torch.no_grad():
        out = block(x_float, attention_mask=None, pe=pe)

    print("\nBlock 0 output:")
    block0_range = analyze(out, "  Output")

    print(f"\n>>> Per-dim offset INCREASE: {block0_range - after_reg_range:.4f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
