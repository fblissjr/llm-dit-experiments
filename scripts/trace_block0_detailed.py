"""
Detailed trace of Block 0 to find where explosion happens.

Last Updated: 2026-01-20

Block 0 causes per-dim range to explode from 0.26 to 780.
This script traces each operation to find the culprit.
"""

import torch
from pathlib import Path


def analyze(x, name, show_range=True):
    """Print statistics."""
    x = x.float()
    per_dim_mean = x.mean(dim=(0, 1)) if x.ndim == 3 else x.mean(dim=0)
    mean_range = (per_dim_mean.max() - per_dim_mean.min()).item()
    if show_range:
        print(f"{name}: mean={x.mean():.4f}, std={x.std():.4f}, per-dim-range={mean_range:.4f}")
    else:
        print(f"{name}: mean={x.mean():.4f}, std={x.std():.4f}")
    return mean_range


def main():
    print("=" * 80)
    print("DETAILED BLOCK 0 TRACE")
    print("=" * 80)

    # Load encoder WITHOUT connector
    print("\nLoading encoder...")
    from llm_dit.encoders.gemma3 import Gemma3Encoder

    encoder = Gemma3Encoder(
        model_id="models/LTX-2/text_encoder",
        device="cuda",
        dtype=torch.bfloat16,
        max_sequence_length=256,
        connectors_path="models/LTX-2/connectors/diffusion_pytorch_model.safetensors",
        use_connector=False,
    )

    prompt = "A fluffy orange cat walking through a sunny garden"
    output = encoder.encode(prompt, return_padded=True)
    assert output.padded_embeddings is not None
    embeddings = output.padded_embeddings[0].clone()

    # Get token count
    if encoder._tokenizer is None:
        encoder._load_model()
    tokenizer = encoder._tokenizer
    assert tokenizer is not None
    encoded = tokenizer(prompt, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
    attention_mask = encoded["attention_mask"][0]

    print(f"\nPrompt: {prompt}")

    # Load connector
    from llm_dit.encoders.embeddings_connector import (
        Embeddings1DConnector, RopeType, load_connector_weights,
        precompute_freqs_cis, rms_norm, apply_rotary_emb,
    )

    connector = Embeddings1DConnector(
        attention_head_dim=128, num_attention_heads=30, num_layers=2,
        positional_embedding_theta=10000.0, positional_embedding_max_pos=[4096],
        num_learnable_registers=128, rope_type=RopeType.SPLIT,
        use_double_precision_rope=True,
    )
    load_connector_weights(connector, Path("models/LTX-2/connectors/diffusion_pytorch_model.safetensors"), "video_connector.")
    connector = connector.to(embeddings.device, dtype=embeddings.dtype)

    # Prepare input
    x = embeddings.unsqueeze(0)
    additive_mask = (1.0 - attention_mask.float()) * -10000.0
    additive_mask = additive_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(x.device).to(x.dtype)

    # Replace padding with registers
    x, new_mask = connector._replace_padded_with_learnable_registers(x, additive_mask)
    print(f"\nInput shape: {x.shape}")
    analyze(x, "Input (after register insertion)")

    # Compute RoPE
    indices_grid = torch.arange(256, dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(0)
    pe = precompute_freqs_cis(
        indices_grid=indices_grid, dim=3840, out_dtype=x.dtype, theta=10000.0,
        max_pos=[4096], num_attention_heads=30, rope_type=RopeType.SPLIT,
        use_double_precision=True,
    )
    print(f"\nRoPE pe shapes: cos={pe[0].shape}, sin={pe[1].shape}")

    # Manual Block 0 forward
    block = connector.transformer_blocks[0]

    print("\n" + "=" * 40)
    print("BLOCK 0 MANUAL FORWARD")
    print("=" * 40)

    # Step 1: Pre-norm
    norm_x = rms_norm(x)
    if norm_x.ndim == 4:
        norm_x = norm_x.squeeze(1)
    analyze(norm_x, "\n1. After RMSNorm (pre-attention)")

    # Step 2: Q, K, V projections
    q = block.attn1.to_q(norm_x)
    k = block.attn1.to_k(norm_x)
    v = block.attn1.to_v(norm_x)
    analyze(q, "2a. Q (after to_q)")
    analyze(k, "2b. K (after to_k)")
    analyze(v, "2c. V (after to_v)")

    # Step 3: QK normalization
    q_normed = block.attn1.norm_q(q)
    k_normed = block.attn1.norm_k(k)
    analyze(q_normed, "3a. Q (after norm_q)")
    analyze(k_normed, "3b. K (after norm_k)")

    # Step 4: Apply RoPE
    q_rope = apply_rotary_emb(q_normed, pe, RopeType.SPLIT)
    k_rope = apply_rotary_emb(k_normed, pe, RopeType.SPLIT)
    analyze(q_rope, "4a. Q (after RoPE)")
    analyze(k_rope, "4b. K (after RoPE)")

    # Step 5: Reshape for multi-head attention
    b, seq_len, _ = q_rope.shape
    heads, dim_head = 30, 128
    q_mh = q_rope.view(b, seq_len, heads, dim_head).transpose(1, 2)  # [B, H, T, D]
    k_mh = k_rope.view(b, seq_len, heads, dim_head).transpose(1, 2)
    v_mh = v.view(b, seq_len, heads, dim_head).transpose(1, 2)

    print(f"\n5. Multi-head shapes: Q={q_mh.shape}, K={k_mh.shape}, V={v_mh.shape}")

    # Step 6: Attention
    attn_out = torch.nn.functional.scaled_dot_product_attention(q_mh, k_mh, v_mh, attn_mask=new_mask, dropout_p=0.0)
    print(f"6. Attention output shape: {attn_out.shape}")
    attn_out_flat = attn_out.transpose(1, 2).reshape(b, seq_len, heads * dim_head)
    analyze(attn_out_flat, "6. Attention output (reshaped)")

    # Step 7: Output projection
    out_proj = block.attn1.to_out(attn_out_flat)
    analyze(out_proj, "7. After to_out projection")

    # Step 8: Residual connection
    residual = out_proj + x
    analyze(residual, "8. After attention residual (attn_out + x)")

    # Step 9: Pre-norm for FFN
    norm_hidden = rms_norm(residual)
    analyze(norm_hidden, "\n9. After RMSNorm (pre-FFN)")

    # Step 10: FFN
    ff_out = block.ff(norm_hidden)
    analyze(ff_out, "10. After FFN")

    # Step 11: FFN residual
    final = ff_out + residual
    final_range = analyze(final, "11. After FFN residual (ff_out + residual)")

    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    if final_range > 100:
        print("\n>>> MASSIVE EXPLOSION detected!")
        print(">>> Check steps above for first large increase in per-dim-range")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
