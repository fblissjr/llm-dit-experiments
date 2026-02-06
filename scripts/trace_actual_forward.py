"""
Trace connector forward pass with ACTUAL encoder output.

Last Updated: 2026-01-20

Use actual feature extractor output to see the real per-dim offset creation.
"""

import torch
from safetensors import safe_open
from pathlib import Path


def rms_norm(x, weight=None, eps=1e-6):
    """RMSNorm implementation."""
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)


def analyze(x, name):
    """Print per-dim statistics."""
    x = x.float()
    per_dim_mean = x.mean(dim=(0, 1)) if x.ndim == 3 else x.mean(dim=0)
    mean_range = (per_dim_mean.max() - per_dim_mean.min()).item()
    print(f"{name}:")
    print(f"  Mean: {x.mean():.4f}, Std: {x.std():.4f}")
    print(f"  Per-dim mean range: {mean_range:.4f}")
    return mean_range


def main():
    print("=" * 80)
    print("ACTUAL FORWARD PASS TRACING")
    print("=" * 80)

    # Load encoder WITHOUT connector to get feature extractor output
    print("\nLoading encoder WITHOUT connector...")
    from llm_dit.encoders.gemma3 import Gemma3Encoder

    encoder = Gemma3Encoder(
        model_id="models/LTX-2/text_encoder",
        device="cuda",
        dtype=torch.bfloat16,
        max_sequence_length=256,
        connectors_path="models/LTX-2/connectors/diffusion_pytorch_model.safetensors",
        use_connector=False,  # No connector
    )

    prompt = "A fluffy orange cat walking through a sunny garden"
    output = encoder.encode(prompt, return_padded=True)
    assert output.padded_embeddings is not None

    # Get embeddings (before connector)
    embeddings = output.padded_embeddings[0].clone()  # [256, 3840]

    # Get actual token count
    if encoder._tokenizer is None:
        encoder._load_model()
    tokenizer = encoder._tokenizer
    assert tokenizer is not None
    encoded = tokenizer(prompt, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
    attention_mask = encoded["attention_mask"][0]
    num_text = int(attention_mask.sum().item())

    print(f"\nPrompt: {prompt}")
    print(f"Text tokens: {num_text}, Padding: {256 - num_text}")

    # Analyze input
    valid_start = 256 - num_text  # Left-padding
    valid_tokens = embeddings[valid_start:].float()
    print("\n" + "=" * 40)
    print("FEATURE EXTRACTOR OUTPUT (valid tokens only)")
    print("=" * 40)
    input_range = analyze(valid_tokens.unsqueeze(0), "  Input")

    # Load connector
    print("\n\nLoading connector for manual forward pass...")
    from llm_dit.encoders.embeddings_connector import (
        Embeddings1DConnector,
        RopeType,
        load_connector_weights,
        precompute_freqs_cis,
        rms_norm as connector_rms_norm,
    )

    connector = Embeddings1DConnector(
        attention_head_dim=128,
        num_attention_heads=30,
        num_layers=2,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=[4096],
        num_learnable_registers=128,
        rope_type=RopeType.SPLIT,
        use_double_precision_rope=True,
    )

    load_connector_weights(
        connector,
        Path("models/LTX-2/connectors/diffusion_pytorch_model.safetensors"),
        prefix="video_connector.",
    )
    connector = connector.to(embeddings.device, dtype=embeddings.dtype)

    # Prepare input in correct format
    x = embeddings.unsqueeze(0)  # [1, 256, 3840]

    # Create additive attention mask (0=valid, -10000=padding)
    additive_mask = (1.0 - attention_mask.float()) * -10000.0
    additive_mask = additive_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(x.device).to(x.dtype)

    print("\n" + "=" * 40)
    print("TRACING CONNECTOR STEPS")
    print("=" * 40)

    # Step 1: _replace_padded_with_learnable_registers
    x_with_regs, new_mask = connector._replace_padded_with_learnable_registers(x, additive_mask)
    print("\nAfter register insertion:")
    reg_range = analyze(x_with_regs, "  Combined")

    # Step 2: Compute RoPE
    indices_grid = torch.arange(256, dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(0)
    pe = precompute_freqs_cis(
        indices_grid=indices_grid,
        dim=connector.inner_dim,
        out_dtype=x_with_regs.dtype,
        theta=connector.positional_embedding_theta,
        max_pos=connector.positional_embedding_max_pos,
        num_attention_heads=connector.num_attention_heads,
        rope_type=connector.rope_type,
        use_double_precision=connector.use_double_precision_rope,
    )

    # Step 3: Transformer block 0
    hidden = x_with_regs
    block0 = connector.transformer_blocks[0]

    print("\n--- BLOCK 0 ---")

    # Pre-norm
    norm_hidden = connector_rms_norm(hidden)
    print("\nAfter pre-norm (before attention):")
    analyze(norm_hidden, "  Normalized")

    # Full block forward
    with torch.no_grad():
        hidden = block0(hidden, attention_mask=new_mask, pe=pe)
    print("\nAfter block 0:")
    block0_range = analyze(hidden, "  Output")

    # Step 4: Transformer block 1
    block1 = connector.transformer_blocks[1]

    print("\n--- BLOCK 1 ---")
    with torch.no_grad():
        hidden = block1(hidden, attention_mask=new_mask, pe=pe)
    print("\nAfter block 1:")
    block1_range = analyze(hidden, "  Output")

    # Step 5: Final RMSNorm
    print("\n--- FINAL RMSNORM ---")
    hidden = connector_rms_norm(hidden)
    final_range = analyze(hidden, "  Final output")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nPer-dim mean range at each stage:")
    print(f"  Input (feature extractor): {input_range:.4f}")
    print(f"  After register insertion:  {reg_range:.4f}")
    print(f"  After block 0:             {block0_range:.4f}")
    print(f"  After block 1:             {block1_range:.4f}")
    print(f"  After final RMSNorm:       {final_range:.4f}")

    if block0_range > reg_range * 5:
        print("\n>>> Block 0 creates most of the offset!")
    elif block1_range > block0_range * 5:
        print("\n>>> Block 1 creates most of the offset!")
    else:
        print("\n>>> Offsets accumulate gradually across blocks")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
