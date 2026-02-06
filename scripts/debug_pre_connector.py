"""
Debug embeddings BEFORE connector to isolate source of per-dim offsets.

Last Updated: 2026-01-20

The debug_text_vs_registers.py showed both text and registers have large
per-dim mean offsets AFTER the connector. This script checks if the offsets
exist BEFORE the connector (from feature extractor) or are created BY the connector.
"""

import torch
from pathlib import Path


def main():
    print("=" * 80)
    print("PRE-CONNECTOR EMBEDDING ANALYSIS")
    print("=" * 80)

    # Load encoder WITHOUT connector to see feature extractor output
    print("\nLoading Gemma3Encoder WITHOUT connector...")
    from llm_dit.encoders.gemma3 import Gemma3Encoder

    encoder_no_conn = Gemma3Encoder(
        model_id="models/LTX-2/text_encoder",
        device="cuda",
        dtype=torch.bfloat16,
        max_sequence_length=256,
        connectors_path="models/LTX-2/connectors/diffusion_pytorch_model.safetensors",
        use_connector=False,  # NO CONNECTOR
    )

    prompt = "A fluffy orange cat walking through a sunny garden"
    print(f"\nPrompt: {prompt}")

    # Encode without connector
    output_no_conn = encoder_no_conn.encode(prompt, return_padded=True)
    assert output_no_conn.padded_embeddings is not None

    # Get the valid token count
    if encoder_no_conn._tokenizer is None:
        encoder_no_conn._load_model()
    tokenizer = encoder_no_conn._tokenizer
    assert tokenizer is not None
    encoded = tokenizer(
        prompt,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    )
    attention_mask = encoded["attention_mask"][0]
    num_text_tokens = int(attention_mask.sum().item())

    print(f"\nText tokens: {num_text_tokens}")
    print(f"Padding: {256 - num_text_tokens}")

    # Analyze feature extractor output
    embeddings = output_no_conn.padded_embeddings[0]  # [256, 3840]

    print(f"\n--- FEATURE EXTRACTOR OUTPUT (no connector) ---")
    print(f"Shape: {embeddings.shape}")

    # Only analyze valid tokens (not padding)
    valid_start = 256 - num_text_tokens  # Left-padding
    valid_tokens = embeddings[valid_start:].float()

    print(f"\nVALID TOKENS ONLY (text tokens):")
    print(f"  Shape: {valid_tokens.shape}")
    print(f"  Mean: {valid_tokens.mean():.4f}, Std: {valid_tokens.std():.4f}")
    print(f"  Range: [{valid_tokens.min():.4f}, {valid_tokens.max():.4f}]")

    per_dim_mean = valid_tokens.mean(dim=0)
    per_dim_std = valid_tokens.std(dim=0)
    print(f"  Per-dim mean range: [{per_dim_mean.min():.4f}, {per_dim_mean.max():.4f}]")
    print(f"  Per-dim std range: [{per_dim_std.min():.4f}, {per_dim_std.max():.4f}]")
    print(f"  Dims with |mean| > 1.0: {(per_dim_mean.abs() > 1.0).sum().item()} / 3840")
    print(f"  Dims with |mean| > 5.0: {(per_dim_mean.abs() > 5.0).sum().item()} / 3840")
    print(f"  Dims with std < 0.5: {(per_dim_std < 0.5).sum().item()} / 3840")

    # Clean up before loading with connector
    del encoder_no_conn
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Now load WITH connector
    print("\n\nLoading Gemma3Encoder WITH connector...")
    encoder_with_conn = Gemma3Encoder(
        model_id="models/LTX-2/text_encoder",
        device="cuda",
        dtype=torch.bfloat16,
        max_sequence_length=256,
        connectors_path="models/LTX-2/connectors/diffusion_pytorch_model.safetensors",
        use_connector=True,  # WITH CONNECTOR
    )

    output_with_conn = encoder_with_conn.encode(prompt, return_padded=True)
    assert output_with_conn.padded_embeddings is not None
    embeddings_with_conn = output_with_conn.padded_embeddings[0]  # [256, 3840]

    # After connector, registers are first, text tokens are last
    num_registers = 256 - num_text_tokens
    text_tokens = embeddings_with_conn[num_registers:].float()

    print(f"\n--- CONNECTOR OUTPUT (text tokens only) ---")
    print(f"  Shape: {text_tokens.shape}")
    print(f"  Mean: {text_tokens.mean():.4f}, Std: {text_tokens.std():.4f}")
    print(f"  Range: [{text_tokens.min():.4f}, {text_tokens.max():.4f}]")

    per_dim_mean_conn = text_tokens.mean(dim=0)
    per_dim_std_conn = text_tokens.std(dim=0)
    print(f"  Per-dim mean range: [{per_dim_mean_conn.min():.4f}, {per_dim_mean_conn.max():.4f}]")
    print(f"  Per-dim std range: [{per_dim_std_conn.min():.4f}, {per_dim_std_conn.max():.4f}]")
    print(f"  Dims with |mean| > 1.0: {(per_dim_mean_conn.abs() > 1.0).sum().item()} / 3840")
    print(f"  Dims with |mean| > 5.0: {(per_dim_mean_conn.abs() > 5.0).sum().item()} / 3840")
    print(f"  Dims with std < 0.5: {(per_dim_std_conn < 0.5).sum().item()} / 3840")

    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON: PRE-CONNECTOR vs POST-CONNECTOR (text tokens)")
    print("=" * 80)

    pre_range = (per_dim_mean.max() - per_dim_mean.min()).item()
    post_range = (per_dim_mean_conn.max() - per_dim_mean_conn.min()).item()

    print(f"\nPer-dim mean range:")
    print(f"  Pre-connector:  {pre_range:.4f}")
    print(f"  Post-connector: {post_range:.4f}")
    print(f"  Change: {post_range - pre_range:.4f}")

    if pre_range > 5.0:
        print("\n>>> Feature extractor ALREADY creates large per-dim offsets!")
        print(">>> The issue is BEFORE the connector.")
    elif post_range > 5.0:
        print("\n>>> Connector CREATES the large per-dim offsets!")
        print(">>> The issue is IN the connector's transformer blocks.")
    else:
        print("\n>>> Both stages have small per-dim offsets. Issue is elsewhere.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
