"""
Debug script to separate text token vs register token statistics.

Last Updated: 2026-01-20

Gemini's hypothesis: The large per-dim mean offsets might come from registers,
not from text tokens. If text tokens are healthy (mean ~0) and registers have
large offsets, the "centering fix" might be wrong.

This script:
1. Encodes a prompt WITH connector
2. Separates text tokens (first N) from register tokens (remaining)
3. Computes statistics for each group separately
4. Traces through caption_projection for each group
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


def analyze_group(x, name, weights=None):
    """Analyze a token group."""
    x = x.to(torch.float32)
    print(f"\n{name}:")
    print(f"  Shape: {x.shape}")
    print(f"  Mean: {x.mean():.4f}, Std: {x.std():.4f}")
    print(f"  Range: [{x.min():.4f}, {x.max():.4f}]")

    # Per-dimension statistics (across sequence)
    per_dim_mean = x.mean(dim=0)  # [3840]
    per_dim_std = x.std(dim=0)    # [3840]
    print(f"  Per-dim mean range: [{per_dim_mean.min():.4f}, {per_dim_mean.max():.4f}]")
    print(f"  Per-dim std range: [{per_dim_std.min():.4f}, {per_dim_std.max():.4f}]")
    print(f"  Dims with std < 0.5: {(per_dim_std < 0.5).sum().item()} / 3840")

    if weights is not None:
        # Trace through caption_projection
        W1 = weights["linear_1_weight"].to(torch.float32).cpu()
        b1 = weights["linear_1_bias"].to(torch.float32).cpu()
        W2 = weights["linear_2_weight"].to(torch.float32).cpu()
        b2 = weights["linear_2_bias"].to(torch.float32).cpu()

        x_batch = x.unsqueeze(0).cpu()  # [1, seq, 3840]

        x1 = torch.nn.functional.linear(x_batch, W1, b1)
        print(f"  After linear_1: mean={x1.mean():.4f}, std={x1.std():.4f}")

        x2 = torch.nn.functional.gelu(x1, approximate="tanh")
        print(f"  After GELU: mean={x2.mean():.4f}, std={x2.std():.4f}")

        x3 = torch.nn.functional.linear(x2, W2, b2)
        print(f"  After linear_2: mean={x3.mean():.4f}, std={x3.std():.4f}")


def main():
    print("=" * 80)
    print("TEXT VS REGISTER STATISTICS COMPARISON")
    print("=" * 80)

    # Load encoder with connector
    print("\nLoading Gemma3Encoder...")
    from llm_dit.encoders.gemma3 import Gemma3Encoder

    encoder = Gemma3Encoder(
        model_id="models/LTX-2/text_encoder",
        device="cuda",
        dtype=torch.bfloat16,
        max_sequence_length=256,
        connectors_path="models/LTX-2/connectors/diffusion_pytorch_model.safetensors",
        use_connector=True,
        load_in_8bit=True,
    )

    # Encode test prompt
    prompt = "A fluffy orange cat walking through a sunny garden"
    print(f"\nPrompt: {prompt}")

    # We need to track the original token count before connector
    # The connector pads to max_sequence_length (256)
    # We need to know how many were actual text tokens

    # First, get tokenization to find real text length
    if encoder._tokenizer is None:
        encoder._load_model()  # This loads the tokenizer

    tokenizer = encoder._tokenizer
    assert tokenizer is not None, "Tokenizer not loaded"

    encoded = tokenizer(
        prompt,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    )
    attention_mask = encoded["attention_mask"][0]
    num_text_tokens = int(attention_mask.sum().item())
    print(f"\nOriginal text tokens: {num_text_tokens}")
    print(f"Register tokens: {256 - num_text_tokens}")

    # Now encode with connector
    output = encoder.encode(prompt, return_padded=True)
    assert output.padded_embeddings is not None, "padded_embeddings not returned"
    embeddings = output.padded_embeddings[0]  # [256, 3840]

    print(f"\nTotal embeddings shape: {embeddings.shape}")

    # The connector REVERSES the order: registers first, then text tokens at the end
    # This is from _replace_padded_with_learnable_registers:
    # "This moves valid tokens to the end of the sequence and fills
    #  the beginning with learnable register tokens."

    num_registers = 256 - num_text_tokens
    register_tokens = embeddings[:num_registers]  # First N are registers
    text_tokens = embeddings[num_registers:]      # Last tokens are text

    print(f"\nAfter connector processing (registers first, text last):")
    print(f"  Register tokens: positions 0-{num_registers-1} ({num_registers} tokens)")
    print(f"  Text tokens: positions {num_registers}-255 ({num_text_tokens} tokens)")

    # Load caption_projection weights for full analysis
    print("\nLoading caption_projection weights...")
    weights = load_caption_projection_weights()

    # Analyze each group
    print("\n" + "=" * 80)
    print("SEPARATED STATISTICS")
    print("=" * 80)

    analyze_group(text_tokens, "TEXT TOKENS ONLY", weights)
    analyze_group(register_tokens, "REGISTER TOKENS ONLY", weights)
    analyze_group(embeddings, "ALL TOKENS (text + registers)", weights)

    # Check correlation between groups
    print("\n" + "=" * 80)
    print("CROSS-GROUP ANALYSIS")
    print("=" * 80)

    text_per_dim_mean = text_tokens.float().mean(dim=0)
    reg_per_dim_mean = register_tokens.float().mean(dim=0)
    correlation = torch.corrcoef(torch.stack([text_per_dim_mean, reg_per_dim_mean]))[0, 1]
    print(f"\nPer-dim mean correlation (text vs registers): {correlation:.4f}")

    # Check if dimensions with large offsets are same in both groups
    text_large_offset_dims = (text_per_dim_mean.abs() > 1.0)
    reg_large_offset_dims = (reg_per_dim_mean.abs() > 1.0)
    both_large = (text_large_offset_dims & reg_large_offset_dims).sum().item()
    print(f"Dims with |mean| > 1.0: text={text_large_offset_dims.sum().item()}, "
          f"registers={reg_large_offset_dims.sum().item()}, overlap={both_large}")

    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    text_mean = text_tokens.float().mean()
    reg_mean = register_tokens.float().mean()
    text_per_dim_range = (text_per_dim_mean.max() - text_per_dim_mean.min()).item()
    reg_per_dim_range = (reg_per_dim_mean.max() - reg_per_dim_mean.min()).item()

    print(f"\nText tokens: overall mean={text_mean:.4f}, per-dim mean range={text_per_dim_range:.4f}")
    print(f"Registers:   overall mean={reg_mean:.4f}, per-dim mean range={reg_per_dim_range:.4f}")

    if text_per_dim_range < 2.0 and reg_per_dim_range > 5.0:
        print("\n>>> DIAGNOSIS: Registers have large per-dim offsets, text is healthy")
        print(">>> The 'centering fix' may shift text tokens incorrectly!")
    elif text_per_dim_range > 5.0 and reg_per_dim_range < 2.0:
        print("\n>>> DIAGNOSIS: Text has large per-dim offsets, registers are healthy")
        print(">>> The connector's transformer blocks may be creating offsets")
    elif text_per_dim_range > 5.0 and reg_per_dim_range > 5.0:
        print("\n>>> DIAGNOSIS: BOTH groups have large per-dim offsets")
        print(">>> The connector's processing affects both equally")
    else:
        print("\n>>> DIAGNOSIS: Both groups have small per-dim offsets")
        print(">>> The issue may be elsewhere")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
