"""
Verify encoder output is non-zero and properly projected.

Last Updated: 2026-01-19

Checks that feature extractor weights are loaded and producing valid embeddings.
"""
import torch


def main():
    print("=== Encoder Output Verification ===\n")

    from llm_dit.encoders.gemma3 import Gemma3Encoder

    print("Loading text encoder...")
    encoder = Gemma3Encoder(
        model_id="models/LTX-2/text_encoder",
        device="cuda",
        dtype=torch.bfloat16,
    )

    # Encode a test prompt FIRST to trigger lazy loading
    print("\n--- Encoding Test (triggers lazy loading) ---")
    prompts = ["A cat walking through a sunny garden"]

    output = encoder.encode(prompts)
    embeddings = output.embeddings[0]
    mask = output.attention_masks[0]

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")
    print(f"Mask shape: {mask.shape}")

    # NOW check if feature extractor has loaded weights
    print("\n--- Feature Extractor Weights (after lazy load) ---")
    if hasattr(encoder, "_feature_extractor") and encoder._feature_extractor is not None:
        fe = encoder._feature_extractor
        if hasattr(fe, "aggregate_embed"):
            w = fe.aggregate_embed.weight
            print(f"Weight shape: {w.shape}")
            print(f"Weight dtype: {w.dtype}")
            print(f"Weight stats: mean={w.float().mean():.6f}, std={w.float().std():.6f}")
            print(f"Weight range: [{w.float().min():.6f}, {w.float().max():.6f}]")
            print(f"Weight non-zero: {(w != 0).sum().item()} / {w.numel()}")

            if w.abs().sum() < 1e-6:
                print("WARNING: Feature extractor weights appear to be zero/uninitialized!")
        else:
            print("WARNING: Feature extractor has no aggregate_embed attribute!")
    else:
        print("WARNING: No feature extractor found!")

    # Check embedding statistics (using embeddings from earlier)
    emb_float = embeddings.float()
    print(f"\nEmbedding stats:")
    print(f"  Mean: {emb_float.mean():.6f}")
    print(f"  Std: {emb_float.std():.6f}")
    print(f"  Min: {emb_float.min():.6f}")
    print(f"  Max: {emb_float.max():.6f}")
    print(f"  Abs max: {emb_float.abs().max():.6f}")

    # Check if embeddings are zero
    is_zero = emb_float.abs().max() < 1e-6
    if is_zero:
        print("\nFAILED: Embeddings are all zeros!")
    else:
        print("\nPASSED: Embeddings are non-zero")

    # Per-token stats
    print("\n--- Per-Token Statistics ---")
    for i in range(min(4, embeddings.shape[0])):
        tok = embeddings[i].float()
        print(f"Token {i}: mean={tok.mean():.4f}, std={tok.std():.4f}, "
              f"abs_max={tok.abs().max():.4f}")

    # Cleanup
    del encoder
    import gc
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
