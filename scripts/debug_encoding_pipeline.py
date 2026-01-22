"""
Debug the encoding pipeline to find where zeros appear.

Last Updated: 2026-01-19

Traces hidden states through each stage:
1. Gemma3 forward → hidden states
2. Stack 49 layers
3. Normalize (RMS + concat)
4. Feature extractor projection
"""
import torch


def check_tensor(name: str, t: torch.Tensor):
    """Print tensor statistics."""
    t_float = t.float()
    print(f"{name}:")
    print(f"  Shape: {t.shape}")
    print(f"  Dtype: {t.dtype}")
    print(f"  Mean: {t_float.mean():.6f}")
    print(f"  Std: {t_float.std():.6f}")
    print(f"  Abs max: {t_float.abs().max():.6f}")
    is_zero = t_float.abs().max() < 1e-6
    if is_zero:
        print(f"  STATUS: ALL ZEROS!")
    else:
        print(f"  STATUS: OK (non-zero)")
    print()


def main():
    print("=== Encoding Pipeline Debug ===\n")

    from llm_dit.encoders.gemma3 import Gemma3Encoder

    print("Loading text encoder...")
    encoder = Gemma3Encoder(
        model_id="models/LTX-2/text_encoder",
        device="cuda",
        dtype=torch.bfloat16,
        load_in_8bit=True,
    )

    # Use encode_multilayer to get intermediate outputs
    prompt = "A cat walking through a sunny garden"
    print(f"\nPrompt: {prompt}\n")

    # Trigger loading and get intermediate outputs
    print("=== Calling encode_multilayer ===\n")
    result = encoder.encode_multilayer([prompt], return_projected=True)

    # Check raw layer outputs
    print("--- Layer Stack ---")
    if "layer_stack" in result:
        check_tensor("layer_stack", result["layer_stack"])

    # Check attention mask
    print("--- Attention Mask ---")
    if "attention_mask" in result:
        check_tensor("attention_mask", result["attention_mask"])

    # Check projected embeddings
    print("--- Projected Embeddings ---")
    if "projected" in result and result["projected"] is not None:
        check_tensor("projected_embeddings", result["projected"])

    # Now let's manually trace through the feature extractor
    print("\n=== Manual Feature Extractor Test ===\n")

    if "layer_stack" in result:
        stacked = result["layer_stack"]
        attention_mask = result["attention_mask"]

        # Normalize: RMS norm over features, then concat layers
        print("--- RMS Normalization ---")
        from llm_dit.encoders.gemma3 import _norm_and_concat_layers
        normalized = _norm_and_concat_layers(stacked, attention_mask)
        check_tensor("normalized", normalized)

        # Feature extractor
        print("--- Feature Extractor ---")
        fe = encoder._feature_extractor
        print(f"Feature extractor weight shape: {fe.aggregate_embed.weight.shape}")
        print(f"Feature extractor weight device: {fe.aggregate_embed.weight.device}")
        print(f"Normalized tensor device: {normalized.device}")

        # Move to same device if needed
        if fe.aggregate_embed.weight.device != normalized.device:
            fe = fe.to(normalized.device)

        projected = fe(normalized)
        check_tensor("projected", projected)

    # Cleanup
    del encoder
    import gc
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
