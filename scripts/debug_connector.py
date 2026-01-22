"""
Debug the embeddings connector to find where zeros appear.

Last Updated: 2026-01-19

Tests whether the Embeddings1DConnector is corrupting the embeddings.
"""
import torch


def check_tensor(name: str, t: torch.Tensor):
    """Print tensor statistics."""
    t_float = t.float()
    print(f"{name}:")
    print(f"  Shape: {t.shape}")
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
    print("=== Embeddings Connector Debug ===\n")

    from llm_dit.encoders.gemma3 import Gemma3Encoder

    print("Loading text encoder...")
    encoder = Gemma3Encoder(
        model_id="models/LTX-2/text_encoder",
        device="cuda",
        dtype=torch.bfloat16,
        load_in_8bit=True,
    )

    # Get projected embeddings (before connector)
    prompt = "A cat walking through a sunny garden"
    print(f"\nPrompt: {prompt}\n")

    result = encoder.encode_multilayer([prompt], return_projected=True)
    embeddings = result["projected"]  # [1, 256, 3840]
    attention_mask = result["attention_mask"]  # [1, 256]

    print("--- Before Connector ---")
    check_tensor("embeddings", embeddings)
    check_tensor("attention_mask", attention_mask)

    # Check if connector exists and test it
    print("\n--- Connector Test ---")
    connector = encoder._embeddings_connector

    if connector is None:
        print("Connector is None - this is the problem!")
        print("Embeddings should be fine since we're not using connector")
    else:
        print(f"Connector exists: {type(connector)}")

        # Convert attention mask to additive format
        additive_mask = (1.0 - attention_mask.float()) * -10000.0
        additive_mask = additive_mask[:, None, None, :].to(embeddings.dtype)
        check_tensor("additive_mask", additive_mask)

        # Move connector to same device
        if next(connector.parameters()).device != embeddings.device:
            connector = connector.to(embeddings.device)

        # Run through connector
        connector_out, _ = connector(embeddings, additive_mask)
        check_tensor("after_connector", connector_out)

        # Check if connector corrupted the embeddings
        if connector_out.abs().max() < 1e-6:
            print("PROBLEM: Connector outputs zeros!")

    # Now call encode() to see full pipeline
    print("\n--- Full encode() method ---")
    output = encoder.encode([prompt])
    final_embeddings = output.embeddings[0]
    check_tensor("final_embeddings", final_embeddings)

    # Cleanup
    del encoder
    import gc
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
