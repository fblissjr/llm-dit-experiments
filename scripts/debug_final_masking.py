"""
Debug the final masking step that produces zeros.

Last Updated: 2026-01-19
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
    print(f"  Status: {'ZEROS' if is_zero else 'OK'}")
    print()


def main():
    print("=== Final Masking Debug ===\n")

    from llm_dit.encoders.gemma3 import Gemma3Encoder

    encoder = Gemma3Encoder(
        model_id="models/LTX-2/text_encoder",
        device="cuda",
        dtype=torch.bfloat16,
        load_in_8bit=True,
    )

    prompt = "A cat walking through a sunny garden"

    # Get projected embeddings and connector output manually
    result = encoder.encode_multilayer([prompt], return_projected=True)
    projected = result["projected"]  # [1, 256, 3840]
    attention_mask = result["attention_mask"]  # [1, 256]

    print("--- Before connector ---")
    check_tensor("projected", projected)

    # Check which positions have valid tokens
    seq_length = attention_mask.sum(dim=1).item()
    print(f"Attention mask: {seq_length} valid tokens out of {attention_mask.shape[1]}")
    print(f"Valid positions: {attention_mask[0, :10].tolist()}...")
    print()

    # Run connector
    connector = encoder._embeddings_connector
    additive_mask = (1.0 - attention_mask.float()) * -10000.0
    additive_mask = additive_mask[:, None, None, :].to(projected.dtype)

    connector = connector.to(projected.device)
    connector_out, new_mask = connector(projected, additive_mask)

    print("--- After connector ---")
    check_tensor("connector_out", connector_out)
    if new_mask is not None:
        print(f"New mask shape: {new_mask.shape}")
        print(f"New mask type: {new_mask.dtype}")

    # Check first few and valid tokens specifically
    print("--- Token analysis ---")
    print(f"First 8 tokens abs max: {connector_out[0, :8].abs().max():.6f}")
    print(f"First valid token ({int(seq_length-1)}): {connector_out[0, int(seq_length-1)].abs().max():.6f}")
    print()

    # Apply attention mask like encode() does
    print("--- Apply attention mask ---")
    masked = connector_out * attention_mask[:, :, None].to(connector_out.dtype)
    check_tensor("masked", masked)

    # Check the specific tokens
    print(f"Masked first 8 tokens abs max: {masked[0, :8].abs().max():.6f}")
    print()

    # Extract valid tokens
    print("--- Extract valid tokens ---")
    valid_tokens = masked[0, :int(seq_length)]
    check_tensor("valid_tokens", valid_tokens)

    # Check individual valid positions
    print("Individual token check:")
    for i in range(min(8, int(seq_length))):
        tok = masked[0, i]
        print(f"  Token {i}: abs_max={tok.abs().max():.6f}, mask={attention_mask[0, i].item()}")


if __name__ == "__main__":
    main()
