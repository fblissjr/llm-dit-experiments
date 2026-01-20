"""
Verify connector weights are loaded correctly from checkpoint.

Last Updated: 2026-01-20

Compares our loaded connector weights against the checkpoint directly
to ensure all weights match exactly.
"""

import torch
from safetensors import safe_open
from pathlib import Path


def main():
    print("=" * 80)
    print("CONNECTOR WEIGHTS VERIFICATION")
    print("=" * 80)

    connectors_path = Path("models/LTX-2/connectors/diffusion_pytorch_model.safetensors")

    # Load connector weights directly from checkpoint
    print(f"\nLoading checkpoint: {connectors_path}")
    checkpoint_weights = {}
    with safe_open(connectors_path, framework="pt") as f:
        keys = [k for k in f.keys() if "video_connector" in k]
        print(f"Found {len(keys)} video_connector keys")
        for k in keys:
            checkpoint_weights[k] = f.get_tensor(k)

    # Print checkpoint structure
    print("\nCheckpoint keys:")
    for k in sorted(checkpoint_weights.keys())[:20]:
        tensor = checkpoint_weights[k]
        print(f"  {k}: shape={tensor.shape}, dtype={tensor.dtype}")
    if len(keys) > 20:
        print(f"  ... and {len(keys) - 20} more")

    # Load our encoder with connector
    print("\n\nLoading Gemma3Encoder with connector...")
    from llm_dit.encoders.gemma3 import Gemma3Encoder

    encoder = Gemma3Encoder(
        model_id="models/LTX-2/text_encoder",
        device="cuda",
        dtype=torch.bfloat16,
        max_sequence_length=256,
        connectors_path=str(connectors_path),
        use_connector=True,
        load_in_8bit=True,
    )

    # Force model loading
    encoder._load_model()

    # Get connector
    connector = encoder._embeddings_connector
    assert connector is not None, "Connector not loaded"

    # Print our connector structure
    print("\nOur connector state_dict keys:")
    our_weights = connector.state_dict()
    for k in sorted(our_weights.keys())[:20]:
        tensor = our_weights[k]
        print(f"  {k}: shape={tensor.shape}, dtype={tensor.dtype}")
    if len(our_weights) > 20:
        print(f"  ... and {len(our_weights) - 20} more")

    # Compare key weights
    print("\n" + "=" * 80)
    print("WEIGHT COMPARISON")
    print("=" * 80)

    # Map between checkpoint keys and our keys
    # Note: checkpoint uses "transformer_blocks", not "transformer_1d_blocks"
    key_mapping = {
        # Learnable registers
        "video_connector.learnable_registers": "learnable_registers",
        # Block 0 attention
        "video_connector.transformer_blocks.0.attn1.to_q.weight": "transformer_blocks.0.attn1.to_q.weight",
        "video_connector.transformer_blocks.0.attn1.to_k.weight": "transformer_blocks.0.attn1.to_k.weight",
        "video_connector.transformer_blocks.0.attn1.to_v.weight": "transformer_blocks.0.attn1.to_v.weight",
        "video_connector.transformer_blocks.0.attn1.to_out.0.weight": "transformer_blocks.0.attn1.to_out.0.weight",
        # Block 0 norm_q and norm_k
        "video_connector.transformer_blocks.0.attn1.norm_q.weight": "transformer_blocks.0.attn1.norm_q.weight",
        "video_connector.transformer_blocks.0.attn1.norm_k.weight": "transformer_blocks.0.attn1.norm_k.weight",
        # Block 0 feedforward
        "video_connector.transformer_blocks.0.ff.net.0.proj.weight": "transformer_blocks.0.ff.net.0.proj.weight",
        "video_connector.transformer_blocks.0.ff.net.2.weight": "transformer_blocks.0.ff.net.2.weight",
        # Block 1
        "video_connector.transformer_blocks.1.attn1.to_q.weight": "transformer_blocks.1.attn1.to_q.weight",
    }

    all_match = True
    for ckpt_key, our_key in key_mapping.items():
        if ckpt_key not in checkpoint_weights:
            print(f"WARNING: {ckpt_key} not in checkpoint!")
            continue
        if our_key not in our_weights:
            print(f"WARNING: {our_key} not in our model!")
            all_match = False
            continue

        ckpt_tensor = checkpoint_weights[ckpt_key].to(torch.float32)
        our_tensor = our_weights[our_key].to(torch.float32).cpu()

        if ckpt_tensor.shape != our_tensor.shape:
            print(f"MISMATCH {our_key}: shape {ckpt_tensor.shape} vs {our_tensor.shape}")
            all_match = False
            continue

        max_diff = (ckpt_tensor - our_tensor).abs().max().item()
        mean_diff = (ckpt_tensor - our_tensor).abs().mean().item()

        status = "OK" if max_diff < 1e-4 else "MISMATCH"
        print(f"{status}: {our_key}")
        print(f"       max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        if max_diff >= 1e-4:
            all_match = False

    print("\n" + "=" * 80)
    if all_match:
        print("All checked weights MATCH!")
    else:
        print("Some weights DO NOT MATCH!")
    print("=" * 80)

    # Also check learnable registers statistics
    print("\n\nLearnable registers analysis:")
    registers = our_weights["learnable_registers"].to(torch.float32)
    print(f"  Shape: {registers.shape}")
    print(f"  Dtype: {registers.dtype}")
    print(f"  Mean: {registers.mean():.6f}")
    print(f"  Std: {registers.std():.6f}")
    print(f"  Range: [{registers.min():.6f}, {registers.max():.6f}]")

    per_dim_mean = registers.mean(dim=0)
    print(f"  Per-dim mean range: [{per_dim_mean.min():.4f}, {per_dim_mean.max():.4f}]")


if __name__ == "__main__":
    main()
