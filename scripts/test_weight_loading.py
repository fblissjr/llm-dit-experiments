"""
Verify transformer weights are loaded correctly.

Last Updated: 2026-01-20
"""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from llm_dit.models.ltx2.loader import load_ltx2_transformer


def load_sharded_checkpoint(path: str) -> dict:
    """Load sharded safetensors checkpoint."""
    path = Path(path)
    index_file = path / "diffusion_pytorch_model.safetensors.index.json"

    with open(index_file) as f:
        index = json.load(f)

    all_tensors = {}
    shard_files = set(index["weight_map"].values())
    for shard_file in shard_files:
        shard_path = path / shard_file
        tensors = load_file(str(shard_path))
        all_tensors.update(tensors)

    return all_tensors


def main():
    print("=" * 80)
    print("TRANSFORMER WEIGHT LOADING VERIFICATION")
    print("=" * 80)

    # Load checkpoint directly (sharded)
    ckpt_path = "models/LTX-2/transformer"
    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt = load_sharded_checkpoint(ckpt_path)
    ckpt_keys = set(ckpt.keys())
    print(f"Checkpoint keys: {len(ckpt_keys)}")

    # Filter to video-only keys (skip audio)
    video_keys = {k for k in ckpt_keys if not any(x in k for x in ['audio', 'av_cross_attention'])}
    audio_keys = ckpt_keys - video_keys
    print(f"Video keys: {len(video_keys)}")
    print(f"Audio keys (skipped): {len(audio_keys)}")

    # Load our model
    print("\nLoading our transformer model...")
    model = load_ltx2_transformer(
        "models/LTX-2/transformer",
        dtype=torch.bfloat16,
        device="cpu",
        video_only=True,
    )

    model_keys = set(model.state_dict().keys())
    print(f"Model parameters: {len(model_keys)}")

    # Check for any uninitialized parameters (still default initialized)
    print("\n--- Parameter Statistics Check ---")
    suspicious = []
    for name, param in model.named_parameters():
        if param.abs().mean() == 0:
            suspicious.append(f"{name}: all zeros")
        elif param.abs().max() > 100:
            suspicious.append(f"{name}: unusually large values (max={param.abs().max():.2f})")

    if suspicious:
        print("Suspicious parameters:")
        for s in suspicious[:20]:
            print(f"  {s}")
    else:
        print("All parameters look initialized correctly.")

    # Check specific layers
    print("\n--- Sample Weight Statistics ---")
    layers_to_check = [
        "blocks.0.attn.qkv_proj.weight",
        "blocks.0.ff.linear_1.weight",
        "blocks.23.attn.out_proj.weight",
        "blocks.47.ff.linear_2.weight",
        "proj_in.weight",
        "proj_out.weight",
    ]

    for layer in layers_to_check:
        if layer in model.state_dict():
            w = model.state_dict()[layer]
            print(f"{layer}: shape={w.shape}, mean={w.mean():.6f}, std={w.std():.6f}")
        else:
            print(f"{layer}: NOT FOUND")

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
