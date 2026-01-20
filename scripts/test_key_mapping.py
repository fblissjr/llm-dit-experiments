"""
Verify transformer key mapping between checkpoint and our model.

Last Updated: 2026-01-20
"""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from llm_dit.models.ltx2.loader import load_ltx2_transformer, map_key


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
    print("TRANSFORMER KEY MAPPING ANALYSIS")
    print("=" * 80)

    # Load checkpoint
    ckpt_path = "models/LTX-2/transformer"
    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt = load_sharded_checkpoint(ckpt_path)

    # Separate video and audio keys
    video_keys = sorted([k for k in ckpt.keys()
                         if not any(x in k for x in ['audio', 'av_cross_attention'])])
    audio_keys = sorted([k for k in ckpt.keys()
                         if any(x in k for x in ['audio', 'av_cross_attention'])])

    print(f"Video keys: {len(video_keys)}")
    print(f"Audio keys: {len(audio_keys)}")

    # Apply key mapping to video keys
    print("\n--- Key Mapping Test ---")
    mapped_keys = set()
    mapping_errors = []
    for ckpt_key in video_keys:
        try:
            our_key = map_key(ckpt_key)
            mapped_keys.add(our_key)
        except Exception as e:
            mapping_errors.append((ckpt_key, str(e)))

    print(f"Successfully mapped: {len(mapped_keys)}")
    print(f"Mapping errors: {len(mapping_errors)}")

    # Load our model and get its keys
    print("\n--- Loading our model ---")
    model = load_ltx2_transformer(
        "models/LTX-2/transformer",
        dtype=torch.bfloat16,
        device="cpu",
        video_only=True,
    )
    model_keys = set(model.state_dict().keys())
    print(f"Model parameters: {len(model_keys)}")

    # Compare
    print("\n--- Key Comparison ---")
    in_ckpt_not_model = mapped_keys - model_keys
    in_model_not_ckpt = model_keys - mapped_keys

    print(f"Mapped keys not in model: {len(in_ckpt_not_model)}")
    if in_ckpt_not_model:
        print("  First 10:")
        for k in sorted(in_ckpt_not_model)[:10]:
            print(f"    {k}")

    print(f"Model keys not in mapped: {len(in_model_not_ckpt)}")
    if in_model_not_ckpt:
        print("  First 10:")
        for k in sorted(in_model_not_ckpt)[:10]:
            print(f"    {k}")

    # Show sample mappings
    print("\n--- Sample Key Mappings ---")
    sample_keys = video_keys[:10]
    for k in sample_keys:
        mapped = map_key(k)
        print(f"  {k}")
        print(f"    -> {mapped}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
