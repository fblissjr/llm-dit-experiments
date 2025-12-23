#!/usr/bin/env python3
"""
Convert image encoders to DiffSynth-compatible format.

Converts:
- facebook/dinov3-vit7b16-pretrain-lvd1689m (FP32 -> BF16)
- google/siglip2-giant-opt-patch16-384 (extract vision encoder)

Output format matches DiffSynth-Studio/General-Image-Encoders structure.

Usage:
    uv run scripts/convert_image_encoders.py --output-dir ./image_encoders
    uv run scripts/convert_image_encoders.py --dinov3-only --output-dir ./image_encoders
    uv run scripts/convert_image_encoders.py --siglip2-only --output-dir ./image_encoders
"""

import argparse
import gc
import os
from pathlib import Path

import torch
from safetensors.torch import save_file


def convert_dinov3(
    output_dir: Path,
    source_model: str = "facebook/dinov3-vit7b16-pretrain-lvd1689m/",
):
    """Convert DINOv3-7B from FP32 to BF16 safetensors."""
    from transformers import DINOv3ViTModel

    output_path = output_dir / "DINOv3-7B"
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "model.safetensors"

    if output_file.exists():
        print(f"[DINOv3] Output already exists: {output_file}")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != "y":
            print("[DINOv3] Skipping")
            return

    print(f"[DINOv3] Loading {source_model}...")
    print("[DINOv3] This will use ~27GB RAM temporarily")

    # Load in FP32 (original format)
    model = DINOv3ViTModel.from_pretrained(
        source_model,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    print("[DINOv3] Converting to BF16...")
    state_dict = model.state_dict()

    # Convert each tensor to BF16
    bf16_state_dict = {}
    for key, tensor in state_dict.items():
        bf16_state_dict[key] = tensor.to(torch.bfloat16).contiguous()

    # Verify shapes match DiffSynth expectations
    expected_keys = [
        "embeddings.cls_token",
        "embeddings.mask_token",
        "embeddings.patch_embeddings.weight",
        "embeddings.patch_embeddings.bias",
        "embeddings.register_tokens",
        "layer.0.attention.k_proj.weight",
        "layer.0.attention.q_proj.weight",
        "layer.0.attention.v_proj.weight",
        "layer.0.attention.o_proj.weight",
        "layer.0.attention.o_proj.bias",
        "layer.0.mlp.gate_proj.weight",
        "layer.0.mlp.up_proj.weight",
        "layer.0.mlp.down_proj.weight",
        "layer.0.layer_scale1.lambda1",
        "layer.0.layer_scale2.lambda1",
        "norm.weight",
        "norm.bias",
    ]

    missing = [k for k in expected_keys if k not in bf16_state_dict]
    if missing:
        print(f"[DINOv3] Warning: Missing expected keys: {missing}")

    # Check dimensions
    print(f"[DINOv3] Verifying dimensions...")
    embed_dim = bf16_state_dict["embeddings.cls_token"].shape[-1]
    num_layers = sum(
        1 for k in bf16_state_dict if k.startswith("layer.") and ".attention.k_proj.weight" in k
    )
    print(f"  - Hidden dim: {embed_dim} (expected: 4096)")
    print(f"  - Num layers: {num_layers} (expected: 40)")

    if embed_dim != 4096 or num_layers != 40:
        print("[DINOv3] ERROR: Dimension mismatch!")
        return

    # Save
    print(f"[DINOv3] Saving to {output_file}...")
    save_file(bf16_state_dict, output_file)

    # Calculate size
    size_bytes = output_file.stat().st_size
    size_gb = size_bytes / (1024**3)
    print(f"[DINOv3] Saved: {size_gb:.2f} GB")
    print(f"[DINOv3] Expected: ~13.4 GB")

    # Cleanup
    del model, state_dict, bf16_state_dict
    gc.collect()

    print("[DINOv3] Done!")


def convert_siglip2(output_dir: Path, source_model: str = "google/siglip2-giant-opt-patch16-384"):
    """Extract SigLIP2 vision encoder and convert to safetensors."""
    from transformers import SiglipModel

    output_path = output_dir / "SigLIP2-G384"
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "model.safetensors"

    if output_file.exists():
        print(f"[SigLIP2] Output already exists: {output_file}")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != "y":
            print("[SigLIP2] Skipping")
            return

    print(f"[SigLIP2] Loading {source_model}...")

    # Load full model (vision + text)
    model = SiglipModel.from_pretrained(
        source_model,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    print("[SigLIP2] Extracting vision encoder...")
    full_state_dict = model.state_dict()

    # Extract only vision_model weights and strip prefix
    vision_state_dict = {}
    for key, tensor in full_state_dict.items():
        if key.startswith("vision_model."):
            # Strip 'vision_model.' prefix
            new_key = key[len("vision_model.") :]
            vision_state_dict[new_key] = tensor.contiguous()

    print(f"[SigLIP2] Extracted {len(vision_state_dict)} tensors from vision encoder")

    # Verify expected structure
    expected_keys = [
        "embeddings.patch_embedding.weight",
        "embeddings.patch_embedding.bias",
        "embeddings.position_embedding.weight",
        "encoder.layers.0.self_attn.k_proj.weight",
        "encoder.layers.0.self_attn.q_proj.weight",
        "encoder.layers.0.self_attn.v_proj.weight",
        "encoder.layers.0.self_attn.out_proj.weight",
        "encoder.layers.0.mlp.fc1.weight",
        "encoder.layers.0.mlp.fc2.weight",
        "post_layernorm.weight",
        "post_layernorm.bias",
        "head.attention.in_proj_weight",
        "head.attention.out_proj.weight",
        "head.mlp.fc1.weight",
        "head.mlp.fc2.weight",
        "head.layernorm.weight",
        "head.probe",
    ]

    missing = [k for k in expected_keys if k not in vision_state_dict]
    if missing:
        print(f"[SigLIP2] Warning: Missing expected keys: {missing[:5]}...")  # Show first 5

    # Check dimensions
    print(f"[SigLIP2] Verifying dimensions...")
    if "embeddings.patch_embedding.weight" in vision_state_dict:
        embed_dim = vision_state_dict["embeddings.patch_embedding.weight"].shape[0]
    else:
        embed_dim = "unknown"

    num_layers = sum(
        1 for k in vision_state_dict if "encoder.layers." in k and ".self_attn.k_proj.weight" in k
    )
    print(f"  - Hidden dim: {embed_dim} (expected: 1536)")
    print(f"  - Num layers: {num_layers} (expected: 40)")

    if embed_dim != 1536 or num_layers != 40:
        print("[SigLIP2] WARNING: Dimension mismatch - check model version")

    # Save (keep FP32 like DiffSynth's version)
    print(f"[SigLIP2] Saving to {output_file}...")
    save_file(vision_state_dict, output_file)

    # Calculate size
    size_bytes = output_file.stat().st_size
    size_gb = size_bytes / (1024**3)
    print(f"[SigLIP2] Saved: {size_gb:.2f} GB")
    print(f"[SigLIP2] Expected: ~4.65 GB")

    # Cleanup
    del model, full_state_dict, vision_state_dict
    gc.collect()

    print("[SigLIP2] Done!")


def verify_outputs(output_dir: Path):
    """Verify converted models can be loaded."""
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    dinov3_path = output_dir / "DINOv3-7B" / "model.safetensors"
    siglip2_path = output_dir / "SigLIP2-G384" / "model.safetensors"

    if dinov3_path.exists():
        print(f"\n[DINOv3] Testing load from {dinov3_path}...")
        try:
            from safetensors.torch import load_file

            state_dict = load_file(dinov3_path)
            print(f"  - Loaded {len(state_dict)} tensors")
            print(f"  - Sample key: {list(state_dict.keys())[0]}")
            print(f"  - Sample dtype: {state_dict[list(state_dict.keys())[0]].dtype}")
            del state_dict
            gc.collect()
            print("  - OK")
        except Exception as e:
            print(f"  - FAILED: {e}")

    if siglip2_path.exists():
        print(f"\n[SigLIP2] Testing load from {siglip2_path}...")
        try:
            from safetensors.torch import load_file

            state_dict = load_file(siglip2_path)
            print(f"  - Loaded {len(state_dict)} tensors")
            print(f"  - Sample key: {list(state_dict.keys())[0]}")
            print(f"  - Sample dtype: {state_dict[list(state_dict.keys())[0]].dtype}")
            del state_dict
            gc.collect()
            print("  - OK")
        except Exception as e:
            print(f"  - FAILED: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert image encoders to DiffSynth-compatible format"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory (will create DINOv3-7B/ and SigLIP2-G384/ subdirs)",
    )
    parser.add_argument(
        "--dinov3-only",
        action="store_true",
        help="Only convert DINOv3",
    )
    parser.add_argument(
        "--siglip2-only",
        action="store_true",
        help="Only convert SigLIP2",
    )
    parser.add_argument(
        "--dinov3-source",
        type=str,
        default="facebook/dinov3-vit7b16-pretrain-lvd1689m",
        help="DINOv3 source model ID",
    )
    parser.add_argument(
        "--siglip2-source",
        type=str,
        default="google/siglip2-giant-opt-patch16-384",
        help="SigLIP2 source model ID",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip verification step",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Image Encoder Conversion for DiffSynth/Image2LoRA")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()

    convert_dinov3_flag = not args.siglip2_only
    convert_siglip2_flag = not args.dinov3_only

    if convert_dinov3_flag:
        convert_dinov3(output_dir, args.dinov3_source)
        print()

    if convert_siglip2_flag:
        convert_siglip2(output_dir, args.siglip2_source)
        print()

    if not args.skip_verify:
        verify_outputs(output_dir)

    print("\n" + "=" * 60)
    print("USAGE WITH DIFFSYNTH")
    print("=" * 60)
    print(f"""
To use these encoders with DiffSynth Image2LoRA:

    from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig

    pipe = ZImagePipeline.from_pretrained(
        model_configs=[
            # ... other configs ...
            ModelConfig("{output_dir}/SigLIP2-G384/model.safetensors"),
            ModelConfig("{output_dir}/DINOv3-7B/model.safetensors"),
        ],
    )
""")


if __name__ == "__main__":
    main()
