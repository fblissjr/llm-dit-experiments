"""
Test reference VideoDecoder directly without full pipeline dependencies.

This script loads the reference VideoDecoder and runs it with the same
random latents to compare with our implementation.

Last Updated: 2026-01-20
"""

import sys
from pathlib import Path

# Add coderef to path (only ltx-core, not pipelines)
coderef_root = Path(__file__).parent.parent / "coderef" / "LTX-2" / "packages"
sys.path.insert(0, str(coderef_root / "ltx-core" / "src"))

import torch
from safetensors.torch import load_file

# Import only video VAE components (no audio)
from ltx_core.model.video_vae.video_vae import VideoDecoder
from ltx_core.model.video_vae.enums import NormLayerType, PaddingModeType


def print_stats(name: str, tensor: torch.Tensor) -> None:
    """Print tensor statistics."""
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}, dtype: {tensor.dtype}")
    print(f"  Mean: {tensor.mean():.4f}, Std: {tensor.std():.4f}")
    print(f"  Range: [{tensor.min():.4f}, {tensor.max():.4f}]")


def create_reference_decoder(dtype: torch.dtype = torch.bfloat16):
    """Create reference decoder with LTX-2 architecture."""
    # LTX-2 VAE decoder architecture based on checkpoint
    # The architecture uses named blocks that get reversed
    decoder_blocks = [
        # First res_x block (becomes mid_block after reverse)
        ("res_x", {"num_layers": 5}),
        # Alternating compress_all + res_x
        ("compress_all", {"residual": True, "multiplier": 2}),
        ("res_x", {"num_layers": 5}),
        ("compress_all", {"residual": True, "multiplier": 2}),
        ("res_x", {"num_layers": 5}),
        ("compress_all", {"residual": True, "multiplier": 2}),
        ("res_x", {"num_layers": 5}),
    ]

    decoder = VideoDecoder(
        convolution_dimensions=3,
        in_channels=128,
        out_channels=3,
        decoder_blocks=decoder_blocks,
        patch_size=4,
        norm_layer=NormLayerType.PIXEL_NORM,
        causal=False,
        timestep_conditioning=False,
        decoder_spatial_padding_mode=PaddingModeType.REFLECT,
    )

    return decoder.to(dtype)


def load_weights(decoder, checkpoint_path: str, dtype: torch.dtype):
    """Load weights from checkpoint into decoder."""
    import re
    ckpt = load_file(checkpoint_path)

    # Map keys from checkpoint to reference decoder structure
    # Checkpoint: decoder.mid_block.resnets.N.* → up_blocks.0.res_blocks.N.*
    # Checkpoint: decoder.up_blocks.K.upsamplers.0.* → up_blocks.{2K+1}.*
    # Checkpoint: decoder.up_blocks.K.resnets.N.* → up_blocks.{2K+2}.res_blocks.N.*
    mapped = {}
    for k, v in ckpt.items():
        if k == "latents_mean":
            mapped["per_channel_statistics.mean-of-means"] = v.to(dtype)
        elif k == "latents_std":
            mapped["per_channel_statistics.std-of-means"] = v.to(dtype)
        elif k.startswith("decoder."):
            new_key = k[8:]  # Remove 'decoder.'

            # Map mid_block.resnets → up_blocks.0.res_blocks
            if new_key.startswith("mid_block.resnets."):
                new_key = new_key.replace("mid_block.resnets.", "up_blocks.0.res_blocks.")
            else:
                # Map up_blocks.K.upsamplers/resnets
                match = re.match(r"up_blocks\.(\d+)\.(upsamplers|resnets)\.", new_key)
                if match:
                    block_idx = int(match.group(1))
                    block_type = match.group(2)

                    if block_type == "upsamplers":
                        new_idx = 2 * block_idx + 1
                        new_key = re.sub(
                            r"up_blocks\.\d+\.upsamplers\.0\.",
                            f"up_blocks.{new_idx}.",
                            new_key
                        )
                    else:  # resnets
                        new_idx = 2 * block_idx + 2
                        new_key = re.sub(
                            r"up_blocks\.\d+\.resnets\.",
                            f"up_blocks.{new_idx}.res_blocks.",
                            new_key
                        )

            mapped[new_key] = v.to(dtype)

    result = decoder.load_state_dict(mapped, strict=False)
    print(f"Load result: {len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected")
    if result.missing_keys:
        print(f"  First 3 missing: {result.missing_keys[:3]}")
    if result.unexpected_keys:
        print(f"  First 3 unexpected: {result.unexpected_keys[:3]}")

    return decoder


def main():
    print("=" * 80)
    print("REFERENCE VAE DECODER TEST")
    print("=" * 80)

    device = "cuda"
    dtype = torch.bfloat16
    checkpoint_path = "models/LTX-2/vae/diffusion_pytorch_model.safetensors"

    # Generate fixed random latents
    torch.manual_seed(42)
    latents = torch.randn(1, 128, 2, 8, 12, device=device, dtype=dtype)
    print_stats("\nTest latents", latents)

    # Create and load reference decoder
    print("\n--- Creating Reference Decoder ---")
    try:
        ref_decoder = create_reference_decoder(dtype)
        ref_decoder = load_weights(ref_decoder, checkpoint_path, dtype)
        ref_decoder = ref_decoder.to(device)

        print(f"\nReference decoder structure:")
        print(f"  up_blocks: {len(ref_decoder.up_blocks)}")
        for i, block in enumerate(ref_decoder.up_blocks):
            print(f"    [{i}] {type(block).__name__}")

        # Run forward pass
        print("\n--- Running Reference Forward Pass ---")
        with torch.no_grad():
            ref_output = ref_decoder(latents.clone())
        print_stats("Reference output", ref_output)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Now run our decoder for comparison
    print("\n--- Running Our Decoder ---")
    from llm_dit.models.ltx2.vae import load_ltx2_vae_decoder

    our_decoder = load_ltx2_vae_decoder("models/LTX-2/vae", dtype=dtype, device="cpu")
    our_decoder = our_decoder.to(device)

    with torch.no_grad():
        our_output = our_decoder(latents.clone())
    print_stats("Our output", our_output)

    # Compare
    print("\n--- Comparison ---")
    diff = (ref_output - our_output).abs()
    print_stats("Absolute difference", diff)
    print(f"Max diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")

    if diff.max() < 0.01:
        print("\n✓ Outputs match!")
    else:
        print("\n✗ Outputs differ - investigating...")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
