#!/usr/bin/env python3
"""
Qwen3-VL Integration Proof-of-Concept for Z-Image

This script uses the core VLEmbeddingExtractor from src/llm_dit/vl/ and
respects config.toml settings. It provides additional exploration features
like projection layers and Z-Image compatibility testing.

Usage:
    # Extract vision features using config.toml
    uv run experiments/qwen3_vl_poc.py \
        --config config.toml --profile rtx4090 \
        --image test.jpg \
        --extract-only

    # Test Z-Image compatibility
    uv run experiments/qwen3_vl_poc.py \
        --config config.toml \
        --image test.jpg

    # Generate an image from vision embeddings
    uv run experiments/qwen3_vl_poc.py \
        --config config.toml \
        --image reference.jpg \
        --generate \
        --output generated.png \
        --seed 42

    # Load pre-saved embeddings (skip VL extraction)
    uv run experiments/qwen3_vl_poc.py \
        --config config.toml \
        --load-embeddings embeddings.pt \
        --generate

Note: This script uses the same CLI flags as the main tools for consistency.
Config.toml is the single source of truth.
"""

import argparse
import gc
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class ProjectionLayer(nn.Module):
    """Simple projection layer for 8B -> Z-Image compatibility (experimental)."""

    def __init__(self, source_dim: int = 4096, target_dim: int = 2560):
        super().__init__()
        self.projection = nn.Linear(source_dim, target_dim)
        nn.init.orthogonal_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


def test_z_image_compatibility(
    embeddings: torch.Tensor,
    runtime_config,
    use_projection: bool = False,
    device: str = "cuda",
    generate: bool = False,
    output_path: str = None,
    seed: int = 42,
):
    """Test if vision embeddings can be fed to Z-Image DiT."""
    print(f"\nTesting Z-Image compatibility...")

    current_dim = embeddings.shape[-1]
    target_dim = 2560

    if current_dim != target_dim:
        if use_projection:
            print(f"Applying projection: {current_dim} -> {target_dim}")
            projection = ProjectionLayer(current_dim, target_dim).to(device, dtype=embeddings.dtype)
            embeddings = projection(embeddings)
        else:
            print(f"Dimension mismatch: {current_dim} != {target_dim}")
            print("Use --use-projection for 8B model")
            return None, None

    print(f"Vision embeddings ready for DiT: {embeddings.shape}")

    # Get Z-Image model path from config
    z_image_path = runtime_config.model_path if runtime_config else None
    if not z_image_path:
        print("No model_path in config. Provide --model-path or set in config.toml")
        return embeddings, None

    try:
        from llm_dit import ZImagePipeline

        print(f"Loading Z-Image generator from {z_image_path}...")
        pipe = ZImagePipeline.from_pretrained_generator_only(
            z_image_path,
            dit_device=device,
            vae_device=device,
            enable_cpu_offload=True,
        )

        print(f"Embeddings shape for DiT: {embeddings.shape}")
        print("Shape compatible with Z-Image DiT!")

        if generate:
            print(f"\nGenerating image from vision embeddings (seed={seed})...")
            generator = torch.Generator(device="cpu").manual_seed(seed)

            # Use config for dimensions
            height = runtime_config.height if runtime_config and runtime_config.height else 256
            width = runtime_config.width if runtime_config and runtime_config.width else 256
            steps = runtime_config.num_inference_steps if runtime_config and runtime_config.num_inference_steps else 9

            image = pipe.generate_from_embeddings(
                prompt_embeds=embeddings,
                height=height,
                width=width,
                num_inference_steps=steps,
                generator=generator,
            )

            if output_path:
                image.save(output_path)
                print(f"Image saved to: {output_path}")
            else:
                default_path = "/tmp/claude/qwen3_vl_generated.png"
                Path(default_path).parent.mkdir(parents=True, exist_ok=True)
                image.save(default_path)
                print(f"Image saved to: {default_path}")

            return embeddings, image

        return embeddings, pipe

    except ImportError as e:
        print(f"Could not load Z-Image: {e}")
        return embeddings, None


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL Integration PoC using core VLEmbeddingExtractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config file options (same as main CLI)
    parser.add_argument("--config", type=str, help="Path to config.toml file")
    parser.add_argument("--profile", type=str, default="default", help="Config profile")

    # VL options (same flags as main CLI)
    parser.add_argument("--vl-model-path", type=str, help="Path to Qwen3-VL model")
    parser.add_argument("--vl-device", type=str, help="Device for VL model")
    parser.add_argument("--vl-hidden-layer", type=int, help="Hidden layer to extract")

    # Z-Image options
    parser.add_argument("--model-path", type=str, help="Path to Z-Image model")

    # Script-specific options
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--extract-only", action="store_true", help="Only extract features")
    parser.add_argument("--use-projection", action="store_true", help="Use projection (8B model)")
    parser.add_argument("--generate", action="store_true", help="Generate an image")
    parser.add_argument("--output", type=str, help="Path to save generated image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-embeddings", type=str, help="Save embeddings to .pt file")
    parser.add_argument("--load-embeddings", type=str, help="Load embeddings from .pt file")
    parser.add_argument("--text", type=str, help="Optional text with image")

    args = parser.parse_args()

    # Load config using the standard config loading
    from llm_dit.cli import load_runtime_config

    config_args = argparse.Namespace(
        config=args.config,
        profile=args.profile,
        vl_model_path=args.vl_model_path,
        vl_device=args.vl_device,
        vl_hidden_layer=args.vl_hidden_layer,
        vl_auto_unload=True,
        vl_alpha=0.3,
        vl_blend_mode="linear",
        model_path=args.model_path,
        encoder_device=None,
        dit_device=None,
        vae_device=None,
        templates_dir=None,
    )

    # Add other required fields
    for field in [
        'torch_dtype', 'quantization', 'encoder_cpu_offload', 'encoder_max_length',
        'pipeline_torch_dtype', 'pipeline_device', 'enable_model_cpu_offload',
        'enable_sequential_cpu_offload', 'width', 'height', 'num_inference_steps',
        'guidance_scale', 'enable_thinking', 'default_template', 'shift',
        'flash_attn', 'compile', 'cpu_offload', 'attention_backend',
        'use_custom_scheduler', 'tiled_vae', 'tile_size', 'tile_overlap',
        'embedding_cache', 'cache_size', 'long_prompt_mode', 'hidden_layer',
        'api_url', 'api_model', 'use_api_encoder', 'lora_paths', 'lora_scales',
        'rewriter_use_api', 'rewriter_api_url', 'rewriter_api_model',
        'rewriter_temperature', 'rewriter_top_p', 'rewriter_top_k',
        'rewriter_min_p', 'rewriter_presence_penalty', 'rewriter_max_tokens',
        'debug',
    ]:
        if not hasattr(config_args, field):
            setattr(config_args, field, None)

    try:
        runtime_config = load_runtime_config(config_args)
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        runtime_config = None

    # Mode 1: Load pre-saved embeddings
    if args.load_embeddings:
        print(f"Loading embeddings from {args.load_embeddings}...")
        saved = torch.load(args.load_embeddings, weights_only=True)
        embeddings = saved["embeddings"]
        device = args.vl_device or (runtime_config.vl_device if runtime_config else "cuda")
        embeddings = embeddings.to(device)
        print(f"Loaded embeddings shape: {embeddings.shape}")

        test_z_image_compatibility(
            embeddings,
            runtime_config,
            use_projection=args.use_projection,
            device=device,
            generate=args.generate,
            output_path=args.output,
            seed=args.seed,
        )
        print("\nDone!")
        return

    # Mode 2: Extract from image using core VLEmbeddingExtractor
    if not args.image:
        print("Error: --image required (or use --load-embeddings)")
        sys.exit(1)

    if not Path(args.image).exists():
        print(f"Image not found: {args.image}")
        sys.exit(1)

    # Get VL model path from config or CLI
    vl_model_path = args.vl_model_path
    if not vl_model_path and runtime_config:
        vl_model_path = runtime_config.vl_model_path

    if not vl_model_path:
        # Try common locations
        for candidate in [
            Path.home() / "Storage" / "Qwen3-VL-4B-Instruct",
            Path.home() / "models" / "Qwen3-VL-4B-Instruct",
        ]:
            if candidate.exists():
                vl_model_path = str(candidate)
                break

    if not vl_model_path:
        print("Could not find Qwen3-VL model. Set vl.model_path in config.toml")
        sys.exit(1)

    # Get device from config or CLI
    device = args.vl_device
    if not device and runtime_config:
        device = runtime_config.vl_device
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get hidden layer from config or CLI
    hidden_layer = args.vl_hidden_layer
    if hidden_layer is None and runtime_config:
        hidden_layer = runtime_config.vl_hidden_layer
    if hidden_layer is None:
        hidden_layer = -2

    # Use core VLEmbeddingExtractor
    from PIL import Image
    from llm_dit.vl import VLEmbeddingExtractor

    print(f"Loading VLEmbeddingExtractor from {vl_model_path}...")
    vl_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    extractor = VLEmbeddingExtractor.from_pretrained(
        vl_model_path,
        device=device,
        torch_dtype=vl_dtype,
    )

    # Extract embeddings
    image = Image.open(args.image).convert("RGB")
    print(f"Image size: {image.size}")

    result = extractor.extract(
        image=image,
        text=args.text,
        hidden_layer=hidden_layer,
        scale_to_text=True,
    )

    embeddings = result.embeddings
    print(f"Extracted embeddings: {embeddings.shape}")
    print(f"Stats: mean={embeddings.mean():.4f}, std={embeddings.std():.4f}")

    # Save if requested
    if args.save_embeddings:
        torch.save({
            "embeddings": embeddings.cpu(),
            "shape": embeddings.shape,
            "source_image": str(args.image),
            "hidden_layer": result.hidden_layer,
        }, args.save_embeddings)
        print(f"Embeddings saved to {args.save_embeddings}")

    # Offload VL model before loading Z-Image
    print("Offloading VL model to free memory...")
    del extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Test Z-Image compatibility
    if not args.extract_only:
        test_z_image_compatibility(
            embeddings,
            runtime_config,
            use_projection=args.use_projection,
            device=device,
            generate=args.generate,
            output_path=args.output,
            seed=args.seed,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
