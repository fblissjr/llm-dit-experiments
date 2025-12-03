#!/usr/bin/env python3
"""
End-to-end Z-Image generation script.

Usage:
    # With config file (recommended)
    uv run scripts/generate.py --config config.toml "A cat sleeping in sunlight"

    # With config profile
    uv run scripts/generate.py --config config.toml --profile low_vram "A cat"

    # Basic generation (no config)
    uv run scripts/generate.py --model-path /path/to/z-image "A cat sleeping in sunlight"

    # With template
    uv run scripts/generate.py --model-path /path/to/z-image --template photorealistic "A cat"

    # With seed
    uv run scripts/generate.py --model-path /path/to/z-image --seed 42 "A cat"

    # Encoder-only mode (for experiments or distributed inference)
    uv run scripts/generate.py --model-path /path/to/z-image --encoder-only "A cat"

    # Save embeddings for distributed inference
    uv run scripts/generate.py --model-path /path/to/z-image --save-embeddings emb.safetensors "A cat"

    # DISTRIBUTED: Encode via remote API (Mac), generate locally (CUDA)
    uv run scripts/generate.py --api-url http://mac-ip:8080 --model-path /path/to/z-image "A cat"
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate images with Z-Image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument(
        "prompt",
        type=str,
        help="Text prompt for image generation",
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to TOML config file",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="default",
        help="Config profile to use (default: default)",
    )

    # Model path (can be overridden by config)
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to Z-Image model or HuggingFace ID",
    )

    # Optional
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output image path (default: output.png)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height (default: 1024, must be divisible by 16)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width (default: 1024, must be divisible by 16)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=9,
        help="Number of inference steps (default: 9 for turbo)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Template name to use for encoding",
    )
    parser.add_argument(
        "--templates-dir",
        type=str,
        default=None,
        help="Path to templates directory",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable thinking tags in prompt",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=0.0,
        help="CFG scale (default: 0.0, not needed for Z-Image-Turbo)",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Negative prompt for CFG",
    )

    # Mode flags
    parser.add_argument(
        "--encoder-only",
        action="store_true",
        help="Only run encoder (for experiments)",
    )
    parser.add_argument(
        "--save-embeddings",
        type=str,
        default=None,
        help="Save embeddings to file (for distributed inference)",
    )
    parser.add_argument(
        "--load-embeddings",
        type=str,
        default=None,
        help="Load embeddings from file (skip encoding)",
    )

    # API backend (distributed inference)
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="Use remote API for encoding (e.g., http://mac-ip:8080)",
    )
    parser.add_argument(
        "--api-model",
        type=str,
        default="Qwen3-4B-mxfp4-mlx",
        help="Model ID for API backend (default: Qwen3-4B-mxfp4-mlx)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    # Load config if provided
    config = None
    if args.config:
        from llm_dit.config import Config
        config = Config.from_toml(args.config, args.profile)
        logger.info(f"Loaded config profile: {args.profile}")

    # Resolve model path (CLI > config)
    model_path = args.model_path
    if model_path is None and config:
        model_path = config.model_path
    if model_path is None:
        logger.error("No model path specified. Use --model-path or --config.")
        return 1

    # Find templates directory
    templates_dir = args.templates_dir
    if templates_dir is None and config and config.templates_dir:
        templates_dir = config.templates_dir
    if templates_dir is None:
        # Try default location relative to this script
        default_templates = Path(__file__).parent.parent / "templates" / "z_image"
        if default_templates.exists():
            templates_dir = str(default_templates)
            logger.info(f"Using default templates: {templates_dir}")

    # Set up generator
    generator = None
    if args.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        logger.info(f"Using seed: {args.seed}")

    if args.encoder_only or args.save_embeddings:
        # Encoder-only mode for experiments or distributed inference
        logger.info("Running in encoder-only mode")

        from llm_dit.encoders import ZImageTextEncoder

        logger.info(f"Loading encoder from {model_path}...")
        start = time.time()
        encoder = ZImageTextEncoder.from_pretrained(
            model_path,
            templates_dir=templates_dir,
        )
        load_time = time.time() - start
        logger.info(f"Encoder loaded in {load_time:.1f}s")

        # Encode prompt
        logger.info(f"Encoding prompt: {args.prompt[:50]}...")
        start = time.time()
        output = encoder.encode(
            args.prompt,
            template=args.template,
            enable_thinking=not args.no_thinking,
        )
        encode_time = time.time() - start

        embeds = output.embeddings[0]
        logger.info(f"Encoding complete in {encode_time:.3f}s")
        logger.info(f"  - Sequence length: {embeds.shape[0]}")
        logger.info(f"  - Embedding dim: {embeds.shape[1]}")
        logger.info(f"  - Device: {embeds.device}")
        logger.info(f"  - Dtype: {embeds.dtype}")

        # Print embedding stats
        logger.info(f"  - Mean: {embeds.mean().item():.4f}")
        logger.info(f"  - Std: {embeds.std().item():.4f}")
        logger.info(f"  - Min: {embeds.min().item():.4f}")
        logger.info(f"  - Max: {embeds.max().item():.4f}")

        # Save embeddings if requested
        if args.save_embeddings:
            from llm_dit.distributed import save_embeddings
            save_path = save_embeddings(
                embeddings=embeds,
                path=args.save_embeddings,
                prompt=args.prompt,
                model_path=model_path,
                template=args.template,
                enable_thinking=not args.no_thinking,
                encoder_device=str(encoder.device),
            )
            logger.info(f"Embeddings saved to: {save_path}")

        return 0

    # Get generation params from config or CLI
    height = args.height
    width = args.width
    steps = args.steps
    guidance_scale = args.guidance_scale
    enable_thinking = not args.no_thinking

    if config:
        if height == 1024:  # default
            height = config.generation.height
        if width == 1024:  # default
            width = config.generation.width
        if steps == 9:  # default
            steps = config.generation.num_inference_steps
        if guidance_scale == 0.0:  # default
            guidance_scale = config.generation.guidance_scale
        if not args.no_thinking:
            enable_thinking = config.generation.enable_thinking

    # Check if using API for encoding (distributed inference - encode remote, generate local)
    if args.api_url:
        logger.info("Running in distributed mode (API encoding + local generation)")
        logger.info(f"Using API backend: {args.api_url}")

        from llm_dit.backends.api import APIBackend, APIBackendConfig
        from llm_dit.encoders import ZImageTextEncoder
        from llm_dit.pipelines import ZImagePipeline
        from llm_dit.templates import TemplateRegistry

        # Create API backend for encoding
        api_config = APIBackendConfig(
            base_url=args.api_url,
            model_id=args.api_model,
            encoding_format="base64",
        )
        backend = APIBackend(api_config)

        # Load templates locally
        templates = None
        if templates_dir:
            templates = TemplateRegistry.from_directory(templates_dir)
            logger.info(f"Loaded {len(templates)} templates")

        # Create encoder with API backend
        encoder = ZImageTextEncoder(
            backend=backend,
            templates=templates,
        )

        # Encode via API
        logger.info(f"Encoding prompt via API: {args.prompt[:50]}...")
        start = time.time()
        output = encoder.encode(
            args.prompt,
            template=args.template,
            enable_thinking=enable_thinking,
        )
        encode_time = time.time() - start
        embeds = output.embeddings[0]
        logger.info(f"Encoding complete in {encode_time:.3f}s")
        logger.info(f"  Shape: {embeds.shape}")

        # Load generator-only pipeline (no LLM)
        logger.info(f"Loading generator from {model_path}...")
        start = time.time()

        try:
            pipe = ZImagePipeline.from_pretrained_generator_only(
                model_path,
                torch_dtype=torch.bfloat16,
            )
        except ImportError as e:
            logger.error(f"Missing diffusers components: {e}")
            return 1
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            return 1

        load_time = time.time() - start
        logger.info(f"Generator loaded in {load_time:.1f}s")

        # Progress callback
        def progress_callback(step: int, total: int, latents: torch.Tensor):
            logger.info(f"Step {step + 1}/{total}")

        # Generate from embeddings
        logger.info(f"Generating {width}x{height} image...")

        start = time.time()
        image = pipe.generate_from_embeddings(
            embeds,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback=progress_callback if args.verbose else None,
        )
        gen_time = time.time() - start

        # Save
        output_path = Path(args.output)
        image.save(output_path)
        logger.info(f"Image saved to {output_path}")
        logger.info(f"Total time: encode={encode_time:.1f}s + generate={gen_time:.1f}s")

        return 0

    # Check if loading embeddings (distributed inference - CUDA side)
    if args.load_embeddings:
        logger.info("Running in distributed mode (generator only)")
        logger.info(f"Loading embeddings from {args.load_embeddings}")

        from llm_dit.distributed import load_embeddings
        from llm_dit.pipelines import ZImagePipeline

        # Load pre-computed embeddings
        emb_file = load_embeddings(args.load_embeddings)
        logger.info(f"Loaded embeddings: {emb_file.metadata.prompt[:50]}...")
        logger.info(f"  Shape: {emb_file.embeddings.shape}")
        logger.info(f"  Original device: {emb_file.metadata.encoder_device}")

        # Load generator-only pipeline (no LLM)
        logger.info(f"Loading generator from {model_path}...")
        start = time.time()

        try:
            pipe = ZImagePipeline.from_pretrained_generator_only(
                model_path,
                torch_dtype=torch.bfloat16,
            )
        except ImportError as e:
            logger.error(f"Missing diffusers components: {e}")
            return 1
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            return 1

        load_time = time.time() - start
        logger.info(f"Generator loaded in {load_time:.1f}s")

        # Progress callback
        def progress_callback(step: int, total: int, latents: torch.Tensor):
            logger.info(f"Step {step + 1}/{total}")

        # Generate from embeddings
        logger.info(f"Generating {width}x{height} image...")

        start = time.time()
        image = pipe.generate_from_embeddings(
            emb_file.embeddings,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback=progress_callback if args.verbose else None,
        )
        gen_time = time.time() - start

        # Save
        output_path = Path(args.output)
        image.save(output_path)
        logger.info(f"Image saved to {output_path}")
        logger.info(f"Generation time: {gen_time:.1f}s")
        logger.info(f"Original prompt: {emb_file.metadata.prompt}")

        return 0

    # Full generation mode (encode + generate)
    logger.info("Running full generation")

    from llm_dit.pipelines import ZImagePipeline

    logger.info(f"Loading pipeline from {model_path}...")
    start = time.time()

    try:
        pipe = ZImagePipeline.from_pretrained(
            model_path,
            templates_dir=templates_dir,
            torch_dtype=torch.bfloat16,
        )
    except ImportError as e:
        logger.error(f"Missing diffusers components: {e}")
        logger.error("Full generation requires diffusers with Z-Image support.")
        logger.error("Try encoder-only mode with --encoder-only")
        return 1
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        return 1

    load_time = time.time() - start
    logger.info(f"Pipeline loaded in {load_time:.1f}s")

    # Progress callback
    def progress_callback(step: int, total: int, latents: torch.Tensor):
        logger.info(f"Step {step + 1}/{total}")

    # Generate
    logger.info(f"Generating {width}x{height} image...")
    logger.info(f"Prompt: {args.prompt}")
    if args.template:
        logger.info(f"Template: {args.template}")

    start = time.time()
    image = pipe(
        args.prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        negative_prompt=args.negative_prompt,
        generator=generator,
        template=args.template,
        enable_thinking=enable_thinking,
        callback=progress_callback if args.verbose else None,
    )
    gen_time = time.time() - start

    # Save
    output_path = Path(args.output)
    image.save(output_path)
    logger.info(f"Image saved to {output_path}")
    logger.info(f"Generation time: {gen_time:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
