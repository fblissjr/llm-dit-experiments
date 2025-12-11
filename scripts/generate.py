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

    # With LoRA
    uv run scripts/generate.py --model-path /path/to/z-image --lora /path/to/lora.safetensors:0.8 "A cat"

    # With custom scheduler shift
    uv run scripts/generate.py --model-path /path/to/z-image --shift 5.0 "A cat"
"""

import logging
import sys
import time
from pathlib import Path

import torch

from llm_dit.cli import create_base_parser, load_runtime_config, setup_logging


def main():
    # Create parser with generation args
    parser = create_base_parser(
        description="Generate images with Z-Image",
        include_generation_args=True,
        include_server_args=False,
    )

    # Add generate-specific arguments
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",  # Optional when using --load-embeddings
        default=None,
        help="Text prompt for image generation (optional if using --load-embeddings)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output image path (default: output.png)",
    )
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

    args = parser.parse_args()

    # Load unified config
    config = load_runtime_config(args)
    setup_logging(config)

    logger = logging.getLogger(__name__)

    # Validate model path
    if config.model_path == "" and not args.load_embeddings:
        logger.error("No model path specified. Use --model-path or --config.")
        return 1

    # Find templates directory
    templates_dir = config.templates_dir
    if templates_dir is None:
        # Try default location relative to this script
        default_templates = Path(__file__).parent.parent / "templates" / "z_image"
        if default_templates.exists():
            templates_dir = str(default_templates)
            logger.info(f"Using default templates: {templates_dir}")

    # Set up generator
    generator = None
    seed = getattr(args, 'seed', None)
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        logger.info(f"Using seed: {seed}")

    if args.encoder_only or args.save_embeddings:
        # Encoder-only mode for experiments or distributed inference
        logger.info("Running in encoder-only mode")

        from llm_dit.startup import PipelineLoader

        loader = PipelineLoader(config)
        result = loader.load_encoder()
        encoder = result.encoder

        # Encode prompt
        logger.info(f"Encoding prompt: {args.prompt[:50]}...")
        start = time.time()
        output = encoder.encode(
            args.prompt,
            template=config.default_template,
            system_prompt=config.system_prompt,
            thinking_content=config.thinking_content,
            assistant_content=config.assistant_content,
            enable_thinking=config.enable_thinking,
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
                model_path=config.model_path,
                template=config.default_template,
                enable_thinking=config.enable_thinking,
                encoder_device=str(encoder.device),
            )
            logger.info(f"Embeddings saved to: {save_path}")

        return 0

    # Check if using API for encoding (distributed inference - encode remote, generate local)
    if config.api_url:
        logger.info("Running in distributed mode (API encoding + local generation)")

        from llm_dit.startup import PipelineLoader

        loader = PipelineLoader(config)

        try:
            result = loader.load_api_pipeline()
            pipe = result.pipeline
            encoder = result.encoder
        except ImportError as e:
            logger.error(f"Missing diffusers components: {e}")
            return 1
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            return 1

        # Encode via API
        logger.info(f"Encoding prompt via API: {args.prompt[:50]}...")
        start = time.time()
        output = encoder.encode(
            args.prompt,
            template=config.default_template,
            system_prompt=config.system_prompt,
            thinking_content=config.thinking_content,
            assistant_content=config.assistant_content,
            enable_thinking=config.enable_thinking,
        )
        encode_time = time.time() - start
        embeds = output.embeddings[0]
        logger.info(f"Encoding complete in {encode_time:.3f}s")
        logger.info(f"  Shape: {embeds.shape}")

        # Progress callback
        def progress_callback(step: int, total: int, latents: torch.Tensor):
            logger.info(f"Step {step + 1}/{total}")

        # Generate from embeddings
        logger.info(f"Generating {config.width}x{config.height} image...")

        start = time.time()
        image = pipe.generate_from_embeddings(
            embeds,
            height=config.height,
            width=config.width,
            num_inference_steps=config.steps,
            guidance_scale=config.guidance_scale,
            generator=generator,
            callback=progress_callback if config.verbose else None,
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
        logger.info("Running in embeddings mode (skip text encoding)")
        logger.info(f"Loading embeddings from {args.load_embeddings}")

        # Determine file format and load embeddings
        emb_path = Path(args.load_embeddings)
        if emb_path.suffix == ".pt":
            # Simple PyTorch format (from Qwen3-VL or other sources)
            saved = torch.load(args.load_embeddings, weights_only=True)
            embeddings = saved["embeddings"]
            source_info = saved.get("source_image", "unknown")
            logger.info(f"Loaded embeddings: shape={embeddings.shape}")
            logger.info(f"  Source: {source_info}")
        else:
            # Safetensors format (from distributed module)
            from llm_dit.distributed import load_embeddings
            emb_file = load_embeddings(args.load_embeddings)
            embeddings = emb_file.embeddings
            source_info = emb_file.metadata.prompt[:50] if emb_file.metadata.prompt else "unknown"
            logger.info(f"Loaded embeddings: shape={embeddings.shape}")
            logger.info(f"  Source: {source_info}...")
            logger.info(f"  Original device: {emb_file.metadata.encoder_device}")

        # Load pipeline using optimized PipelineLoader (same as full generation)
        # This uses the correct device placement from config, avoids the OOM from generator_only
        from llm_dit.startup import PipelineLoader

        try:
            loader = PipelineLoader(config)
            result = loader.load_pipeline()
            pipe = result.pipeline
        except ImportError as e:
            logger.error(f"Missing diffusers components: {e}")
            return 1
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            return 1

        # Progress callback
        def progress_callback(step: int, total: int, latents: torch.Tensor):
            logger.info(f"Step {step + 1}/{total}")

        # Generate using prompt_embeds (skips text encoding)
        logger.info(f"Generating {config.width}x{config.height} image from embeddings...")

        start = time.time()
        image = pipe(
            prompt_embeds=embeddings,  # Use pre-computed embeddings
            height=config.height,
            width=config.width,
            num_inference_steps=config.steps,
            guidance_scale=config.guidance_scale,
            generator=generator,
            long_prompt_mode=config.long_prompt_mode,
            callback=progress_callback if config.verbose else None,
        )
        gen_time = time.time() - start

        # Save
        output_path = Path(args.output)
        image.save(output_path)
        logger.info(f"Image saved to {output_path}")
        logger.info(f"Generation time: {gen_time:.1f}s")
        logger.info(f"Embeddings source: {source_info}")

        return 0

    # Full generation mode (encode + generate)
    logger.info("Running full generation")

    from llm_dit.startup import PipelineLoader

    try:
        loader = PipelineLoader(config)
        result = loader.load_pipeline()
        pipe = result.pipeline
    except ImportError as e:
        logger.error(f"Missing diffusers components: {e}")
        logger.error("Full generation requires diffusers with Z-Image support.")
        logger.error("Try encoder-only mode with --encoder-only")
        return 1
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        return 1

    # Progress callback
    def progress_callback(step: int, total: int, latents: torch.Tensor):
        logger.info(f"Step {step + 1}/{total}")

    # Generate
    logger.info(f"Generating {config.width}x{config.height} image...")
    logger.info(f"Prompt: {args.prompt}")
    if config.default_template:
        logger.info(f"Template: {config.default_template}")

    # Get negative prompt from CLI
    negative_prompt = getattr(args, 'negative_prompt', None)

    start = time.time()
    image = pipe(
        args.prompt,
        height=config.height,
        width=config.width,
        num_inference_steps=config.steps,
        guidance_scale=config.guidance_scale,
        negative_prompt=negative_prompt,
        generator=generator,
        template=config.default_template,
        system_prompt=config.system_prompt,
        thinking_content=config.thinking_content,
        assistant_content=config.assistant_content,
        force_think_block=config.enable_thinking,  # enable_thinking maps to force_think_block
        long_prompt_mode=config.long_prompt_mode,
        hidden_layer=config.hidden_layer,
        callback=progress_callback if config.verbose else None,
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
