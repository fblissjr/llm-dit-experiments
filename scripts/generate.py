#!/usr/bin/env python3
"""
End-to-end image generation script.

Supports four model types:
  - Z-Image (zimage): Text-to-image generation (turbo, 8-9 steps)
  - Qwen-Image Layered (qwenimage-layered): Image-to-layers decomposition [legacy]
  - Qwen-Image T2I (qwenimage-t2i): Text-to-image generation (40 steps, FP8)
  - Qwen-Image Edit (qwenimage-edit): Image editing with instructions

Usage:
    # Z-Image (default)
    uv run scripts/generate.py --model-path /path/to/z-image "A cat sleeping in sunlight"

    # Qwen-Image T2I (text-to-image)
    uv run scripts/generate.py --model-type qwenimage-t2i \\
        --qwen-image-2512-model-path /path/to/Qwen-Image-2512 \\
        "A majestic mountain peak at golden hour"

    # Qwen-Image Edit (image editing)
    uv run scripts/generate.py --model-type qwenimage-edit \\
        --qwen-image-edit-model-path /path/to/Qwen-Image-Edit-2511 \\
        --img2img input.jpg \\
        "Make the sky more vibrant"

    # Qwen-Image Layered (legacy - image decomposition)
    uv run scripts/generate.py --model-type qwenimage-layered \\
        --qwen-image-model-path /path/to/Qwen_Qwen-Image-Layered \\
        --img2img input.jpg \\
        "A cheerful child waving under a blue sky"

    # With config file (recommended)
    uv run scripts/generate.py --config config.toml "A cat sleeping in sunlight"

    # With config profile
    uv run scripts/generate.py --config config.toml --profile low_vram "A cat"

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


def run_qwen_image_generation(args, config, logger) -> int:
    """
    Run Qwen-Image generation (decomposition or edit-only mode).

    Args:
        args: Parsed CLI arguments
        config: RuntimeConfig with all settings
        logger: Logger instance

    Returns:
        Exit code (0 for success)
    """
    from PIL import Image

    # Validate model path based on mode
    edit_only = getattr(config, 'qwen_image_edit_only', False)
    if edit_only:
        # Edit-only mode uses edit_model_path
        if not config.qwen_image_edit_model_path:
            logger.error(
                "No Qwen-Image-Edit model path specified. "
                "Use --qwen-image-edit-model-path or set qwen_image.edit_model_path in config."
            )
            return 1
    else:
        # Decompose mode uses model_path
        if not config.qwen_image_model_path:
            logger.error(
                "No Qwen-Image model path specified. "
                "Use --qwen-image-model-path or set qwen_image.model_path in config."
            )
            return 1

        # Decompose mode requires an input image
        if not args.img2img:
            logger.error(
                "Qwen-Image-Layered requires an input image. "
                "Use --img2img /path/to/image.jpg"
            )
            return 1

    # Validate resolution
    resolution = config.qwen_image_resolution
    if resolution not in (640, 1024):
        logger.error(
            f"Qwen-Image only supports 640 or 1024 resolution. Got: {resolution}"
        )
        return 1

    # Branch: Edit-only mode vs Decompose mode
    if edit_only:
        return _run_qwen_image_edit_only(args, config, logger, resolution)
    else:
        return _run_qwen_image_decompose(args, config, logger, resolution)


def _run_qwen_image_edit_only(args, config, logger, resolution: int) -> int:
    """Run Qwen-Image-Edit in standalone mode (text-to-image or image editing)."""
    from PIL import Image

    logger.info("=" * 60)
    logger.info("Qwen-Image-Edit (Standalone)")
    logger.info("=" * 60)
    logger.info(f"Model: {config.qwen_image_edit_model_path}")
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Input: {args.img2img or 'None (text-to-image)'}")
    logger.info(f"Resolution: {resolution}x{resolution}")
    logger.info(f"CFG Scale: {config.qwen_image_cfg_scale}")
    logger.info(f"Steps: {config.qwen_image_steps}")
    logger.info(f"Transformer quant: {config.qwen_image_quantize_transformer}")
    logger.info(f"Text encoder quant: {config.qwen_image_quantize_text_encoder}")
    logger.info(f"CPU offload: {config.qwen_image_cpu_offload}")

    # Load input image if provided
    input_image = None
    if args.img2img:
        input_image = Image.open(args.img2img)
        logger.info(f"Input image size: {input_image.size}")

    # Load pipeline
    logger.info("Loading Qwen-Image-Edit pipeline...")
    start_load = time.time()

    from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

    try:
        # Map quantization strings
        quant_transformer = config.qwen_image_quantize_transformer
        if quant_transformer == "none":
            quant_transformer = None

        quant_text_encoder = config.qwen_image_quantize_text_encoder
        if quant_text_encoder == "none":
            quant_text_encoder = None

        pipe = QwenImageDiffusersPipeline.from_pretrained(
            model_path=None,  # Not needed for edit-only
            edit_model_path=config.qwen_image_edit_model_path,
            edit_only=True,
            device=torch.device(config.dit_device_resolved),
            torch_dtype=config.get_torch_dtype(),
            quantize_transformer=quant_transformer,
            quantize_text_encoder=quant_text_encoder,
            cpu_offload=config.qwen_image_cpu_offload,
        )
    except Exception as e:
        logger.error(f"Failed to load Qwen-Image-Edit pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1

    load_time = time.time() - start_load
    logger.info(f"Pipeline loaded in {load_time:.1f}s")

    # Set up seed
    seed = getattr(args, 'seed', None)

    # Detect mode: text-to-image vs image editing
    is_text_to_image = input_image is None
    logger.info(f"Mode: {'Text-to-image' if is_text_to_image else 'Image editing'}")

    # Run generation
    logger.info("Running image generation...")
    start_gen = time.time()

    try:
        if is_text_to_image:
            # Pure text-to-image generation
            result = pipe.generate(
                prompt=args.prompt,
                negative_prompt=getattr(config, 'negative_prompt', ' '),
                height=resolution,
                width=resolution,
                num_inference_steps=config.qwen_image_steps,
                cfg_scale=config.qwen_image_cfg_scale,
                seed=seed,
            )
        else:
            # Image editing mode
            result = pipe.edit_layer(
                layer_image=input_image,
                instruction=args.prompt,
                num_inference_steps=config.qwen_image_steps,
                cfg_scale=config.qwen_image_cfg_scale,
                seed=seed,
                max_size=resolution,
            )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    gen_time = time.time() - start_gen
    logger.info(f"Generation complete in {gen_time:.1f}s")

    # Save output
    output_path = Path(args.output)
    result.save(output_path)
    logger.info(f"Saved: {output_path}")

    logger.info("=" * 60)
    logger.info(f"Total time: load={load_time:.1f}s + generate={gen_time:.1f}s")

    return 0


def _run_qwen_image_decompose(args, config, logger, resolution: int) -> int:
    """Run Qwen-Image-Layered image decomposition."""
    from PIL import Image

    logger.info("=" * 60)
    logger.info("Qwen-Image-Layered Image Decomposition")
    logger.info("=" * 60)
    logger.info(f"Model: {config.qwen_image_model_path}")
    logger.info(f"Input: {args.img2img}")
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Resolution: {resolution}x{resolution}")
    logger.info(f"Layers: {config.qwen_image_layer_num}")
    logger.info(f"CFG Scale: {config.qwen_image_cfg_scale}")
    logger.info(f"Steps: {config.steps}")

    # Load input image
    input_image = Image.open(args.img2img)
    logger.info(f"Input image size: {input_image.size}")

    # Load pipeline
    logger.info("Loading Qwen-Image-Layered pipeline...")
    start_load = time.time()

    from llm_dit.pipelines.qwen_image import QwenImagePipeline

    try:
        pipe = QwenImagePipeline.from_pretrained(
            config.qwen_image_model_path,
            device=config.dit_device_resolved,
            text_encoder_device=config.encoder_device_resolved,
            torch_dtype=config.get_torch_dtype(),
        )
    except Exception as e:
        logger.error(f"Failed to load Qwen-Image pipeline: {e}")
        return 1

    load_time = time.time() - start_load
    logger.info(f"Pipeline loaded in {load_time:.1f}s")

    # Set up seed
    seed = getattr(args, 'seed', None)

    # Progress callback
    def progress_callback(step: int, total: int):
        logger.info(f"Step {step}/{total}")

    # Run decomposition
    logger.info("Running image decomposition...")
    start_gen = time.time()

    try:
        layers = pipe.decompose(
            image=input_image,
            prompt=args.prompt,
            layer_num=config.qwen_image_layer_num,
            height=resolution,
            width=resolution,
            num_inference_steps=config.steps,
            cfg_scale=config.qwen_image_cfg_scale,
            seed=seed,
            shift=config.shift if config.shift != 3.0 else None,  # Use dynamic if default
            progress_callback=progress_callback if config.verbose else None,
        )
    except Exception as e:
        logger.error(f"Decomposition failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    gen_time = time.time() - start_gen
    logger.info(f"Decomposition complete in {gen_time:.1f}s")
    logger.info(f"Generated {len(layers)} layers")

    # Save layers
    output_base = Path(args.output)
    output_dir = output_base.parent
    output_stem = output_base.stem
    output_suffix = output_base.suffix or ".png"

    saved_paths = []
    for i, layer_img in enumerate(layers):
        if i == 0:
            layer_name = "composite"
        else:
            layer_name = f"layer_{i}"

        layer_path = output_dir / f"{output_stem}_{layer_name}{output_suffix}"
        layer_img.save(layer_path)
        saved_paths.append(layer_path)
        logger.info(f"  Saved: {layer_path}")

    logger.info("=" * 60)
    logger.info(f"Total time: load={load_time:.1f}s + generate={gen_time:.1f}s")
    logger.info(f"Output files: {len(saved_paths)}")

    return 0


def run_qwen_image_t2i_generation(args, config, logger) -> int:
    """
    Run Qwen-Image text-to-image generation (T2I variant).

    Args:
        args: Parsed CLI arguments
        config: RuntimeConfig with all settings
        logger: Logger instance

    Returns:
        Exit code (0 for success)
    """
    from pathlib import Path

    # Validate model path
    model_path = config.qwen_image_model_path
    if not model_path:
        logger.error(
            "No Qwen-Image model path specified. "
            "Use --qwen-image-model-path or set qwen_image.model_path in config."
        )
        return 1

    # Get variant-aware defaults
    steps = config.get_qwen_image_steps()
    resolution = config.get_qwen_image_resolution()
    quant_transformer = config.get_qwen_image_quantize_transformer()
    cfg_scale = config.qwen_image_cfg_scale

    logger.info("=" * 60)
    logger.info("Qwen-Image Text-to-Image Generation")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Resolution: {config.width}x{config.height}")
    logger.info(f"CFG Scale: {cfg_scale}")
    logger.info(f"Steps: {steps}")
    logger.info(f"Transformer quant: {quant_transformer}")
    logger.info(f"Text encoder quant: {config.qwen_image_quantize_text_encoder}")

    # Load pipeline
    logger.info("Loading Qwen-Image T2I pipeline...")
    start_load = time.time()

    from llm_dit.pipelines import QwenImage2512Pipeline

    try:
        # Map quantization strings
        quant_tf = quant_transformer if quant_transformer != "none" else None
        quant_te = config.qwen_image_quantize_text_encoder
        quant_te = quant_te if quant_te != "none" else None

        pipe = QwenImage2512Pipeline.from_pretrained(
            model_path,
            quantize_transformer=quant_tf,
            quantize_text_encoder=quant_te,
            cpu_offload=config.qwen_image_cpu_offload,
        )
    except Exception as e:
        logger.error(f"Failed to load Qwen-Image T2I pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1

    load_time = time.time() - start_load
    logger.info(f"Pipeline loaded in {load_time:.1f}s")

    # Set up seed
    seed = getattr(args, 'seed', None)

    # Generate image
    logger.info("Generating image...")
    start_gen = time.time()

    try:
        image = pipe(
            prompt=args.prompt,
            negative_prompt=config.negative_prompt or " ",
            height=config.height,
            width=config.width,
            num_inference_steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    gen_time = time.time() - start_gen
    logger.info(f"Generation complete in {gen_time:.1f}s")

    # Save output
    output_path = Path(args.output)
    image.save(output_path)
    logger.info(f"Saved: {output_path}")

    logger.info("=" * 60)
    logger.info(f"Total time: load={load_time:.1f}s + generate={gen_time:.1f}s")

    return 0


def main():
    # Create parser with generation args
    parser = create_base_parser(
        description="Generate images with Z-Image, Qwen-Image-Layered, or Qwen-Image-2512",
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
    parser.add_argument(
        "--img2img",
        type=str,
        default=None,
        help="Input image for img2img generation",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.7,
        help="img2img strength (0.0=no change, 1.0=full regeneration, default: 0.7)",
    )

    args = parser.parse_args()

    # Load unified config
    config = load_runtime_config(args)
    setup_logging(config)

    logger = logging.getLogger(__name__)

    # Handle Qwen-Image model types
    if config.model_type == "qwenimage-layered":
        return run_qwen_image_generation(args, config, logger)

    if config.model_type == "qwenimage-t2i":
        return run_qwen_image_t2i_generation(args, config, logger)

    if config.model_type == "qwenimage-edit":
        # TODO: Implement dedicated edit-only generation flow
        logger.error("qwenimage-edit requires --img2img. Use qwenimage-t2i for text-to-image.")
        return 1

    # Z-Image flow continues below
    # Validate model path
    if config.model_path == "" and not args.load_embeddings:
        logger.error("No model path specified. Use --model-path or --config.")
        return 1

    # Validate and fix resolution
    from llm_dit.constants import VAE_MULTIPLE, MIN_RESOLUTION, MAX_RESOLUTION, snap_to_multiple

    width_valid = config.width % VAE_MULTIPLE == 0
    height_valid = config.height % VAE_MULTIPLE == 0

    if not width_valid or not height_valid:
        orig_width, orig_height = config.width, config.height
        config.width = snap_to_multiple(config.width, VAE_MULTIPLE)
        config.height = snap_to_multiple(config.height, VAE_MULTIPLE)
        logger.warning(
            f"Resolution {orig_width}x{orig_height} not divisible by {VAE_MULTIPLE}. "
            f"Snapped to {config.width}x{config.height}"
        )

    if config.width < MIN_RESOLUTION or config.height < MIN_RESOLUTION:
        logger.warning(f"Resolution below minimum {MIN_RESOLUTION}px may produce poor results")
    if config.width > MAX_RESOLUTION or config.height > MAX_RESOLUTION:
        logger.warning(f"Resolution above {MAX_RESOLUTION}px may require tiled VAE (--tiled-vae)")

    # Find templates directory
    templates_dir = config.templates_dir
    if templates_dir is None:
        # Try default location relative to this script
        default_templates = Path(__file__).parent.parent / "templates" / "z_image"
        if default_templates.exists():
            templates_dir = str(default_templates)
            logger.info(f"Using default templates: {templates_dir}")

    # Set up seed (generator created later based on code path)
    seed = getattr(args, 'seed', None)
    if seed is not None:
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

        # CPU generator for generate_from_embeddings
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)

        start = time.time()
        image = pipe.generate_from_embeddings(
            embeds,
            height=config.height,
            width=config.width,
            num_inference_steps=config.steps,
            guidance_scale=config.guidance_scale,
            generator=generator,
            shift=config.shift,
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

        # CPU generator for txt2img
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)

        start = time.time()
        image = pipe(
            prompt_embeds=embeddings,  # Use pre-computed embeddings
            height=config.height,
            width=config.width,
            num_inference_steps=config.steps,
            guidance_scale=config.guidance_scale,
            generator=generator,
            long_prompt_mode=config.long_prompt_mode,
            skip_layer_guidance_scale=config.slg_scale,
            skip_layer_indices=config.slg_layers,
            skip_layer_start=config.slg_start,
            skip_layer_stop=config.slg_stop,
            shift=config.shift,
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

    # Get negative prompt from CLI
    negative_prompt = getattr(args, 'negative_prompt', None)

    # Check for img2img mode
    if args.img2img:
        from PIL import Image

        logger.info(f"Running img2img with strength={args.strength}")
        logger.info(f"Input image: {args.img2img}")

        input_image = Image.open(args.img2img)
        logger.info(f"  Size: {input_image.size}")

        logger.info(f"Prompt: {args.prompt}")

        # img2img needs CUDA generator (creates noise directly on device)
        generator = None
        if seed is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        start = time.time()
        image = pipe.img2img(
            prompt=args.prompt,
            image=input_image,
            strength=args.strength,
            num_inference_steps=config.steps,
            guidance_scale=config.guidance_scale,
            negative_prompt=negative_prompt,
            generator=generator,
            template=config.default_template,
            system_prompt=config.system_prompt,
            thinking_content=config.thinking_content,
            assistant_content=config.assistant_content,
            force_think_block=config.enable_thinking,
            long_prompt_mode=config.long_prompt_mode,
            hidden_layer=config.hidden_layer,
            layer_weights=config.layer_weights,
            shift=config.shift,
            callback=progress_callback if config.verbose else None,
        )
        gen_time = time.time() - start
    else:
        # Normal txt2img generation
        logger.info(f"Generating {config.width}x{config.height} image...")
        logger.info(f"Prompt: {args.prompt}")
        if config.default_template:
            logger.info(f"Template: {config.default_template}")

        # txt2img needs CPU generator (creates noise on CPU then moves to device)
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)

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
            layer_weights=config.layer_weights,
            skip_layer_guidance_scale=config.slg_scale,
            skip_layer_indices=config.slg_layers,
            skip_layer_start=config.slg_start,
            skip_layer_stop=config.slg_stop,
            shift=config.shift,
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
