#!/usr/bin/env python3
"""
Blend VL embeddings with text embeddings and generate images.

This script takes pre-extracted VL embeddings (from extract_embeddings.py) and
blends them with text embeddings to condition Z-Image generation.

Usage:
    # Basic usage with pre-extracted VL embeddings
    uv run experiments/qwen3_vl/blend_and_generate.py \
        --vl-embeddings vl.pt \
        --prompt "Your text prompt" \
        --output result.png

    # Adjust VL influence (0.0 = pure text, 1.0 = pure VL)
    uv run experiments/qwen3_vl/blend_and_generate.py \
        --vl-embeddings vl.pt \
        --prompt "Your text prompt" \
        --alpha 0.5 \
        --output result.png

    # Pure VL (no text blending)
    uv run experiments/qwen3_vl/blend_and_generate.py \
        --vl-embeddings vl.pt \
        --alpha 1.0 \
        --output result.png

    # Generate multiple alpha values for comparison
    uv run experiments/qwen3_vl/blend_and_generate.py \
        --vl-embeddings vl.pt \
        --prompt "Your text prompt" \
        --sweep-alpha 0.0,0.1,0.2,0.3,0.5,0.7,1.0 \
        --output-dir results/
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

# Import blending functions from core module (avoid duplication)
from llm_dit.vl.blending import blend_embeddings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def load_vl_embeddings(path: str) -> dict:
    """Load VL embeddings from file."""
    saved = torch.load(path, weights_only=True)
    logger.info(f"Loaded VL embeddings: shape={saved['embeddings'].shape}")
    if "source_image" in saved:
        logger.info(f"  Source: {saved['source_image']}")
    return saved


@dataclass
class TextEncodingResult:
    """Result of text prompt encoding."""
    embeddings: torch.Tensor  # (seq_len, hidden_dim)
    formatted_prompt: str  # Full prompt with all special tokens


def encode_text_prompt(
    prompt: str,
    config,
    force_think_block: bool = False,
    system_prompt: str | None = None,
) -> TextEncodingResult:
    """
    Encode text prompt using Z-Image's standard encoder.

    Args:
        prompt: Text prompt to encode
        config: RuntimeConfig with encoder settings
        force_think_block: If True, add empty think block (matches VL's injected format)
        system_prompt: Optional system message

    Returns:
        TextEncodingResult with embeddings and full formatted prompt
    """
    # Import here to avoid loading until needed
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from llm_dit.startup import PipelineLoader

    loader = PipelineLoader(config)
    result = loader.load_encoder()
    encoder = result.encoder

    output = encoder.encode(
        prompt,
        force_think_block=force_think_block,
        system_prompt=system_prompt,
    )
    return TextEncodingResult(
        embeddings=output.embeddings[0],
        formatted_prompt=output.formatted_prompts[0] if output.formatted_prompts else "",
    )


def generate_from_embeddings(
    embeddings: torch.Tensor,
    config,
    output_path: str,
    seed: int | None = None,
):
    """Generate image from embeddings using Z-Image pipeline."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from llm_dit.startup import PipelineLoader

    loader = PipelineLoader(config)
    result = loader.load_pipeline()
    pipe = result.pipeline

    # Set up generator
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    # Generate
    logger.info(f"Generating {config.width}x{config.height} image...")
    start = time.time()

    image = pipe(
        prompt_embeds=embeddings,
        height=config.height,
        width=config.width,
        num_inference_steps=config.steps,
        guidance_scale=config.guidance_scale,
        generator=generator,
    )

    gen_time = time.time() - start
    logger.info(f"Generation time: {gen_time:.1f}s")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    logger.info(f"Saved to {output_path}")

    return image


def main():
    parser = argparse.ArgumentParser(
        description="Blend VL embeddings with text and generate images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input
    parser.add_argument(
        "--vl-embeddings",
        type=str,
        required=True,
        help="Path to VL embeddings file (.pt)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt (required unless alpha=1.0)",
    )

    # Blending
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="VL influence ratio (0.0=pure text, 1.0=pure VL, default: 0.3)",
    )
    parser.add_argument(
        "--sweep-alpha",
        type=str,
        default=None,
        help="Comma-separated alpha values to sweep",
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output.png",
        help="Output image path (default: output.png)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (for sweep mode)",
    )

    # Generation params
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to config file",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="rtx4090",
        help="Config profile to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=9,
        help="Number of inference steps (default: 9 for Z-Image Turbo)",
    )

    args = parser.parse_args()

    # Load config
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from llm_dit.cli import load_runtime_config

    # Create a minimal args object for config loading
    class ConfigArgs:
        pass

    config_args = ConfigArgs()
    config_args.config = args.config
    config_args.profile = args.profile
    config_args.steps = args.steps

    # Set defaults for required fields
    for attr in ['model_path', 'text_encoder_device', 'dit_device', 'vae_device',
                 'cpu_offload', 'flash_attn', 'compile', 'debug', 'verbose',
                 'attention_backend', 'use_custom_scheduler', 'tiled_vae',
                 'embedding_cache', 'long_prompt_mode', 'hidden_layer', 'shift',
                 'lora', 'api_url', 'api_model', 'local_encoder', 'templates_dir',
                 'torch_dtype', 'text_encoder_path', 'tile_size', 'tile_overlap',
                 'cache_size', 'rewriter_use_api', 'rewriter_api_url', 'rewriter_api_model',
                 'rewriter_temperature', 'rewriter_top_p', 'rewriter_top_k',
                 'rewriter_min_p', 'rewriter_presence_penalty', 'rewriter_max_tokens',
                 'width', 'height', 'guidance_scale', 'negative_prompt', 'seed',
                 'embeddings_file', 'template', 'system_prompt', 'thinking_content',
                 'assistant_content', 'enable_thinking']:
        if not hasattr(config_args, attr):
            setattr(config_args, attr, None)

    config = load_runtime_config(config_args)

    # Load VL embeddings
    vl_data = load_vl_embeddings(args.vl_embeddings)
    vl_emb = vl_data["embeddings"]

    # Determine alpha values to process
    if args.sweep_alpha:
        alphas = [float(a.strip()) for a in args.sweep_alpha.split(",")]
    else:
        alphas = [args.alpha]

    # Validate prompt requirement
    if args.prompt is None and any(a < 1.0 for a in alphas):
        logger.error("--prompt is required when alpha < 1.0")
        return 1

    # Encode text if needed
    text_emb = None
    if args.prompt and any(a < 1.0 for a in alphas):
        logger.info(f"Encoding text prompt: {args.prompt[:50]}...")
        text_emb = encode_text_prompt(args.prompt, config)
        logger.info(f"Text embeddings: shape={text_emb.shape}, std={text_emb.std():.2f}")

    # Generate for each alpha
    for alpha in alphas:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing alpha={alpha}")
        logger.info(f"{'='*60}")

        # Blend embeddings
        if alpha == 1.0:
            blended = vl_emb
        elif alpha == 0.0:
            blended = text_emb
        else:
            blended = blend_embeddings(vl_emb, text_emb, alpha)

        # Determine output path
        if len(alphas) > 1:
            output_dir = Path(args.output_dir or "results/vl_sweep")
            output_path = output_dir / f"alpha_{alpha:.2f}.png"
        else:
            output_path = args.output

        # Generate
        generate_from_embeddings(
            embeddings=blended,
            config=config,
            output_path=str(output_path),
            seed=args.seed,
        )

    logger.info("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
