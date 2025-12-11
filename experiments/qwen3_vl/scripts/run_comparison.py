#!/usr/bin/env python3
"""
Run comprehensive VL conditioning comparisons.

This script runs experiments comparing different VL conditioning approaches:
- Pure Qwen3-4B text (baseline, alpha=0.0)
- Pure Qwen3-VL (alpha=1.0, no blending)
- Blended VL + text at various alpha ratios
- Different token selections (full sequence, image only, text only)
- Different hidden layers

Token Selection Modes:
    - full_sequence: All tokens (system + image + text + assistant)
    - image_w_markers: Image region including special tokens (151652, 151653)
    - image_no_markers: Image region excluding special marker tokens
    - text_tokens: Everything except image region

Note on token positions:
    After transformer self-attention, ALL positions carry mixed information
    from the entire sequence. Token selection tests whether position affects
    quality, not whether information is "pure" image or text.

Experiments:
    pure_vl              - ISOLATED Qwen3-VL only, NO text prompt in VL input (4 configs)
                           Tests what Z-Image DiT does with pure image-only VL embeddings
    pure_vl_with_prompt  - ISOLATED Qwen3-VL with text prompt INCLUDED in VL input (4 configs)
                           Tests if including text in VL extraction influences output
                           This is the key test: after self-attention, do text tokens
                           carry enough prompt information to guide generation?
    token_selection      - Blended VL + Qwen3-4B at various alphas (5-17 configs)
                           Tests whether token position affects blending quality
    alpha_sweep          - Sweep alpha from 0.0 to 1.0 (4-11 configs)
    hidden_layer         - Sweep hidden layers -1 to -6 (3-6 configs)
    style_transfer       - Low alpha values for style transfer (4-9 configs)
    full                 - Comprehensive test across all dimensions

Usage:
    # ISOLATED Qwen3-VL test (no Qwen3-4B blending)
    uv run experiments/qwen3_vl/scripts/run_comparison.py \\
        --image experiments/inputs/test_scene.png \\
        --prompt "Homer Simpson eating spaghetti" \\
        --experiment pure_vl \\
        --output-dir experiments/results/pure_vl

    # Token selection with blending (uses both models)
    uv run experiments/qwen3_vl/scripts/run_comparison.py \\
        --image experiments/inputs/test_scene.png \\
        --prompt "Homer Simpson eating spaghetti" \\
        --experiment token_selection \\
        --quick \\
        --output-dir experiments/results/token_selection

    # Alpha sweep
    uv run experiments/qwen3_vl/scripts/run_comparison.py \\
        --image experiments/inputs/test_scene.png \\
        --prompt "Your text prompt" \\
        --experiment alpha_sweep \\
        --output-dir experiments/results/alpha_sweep

    # Hidden layer sweep
    uv run experiments/qwen3_vl/scripts/run_comparison.py \\
        --image experiments/inputs/test_scene.png \\
        --prompt "Your text prompt" \\
        --experiment hidden_layer \\
        --output-dir experiments/results/layer_sweep
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    name: str
    alpha: float
    hidden_layer: int = -2
    image_tokens_only: bool = False
    image_tokens_no_markers: bool = False
    text_tokens_only: bool = False
    text: str | None = None  # Text to include with image in VL
    scale_to_text: bool = True


def get_experiment_configs(experiment_type: str, quick: bool = False) -> list[ExperimentConfig]:
    """Get experiment configurations for different experiment types."""

    if experiment_type == "alpha_sweep":
        # Sweep interpolation ratios
        if quick:
            alphas = [0.0, 0.3, 0.5, 1.0]
        else:
            alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        return [
            ExperimentConfig(name=f"alpha_{a:.1f}", alpha=a)
            for a in alphas
        ]

    elif experiment_type == "style_transfer":
        # Style transfer: low VL ratios to transfer style without content
        if quick:
            alphas = [0.0, 0.2, 0.3, 0.4]
        else:
            alphas = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]

        return [
            ExperimentConfig(name=f"style_{a:.2f}", alpha=a)
            for a in alphas
        ]

    elif experiment_type == "hidden_layer":
        # Compare different hidden layers
        layers = [-1, -2, -3, -4, -5, -6] if not quick else [-1, -2, -4]
        return [
            ExperimentConfig(name=f"layer_{l}", alpha=0.3, hidden_layer=l)
            for l in layers
        ]

    elif experiment_type == "token_selection":
        # Compare token selection modes WITH blending (uses both Qwen3-VL and Qwen3-4B)
        configs = []

        # Pure text baseline (alpha=0.0, only Qwen3-4B)
        configs.append(ExperimentConfig(name="pure_text_baseline", alpha=0.0))

        # Blended at alpha=0.3 - test all 4 token selection modes
        configs.extend([
            ExperimentConfig(name="blend_full_sequence_a03", alpha=0.3),
            ExperimentConfig(name="blend_image_w_markers_a03", alpha=0.3, image_tokens_only=True),
            ExperimentConfig(name="blend_image_no_markers_a03", alpha=0.3, image_tokens_no_markers=True),
            ExperimentConfig(name="blend_text_tokens_a03", alpha=0.3, text_tokens_only=True),
        ])

        if not quick:
            # Also test at other alpha levels
            for alpha in [0.1, 0.2, 0.5]:
                configs.extend([
                    ExperimentConfig(name=f"blend_full_sequence_a{int(alpha*10):02d}", alpha=alpha),
                    ExperimentConfig(name=f"blend_image_w_markers_a{int(alpha*10):02d}", alpha=alpha, image_tokens_only=True),
                    ExperimentConfig(name=f"blend_image_no_markers_a{int(alpha*10):02d}", alpha=alpha, image_tokens_no_markers=True),
                    ExperimentConfig(name=f"blend_text_tokens_a{int(alpha*10):02d}", alpha=alpha, text_tokens_only=True),
                ])

        return configs

    elif experiment_type == "pure_vl":
        # ISOLATED Qwen3-VL test - NO Qwen3-4B blending (alpha=1.0)
        # Tests what Z-Image DiT does with ONLY VL embeddings
        # This directly tests: "Are VL image token positions garbage to Z-Image?"
        # NOTE: text=None means only the image is passed to VL, not the text prompt
        return [
            ExperimentConfig(name="pure_vl_full_sequence", alpha=1.0),
            ExperimentConfig(name="pure_vl_image_w_markers", alpha=1.0, image_tokens_only=True),
            ExperimentConfig(name="pure_vl_image_no_markers", alpha=1.0, image_tokens_no_markers=True),
            ExperimentConfig(name="pure_vl_text_tokens", alpha=1.0, text_tokens_only=True),
        ]

    elif experiment_type == "pure_vl_with_prompt":
        # ISOLATED Qwen3-VL test WITH text prompt included in VL input
        # Tests: Does including the text prompt in VL extraction influence output?
        # The text is passed to Qwen3-VL along with the image, so after self-attention
        # the embeddings should carry information about both image AND text.
        # Uses special marker "__PROMPT__" which run_experiments() replaces with actual prompt
        return [
            ExperimentConfig(name="pure_vl_prompt_full", alpha=1.0, text="__PROMPT__"),
            ExperimentConfig(name="pure_vl_prompt_image_tokens", alpha=1.0, text="__PROMPT__", image_tokens_only=True),
            ExperimentConfig(name="pure_vl_prompt_text_tokens", alpha=1.0, text="__PROMPT__", text_tokens_only=True),
            # Compare to no-prompt baseline
            ExperimentConfig(name="pure_vl_no_prompt_full", alpha=1.0, text=None),
        ]

    elif experiment_type == "text_guidance":
        # Compare VL with and without text description
        return [
            ExperimentConfig(name="vl_no_text", alpha=0.5, text=None),
            ExperimentConfig(name="vl_with_desc", alpha=0.5, text="Describe this image in detail."),
        ]

    elif experiment_type == "vl_intra_token_blend":
        # NEW: Blend image tokens (visual style) with text tokens (prompt following) from SAME VL extraction
        # Key insight: At alpha=1.0, text tokens already follow the prompt ("Homer Simpson")
        # while image tokens carry visual style. Can we get BOTH by blending within VL?
        configs = []

        # Baselines: Pure VL with different token selections (alpha=1.0, no Qwen3-4B)
        configs.extend([
            ExperimentConfig(name="baseline_vl_full", alpha=1.0, text="__PROMPT__"),
            ExperimentConfig(name="baseline_vl_image_only", alpha=1.0, text="__PROMPT__", image_tokens_no_markers=True),
            ExperimentConfig(name="baseline_vl_text_only", alpha=1.0, text="__PROMPT__", text_tokens_only=True),
        ])

        if not quick:
            # TODO: Implement intra-VL blending in extraction or post-processing
            # This requires extracting BOTH image and text token subsets separately,
            # then blending them at various ratios BEFORE passing to Z-Image
            # Ratios to test: 30% image style + 70% text prompt guidance
            configs.extend([
                # These would use a new parameter like vl_intra_blend_ratio
                # ExperimentConfig(name="intra_blend_30img_70text", alpha=1.0, text="__PROMPT__", vl_intra_blend_ratio=0.3),
                # ExperimentConfig(name="intra_blend_50img_50text", alpha=1.0, text="__PROMPT__", vl_intra_blend_ratio=0.5),
                # ExperimentConfig(name="intra_blend_70img_30text", alpha=1.0, text="__PROMPT__", vl_intra_blend_ratio=0.7),
            ])

        return configs

    elif experiment_type == "vl_only_vs_qwen3":
        # NEW: Direct comparison - Can VL alone replace Qwen3-4B entirely?
        # Tests if blending Qwen3-4B is even necessary when VL text tokens follow prompts
        return [
            # Pure Qwen3-4B (baseline)
            ExperimentConfig(name="pure_qwen3_4b", alpha=0.0),
            # Pure VL with prompt in text tokens (our new finding)
            ExperimentConfig(name="pure_vl_text_tokens", alpha=1.0, text="__PROMPT__", text_tokens_only=True),
            # Pure VL full sequence (includes both)
            ExperimentConfig(name="pure_vl_full_seq", alpha=1.0, text="__PROMPT__"),
            # Traditional blend (for comparison)
            ExperimentConfig(name="blend_30_vl_70_qwen3", alpha=0.3),
        ]

    elif experiment_type == "vl_layer_by_token":
        # NEW: Test different hidden layers for VL extraction, isolated by token type
        # Research question: Do image vs text tokens benefit from different layers?
        configs = []

        # Test image tokens across layers
        for layer in [-1, -2, -5, -10, -15]:
            configs.append(
                ExperimentConfig(
                    name=f"image_tokens_layer{layer}",
                    alpha=1.0,
                    text="__PROMPT__",
                    image_tokens_no_markers=True,
                    hidden_layer=layer,
                )
            )

        # Test text tokens across layers
        for layer in [-1, -2, -5, -10, -15]:
            configs.append(
                ExperimentConfig(
                    name=f"text_tokens_layer{layer}",
                    alpha=1.0,
                    text="__PROMPT__",
                    text_tokens_only=True,
                    hidden_layer=layer,
                )
            )

        if quick:
            # Quick mode: only test -2 and -10
            configs = [c for c in configs if c.hidden_layer in [-2, -10]]

        return configs

    elif experiment_type == "vl_double_conditioning":
        # NEW: Use TWO VL extractions - one for style (image tokens), one for prompt (text tokens)
        # Then blend them together. This tests if we can get best of both worlds.
        # NOTE: This requires extracting TWICE with different token selections
        return [
            # Single extractions (baselines)
            ExperimentConfig(name="single_image_tokens", alpha=1.0, text=None, image_tokens_no_markers=True),
            ExperimentConfig(name="single_text_tokens", alpha=1.0, text="__PROMPT__", text_tokens_only=True),
            # TODO: Implement dual extraction blending
            # Would need new infrastructure to:
            # 1. Extract with image_tokens_no_markers, text=None -> style_emb
            # 2. Extract with text_tokens_only, text=prompt -> prompt_emb
            # 3. Blend style_emb + prompt_emb at various ratios
            # ExperimentConfig(name="dual_30style_70prompt", ...),
        ]

    elif experiment_type == "vl_prompt_variations":
        # NEW: Test how TEXT content in VL extraction affects prompt following
        # Key question: What text gives the best prompt adherence in text tokens?
        return [
            # No text in VL (pure visual)
            ExperimentConfig(name="no_text_image_tokens", alpha=1.0, text=None, image_tokens_only=True),
            ExperimentConfig(name="no_text_text_tokens", alpha=1.0, text=None, text_tokens_only=True),
            # Exact prompt in VL (our finding)
            ExperimentConfig(name="with_prompt_image_tokens", alpha=1.0, text="__PROMPT__", image_tokens_only=True),
            ExperimentConfig(name="with_prompt_text_tokens", alpha=1.0, text="__PROMPT__", text_tokens_only=True),
            # Generic description in VL
            ExperimentConfig(name="with_desc_text_tokens", alpha=1.0, text="Describe what you see.", text_tokens_only=True),
            # Instruction in VL
            ExperimentConfig(name="with_instruction_text_tokens", alpha=1.0, text="Generate an image following this description.", text_tokens_only=True),
        ]

    elif experiment_type == "full":
        # Full comparison (all dimensions)
        configs = []

        # Alpha sweep
        for a in [0.0, 0.3, 0.5, 0.7, 1.0]:
            configs.append(ExperimentConfig(name=f"alpha_{a:.1f}", alpha=a))

        # Hidden layers at optimal alpha
        for l in [-1, -2, -3, -4]:
            if l != -2:  # -2 already covered
                configs.append(ExperimentConfig(name=f"layer_{l}", alpha=0.3, hidden_layer=l))

        # Token selection
        configs.append(ExperimentConfig(name="img_tokens", alpha=0.3, image_tokens_only=True))

        return configs

    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")


def generate_comparison_grid(
    results: list[dict],
    output_dir: Path,
    reference_image_path: str,
    prompt: str,
    thumbnail_size: int = 512,
) -> Path:
    """
    Generate a comparison grid from experiment results.

    Creates a horizontal strip of all generated images with labels.

    Args:
        results: List of result dicts with 'output_path' and 'name' keys
        output_dir: Directory to save the grid
        reference_image_path: Path to the reference image (shown first)
        prompt: Text prompt used (shown in header)
        thumbnail_size: Size for each thumbnail

    Returns:
        Path to the generated grid image
    """
    from PIL import Image as PILImage, ImageDraw, ImageFont

    # Filter to successful results only
    successful = [r for r in results if r.get("success", False)]
    if not successful:
        logger.warning("No successful results to create grid from")
        return None

    # Load reference image
    ref_img = PILImage.open(reference_image_path).convert("RGB")

    # Load all result images
    images = []
    labels = ["Reference"]
    images.append(ref_img)

    for r in successful:
        try:
            img = PILImage.open(r["output_path"]).convert("RGB")
            images.append(img)
            labels.append(r["name"])
        except Exception as e:
            logger.warning(f"Could not load {r['output_path']}: {e}")

    if len(images) <= 1:
        logger.warning("No result images to create grid from")
        return None

    # Resize all to thumbnail size
    thumbs = []
    for img in images:
        thumb = img.copy()
        thumb.thumbnail((thumbnail_size, thumbnail_size), PILImage.Resampling.LANCZOS)
        thumbs.append(thumb)

    # Calculate grid dimensions
    n_images = len(thumbs)
    padding = 10
    label_height = 30
    header_height = 50

    # Create canvas
    total_width = n_images * (thumbnail_size + padding) + padding
    total_height = header_height + thumbnail_size + label_height + padding * 2

    canvas = PILImage.new("RGB", (total_width, total_height), color=(30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        header_font = font

    # Draw header with prompt
    prompt_display = prompt[:80] + "..." if len(prompt) > 80 else prompt
    draw.text(
        (total_width // 2, header_height // 2),
        f"Prompt: {prompt_display}",
        fill=(200, 200, 200),
        font=header_font,
        anchor="mm",
    )

    # Paste images and labels
    for i, (thumb, label) in enumerate(zip(thumbs, labels)):
        x = padding + i * (thumbnail_size + padding)
        y = header_height + padding

        # Center thumbnail if smaller than cell
        paste_x = x + (thumbnail_size - thumb.width) // 2
        paste_y = y + (thumbnail_size - thumb.height) // 2
        canvas.paste(thumb, (paste_x, paste_y))

        # Draw label below
        label_y = y + thumbnail_size + 5
        # Truncate label if too long
        label_display = label[:20] + "..." if len(label) > 20 else label
        draw.text(
            (x + thumbnail_size // 2, label_y),
            label_display,
            fill=(180, 180, 180),
            font=font,
            anchor="mt",
        )

    # Save grid
    grid_path = output_dir / "comparison_grid.png"
    canvas.save(grid_path)
    logger.info(f"Saved comparison grid to {grid_path}")

    return grid_path


def run_experiments(
    image_path: str,
    prompt: str,
    output_dir: str,
    configs: list[ExperimentConfig],
    model_path: str | None = None,
    z_image_config: str = "config.toml",
    z_image_profile: str = "rtx4090",
    seed: int = 42,
    steps: int = 9,
):
    """Run all experiment configurations.

    Models are loaded once and reused across all experiments for efficiency.
    A comparison grid is generated at the end.
    """
    from llm_dit.vl import VLEmbeddingExtractor
    from llm_dit.startup import PipelineLoader
    from blend_and_generate import encode_text_prompt, blend_embeddings

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    image = Image.open(image_path).convert("RGB")
    logger.info(f"Loaded image: {image_path} ({image.size[0]}x{image.size[1]})")

    # Load Z-Image config
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from llm_dit.cli import load_runtime_config

    class ConfigArgs:
        pass
    config_args = ConfigArgs()
    config_args.config = z_image_config
    config_args.profile = z_image_profile
    config_args.steps = steps
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
                 'assistant_content', 'enable_thinking',
                 'vl_model_path', 'vl_device', 'vl_hidden_layer', 'vl_alpha',
                 'vl_blend_mode', 'vl_auto_unload']:
        if not hasattr(config_args, attr):
            setattr(config_args, attr, None)
    z_config = load_runtime_config(config_args)

    # Encode text prompt once
    logger.info(f"Encoding text prompt: {prompt[:50]}...")
    text_emb = encode_text_prompt(prompt, z_config)
    logger.info(f"Text embeddings: shape={text_emb.shape}, std={text_emb.std():.2f}")

    # Get VL model path from CLI, config, or auto-detect
    vl_model_path = model_path
    if not vl_model_path and hasattr(z_config, 'vl_model_path'):
        vl_model_path = z_config.vl_model_path
    if not vl_model_path:
        candidates = [
            Path.home() / "Storage" / "Qwen3-VL-4B-Instruct",
            Path.home() / "models" / "Qwen3-VL-4B-Instruct",
        ]
        for candidate in candidates:
            if candidate.exists():
                vl_model_path = str(candidate)
                break

    if not vl_model_path:
        raise ValueError("Could not find Qwen3-VL model. Set vl.model_path in config.toml or use --model-path")

    # Load VL extractor using core module (loaded once, reused)
    vl_device = getattr(z_config, 'vl_device', None) or "cuda"
    vl_dtype = torch.bfloat16 if vl_device == "cuda" else torch.float32
    logger.info(f"Loading VLEmbeddingExtractor from {vl_model_path}")
    vl_extractor = VLEmbeddingExtractor.from_pretrained(
        vl_model_path,
        device=vl_device,
        torch_dtype=vl_dtype,
    )

    # Load Z-Image pipeline once (reused for all generations)
    logger.info("Loading Z-Image pipeline (will be reused for all experiments)...")
    loader = PipelineLoader(z_config)
    pipeline_result = loader.load_pipeline()
    pipe = pipeline_result.pipeline
    logger.info("Pipeline loaded successfully")

    # Set up generator for reproducibility
    generator = torch.Generator()

    # Results tracking
    results = []

    # Run each configuration
    for i, config in enumerate(configs):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{len(configs)}] Running: {config.name}")
        logger.info(f"{'='*60}")

        start_time = time.time()

        try:
            # Handle __PROMPT__ placeholder - replace with actual prompt
            vl_text = config.text
            if vl_text == "__PROMPT__":
                vl_text = prompt
                logger.info(f"  Including text prompt in VL: {prompt[:50]}...")

            # Extract VL embeddings with this config's settings using core module
            vl_result = vl_extractor.extract(
                image=image,
                text=vl_text,
                hidden_layer=config.hidden_layer,
                image_tokens_only=config.image_tokens_only,
                image_tokens_no_markers=config.image_tokens_no_markers,
                text_tokens_only=config.text_tokens_only,
                scale_to_text=config.scale_to_text,
            )
            vl_emb = vl_result.embeddings

            # Blend embeddings
            if config.alpha == 0.0:
                blended = text_emb
            elif config.alpha == 1.0:
                blended = vl_emb
            else:
                blended = blend_embeddings(vl_emb, text_emb, config.alpha)

            # Generate using the pre-loaded pipeline
            output_path = output_dir / f"{config.name}.png"
            generator.manual_seed(seed)

            logger.info(f"  Generating {z_config.width}x{z_config.height} image...")
            gen_start = time.time()

            result_image = pipe(
                prompt_embeds=blended,
                height=z_config.height,
                width=z_config.width,
                num_inference_steps=z_config.steps,
                guidance_scale=z_config.guidance_scale,
                generator=generator,
            )

            gen_time = time.time() - gen_start
            logger.info(f"  Generation time: {gen_time:.1f}s")

            # Save image
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result_image.save(output_path)
            logger.info(f"  Saved to {output_path}")

            elapsed = time.time() - start_time

            # Record result
            results.append({
                **asdict(config),
                "output_path": str(output_path),
                "vl_shape": list(vl_emb.shape),
                "vl_std": vl_result.scaled_std,
                "token_selection": vl_result.token_selection,
                "chat_template_format": vl_result.chat_template_format,
                "blended_std": blended.std().item(),
                "elapsed_seconds": elapsed,
                "generation_time": gen_time,
                "success": True,
            })

        except Exception as e:
            logger.error(f"Failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                **asdict(config),
                "success": False,
                "error": str(e),
            })

    # Cleanup
    logger.info("\nCleaning up models...")
    del vl_extractor
    del pipe
    del pipeline_result
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save results metadata
    metadata = {
        "image_path": str(image_path),
        "prompt": prompt,
        "seed": seed,
        "steps": steps,
        "results": results,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Generate comparison grid
    logger.info("\nGenerating comparison grid...")
    generate_comparison_grid(
        results=results,
        output_dir=output_dir,
        reference_image_path=image_path,
        prompt=prompt,
    )

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Total experiments: {len(results)}")
    logger.info(f"Successful: {sum(1 for r in results if r.get('success', False))}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive VL conditioning comparisons",
    )

    parser.add_argument("--image", "-i", required=True, help="Reference image path")
    parser.add_argument("--prompt", "-p", required=True, help="Text prompt")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory (default: experiments/results/{experiment}_{timestamp})")
    parser.add_argument("--experiment", "-e", default="alpha_sweep",
                        choices=["alpha_sweep", "style_transfer", "hidden_layer",
                                 "token_selection", "pure_vl", "pure_vl_with_prompt",
                                 "text_guidance", "vl_intra_token_blend", "vl_only_vs_qwen3",
                                 "vl_layer_by_token", "vl_double_conditioning", "vl_prompt_variations",
                                 "full"],
                        help="Experiment type. NEW experiments based on token position findings: "
                             "vl_intra_token_blend=blend VL image+text tokens, "
                             "vl_only_vs_qwen3=compare VL-only vs Qwen3-4B, "
                             "vl_layer_by_token=test layers separately for image/text tokens, "
                             "vl_double_conditioning=dual VL extractions, "
                             "vl_prompt_variations=test different VL text inputs")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer variations")
    parser.add_argument("--model-path", help="Qwen3-VL model path")
    parser.add_argument("--config", default="config.toml", help="Z-Image config file")
    parser.add_argument("--profile", default="rtx4090", help="Z-Image config profile")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--steps", type=int, default=9, help="Inference steps (default: 9 for Z-Image Turbo)")

    args = parser.parse_args()

    # Generate output directory with experiment name and timestamp if not specified
    if args.output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"experiments/results/{args.experiment}_{timestamp}"

    # Get experiment configs
    configs = get_experiment_configs(args.experiment, quick=args.quick)
    logger.info(f"Running {len(configs)} experiment configurations")
    logger.info(f"Output directory: {args.output_dir}")

    # Run experiments
    run_experiments(
        image_path=args.image,
        prompt=args.prompt,
        output_dir=args.output_dir,
        configs=configs,
        model_path=args.model_path,
        z_image_config=args.config,
        z_image_profile=args.profile,
        seed=args.seed,
        steps=args.steps,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
