#!/usr/bin/env python3
"""FMTT Proof-of-Concept for Z-Image.

This standalone script validates the FMTT (Flow Map Trajectory Tilting) algorithm
with Z-Image Turbo. It implements reward-guided sampling using gradient-based
trajectory modification.

Algorithm overview:
1. At each denoising step, predict the final clean image via flow map
2. Evaluate SigLIP2 reward on the prediction
3. Backprop gradient through flow map to current noisy latents
4. Nudge the velocity toward higher-reward regions

Run from repository root:
    uv run experiments/fmtt/poc.py --config config.toml --profile rtx4090 "A cat with five fingers"
    uv run experiments/fmtt/poc.py --model-path /path/to/z-image "A cat with five fingers"

Key components:
- DifferentiableSigLIP: Gradient-enabled reward function
- flow_map_direct: Single-step trajectory prediction
- fmtt_sampling: Modified denoising loop with reward guidance

Reference: arXiv:2511.22688 (Test-Time Scaling of Diffusion Models with Flow Maps)
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.fmtt.differentiable_siglip import DifferentiableSigLIP
from experiments.prompts import get_prompt_by_id, get_categories, get_prompts_by_category


def create_comparison_image(
    baseline_img: Image.Image,
    fmtt_img: Image.Image,
    baseline_reward: float,
    fmtt_reward: float,
    prompt: str,
    guidance_scale: float,
) -> Image.Image:
    """Create side-by-side comparison with labels and metrics."""
    width, height = baseline_img.size

    # Create canvas with space for labels
    label_height = 60
    canvas = Image.new("RGB", (width * 2 + 10, height + label_height), (30, 30, 30))

    # Paste images
    canvas.paste(baseline_img, (0, label_height))
    canvas.paste(fmtt_img, (width + 10, label_height))

    # Add labels
    draw = ImageDraw.Draw(canvas)

    # Try to use a nice font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
        small_font = font

    # Baseline label
    draw.text((10, 5), f"BASELINE", fill=(200, 200, 200), font=font)
    draw.text((10, 28), f"Reward: {baseline_reward:.4f}", fill=(150, 150, 150), font=small_font)

    # FMTT label
    improvement = fmtt_reward - baseline_reward
    color = (100, 255, 100) if improvement > 0 else (255, 100, 100)
    draw.text((width + 20, 5), f"FMTT (scale={guidance_scale})", fill=(200, 200, 200), font=font)
    draw.text((width + 20, 28), f"Reward: {fmtt_reward:.4f} ({improvement:+.4f})", fill=color, font=small_font)

    # Truncate prompt for display
    display_prompt = prompt[:80] + "..." if len(prompt) > 80 else prompt
    draw.text((10, 45), display_prompt, fill=(100, 100, 100), font=small_font)

    return canvas

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def flow_map_direct(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Predict clean latents from noisy using single-step flow map.

    For Z-Image flow matching:
        x_clean = x_t + velocity * sigma

    This is the single-step Euler approximation. It's sufficient because:
    - Turbo model has only 8-9 steps (each covers significant trajectory)
    - Even imperfect gradients nudge in the right direction
    - Direct approximation is essentially free (reuses DiT velocity)

    Args:
        x_t: Current noisy latents, shape (B, C, H, W)
        velocity: Raw velocity from DiT (before pipeline negation), shape (B, C, H, W)
        sigma: Current sigma value (noise level), in [0, 1]

    Returns:
        Predicted clean latents, shape (B, C, H, W)
    """
    return x_t + velocity * sigma


def compute_fmtt_gradient(
    latents: torch.Tensor,
    velocity: torch.Tensor,
    sigma: float,
    vae,
    reward_fn: DifferentiableSigLIP,
    prompt: str,
    normalize_mode: str = "unit",
    clip_value: float = 1.0,
    decode_scale: float = 0.5,
) -> tuple[torch.Tensor, float]:
    """Compute gradient of reward w.r.t. latents through flow map.

    This is the core FMTT computation:
    1. Flow map: latents -> predicted clean latents
    2. Downscale latents (saves ~4x VRAM during VAE decode)
    3. VAE decode: latents -> image
    4. Reward: image -> scalar
    5. Backprop: scalar -> gradient w.r.t. latents

    Args:
        latents: Current noisy latents
        velocity: DiT velocity prediction (raw, before negation)
        sigma: Current noise level
        vae: VAE decoder
        reward_fn: Differentiable reward function
        prompt: Text prompt for reward
        normalize_mode: Gradient normalization ("unit", "clip", "none")
        clip_value: Max gradient norm for "clip" mode
        decode_scale: Scale factor for intermediate decoding (0.5 = 512px for 1024px input)
                     Lower values save VRAM but reduce gradient precision

    Returns:
        Tuple of (gradient tensor, reward value)
    """
    # Enable gradients on latents
    latents_grad = latents.detach().requires_grad_(True)

    # Flow map prediction (gradients flow through)
    predicted_clean = flow_map_direct(latents_grad, velocity.detach(), sigma)

    # Downscale latents before VAE decode to save VRAM
    # The gradient will still flow back to full-resolution latents via interpolate
    if decode_scale < 1.0:
        h, w = predicted_clean.shape[-2:]
        new_h, new_w = int(h * decode_scale), int(w * decode_scale)
        predicted_clean_small = torch.nn.functional.interpolate(
            predicted_clean, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
    else:
        predicted_clean_small = predicted_clean

    # VAE decode (with gradients for backprop to latents)
    scaled = (predicted_clean_small / vae.config.scaling_factor) + vae.config.shift_factor
    image = vae.decode(scaled.to(vae.dtype)).sample

    # Reward computation
    reward = reward_fn.compute_reward(image, prompt)

    # Backprop to latents
    grad = torch.autograd.grad(
        reward.mean(),
        latents_grad,
        create_graph=False,
    )[0]

    # Check for numerical issues
    if grad.isnan().any() or grad.isinf().any():
        logger.warning("Gradient instability detected, returning zero gradient")
        return torch.zeros_like(grad), reward.mean().item()

    # Normalize gradient
    grad_norm = grad.norm()
    if normalize_mode == "unit":
        grad = grad / (grad_norm + 1e-8)
    elif normalize_mode == "clip" and grad_norm > clip_value:
        grad = grad * (clip_value / grad_norm)
    # "none" leaves gradient as-is

    return grad, reward.mean().item()


def fmtt_sampling(
    pipe,
    prompt: str,
    reward_fn: DifferentiableSigLIP,
    guidance_scale: float = 1.0,
    guidance_start: float = 0.0,
    guidance_stop: float = 0.5,
    normalize_mode: str = "unit",
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 9,
    generator: Optional[torch.Generator] = None,
    verbose: bool = True,
    decode_scale: float = 0.5,
) -> tuple[Image.Image, dict]:
    """FMTT-guided sampling for Z-Image.

    This modifies the standard denoising loop to:
    1. Predict where trajectory will end (flow map)
    2. Evaluate reward on prediction
    3. Compute gradient of reward w.r.t. current latents
    4. Add gradient to velocity to nudge toward higher reward

    Args:
        pipe: ZImagePipeline instance
        prompt: Text prompt
        reward_fn: Differentiable SigLIP2 reward function
        guidance_scale: FMTT guidance strength (0.5-2.0 typical)
        guidance_start: When to start FMTT (0.0 = from beginning)
        guidance_stop: When to stop FMTT (0.5 = first half only)
        normalize_mode: Gradient normalization mode
        height: Image height
        width: Image width
        num_inference_steps: Number of denoising steps
        generator: Random generator for reproducibility
        verbose: Whether to log per-step info

    Returns:
        Tuple of (PIL Image, metrics dict)
    """
    device = pipe.device
    dtype = next(pipe.transformer.parameters()).dtype

    # Encode prompt
    logger.info(f"Encoding prompt: '{prompt[:50]}...'")
    prompt_output = pipe.encoder.encode(prompt, force_think_block=True)
    prompt_embeds = [prompt_output.embeddings[0].to(device=device, dtype=dtype)]

    # Initialize latents
    vae_scale = pipe.vae_scale_factor * 2
    latent_h = 2 * (height // vae_scale)
    latent_w = 2 * (width // vae_scale)
    num_channels = pipe.transformer.config.in_channels

    latents = torch.randn(
        (1, num_channels, latent_h, latent_w),
        generator=generator,
        dtype=torch.float32,
        device=device,
    )

    # Setup scheduler
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # Get sigmas for flow map
    sigmas = pipe.scheduler.sigmas

    # Tracking
    rewards = []
    grad_norms = []
    step_times = []

    if verbose:
        logger.info(f"Starting FMTT sampling: {num_inference_steps} steps")
        logger.info(
            f"Guidance: scale={guidance_scale}, range=[{guidance_start:.0%}, {guidance_stop:.0%}], normalize={normalize_mode}"
        )

    for i, t in enumerate(timesteps):
        step_start = time.time()
        progress = i / num_inference_steps
        apply_fmtt = guidance_start <= progress < guidance_stop

        # Get sigma for flow map
        sigma = sigmas[i].item()

        # Prepare inputs for transformer
        timestep = t.expand(latents.shape[0])
        timestep_normalized = (1000 - timestep) / 1000

        latent_input = latents.to(dtype).unsqueeze(2)
        latent_list = list(latent_input.unbind(dim=0))

        if apply_fmtt:
            # FMTT path: compute gradient and apply guidance
            #
            # NOTE: We use no_grad for the DiT forward pass because:
            # 1. compute_fmtt_gradient() creates its own fresh computation graph
            # 2. It detaches latents and velocity, then builds: latents -> flow_map -> VAE -> SigLIP
            # 3. Running DiT under enable_grad() would store all activations (~20GB) causing OOM
            with torch.no_grad():
                model_output = pipe.transformer(
                    latent_list,
                    timestep_normalized,
                    prompt_embeds,
                )[0]

                # Extract velocity (before negation)
                velocity = torch.stack([o.float() for o in model_output], dim=0).squeeze(2)

            # Compute FMTT gradient (creates its own grad-enabled graph)
            grad, reward = compute_fmtt_gradient(
                latents,
                velocity,
                sigma,
                pipe.vae,
                reward_fn,
                prompt,
                normalize_mode=normalize_mode,
                decode_scale=decode_scale,
            )

            rewards.append(reward)
            grad_norms.append(grad.norm().item())

            # Apply guidance to velocity (before negation)
            # noise_pred = -(velocity + scale * grad)
            noise_pred = -(velocity + guidance_scale * grad)

            if verbose:
                logger.info(
                    f"Step {i + 1}/{num_inference_steps}: sigma={sigma:.4f}, "
                    f"reward={reward:.4f}, grad_norm={grad_norms[-1]:.4f}"
                )
        else:
            # Standard path (no FMTT)
            with torch.no_grad():
                model_output = pipe.transformer(
                    latent_list,
                    timestep_normalized,
                    prompt_embeds,
                )[0]

                velocity = torch.stack([o.float() for o in model_output], dim=0).squeeze(2)
                noise_pred = -velocity

        # Scheduler step
        latents = pipe.scheduler.step(
            noise_pred.to(torch.float32),
            t,
            latents,
            return_dict=False,
        )[0]

        step_times.append(time.time() - step_start)

    # Decode final image
    logger.info("Decoding final image...")
    with torch.no_grad():
        latents_decoded = latents.to(pipe.vae.dtype)
        latents_decoded = (latents_decoded / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latents_decoded).sample[0]

    # Convert to PIL
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(1, 2, 0).float().numpy()
    image = (image * 255).round().astype("uint8")
    pil_image = Image.fromarray(image)

    # Compute final reward on actual output
    with torch.no_grad():
        final_image_tensor = torch.from_numpy(image.astype("float32") / 255.0)
        final_image_tensor = final_image_tensor.permute(2, 0, 1).unsqueeze(0)
        final_image_tensor = final_image_tensor * 2 - 1  # [0,1] -> [-1,1]
        final_image_tensor = final_image_tensor.to(device)
        final_reward = reward_fn.compute_reward(final_image_tensor, prompt).item()

    metrics = {
        "intermediate_rewards": rewards,
        "mean_intermediate_reward": sum(rewards) / len(rewards) if rewards else 0,
        "final_reward": final_reward,
        "grad_norms": grad_norms,
        "mean_grad_norm": sum(grad_norms) / len(grad_norms) if grad_norms else 0,
        "guidance_steps": len(rewards),
        "total_steps": num_inference_steps,
        "step_times": step_times,
        "mean_step_time": sum(step_times) / len(step_times),
        "total_time": sum(step_times),
    }

    return pil_image, metrics


def generate_baseline(
    pipe,
    prompt: str,
    reward_fn: DifferentiableSigLIP,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 9,
    generator: Optional[torch.Generator] = None,
) -> tuple[Image.Image, float]:
    """Generate baseline image without FMTT for comparison."""
    start = time.time()

    image = pipe(
        prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        generator=generator,
        force_think_block=True,
    )

    elapsed = time.time() - start

    # Compute reward
    device = pipe.device
    with torch.no_grad():
        import numpy as np

        image_tensor = torch.from_numpy(np.array(image).astype("float32") / 255.0)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor * 2 - 1
        image_tensor = image_tensor.to(device)
        reward = reward_fn.compute_reward(image_tensor, prompt).item()

    return image, reward, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="FMTT Proof-of-Concept for Z-Image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using standard prompt by ID (from experiments/prompts/standard_prompts.yaml)
    uv run experiments/fmtt/poc.py --config config.toml --profile rtx4090 --prompt-id technical_001

    # Run all prompts in a category
    uv run experiments/fmtt/poc.py --config config.toml --profile rtx4090 --prompt-category technical

    # Custom prompt
    uv run experiments/fmtt/poc.py --config config.toml --profile rtx4090 "A cat"

    # With baseline comparison
    uv run experiments/fmtt/poc.py --config config.toml --profile rtx4090 --compare-baseline --prompt-id animal_002

    # Sweep guidance scale
    uv run experiments/fmtt/poc.py --config config.toml --profile rtx4090 --sweep-scale --prompt-id technical_001
        """,
    )
    parser.add_argument("prompt", nargs="?", default=None, help="Text prompt (or use --prompt-id)")
    parser.add_argument("--config", type=str, help="Config file (e.g., config.toml)")
    parser.add_argument("--profile", type=str, default="default", help="Config profile (default: default)")
    parser.add_argument("--model-path", type=str, help="Path to Z-Image model (alternative to config)")
    parser.add_argument("--prompt-id", type=str, help="Prompt ID from standard_prompts.yaml (e.g., technical_001, animal_002)")
    parser.add_argument("--prompt-category", type=str, choices=get_categories(), help="Run all prompts in category")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="FMTT guidance scale")
    parser.add_argument("--guidance-start", type=float, default=0.0, help="When to start FMTT (0.0-1.0)")
    parser.add_argument("--guidance-stop", type=float, default=0.5, help="When to stop FMTT (0.0-1.0)")
    parser.add_argument("--normalize-mode", type=str, default="unit", choices=["unit", "clip", "none"])
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="experiments/results/fmtt")
    parser.add_argument("--compare-baseline", action="store_true", help="Also generate baseline for comparison")
    parser.add_argument("--sweep-scale", action="store_true", help="Sweep guidance scale values")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    parser.add_argument("--siglip-device", type=str, default="cuda", help="Device for SigLIP (cuda/cpu)")
    parser.add_argument("--encoder-cpu", action="store_true", default=True, help="Run text encoder on CPU (default: True, required for 24GB cards)")
    parser.add_argument("--encoder-cuda", action="store_true", help="Run text encoder on CUDA (only for 48GB+ cards)")
    parser.add_argument("--decode-scale", type=float, default=0.5, help="Scale for intermediate VAE decode (0.5=512px, saves VRAM)")

    args = parser.parse_args()

    # Determine prompt(s)
    prompts_to_run = []

    if args.prompt_category:
        # Run all prompts in category
        category_prompts = get_prompts_by_category(args.prompt_category)
        for p in category_prompts:
            prompts_to_run.append((p["id"], p["prompt"]))
        logger.info(f"Running {len(prompts_to_run)} prompts from category: {args.prompt_category}")
    elif args.prompt_id:
        # Use standard prompt by ID
        prompt_data = get_prompt_by_id(args.prompt_id)
        if prompt_data is None:
            parser.error(f"Prompt ID '{args.prompt_id}' not found. Use --help to see categories.")
        prompts_to_run.append((args.prompt_id, prompt_data["prompt"]))
    elif args.prompt:
        # Custom prompt
        prompts_to_run.append(("custom", args.prompt))
    else:
        parser.error("Provide a prompt, --prompt-id, or --prompt-category")

    # Handle encoder device (--encoder-cuda overrides default --encoder-cpu)
    encoder_on_cpu = args.encoder_cpu and not args.encoder_cuda

    # For single prompt, log it
    if len(prompts_to_run) == 1:
        prompt_name, prompt = prompts_to_run[0]
        logger.info(f"Prompt [{prompt_name}]: '{prompt[:80]}...'" if len(prompt) > 80 else f"Prompt [{prompt_name}]: '{prompt}'")

    # Load pipeline - either from config or direct model path
    if args.config:
        logger.info(f"Loading Z-Image pipeline from config: {args.config} (profile: {args.profile})")
        from llm_dit.config import Config
        from llm_dit.pipelines.z_image import ZImagePipeline

        # Load TOML config
        config = Config.from_toml(args.config, args.profile)

        # Extract values from config
        model_path = config.model_path
        templates_dir = config.templates_dir
        encoder_device = "cpu" if encoder_on_cpu else config.encoder.device
        hidden_layer = config.encoder.hidden_layer

        # Resolve dtype
        dtype_str = config.encoder.torch_dtype
        if dtype_str == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype_str == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        logger.info(f"  Model: {model_path}")
        logger.info(f"  Encoder device: {encoder_device}")
        logger.info(f"  Hidden layer: {hidden_layer}")
        logger.info(f"  SLG/DyPE: N/A (custom sampling loop)")

        pipe = ZImagePipeline.from_pretrained(
            model_path,
            templates_dir=templates_dir,
            torch_dtype=torch_dtype,
            encoder_device=encoder_device,
            hidden_layer=hidden_layer,
        )

        # Use config values for height/width/steps if not overridden
        if args.height == 1024:
            args.height = config.generation.height
        if args.width == 1024:
            args.width = config.generation.width
        if args.steps == 9:
            args.steps = config.generation.num_inference_steps

    elif args.model_path:
        logger.info(f"Loading Z-Image pipeline from {args.model_path}...")
        from llm_dit.pipelines.z_image import ZImagePipeline

        pipe = ZImagePipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
        )
    else:
        parser.error("Either --config or --model-path is required")

    # Load reward function
    logger.info(f"Loading SigLIP2 reward function on {args.siglip_device}...")
    reward_fn = DifferentiableSigLIP(device=args.siglip_device)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run generation for each prompt
    all_results = []

    for prompt_name, prompt in prompts_to_run:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {prompt_name}")
        logger.info(f"{'='*60}")

        if args.sweep_scale:
            # Sweep guidance scale
            scales = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0]
            logger.info(f"Sweeping guidance scales: {scales}")

            results = []
            for scale in scales:
                generator = torch.Generator(device=pipe.device).manual_seed(args.seed)

                image, metrics = fmtt_sampling(
                    pipe,
                    prompt,
                    reward_fn,
                    guidance_scale=scale,
                    guidance_start=args.guidance_start,
                    guidance_stop=args.guidance_stop,
                    normalize_mode=args.normalize_mode,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.steps,
                    generator=generator,
                    verbose=not args.quiet,
                    decode_scale=args.decode_scale,
                )

                image.save(output_dir / f"fmtt_{prompt_name}_scale{scale:.1f}.png")
                results.append({"scale": scale, "reward": metrics["final_reward"]})
                logger.info(f"Scale {scale}: reward={metrics['final_reward']:.4f}")

            # Find best
            best = max(results, key=lambda x: x["reward"])
            logger.info(f"Best scale: {best['scale']} (reward={best['reward']:.4f})")
            all_results.append({"prompt_id": prompt_name, "best_scale": best["scale"], "best_reward": best["reward"]})

        else:
            # Single generation
            generator = torch.Generator(device=pipe.device).manual_seed(args.seed)

            # Generate with FMTT
            logger.info("Generating with FMTT...")
            fmtt_image, fmtt_metrics = fmtt_sampling(
                pipe,
                prompt,
                reward_fn,
                guidance_scale=args.guidance_scale,
                guidance_start=args.guidance_start,
                guidance_stop=args.guidance_stop,
                normalize_mode=args.normalize_mode,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                generator=generator,
                verbose=not args.quiet,
                decode_scale=args.decode_scale,
            )

            fmtt_image.save(output_dir / f"fmtt_{prompt_name}.png")
            logger.info(f"FMTT saved to: {output_dir / f'fmtt_{prompt_name}.png'}")
            logger.info(f"FMTT final reward: {fmtt_metrics['final_reward']:.4f}")
            logger.info(f"FMTT mean intermediate reward: {fmtt_metrics['mean_intermediate_reward']:.4f}")
            logger.info(f"FMTT total time: {fmtt_metrics['total_time']:.2f}s")

            result = {
                "prompt_id": prompt_name,
                "fmtt_reward": fmtt_metrics["final_reward"],
                "fmtt_time": fmtt_metrics["total_time"],
            }

            if args.compare_baseline:
                # Generate baseline (use CPU generator - pipeline moves tensors to GPU)
                generator = torch.Generator(device="cpu").manual_seed(args.seed)

                logger.info("Generating baseline (no FMTT)...")
                baseline_image, baseline_reward, baseline_time = generate_baseline(
                    pipe,
                    prompt,
                    reward_fn,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.steps,
                    generator=generator,
                )

                baseline_image.save(output_dir / f"baseline_{prompt_name}.png")
                logger.info(f"Baseline saved to: {output_dir / f'baseline_{prompt_name}.png'}")
                logger.info(f"Baseline reward: {baseline_reward:.4f}")
                logger.info(f"Baseline time: {baseline_time:.2f}s")

                # Comparison
                improvement = fmtt_metrics["final_reward"] - baseline_reward
                result["baseline_reward"] = baseline_reward
                result["improvement"] = improvement

                logger.info("=" * 50)
                logger.info("COMPARISON:")
                logger.info(f"  Reward improvement: {improvement:+.4f}")
                logger.info(f"  FMTT / Baseline time: {fmtt_metrics['total_time']:.2f}s / {baseline_time:.2f}s")
                logger.info(f"  Time ratio: {fmtt_metrics['total_time'] / baseline_time:.2f}x")
                logger.info("=" * 50)

                # Save side-by-side comparison image
                comparison = create_comparison_image(
                    baseline_image,
                    fmtt_image,
                    baseline_reward,
                    fmtt_metrics["final_reward"],
                    prompt,
                    args.guidance_scale,
                )
                comparison_path = output_dir / f"compare_{prompt_name}.png"
                comparison.save(comparison_path)
                logger.info(f"Comparison saved to: {comparison_path}")

            all_results.append(result)

    # Summary for category runs
    if len(all_results) > 1:
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info(f"{'='*60}")
        for r in all_results:
            if "improvement" in r:
                logger.info(f"{r['prompt_id']}: FMTT={r['fmtt_reward']:.4f}, baseline={r['baseline_reward']:.4f}, delta={r['improvement']:+.4f}")
            elif "best_scale" in r:
                logger.info(f"{r['prompt_id']}: best_scale={r['best_scale']}, reward={r['best_reward']:.4f}")
            else:
                logger.info(f"{r['prompt_id']}: FMTT={r['fmtt_reward']:.4f}")


if __name__ == "__main__":
    main()
