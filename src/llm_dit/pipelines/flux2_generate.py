"""
FLUX.2 Klein Image Generation Pipeline.

Last Updated: 2026-01-23

Pure PyTorch implementation of FLUX.2 Klein image generation with
three-stage offloading for memory efficiency on consumer GPUs.

Stages:
1. Text Encoding: Qwen3 encoder extracts multi-layer embeddings
2. Denoising: FLUX.2 transformer performs diffusion denoising
3. VAE Decode: AutoEncoder converts latents to pixels

Memory Optimization:
- Three-stage offloading: Only one component on GPU at a time
- Peak VRAM = max(encoder, transformer, vae) instead of sum
- Suitable for 24GB GPUs with Klein 9B BF16

Ported from: coderef/flux2/src/flux2/sampling.py

Usage:
    from llm_dit.pipelines.flux2_generate import generate_image, Flux2GenerationConfig

    config = Flux2GenerationConfig(
        prompt="A photo of a cat",
        height=1024,
        width=1024,
        num_steps=4,
    )
    image = generate_image(config, model_name="klein-9b")
    image.save("output.png")
"""

import gc
import math
import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torchvision
from PIL import Image
from tqdm import tqdm

from llm_dit.models.flux2.rope import create_image_ids, create_text_ids
from llm_dit.models.flux2.constants import (
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_NUM_STEPS_DISTILLED,
    DEFAULT_GUIDANCE_DISTILLED,
    DEFAULT_SEED,
    TOTAL_SPATIAL_COMPRESSION,
    LATENT_CHANNELS_AFTER_PATCHIFY,
    FLUX2_MODEL_INFO,
)

logger = logging.getLogger(__name__)


def cleanup_memory() -> None:
    """Free GPU memory between stages."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@dataclass
class Flux2GenerationConfig:
    """Configuration for FLUX.2 image generation."""

    prompt: str
    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH
    num_steps: int = DEFAULT_NUM_STEPS_DISTILLED
    guidance: float = DEFAULT_GUIDANCE_DISTILLED
    seed: Optional[int] = DEFAULT_SEED

    # Device and dtype
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    # Offloading
    offload_between_stages: bool = True

    @property
    def latent_height(self) -> int:
        """Latent height after patchify (16x compression)."""
        return self.height // TOTAL_SPATIAL_COMPRESSION

    @property
    def latent_width(self) -> int:
        """Latent width after patchify (16x compression)."""
        return self.width // TOTAL_SPATIAL_COMPRESSION

    @property
    def num_tokens(self) -> int:
        """Number of image tokens."""
        return self.latent_height * self.latent_width


def generalized_time_snr_shift(t: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """
    Apply SNR-based timestep shift.

    Args:
        t: Linear timesteps in [0, 1]
        mu: Shift parameter (computed from image size)
        sigma: Scale parameter (typically 1.0)

    Returns:
        Shifted timesteps
    """
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """
    Compute empirical mu parameter for timestep shifting.

    Higher resolution images need different shift schedules.

    Args:
        image_seq_len: Number of image tokens
        num_steps: Number of denoising steps

    Returns:
        Computed mu value
    """
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


def get_schedule(num_steps: int, image_seq_len: int) -> list[float]:
    """
    Generate timestep schedule with SNR-based shifting.

    Args:
        num_steps: Number of denoising steps
        image_seq_len: Number of image tokens

    Returns:
        List of timesteps from 1.0 to ~0
    """
    mu = compute_empirical_mu(image_seq_len, num_steps)
    timesteps = torch.linspace(1, 0, num_steps + 1)
    timesteps = generalized_time_snr_shift(timesteps, mu, 1.0)
    return timesteps.tolist()


def denoise(
    model,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    timesteps: list[float],
    guidance: float | None = None,
) -> torch.Tensor:
    """
    FLUX.2 denoising loop with flow matching.

    Args:
        model: FLUX.2 transformer
        img: Initial noise [B, seq_len, channels]
        img_ids: Image position IDs [B, seq_len, 4]
        txt: Text embeddings [B, txt_len, context_dim]
        txt_ids: Text position IDs [B, txt_len, 4]
        timesteps: List of timesteps from ~1 to ~0
        guidance: Guidance scale (None for distilled models)

    Returns:
        Denoised latents [B, seq_len, channels]
    """
    # Prepare guidance vector (for non-distilled models)
    guidance_vec = None
    if guidance is not None:
        guidance_vec = torch.full(
            (img.shape[0],), guidance, device=img.device, dtype=img.dtype
        )

    for t_curr, t_prev in tqdm(zip(timesteps[:-1], timesteps[1:]), total=len(timesteps) - 1, desc="Denoising"):
        # Current timestep vector
        t_vec = torch.full(
            (img.shape[0],), t_curr, dtype=img.dtype, device=img.device
        )

        # Predict velocity
        pred = model(
            x=img,
            x_ids=img_ids,
            timesteps=t_vec,
            ctx=txt,
            ctx_ids=txt_ids,
            guidance=guidance_vec,
        )

        # Euler step: x_{t-1} = x_t + (t_prev - t_curr) * v
        img = img + (t_prev - t_curr) * pred

    return img


def latents_to_image(latents: torch.Tensor, vae) -> Image.Image:
    """
    Decode latents to PIL Image.

    Args:
        latents: Patchified latents [B, seq_len, C] or [B, C, H, W]
        vae: FLUX.2 VAE

    Returns:
        PIL Image
    """
    # Reshape from sequence to spatial if needed
    if latents.ndim == 3:
        # [B, seq_len, C] -> [B, C, H, W]
        b, seq_len, c = latents.shape
        h = w = int(math.sqrt(seq_len))
        latents = latents.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    # Decode
    with torch.no_grad():
        pixels = vae.decode(latents)

    # Convert to PIL
    # pixels is in [-1, 1], convert to [0, 1]
    pixels = (pixels + 1) / 2
    pixels = pixels.clamp(0, 1)

    # Convert to PIL Image
    pixels = pixels[0]  # Remove batch dimension
    return torchvision.transforms.functional.to_pil_image(pixels)


def generate_image(
    config: Flux2GenerationConfig,
    model_name: str = "klein-9b",
    encoder=None,
    transformer=None,
    vae=None,
) -> Image.Image:
    """
    Generate an image using FLUX.2 Klein.

    Uses three-stage offloading to minimize peak VRAM usage.

    Args:
        config: Generation configuration
        model_name: Model variant ("klein-4b", "klein-9b")
        encoder: Pre-loaded encoder (optional)
        transformer: Pre-loaded transformer (optional)
        vae: Pre-loaded VAE (optional)

    Returns:
        Generated PIL Image
    """
    device = torch.device(config.device)
    dtype = config.dtype

    # Set seed for reproducibility
    if config.seed is not None:
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

    logger.info(f"Generating {config.width}x{config.height} image with {config.num_steps} steps")

    # ===========================================================================
    # Stage 1: Text Encoding
    # ===========================================================================
    logger.info("Stage 1: Encoding text...")

    if encoder is None:
        from llm_dit.encoders.qwen3_flux2 import Qwen3Flux2Encoder

        model_info = FLUX2_MODEL_INFO[model_name.lower()]
        text_encoder_spec = model_info["text_encoder"]
        encoder = Qwen3Flux2Encoder.from_pretrained(text_encoder_spec, device=config.device)

    # Encode text
    txt_embeddings = encoder.encode([config.prompt])  # [1, seq_len, context_dim]
    txt_embeddings = txt_embeddings.to(dtype)

    # Create text position IDs
    txt_ids = create_text_ids(
        batch_size=1,
        seq_len=txt_embeddings.shape[1],
        device=device,
        dtype=torch.float32,
    )

    # Offload encoder
    if config.offload_between_stages:
        logger.info("Offloading encoder...")
        encoder.offload()
        del encoder
        cleanup_memory()

    # ===========================================================================
    # Stage 2: Denoising
    # ===========================================================================
    logger.info("Stage 2: Denoising...")

    if transformer is None:
        from llm_dit.models.flux2.loader import load_flux2_transformer

        transformer = load_flux2_transformer(model_name, device=config.device, dtype=dtype)

    # Move embeddings to device
    txt_embeddings = txt_embeddings.to(device)
    txt_ids = txt_ids.to(device)

    # Create initial noise
    img = torch.randn(
        1,
        config.num_tokens,
        LATENT_CHANNELS_AFTER_PATCHIFY,
        device=device,
        dtype=dtype,
    )

    # Create image position IDs
    img_ids = create_image_ids(
        batch_size=1,
        height=config.latent_height,
        width=config.latent_width,
        device=device,
        dtype=torch.float32,
    )

    # Get timestep schedule
    timesteps = get_schedule(config.num_steps, config.num_tokens)

    # Determine guidance (None for distilled models)
    guidance = None if FLUX2_MODEL_INFO[model_name.lower()]["distilled"] else config.guidance

    # Denoise
    latents = denoise(
        model=transformer,
        img=img,
        img_ids=img_ids,
        txt=txt_embeddings,
        txt_ids=txt_ids,
        timesteps=timesteps,
        guidance=guidance,
    )

    # Move latents to CPU and offload transformer
    latents = latents.cpu()
    if config.offload_between_stages:
        logger.info("Offloading transformer...")
        del transformer
        cleanup_memory()

    # ===========================================================================
    # Stage 3: VAE Decode
    # ===========================================================================
    logger.info("Stage 3: Decoding latents...")

    if vae is None:
        from llm_dit.models.flux2.loader import load_flux2_vae

        vae = load_flux2_vae(model_name, device=config.device, dtype=dtype)

    # Move latents back to device for decoding
    latents = latents.to(device).to(dtype)

    # Decode to image
    image = latents_to_image(latents, vae)

    if config.offload_between_stages:
        logger.info("Offloading VAE...")
        del vae
        cleanup_memory()

    logger.info("Generation complete!")
    return image


def quick_generate(
    prompt: str,
    model_name: str = "klein-9b",
    height: int = 1024,
    width: int = 1024,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Quick generation helper with minimal configuration.

    Args:
        prompt: Text prompt
        model_name: Model variant
        height: Image height
        width: Image width
        seed: Random seed

    Returns:
        Generated PIL Image
    """
    config = Flux2GenerationConfig(
        prompt=prompt,
        height=height,
        width=width,
        seed=seed,
    )
    return generate_image(config, model_name=model_name)
