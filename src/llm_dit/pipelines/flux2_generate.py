"""
FLUX.2 Klein Image Generation Pipeline.

Last Updated: 2026-01-23

Pure PyTorch implementation of FLUX.2 Klein image generation with
three-stage offloading for memory efficiency on consumer GPUs.

Supports:
- Text-to-image generation
- Image editing with reference images (N input images)

Stages:
1. Text Encoding: Qwen3 encoder extracts multi-layer embeddings
2. Reference Encoding: VAE-encode input images (if provided)
3. Denoising: FLUX.2 transformer performs diffusion denoising
4. VAE Decode: AutoEncoder converts latents to pixels

Memory Optimization:
- Three-stage offloading: Only one component on GPU at a time
- Peak VRAM = max(encoder, transformer, vae) instead of sum
- Suitable for 24GB GPUs with Klein 9B BF16

Ported from: coderef/flux2/src/flux2/sampling.py

Usage:
    # Text-to-image
    from llm_dit.pipelines.flux2_generate import generate_image, Flux2GenerationConfig

    config = Flux2GenerationConfig(
        prompt="A photo of a cat",
        height=1024,
        width=1024,
        num_steps=4,
    )
    image = generate_image(config, model_name="klein-9b")
    image.save("output.png")

    # Image editing with reference images
    from PIL import Image
    ref_images = [Image.open("input1.jpg"), Image.open("input2.jpg")]
    config = Flux2GenerationConfig(
        prompt="Make the cat wear a hat",
        reference_images=ref_images,
    )
    image = generate_image(config, model_name="klein-9b")
"""

import gc
import math
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torchvision
from einops import rearrange
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
    """Configuration for FLUX.2 image generation.

    Supports both text-to-image and image editing modes.
    For image editing, provide reference_images with input images.
    """

    prompt: str
    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH
    num_steps: int = DEFAULT_NUM_STEPS_DISTILLED
    guidance: float = DEFAULT_GUIDANCE_DISTILLED
    seed: Optional[int] = DEFAULT_SEED

    # Reference images for editing (list of PIL Images or file paths)
    reference_images: list[Image.Image | str] = field(default_factory=list)

    # Pixel limits for reference images (higher = more detail, more VRAM)
    # Single ref: up to 2024^2, multiple refs: up to 1024^2 each
    ref_limit_pixels: Optional[int] = None

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

    @property
    def is_editing_mode(self) -> bool:
        """Whether reference images are provided for editing."""
        return len(self.reference_images) > 0


# =============================================================================
# Reference Image Processing (for editing mode)
# =============================================================================


def load_reference_images(config: Flux2GenerationConfig) -> list[Image.Image]:
    """Load and prepare reference images for encoding.

    Args:
        config: Generation config with reference_images field

    Returns:
        List of PIL Images ready for encoding
    """
    images = []
    for ref in config.reference_images:
        if isinstance(ref, str):
            # Load from file path
            img = Image.open(ref).convert("RGB")
        else:
            img = ref.convert("RGB") if ref.mode != "RGB" else ref
        images.append(img)
    return images


def preprocess_reference_image(
    img: Image.Image,
    limit_pixels: Optional[int] = None,
    ensure_multiple: int = 16,
) -> torch.Tensor:
    """Preprocess a single reference image for VAE encoding.

    Args:
        img: PIL Image
        limit_pixels: Maximum total pixels (resizes if exceeded)
        ensure_multiple: Ensure dimensions are multiples of this

    Returns:
        Tensor in [-1, 1] range, shape [3, H, W]
    """
    # Ensure RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Cap pixels if needed
    if limit_pixels is not None:
        w, h = img.size
        if w * h > limit_pixels:
            scale = math.sqrt(limit_pixels / (w * h))
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Center crop to multiple of ensure_multiple
    w, h = img.size
    new_w = (w // ensure_multiple) * ensure_multiple
    new_h = (h // ensure_multiple) * ensure_multiple
    if new_w != w or new_h != h:
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        img = img.crop((left, top, left + new_w, top + new_h))

    # Convert to tensor in [-1, 1]
    tensor = torchvision.transforms.ToTensor()(img)
    return 2 * tensor - 1


def encode_reference_images(
    vae,
    images: list[Image.Image],
    limit_pixels: Optional[int] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """Encode reference images into latent tokens with position IDs.

    Each reference image gets a unique time coordinate (t=10, t=20, etc.)
    to distinguish it from the generated image (t=0).

    Args:
        vae: FLUX.2 VAE
        images: List of PIL Images
        limit_pixels: Maximum pixels per image
        device: Target device
        dtype: Target dtype

    Returns:
        Tuple of (ref_tokens, ref_ids) or (None, None) if no images
        - ref_tokens: [1, total_ref_tokens, 128]
        - ref_ids: [1, total_ref_tokens, 4]
    """
    if not images:
        return None, None

    # Set pixel limit based on number of images
    if limit_pixels is None:
        if len(images) > 1:
            limit_pixels = 1024**2  # 1MP each for multiple refs
        else:
            limit_pixels = 2024**2  # 4MP for single ref

    # Time offset scale for reference images
    t_scale = 10  # Each ref gets t=10, t=20, etc.

    encoded_refs = []
    ref_ids_list = []

    for idx, img in enumerate(images):
        # Preprocess
        img_tensor = preprocess_reference_image(img, limit_pixels=limit_pixels)
        img_tensor = img_tensor.unsqueeze(0).to(device).to(dtype)  # [1, 3, H, W]

        # VAE encode
        with torch.no_grad():
            latent = vae.encode(img_tensor)  # [1, 128, H/16, W/16]

        # Reshape to sequence: [1, H*W, 128]
        b, c, h, w = latent.shape
        latent_seq = rearrange(latent, "b c h w -> b (h w) c")

        # Create position IDs with unique time coordinate
        t_coord = torch.tensor([t_scale + t_scale * idx], device=device)
        ids = _create_ref_image_ids(h, w, t_coord, device)  # [h*w, 4]

        encoded_refs.append(latent_seq)
        ref_ids_list.append(ids)

    # Concatenate all references
    ref_tokens = torch.cat(encoded_refs, dim=1)  # [1, total_tokens, 128]
    ref_ids = torch.cat(ref_ids_list, dim=0).unsqueeze(0)  # [1, total_tokens, 4]

    return ref_tokens.to(dtype), ref_ids.to(torch.float32)


def _create_ref_image_ids(
    h: int,
    w: int,
    t_coord: torch.Tensor,
    device: torch.device | str,
) -> torch.Tensor:
    """Create 4D position IDs for a reference image.

    Args:
        h: Latent height
        w: Latent width
        t_coord: Time coordinate tensor
        device: Target device

    Returns:
        Position IDs [h*w, 4] with (t, h, w, l) coordinates
    """
    coords = {
        "t": t_coord,
        "h": torch.arange(h, device=device),
        "w": torch.arange(w, device=device),
        "l": torch.arange(1, device=device),
    }
    # Cartesian product: (t, h, w, l) for all spatial positions
    ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    return ids


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
    img_cond_seq: torch.Tensor | None = None,
    img_cond_seq_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    FLUX.2 denoising loop with flow matching.

    Supports optional reference image conditioning for editing mode.

    Args:
        model: FLUX.2 transformer
        img: Initial noise [B, seq_len, channels]
        img_ids: Image position IDs [B, seq_len, 4]
        txt: Text embeddings [B, txt_len, context_dim]
        txt_ids: Text position IDs [B, txt_len, 4]
        timesteps: List of timesteps from ~1 to ~0
        guidance: Guidance scale (None for distilled models)
        img_cond_seq: Reference image tokens [B, ref_tokens, channels] (optional)
        img_cond_seq_ids: Reference image position IDs [B, ref_tokens, 4] (optional)

    Returns:
        Denoised latents [B, seq_len, channels]
    """
    # Prepare guidance vector (for non-distilled models)
    guidance_vec = None
    if guidance is not None:
        guidance_vec = torch.full(
            (img.shape[0],), guidance, device=img.device, dtype=img.dtype
        )

    num_img_tokens = img.shape[1]

    for t_curr, t_prev in tqdm(zip(timesteps[:-1], timesteps[1:]), total=len(timesteps) - 1, desc="Denoising"):
        # Current timestep vector
        t_vec = torch.full(
            (img.shape[0],), t_curr, dtype=img.dtype, device=img.device
        )

        # Prepare input - concatenate reference tokens if provided
        img_input = img
        img_input_ids = img_ids
        if img_cond_seq is not None:
            assert img_cond_seq_ids is not None, "Must provide both img_cond_seq and img_cond_seq_ids"
            img_input = torch.cat([img, img_cond_seq], dim=1)
            img_input_ids = torch.cat([img_ids, img_cond_seq_ids], dim=1)

        # Predict velocity
        pred = model(
            x=img_input,
            x_ids=img_input_ids,
            timesteps=t_vec,
            ctx=txt,
            ctx_ids=txt_ids,
            guidance=guidance_vec,
        )

        # Only take prediction for noise tokens (not reference tokens)
        if img_cond_seq is not None:
            pred = pred[:, :num_img_tokens]

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
    encoder_path: Optional[str] = None,
    model_path: Optional[str] = None,
    vae_path: Optional[str] = None,
) -> Image.Image:
    """
    Generate an image using FLUX.2 Klein.

    Supports both text-to-image and image editing with reference images.
    Uses three-stage offloading to minimize peak VRAM usage.

    Args:
        config: Generation configuration (can include reference_images for editing)
        model_name: Model variant ("klein-4b", "klein-9b", "klein-9b-fp8", etc.)
        encoder: Pre-loaded encoder (optional)
        transformer: Pre-loaded transformer (optional)
        vae: Pre-loaded VAE (optional)
        encoder_path: Custom path for text encoder (overrides model default, auto-detects dtype)
        model_path: Local path to transformer weights (file or directory)
        vae_path: Local path to VAE weights (file or directory)

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

    mode = "editing" if config.is_editing_mode else "text-to-image"
    logger.info(f"Generating {config.width}x{config.height} image ({mode} mode) with {config.num_steps} steps")

    # ===========================================================================
    # Stage 1: Text Encoding
    # ===========================================================================
    logger.info("Stage 1: Encoding text...")

    if encoder is None:
        from llm_dit.encoders.qwen3_flux2 import Qwen3Flux2Encoder

        # Use custom encoder path if provided, otherwise use model default
        model_info = FLUX2_MODEL_INFO[model_name.lower()]
        text_encoder_spec = encoder_path or model_info["text_encoder"]
        logger.info(f"Loading encoder from: {text_encoder_spec}")
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
    # Stage 1.5: Reference Image Encoding (if editing mode)
    # ===========================================================================
    ref_tokens = None
    ref_ids = None

    if config.is_editing_mode:
        logger.info(f"Stage 1.5: Encoding {len(config.reference_images)} reference image(s)...")

        # Load VAE for encoding (will be reused for decoding)
        if vae is None:
            from llm_dit.models.flux2.loader import load_flux2_vae
            vae = load_flux2_vae(model_name, device=config.device, dtype=dtype, vae_path=vae_path)

        # Load and encode reference images
        ref_images = load_reference_images(config)
        ref_tokens, ref_ids = encode_reference_images(
            vae=vae,
            images=ref_images,
            limit_pixels=config.ref_limit_pixels,
            device=config.device,
            dtype=dtype,
        )

        if ref_tokens is not None:
            logger.info(f"Encoded reference images: {ref_tokens.shape[1]} tokens")

        # Optionally offload VAE (will reload for decode)
        if config.offload_between_stages:
            logger.info("Offloading VAE (will reload for decode)...")
            del vae
            vae = None
            cleanup_memory()

    # ===========================================================================
    # Stage 2: Denoising
    # ===========================================================================
    logger.info("Stage 2: Denoising...")

    if transformer is None:
        from llm_dit.models.flux2.loader import load_flux2_transformer

        transformer = load_flux2_transformer(model_name, device=config.device, dtype=dtype, model_path=model_path)

    # Move embeddings to device
    txt_embeddings = txt_embeddings.to(device)
    txt_ids = txt_ids.to(device)

    # Move reference tokens to device if present
    if ref_tokens is not None:
        ref_tokens = ref_tokens.to(device)
        ref_ids = ref_ids.to(device)

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

    # Denoise with optional reference image conditioning
    latents = denoise(
        model=transformer,
        img=img,
        img_ids=img_ids,
        txt=txt_embeddings,
        txt_ids=txt_ids,
        timesteps=timesteps,
        guidance=guidance,
        img_cond_seq=ref_tokens,
        img_cond_seq_ids=ref_ids,
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

        vae = load_flux2_vae(model_name, device=config.device, dtype=dtype, vae_path=vae_path)

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
    encoder_path: Optional[str] = None,
    model_path: Optional[str] = None,
    vae_path: Optional[str] = None,
) -> Image.Image:
    """
    Quick generation helper with minimal configuration.

    Args:
        prompt: Text prompt
        model_name: Model variant (e.g., "klein-9b", "klein-9b-fp8")
        height: Image height
        width: Image width
        seed: Random seed
        encoder_path: Custom encoder path (auto-detects dtype)
        model_path: Local path to transformer weights
        vae_path: Local path to VAE weights

    Returns:
        Generated PIL Image
    """
    config = Flux2GenerationConfig(
        prompt=prompt,
        height=height,
        width=width,
        seed=seed,
    )
    return generate_image(
        config,
        model_name=model_name,
        encoder_path=encoder_path,
        model_path=model_path,
        vae_path=vae_path,
    )
