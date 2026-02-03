"""
FLUX.2 Klein Image Generation Pipeline.

Last Updated: 2026-01-24

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


def log_gpu_memory(stage: str) -> None:
    """Log current GPU memory usage for debugging."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.debug(f"[GPU] {stage}: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")


def log_memory_snapshot(label: str) -> None:
    """Log detailed memory snapshot including fragmentation analysis (P1 debug)."""
    if not torch.cuda.is_available():
        return
    stats = torch.cuda.memory_stats()
    allocated = stats.get("allocated_bytes.all.current", 0) / 1e9
    reserved = stats.get("reserved_bytes.all.current", 0) / 1e9
    frag_gap = reserved - allocated
    logger.debug(
        f"[{label}] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, "
        f"Fragmentation: {frag_gap:.2f}GB"
    )


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

    # LoRA weights: list of "path:scale" or just "path" (default scale 0.8)
    # Example: ["style.safetensors:0.7", "detail.safetensors:0.5"]
    loras: Optional[list[str]] = None

    # Text encoding options
    max_text_length: int = 512  # Maximum text tokens (can increase for longer prompts)
    pad_to_max: bool = True  # Whether to pad all sequences to max_text_length
    # Which 3 hidden layers to extract from Qwen3 (must be exactly 3)
    # Default: [9, 18, 27] for early/middle/late representations
    # Options: any 3 layer indices in [0, 27] for 8B or [0, 17] for 4B
    output_layers: Optional[list[int]] = None

    # Device and dtype
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    # Offloading
    offload_between_stages: bool = True
    block_offload: bool = False  # Block-by-block GPU offloading (slower but uses less VRAM)

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

    # Log original dimensions
    orig_w, orig_h = img.size
    logger.debug(f"[REF:Preprocess] Original dimensions: {orig_w}x{orig_h}")

    # Cap pixels if needed
    if limit_pixels is not None:
        w, h = img.size
        if w * h > limit_pixels:
            scale = math.sqrt(limit_pixels / (w * h))
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            logger.debug(f"[REF:Preprocess] After resize: {new_w}x{new_h} (limit={limit_pixels} pixels)")

    # Center crop to multiple of ensure_multiple
    w, h = img.size
    new_w = (w // ensure_multiple) * ensure_multiple
    new_h = (h // ensure_multiple) * ensure_multiple
    if new_w != w or new_h != h:
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        img = img.crop((left, top, left + new_w, top + new_h))
        logger.debug(f"[REF:Preprocess] After crop to multiple of {ensure_multiple}: {new_w}x{new_h}")

    # Convert to tensor in [-1, 1]
    tensor = torchvision.transforms.ToTensor()(img)
    logger.debug(f"[REF:Preprocess] Final tensor shape: {list(tensor.shape)}")
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

    logger.info(f"[REF:Encode] Encoding {len(images)} reference image(s)...")

    for idx, img in enumerate(images):
        logger.debug(f"[REF:Encode] Processing reference image {idx}...")

        # Preprocess
        img_tensor = preprocess_reference_image(img, limit_pixels=limit_pixels)
        img_tensor = img_tensor.unsqueeze(0).to(device).to(dtype)  # [1, 3, H, W]
        logger.debug(f"[REF:Encode] Image {idx} input tensor shape: {list(img_tensor.shape)}")

        # VAE encode
        with torch.no_grad():
            latent = vae.encode(img_tensor)  # [1, 128, H/16, W/16]

        logger.debug(f"[REF:Encode] Image {idx} latent shape after VAE: {list(latent.shape)}")

        # Reshape to sequence: [1, H*W, 128]
        b, c, h, w = latent.shape
        latent_seq = rearrange(latent, "b c h w -> b (h w) c")
        token_count = h * w
        logger.debug(f"[REF:Encode] Image {idx} latent h={h}, w={w}, token_count={token_count}")

        # Create position IDs with unique time coordinate
        t_coord = torch.tensor([t_scale + t_scale * idx], device=device)
        logger.debug(f"[REF:Encode] Image {idx} t_coord value: {t_coord.item()}")
        logger.debug(f"[REF:Encode] Image {idx} position ID range: h=[0,{h-1}], w=[0,{w-1}]")
        ids = _create_ref_image_ids(h, w, t_coord, device)  # [h*w, 4]
        logger.debug(f"[REF:Encode] Image {idx} position IDs first: {ids[0].tolist()}, last: {ids[-1].tolist()}")

        encoded_refs.append(latent_seq)
        ref_ids_list.append(ids)

    # Concatenate all references
    ref_tokens = torch.cat(encoded_refs, dim=1)  # [1, total_tokens, 128]
    ref_ids = torch.cat(ref_ids_list, dim=0).unsqueeze(0)  # [1, total_tokens, 4]

    logger.debug(f"[REF:Encode] Total ref tokens after concatenation: {ref_tokens.shape[1]}")
    logger.debug(f"[REF:Encode] Final ref_tokens shape: {list(ref_tokens.shape)}")
    logger.debug(f"[REF:Encode] Final ref_ids shape: {list(ref_ids.shape)}")

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
    logger.debug(f"[REF:IDs] Creating position IDs for h={h}, w={w}, t={t_coord.item()}")

    coords = {
        "t": t_coord,
        "h": torch.arange(h, device=device),
        "w": torch.arange(w, device=device),
        "l": torch.arange(1, device=device),
    }
    # Cartesian product: (t, h, w, l) for all spatial positions
    ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])

    logger.debug(f"[REF:IDs] Created {ids.shape[0]} position IDs")
    logger.debug(f"[REF:IDs] First 3 IDs: {ids[:3].tolist()}")
    logger.debug(f"[REF:IDs] Last 3 IDs: {ids[-3:].tolist()}")

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


def _format_gb(bytes_val: int | float) -> str:
    """Format bytes to GB string."""
    return f"{bytes_val / 1e9:.2f}GB"


def _log_denoise_memory(step: int, label: str = "") -> tuple[float, float]:
    """Log memory state during denoising. Returns (allocated, reserved) in bytes."""
    if not torch.cuda.is_available():
        return 0.0, 0.0

    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()

    if logger.isEnabledFor(logging.DEBUG):
        prefix = f"[Denoise:Step {step}:{label}]" if label else f"[Denoise:Step {step}]"
        logger.debug(f"{prefix} GPU: {_format_gb(allocated)} allocated, {_format_gb(reserved)} reserved")

    return allocated, reserved


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
    num_steps = len(timesteps) - 1
    num_img_tokens = img.shape[1]

    # =========================================================================
    # Denoising Loop Initialization - Granular Debug Logging
    # =========================================================================
    logger.debug("=" * 60)
    logger.debug("[Denoise] Starting denoising loop")
    logger.debug(f"[Denoise] num_steps={num_steps}, num_img_tokens={num_img_tokens}")
    logger.debug(f"[Denoise] img shape={img.shape}, dtype={img.dtype}, device={img.device}")
    logger.debug(f"[Denoise] txt shape={txt.shape}, dtype={txt.dtype}")
    logger.debug(f"[Denoise] img_ids shape={img_ids.shape}")
    logger.debug(f"[Denoise] txt_ids shape={txt_ids.shape}")

    if img_cond_seq is not None:
        logger.debug(f"[Denoise:Setup] Generated image token count: {num_img_tokens}")
        logger.debug(f"[Denoise:Setup] Reference token count: {img_cond_seq.shape[1]}")
        logger.debug(f"[Denoise:Setup] Combined sequence length: {num_img_tokens + img_cond_seq.shape[1]}")
        logger.debug(f"[Denoise] Reference tokens: {img_cond_seq.shape[1]} tokens")

    # Log SDPA backend status
    if torch.cuda.is_available():
        try:
            from torch.backends.cuda import (
                flash_sdp_enabled,
                math_sdp_enabled,
                mem_efficient_sdp_enabled,
            )
            logger.debug(f"[Denoise] SDPA backends: flash={flash_sdp_enabled()}, "
                        f"mem_efficient={mem_efficient_sdp_enabled()}, math={math_sdp_enabled()}")
        except ImportError:
            logger.debug("[Denoise] SDPA backend info not available (PyTorch < 2.0)")

    # Track peak memory
    peak_allocated = 0.0
    peak_reserved = 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    initial_alloc, initial_reserved = _log_denoise_memory(0, "init")
    logger.debug("=" * 60)

    # Prepare guidance vector (for non-distilled models)
    guidance_vec = None
    if guidance is not None:
        guidance_vec = torch.full(
            (img.shape[0],), guidance, device=img.device, dtype=img.dtype
        )
        logger.debug(f"[Denoise] Using guidance: {guidance}")
    else:
        logger.debug("[Denoise] No guidance (distilled model)")

    for step_idx, (t_curr, t_prev) in enumerate(
        tqdm(zip(timesteps[:-1], timesteps[1:]), total=num_steps, desc="Denoising")
    ):
        # =====================================================================
        # Step Start
        # =====================================================================
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("-" * 40)
            logger.debug(f"[Denoise:Step {step_idx}] t_curr={t_curr:.4f} → t_prev={t_prev:.4f}")
            _log_denoise_memory(step_idx, "start")

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

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Denoise:Step {step_idx}] model input shape: {img_input.shape}")
            _log_denoise_memory(step_idx, "pre_forward")

        # Detailed logging on first step for diagnosis
        if img_cond_seq is not None and step_idx == 0:
            logger.debug(f"[Denoise:Step {step_idx}] Combined sequence breakdown:")
            logger.debug(f"[Denoise:Step {step_idx}]   Generated tokens: {num_img_tokens}")
            logger.debug(f"[Denoise:Step {step_idx}]   Reference tokens: {img_cond_seq.shape[1]}")
            logger.debug(f"[Denoise:Step {step_idx}]   Total tokens: {img_input.shape[1]}")
            logger.debug(f"[Denoise:Step {step_idx}]   img_input shape: {list(img_input.shape)}")
            logger.debug(f"[Denoise:Step {step_idx}]   img_input_ids shape: {list(img_input_ids.shape)}")
            # Show generated image position ID range
            gen_ids = img_input_ids[0, :num_img_tokens]
            logger.debug(f"[Denoise:Step {step_idx}]   Generated IDs - first: {gen_ids[0].tolist()}, last: {gen_ids[-1].tolist()}")
            # Show reference position ID ranges
            ref_ids_section = img_input_ids[0, num_img_tokens:]
            if ref_ids_section.shape[0] > 0:
                logger.debug(f"[Denoise:Step {step_idx}]   Reference IDs - first: {ref_ids_section[0].tolist()}, last: {ref_ids_section[-1].tolist()}")

        # =====================================================================
        # Model Forward Pass
        # =====================================================================
        # CRITICAL: Use torch.no_grad() to prevent activation accumulation.
        # Without this, PyTorch builds an autograd graph and stores all
        # intermediate activations, causing memory to grow with each block.
        with torch.no_grad():
            pred = model(
                x=img_input,
                x_ids=img_input_ids,
                timesteps=t_vec,
                ctx=txt,
                ctx_ids=txt_ids,
                guidance=guidance_vec,
            )

        if logger.isEnabledFor(logging.DEBUG):
            _log_denoise_memory(step_idx, "post_forward")

        # Only take prediction for noise tokens (not reference tokens)
        if img_cond_seq is not None:
            pred = pred[:, :num_img_tokens]

        # Euler step: x_{t-1} = x_t + (t_prev - t_curr) * v
        img = img + (t_prev - t_curr) * pred

        # =====================================================================
        # Step End - Memory Tracking
        # =====================================================================
        if torch.cuda.is_available():
            step_alloc, step_reserved = _log_denoise_memory(step_idx, "end")
            peak_allocated = max(peak_allocated, step_alloc)
            peak_reserved = max(peak_reserved, step_reserved)

            # Log delta from initial
            delta = step_alloc - initial_alloc
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[Denoise:Step {step_idx}] Memory delta from init: {_format_gb(delta)}")

    # =========================================================================
    # Denoising Complete - Summary
    # =========================================================================
    logger.debug("=" * 60)
    logger.debug("[Denoise] Denoising complete")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        final_alloc = torch.cuda.memory_allocated()
        final_reserved = torch.cuda.memory_reserved()
        pytorch_peak = torch.cuda.max_memory_allocated()

        logger.debug(f"[Denoise] Final: {_format_gb(final_alloc)} allocated, {_format_gb(final_reserved)} reserved")
        logger.debug(f"[Denoise] Peak (tracked): {_format_gb(peak_allocated)} allocated")
        logger.debug(f"[Denoise] Peak (PyTorch): {_format_gb(pytorch_peak)} allocated")
        logger.debug(f"[Denoise] Memory retained: {_format_gb(final_alloc - initial_alloc)}")

        # Log if memory usage seems problematic
        if pytorch_peak > 20e9:  # > 20GB
            logger.warning(f"[Denoise] HIGH PEAK MEMORY: {_format_gb(pytorch_peak)} - may cause OOM")

    logger.debug("=" * 60)

    return img


def latents_to_image(
    latents: torch.Tensor,
    vae,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> Image.Image:
    """
    Decode latents to PIL Image.

    Args:
        latents: Patchified latents [B, seq_len, C] or [B, C, H, W]
        vae: FLUX.2 VAE
        height: Target image height in pixels (needed for non-square images)
        width: Target image width in pixels (needed for non-square images)

    Returns:
        PIL Image
    """
    logger.debug(f"[Decode] Input latents shape: {list(latents.shape)}")
    logger.debug(f"[Decode] Target height={height}, width={width}")

    # Reshape from sequence to spatial if needed
    if latents.ndim == 3:
        # [B, seq_len, C] -> [B, C, H, W]
        b, seq_len, c = latents.shape

        # Calculate patch dimensions from target size or assume square
        if height is not None and width is not None:
            h = height // 16  # FLUX.2 uses 16x16 patches
            w = width // 16
            logger.debug(f"[Decode] Computed h={h}, w={w} (from height={height}, width={width})")
        else:
            # Fallback to square (only works for square outputs)
            h = w = int(math.sqrt(seq_len))
            logger.debug(f"[Decode] Computed square h={h}, w={w} (from seq_len={seq_len})")

        logger.debug(f"[Decode] Reshaping from [B={b}, seq_len={seq_len}, C={c}] to [B={b}, C={c}, H={h}, W={w}]")
        latents = latents.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        logger.debug(f"[Decode] Reshaped tensor dimensions: {list(latents.shape)}")

    # Decode
    with torch.no_grad():
        pixels = vae.decode(latents)

    # Convert to PIL
    # pixels is in [-1, 1], convert to [0, 1]
    pixels = (pixels + 1) / 2
    pixels = pixels.clamp(0, 1)

    # Convert to PIL Image
    pixels = pixels[0]  # Remove batch dimension
    pixels = pixels.float()  # to_pil_image doesn't support bfloat16
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
    log_gpu_memory("before encoder load")
    logger.info("Stage 1: Encoding text...")

    if encoder is None:
        from llm_dit.encoders.qwen3_unified import Qwen3UnifiedEncoder

        # Use custom encoder path if provided, otherwise use model default
        model_info = FLUX2_MODEL_INFO[model_name.lower()]
        text_encoder_spec = encoder_path or model_info["text_encoder"]
        logger.info(f"Loading encoder from: {text_encoder_spec}")

        # Determine preset based on model
        preset = "klein-9b" if "9b" in model_name.lower() or "8b" in model_name.lower() else "klein-4b"
        layers_str = str(config.output_layers) if config.output_layers else "[9, 18, 27]"
        logger.debug(f"[TextEnc] preset={preset}, max_length={config.max_text_length}, pad_to_max={config.pad_to_max}, layers={layers_str}")

        # Use unified encoder with FLUX.2 Klein preset
        encoder = Qwen3UnifiedEncoder.from_preset(
            preset,
            model_path=text_encoder_spec,
            device=config.device,
        )

    # Encode text
    log_gpu_memory("after encoder load")
    txt_embeddings = encoder.encode([config.prompt])  # [1, seq_len, context_dim]
    txt_embeddings = txt_embeddings.to(dtype)
    log_gpu_memory("after text encoding")

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
        log_gpu_memory("after encoder offload + cleanup")

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

        transformer = load_flux2_transformer(
            model_name,
            device=config.device,
            dtype=dtype,
            model_path=model_path,
            block_offload=config.block_offload,
        )
        log_gpu_memory("after transformer load")

    # Load LoRA weights if specified
    if config.loras:
        from llm_dit.utils.lora import load_lora, parse_lora_spec

        total_updated = 0
        for spec in config.loras:
            path, scale = parse_lora_spec(spec)
            logger.info(f"Loading LoRA: {path} (scale={scale})")
            updated = load_lora(transformer, path, scale=scale)
            total_updated += updated
        logger.info(f"LoRA complete: {total_updated} layers updated")
        log_gpu_memory("after LoRA fusion")

    # Move embeddings to device
    txt_embeddings = txt_embeddings.to(device)
    txt_ids = txt_ids.to(device)

    # Move reference tokens to device if present
    if ref_tokens is not None and ref_ids is not None:
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
    logger.debug(f"[GenImage:Setup] Creating position IDs for generated image:")
    logger.debug(f"[GenImage:Setup]   config.height={config.height}, config.width={config.width} (pixels)")
    logger.debug(f"[GenImage:Setup]   config.latent_height={config.latent_height}, config.latent_width={config.latent_width}")
    logger.debug(f"[GenImage:Setup]   config.num_tokens={config.num_tokens}")

    img_ids = create_image_ids(
        batch_size=1,
        height=config.latent_height,
        width=config.latent_width,
        device=device,
        dtype=torch.float32,
    )
    logger.debug(f"[GenImage:Setup] Generated img_ids shape: {list(img_ids.shape)}")
    logger.debug(f"[GenImage:Setup] Generated img_ids first 3: {img_ids[0, :3].tolist()}")
    logger.debug(f"[GenImage:Setup] Generated img_ids last 3: {img_ids[0, -3:].tolist()}")

    # Validate dimensions match
    if img.shape[1] != img_ids.shape[1]:
        logger.error(f"[GenImage:Setup] MISMATCH: img tokens={img.shape[1]}, img_ids tokens={img_ids.shape[1]}")
    else:
        logger.debug(f"[GenImage:Setup] Dimension check OK: {img.shape[1]} tokens")

    # Get timestep schedule
    timesteps = get_schedule(config.num_steps, config.num_tokens)

    # Determine guidance (None for distilled models)
    guidance = None if FLUX2_MODEL_INFO[model_name.lower()]["distilled"] else config.guidance

    log_gpu_memory("before denoising loop")
    log_memory_snapshot("Pre-denoise")

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
    image = latents_to_image(latents, vae, height=config.height, width=config.width)

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
    block_offload: bool = False,
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
        block_offload: Block-by-block offloading (slower but uses less VRAM)

    Returns:
        Generated PIL Image
    """
    config = Flux2GenerationConfig(
        prompt=prompt,
        height=height,
        width=width,
        seed=seed,
        block_offload=block_offload,
    )
    return generate_image(
        config,
        model_name=model_name,
        encoder_path=encoder_path,
        model_path=model_path,
        vae_path=vae_path,
    )
