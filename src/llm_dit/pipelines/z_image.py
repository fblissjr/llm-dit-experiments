"""
Z-Image pipeline for end-to-end text-to-image generation.

This pipeline combines our ZImageTextEncoder with diffusers components:
- Scheduler: FlowMatchEulerDiscreteScheduler (shift=3) or our FlowMatchScheduler
- Transformer: ZImageTransformer2DModel (S3-DiT, 6B params)
- VAE: AutoencoderKL (16-channel, Flux-derived) with optional tiled decode

Example:
    pipe = ZImagePipeline.from_pretrained("/path/to/z-image")
    image = pipe("A cat sleeping in sunlight")

Key differences from diffusers ZImagePipeline:
1. Uses our ZImageTextEncoder with template support
2. Supports template-based prompt customization
3. Exposes thinking block control
4. More explicit control over encoding
5. Optional pure-PyTorch scheduler (use_custom_scheduler=True)
6. Optional tiled VAE decode for large images (tiled_vae=True)
7. Selectable attention backend (flash_attn_2, sdpa, etc.)
8. DyPE (Dynamic Position Extrapolation) for high-resolution generation
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING

import torch
from PIL import Image

from llm_dit.conversation import Conversation
from llm_dit.encoders import ZImageTextEncoder
from llm_dit.utils.long_prompt import compress_embeddings, LongPromptMode

if TYPE_CHECKING:
    from llm_dit.utils.dype import DyPEConfig
    from llm_dit.guidance import SkipLayerGuidance, LayerSkipConfig

logger = logging.getLogger(__name__)

# Maximum text sequence length supported by the DiT transformer.
# The Z-Image DiT uses multi-axis RoPE with config axes_lens=[1536, 512, 512].
# However, the actual working limit is 1504 tokens (47 * 32 = 1504, where 32 is axes_dims[0]).
# This appears to be an off-by-one in RoPE frequency table indexing.
# Exceeding 1504 causes "vectorized_gather_kernel: index out of bounds" errors.
MAX_TEXT_SEQ_LEN = 1504


def setup_attention_backend(backend: Optional[str] = None) -> str:
    """
    Configure attention backend on startup.

    Args:
        backend: Backend name or "auto" for auto-detection.
                Options: flash_attn_3, flash_attn_2, sage, xformers, sdpa, auto

    Returns:
        Name of the selected backend.
    """
    from llm_dit.utils.attention import (
        get_available_backends,
        get_attention_backend,
        set_attention_backend,
        log_attention_info,
    )

    if backend is not None and backend != "auto":
        try:
            set_attention_backend(backend)
        except ValueError as e:
            logger.warning(f"Failed to set attention backend: {e}")
            logger.info("Falling back to auto-detect")

    log_attention_info()
    return get_attention_backend()


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Calculate shift for flow matching scheduler."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class ZImagePipeline:
    """
    End-to-end pipeline for Z-Image text-to-image generation.

    This pipeline wraps diffusers components with our custom encoder,
    providing template support and more control over the encoding process.

    Attributes:
        encoder: ZImageTextEncoder for prompt encoding
        transformer: diffusers ZImageTransformer2DModel
        vae: diffusers AutoencoderKL
        scheduler: diffusers FlowMatchEulerDiscreteScheduler

    Note:
        This pipeline requires diffusers>=0.30.0 with Z-Image support.
        Install via: pip install diffusers[torch]
    """

    def __init__(
        self,
        encoder: ZImageTextEncoder,
        transformer: Any,  # ZImageTransformer2DModel
        vae: Any,  # AutoencoderKL
        scheduler: Any,  # FlowMatchEulerDiscreteScheduler or FlowMatchScheduler
        tiled_vae: bool = False,
        tile_size: int = 512,
        tile_overlap: int = 64,
        dype_config: Optional["DyPEConfig"] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            encoder: ZImageTextEncoder instance
            transformer: diffusers ZImageTransformer2DModel
            vae: diffusers AutoencoderKL
            scheduler: diffusers FlowMatchEulerDiscreteScheduler or our FlowMatchScheduler
            tiled_vae: Enable tiled VAE decode for large images
            tile_size: Tile size in pixels (default: 512)
            tile_overlap: Overlap between tiles in pixels (default: 64)
            dype_config: Optional DyPE configuration for high-resolution generation
        """
        self.encoder = encoder
        self.transformer = transformer
        self.scheduler = scheduler
        self.dype_config = dype_config

        # VAE scale factor (8 for Z-Image)
        self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

        # Wrap VAE with tiled decoder if requested
        self._tiled_vae_enabled = tiled_vae
        if tiled_vae:
            from llm_dit.utils.tiled_vae import TiledVAEDecoder
            self.vae = TiledVAEDecoder(vae, tile_size=tile_size, tile_overlap=tile_overlap)
            logger.info(f"Tiled VAE enabled: tile_size={tile_size}, overlap={tile_overlap}")
        else:
            self.vae = vae

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        text_encoder_path: str | None = None,
        templates_dir: str | Path | None = None,
        default_template: str | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str | None = None,
        encoder_device: str = "auto",
        dit_device: str = "auto",
        vae_device: str = "auto",
        quantization: str = "none",
        # PyTorch-native component options
        use_custom_scheduler: bool = False,
        tiled_vae: bool = False,
        tile_size: int = 512,
        tile_overlap: int = 64,
        attention_backend: str | None = None,
        **kwargs,
    ) -> "ZImagePipeline":
        """
        Load pipeline from pretrained model.

        Args:
            model_path: Path to Z-Image model (DiT + VAE) or HuggingFace ID
            text_encoder_path: Path to text encoder (Qwen3-4B). If None, uses model_path/text_encoder/
            templates_dir: Optional path to templates directory
            default_template: Optional default template name
            torch_dtype: Model dtype (default: bfloat16)
            device_map: Device mapping (default: "auto", legacy parameter)
            encoder_device: Device for text encoder (cpu, cuda, mps, auto)
            dit_device: Device for DiT/transformer (cpu, cuda, mps, auto)
            vae_device: Device for VAE (cpu, cuda, mps, auto)
            quantization: Text encoder quantization mode (none, 4bit, 8bit, int8_dynamic)
            use_custom_scheduler: Use our pure-PyTorch FlowMatchScheduler
            tiled_vae: Enable tiled VAE decode for large images (2K+)
            tile_size: Tile size in pixels for VAE decode (default: 512)
            tile_overlap: Overlap between tiles in pixels (default: 64)
            attention_backend: Attention backend (auto, flash_attn_2, sdpa, etc.)
            **kwargs: Additional arguments

        Returns:
            Initialized ZImagePipeline

        Example:
            pipe = ZImagePipeline.from_pretrained(
                "Tongyi-MAI/Z-Image-Turbo",
                templates_dir="templates/z_image",
                torch_dtype=torch.bfloat16,
                encoder_device="cpu",
                dit_device="cuda",
                vae_device="cuda",
            )

            # With PyTorch-native components:
            pipe = ZImagePipeline.from_pretrained(
                "/path/to/z-image",
                use_custom_scheduler=True,
                tiled_vae=True,
                attention_backend="flash_attn_2",
            )

        Note:
            Z-Image-Turbo requires diffusers with Z-Image support.
            As of diffusers 0.35.x, the model architecture may not be
            fully supported. Check diffusers releases for Z-Image support.
        """
        # Import diffusers components
        try:
            from diffusers import DiffusionPipeline
        except ImportError as e:
            raise ImportError(
                "diffusers is required for ZImagePipeline. "
                "Install with: pip install diffusers[torch]"
            ) from e

        # Set up attention backend
        if attention_backend:
            setup_attention_backend(attention_backend)

        # Resolve device strings
        def resolve_device(device_str: str) -> str:
            if device_str == "auto":
                if torch.cuda.is_available():
                    return "cuda"
                elif torch.backends.mps.is_available():
                    return "mps"
                else:
                    return "cpu"
            return device_str

        encoder_device_resolved = resolve_device(encoder_device)
        dit_device_resolved = resolve_device(dit_device)
        vae_device_resolved = resolve_device(vae_device)

        # Use separate text encoder path if provided, otherwise use model_path
        encoder_path = text_encoder_path if text_encoder_path else model_path

        logger.info("Loading pipeline components with device placement:")
        logger.info(f"  Encoder: {encoder_device_resolved} (from {encoder_path})")
        logger.info(f"  DiT: {dit_device_resolved}")
        logger.info(f"  VAE: {vae_device_resolved}")

        # Load encoder (our custom encoder with template support)
        encoder = ZImageTextEncoder.from_pretrained(
            encoder_path,
            templates_dir=templates_dir,
            default_template=default_template,
            torch_dtype=torch_dtype,
            device_map=encoder_device_resolved,
            quantization=quantization,
        )

        # Load the diffusers pipeline (auto-detect pipeline class)
        logger.info("Loading diffusers pipeline...")
        load_kwargs = {
            "torch_dtype": torch_dtype,  # diffusers uses torch_dtype, not dtype
            **kwargs,
        }
        # Use device_map for initial loading if provided (legacy)
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        diffusers_pipe = DiffusionPipeline.from_pretrained(model_path, **load_kwargs)

        # Extract components from diffusers pipeline
        transformer = diffusers_pipe.transformer
        vae = diffusers_pipe.vae

        # Use our scheduler or diffusers scheduler
        if use_custom_scheduler:
            from llm_dit.schedulers import FlowMatchScheduler
            scheduler = FlowMatchScheduler(shift=3.0)
            logger.info("Using custom FlowMatchScheduler (pure PyTorch)")
        else:
            scheduler = diffusers_pipe.scheduler

        # Move components to their designated devices
        logger.info(f"Moving transformer to {dit_device_resolved}...")
        transformer = transformer.to(dit_device_resolved)
        logger.info(f"Moving VAE to {vae_device_resolved}...")
        vae = vae.to(vae_device_resolved)

        logger.info("Pipeline loaded successfully")
        return cls(
            encoder,
            transformer,
            vae,
            scheduler,
            tiled_vae=tiled_vae,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            dype_config=kwargs.get("dype_config"),
        )

    @classmethod
    def from_diffusers_pipeline(
        cls,
        diffusers_pipe,
        model_path: str,
        templates_dir: str | Path | None = None,
        default_template: str | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ZImagePipeline":
        """
        Create ZImagePipeline from an existing diffusers pipeline.

        This is useful when you've already loaded a diffusers pipeline
        and want to use our custom encoder with template support.

        Args:
            diffusers_pipe: An existing diffusers pipeline with transformer, vae, scheduler
            model_path: Path to model (for loading our encoder)
            templates_dir: Optional path to templates directory
            default_template: Optional default template name
            torch_dtype: Model dtype

        Returns:
            ZImagePipeline with custom encoder

        Example:
            from diffusers import ZImagePipeline as DiffusersPipe
            diffusers_pipe = DiffusersPipe.from_pretrained("model_path")

            # Wrap with our encoder
            pipe = ZImagePipeline.from_diffusers_pipeline(
                diffusers_pipe,
                "model_path",
                templates_dir="templates/z_image",
            )
        """
        # Load our encoder
        encoder = ZImageTextEncoder.from_pretrained(
            model_path,
            templates_dir=templates_dir,
            default_template=default_template,
            torch_dtype=torch_dtype,
        )

        return cls(
            encoder=encoder,
            transformer=diffusers_pipe.transformer,
            vae=diffusers_pipe.vae,
            scheduler=diffusers_pipe.scheduler,
        )

    @property
    def device(self) -> torch.device:
        """Return pipeline device (based on transformer, not encoder)."""
        # Use transformer device since encoder might be API-backed (returns CPU)
        if self.transformer is not None:
            return next(self.transformer.parameters()).device
        if self.encoder is not None:
            return self.encoder.device
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        """Return pipeline dtype."""
        if self.transformer is not None:
            return next(self.transformer.parameters()).dtype
        if self.encoder is not None:
            return self.encoder.dtype
        return torch.float32

    def to(self, device: torch.device) -> "ZImagePipeline":
        """Move pipeline to device."""
        self.encoder.to(device)
        self.transformer.to(device)
        self.vae.to(device)
        return self

    def load_lora(
        self,
        lora_path: Union[str, Path, List[str], List[Path]],
        scale: Union[float, List[float]] = 1.0,
    ) -> int:
        """
        Load and fuse LoRA weights into the transformer.

        Args:
            lora_path: Path to LoRA file(s). Can be a single path or list of paths.
            scale: LoRA scale factor(s). Can be a single value or list matching lora_path.

        Returns:
            Total number of layers updated

        Example:
            # Single LoRA
            pipe.load_lora("anime_style.safetensors", scale=0.7)

            # Multiple LoRAs
            pipe.load_lora(
                ["style1.safetensors", "style2.safetensors"],
                scale=[0.5, 0.3]
            )

        Note:
            LoRAs are fused (permanently merged) into the transformer weights.
            To remove a LoRA, you must reload the pipeline.
        """
        from llm_dit.utils.lora import load_lora as _load_lora

        # Normalize to lists
        if isinstance(lora_path, (str, Path)):
            lora_paths = [lora_path]
        else:
            lora_paths = list(lora_path)

        if isinstance(scale, (int, float)):
            scales = [scale] * len(lora_paths)
        else:
            scales = list(scale)

        if len(lora_paths) != len(scales):
            raise ValueError(
                f"Number of LoRA paths ({len(lora_paths)}) must match "
                f"number of scales ({len(scales)})"
            )

        total_updated = 0
        for path, s in zip(lora_paths, scales):
            updated = _load_lora(
                self.transformer,
                path,
                scale=s,
                device=next(self.transformer.parameters()).device,
                torch_dtype=next(self.transformer.parameters()).dtype,
            )
            total_updated += updated

        return total_updated

    def enable_gradient_checkpointing(self, enable: bool = True) -> None:
        """
        Enable or disable gradient checkpointing on the transformer.

        Gradient checkpointing trades compute for memory by not storing
        intermediate activations during the forward pass, recomputing them
        during the backward pass instead.

        Args:
            enable: Whether to enable (True) or disable (False) checkpointing.

        Use cases:
            - Training/fine-tuning with limited VRAM
            - LoRA training on consumer GPUs
            - Large batch sizes

        Example:
            pipe.enable_gradient_checkpointing(True)
            # Now training uses less VRAM but is slower

        Note:
            This only affects training. Inference is unaffected.
        """
        if self.transformer is None:
            logger.warning("No transformer loaded, cannot set gradient checkpointing")
            return

        if hasattr(self.transformer, "enable_gradient_checkpointing"):
            if enable:
                self.transformer.enable_gradient_checkpointing()
                logger.info("Gradient checkpointing enabled on transformer")
            else:
                self.transformer.disable_gradient_checkpointing()
                logger.info("Gradient checkpointing disabled on transformer")
        elif hasattr(self.transformer, "gradient_checkpointing_enable"):
            # diffusers style
            if enable:
                self.transformer.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled on transformer")
            else:
                self.transformer.gradient_checkpointing_disable()
                logger.info("Gradient checkpointing disabled on transformer")
        else:
            logger.warning(
                "Transformer does not support gradient checkpointing. "
                "This may be a model architecture limitation."
            )

    def unload_fmtt(self) -> bool:
        """
        Unload cached FMTT reward function (SigLIP) to free GPU memory.

        The FMTT reward function is cached on first use to avoid reloading
        on subsequent generations. This method releases that memory (~4-6GB)
        when FMTT is no longer needed.

        Returns:
            True if a cached reward function was unloaded, False if none was cached.

        Example:
            pipe("A cat", fmtt_guidance_scale=1.0)  # Loads SigLIP
            pipe("A dog", fmtt_guidance_scale=0.0)  # SigLIP still in memory!
            pipe.unload_fmtt()                       # Now SigLIP is freed
            pipe("A bird")                           # Full VRAM available
        """
        if hasattr(self, '_fmtt_reward_fn') and self._fmtt_reward_fn is not None:
            # Clear text cache if it exists
            if hasattr(self._fmtt_reward_fn, 'clear_cache'):
                self._fmtt_reward_fn.clear_cache()

            # Delete the model
            del self._fmtt_reward_fn
            self._fmtt_reward_fn = None

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("[Pipeline] FMTT reward function unloaded, VRAM freed")
            return True

        return False

    def encode_image(
        self,
        image: Union[Image.Image, torch.Tensor],
    ) -> torch.Tensor:
        """
        Encode an image to latent space using the VAE.

        Args:
            image: PIL Image or tensor in [0, 1] range with shape (B, C, H, W)

        Returns:
            Latent tensor ready for denoising
        """
        import numpy as np

        # Get the underlying VAE (unwrap TiledVAEDecoder if needed)
        vae = self.vae.vae if hasattr(self.vae, 'vae') else self.vae

        # Convert PIL to tensor
        if isinstance(image, Image.Image):
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

        # Ensure correct shape and range
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Move to VAE device and dtype
        vae_device = next(vae.parameters()).device
        vae_dtype = next(vae.parameters()).dtype
        image = image.to(device=vae_device, dtype=vae_dtype)

        # Normalize from [0, 1] to [-1, 1]
        image = 2.0 * image - 1.0

        # Encode
        with torch.no_grad():
            latent_dist = vae.encode(image)
            if hasattr(latent_dist, 'latent_dist'):
                latents = latent_dist.latent_dist.sample()
            else:
                latents = latent_dist.sample() if hasattr(latent_dist, 'sample') else latent_dist

        # Apply VAE scaling
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

        return latents

    def img2img(
        self,
        prompt: Union[str, "Conversation", None] = None,
        image: Union[Image.Image, torch.Tensor] = None,
        strength: float = 0.75,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 9,
        guidance_scale: float = 0.0,
        negative_prompt: Optional[str] = None,
        generator: Optional[torch.Generator] = None,
        template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        thinking_content: Optional[str] = None,
        assistant_content: Optional[str] = None,
        force_think_block: bool = False,
        remove_quotes: bool = False,
        output_type: str = "pil",
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        shift: Optional[float] = None,
        long_prompt_mode: str = "truncate",
        hidden_layer: int = -2,
        prompt_embeds: Optional[torch.Tensor] = None,
        cfg_normalization: float = 0.0,
        cfg_truncation: float = 1.0,
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor]:
        """
        Generate an image from a text prompt and an input image.

        The strength parameter controls how much the output differs from the input:
        - strength=0.0: Output is identical to input (no denoising)
        - strength=1.0: Output ignores input completely (like txt2img)
        - strength=0.5-0.8: Good range for style transfer / modifications

        Args:
            prompt: Text prompt describing the desired output
            image: Input image (PIL Image or tensor)
            strength: How much to transform the input (0.0-1.0)
            height: Output height (default: input image height)
            width: Output width (default: input image width)
            num_inference_steps: Total denoising steps (actual steps = steps * strength)
            guidance_scale: CFG scale (default: 0.0 for Z-Image-Turbo)
            negative_prompt: Negative prompt for CFG
            generator: Random generator for reproducibility
            template: Template name for encoding
            system_prompt: System prompt (optional)
            thinking_content: Content inside <think>...</think>
            assistant_content: Content after </think>
            force_think_block: Add empty think block
            remove_quotes: Strip " characters
            output_type: Output format ("pil", "latent", or "pt")
            callback: Progress callback
            shift: Override scheduler shift/mu
            long_prompt_mode: How to handle prompts > 1504 tokens (truncate/interpolate/pool/attention_pool)
            hidden_layer: Which LLM hidden layer to extract embeddings from (default: -2)

        Returns:
            Generated image in specified format

        Example:
            # Style transfer
            input_img = Image.open("photo.jpg")
            output = pipe.img2img(
                "A watercolor painting",
                image=input_img,
                strength=0.6,
            )

            # Light modifications
            output = pipe.img2img(
                "Add a sunset sky",
                image=input_img,
                strength=0.3,
            )
        """
        import numpy as np

        # Get image dimensions
        if isinstance(image, Image.Image):
            img_width, img_height = image.size
        else:
            img_height, img_width = image.shape[-2:]

        # Use image dimensions if not specified
        if height is None:
            height = img_height
        if width is None:
            width = img_width

        # Validate dimensions
        vae_scale = self.vae_scale_factor * 2  # 16 for Z-Image
        if height % vae_scale != 0:
            # Round to nearest valid size
            height = (height // vae_scale) * vae_scale
            logger.info(f"Adjusted height to {height} (must be divisible by {vae_scale})")
        if width % vae_scale != 0:
            width = (width // vae_scale) * vae_scale
            logger.info(f"Adjusted width to {width} (must be divisible by {vae_scale})")

        # Resize image if dimensions changed
        if isinstance(image, Image.Image):
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.LANCZOS)
        else:
            if image.shape[-2:] != (height, width):
                image = torch.nn.functional.interpolate(
                    image, size=(height, width), mode="bilinear", align_corners=False
                )

        device = self.device
        dtype = self.dtype

        # Calculate actual number of steps based on strength
        # strength=1.0 means all steps, strength=0.5 means half the steps
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)

        logger.info(f"[img2img] strength={strength}, steps={num_inference_steps}, actual_steps={init_timestep}")

        # 1. Encode prompt (or use provided embeddings)
        if prompt_embeds is not None:
            logger.info(f"[img2img] Using provided prompt_embeds: shape={prompt_embeds.shape}")
            raw_embeds = prompt_embeds
            if raw_embeds.shape[0] > MAX_TEXT_SEQ_LEN:
                raw_embeds = compress_embeddings(raw_embeds, MAX_TEXT_SEQ_LEN, mode=long_prompt_mode)
            prompt_embeds_list = [raw_embeds.to(device=device, dtype=dtype)]
        else:
            logger.info(f"[img2img] Encoding prompt...")
            prompt_output = self.encoder.encode(
                prompt,
                template=template,
                system_prompt=system_prompt,
                thinking_content=thinking_content,
                assistant_content=assistant_content,
                force_think_block=force_think_block,
                remove_quotes=remove_quotes,
                layer_index=hidden_layer,
            )
            raw_embeds = prompt_output.embeddings[0]
            # Compress if needed
            if raw_embeds.shape[0] > MAX_TEXT_SEQ_LEN:
                raw_embeds = compress_embeddings(raw_embeds, MAX_TEXT_SEQ_LEN, mode=long_prompt_mode)
            prompt_embeds_list = [raw_embeds.to(device=device, dtype=dtype)]

        # 2. Encode input image
        logger.info(f"[img2img] Encoding input image...")
        init_latents = self.encode_image(image)
        init_latents = init_latents.to(device=device, dtype=dtype)
        logger.info(f"[img2img] Init latents shape: {init_latents.shape}")

        # 3. Prepare timesteps
        latent_height = 2 * (height // vae_scale)
        latent_width = 2 * (width // vae_scale)
        image_seq_len = (latent_height // 2) * (latent_width // 2)

        if shift is not None:
            mu = shift
        else:
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )

        self.scheduler.sigma_min = 0.0
        # IMPORTANT: For diffusers FlowMatchEulerDiscreteScheduler, when use_dynamic_shifting=False,
        # set_timesteps() ignores the mu parameter and uses self.shift instead.
        # Diffusers uses set_shift() method (shift property is read-only).
        # Our FlowMatchScheduler uses direct attribute assignment.
        if hasattr(self.scheduler, 'set_shift'):
            self.scheduler.set_shift(mu)  # diffusers scheduler
        elif hasattr(self.scheduler, 'shift'):
            self.scheduler.shift = mu  # our FlowMatchScheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
        timesteps = self.scheduler.timesteps[t_start:]

        # 4. Add noise to init latents
        noise = torch.randn(
            init_latents.shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )

        # Get the starting sigma (noise level)
        if t_start < len(self.scheduler.sigmas):
            sigma_start = self.scheduler.sigmas[t_start]
        else:
            sigma_start = self.scheduler.sigmas[-1]

        # Add noise for flow matching: latents = (1 - sigma) * init + sigma * noise
        # FlowMatchEulerDiscreteScheduler doesn't have add_noise, so we do it manually
        sigma = sigma_start.to(device=device, dtype=dtype)
        latents = (1 - sigma) * init_latents + sigma * noise
        latents = latents.to(dtype=torch.float32)

        logger.info(f"[img2img] Starting denoising from step {t_start}, {len(timesteps)} steps remaining")

        # 5. Encode negative prompt if using CFG
        negative_prompt_embeds = []
        if guidance_scale > 0:
            if negative_prompt is not None:
                neg_output = self.encoder.encode(negative_prompt, force_think_block=force_think_block, layer_index=hidden_layer)
                neg_embeds = neg_output.embeddings[0]
                # Compress if needed
                if neg_embeds.shape[0] > MAX_TEXT_SEQ_LEN:
                    neg_embeds = compress_embeddings(neg_embeds, MAX_TEXT_SEQ_LEN, mode=long_prompt_mode)
                negative_prompt_embeds = [neg_embeds.to(device=device, dtype=dtype)]
            else:
                neg_output = self.encoder.encode("", force_think_block=force_think_block, layer_index=hidden_layer)
                negative_prompt_embeds = [neg_output.embeddings[0].to(device=device, dtype=dtype)]

        # 6. Denoising loop (same as txt2img but starting from noised latents)
        cpu_offload = getattr(self, '_enable_cpu_offload', False)

        with torch.no_grad():
            for i, t in enumerate(timesteps):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if cpu_offload:
                    self.transformer.to(device)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                timestep = t.expand(latents.shape[0])
                timestep = (1000 - timestep) / 1000

                # Calculate denoising progress (0 to 1) for CFG truncation
                progress = timestep[0].item()

                # Handle CFG with optional truncation
                current_cfg_scale = guidance_scale
                if cfg_truncation < 1.0 and progress > cfg_truncation:
                    current_cfg_scale = 0.0

                apply_cfg = current_cfg_scale > 0

                if apply_cfg:
                    latent_input = latents.to(dtype).repeat(2, 1, 1, 1)
                    embeds_input = prompt_embeds_list + negative_prompt_embeds
                    timestep_input = timestep.repeat(2)
                else:
                    latent_input = latents.to(dtype)
                    embeds_input = prompt_embeds_list
                    timestep_input = timestep

                latent_input = latent_input.unsqueeze(2)
                latent_list = list(latent_input.unbind(dim=0))

                model_output = self.transformer(
                    latent_list,
                    timestep_input,
                    embeds_input,
                )[0]

                if cpu_offload:
                    self.transformer.to("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if apply_cfg:
                    pos_out = model_output[:1]
                    neg_out = model_output[1:]
                    noise_pred = []
                    for pos, neg in zip(pos_out, neg_out):
                        pos_f = pos.float()
                        neg_f = neg.float()
                        pred = pos_f + current_cfg_scale * (pos_f - neg_f)

                        # CFG normalization
                        if cfg_normalization > 0:
                            pos_norm = torch.linalg.vector_norm(pos_f)
                            pred_norm = torch.linalg.vector_norm(pred)
                            max_allowed_norm = pos_norm * cfg_normalization
                            pred_norm = torch.where(pred_norm < 1e-6, torch.ones_like(pred_norm), pred_norm)
                            scale_factor = torch.clamp(max_allowed_norm / pred_norm, max=1.0)
                            pred = pred * scale_factor

                        noise_pred.append(pred)
                    noise_pred = torch.stack(noise_pred, dim=0)
                else:
                    noise_pred = torch.stack([o.float() for o in model_output], dim=0)

                noise_pred = noise_pred.squeeze(2)
                noise_pred = -noise_pred

                latents = self.scheduler.step(
                    noise_pred.to(torch.float32),
                    t,
                    latents,
                    return_dict=False,
                )[0]

                if callback is not None:
                    callback(i, len(timesteps), latents)

        # 7. Decode latents
        if output_type == "latent":
            return latents

        logger.info("[img2img] Decoding latents...")
        if cpu_offload:
            # Get underlying VAE
            vae = self.vae.vae if hasattr(self.vae, 'vae') else self.vae
            vae.to(device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Get underlying VAE for config access
        vae = self.vae.vae if hasattr(self.vae, 'vae') else self.vae
        latents = latents.to(vae.dtype)
        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor

        with torch.no_grad():
            image_out = self.vae.decode(latents, return_dict=False)
            if isinstance(image_out, tuple):
                image_out = image_out[0]

        if cpu_offload:
            vae = self.vae.vae if hasattr(self.vae, 'vae') else self.vae
            vae.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if output_type == "pt":
            return image_out

        # Convert to PIL
        image_out = (image_out / 2 + 0.5).clamp(0, 1)
        image_out = image_out.cpu().permute(0, 2, 3, 1).float().numpy()
        image_out = (image_out * 255).round().astype("uint8")

        if image_out.shape[0] == 1:
            return Image.fromarray(image_out[0])
        return [Image.fromarray(img) for img in image_out]

    def __call__(
        self,
        prompt: Union[str, Conversation, None] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 9,
        guidance_scale: float = 0.0,
        negative_prompt: Optional[str] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        thinking_content: Optional[str] = None,
        assistant_content: Optional[str] = None,
        force_think_block: bool = False,
        remove_quotes: bool = False,
        output_type: str = "pil",
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        shift: Optional[float] = None,
        long_prompt_mode: str = "truncate",
        hidden_layer: int = -2,
        layer_weights: dict[int, float] | None = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        skip_layer_guidance_scale: float = 0.0,
        skip_layer_indices: Optional[List[int]] = None,
        skip_layer_start: float = 0.01,
        skip_layer_stop: float = 0.2,
        # Flow Map Trajectory Tilting (FMTT) for test-time reward optimization
        fmtt_guidance_scale: float = 0.0,
        fmtt_guidance_start: float = 0.0,
        fmtt_guidance_stop: float = 0.5,
        fmtt_normalize_mode: str = "unit",
        fmtt_decode_scale: float = 0.5,
        fmtt_siglip_model: str = "google/siglip2-giant-opt-patch16-384",
        fmtt_siglip_device: Optional[str] = None,  # None = use pipeline device
        fmtt_reward_fn: Optional[Any] = None,
        # DyPE (Dynamic Position Extrapolation) for high-resolution generation
        dype_config: Optional["DyPEConfig"] = None,  # Per-request DyPE config (overrides self.dype_config)
        # CFG enhancements (useful for non-distilled models like Qwen-Image-Layered)
        cfg_normalization: float = 0.0,  # 0.0 = disabled, >0 = clamp CFG norm relative to positive pred
        cfg_truncation: float = 1.0,  # 1.0 = never truncate, <1.0 = stop CFG at that progress fraction
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor]:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text prompt (user message) or Conversation object
            height: Output image height (default: 1024, must be divisible by 16)
            width: Output image width (default: 1024, must be divisible by 16)
            num_inference_steps: Number of denoising steps (default: 9 for turbo)
            guidance_scale: CFG scale (default: 0.0, CFG is baked into Z-Image)
            negative_prompt: Negative prompt for CFG (only used if guidance_scale > 0)
            generator: Random generator for reproducibility
            latents: Pre-generated latents (optional)
            template: Template name for encoding
            system_prompt: System prompt (optional, e.g., "You are a painter.")
            thinking_content: Content inside <think>...</think> (triggers think block)
            assistant_content: Content after </think> (optional)
            force_think_block: If True, add empty think block even without content
            remove_quotes: If True, strip " characters (for JSON-type prompts)
            output_type: Output format ("pil", "latent", or "pt")
            callback: Optional callback for progress updates
            shift: Override scheduler shift/mu (default: calculated based on resolution).
                   Higher values generally produce more detailed but potentially less stable results.
                   Typical range: 0.5-10.0. Default is dynamically calculated (~3.0 for 1024x1024).
            long_prompt_mode: How to handle prompts exceeding 1504 tokens:
                   "truncate" (default) - cut off at 1504 tokens
                   "interpolate" - resample embeddings using linear interpolation
                   "pool" - use adaptive average pooling
                   "attention_pool" - use importance-weighted pooling
            hidden_layer: Which LLM hidden layer to extract embeddings from (default: -2).
                   -1 = last layer, -2 = penultimate (default), -3 to -6 for deeper layers.
                   Useful for experimenting with embedding quality.
            layer_weights: Optional dict mapping layer indices to blend weights, e.g.:
                   {-2: 0.7, -5: 0.3} blends 70% penultimate + 30% layer -5.
                   If provided, overrides hidden_layer. Weights are normalized to sum to 1.0.
            prompt_embeds: Pre-computed embeddings tensor of shape (seq_len, hidden_dim).
                   If provided, skips text encoding entirely. Useful for:
                   - Vision-conditioned generation with Qwen3-VL embeddings
                   - Distributed inference with pre-computed embeddings
                   - Caching embeddings across multiple generations
            skip_layer_guidance_scale: Scale for Skip Layer Guidance (default: 0.0 = disabled).
                   Typical values: 2.0-3.0. Higher values improve structure/anatomy but may
                   cause artifacts. Only applied when skip_layer_indices is provided.
            skip_layer_indices: List of transformer layer indices to skip for SLG.
                   For Z-Image (30 layers), recommended: [7, 8, 9, 10, 11, 12] (middle layers).
                   If None, SLG is disabled.
            skip_layer_start: Start SLG at this fraction of total steps (default: 0.05).
            skip_layer_stop: Stop SLG at this fraction of total steps (default: 0.5).
                   Wider range needed for turbo-distilled models (8-9 steps).
            fmtt_guidance_scale: Scale for FMTT reward guidance (default: 0.0 = disabled).
                   Typical values: 0.5-2.0. Uses SigLIP2 to guide generation toward text-aligned images.
                   Note: Loads SigLIP (~4GB VRAM). Consider using encoder_device="cpu" if VRAM is limited.
            fmtt_guidance_start: Start FMTT at this fraction of total steps (default: 0.0).
            fmtt_guidance_stop: Stop FMTT at this fraction of total steps (default: 0.5).
            fmtt_normalize_mode: Gradient normalization mode for FMTT (default: "unit").
                   Options: "unit" (normalize to unit norm), "clip", "none".
            fmtt_decode_scale: Scale for intermediate VAE decode during FMTT (default: 0.5).
                   Lower values save VRAM but reduce gradient precision.
            fmtt_reward_fn: Pre-loaded reward function (optional). If None, loads SigLIP on first use.
                   Pass a DifferentiableSigLIP instance to avoid reloading between generations.
            cfg_normalization: CFG norm clamping factor (default: 0.0 = disabled).
                   When >0, clamps the combined prediction norm to cfg_normalization times the
                   positive prediction norm. Prevents CFG from over-amplifying, reducing artifacts
                   at high CFG values. Typical values: 1.0-2.0. Useful for non-distilled models.
            cfg_truncation: CFG truncation threshold (default: 1.0 = never truncate).
                   When <1.0, stops applying CFG after this fraction of denoising progress.
                   E.g., 0.7 means CFG is disabled for the final 30% of steps.
                   Reduces late-stage artifacts. Typical values: 0.5-0.8.

        Returns:
            Generated image(s) in specified format

        Example:
            # Basic generation
            image = pipe("A cat sleeping")

            # With system prompt
            image = pipe("Paint a cat", system_prompt="You are a painter.")

            # With thinking content (automatically adds think block)
            image = pipe(
                "A sunset",
                thinking_content="Warm orange and pink hues.",
            )

            # Force empty think block
            image = pipe("A sunset", force_think_block=True)

            # With seed
            image = pipe(
                "A cat sleeping",
                generator=torch.Generator().manual_seed(42),
            )

            # With pre-computed embeddings (e.g., from Qwen3-VL vision encoder)
            image = pipe(
                prompt_embeds=vision_embeddings,  # shape: (seq_len, 2560)
                generator=torch.Generator().manual_seed(42),
            )
        """
        # Validate dimensions (must be divisible by 16 for Z-Image)
        vae_scale = self.vae_scale_factor * 2  # 16 for Z-Image
        if height % vae_scale != 0:
            raise ValueError(
                f"Height must be divisible by {vae_scale} (got {height})"
            )
        if width % vae_scale != 0:
            raise ValueError(
                f"Width must be divisible by {vae_scale} (got {width})"
            )

        device = self.device
        dtype = self.dtype

        # 1. Encode prompt OR use pre-computed embeddings
        if prompt_embeds is not None:
            # Use pre-computed embeddings (e.g., from Qwen3-VL vision encoder)
            logger.info(f"[Pipeline] Using pre-computed embeddings: shape={prompt_embeds.shape}")
            raw_embeds = prompt_embeds
        else:
            # Standard text encoding path
            if prompt is None:
                raise ValueError("Either 'prompt' or 'prompt_embeds' must be provided")

            logger.info(f"[Pipeline] Encoding prompt on device={device}, dtype={dtype}")
            logger.info(f"[Pipeline] Encoder type: {type(self.encoder).__name__}")
            backend = getattr(self.encoder, 'backend', None)
            logger.info(f"[Pipeline] Encoder backend: {type(backend).__name__ if backend else 'local'}")

            # Use blended encoding if layer_weights provided, otherwise standard encoding
            if layer_weights is not None:
                logger.info(f"[Pipeline] Using blended encoding with layer_weights={layer_weights}")
                prompt_output = self.encoder.encode_blended(
                    prompt,
                    layer_weights=layer_weights,
                    template=template,
                    system_prompt=system_prompt,
                    thinking_content=thinking_content,
                    assistant_content=assistant_content,
                    force_think_block=force_think_block,
                    remove_quotes=remove_quotes,
                )
            else:
                prompt_output = self.encoder.encode(
                    prompt,
                    template=template,
                    system_prompt=system_prompt,
                    thinking_content=thinking_content,
                    assistant_content=assistant_content,
                    force_think_block=force_think_block,
                    remove_quotes=remove_quotes,
                    layer_index=hidden_layer,
                )

            # Log formatted prompt for debugging
            if prompt_output.formatted_prompts:
                formatted = prompt_output.formatted_prompts[0]
                logger.info(f"[Pipeline] Formatted prompt ({len(formatted)} chars):")
                # Show the full prompt with special tokens visible
                logger.info(f"[Pipeline] ---BEGIN FORMATTED PROMPT---")
                logger.info(formatted)
                logger.info(f"[Pipeline] ---END FORMATTED PROMPT---")

            raw_embeds = prompt_output.embeddings[0]
        logger.info(f"[Pipeline] Raw embeddings: shape={raw_embeds.shape}, device={raw_embeds.device}, dtype={raw_embeds.dtype}")
        logger.info(f"[Pipeline] Embedding stats: min={raw_embeds.min().item():.4f}, max={raw_embeds.max().item():.4f}, mean={raw_embeds.mean().item():.4f}, std={raw_embeds.std().item():.4f}")

        # Handle embeddings exceeding DiT's max text sequence length
        if raw_embeds.shape[0] > MAX_TEXT_SEQ_LEN:
            raw_embeds = compress_embeddings(raw_embeds, MAX_TEXT_SEQ_LEN, mode=long_prompt_mode)
            logger.info(f"[Pipeline] Compressed embeddings shape: {raw_embeds.shape}")

        # Move embeddings to device (API backend returns CPU tensors)
        prompt_embeds = [raw_embeds.to(device=device, dtype=dtype)]
        logger.info(f"[Pipeline] Moved embeddings to: device={prompt_embeds[0].device}, dtype={prompt_embeds[0].dtype}")

        # Encode negative prompt if using CFG
        negative_prompt_embeds = []
        if guidance_scale > 0 and negative_prompt is not None:
            neg_output = self.encoder.encode(
                negative_prompt,
                force_think_block=force_think_block,
                layer_index=hidden_layer,
            )
            neg_embeds = neg_output.embeddings[0]
            # Compress if needed
            if neg_embeds.shape[0] > MAX_TEXT_SEQ_LEN:
                neg_embeds = compress_embeddings(neg_embeds, MAX_TEXT_SEQ_LEN, mode=long_prompt_mode)
            negative_prompt_embeds = [neg_embeds.to(device=device, dtype=dtype)]
        elif guidance_scale > 0:
            # Empty negative prompt
            neg_output = self.encoder.encode(
                "",
                force_think_block=force_think_block,
                layer_index=hidden_layer,
            )
            negative_prompt_embeds = [neg_output.embeddings[0].to(device=device, dtype=dtype)]

        # 2. Prepare latents
        latent_height = 2 * (height // vae_scale)
        latent_width = 2 * (width // vae_scale)
        num_channels = self.transformer.config.in_channels

        if latents is None:
            # Generate latents on CPU if generator is CPU (for reproducibility)
            # then move to device
            latents = torch.randn(
                (1, num_channels, latent_height, latent_width),
                generator=generator,
                dtype=torch.float32,
            ).to(device)
        else:
            latents = latents.to(device)

        # 3. Prepare timesteps
        image_seq_len = (latent_height // 2) * (latent_width // 2)
        if shift is not None:
            # Use user-provided shift value
            mu = shift
            logger.info(f"[Pipeline] Using user-provided shift/mu: {mu}")
        else:
            # Calculate shift based on resolution (dynamic shift)
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
            logger.info(f"[Pipeline] Calculated shift/mu for resolution: {mu:.4f}")
        self.scheduler.sigma_min = 0.0
        # IMPORTANT: For diffusers FlowMatchEulerDiscreteScheduler, when use_dynamic_shifting=False,
        # set_timesteps() ignores the mu parameter and uses self.shift instead.
        # Diffusers uses set_shift() method (shift property is read-only).
        # Our FlowMatchScheduler uses direct attribute assignment.
        if hasattr(self.scheduler, 'set_shift'):
            self.scheduler.set_shift(mu)  # diffusers scheduler
        elif hasattr(self.scheduler, 'shift'):
            self.scheduler.shift = mu  # our FlowMatchScheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
        timesteps = self.scheduler.timesteps

        logger.info(f"[Pipeline] Latent shape: {latents.shape}, device={latents.device}")
        logger.info(f"[Pipeline] Prompt embeds: shape={prompt_embeds[0].shape}, device={prompt_embeds[0].device}")
        logger.info(f"[Pipeline] Timesteps: {len(timesteps)}, device={timesteps.device}")
        logger.info(f"[Pipeline] Timestep values: {timesteps.tolist()}")
        logger.info(f"[Pipeline] Scheduler sigmas: {self.scheduler.sigmas.tolist() if hasattr(self.scheduler, 'sigmas') else 'N/A'}")

        # 4. Denoising loop
        logger.info(f"[Pipeline] Starting {num_inference_steps} denoising steps...")

        # Check if using CPU offload mode
        cpu_offload = getattr(self, '_enable_cpu_offload', False)
        if cpu_offload:
            logger.info("[Pipeline] CPU offload mode - moving transformer to GPU for forward pass")

        # Patch transformer with DyPE if enabled
        # Per-request dype_config overrides self.dype_config
        active_dype_config = dype_config if dype_config is not None else self.dype_config
        dype_patched = False
        if active_dype_config is not None and active_dype_config.enabled:
            from llm_dit.utils.dype import patch_zimage_rope, set_zimage_timestep
            logger.info(f"[Pipeline] DyPE enabled: method={active_dype_config.method}, scale={active_dype_config.dype_scale}")
            try:
                patch_zimage_rope(self.transformer, active_dype_config, width, height)
                dype_patched = True
                logger.info(f"[Pipeline] DyPE patched transformer for {width}x{height}")
            except Exception as e:
                logger.warning(f"[Pipeline] Failed to patch DyPE: {e}. Continuing without DyPE.")

        # Initialize Skip Layer Guidance if enabled
        slg = None
        if skip_layer_guidance_scale > 0 and skip_layer_indices is not None:
            from llm_dit.guidance import SkipLayerGuidance
            slg = SkipLayerGuidance(
                skip_layers=skip_layer_indices,
                guidance_scale=skip_layer_guidance_scale,
                guidance_start=skip_layer_start,
                guidance_stop=skip_layer_stop,
                fqn="layers",  # Z-Image transformer uses "layers" for transformer blocks
            )
            logger.info(
                f"[Pipeline] Skip Layer Guidance enabled: "
                f"scale={skip_layer_guidance_scale}, layers={skip_layer_indices}, "
                f"range=[{skip_layer_start:.0%}, {skip_layer_stop:.0%}]"
            )

        # Initialize FMTT (Flow Map Trajectory Tilting) if enabled
        fmtt = None
        fmtt_prompt = None
        if fmtt_guidance_scale > 0:
            from llm_dit.guidance import FMTTGuidance
            from llm_dit.rewards import DifferentiableSigLIP

            # Load or reuse reward function
            # FMTT needs ~4GB for SigLIP - check VRAM and manage encoder if needed
            if fmtt_reward_fn is None:
                # Check if we have a cached reward function on the pipeline
                if hasattr(self, '_fmtt_reward_fn') and self._fmtt_reward_fn is not None:
                    fmtt_reward_fn = self._fmtt_reward_fn
                    logger.info("[Pipeline] Reusing cached SigLIP for FMTT")
                else:
                    # Check available VRAM before loading SigLIP
                    if torch.cuda.is_available():
                        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
                        logger.info(f"[Pipeline] FMTT requested, {free_mem:.1f}GB VRAM free")

                        # SigLIP needs ~4GB, we want some headroom
                        if free_mem < 5.0:
                            # Check if encoder is on CUDA and can be moved
                            if self.encoder is not None:
                                encoder_device = getattr(self.encoder, 'device', None)
                                if encoder_device is not None and 'cuda' in str(encoder_device):
                                    logger.warning(
                                        f"[Pipeline] Insufficient VRAM for FMTT ({free_mem:.1f}GB free, need ~5GB). "
                                        f"Moving encoder to CPU to make room for SigLIP..."
                                    )
                                    self.encoder.to("cpu")
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    free_mem = torch.cuda.mem_get_info()[0] / 1024**3
                                    logger.info(f"[Pipeline] After encoder move: {free_mem:.1f}GB VRAM free")

                        # If still not enough, error out with helpful message
                        if free_mem < 4.0:
                            raise RuntimeError(
                                f"Insufficient VRAM for FMTT: {free_mem:.1f}GB free, need ~4GB for SigLIP. "
                                f"Try: 1) Set encoder_device='cpu' in config, or "
                                f"2) Disable FMTT (fmtt_scale=0)"
                            )

                    # Determine SigLIP device (use param, or fall back to pipeline device)
                    siglip_device = fmtt_siglip_device if fmtt_siglip_device else device
                    logger.info(f"[Pipeline] Loading SigLIP for FMTT: {fmtt_siglip_model} on {siglip_device}")
                    fmtt_reward_fn = DifferentiableSigLIP(
                        model_name=fmtt_siglip_model,
                        device=siglip_device,
                    )

                    # Cache for reuse in future generations
                    self._fmtt_reward_fn = fmtt_reward_fn
                    logger.info("[Pipeline] SigLIP loaded and cached for FMTT")

            fmtt = FMTTGuidance(
                vae=self.vae.vae if hasattr(self.vae, 'vae') else self.vae,
                reward_fn=fmtt_reward_fn,
                guidance_scale=fmtt_guidance_scale,
                guidance_start=fmtt_guidance_start,
                guidance_stop=fmtt_guidance_stop,
                normalize_mode=fmtt_normalize_mode,
                decode_scale=fmtt_decode_scale,
            )

            # Get prompt text for reward computation
            if isinstance(prompt, str):
                fmtt_prompt = prompt
            elif hasattr(prompt, 'messages') and prompt.messages:
                # Extract user message from Conversation
                fmtt_prompt = prompt.messages[-1].content if prompt.messages else ""
            else:
                fmtt_prompt = str(prompt) if prompt is not None else ""

            logger.info(
                f"[Pipeline] FMTT enabled: "
                f"scale={fmtt_guidance_scale}, range=[{fmtt_guidance_start:.0%}, {fmtt_guidance_stop:.0%}]"
            )

        # Run denoising loop with no_grad to prevent gradient accumulation
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                # Update DyPE timestep if patched
                if dype_patched:
                    # Get sigma from scheduler
                    sigma = self.scheduler.sigmas[i].item() if hasattr(self.scheduler, 'sigmas') else t.item() / 1000.0
                    set_zimage_timestep(self.transformer, sigma)
                # Clear cache before each step to minimize peak memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Move transformer to GPU for this step if using CPU offload
                if cpu_offload:
                    self.transformer.to(device)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                # Prepare timestep (inverted for Z-Image)
                timestep = t.expand(latents.shape[0])
                timestep = (1000 - timestep) / 1000

                # Calculate denoising progress (0 to 1) for CFG truncation
                progress = timestep[0].item()

                # Handle CFG with optional truncation
                # CFG truncation: disable CFG after cfg_truncation fraction of progress
                current_cfg_scale = guidance_scale
                if cfg_truncation < 1.0 and progress > cfg_truncation:
                    current_cfg_scale = 0.0

                apply_cfg = current_cfg_scale > 0

                if apply_cfg:
                    latent_input = latents.to(dtype).repeat(2, 1, 1, 1)
                    embeds_input = prompt_embeds + negative_prompt_embeds
                    timestep_input = timestep.repeat(2)
                else:
                    latent_input = latents.to(dtype)
                    embeds_input = prompt_embeds
                    timestep_input = timestep

                # Add temporal dimension for transformer
                latent_input = latent_input.unsqueeze(2)
                latent_list = list(latent_input.unbind(dim=0))

                # Check if SLG is active at this step
                use_slg = slg is not None and slg.is_active(i, num_inference_steps)

                # Run transformer (normal forward pass)
                model_output = self.transformer(
                    latent_list,
                    timestep_input,
                    embeds_input,
                )[0]

                # Run skip-layer forward pass if SLG is active
                if use_slg:
                    with slg.skip_layers_context(self.transformer):
                        model_output_skip = self.transformer(
                            latent_list,
                            timestep_input,
                            embeds_input,
                        )[0]

                # Move transformer back to CPU after forward pass if using CPU offload
                if cpu_offload:
                    self.transformer.to("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Process output with CFG
                if apply_cfg:
                    pos_out = model_output[:1]
                    neg_out = model_output[1:]
                    noise_pred = []
                    for pos, neg in zip(pos_out, neg_out):
                        pos_f = pos.float()
                        neg_f = neg.float()
                        # Apply CFG: positive + scale * (positive - negative)
                        pred = pos_f + current_cfg_scale * (pos_f - neg_f)

                        # CFG normalization: clamp combined norm relative to positive norm
                        if cfg_normalization > 0:
                            pos_norm = torch.linalg.vector_norm(pos_f)
                            pred_norm = torch.linalg.vector_norm(pred)
                            max_allowed_norm = pos_norm * cfg_normalization
                            # Avoid division by zero
                            pred_norm = torch.where(pred_norm < 1e-6, torch.ones_like(pred_norm), pred_norm)
                            scale_factor = torch.clamp(max_allowed_norm / pred_norm, max=1.0)
                            pred = pred * scale_factor

                        noise_pred.append(pred)
                    noise_pred = torch.stack(noise_pred, dim=0)
                else:
                    noise_pred = torch.stack([o.float() for o in model_output], dim=0)

                # Apply Skip Layer Guidance if active
                if use_slg:
                    noise_pred_skip = torch.stack([o.float() for o in model_output_skip], dim=0)
                    # For Z-Image (no CFG), SLG formula: pred = pred_cond + scale * (pred_cond - pred_skip)
                    noise_pred = slg.guide(noise_pred, noise_pred_skip, cfg_scale=current_cfg_scale)

                noise_pred = noise_pred.squeeze(2)

                # Apply FMTT guidance if active at this step
                # FMTT modifies velocity to guide toward higher-reward regions
                use_fmtt = fmtt is not None and fmtt.is_active(i, num_inference_steps)
                if use_fmtt:
                    # Get sigma for flow map prediction
                    sigma = self.scheduler.sigmas[i].item() if hasattr(self.scheduler, 'sigmas') else t.item() / 1000.0

                    # noise_pred is the raw velocity from DiT (before negation)
                    velocity = noise_pred

                    # Compute FMTT gradient (this enables gradients internally)
                    fmtt_grad, reward = fmtt.compute_gradient(
                        latents=latents,
                        velocity=velocity.detach(),
                        sigma=sigma,
                        prompt=fmtt_prompt,
                    )

                    # Apply gradient to velocity
                    velocity = fmtt.guide_velocity(velocity, fmtt_grad)
                    noise_pred = velocity

                    if i % 3 == 0:  # Log every few steps
                        logger.info(f"[FMTT] Step {i}: reward={reward:.4f}, grad_norm={fmtt_grad.norm().item():.4f}")

                noise_pred = -noise_pred  # Negate output for Z-Image

                # Scheduler step
                latents = self.scheduler.step(
                    noise_pred.to(torch.float32),
                    t,
                    latents,
                    return_dict=False,
                )[0]

                # Free intermediate tensors
                del latent_input, latent_list, model_output, noise_pred, timestep
                if apply_cfg:
                    del pos_out, neg_out

                # Callback
                if callback is not None:
                    callback(i, len(timesteps), latents)

                # Progress logging (every few steps)
                if i == 0 or (i + 1) % 3 == 0 or i == len(timesteps) - 1:
                    if torch.cuda.is_available():
                        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
                        alloc_mem = torch.cuda.memory_allocated() / 1024**3
                        logger.info(f"[Pipeline] Step {i+1}/{len(timesteps)} complete (GPU: {alloc_mem:.1f}GB alloc, {free_mem:.1f}GB free)")
                    else:
                        logger.info(f"[Pipeline] Step {i+1}/{len(timesteps)} complete")

        logger.info(f"[Pipeline] Denoising complete, latents shape: {latents.shape}")

        # 5. Decode latents
        if output_type == "latent":
            return latents

        logger.info("[Pipeline] Decoding latents with VAE...")

        # Move VAE to GPU if using CPU offload
        if cpu_offload:
            logger.info("[Pipeline] Moving VAE to GPU for decode...")
            self.vae.to(device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        latents = latents.to(self.vae.dtype)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

        with torch.no_grad():
            image = self.vae.decode(latents, return_dict=False)[0]

        # Move VAE back to CPU if using CPU offload
        if cpu_offload:
            self.vae.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Post-process
        if output_type == "pt":
            return image

        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")

        if image.shape[0] == 1:
            return Image.fromarray(image[0])
        return [Image.fromarray(img) for img in image]

    def encode_prompt(
        self,
        prompt: Union[str, Conversation],
        template: Optional[str] = None,
        force_think_block: bool = False,
        truncate: bool = True,
        long_prompt_mode: str = "truncate",
    ) -> torch.Tensor:
        """
        Encode a prompt without running generation.

        Useful for:
        - Pre-computing embeddings
        - Analyzing embedding differences
        - Experiments comparing prompts

        Args:
            prompt: Text prompt or Conversation object
            template: Optional template name
            force_think_block: If True, add empty think block
            truncate: If True, compress to MAX_TEXT_SEQ_LEN (default: True)
            long_prompt_mode: How to handle long prompts (truncate/interpolate/pool/attention_pool)

        Returns:
            Embedding tensor [seq_len, embed_dim]
        """
        output = self.encoder.encode(
            prompt,
            template=template,
            force_think_block=force_think_block,
        )
        embeddings = output.embeddings[0]
        if truncate and embeddings.shape[0] > MAX_TEXT_SEQ_LEN:
            embeddings = compress_embeddings(embeddings, MAX_TEXT_SEQ_LEN, mode=long_prompt_mode)
        return embeddings

    def generate_from_embeddings(
        self,
        prompt_embeds: torch.Tensor,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 9,
        guidance_scale: float = 0.0,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        shift: Optional[float] = None,
        long_prompt_mode: str = "truncate",
        cfg_normalization: float = 0.0,
        cfg_truncation: float = 1.0,
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor]:
        """
        Generate an image from pre-computed embeddings.

        This is the "generator side" of distributed inference.
        Use this on the CUDA server with embeddings saved from the encoder.

        Args:
            prompt_embeds: Pre-computed prompt embeddings [seq_len, embed_dim]
            height: Output image height (default: 1024, must be divisible by 16)
            width: Output image width (default: 1024, must be divisible by 16)
            num_inference_steps: Number of denoising steps (default: 9 for turbo)
            guidance_scale: CFG scale (default: 0.0, CFG is baked into Z-Image)
            negative_prompt_embeds: Pre-computed negative embeddings for CFG
            generator: Random generator for reproducibility
            latents: Pre-generated latents (optional)
            output_type: Output format ("pil", "latent", or "pt")
            callback: Optional callback for progress updates
            cfg_normalization: CFG norm clamping (0.0 = disabled, typical: 1.0-2.0)
            cfg_truncation: CFG truncation threshold (1.0 = never, typical: 0.5-0.8)

        Returns:
            Generated image(s) in specified format

        Example:
            from llm_dit.distributed import load_embeddings

            # Load embeddings generated on Mac
            emb_file = load_embeddings("sunset.safetensors", device="cuda")

            # Generate on CUDA
            pipe = ZImagePipeline.from_pretrained_generator_only(...)
            image = pipe.generate_from_embeddings(
                emb_file.embeddings,
                generator=torch.Generator().manual_seed(42),
            )
        """
        # Validate dimensions (must be divisible by 16 for Z-Image)
        vae_scale = self.vae_scale_factor * 2  # 16 for Z-Image
        if height % vae_scale != 0:
            raise ValueError(
                f"Height must be divisible by {vae_scale} (got {height})"
            )
        if width % vae_scale != 0:
            raise ValueError(
                f"Width must be divisible by {vae_scale} (got {width})"
            )

        device = self.transformer.device
        dtype = next(self.transformer.parameters()).dtype

        # Compress embeddings if they exceed DiT's max text sequence length
        if prompt_embeds.shape[0] > MAX_TEXT_SEQ_LEN:
            prompt_embeds = compress_embeddings(prompt_embeds, MAX_TEXT_SEQ_LEN, mode=long_prompt_mode)

        # Move embeddings to device
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)

        # Wrap embeddings in list format expected by transformer
        prompt_embeds_list = [prompt_embeds]

        # Prepare negative embeddings if using CFG
        negative_embeds_list = []
        if guidance_scale > 0:
            if negative_prompt_embeds is not None:
                # Compress negative embeddings if needed
                if negative_prompt_embeds.shape[0] > MAX_TEXT_SEQ_LEN:
                    negative_prompt_embeds = compress_embeddings(
                        negative_prompt_embeds, MAX_TEXT_SEQ_LEN, mode=long_prompt_mode
                    )
                negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
                negative_embeds_list = [negative_prompt_embeds]
            else:
                raise ValueError(
                    "negative_prompt_embeds required when guidance_scale > 0. "
                    "For Z-Image-Turbo, guidance_scale=0.0 is recommended."
                )

        # Prepare latents
        latent_height = 2 * (height // vae_scale)
        latent_width = 2 * (width // vae_scale)
        num_channels = self.transformer.config.in_channels

        if latents is None:
            latents = torch.randn(
                (1, num_channels, latent_height, latent_width),
                generator=generator,
                dtype=torch.float32,
            ).to(device)
        else:
            latents = latents.to(device)

        # Prepare timesteps
        image_seq_len = (latent_height // 2) * (latent_width // 2)
        if shift is not None:
            # Use user-provided shift value
            mu = shift
            logger.info(f"[Pipeline] Using user-provided shift/mu: {mu}")
        else:
            # Calculate shift based on resolution (dynamic shift)
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
            logger.info(f"[Pipeline] Calculated shift/mu for resolution: {mu:.4f}")
        self.scheduler.sigma_min = 0.0
        # IMPORTANT: For diffusers FlowMatchEulerDiscreteScheduler, when use_dynamic_shifting=False,
        # set_timesteps() ignores the mu parameter and uses self.shift instead.
        # Diffusers uses set_shift() method (shift property is read-only).
        # Our FlowMatchScheduler uses direct attribute assignment.
        if hasattr(self.scheduler, 'set_shift'):
            self.scheduler.set_shift(mu)  # diffusers scheduler
        elif hasattr(self.scheduler, 'shift'):
            self.scheduler.shift = mu  # our FlowMatchScheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
        timesteps = self.scheduler.timesteps

        # Denoising loop
        logger.debug(f"Running {num_inference_steps} denoising steps...")
        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0])
            timestep = (1000 - timestep) / 1000

            # Calculate denoising progress (0 to 1) for CFG truncation
            progress = timestep[0].item()

            # Handle CFG with optional truncation
            current_cfg_scale = guidance_scale
            if cfg_truncation < 1.0 and progress > cfg_truncation:
                current_cfg_scale = 0.0

            apply_cfg = current_cfg_scale > 0

            if apply_cfg:
                latent_input = latents.to(dtype).repeat(2, 1, 1, 1)
                embeds_input = prompt_embeds_list + negative_embeds_list
                timestep_input = timestep.repeat(2)
            else:
                latent_input = latents.to(dtype)
                embeds_input = prompt_embeds_list
                timestep_input = timestep

            latent_input = latent_input.unsqueeze(2)
            latent_list = list(latent_input.unbind(dim=0))

            model_output = self.transformer(
                latent_list,
                timestep_input,
                embeds_input,
            )[0]

            if apply_cfg:
                pos_out = model_output[:1]
                neg_out = model_output[1:]
                noise_pred = []
                for pos, neg in zip(pos_out, neg_out):
                    pos_f = pos.float()
                    neg_f = neg.float()
                    pred = pos_f + current_cfg_scale * (pos_f - neg_f)

                    # CFG normalization
                    if cfg_normalization > 0:
                        pos_norm = torch.linalg.vector_norm(pos_f)
                        pred_norm = torch.linalg.vector_norm(pred)
                        max_allowed_norm = pos_norm * cfg_normalization
                        pred_norm = torch.where(pred_norm < 1e-6, torch.ones_like(pred_norm), pred_norm)
                        scale_factor = torch.clamp(max_allowed_norm / pred_norm, max=1.0)
                        pred = pred * scale_factor

                    noise_pred.append(pred)
                noise_pred = torch.stack(noise_pred, dim=0)
            else:
                noise_pred = torch.stack([t.float() for t in model_output], dim=0)

            noise_pred = noise_pred.squeeze(2)
            noise_pred = -noise_pred

            latents = self.scheduler.step(
                noise_pred.to(torch.float32),
                t,
                latents,
                return_dict=False,
            )[0]

            if callback is not None:
                callback(i, len(timesteps), latents)

        # Decode latents
        if output_type == "latent":
            return latents

        logger.debug("Decoding latents...")
        latents = latents.to(self.vae.dtype)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

        image = self.vae.decode(latents, return_dict=False)[0]

        if output_type == "pt":
            return image

        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")

        if image.shape[0] == 1:
            return Image.fromarray(image[0])
        return [Image.fromarray(img) for img in image]

    @classmethod
    def from_pretrained_generator_only(
        cls,
        model_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
        enable_cpu_offload: bool = False,
        dit_device: str = "auto",
        vae_device: str = "auto",
        # PyTorch-native component options
        use_custom_scheduler: bool = False,
        tiled_vae: bool = False,
        tile_size: int = 512,
        tile_overlap: int = 64,
        attention_backend: str | None = None,
        **kwargs,
    ) -> "ZImagePipeline":
        """
        Load only the generator components (transformer, VAE, scheduler).

        Use this for the CUDA side of distributed inference when you
        already have pre-computed embeddings.

        Args:
            model_path: Path to Z-Image model or HuggingFace ID
            torch_dtype: Model dtype (default: bfloat16)
            device: Device to load to (default: cuda, legacy parameter)
            enable_cpu_offload: Enable model CPU offload for low VRAM
            dit_device: Device for DiT/transformer (cpu, cuda, mps, auto)
            vae_device: Device for VAE (cpu, cuda, mps, auto)
            use_custom_scheduler: Use our pure-PyTorch FlowMatchScheduler
            tiled_vae: Enable tiled VAE decode for large images (2K+)
            tile_size: Tile size in pixels for VAE decode (default: 512)
            tile_overlap: Overlap between tiles in pixels (default: 64)
            attention_backend: Attention backend (auto, flash_attn_2, sdpa, etc.)
            **kwargs: Additional arguments for diffusers

        Returns:
            ZImagePipeline with encoder=None (use generate_from_embeddings only)

        Example:
            pipe = ZImagePipeline.from_pretrained_generator_only(
                "/path/to/z-image",
                dit_device="cuda",
                vae_device="cuda",
                tiled_vae=True,  # For 2K+ images
            )
            image = pipe.generate_from_embeddings(embeddings)
        """
        try:
            from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
            from diffusers.models import ZImageTransformer2DModel
        except ImportError as e:
            raise ImportError(
                "diffusers is required for ZImagePipeline. "
                "Install with: pip install diffusers[torch]"
            ) from e

        # Set up attention backend
        if attention_backend:
            setup_attention_backend(attention_backend)

        # Resolve device strings
        def resolve_device(device_str: str) -> str:
            if device_str == "auto":
                if torch.cuda.is_available():
                    return "cuda"
                elif torch.backends.mps.is_available():
                    return "mps"
                else:
                    return "cpu"
            return device_str

        # Use new device params if provided, otherwise fall back to legacy 'device'
        dit_device_resolved = resolve_device(dit_device) if dit_device != "auto" or device == "cuda" else str(device)
        vae_device_resolved = resolve_device(vae_device) if vae_device != "auto" or device == "cuda" else str(device)

        # Re-resolve in case we got legacy device
        dit_device_resolved = resolve_device(dit_device_resolved)
        vae_device_resolved = resolve_device(vae_device_resolved)

        model_path = Path(model_path)
        logger.info("=" * 60)
        logger.info("LOADING GENERATOR COMPONENTS (encoder-free mode)")
        logger.info("=" * 60)
        logger.info(f"  model_path: {model_path}")
        logger.info(f"  torch_dtype: {torch_dtype}")
        logger.info(f"  dit_device: {dit_device_resolved}")
        logger.info(f"  vae_device: {vae_device_resolved}")
        logger.info(f"  enable_cpu_offload: {enable_cpu_offload}")
        logger.info("-" * 60)

        # Check available GPU memory before loading
        if torch.cuda.is_available():
            free_mem = torch.cuda.mem_get_info()[0] / 1024**3
            total_mem = torch.cuda.mem_get_info()[1] / 1024**3
            logger.info(f"  GPU memory: {free_mem:.1f}GB free / {total_mem:.1f}GB total")

        # Load only the components we need (skip text encoder entirely)
        # Use low_cpu_mem_usage to avoid OOM during dtype conversion
        logger.info("Loading transformer (low_cpu_mem_usage=True)...")
        transformer = ZImageTransformer2DModel.from_pretrained(
            model_path / "transformer",
            torch_dtype=torch_dtype,  # diffusers uses torch_dtype
            low_cpu_mem_usage=True,
            **kwargs,
        )
        logger.info(f"  Transformer loaded: {transformer.__class__.__name__}")
        logger.info(f"  Transformer dtype: {next(transformer.parameters()).dtype}")
        logger.info(f"  Transformer device (before move): {next(transformer.parameters()).device}")

        if torch.cuda.is_available():
            free_mem = torch.cuda.mem_get_info()[0] / 1024**3
            logger.info(f"  GPU memory after transformer load: {free_mem:.1f}GB free")

        logger.info("Loading VAE (low_cpu_mem_usage=True)...")
        vae = AutoencoderKL.from_pretrained(
            model_path / "vae",
            torch_dtype=torch_dtype,  # diffusers uses torch_dtype
            low_cpu_mem_usage=True,
            **kwargs,
        )
        logger.info(f"  VAE loaded: {vae.__class__.__name__}")
        logger.info(f"  VAE dtype: {next(vae.parameters()).dtype}")

        # Load scheduler (custom or diffusers)
        if use_custom_scheduler:
            from llm_dit.schedulers import FlowMatchScheduler
            scheduler = FlowMatchScheduler(shift=3.0)
            logger.info("Using custom FlowMatchScheduler (pure PyTorch)")
        else:
            logger.info("Loading scheduler...")
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                model_path / "scheduler",
            )
            logger.info(f"  Scheduler: {scheduler.__class__.__name__}")

        # Move to device (unless using CPU offload)
        logger.info("-" * 60)
        if enable_cpu_offload:
            logger.info("CPU offload enabled - both transformer and VAE stay on CPU")
            logger.info("  Transformer: moves to GPU for each denoising step")
            logger.info("  VAE: moves to GPU only for final decode")
        else:
            logger.info(f"Moving transformer to {dit_device_resolved}...")
            transformer = transformer.to(dit_device_resolved)
            logger.info(f"  Transformer now on: {next(transformer.parameters()).device}")

            if torch.cuda.is_available():
                free_mem = torch.cuda.mem_get_info()[0] / 1024**3
                logger.info(f"  GPU memory after transformer move: {free_mem:.1f}GB free")

            logger.info(f"Moving VAE to {vae_device_resolved}...")
            vae = vae.to(vae_device_resolved)
            logger.info(f"  VAE now on: {next(vae.parameters()).device}")

        # Create pipeline without encoder
        pipeline = cls.__new__(cls)
        pipeline.encoder = None
        pipeline.transformer = transformer
        pipeline.scheduler = scheduler
        pipeline.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        pipeline._enable_cpu_offload = enable_cpu_offload
        pipeline._tiled_vae_enabled = tiled_vae

        # Wrap VAE with tiled decoder if requested
        if tiled_vae:
            from llm_dit.utils.tiled_vae import TiledVAEDecoder
            pipeline.vae = TiledVAEDecoder(vae, tile_size=tile_size, tile_overlap=tile_overlap)
            logger.info(f"Tiled VAE enabled: tile_size={tile_size}, overlap={tile_overlap}")
        else:
            pipeline.vae = vae

        logger.info("-" * 60)
        logger.info("Generator loaded successfully (encoder-free mode)")
        if device == "cuda" and torch.cuda.is_available():
            free_mem = torch.cuda.mem_get_info()[0] / 1024**3
            used_mem = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1024**3
            logger.info(f"  Final GPU memory: {used_mem:.1f}GB used, {free_mem:.1f}GB free")
        logger.info("=" * 60)
        return pipeline

    def generate_multipass(
        self,
        prompt: Union[str, Conversation, None] = None,
        final_width: int = 2048,
        final_height: int = 2048,
        passes: Optional[List[Dict[str, Any]]] = None,
        generator: Optional[torch.Generator] = None,
        template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        thinking_content: Optional[str] = None,
        assistant_content: Optional[str] = None,
        force_think_block: bool = False,
        remove_quotes: bool = False,
        output_type: str = "pil",
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        long_prompt_mode: str = "truncate",
        hidden_layer: int = -2,
        prompt_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor]:
        """
        Generate a high-resolution image using multi-pass workflow.

        Multi-pass generation produces higher quality results at high resolutions
        (2K-4K+) by first generating at a lower resolution, then upscaling through
        img2img passes. Each pass can independently apply DyPE position extrapolation.

        Why multi-pass works better:
        - First pass: Establishes global composition and structure at lower res
        - Second pass: Refines details at target resolution with DyPE support
        - Each pass benefits from DyPE's timestep-aware frequency modulation

        Args:
            prompt: Text prompt or Conversation object
            final_width: Target output width in pixels (default: 2048)
            final_height: Target output height in pixels (default: 2048)
            passes: List of pass configurations. Each dict can contain:
                - scale: Resolution scale relative to final (0.5 = half res)
                - steps: Number of inference steps for this pass
                - strength: img2img strength (for passes after the first)
                - shift: Optional scheduler shift override
                Default: [{"scale": 0.5, "steps": 9}, {"scale": 1.0, "steps": 9, "strength": 0.5}]
            generator: Random generator for reproducibility
            template: Template name for encoding
            system_prompt: System prompt (optional)
            thinking_content: Content inside <think>...</think>
            assistant_content: Content after </think>
            force_think_block: Add empty think block even without content
            remove_quotes: Strip " characters
            output_type: Output format ("pil", "latent", or "pt")
            callback: Progress callback (called for each pass)
            long_prompt_mode: How to handle prompts > 1504 tokens
            hidden_layer: Which LLM hidden layer to extract embeddings from
            prompt_embeds: Pre-computed embeddings (skip text encoding)
            **kwargs: Additional arguments passed to each pass

        Returns:
            Generated image in specified format

        Example:
            # Two-pass 4K generation (recommended)
            image = pipe.generate_multipass(
                "A detailed cityscape",
                final_width=4096,
                final_height=4096,
                passes=[
                    {"scale": 0.5, "steps": 9},      # 2K first pass
                    {"scale": 1.0, "steps": 9, "strength": 0.5},  # 4K refinement
                ],
            )

            # Three-pass for maximum quality
            image = pipe.generate_multipass(
                "A portrait",
                final_width=4096,
                final_height=4096,
                passes=[
                    {"scale": 0.25, "steps": 9},     # 1K base
                    {"scale": 0.5, "steps": 9, "strength": 0.6},  # 2K
                    {"scale": 1.0, "steps": 9, "strength": 0.4},  # 4K
                ],
            )
        """
        # Default two-pass configuration
        if passes is None:
            passes = [
                {"scale": 0.5, "steps": 9},  # First pass: half resolution
                {"scale": 1.0, "steps": 9, "strength": 0.5},  # Second pass: full res img2img
            ]

        # Validate passes
        if len(passes) < 1:
            raise ValueError("At least one pass is required")
        if passes[0].get("strength") is not None:
            logger.warning("First pass strength is ignored (txt2img)")

        # Validate final dimensions
        vae_scale = self.vae_scale_factor * 2  # 16 for Z-Image
        if final_width % vae_scale != 0:
            final_width = (final_width // vae_scale) * vae_scale
            logger.info(f"Adjusted final_width to {final_width}")
        if final_height % vae_scale != 0:
            final_height = (final_height // vae_scale) * vae_scale
            logger.info(f"Adjusted final_height to {final_height}")

        logger.info(f"[Multipass] Starting {len(passes)}-pass generation: {final_width}x{final_height}")

        result = None
        for pass_idx, pass_config in enumerate(passes):
            scale = pass_config.get("scale", 1.0)
            steps = pass_config.get("steps", 9)
            strength = pass_config.get("strength")
            shift = pass_config.get("shift")
            guidance_scale = pass_config.get("guidance_scale", 0.0)

            # Compute dimensions for this pass
            pass_width = int(final_width * scale)
            pass_height = int(final_height * scale)

            # Ensure divisible by VAE scale
            pass_width = (pass_width // vae_scale) * vae_scale
            pass_height = (pass_height // vae_scale) * vae_scale

            logger.info(f"[Multipass] Pass {pass_idx + 1}/{len(passes)}: {pass_width}x{pass_height}, steps={steps}")

            if result is None:
                # First pass: txt2img
                logger.info(f"[Multipass] Pass {pass_idx + 1}: txt2img at {pass_width}x{pass_height}")
                result = self(
                    prompt=prompt,
                    width=pass_width,
                    height=pass_height,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    template=template,
                    system_prompt=system_prompt,
                    thinking_content=thinking_content,
                    assistant_content=assistant_content,
                    force_think_block=force_think_block,
                    remove_quotes=remove_quotes,
                    output_type="pil",  # Always PIL for intermediate passes
                    callback=callback,
                    shift=shift,
                    long_prompt_mode=long_prompt_mode,
                    hidden_layer=hidden_layer,
                    prompt_embeds=prompt_embeds,
                    **kwargs,  # Pass through CFG, SLG, FMTT, DyPE, etc.
                )
            else:
                # Subsequent passes: img2img
                if strength is None:
                    strength = 0.5  # Default strength for refinement

                logger.info(f"[Multipass] Pass {pass_idx + 1}: img2img at {pass_width}x{pass_height}, strength={strength}")
                result = self.img2img(
                    prompt=prompt,
                    image=result,
                    strength=strength,
                    width=pass_width,
                    height=pass_height,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    template=template,
                    system_prompt=system_prompt,
                    thinking_content=thinking_content,
                    assistant_content=assistant_content,
                    force_think_block=force_think_block,
                    remove_quotes=remove_quotes,
                    output_type="pil",  # Always PIL for intermediate passes
                    callback=callback,
                    shift=shift,
                    long_prompt_mode=long_prompt_mode,
                    hidden_layer=hidden_layer,
                    prompt_embeds=prompt_embeds,
                    **kwargs,  # Pass through CFG, SLG, FMTT, DyPE, etc.
                )

            logger.info(f"[Multipass] Pass {pass_idx + 1} complete")

        # Convert final result to requested output type
        if output_type == "pil":
            return result
        elif output_type == "pt":
            import numpy as np
            if isinstance(result, Image.Image):
                img_array = np.array(result).astype(np.float32) / 255.0
                return torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
            return result
        elif output_type == "latent":
            # Re-encode to latent space
            return self.encode_image(result)

        return result
