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
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from PIL import Image

from llm_dit.conversation import Conversation
from llm_dit.encoders import ZImageTextEncoder

logger = logging.getLogger(__name__)


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
        """
        self.encoder = encoder
        self.transformer = transformer
        self.scheduler = scheduler

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
        )

        # Load the diffusers pipeline (auto-detect pipeline class)
        logger.info("Loading diffusers pipeline...")
        load_kwargs = {
            "torch_dtype": torch_dtype,
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

    def __call__(
        self,
        prompt: Union[str, Conversation],
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

        # 1. Encode prompt
        logger.info(f"[Pipeline] Encoding prompt on device={device}, dtype={dtype}")
        logger.info(f"[Pipeline] Encoder type: {type(self.encoder).__name__}")
        backend = getattr(self.encoder, 'backend', None)
        logger.info(f"[Pipeline] Encoder backend: {type(backend).__name__ if backend else 'local'}")

        prompt_output = self.encoder.encode(
            prompt,
            template=template,
            system_prompt=system_prompt,
            thinking_content=thinking_content,
            assistant_content=assistant_content,
            force_think_block=force_think_block,
            remove_quotes=remove_quotes,
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

        # Move embeddings to device (API backend returns CPU tensors)
        prompt_embeds = [raw_embeds.to(device=device, dtype=dtype)]
        logger.info(f"[Pipeline] Moved embeddings to: device={prompt_embeds[0].device}, dtype={prompt_embeds[0].dtype}")

        # Encode negative prompt if using CFG
        negative_prompt_embeds = []
        if guidance_scale > 0 and negative_prompt is not None:
            neg_output = self.encoder.encode(
                negative_prompt,
                force_think_block=force_think_block,
            )
            negative_prompt_embeds = [neg_output.embeddings[0].to(device=device, dtype=dtype)]
        elif guidance_scale > 0:
            # Empty negative prompt
            neg_output = self.encoder.encode(
                "",
                force_think_block=force_think_block,
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

        # Run denoising loop with no_grad to prevent gradient accumulation
        with torch.no_grad():
            for i, t in enumerate(timesteps):
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

                # Handle CFG
                apply_cfg = guidance_scale > 0

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

                # Run transformer
                model_output = self.transformer(
                    latent_list,
                    timestep_input,
                    embeds_input,
                )[0]

                # Move transformer back to CPU after forward pass if using CPU offload
                if cpu_offload:
                    self.transformer.to("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Process output
                if apply_cfg:
                    pos_out = model_output[:1]
                    neg_out = model_output[1:]
                    noise_pred = []
                    for pos, neg in zip(pos_out, neg_out):
                        pred = pos.float() + guidance_scale * (pos.float() - neg.float())
                        noise_pred.append(pred)
                    noise_pred = torch.stack(noise_pred, dim=0)
                else:
                    noise_pred = torch.stack([o.float() for o in model_output], dim=0)

                noise_pred = noise_pred.squeeze(2)
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

        Returns:
            Embedding tensor [seq_len, embed_dim]
        """
        output = self.encoder.encode(
            prompt,
            template=template,
            force_think_block=force_think_block,
        )
        return output.embeddings[0]

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

        # Move embeddings to device
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)

        # Wrap embeddings in list format expected by transformer
        prompt_embeds_list = [prompt_embeds]

        # Prepare negative embeddings if using CFG
        negative_embeds_list = []
        if guidance_scale > 0:
            if negative_prompt_embeds is not None:
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
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.sigma_min = 0.0
        self.scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
        timesteps = self.scheduler.timesteps

        # Denoising loop
        logger.debug(f"Running {num_inference_steps} denoising steps...")
        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0])
            timestep = (1000 - timestep) / 1000

            apply_cfg = guidance_scale > 0

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
                    pred = pos.float() + guidance_scale * (pos.float() - neg.float())
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
            torch_dtype=torch_dtype,
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
            torch_dtype=torch_dtype,
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
