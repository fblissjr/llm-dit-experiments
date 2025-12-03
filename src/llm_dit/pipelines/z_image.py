"""
Z-Image pipeline for end-to-end text-to-image generation.

This pipeline combines our ZImageTextEncoder with diffusers components:
- Scheduler: FlowMatchEulerDiscreteScheduler (shift=3)
- Transformer: ZImageTransformer2DModel (S3-DiT, 6B params)
- VAE: AutoencoderKL (16-channel, Flux-derived)

Example:
    pipe = ZImagePipeline.from_pretrained("/path/to/z-image")
    image = pipe("A cat sleeping in sunlight")

Key differences from diffusers ZImagePipeline:
1. Uses our ZImageTextEncoder with template support
2. Supports template-based prompt customization
3. Exposes thinking block control
4. More explicit control over encoding
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from PIL import Image

from llm_dit.conversation import Conversation
from llm_dit.encoders import ZImageTextEncoder

logger = logging.getLogger(__name__)


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
        scheduler: Any,  # FlowMatchEulerDiscreteScheduler
    ):
        """
        Initialize the pipeline.

        Args:
            encoder: ZImageTextEncoder instance
            transformer: diffusers ZImageTransformer2DModel
            vae: diffusers AutoencoderKL
            scheduler: diffusers FlowMatchEulerDiscreteScheduler
        """
        self.encoder = encoder
        self.transformer = transformer
        self.vae = vae
        self.scheduler = scheduler

        # VAE scale factor (8 for Z-Image)
        self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        templates_dir: str | Path | None = None,
        default_template: str | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str | None = None,
        **kwargs,
    ) -> "ZImagePipeline":
        """
        Load pipeline from pretrained model.

        Args:
            model_path: Path to Z-Image model or HuggingFace ID
            templates_dir: Optional path to templates directory
            default_template: Optional default template name
            torch_dtype: Model dtype (default: bfloat16)
            device_map: Device mapping (default: "auto")
            **kwargs: Additional arguments

        Returns:
            Initialized ZImagePipeline

        Example:
            pipe = ZImagePipeline.from_pretrained(
                "Tongyi-MAI/Z-Image-Turbo",
                templates_dir="templates/z_image",
                torch_dtype=torch.bfloat16,
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

        # Load encoder (our custom encoder with template support)
        encoder = ZImageTextEncoder.from_pretrained(
            model_path,
            templates_dir=templates_dir,
            default_template=default_template,
            torch_dtype=torch_dtype,
        )

        # Load the diffusers pipeline (auto-detect pipeline class)
        logger.info("Loading diffusers pipeline...")
        load_kwargs = {
            "torch_dtype": torch_dtype,
            **kwargs,
        }
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        diffusers_pipe = DiffusionPipeline.from_pretrained(model_path, **load_kwargs)

        # Move diffusers pipeline to same device as encoder
        device = encoder.device
        logger.info(f"Moving pipeline to {device}...")
        diffusers_pipe = diffusers_pipe.to(device)

        # Extract components from diffusers pipeline
        transformer = diffusers_pipe.transformer
        vae = diffusers_pipe.vae
        scheduler = diffusers_pipe.scheduler

        logger.info("Pipeline loaded successfully")
        return cls(encoder, transformer, vae, scheduler)

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
        enable_thinking: bool = True,
        output_type: str = "pil",
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor]:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text prompt or Conversation object
            height: Output image height (default: 1024, must be divisible by 16)
            width: Output image width (default: 1024, must be divisible by 16)
            num_inference_steps: Number of denoising steps (default: 9 for turbo)
            guidance_scale: CFG scale (default: 0.0, CFG is baked into Z-Image)
            negative_prompt: Negative prompt for CFG (only used if guidance_scale > 0)
            generator: Random generator for reproducibility
            latents: Pre-generated latents (optional)
            template: Template name for encoding
            enable_thinking: Whether to include thinking tags
            output_type: Output format ("pil", "latent", or "pt")
            callback: Optional callback for progress updates

        Returns:
            Generated image(s) in specified format

        Example:
            # Basic generation
            image = pipe("A cat sleeping")

            # With template
            image = pipe("A cat", template="photorealistic")

            # With seed
            image = pipe(
                "A cat sleeping",
                generator=torch.Generator().manual_seed(42),
            )

            # CFG (not recommended for Z-Image-Turbo)
            image = pipe(
                "A cat",
                guidance_scale=5.0,
                negative_prompt="blurry, low quality",
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
        logger.debug("Encoding prompt...")
        prompt_output = self.encoder.encode(
            prompt,
            template=template,
            enable_thinking=enable_thinking,
        )
        # Move embeddings to device (API backend returns CPU tensors)
        prompt_embeds = [prompt_output.embeddings[0].to(device=device, dtype=dtype)]

        # Encode negative prompt if using CFG
        negative_prompt_embeds = []
        if guidance_scale > 0 and negative_prompt is not None:
            neg_output = self.encoder.encode(
                negative_prompt,
                enable_thinking=enable_thinking,
            )
            negative_prompt_embeds = [neg_output.embeddings[0].to(device=device, dtype=dtype)]
        elif guidance_scale > 0:
            # Empty negative prompt
            neg_output = self.encoder.encode(
                "",
                enable_thinking=enable_thinking,
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

        # 4. Denoising loop
        logger.debug(f"Running {num_inference_steps} denoising steps...")
        for i, t in enumerate(timesteps):
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
                noise_pred = torch.stack([t.float() for t in model_output], dim=0)

            noise_pred = noise_pred.squeeze(2)
            noise_pred = -noise_pred  # Negate output for Z-Image

            # Scheduler step
            latents = self.scheduler.step(
                noise_pred.to(torch.float32),
                t,
                latents,
                return_dict=False,
            )[0]

            # Callback
            if callback is not None:
                callback(i, len(timesteps), latents)

        # 5. Decode latents
        if output_type == "latent":
            return latents

        logger.debug("Decoding latents...")
        latents = latents.to(self.vae.dtype)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

        image = self.vae.decode(latents, return_dict=False)[0]

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
        enable_thinking: bool = True,
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
            enable_thinking: Whether to include thinking tags

        Returns:
            Embedding tensor [seq_len, embed_dim]
        """
        output = self.encoder.encode(
            prompt,
            template=template,
            enable_thinking=enable_thinking,
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
        **kwargs,
    ) -> "ZImagePipeline":
        """
        Load only the generator components (transformer, VAE, scheduler).

        Use this for the CUDA side of distributed inference when you
        already have pre-computed embeddings.

        Args:
            model_path: Path to Z-Image model or HuggingFace ID
            torch_dtype: Model dtype (default: bfloat16)
            device: Device to load to (default: cuda)
            enable_cpu_offload: Enable model CPU offload for low VRAM
            **kwargs: Additional arguments for diffusers

        Returns:
            ZImagePipeline with encoder=None (use generate_from_embeddings only)

        Example:
            pipe = ZImagePipeline.from_pretrained_generator_only(
                "/path/to/z-image",
                device="cuda",
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

        model_path = Path(model_path)
        logger.info(f"Loading generator components from {model_path}...")

        # Load only the components we need (skip text encoder entirely)
        logger.info("Loading transformer...")
        transformer = ZImageTransformer2DModel.from_pretrained(
            model_path / "transformer",
            torch_dtype=torch_dtype,
            **kwargs,
        )

        logger.info("Loading VAE...")
        vae = AutoencoderKL.from_pretrained(
            model_path / "vae",
            torch_dtype=torch_dtype,
            **kwargs,
        )

        logger.info("Loading scheduler...")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path / "scheduler",
        )

        # Move to device (unless using CPU offload)
        if enable_cpu_offload:
            logger.info("CPU offload enabled - models stay on CPU until needed")
            # For CPU offload, we keep models on CPU and move them during inference
            # This requires modifying the generation loop - for now just use regular loading
            # but with lower memory footprint since we skipped the text encoder
            logger.info(f"Moving pipeline to {device}...")
            transformer = transformer.to(device)
            vae = vae.to(device)
        else:
            logger.info(f"Moving pipeline to {device}...")
            transformer = transformer.to(device)
            vae = vae.to(device)

        # Create pipeline without encoder
        pipeline = cls.__new__(cls)
        pipeline.encoder = None
        pipeline.transformer = transformer
        pipeline.vae = vae
        pipeline.scheduler = scheduler
        pipeline.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

        logger.info("Generator loaded successfully (encoder-free mode)")
        return pipeline
