"""
Qwen-Image-Layered pipeline for image-to-layers decomposition.

This pipeline implements multi-layer decomposition using the Qwen-Image-Layered model:
- Text encoder: Qwen2.5-VL-7B-Instruct (3584 hidden dim)
- DiT: 60-layer dual-stream transformer
- VAE: 3D causal VAE (16 channels)

The model takes an input image and decomposes it into N+1 layers:
- Layer 0: Composite/base layer
- Layers 1-N: RGBA decomposition layers

Example:
    pipe = QwenImagePipeline.from_pretrained("/path/to/Qwen_Qwen-Image-Layered")
    layers = pipe.decompose(
        image=input_image,
        prompt="A cheerful child waving under a blue sky",
        layer_num=3,
    )
    # layers is a list of PIL.Image (RGBA format)
"""

import logging
import math
from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image
from tqdm import tqdm

from llm_dit.backends.qwen_image import QwenImageTextEncoderBackend
from llm_dit.models.qwen_image_vae import QwenImageVAE
from llm_dit.models.qwen_image_dit import QwenImageDiT
from llm_dit.utils.latent_packing import (
    pack_latents_2x2,
    unpack_latents_2x2,
    pack_multi_layer_latents,
    unpack_multi_layer_latents,
    get_img_shapes_for_rope,
)

logger = logging.getLogger(__name__)

# Supported resolutions (from training data)
SUPPORTED_RESOLUTIONS = (640, 1024)


def calculate_dynamic_shift(seq_len: int, base_seq_len: int = 256) -> float:
    """
    Calculate dynamic shift for flow matching scheduler.

    Based on DiffSynth FlowMatchScheduler shift calculation:
        shift_mu = (seq_len / base_seqlen) ** 0.5

    Args:
        seq_len: Current sequence length (latent H * W / 4)
        base_seq_len: Base sequence length (256 * 256 / 16 / 16 = 256)

    Returns:
        Dynamic shift value
    """
    return (seq_len / base_seq_len) ** 0.5


class QwenImagePipeline:
    """
    Pipeline for Qwen-Image-Layered image decomposition.

    This pipeline performs image-to-layers decomposition using the Qwen-Image-Layered
    model, outputting multiple RGBA layers that can be composited.

    Attributes:
        text_encoder: QwenImageTextEncoderBackend
        dit: QwenImageDiT transformer
        vae: QwenImageVAE encoder/decoder
    """

    def __init__(
        self,
        text_encoder: QwenImageTextEncoderBackend,
        dit: QwenImageDiT,
        vae: QwenImageVAE,
    ):
        """
        Initialize the pipeline.

        Args:
            text_encoder: Qwen2.5-VL text encoder backend
            dit: QwenImageDiT transformer
            vae: QwenImageVAE encoder/decoder
        """
        self.text_encoder = text_encoder
        self.dit = dit
        self.vae = vae

        # VAE scale factor (8 for Qwen-Image)
        self.vae_scale_factor = 8

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cuda",
        text_encoder_device: str | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> "QwenImagePipeline":
        """
        Load pipeline from pretrained Qwen-Image-Layered model.

        Args:
            model_path: Path to Qwen-Image-Layered model directory
            device: Device for DiT and VAE
            text_encoder_device: Device for text encoder (defaults to device)
            torch_dtype: Model dtype (default: bfloat16)
            **kwargs: Additional arguments

        Returns:
            Initialized QwenImagePipeline

        Example:
            pipe = QwenImagePipeline.from_pretrained(
                "/path/to/Qwen_Qwen-Image-Layered",
                device="cuda",
                torch_dtype=torch.bfloat16,
            )
        """
        model_path = Path(model_path)
        text_encoder_device = text_encoder_device or device

        logger.info(f"Loading Qwen-Image-Layered from {model_path}")

        # Load text encoder
        logger.info(f"Loading text encoder on {text_encoder_device}")
        text_encoder = QwenImageTextEncoderBackend.from_pretrained(
            model_path,
            device=text_encoder_device,
            torch_dtype=torch_dtype,
        )

        # Load VAE
        logger.info(f"Loading VAE on {device}")
        vae = QwenImageVAE.from_pretrained(
            model_path,
            device=device,
            torch_dtype=torch_dtype,
        )

        # Load DiT
        logger.info(f"Loading DiT on {device}")
        dit = QwenImageDiT.from_pretrained(
            model_path,
            device=device,
            torch_dtype=torch_dtype,
            use_layer3d_rope=True,  # Enable layer-aware RoPE for decomposition
        )

        logger.info("Qwen-Image-Layered pipeline loaded successfully")
        return cls(text_encoder, dit, vae)

    @property
    def device(self) -> torch.device:
        """Return DiT device."""
        return self.dit.device

    @property
    def dtype(self) -> torch.dtype:
        """Return model dtype."""
        return self.dit.dtype

    def _validate_resolution(self, height: int, width: int) -> None:
        """Validate that resolution is supported."""
        # Both dimensions must be divisible by 16 (VAE constraint)
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Height and width must be divisible by 16. "
                f"Got height={height}, width={width}"
            )

        # Check if resolution matches supported bases
        # Qwen-Image-Layered was trained on 640 and 1024 base resolutions
        base_res = min(height, width)
        if base_res not in SUPPORTED_RESOLUTIONS:
            logger.warning(
                f"Resolution {width}x{height} may produce suboptimal results. "
                f"Supported base resolutions: {SUPPORTED_RESOLUTIONS}"
            )

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode PIL image to VAE latents."""
        # Convert to tensor
        import torchvision.transforms as T

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),  # Scale to [-1, 1]
        ])

        img_tensor = transform(image).unsqueeze(0).to(
            device=self.vae.device,
            dtype=self.vae.dtype,
        )

        # Encode
        latents = self.vae.encode(img_tensor)
        return latents

    def _decode_latents(
        self,
        latents: torch.Tensor,
        layer_num: int | None = None,
    ) -> Union[Image.Image, List[Image.Image]]:
        """Decode latents to PIL images."""
        # Decode
        decoded = self.vae.decode(latents, num_layers=layer_num)

        # Convert to PIL
        if isinstance(decoded, list):
            # Multi-layer output
            images = []
            for layer_tensor in decoded:
                images.append(self._tensor_to_pil(layer_tensor))
            return images
        else:
            return self._tensor_to_pil(decoded)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image."""
        # Tensor is (C, H, W) in [-1, 1] range
        tensor = tensor.cpu().float()
        tensor = (tensor + 1) / 2  # Scale to [0, 1]
        tensor = tensor.clamp(0, 1)

        # Convert channels
        if tensor.shape[0] == 4:
            # RGBA
            mode = "RGBA"
        else:
            # RGB
            mode = "RGB"

        # Convert to numpy and PIL
        tensor = (tensor * 255).byte().permute(1, 2, 0).numpy()
        return Image.fromarray(tensor, mode=mode)

    @torch.no_grad()
    def decompose(
        self,
        image: Image.Image,
        prompt: str,
        layer_num: int = 3,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 30,
        cfg_scale: float = 4.0,
        seed: int | None = None,
        shift: float | None = None,
        progress_callback: Optional[callable] = None,
    ) -> List[Image.Image]:
        """
        Decompose an image into multiple RGBA layers.

        Args:
            image: Input PIL image to decompose
            prompt: Text description of the image
            layer_num: Number of decomposition layers (default: 3)
            height: Output height (defaults to image height, must be divisible by 16)
            width: Output width (defaults to image width, must be divisible by 16)
            num_inference_steps: Number of denoising steps (default: 30)
            cfg_scale: Classifier-free guidance scale (default: 4.0)
            seed: Random seed for reproducibility
            shift: Scheduler shift (computed dynamically if None)
            progress_callback: Optional callback for progress updates

        Returns:
            List of PIL Images:
            - images[0]: Composite (input) layer
            - images[1:layer_num+1]: Decomposed RGBA layers
        """
        # Determine output size
        if height is None:
            height = image.height
        if width is None:
            width = image.width

        # Validate resolution
        self._validate_resolution(height, width)

        # Resize input image if needed
        if image.size != (width, height):
            image = image.resize((width, height), Image.LANCZOS)

        # Ensure RGBA
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        logger.info(
            f"Decomposing image: {width}x{height}, "
            f"layers={layer_num}, steps={num_inference_steps}"
        )

        # Encode input image
        input_latents = self._encode_image(image)
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor

        # Encode prompt
        encode_output = self.text_encoder.encode(
            [prompt],
            return_padded=True,
        )
        prompt_embeds = encode_output.padded_embeddings.to(self.device, self.dtype)
        prompt_mask = encode_output.padded_mask.to(self.device)

        # Generate noise for decomposition layers
        total_layers = layer_num + 1  # Decomposition layers + composite
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)
        else:
            generator = None

        noise = torch.randn(
            (total_layers, 16, latent_h, latent_w),
            generator=generator,
            device="cpu",
            dtype=self.dtype,
        ).to(self.device)

        # Pack latents for DiT
        packed_noise = pack_multi_layer_latents(noise, height, width, layer_num)

        # Compute img_shapes for RoPE
        img_shapes = get_img_shapes_for_rope(
            height, width, layer_num,
            include_condition=True,
            condition_height=height,
            condition_width=width,
        )

        # Compute dynamic shift if not provided
        if shift is None:
            seq_len = (latent_h // 2) * (latent_w // 2)
            shift = calculate_dynamic_shift(seq_len)
            logger.debug(f"Using dynamic shift: {shift:.4f}")

        # Set up scheduler (simple Euler discrete)
        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
        # Apply shift transformation
        shifted_sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = shifted_sigmas.to(self.device, self.dtype)

        # Denoising loop
        latents = packed_noise
        for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
            # Prepare timestep
            t_batch = t.unsqueeze(0).expand(latents.shape[0])

            # Classifier-free guidance
            if cfg_scale > 1.0:
                # Unconditional pass
                uncond_embeds = torch.zeros_like(prompt_embeds)
                uncond_mask = torch.ones_like(prompt_mask)

                noise_pred_uncond = self.dit(
                    latents=latents,
                    timestep=t_batch,
                    prompt_embeds=uncond_embeds,
                    prompt_mask=uncond_mask,
                    height=height,
                    width=width,
                    img_shapes=img_shapes,
                    layer_num=layer_num,
                )

                # Conditional pass
                noise_pred_cond = self.dit(
                    latents=latents,
                    timestep=t_batch,
                    prompt_embeds=prompt_embeds,
                    prompt_mask=prompt_mask,
                    height=height,
                    width=width,
                    img_shapes=img_shapes,
                    layer_num=layer_num,
                )

                # Apply CFG
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # No guidance
                noise_pred = self.dit(
                    latents=latents,
                    timestep=t_batch,
                    prompt_embeds=prompt_embeds,
                    prompt_mask=prompt_mask,
                    height=height,
                    width=width,
                    img_shapes=img_shapes,
                    layer_num=layer_num,
                )

            # Euler step
            dt = 1.0 / num_inference_steps if i < len(timesteps) - 1 else 0.0
            latents = latents - dt * noise_pred

            # Progress callback
            if progress_callback is not None:
                progress_callback(i + 1, num_inference_steps)

        # Unpack latents
        unpacked_latents = unpack_multi_layer_latents(
            latents, height, width, layer_num
        )

        # Decode to images
        images = self._decode_latents(unpacked_latents, layer_num=layer_num)

        logger.info(f"Generated {len(images)} layers")
        return images

    def to(self, device: torch.device) -> "QwenImagePipeline":
        """Move pipeline components to device."""
        self.text_encoder.to(device)
        self.dit.to(device)
        self.vae.to(device)
        return self
