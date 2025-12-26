"""
Qwen-Image pipeline wrapper using official diffusers.

This module wraps the official diffusers QwenImageLayeredPipeline and
QwenImageEditPlusPipeline, providing a clean API consistent with our
project structure while leveraging the battle-tested diffusers implementation.

Capabilities:
- decompose(): Image-to-RGBA-layers decomposition
- edit_layer(): Edit individual RGBA layers with text instructions

Example:
    pipe = QwenImageDiffusersPipeline.from_pretrained(
        "/path/to/Qwen_Qwen-Image-Layered"
    )

    # Decompose an image into layers
    layers = pipe.decompose(
        image=input_image,
        prompt="A cheerful scene",
        layer_num=4,
    )

    # Edit a specific layer
    edited = pipe.edit_layer(
        layer_image=layers[1],
        instruction="Change the color to blue",
    )
"""

import logging
import sys
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Add coderef diffusers to path for imports
_CODEREF_DIFFUSERS = Path(__file__).parent.parent.parent.parent / "coderef" / "diffusers" / "src"
if _CODEREF_DIFFUSERS.exists() and str(_CODEREF_DIFFUSERS) not in sys.path:
    sys.path.insert(0, str(_CODEREF_DIFFUSERS))
    logger.debug(f"Added coderef diffusers to path: {_CODEREF_DIFFUSERS}")

# Supported resolutions (fixed buckets from training)
SUPPORTED_RESOLUTIONS = (640, 1024)

# Default parameters from technical report
DEFAULT_CFG_SCALE = 4.0
DEFAULT_STEPS = 40  # Updated for Qwen-Image-Edit-2511 (was 50 for 2509)
DEFAULT_LAYER_NUM = 4
DEFAULT_RESOLUTION = 640


class QwenImageDiffusersPipeline:
    """
    Pipeline wrapper for Qwen-Image using official diffusers.

    Wraps QwenImageLayeredPipeline for decomposition and optionally
    QwenImageEditPlusPipeline for layer editing.

    Attributes:
        decompose_pipe: The diffusers QwenImageLayeredPipeline
        edit_pipe: Optional QwenImageEditPlusPipeline (lazy loaded)
        device: Primary device for inference
        dtype: Model dtype (bfloat16 recommended)
    """

    def __init__(
        self,
        decompose_pipe,
        edit_pipe=None,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the pipeline wrapper.

        Args:
            decompose_pipe: Loaded QwenImageLayeredPipeline
            edit_pipe: Optional loaded QwenImageEditPlusPipeline
            device: Device for inference
            dtype: Model dtype
        """
        self.decompose_pipe = decompose_pipe
        self.edit_pipe = edit_pipe
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = dtype
        self._cpu_offload_enabled = False
        self._edit_model_path = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        edit_model_path: Optional[Union[str, Path]] = None,
        device: Union[str, torch.device] = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        cpu_offload: bool = True,
        load_edit_model: bool = False,
    ) -> "QwenImageDiffusersPipeline":
        """
        Load the pipeline from pretrained weights.

        Args:
            model_path: Path to Qwen-Image-Layered model
            edit_model_path: Optional path to Qwen-Image-Edit model
                (defaults to "Qwen/Qwen-Image-Edit-2511" from HuggingFace)
            device: Device for inference
            torch_dtype: Model dtype (bfloat16 recommended)
            cpu_offload: Enable sequential CPU offload for memory efficiency
            load_edit_model: If True, also load the edit model

        Returns:
            Initialized QwenImageDiffusersPipeline

        Example:
            # Basic loading with CPU offload (recommended for RTX 4090)
            pipe = QwenImageDiffusersPipeline.from_pretrained(
                "/path/to/Qwen_Qwen-Image-Layered",
                cpu_offload=True,
            )

            # Also load edit model at startup
            pipe = QwenImageDiffusersPipeline.from_pretrained(
                "/path/to/Qwen_Qwen-Image-Layered",
                load_edit_model=True,
            )
        """
        model_path = Path(model_path)
        device = torch.device(device)

        if not model_path.exists():
            raise ValueError(f"Model not found at {model_path}")

        # Import from diffusers (using coderef)
        from diffusers import QwenImageLayeredPipeline

        logger.info(f"Loading QwenImageLayeredPipeline from {model_path}")
        decompose_pipe = QwenImageLayeredPipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch_dtype,  # diffusers uses torch_dtype
        )

        # Enable CPU offload for memory efficiency
        if cpu_offload:
            logger.info("Enabling sequential CPU offload")
            decompose_pipe.enable_sequential_cpu_offload()
            cpu_offload_enabled = True
        else:
            decompose_pipe.to(device)
            cpu_offload_enabled = False

        # Resolve edit model path (expand ~ and convert to string)
        resolved_edit_path = None
        if edit_model_path:
            resolved_edit_path = str(Path(edit_model_path).expanduser())
        else:
            resolved_edit_path = "Qwen/Qwen-Image-Edit-2511"

        # Optionally load edit model
        edit_pipe = None
        if load_edit_model:
            logger.info(f"Loading QwenImageEditPlusPipeline from {resolved_edit_path}")
            from diffusers import QwenImageEditPlusPipeline
            edit_pipe = QwenImageEditPlusPipeline.from_pretrained(
                resolved_edit_path,
                torch_dtype=torch_dtype,  # diffusers uses torch_dtype
            )
            if cpu_offload:
                edit_pipe.enable_sequential_cpu_offload()
            else:
                edit_pipe.to(device)

        instance = cls(
            decompose_pipe=decompose_pipe,
            edit_pipe=edit_pipe,
            device=device,
            dtype=torch_dtype,
        )
        instance._cpu_offload_enabled = cpu_offload_enabled
        instance._edit_model_path = resolved_edit_path

        logger.info(
            f"QwenImageDiffusersPipeline loaded: "
            f"decompose=True, edit={edit_pipe is not None}, "
            f"cpu_offload={cpu_offload_enabled}"
        )

        return instance

    @property
    def device(self) -> torch.device:
        """Return primary device."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Return model dtype."""
        return self._dtype

    @property
    def has_edit_model(self) -> bool:
        """Check if edit model is loaded."""
        return self.edit_pipe is not None

    def decompose(
        self,
        image: Image.Image,
        prompt: str = "",
        layer_num: int = DEFAULT_LAYER_NUM,
        resolution: int = DEFAULT_RESOLUTION,
        num_inference_steps: int = DEFAULT_STEPS,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        seed: Optional[int] = None,
        negative_prompt: str = " ",
        use_en_prompt: bool = True,
        cfg_normalize: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Image.Image]:
        """
        Decompose an image into RGBA layers.

        Args:
            image: Input image (will be converted to RGBA)
            prompt: Optional text description of the image content.
                If empty and use_en_prompt=True, auto-captioning is used.
            layer_num: Number of decomposition layers (2-10, default 4)
            resolution: Output resolution (640 or 1024, default 640)
            num_inference_steps: Diffusion steps (default 50)
            cfg_scale: Classifier-free guidance scale (default 4.0)
            seed: Random seed for reproducibility
            negative_prompt: Negative prompt (default " ")
            use_en_prompt: Use English auto-captioning if no prompt (default True)
            cfg_normalize: Enable CFG normalization (default True)
            progress_callback: Optional callback(step, total_steps)

        Returns:
            List of RGBA PIL Images (layer_num + 1 images: composite + layers)

        Example:
            layers = pipe.decompose(
                image=Image.open("scene.png"),
                prompt="A house with a garden",
                layer_num=4,
                resolution=640,
            )
            for i, layer in enumerate(layers):
                layer.save(f"layer_{i}.png")
        """
        # Validate resolution
        if resolution not in SUPPORTED_RESOLUTIONS:
            raise ValueError(
                f"Resolution must be one of {SUPPORTED_RESOLUTIONS}, got {resolution}"
            )

        # Validate layer_num
        if not 1 <= layer_num <= 10:
            raise ValueError(f"layer_num must be 1-10, got {layer_num}")

        # Convert to RGBA
        if image.mode != "RGBA":
            image = image.convert("RGB").convert("RGBA")

        # Setup generator for seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        logger.info(
            f"Decomposing image: resolution={resolution}, layers={layer_num}, "
            f"steps={num_inference_steps}, cfg={cfg_scale}"
        )

        # Run decomposition
        result = self.decompose_pipe(
            image=image,
            prompt=prompt if prompt else None,
            negative_prompt=negative_prompt,
            layers=layer_num,
            resolution=resolution,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=cfg_scale,
            cfg_normalize=cfg_normalize,
            use_en_prompt=use_en_prompt,
            generator=generator,
        )

        # Extract layers from result (handle nested list structure)
        layers = result.images[0] if isinstance(result.images[0], list) else result.images

        logger.info(f"Decomposition complete: {len(layers)} layers generated")

        return layers

    def load_edit_model(self, model_path: Optional[Union[str, Path]] = None) -> None:
        """
        Load the edit model (lazy loading).

        Args:
            model_path: Path to edit model (defaults to HuggingFace model)
        """
        if self.edit_pipe is not None:
            logger.info("Edit model already loaded")
            return

        # Resolve path (expand ~ if present)
        raw_path = model_path or self._edit_model_path or "Qwen/Qwen-Image-Edit-2511"
        if raw_path and not raw_path.startswith("Qwen/"):
            edit_path = str(Path(raw_path).expanduser())
        else:
            edit_path = raw_path

        logger.info(f"Loading QwenImageEditPlusPipeline from {edit_path}")
        from diffusers import QwenImageEditPlusPipeline

        self.edit_pipe = QwenImageEditPlusPipeline.from_pretrained(
            edit_path,
            torch_dtype=self._dtype,
        )

        if self._cpu_offload_enabled:
            self.edit_pipe.enable_sequential_cpu_offload()
        else:
            self.edit_pipe.to(self._device)

        logger.info("Edit model loaded successfully")

    def edit_layer(
        self,
        layer_image: Image.Image,
        instruction: str,
        num_inference_steps: int = DEFAULT_STEPS,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Edit a layer using text instructions.

        Args:
            layer_image: RGBA layer image to edit
            instruction: Text instruction for editing (e.g., "Change color to blue")
            num_inference_steps: Diffusion steps (default 50)
            cfg_scale: Classifier-free guidance scale (default 4.0)
            seed: Random seed for reproducibility

        Returns:
            Edited RGBA image

        Example:
            edited = pipe.edit_layer(
                layer_image=layers[1],
                instruction="Make the house red",
            )
            edited.save("edited_layer.png")
        """
        # Lazy load edit model if needed
        if self.edit_pipe is None:
            self.load_edit_model()

        # Handle RGBA -> RGB conversion for edit pipeline (VAE expects 3 channels)
        # Store alpha channel to reapply after editing
        alpha_channel = None
        if layer_image.mode == "RGBA":
            # Split channels and store alpha
            r, g, b, a = layer_image.split()
            alpha_channel = a
            rgb_image = Image.merge("RGB", (r, g, b))
        elif layer_image.mode == "RGB":
            rgb_image = layer_image
        else:
            rgb_image = layer_image.convert("RGB")

        # Setup generator
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        logger.info(f"Editing layer: instruction='{instruction}', steps={num_inference_steps}")

        # Run edit on RGB image
        result = self.edit_pipe(
            image=rgb_image,
            prompt=instruction,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=cfg_scale,
            generator=generator,
        )

        edited_rgb = result.images[0]

        # Reapply alpha channel if original was RGBA
        if alpha_channel is not None:
            # Resize alpha if needed (in case edit changed resolution)
            if alpha_channel.size != edited_rgb.size:
                alpha_channel = alpha_channel.resize(edited_rgb.size, Image.LANCZOS)
            r, g, b = edited_rgb.split()
            edited = Image.merge("RGBA", (r, g, b, alpha_channel))
        else:
            edited = edited_rgb

        logger.info("Layer edit complete")

        return edited

    def edit_multi(
        self,
        images: List[Image.Image],
        instruction: str,
        num_inference_steps: int = DEFAULT_STEPS,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Combine multiple images based on text instructions.

        New capability in Qwen-Image-Edit-2511 for multi-person consistency
        and creative image merging. Supports combining 2+ images into a
        single coherent output.

        Args:
            images: List of 2+ PIL images to combine
            instruction: Text describing how to combine them
                (e.g., "Place both subjects side by side in a park")
            num_inference_steps: Diffusion steps (default 40)
            cfg_scale: Classifier-free guidance scale (default 4.0)
            seed: Random seed for reproducibility

        Returns:
            Combined output image

        Example:
            combined = pipe.edit_multi(
                images=[Image.open("person1.jpg"), Image.open("person2.jpg")],
                instruction="The two people standing together at a beach",
                seed=42,
            )
            combined.save("combined.png")
        """
        # Validate input
        if len(images) < 2:
            raise ValueError(
                f"edit_multi requires at least 2 images, got {len(images)}. "
                "For single-image editing, use edit_layer() instead."
            )

        # Lazy load edit model if needed
        if self.edit_pipe is None:
            self.load_edit_model()

        # Convert all images to RGB (edit pipeline requires RGB input)
        rgb_images = []
        for img in images:
            if img.mode == "RGBA":
                # Convert RGBA to RGB (composite onto white background)
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                rgb_images.append(background)
            elif img.mode == "RGB":
                rgb_images.append(img)
            else:
                rgb_images.append(img.convert("RGB"))

        # Setup generator
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        logger.info(
            f"Multi-image edit: {len(rgb_images)} images, "
            f"instruction='{instruction[:80]}...', steps={num_inference_steps}"
        )

        # Run multi-image edit
        # QwenImageEditPlusPipeline accepts image as a list for multi-image mode
        result = self.edit_pipe(
            image=rgb_images,
            prompt=instruction,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=cfg_scale,
            generator=generator,
        )

        output_image = result.images[0]

        logger.info("Multi-image edit complete")

        return output_image

    def enable_cpu_offload(self) -> None:
        """Enable sequential CPU offload for memory efficiency."""
        if not self._cpu_offload_enabled:
            self.decompose_pipe.enable_sequential_cpu_offload()
            if self.edit_pipe is not None:
                self.edit_pipe.enable_sequential_cpu_offload()
            self._cpu_offload_enabled = True
            logger.info("CPU offload enabled")

    def disable_cpu_offload(self) -> None:
        """Disable CPU offload and move to GPU."""
        if self._cpu_offload_enabled:
            # Note: diffusers pipelines need to be recreated to fully disable offload
            # For now, just log a warning
            logger.warning(
                "Disabling CPU offload requires reloading the pipeline. "
                "Call from_pretrained with cpu_offload=False instead."
            )

    def to(self, device: Union[str, torch.device]) -> "QwenImageDiffusersPipeline":
        """
        Move pipeline to device.

        Note: If CPU offload is enabled, this is a no-op since
        accelerate manages device placement.
        """
        if self._cpu_offload_enabled:
            logger.warning("Cannot move to device when CPU offload is enabled")
            return self

        device = torch.device(device)
        self.decompose_pipe.to(device)
        if self.edit_pipe is not None:
            self.edit_pipe.to(device)
        self._device = device
        return self
