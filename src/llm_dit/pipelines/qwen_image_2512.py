"""
Qwen-Image-2512 text-to-image pipeline wrapper.

This module wraps the official diffusers QwenImagePipeline for the
Qwen-Image-2512 model, providing text-to-image generation with
FP8 quantization support for RTX 4090 (24GB VRAM).

Capabilities:
- Text-to-image generation with high quality
- FP8 quantization for 39GB transformer (required for 24GB GPUs)
- CPU offload for memory management

Example:
    pipe = QwenImage2512Pipeline.from_pretrained(
        "/path/to/Qwen-Image-2512",
        quantize_transformer="fp8",  # Required for RTX 4090
        cpu_offload=True,
    )

    image = pipe(
        prompt="A beautiful sunset over mountains",
        height=1024,
        width=1024,
    )
    image.save("output.png")

last updated: 2025-12-31
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Add coderef diffusers to path for imports
_CODEREF_DIFFUSERS = Path(__file__).parent.parent.parent.parent / "coderef" / "diffusers" / "src"
if _CODEREF_DIFFUSERS.exists() and str(_CODEREF_DIFFUSERS) not in sys.path:
    sys.path.insert(0, str(_CODEREF_DIFFUSERS))
    logger.debug(f"Added coderef diffusers to path: {_CODEREF_DIFFUSERS}")

# Default parameters from Qwen-Image-2512 technical specs
DEFAULT_CFG_SCALE = 4.0
DEFAULT_STEPS = 40  # DiffSynth-Studio uses 40, diffusers default 50
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024


class QwenImage2512Pipeline:
    """
    Pipeline wrapper for Qwen-Image-2512 text-to-image generation.

    This is a pure text-to-image model (unlike Qwen-Image-Edit).
    Requires FP8 quantization for the 39GB transformer to fit in 24GB VRAM.

    Attributes:
        pipe: The underlying diffusers QwenImagePipeline
        device: Primary device for inference
        dtype: Model dtype (bfloat16 recommended)
    """

    def __init__(
        self,
        pipe,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the pipeline wrapper.

        Args:
            pipe: Loaded diffusers QwenImagePipeline
            device: Device for inference
            dtype: Model dtype
        """
        self.pipe = pipe
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = dtype
        self._cpu_offload_enabled = False

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        device: Union[str, torch.device] = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        cpu_offload: bool = True,
        quantize_transformer: str = "fp8",  # Default FP8 for RTX 4090
    ) -> "QwenImage2512Pipeline":
        """
        Load the pipeline from pretrained weights.

        Args:
            model_path: Path to Qwen-Image-2512 model (diffusers format)
            device: Device for inference
            torch_dtype: Model dtype (bfloat16 recommended)
            cpu_offload: Enable sequential CPU offload for memory efficiency
            quantize_transformer: Quantization for transformer (60-layer DiT):
                "fp8" = TorchAO FP8 dynamic (~20GB, default for RTX 4090)
                "int8" = TorchAO INT8 weight-only (~20GB)
                "4bit" = BitsAndBytes NF4 (~10GB, more quality loss)
                "8bit" = BitsAndBytes INT8 (~20GB)
                None = no quantization (~39GB, requires 48GB+ VRAM)

        Returns:
            Initialized QwenImage2512Pipeline

        Example:
            # RTX 4090 (24GB) - recommended settings
            pipe = QwenImage2512Pipeline.from_pretrained(
                "/path/to/Qwen-Image-2512",
                quantize_transformer="fp8",
                cpu_offload=True,
            )
        """
        model_path = Path(model_path).expanduser()
        device = torch.device(device)

        if not model_path.exists():
            raise ValueError(f"Model not found at {model_path}")

        logger.info(f"Loading QwenImage2512Pipeline from {model_path}")

        from diffusers import QwenImagePipeline

        # Build quantization config if needed
        pipe_quant_config = None
        if quantize_transformer:
            pipe_quant_config = cls._build_quantization_config(quantize_transformer)

        # Load pipeline with quantization
        if pipe_quant_config:
            logger.info(f"Loading with transformer quantization: {quantize_transformer}")
            pipe = QwenImagePipeline.from_pretrained(
                str(model_path),
                torch_dtype=torch_dtype,
                quantization_config=pipe_quant_config,
            )
        else:
            logger.warning(
                "Loading without quantization - requires 48GB+ VRAM. "
                "Use quantize_transformer='fp8' for RTX 4090."
            )
            pipe = QwenImagePipeline.from_pretrained(
                str(model_path),
                torch_dtype=torch_dtype,
            )

        # Enable CPU offload for memory management
        if cpu_offload:
            logger.info("Enabling model CPU offload")
            pipe.enable_model_cpu_offload()

        instance = cls(pipe=pipe, device=device, dtype=torch_dtype)
        instance._cpu_offload_enabled = cpu_offload

        logger.info(
            f"QwenImage2512Pipeline loaded: "
            f"quantize={quantize_transformer}, cpu_offload={cpu_offload}"
        )

        return instance

    @staticmethod
    def _build_quantization_config(quantize_transformer: str):
        """Build PipelineQuantizationConfig for transformer quantization."""
        from diffusers import TorchAoConfig
        from diffusers.quantizers import PipelineQuantizationConfig

        # TorchAO quantization (fp8, int8)
        if quantize_transformer in ("fp8", "int8"):
            from llm_dit.quantization import check_fp8_support

            if quantize_transformer == "fp8":
                if not check_fp8_support():
                    raise RuntimeError(
                        "FP8 requires compute capability 8.9+ (RTX 4090/H100). "
                        "Use 'int8' or '8bit' instead."
                    )
                quant_config = TorchAoConfig("float8dq")
                logger.info("Using TorchAO FP8 dynamic quantization for transformer")
            else:
                quant_config = TorchAoConfig("int8wo")
                logger.info("Using TorchAO INT8 weight-only quantization for transformer")

            # Only quantize transformer, not text_encoder or vae
            return PipelineQuantizationConfig(
                quant_mapping={"transformer": quant_config}
            )

        # BitsAndBytes quantization (4bit, 8bit)
        elif quantize_transformer in ("4bit", "8bit"):
            try:
                from diffusers import BitsAndBytesConfig
            except ImportError:
                raise ImportError(
                    "bitsandbytes required for 4bit/8bit. Install with: uv add bitsandbytes"
                )

            if quantize_transformer == "4bit":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                logger.info("Using BitsAndBytes 4-bit quantization for transformer")
            else:
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
                logger.info("Using BitsAndBytes 8-bit quantization for transformer")

            return PipelineQuantizationConfig(
                quant_mapping={"transformer": quant_config}
            )

        else:
            raise ValueError(
                f"Unknown quantization: {quantize_transformer}. "
                "Use 'fp8', 'int8' (TorchAO) or '4bit', '8bit' (BitsAndBytes)"
            )

    @property
    def device(self) -> torch.device:
        """Return primary device."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Return model dtype."""
        return self._dtype

    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = " ",
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        num_inference_steps: int = DEFAULT_STEPS,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        seed: Optional[int] = None,
        max_sequence_length: int = 512,
    ) -> Image.Image:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text description of the desired image
            negative_prompt: What to avoid in the image (default " ")
            height: Output image height (default 1024)
            width: Output image width (default 1024)
            num_inference_steps: Number of diffusion steps (default 40)
            cfg_scale: Classifier-free guidance scale (default 4.0)
            seed: Random seed for reproducibility
            max_sequence_length: Max tokens for prompt (default 512)

        Returns:
            Generated PIL Image

        Example:
            image = pipe(
                prompt="A serene mountain lake at sunset, photorealistic",
                negative_prompt="blurry, low quality",
                height=1024,
                width=1024,
                seed=42,
            )
            image.save("output.png")
        """
        # Setup generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        logger.info(
            f"Generating image: {width}x{height}, steps={num_inference_steps}, "
            f"cfg={cfg_scale}, seed={seed}"
        )

        # Run generation
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=cfg_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            generator=generator,
            max_sequence_length=max_sequence_length,
        )

        image = result.images[0]
        logger.info("Image generation complete")

        return image

    def generate(
        self,
        prompt: Union[str, List[str]],
        **kwargs,
    ) -> Image.Image:
        """Alias for __call__ for consistency with other pipelines."""
        return self(prompt, **kwargs)
