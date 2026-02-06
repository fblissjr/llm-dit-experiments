"""
Qwen-Image-2512 text-to-image pipeline wrapper.

This module wraps the official diffusers QwenImagePipeline for the
Qwen-Image-2512 model, providing text-to-image generation with
FP8 quantization support for RTX 4090 (24GB VRAM).

Capabilities:
- Text-to-image generation with high quality
- FP8 quantization for 39GB transformer (required for 24GB GPUs)
- CPU offload for memory management
- DiffSynth-style FP8 (runtime F.linear patching)

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

last updated: 2026-01-03
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
        dtype: torch.dtype = torch.bfloat16,
        cpu_offload: bool = True,
        quantize_transformer: str = "fp8",  # Default FP8 for RTX 4090
        quantize_text_encoder: Optional[str] = None,  # None = CPU offload (best quality)
    ) -> "QwenImage2512Pipeline":
        """
        Load the pipeline from pretrained weights.

        Args:
            model_path: Path to Qwen-Image-2512 model (diffusers format)
            device: Device for inference
            dtype: Model dtype (bfloat16 recommended)
            cpu_offload: Enable sequential CPU offload for memory efficiency
            quantize_transformer: Quantization for transformer (60-layer DiT, 39GB):
                "fp8-dynamic" = FP8 weights + activations (~20GB, RTX 4090+)
                "fp8-weight-only" = FP8 weights, BF16 activations (~20GB)
                "int8" = INT8 weight-only (~20GB)
                "int4" = INT4 weight-only (~10GB)
                None = no quantization (~39GB, requires 48GB+ VRAM)
            quantize_text_encoder: Quantization for text encoder (Qwen2.5-VL-7B, 16GB):
                None = no quantization (~16GB, best quality, CPU offload recommended)
                "int8" = INT8 weight-only (~8GB)
                "int4" = INT4 weight-only (~4GB)

        Returns:
            Initialized QwenImage2512Pipeline

        Example:
            # RTX 4090 (24GB) - recommended settings (best quality)
            pipe = QwenImage2512Pipeline.from_pretrained(
                "/path/to/Qwen-Image-2512",
                quantize_transformer="fp8",         # ~20GB on GPU
                quantize_text_encoder=None,         # CPU offload (best quality)
                cpu_offload=True,
            )
            # With CPU offload: only one component on GPU at a time (~20GB peak)
        """
        model_path = Path(model_path).expanduser()
        device = torch.device(device)

        if not model_path.exists():
            raise ValueError(f"Model not found at {model_path}")

        logger.info(f"Loading QwenImage2512Pipeline from {model_path}")

        from diffusers import QwenImagePipeline

        # Load pipeline in full precision
        logger.info(
            f"Loading with quantization: transformer={quantize_transformer}, "
            f"text_encoder={quantize_text_encoder}"
        )
        pipe = QwenImagePipeline.from_pretrained(
            str(model_path),
            dtype=dtype,
        )

        # Apply post-load quantization via unified system
        if quantize_transformer and quantize_transformer != "none":
            from llm_dit.quantization import quantize_component

            pipe.transformer, stats = quantize_component(
                pipe.transformer, method=quantize_transformer, component_type="transformer"
            )
            logger.info(
                f"Transformer quantized: {stats['quantized_layers']}/{stats['total_layers']} layers "
                f"({quantize_transformer})"
            )

        if quantize_text_encoder and quantize_text_encoder != "none":
            from llm_dit.quantization import quantize_component

            pipe.text_encoder, stats = quantize_component(
                pipe.text_encoder, method=quantize_text_encoder, component_type="encoder"
            )
            logger.info(
                f"Text encoder quantized: {stats['quantized_layers']}/{stats['total_layers']} layers "
                f"({quantize_text_encoder})"
            )

        # Enable CPU offload for memory management
        if cpu_offload:
            logger.info("Enabling model CPU offload")
            pipe.enable_model_cpu_offload()

        instance = cls(
            pipe=pipe,
            device=device,
            dtype=dtype,
        )
        instance._cpu_offload_enabled = cpu_offload

        logger.info(
            f"QwenImage2512Pipeline loaded: "
            f"quantize_transformer={quantize_transformer}, "
            f"quantize_text_encoder={quantize_text_encoder}, "
            f"cpu_offload={cpu_offload}"
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
        # Forward Block Cache (FBCache) for inference acceleration
        fbcache: bool = False,
        fbcache_threshold: Optional[float] = None,
        fbcache_log: bool = False,
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

        # Initialize FBCache for inference acceleration
        fbcache_ctx = None
        fbcache_state = None
        fbcache_callback = None

        if fbcache:
            # FBCache is not yet supported for Qwen-Image due to different transformer block signatures
            # Qwen-Image uses keyword arguments (hidden_states=...) while FBCache wrapper expects
            # positional arguments. This requires a separate implementation.
            logger.warning(
                "FBCache is not yet supported for Qwen-Image-2512. "
                "The transformer block signatures differ from Z-Image. "
                "Proceeding without FBCache acceleration."
            )
            fbcache = False

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=cfg_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            generator=generator,
            max_sequence_length=max_sequence_length,
            callback_on_step_end=fbcache_callback,
        )

        # Clean up FBCache
        if fbcache_ctx is not None:
            fbcache_ctx.__exit__(None, None, None)
            if fbcache_state is not None:
                stats = fbcache_state.get_stats()
                logger.info(
                    f"FBCache stats: "
                    f"{stats['skips']} skips, {stats['computes']} computes, "
                    f"ratio={stats['skip_ratio']:.1%}, est. speedup={stats['estimated_speedup']:.2f}x"
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
