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
        use_diffsynth_fp8: bool = False,
    ):
        """
        Initialize the pipeline wrapper.

        Args:
            pipe: Loaded diffusers QwenImagePipeline
            device: Device for inference
            dtype: Model dtype
            use_diffsynth_fp8: Use DiffSynth-style FP8 inference
        """
        self.pipe = pipe
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = dtype
        self._cpu_offload_enabled = False
        self._use_diffsynth_fp8 = use_diffsynth_fp8

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
                "fp8" = TorchAO FP8 dynamic (~20GB, default for RTX 4090)
                "diffsynth-fp8" = DiffSynth-style FP8 (runtime F.linear patching)
                "int8" = TorchAO INT8 weight-only (~20GB)
                "4bit" = BitsAndBytes NF4 (~10GB, more quality loss)
                "8bit" = BitsAndBytes INT8 (~20GB)
                None = no quantization (~39GB, requires 48GB+ VRAM)
            quantize_text_encoder: Quantization for text encoder (Qwen2.5-VL-7B, 16GB):
                None = no quantization (~16GB, best quality, CPU offload recommended)
                "8bit" = BitsAndBytes INT8 (~8GB)
                "4bit" = BitsAndBytes NF4 (~4GB, significant quality loss)

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

        # Check for DiffSynth-style FP8 (runtime patching, not TorchAO)
        use_diffsynth_fp8 = quantize_transformer == "diffsynth-fp8"
        if use_diffsynth_fp8:
            # DiffSynth FP8 uses runtime F.linear patching - don't pass to TorchAO
            effective_transformer_quant = None
            logger.info("Using DiffSynth-style FP8 (runtime F.linear patching)")
        else:
            effective_transformer_quant = quantize_transformer

        # Build quantization config if needed
        pipe_quant_config = cls._build_quantization_config(
            quantize_transformer=effective_transformer_quant,
            quantize_text_encoder=quantize_text_encoder,
        )

        # Load pipeline with quantization
        if pipe_quant_config:
            logger.info(
                f"Loading with quantization: transformer={effective_transformer_quant}, "
                f"text_encoder={quantize_text_encoder}"
            )
            pipe = QwenImagePipeline.from_pretrained(
                str(model_path),
                dtype=dtype,
                quantization_config=pipe_quant_config,
            )
        else:
            if not use_diffsynth_fp8:
                logger.warning(
                    "Loading without quantization - requires 48GB+ VRAM. "
                    "Use quantize_transformer='fp8' and quantize_text_encoder='4bit' for RTX 4090."
                )
            pipe = QwenImagePipeline.from_pretrained(
                str(model_path),
                dtype=dtype,
            )

        # Enable CPU offload for memory management
        if cpu_offload:
            logger.info("Enabling model CPU offload")
            pipe.enable_model_cpu_offload()

        # For DiffSynth FP8, pre-convert weights for memory savings
        if use_diffsynth_fp8:
            from llm_dit.quantization import enable_fp8_weights

            logger.info("Converting transformer weights to FP8 for memory savings...")
            enable_fp8_weights(pipe.transformer)

        instance = cls(
            pipe=pipe,
            device=device,
            dtype=dtype,
            use_diffsynth_fp8=use_diffsynth_fp8,
        )
        instance._cpu_offload_enabled = cpu_offload

        logger.info(
            f"QwenImage2512Pipeline loaded: "
            f"quantize_transformer={quantize_transformer}, "
            f"quantize_text_encoder={quantize_text_encoder}, "
            f"cpu_offload={cpu_offload}, "
            f"diffsynth_fp8={use_diffsynth_fp8}"
        )

        return instance

    @staticmethod
    def _build_quantization_config(
        quantize_transformer: Optional[str],
        quantize_text_encoder: Optional[str] = None,
    ):
        """Build PipelineQuantizationConfig for component quantization."""
        from diffusers.quantizers import PipelineQuantizationConfig

        if not quantize_transformer and not quantize_text_encoder:
            return None

        quant_mapping = {}

        # Transformer quantization
        if quantize_transformer:
            if quantize_transformer in ("fp8", "int8"):
                from diffusers import TorchAoConfig

                from llm_dit.quantization import check_fp8_support

                if quantize_transformer == "fp8":
                    if not check_fp8_support():
                        raise RuntimeError(
                            "FP8 requires compute capability 8.9+ (RTX 4090/H100). "
                            "Use 'int8' or '8bit' instead."
                        )
                    quant_mapping["transformer"] = TorchAoConfig("float8dq")
                    logger.info("Transformer: TorchAO FP8 dynamic quantization")
                else:
                    quant_mapping["transformer"] = TorchAoConfig("int8wo")
                    logger.info("Transformer: TorchAO INT8 weight-only quantization")

            elif quantize_transformer in ("4bit", "8bit"):
                try:
                    from diffusers import BitsAndBytesConfig
                except ImportError:
                    raise ImportError(
                        "bitsandbytes required for 4bit/8bit. Install with: uv add bitsandbytes"
                    )

                if quantize_transformer == "4bit":
                    quant_mapping["transformer"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                    logger.info("Transformer: BitsAndBytes 4-bit quantization")
                else:
                    quant_mapping["transformer"] = BitsAndBytesConfig(load_in_8bit=True)
                    logger.info("Transformer: BitsAndBytes 8-bit quantization")
            else:
                raise ValueError(
                    f"Unknown transformer quantization: {quantize_transformer}. "
                    "Use 'fp8', 'int8' (TorchAO) or '4bit', '8bit' (BitsAndBytes)"
                )

        # Text encoder quantization
        if quantize_text_encoder:
            if quantize_text_encoder in ("4bit", "8bit"):
                try:
                    from transformers import BitsAndBytesConfig as TransformersBnbConfig
                except ImportError:
                    raise ImportError(
                        "bitsandbytes required for 4bit/8bit. Install with: uv add bitsandbytes"
                    )

                if quantize_text_encoder == "4bit":
                    quant_mapping["text_encoder"] = TransformersBnbConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                    logger.info("Text encoder: BitsAndBytes 4-bit quantization (~4GB)")
                else:
                    quant_mapping["text_encoder"] = TransformersBnbConfig(load_in_8bit=True)
                    logger.info("Text encoder: BitsAndBytes 8-bit quantization (~8GB)")
            else:
                raise ValueError(
                    f"Unknown text encoder quantization: {quantize_text_encoder}. "
                    "Use '4bit' or '8bit' (BitsAndBytes)"
                )

        if not quant_mapping:
            return None

        return PipelineQuantizationConfig(quant_mapping=quant_mapping)

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

        # Import FP8 context manager if needed
        if self._use_diffsynth_fp8:
            from llm_dit.quantization import fp8_inference

            context_manager = fp8_inference()
            logger.debug("Using DiffSynth-style FP8 inference")
        else:
            from contextlib import nullcontext

            context_manager = nullcontext()

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
            # Disable fbcache for this generation
            fbcache = False

        # Run generation (optionally with FP8 context)
        with context_manager:
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
