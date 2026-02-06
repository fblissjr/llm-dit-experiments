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
        use_diffsynth_fp8: bool = False,  # Deprecated: ignored. Use quantize_transformer instead.
    ):
        """
        Initialize the pipeline wrapper.

        Args:
            decompose_pipe: Loaded QwenImageLayeredPipeline
            edit_pipe: Optional loaded QwenImageEditPlusPipeline
            device: Device for inference
            dtype: Model dtype
            use_diffsynth_fp8: Deprecated, ignored. Use quantize_transformer instead.
        """
        self.decompose_pipe = decompose_pipe
        self.edit_pipe = edit_pipe
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = dtype
        self._cpu_offload_enabled = False
        self._offload_type = "none"
        self._edit_model_path = None
        # DiffSynth FP8 removed: quantization handled by quantize_component()

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        edit_model_path: Optional[Union[str, Path]] = None,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cpu_offload: bool = True,
        offload_type: Optional[str] = None,
        num_blocks_per_group: int = 2,
        load_edit_model: bool = False,
        edit_only: bool = False,
        quantize_text_encoder: Optional[str] = None,
        quantize_transformer: Optional[str] = None,
        quantize_vae: Optional[str] = None,
        compile_transformer: bool = False,
        compile_mode: str = "default",
    ) -> "QwenImageDiffusersPipeline":
        """
        Load the pipeline from pretrained weights.

        Args:
            model_path: Path to Qwen-Image-Layered model
            edit_model_path: Optional path to Qwen-Image-Edit model
                (defaults to "Qwen/Qwen-Image-Edit-2511" from HuggingFace)
            device: Device for inference
            dtype: Model dtype (bfloat16 recommended)
            cpu_offload: Enable CPU offload for memory efficiency (deprecated, use offload_type)
            offload_type: Offloading strategy for VRAM management:
                None = use cpu_offload value for backward compatibility
                "none" = no offloading, keep all on GPU
                "model" = model-level offload (whole components, default)
                "group" = group offload (DiT blocks, best speed/memory trade-off)
                "sequential" = leaf-level offload (minimum VRAM, slowest)
            num_blocks_per_group: DiT blocks per group for "group" offload_type (default: 2)
                Higher = more VRAM, faster. Lower = less VRAM, slower.
            load_edit_model: If True, also load the edit model
            edit_only: If True, skip loading decompose model (for edit-only workflows)
                This saves ~12GB VRAM on 24GB cards.
            quantize_text_encoder: Quantization for text encoder (Qwen2.5-VL-7B):
                None = no quantization (~14GB)
                "int4" = INT4 weight-only (~4GB, 75% reduction)
                "fp8-dynamic" = FP8 weights + activations (~7GB, RTX 4090+)
                "fp8-weight-only" = FP8 weights, BF16 activations (~7GB)
                "int8" = INT8 weight-only (~7GB)
                "int4" = INT4 weight-only (~4GB)
            quantize_transformer: Quantization for transformer (DiT):
                None = no quantization (~8GB)
                "fp8-dynamic" = FP8 weights + activations (~4GB, RTX 4090+)
                "fp8-weight-only" = FP8 weights, BF16 activations (~4GB)
                "int8" = INT8 weight-only (~4GB)
                "int4" = INT4 weight-only (~2GB)
            quantize_vae: Quantization for VAE decoder:
                None = no quantization (~500MB)
                "int8" = TorchAO INT8 dynamic (~250MB, 50% reduction)
                Note: Only int8 recommended for VAE (Conv2d layers)
            compile_transformer: If True, compile DiT with torch.compile for ~1.5-2x speedup.
                First inference will be slower due to compilation.
                NOTE: torch.compile is INCOMPATIBLE with cpu_offload=True. If both are
                specified, compilation will be skipped with a warning.
            compile_mode: torch.compile mode (only used when cpu_offload=False):
                "default" - Minimal optimization, fast compile
                "reduce-overhead" - CUDA graphs, lower latency
                "max-autotune" - CUDA graphs + GEMM autotuning, best performance (slow first compile)
                "max-autotune-no-cudagraphs" - GEMM autotuning only (slow first compile)

        Returns:
            Initialized QwenImageDiffusersPipeline

        Example:
            # Basic loading with CPU offload (recommended for RTX 4090)
            pipe = QwenImageDiffusersPipeline.from_pretrained(
                "/path/to/Qwen_Qwen-Image-Layered",
                cpu_offload=True,
            )

            # Edit-only mode (saves ~12GB VRAM)
            pipe = QwenImageDiffusersPipeline.from_pretrained(
                "/path/to/Qwen_Qwen-Image-Layered",
                edit_only=True,
                edit_model_path="/path/to/Qwen-Image-Edit-2511",
            )

            # Quantized mode for RTX 4090 (edit_only + 4bit text encoder)
            # ~12GB total VRAM: 3.5GB text encoder + 8GB DiT + 0.5GB VAE
            pipe = QwenImageDiffusersPipeline.from_pretrained(
                "/path/to/Qwen_Qwen-Image-Layered",
                edit_only=True,
                quantize_text_encoder="4bit",
                cpu_offload=True,
            )
        """
        device = torch.device(device)

        # Resolve offload_type from new parameter or legacy cpu_offload
        if offload_type is not None:
            effective_offload_type = offload_type
        elif cpu_offload:
            effective_offload_type = "model"  # Legacy default
        else:
            effective_offload_type = "none"

        # Validate offload_type
        valid_offload_types = ("none", "model", "group", "sequential")
        if effective_offload_type not in valid_offload_types:
            raise ValueError(
                f"Invalid offload_type: {effective_offload_type}. "
                f"Valid options: {valid_offload_types}"
            )

        # Resolve effective transformer quantization
        effective_transformer_quant = quantize_transformer

        # Resolve edit model path early (needed for edit_only mode)
        resolved_edit_path = None
        if edit_model_path:
            resolved_edit_path = str(Path(edit_model_path).expanduser())
        else:
            resolved_edit_path = "Qwen/Qwen-Image-Edit-2511"

        # Edit-only mode: skip decompose model entirely
        if edit_only:
            logger.info(f"Edit-only mode: skipping decompose model, loading edit model directly")
            from diffusers import QwenImageEditPlusPipeline

            logger.info(f"Loading QwenImageEditPlusPipeline from {resolved_edit_path}")

            # Check if quantization is requested
            if quantize_text_encoder or effective_transformer_quant or quantize_vae:
                # Use quantized loading - load components separately then assemble pipeline
                edit_pipe = cls._load_edit_pipeline_quantized(
                    resolved_edit_path,
                    dtype=dtype,
                    quantize_text_encoder=quantize_text_encoder,
                    quantize_transformer=effective_transformer_quant,
                    quantize_vae=quantize_vae,
                    cpu_offload=cpu_offload,
                    offload_type=effective_offload_type,
                    num_blocks_per_group=num_blocks_per_group,
                    compile_transformer=compile_transformer,
                    compile_mode=compile_mode,
                )
            elif effective_offload_type != "none":
                # Load pipeline first, then apply offloading
                logger.info(f"Loading with offload_type={effective_offload_type}")
                edit_pipe = QwenImageEditPlusPipeline.from_pretrained(
                    resolved_edit_path,
                    dtype=dtype,
                )
                # NOTE: torch.compile is INCOMPATIBLE with CPU offload
                if compile_transformer and effective_offload_type != "none":
                    logger.warning(
                        "torch.compile is incompatible with CPU offload (accelerate's tensor swapping "
                        "fails with compiled models). Skipping compilation."
                    )
                # Apply offloading based on type
                cls._apply_offloading(
                    edit_pipe, effective_offload_type, device, num_blocks_per_group
                )
            else:
                edit_pipe = QwenImageEditPlusPipeline.from_pretrained(
                    resolved_edit_path,
                    dtype=dtype,
                )
                if compile_transformer:
                    logger.info(
                        f"Compiling transformer with torch.compile (mode={compile_mode})..."
                    )
                    edit_pipe.transformer = torch.compile(
                        edit_pipe.transformer,
                        mode=compile_mode,
                        fullgraph=True,
                    )
                edit_pipe.to(device)

            instance = cls(
                decompose_pipe=None,  # No decompose in edit-only mode
                edit_pipe=edit_pipe,
                device=device,
                dtype=dtype,
            )
            instance._cpu_offload_enabled = effective_offload_type != "none"
            instance._offload_type = effective_offload_type
            instance._edit_model_path = resolved_edit_path

            logger.info(
                f"QwenImageDiffusersPipeline loaded (edit-only): "
                f"decompose=False, edit=True, offload_type={effective_offload_type}"
            )
            return instance

        # Normal mode: load decompose model
        # Convert model_path to Path now (not needed for edit-only mode above)
        if model_path is None:
            raise ValueError("model_path is required for decompose mode")
        model_path = Path(model_path)
        if not model_path.exists():
            raise ValueError(f"Model not found at {model_path}")

        from diffusers import QwenImageLayeredPipeline

        logger.info(f"Loading QwenImageLayeredPipeline from {model_path}")

        # Check if quantization is requested for decompose pipeline (use effective_transformer_quant)
        if quantize_text_encoder or effective_transformer_quant or quantize_vae:
            decompose_pipe = cls._load_decompose_pipeline_quantized(
                str(model_path),
                dtype=dtype,
                quantize_text_encoder=quantize_text_encoder,
                quantize_transformer=effective_transformer_quant,
                quantize_vae=quantize_vae,
                cpu_offload=cpu_offload,
                offload_type=effective_offload_type,
                num_blocks_per_group=num_blocks_per_group,
                compile_transformer=compile_transformer,
                compile_mode=compile_mode,
            )
            cpu_offload_enabled = effective_offload_type != "none"
        elif effective_offload_type != "none":
            # Load pipeline then apply offloading
            decompose_pipe = QwenImageLayeredPipeline.from_pretrained(
                str(model_path),
                dtype=dtype,
            )
            # NOTE: torch.compile is INCOMPATIBLE with CPU offload
            if compile_transformer:
                logger.warning(
                    "torch.compile is incompatible with CPU offload (accelerate's tensor swapping "
                    "fails with compiled models). Skipping compilation."
                )
            # Apply offloading
            cls._apply_offloading(
                decompose_pipe, effective_offload_type, device, num_blocks_per_group
            )
            cpu_offload_enabled = True
        else:
            decompose_pipe = QwenImageLayeredPipeline.from_pretrained(
                str(model_path),
                dtype=dtype,
            )
            if compile_transformer:
                logger.info(f"Compiling transformer with torch.compile (mode={compile_mode})...")
                decompose_pipe.transformer = torch.compile(
                    decompose_pipe.transformer,
                    mode=compile_mode,
                    fullgraph=True,
                )
            decompose_pipe.to(device)
            cpu_offload_enabled = False

        # Optionally load edit model (resolved_edit_path already set above)
        edit_pipe = None
        if load_edit_model:
            logger.info(f"Loading QwenImageEditPlusPipeline from {resolved_edit_path}")
            from diffusers import QwenImageEditPlusPipeline

            edit_pipe = QwenImageEditPlusPipeline.from_pretrained(
                resolved_edit_path,
                dtype=dtype,
            )
            # Apply same offloading as decompose pipeline
            cls._apply_offloading(edit_pipe, effective_offload_type, device, num_blocks_per_group)

        instance = cls(
            decompose_pipe=decompose_pipe,
            edit_pipe=edit_pipe,
            device=device,
            dtype=dtype,
        )
        instance._cpu_offload_enabled = cpu_offload_enabled
        instance._offload_type = effective_offload_type
        instance._edit_model_path = resolved_edit_path

        logger.info(
            f"QwenImageDiffusersPipeline loaded: "
            f"decompose=True, edit={edit_pipe is not None}, "
            f"offload_type={effective_offload_type}"
        )

        return instance

    @staticmethod
    def _apply_offloading(
        pipe,
        offload_type: str,
        device: torch.device,
        num_blocks_per_group: int = 2,
    ) -> None:
        """
        Apply offloading strategy to pipeline.

        Args:
            pipe: Pipeline to apply offloading to
            offload_type: One of "none", "model", "group", "sequential"
            device: Target device for onloading
            num_blocks_per_group: Blocks per group for group offloading
        """
        if offload_type == "none":
            pipe.to(device)
            logger.info(f"Pipeline moved to {device} (no offloading)")

        elif offload_type == "model":
            pipe.enable_model_cpu_offload()
            logger.info("Model CPU offload enabled (component-level)")

        elif offload_type == "group":
            # Group offloading: stream DiT blocks
            try:
                pipe.enable_group_offload(
                    onload_device=device,
                    offload_device=torch.device("cpu"),
                    offload_type="block_level",
                    num_blocks_per_group=num_blocks_per_group,
                    use_stream=True,  # Async data transfer
                    record_stream=True,  # Faster at expense of slightly more memory
                )
                logger.info(
                    f"Group offloading enabled (block_level, {num_blocks_per_group} blocks/group)"
                )
            except AttributeError:
                # Fallback if enable_group_offload not available
                logger.warning(
                    "enable_group_offload not available in this diffusers version, "
                    "falling back to model-level offload"
                )
                pipe.enable_model_cpu_offload()

        elif offload_type == "sequential":
            pipe.enable_sequential_cpu_offload()
            logger.info("Sequential CPU offload enabled (leaf-level, minimum VRAM)")

        else:
            raise ValueError(f"Unknown offload_type: {offload_type}")

    @classmethod
    def _load_edit_pipeline_quantized(
        cls,
        model_path: str,
        dtype: torch.dtype,
        quantize_text_encoder: Optional[str],
        quantize_transformer: Optional[str],
        quantize_vae: Optional[str] = None,
        cpu_offload: bool = True,
        offload_type: Optional[str] = None,
        num_blocks_per_group: int = 2,
        compile_transformer: bool = False,
        compile_mode: str = "default",
    ):
        """
        Load the edit pipeline with quantized components via unified quantize_component().

        Loads components in full precision, applies post-load quantization using the
        unified torchao-based system, then assembles the pipeline.
        """
        from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
        from transformers import Qwen2_5_VLForConditionalGeneration
        from llm_dit.quantization import quantize_component

        logger.info(
            f"Loading pipeline with quantization: text_encoder={quantize_text_encoder}, "
            f"transformer={quantize_transformer}, vae={quantize_vae}"
        )

        # Load text encoder in full precision, then quantize
        text_encoder = None
        if quantize_text_encoder and quantize_text_encoder != "none":
            logger.info(f"Loading text encoder (will quantize with {quantize_text_encoder})...")
            text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                subfolder="text_encoder",
                dtype=dtype,
                low_cpu_mem_usage=True,
            )
            text_encoder, stats = quantize_component(
                text_encoder, method=quantize_text_encoder, component_type="encoder"
            )
            logger.info(
                f"Text encoder quantized: {stats['quantized_layers']}/{stats['total_layers']} layers "
                f"({quantize_text_encoder})"
            )

        # Load transformer in full precision, then quantize
        transformer = None
        if quantize_transformer and quantize_transformer != "none":
            logger.info(f"Loading transformer (will quantize with {quantize_transformer})...")
            transformer = QwenImageTransformer2DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                dtype=dtype,
            )
            transformer, stats = quantize_component(
                transformer, method=quantize_transformer, component_type="transformer"
            )
            logger.info(
                f"Transformer quantized: {stats['quantized_layers']}/{stats['total_layers']} layers "
                f"({quantize_transformer})"
            )

        # Build the pipeline with quantized components
        pipeline_kwargs: dict = {"dtype": dtype}
        if text_encoder is not None:
            pipeline_kwargs["text_encoder"] = text_encoder
        if transformer is not None:
            pipeline_kwargs["transformer"] = transformer

        logger.info("Assembling pipeline with quantized components...")
        edit_pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_path,
            **pipeline_kwargs,
        )

        # Apply VAE quantization after pipeline is assembled
        if quantize_vae and quantize_vae != "none":
            try:
                edit_pipe.vae, stats = quantize_component(
                    edit_pipe.vae, method=quantize_vae, component_type="vae"
                )
                logger.info(f"VAE quantized: {stats['quantized_layers']}/{stats['total_layers']} layers")
            except Exception as e:
                logger.warning(f"VAE quantization failed: {e}, continuing without VAE quantization")

        # Resolve offload_type from new parameter or legacy cpu_offload
        if offload_type is not None:
            effective_offload = offload_type
        elif cpu_offload:
            effective_offload = "model"
        else:
            effective_offload = "none"

        # Apply torch.compile BEFORE CPU offload (compile works on GPU)
        # NOTE: torch.compile is INCOMPATIBLE with CPU offload
        if compile_transformer:
            if effective_offload != "none":
                logger.warning(
                    "torch.compile is incompatible with CPU offload (accelerate's tensor swapping "
                    "fails with compiled models). Skipping compilation. To use torch.compile, "
                    "disable cpu_offload."
                )
            else:
                effective_mode = compile_mode
                # CUDA graphs don't work without CPU offload either if model is too large
                if compile_mode in ("max-autotune", "reduce-overhead"):
                    logger.info(f"Using compile mode: {compile_mode}")

                logger.info(f"Compiling transformer with torch.compile (mode={effective_mode})...")
                logger.info("  Note: First inference will be slower due to compilation")
                edit_pipe.transformer = torch.compile(
                    edit_pipe.transformer,
                    mode=effective_mode,
                    fullgraph=True,
                )
                logger.info("Transformer compiled successfully")

        # Apply offloading
        cls._apply_offloading(
            edit_pipe,
            effective_offload,
            torch.device("cuda"),
            num_blocks_per_group,
        )

        return edit_pipe

    @classmethod
    def _load_decompose_pipeline_quantized(
        cls,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        quantize_text_encoder: Optional[str] = None,
        quantize_transformer: Optional[str] = None,
        quantize_vae: Optional[str] = None,
        cpu_offload: bool = True,
        offload_type: Optional[str] = None,
        num_blocks_per_group: int = 2,
        compile_transformer: bool = False,
        compile_mode: str = "default",
    ):
        """
        Load QwenImageLayeredPipeline with quantized components via unified quantize_component().
        """
        from diffusers import QwenImageLayeredPipeline, QwenImageTransformer2DModel
        from transformers import Qwen2_5_VLForConditionalGeneration
        from llm_dit.quantization import quantize_component

        text_encoder = None
        transformer = None

        # Load and quantize text encoder
        if quantize_text_encoder and quantize_text_encoder != "none":
            logger.info(f"Loading text encoder (will quantize with {quantize_text_encoder})...")
            text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                subfolder="text_encoder",
                dtype=dtype,
                device_map="cpu",
            )
            text_encoder, stats = quantize_component(
                text_encoder, method=quantize_text_encoder, component_type="encoder"
            )
            logger.info(
                f"Text encoder quantized: {stats['quantized_layers']}/{stats['total_layers']} layers "
                f"({quantize_text_encoder})"
            )

        # Load and quantize transformer
        if quantize_transformer and quantize_transformer != "none":
            logger.info(f"Loading transformer (will quantize with {quantize_transformer})...")
            transformer = QwenImageTransformer2DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                dtype=dtype,
            )
            transformer, stats = quantize_component(
                transformer, method=quantize_transformer, component_type="transformer"
            )
            logger.info(
                f"Transformer quantized: {stats['quantized_layers']}/{stats['total_layers']} layers "
                f"({quantize_transformer})"
            )

        # Build the pipeline with quantized components
        pipeline_kwargs: dict = {"dtype": dtype}
        if text_encoder is not None:
            pipeline_kwargs["text_encoder"] = text_encoder
        if transformer is not None:
            pipeline_kwargs["transformer"] = transformer

        logger.info("Assembling decompose pipeline with quantized components...")
        decompose_pipe = QwenImageLayeredPipeline.from_pretrained(
            model_path,
            **pipeline_kwargs,
        )

        # Apply VAE quantization after pipeline is assembled
        if quantize_vae and quantize_vae != "none":
            try:
                decompose_pipe.vae, stats = quantize_component(
                    decompose_pipe.vae, method=quantize_vae, component_type="vae"
                )
                logger.info(f"VAE quantized: {stats['quantized_layers']}/{stats['total_layers']} layers")
            except Exception as e:
                logger.warning(f"VAE quantization failed: {e}, continuing without VAE quantization")

        # Resolve offload_type from new parameter or legacy cpu_offload
        if offload_type is not None:
            effective_offload = offload_type
        elif cpu_offload:
            effective_offload = "model"
        else:
            effective_offload = "none"

        # Apply torch.compile BEFORE CPU offload
        # NOTE: torch.compile is INCOMPATIBLE with CPU offload
        if compile_transformer:
            if effective_offload != "none":
                logger.warning(
                    "torch.compile is incompatible with CPU offload (accelerate's tensor swapping "
                    "fails with compiled models). Skipping compilation."
                )
            else:
                logger.info(f"Compiling transformer with torch.compile (mode={compile_mode})...")
                decompose_pipe.transformer = torch.compile(
                    decompose_pipe.transformer,
                    mode=compile_mode,
                    fullgraph=True,
                )
                logger.info("Transformer compiled successfully")

        # Apply offloading
        cls._apply_offloading(
            decompose_pipe,
            effective_offload,
            torch.device("cuda"),
            num_blocks_per_group,
        )

        return decompose_pipe

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
        # Guard: decompose not available in edit-only mode
        if self.decompose_pipe is None:
            raise RuntimeError(
                "decompose() is not available in edit-only mode. "
                "Instantiate pipeline with edit_only=False to enable decomposition."
            )

        # Validate resolution
        if resolution not in SUPPORTED_RESOLUTIONS:
            raise ValueError(f"Resolution must be one of {SUPPORTED_RESOLUTIONS}, got {resolution}")

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
            dtype=self._dtype,
        )

        if self._cpu_offload_enabled:
            self.edit_pipe.enable_sequential_cpu_offload()
        else:
            self.edit_pipe.to(self._device)

        logger.info("Edit model loaded successfully")

    def unload_decompose_model(self) -> None:
        """Unload decompose model to free VRAM for editing."""
        if self.decompose_pipe is not None:
            logger.info("Unloading decompose model to free VRAM")
            del self.decompose_pipe
            self.decompose_pipe = None
            torch.cuda.empty_cache()

    def edit_layer(
        self,
        layer_image: Image.Image,
        instruction: str,
        num_inference_steps: int = DEFAULT_STEPS,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        seed: Optional[int] = None,
        max_size: int = 1024,
        unload_decompose: bool = True,
    ) -> Image.Image:
        """
        Edit a layer using text instructions.

        Args:
            layer_image: RGBA layer image to edit
            instruction: Text instruction for editing (e.g., "Change color to blue")
            num_inference_steps: Diffusion steps (default 40)
            cfg_scale: Classifier-free guidance scale (default 4.0)
            seed: Random seed for reproducibility
            max_size: Maximum dimension for input image (default 1024, resized if larger)
            unload_decompose: Unload decompose model before editing to save VRAM (default True)

        Returns:
            Edited RGBA image

        Example:
            edited = pipe.edit_layer(
                layer_image=layers[1],
                instruction="Make the house red",
            )
            edited.save("edited_layer.png")
        """
        # Unload decompose model to free VRAM if requested
        if unload_decompose and self.decompose_pipe is not None:
            self.unload_decompose_model()

        # Lazy load edit model if needed
        if self.edit_pipe is None:
            self.load_edit_model()

        # Handle RGBA -> RGB conversion for edit pipeline (VAE expects 3 channels)
        # Store alpha channel to reapply after editing
        original_size = layer_image.size
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

        # Resize if too large (saves VRAM, model works best at 640-1024)
        w, h = rgb_image.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            logger.info(f"Resizing input from {w}x{h} to {new_w}x{new_h} for VRAM efficiency")
            rgb_image = rgb_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            if alpha_channel is not None:
                alpha_channel = alpha_channel.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Setup generator
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        logger.info(f"Editing layer: instruction='{instruction}', steps={num_inference_steps}")

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
        max_size: int = 1024,
        unload_decompose: bool = True,
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
            max_size: Maximum dimension for input images (default 1024)
            unload_decompose: Unload decompose model before editing to save VRAM (default True)

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

        # Unload decompose model to free VRAM if requested
        if unload_decompose and self.decompose_pipe is not None:
            self.unload_decompose_model()

        # Lazy load edit model if needed
        if self.edit_pipe is None:
            self.load_edit_model()

        # Convert all images to RGB and resize if needed
        rgb_images = []
        for img in images:
            # Convert to RGB
            if img.mode == "RGBA":
                # Convert RGBA to RGB (composite onto white background)
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                rgb_img = background
            elif img.mode == "RGB":
                rgb_img = img
            else:
                rgb_img = img.convert("RGB")

            # Resize if too large
            w, h = rgb_img.size
            if max(w, h) > max_size:
                scale = max_size / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                logger.info(f"Resizing image from {w}x{h} to {new_w}x{new_h}")
                rgb_img = rgb_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            rgb_images.append(rgb_img)

        # Setup generator
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        logger.info(
            f"Multi-image edit: {len(rgb_images)} images, "
            f"instruction='{instruction[:80]}...', steps={num_inference_steps}"
        )

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

    def generate(
        self,
        prompt: str,
        negative_prompt: str = " ",
        height: int = 640,
        width: int = 640,
        num_inference_steps: int = DEFAULT_STEPS,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        seed: Optional[int] = None,
        unload_decompose: bool = True,
    ) -> Image.Image:
        """
        Generate image from text prompt only (no input image).

        Pure text-to-image generation using Qwen-Image-Edit-2511.
        Uses text-only encoding (no vision tokens).

        Args:
            prompt: Text description of image to generate
            negative_prompt: Negative prompt (default " ")
            height: Image height (must be multiple of 16, default 640)
            width: Image width (must be multiple of 16, default 640)
            num_inference_steps: Diffusion steps (default 40)
            cfg_scale: Classifier-free guidance scale (default 4.0)
            seed: Random seed for reproducibility
            unload_decompose: Unload decompose model first to save VRAM (default True)

        Returns:
            Generated PIL Image

        Example:
            >>> pipe = QwenImageDiffusersPipeline.from_pretrained(
            ...     model_path=None,
            ...     edit_model_path="/path/to/Qwen-Image-Edit-2511",
            ...     edit_only=True,
            ... )
            >>> image = pipe.generate("A cat sleeping on a couch", seed=42)
            >>> image.save("cat.png")
        """
        # Unload decompose model to free VRAM if requested
        if unload_decompose and self.decompose_pipe is not None:
            self.unload_decompose_model()

        # Lazy load edit model if needed
        if self.edit_pipe is None:
            self.load_edit_model()

        # Validate resolution (must be multiples of 16 for VAE)
        if width % 16 != 0 or height % 16 != 0:
            raise ValueError(
                f"Resolution must be multiples of 16. Got {width}x{height}. "
                f"Try {width // 16 * 16}x{height // 16 * 16} instead."
            )

        # Setup generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        logger.info(
            f"Text-to-image generation: prompt='{prompt[:80]}...', "
            f"resolution={width}x{height}, steps={num_inference_steps}"
        )

        # Create a blank image as starting point
        # The edit model will "edit" this blank canvas based on the prompt
        # This is a workaround since HF QwenImageEditPlusPipeline requires an input image
        blank_image = Image.new("RGB", (width, height), color=(128, 128, 128))
        logger.debug("Using gray canvas as text-to-image starting point")

        result = self.edit_pipe(
            image=blank_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=cfg_scale,
            generator=generator,
        )

        output_image = result.images[0]

        logger.info("Text-to-image generation complete")

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
