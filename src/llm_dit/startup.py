"""
Shared startup and pipeline loading logic.

This module provides unified pipeline/encoder loading used by both
the web server and CLI scripts to ensure consistent behavior.

Usage:
    from llm_dit.startup import PipelineLoader
    from llm_dit.cli import load_runtime_config

    config = load_runtime_config(args)
    loader = PipelineLoader(config)

    # Load full pipeline
    pipeline = loader.load_pipeline()

    # Or encoder only
    encoder = loader.load_encoder()
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from llm_dit.config import DyPEConfig as DyPEConfigType

logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    """Result of loading a pipeline or encoder."""

    pipeline: Optional["ZImagePipeline"] = None
    encoder: Optional["ZImageTextEncoder"] = None
    load_time: float = 0.0
    mode: str = "unknown"
    encoder_device: Optional[str] = None
    dit_device: Optional[str] = None
    vae_device: Optional[str] = None


def build_dype_config(config: "RuntimeConfig") -> Optional["DyPEConfigType"]:
    """Build DyPEConfig from RuntimeConfig.

    Args:
        config: RuntimeConfig with dype_* parameters

    Returns:
        DyPEConfig if dype is configured, None otherwise
    """
    from llm_dit.config import DyPEConfig

    # Check if DyPE is enabled
    if not getattr(config, "dype_enabled", False):
        return None

    return DyPEConfig(
        enabled=True,
        method=getattr(config, "dype_method", "vision_yarn"),
        dype_scale=getattr(config, "dype_scale", 2.0),
        dype_exponent=getattr(config, "dype_exponent", 2.0),
        dype_start_sigma=getattr(config, "dype_start_sigma", 1.0),
        base_shift=getattr(config, "dype_base_shift", 0.5),
        max_shift=getattr(config, "dype_max_shift", 1.15),
        base_resolution=getattr(config, "dype_base_resolution", 1024),
        anisotropic=getattr(config, "dype_anisotropic", False),
        multipass=getattr(config, "dype_multipass", "single"),
        pass2_strength=getattr(config, "dype_pass2_strength", 0.5),
        pass3_strength=getattr(config, "dype_pass3_strength", 0.4),
        frequency_modulation=getattr(config, "dype_frequency_modulation", False),
    )


class PipelineLoader:
    """
    Unified pipeline loading with all optimizations.

    Supports:
    - Full pipeline (encoder + DiT + VAE)
    - Encoder only (for distributed inference)
    - API encoder (remote encoding via heylookitsanllm)
    - Hybrid mode (API encoder + local DiT/VAE)
    """

    def __init__(self, config: "RuntimeConfig"):
        """
        Initialize loader with runtime config.

        Args:
            config: RuntimeConfig from load_runtime_config()
        """
        self.config = config
        self._pipeline = None
        self._encoder = None

    @property
    def pipeline(self):
        return self._pipeline

    @property
    def encoder(self):
        return self._encoder

    def _resolve_templates_dir(self) -> Optional[str]:
        """Find templates directory."""
        if self.config.templates_dir:
            return self.config.templates_dir

        # Try default locations
        candidates = [
            Path.cwd() / "templates" / "z_image",
            Path(__file__).parent.parent.parent / "templates" / "z_image",
        ]
        for path in candidates:
            if path.exists():
                return str(path)
        return None

    def _apply_optimizations(self, pipeline) -> None:
        """Apply flash attention, compile, etc. to pipeline."""
        # Flash Attention
        if self.config.flash_attn:
            logger.info("Enabling Flash Attention...")
            try:
                if hasattr(pipeline.transformer, "set_attention_backend"):
                    pipeline.transformer.set_attention_backend("flash")
                    logger.info("  Flash Attention enabled")
                else:
                    logger.warning("  Transformer does not support attention backend selection")
            except Exception as e:
                logger.warning(f"  Failed to enable Flash Attention: {e}")

        # Attention backend (if specified)
        if self.config.attention_backend and self.config.attention_backend != "auto":
            # Set global attention backend for llm_dit.utils.attention
            try:
                from llm_dit.utils.attention import (
                    set_attention_backend,
                    get_available_backends,
                )

                available = get_available_backends()
                if self.config.attention_backend in available:
                    set_attention_backend(self.config.attention_backend)
                    logger.info(f"Global attention backend set to: {self.config.attention_backend}")
                else:
                    logger.warning(
                        f"Requested attention backend '{self.config.attention_backend}' not available. "
                        f"Available: {available}. Using auto-detection."
                    )
            except Exception as e:
                logger.warning(f"Failed to set global attention backend: {e}")

            # Also set diffusers-specific attention backend
            backend_map = {
                "sdpa": "native",  # diffusers calls SDPA "native"
                "flash_attn_2": "flash",
                "flash_attn_3": "flash",  # diffusers may not distinguish
                "xformers": "xformers",
                "sage": "sage",
            }
            diffusers_backend = backend_map.get(
                self.config.attention_backend, self.config.attention_backend
            )
            logger.info(f"Setting diffusers attention backend to: {diffusers_backend}")
            try:
                if hasattr(pipeline.transformer, "set_attention_backend"):
                    pipeline.transformer.set_attention_backend(diffusers_backend)
                    logger.info(f"  Diffusers attention backend set to {diffusers_backend}")
            except Exception as e:
                logger.warning(f"  Failed to set diffusers attention backend: {e}")

        # torch.compile
        if self.config.compile:
            logger.info(f"Compiling transformer with torch.compile (mode={self.config.compile_mode})...")
            try:
                pipeline.transformer = torch.compile(pipeline.transformer, mode=self.config.compile_mode)
                logger.info("  Transformer compiled (first run will be slow)")
            except Exception as e:
                logger.warning(f"  Failed to compile: {e}")

        # Tiled VAE for large images
        if self.config.tiled_vae:
            logger.info(
                f"Enabling tiled VAE (tile_size={self.config.tile_size}, overlap={self.config.tile_overlap})..."
            )
            try:
                if hasattr(pipeline, "enable_tiled_vae"):
                    pipeline.enable_tiled_vae(
                        tile_size=self.config.tile_size,
                        tile_overlap=self.config.tile_overlap,
                    )
                    logger.info("  Tiled VAE enabled")
                elif hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_tiling"):
                    pipeline.vae.enable_tiling()
                    logger.info("  VAE tiling enabled (basic)")
                else:
                    logger.warning("  Pipeline does not support tiled VAE")
            except Exception as e:
                logger.warning(f"  Failed to enable tiled VAE: {e}")

    def _load_loras(self, pipeline) -> None:
        """Load LoRA weights into pipeline."""
        if not self.config.lora_paths:
            return

        logger.info(f"Loading {len(self.config.lora_paths)} LoRA(s)...")
        scales = self.config.lora_scales or [1.0] * len(self.config.lora_paths)
        try:
            updated = pipeline.load_lora(self.config.lora_paths, scale=scales)
            logger.info(f"  {updated} layers updated by LoRA")
        except Exception as e:
            logger.error(f"  Failed to load LoRA: {e}")

    def load_encoder(self) -> LoadResult:
        """
        Load only the text encoder.

        Returns:
            LoadResult with encoder set
        """
        from llm_dit.encoders import ZImageTextEncoder

        templates_dir = self._resolve_templates_dir()

        logger.info("=" * 60)
        logger.info("LOADING ENCODER")
        logger.info("=" * 60)
        logger.info(f"  Model: {self.config.model_path}")
        logger.info(f"  Device: {self.config.encoder_device_resolved}")
        logger.info(f"  Dtype: {self.config.dtype}")
        logger.info(f"  Quantization: {self.config.quantization}")
        logger.info(f"  Templates: {templates_dir}")
        logger.info(f"  Hidden layer: {self.config.hidden_layer}")
        if self.config.embedding_cache:
            logger.info(f"  Embedding cache: enabled (size={self.config.cache_size})")
        logger.info("-" * 60)

        start = time.time()

        self._encoder = ZImageTextEncoder.from_pretrained(
            self.config.model_path,
            templates_dir=templates_dir,
            device_map=self.config.encoder_device_resolved,
            dtype=self.config.get_dtype(),
            quantization=self.config.quantization,
            enable_cache=self.config.embedding_cache,
            cache_size=self.config.cache_size,
        )

        load_time = time.time() - start

        logger.info(f"Encoder loaded in {load_time:.1f}s")
        logger.info(f"  Device: {self._encoder.device}")
        logger.info("=" * 60)

        return LoadResult(
            encoder=self._encoder,
            load_time=load_time,
            mode="encoder_only",
            encoder_device=str(self._encoder.device),
        )

    def load_pipeline(self) -> LoadResult:
        """
        Load full pipeline (encoder + DiT + VAE).

        Returns:
            LoadResult with pipeline set
        """
        # Route to correct pipeline based on model_type
        model_type = getattr(self.config, "model_type", "zimage")
        if model_type == "qwenimage-layered":
            return self._load_qwen_image_pipeline()
        elif model_type == "qwenimage-t2i":
            # Qwen-Image T2I uses on-demand loading via web API
            # Skip startup loading - pipeline will be loaded on first request
            logger.info("=" * 60)
            logger.info("QWEN-IMAGE T2I MODE")
            logger.info("=" * 60)
            logger.info("Qwen-Image T2I uses on-demand loading via /api/qwen-image/generate")
            logger.info(f"  Model path: {self.config.qwen_image_model_path}")
            logger.info(f"  Steps: {self.config.get_qwen_image_steps()}")
            logger.info(f"  Resolution: {self.config.get_qwen_image_resolution()}")
            logger.info(f"  Transformer quant: {self.config.get_qwen_image_quantize_transformer()}")
            logger.info(f"  Text encoder quant: {self.config.qwen_image_quantize_text_encoder}")
            logger.info("=" * 60)
            return LoadResult(pipeline=None, encoder=None, mode="qwenimage-t2i_ondemand")
        elif model_type == "qwenimage-edit":
            # Qwen-Image Edit uses on-demand loading via web API
            # Skip startup loading - pipeline will be loaded on first request
            logger.info("=" * 60)
            logger.info("QWEN-IMAGE EDIT MODE")
            logger.info("=" * 60)
            logger.info("Qwen-Image Edit uses on-demand loading via /api/qwen-image/edit")
            logger.info(f"  Model path: {self.config.qwen_image_model_path}")
            logger.info(f"  Steps: {self.config.get_qwen_image_steps()}")
            logger.info(f"  Resolution: {self.config.get_qwen_image_resolution()}")
            logger.info(f"  Transformer quant: {self.config.get_qwen_image_quantize_transformer()}")
            logger.info(f"  Text encoder quant: {self.config.qwen_image_quantize_text_encoder}")
            logger.info("=" * 60)
            return LoadResult(pipeline=None, encoder=None, mode="qwenimage-edit_ondemand")
        elif model_type == "ltx2":
            return self._load_ltx2_pipeline()
        elif model_type == "wan":
            return self._load_wan_pipeline()

        from llm_dit.pipelines import ZImagePipeline

        templates_dir = self._resolve_templates_dir()

        logger.info("=" * 60)
        logger.info("LOADING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"  Model: {self.config.model_path}")
        logger.info(f"  Encoder device: {self.config.encoder_device_resolved}")
        logger.info(f"  DiT device: {self.config.dit_device_resolved}")
        logger.info(f"  VAE device: {self.config.vae_device_resolved}")
        logger.info(f"  Dtype: {self.config.dtype}")
        logger.info(f"  Quantization: {self.config.quantization}")
        logger.info(f"  Templates: {templates_dir}")
        logger.info("-" * 60)
        logger.info("Optimizations:")
        logger.info(f"  Flash Attention: {self.config.flash_attn}")
        logger.info(f"  Attention backend: {self.config.attention_backend or 'auto'}")
        logger.info(f"  torch.compile: {self.config.compile}")
        logger.info(f"  CPU offload: {self.config.cpu_offload}")
        logger.info(f"  Tiled VAE: {self.config.tiled_vae}")
        if self.config.tiled_vae:
            logger.info(f"    Tile size: {self.config.tile_size}")
            logger.info(f"    Tile overlap: {self.config.tile_overlap}")
        logger.info(f"  Embedding cache: {self.config.embedding_cache}")
        if self.config.embedding_cache:
            logger.info(f"    Cache size: {self.config.cache_size}")
        logger.info(f"  Long prompt mode: {self.config.long_prompt_mode}")
        logger.info(f"  Custom scheduler: {self.config.use_custom_scheduler}")
        dynamic_shift = getattr(self.config, "dynamic_shift", False)
        if dynamic_shift:
            logger.info(f"  Scheduler shift: dynamic (base=0.5, max=1.15)")
        else:
            logger.info(f"  Scheduler shift: {self.config.shift}")
        d_noise = getattr(self.config, "d_noise", 1.0)
        if d_noise != 1.0:
            logger.info(f"  D-noise: {d_noise:.3f}")
        if self.config.lora_paths:
            logger.info(f"  LoRAs: {len(self.config.lora_paths)}")

        # Build DyPE config if enabled
        dype_config = build_dype_config(self.config)
        if dype_config is not None:
            logger.info(
                f"  DyPE: enabled (method={dype_config.method}, scale={dype_config.dype_scale})"
            )
        else:
            logger.info("  DyPE: disabled")

        # Log SLG (Skip Layer Guidance) config
        slg_scale = getattr(self.config, "slg_scale", 0.0)
        slg_layers = getattr(self.config, "slg_layers", None)
        if slg_scale > 0 and slg_layers:
            slg_start = getattr(self.config, "slg_start", 0.01)
            slg_stop = getattr(self.config, "slg_stop", 0.2)
            logger.info(
                f"  SLG: enabled (scale={slg_scale}, layers={slg_layers}, range=[{slg_start:.0%}, {slg_stop:.0%}])"
            )
        else:
            logger.info("  SLG: disabled")
        logger.info("-" * 60)

        start = time.time()

        self._pipeline = ZImagePipeline.from_pretrained(
            self.config.model_path,
            text_encoder_path=self.config.text_encoder_path,
            templates_dir=templates_dir,
            dtype=self.config.get_dtype(),
            encoder_device=self.config.encoder_device_resolved,
            dit_device=self.config.dit_device_resolved,
            vae_device=self.config.vae_device_resolved,
            quantization=self.config.quantization,
            enable_cache=self.config.embedding_cache,
            cache_size=self.config.cache_size,
            dype_config=dype_config,
        )

        load_time = time.time() - start
        logger.info(f"Pipeline loaded in {load_time:.1f}s")

        # Apply optimizations
        self._apply_optimizations(self._pipeline)

        # Load LoRAs
        self._load_loras(self._pipeline)

        # Store encoder reference
        self._encoder = self._pipeline.encoder

        # Log final state
        logger.info("-" * 60)
        logger.info("Final pipeline state:")
        logger.info(f"  pipeline.device: {self._pipeline.device}")
        logger.info(f"  pipeline.dtype: {self._pipeline.dtype}")
        if self._pipeline.encoder is not None:
            logger.info(f"  encoder.device: {self._pipeline.encoder.device}")
        if self._pipeline.transformer is not None:
            logger.info(
                f"  transformer.device: {next(self._pipeline.transformer.parameters()).device}"
            )
        if self._pipeline.vae is not None:
            logger.info(f"  vae.device: {next(self._pipeline.vae.parameters()).device}")
        logger.info("=" * 60)

        return LoadResult(
            pipeline=self._pipeline,
            encoder=self._encoder,
            load_time=load_time,
            mode="full",
            encoder_device=str(self._pipeline.encoder.device)
            if self._pipeline.encoder is not None
            else None,
            dit_device=str(next(self._pipeline.transformer.parameters()).device)
            if self._pipeline.transformer is not None
            else None,
            vae_device=str(next(self._pipeline.vae.parameters()).device)
            if self._pipeline.vae is not None
            else None,
        )

    def _load_qwen_image_pipeline(self) -> LoadResult:
        """
        Load Qwen-Image pipeline (QwenImageDiffusersPipeline).

        Used for Qwen-Image-Layered and Qwen-Image-Edit models.

        Returns:
            LoadResult with pipeline set
        """
        from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

        # Check if edit_only mode (standalone edit model, no decompose)
        edit_only = getattr(self.config, "qwen_image_edit_only", False)

        logger.info("=" * 60)
        if edit_only:
            logger.info("LOADING QWEN-IMAGE-EDIT PIPELINE (standalone)")
        else:
            logger.info("LOADING QWEN-IMAGE PIPELINE")
        logger.info("=" * 60)
        if not edit_only:
            logger.info(f"  Model: {self.config.model_path}")
        logger.info(f"  Edit model: {self.config.qwen_image_edit_model_path or 'None'}")
        logger.info(f"  Edit only: {edit_only}")
        logger.info(f"  Steps: {self.config.get_qwen_image_steps()}")
        logger.info(f"  Resolution: {self.config.get_qwen_image_resolution()}")
        logger.info(f"  Dtype: {self.config.dtype}")
        logger.info(f"  Text encoder quantization: {self.config.qwen_image_quantize_text_encoder}")
        logger.info(
            f"  Transformer quantization: {self.config.get_qwen_image_quantize_transformer()}"
        )
        logger.info(f"  CPU offload: {self.config.qwen_image_cpu_offload}")
        logger.info(f"  torch.compile: {self.config.compile} (mode: {self.config.compile_mode})")
        logger.info("-" * 60)

        start = time.time()

        # Get quantization settings (convert 'none' to None, use variant-aware getter for transformer)
        quant_te = self.config.qwen_image_quantize_text_encoder
        quant_tf = self.config.get_qwen_image_quantize_transformer()
        quant_te = quant_te if quant_te != "none" else None
        quant_tf = quant_tf if quant_tf != "none" else None

        # Load pipeline
        self._pipeline = QwenImageDiffusersPipeline.from_pretrained(
            self.config.model_path,
            edit_model_path=self.config.qwen_image_edit_model_path or None,
            edit_only=edit_only,
            dtype=self.config.get_dtype(),
            cpu_offload=self.config.qwen_image_cpu_offload,
            quantize_text_encoder=quant_te,
            quantize_transformer=quant_tf,
            compile_transformer=self.config.compile,
            compile_mode=self.config.compile_mode,
        )

        load_time = time.time() - start
        logger.info(f"Qwen-Image pipeline loaded in {load_time:.1f}s")
        logger.info("=" * 60)

        return LoadResult(
            pipeline=self._pipeline,
            load_time=load_time,
            mode="full",
        )

    def _load_ltx2_pipeline(self) -> LoadResult:
        """
        Load LTX-2 video generation pipeline.

        LTX-2 is a 19B video+audio model that uses on-demand loading
        for memory efficiency on 24GB GPUs.

        Returns:
            LoadResult with mode set for on-demand loading
        """
        # Get LTX-2 config
        ltx2_model_path = getattr(self.config, "ltx2_model_path", "")
        ltx2_num_frames = getattr(self.config, "ltx2_num_frames", 33)
        ltx2_fps = getattr(self.config, "ltx2_fps", 24)
        ltx2_steps = getattr(self.config, "ltx2_steps", None)
        ltx2_guidance_scale = getattr(self.config, "ltx2_guidance_scale", 3.5)
        ltx2_offload_mode = getattr(self.config, "ltx2_offload_mode", "model")
        ltx2_lora_path = getattr(self.config, "ltx2_lora_path", "")
        ltx2_lora_scale = getattr(self.config, "ltx2_lora_scale", 1.0)
        ltx2_audio = getattr(self.config, "ltx2_audio", False)

        logger.info("=" * 60)
        logger.info("LTX-2 VIDEO MODE")
        logger.info("=" * 60)
        logger.info("LTX-2 uses on-demand loading via /api/ltx2/generate")
        logger.info(f"  Model path: {ltx2_model_path}")
        logger.info(f"  Frames: {ltx2_num_frames}")
        logger.info(f"  FPS: {ltx2_fps}")
        logger.info(f"  Steps: {ltx2_steps or 'auto (12 for distilled)'}")
        logger.info(f"  Guidance: {ltx2_guidance_scale}")
        logger.info(f"  Offload: {ltx2_offload_mode}")
        if ltx2_lora_path:
            logger.info(f"  LoRA: {ltx2_lora_path} (scale={ltx2_lora_scale})")
        logger.info(f"  Audio: {ltx2_audio}")
        logger.info("=" * 60)

        return LoadResult(pipeline=None, encoder=None, mode="ltx2_ondemand")

    def _load_wan_pipeline(self) -> LoadResult:
        """
        Load Wan/HuMo video generation pipeline.

        Uses HuMo transformer as base, supporting both T2V and audio-conditioned modes.
        Audio conditioning is controlled via audio_scale parameter at runtime.

        Returns:
            LoadResult with pipeline loaded
        """
        from llm_dit.pipelines import WanVideoPipeline

        # Get Wan/HuMo config
        humo_path = getattr(self.config, "wan_humo_path", "")
        wan_path = getattr(self.config, "wan_base_path", "")
        whisper_path = getattr(self.config, "wan_whisper_path", "")
        humo_variant = getattr(self.config, "wan_humo_variant", "17B")
        wan_num_frames = getattr(self.config, "wan_num_frames", 97)
        wan_fps = getattr(self.config, "wan_fps", 25)
        wan_height = getattr(self.config, "wan_height", 720)
        wan_width = getattr(self.config, "wan_width", 1280)
        wan_steps = getattr(self.config, "wan_steps", 50)
        wan_guidance_scale = getattr(self.config, "wan_guidance_scale", 5.0)
        wan_audio_scale = getattr(self.config, "wan_audio_scale", 0.0)
        wan_offload_mode = getattr(self.config, "wan_offload_mode", "model")

        logger.info("=" * 60)
        logger.info("LOADING WAN/HUMO VIDEO PIPELINE")
        logger.info("=" * 60)
        logger.info(f"  HuMo path: {humo_path}")
        logger.info(f"  HuMo variant: {humo_variant}")
        logger.info(f"  Wan path: {wan_path}")
        logger.info(f"  Whisper path: {whisper_path or 'lazy-load'}")
        logger.info(f"  Resolution: {wan_width}x{wan_height}")
        logger.info(
            f"  Frames: {wan_num_frames} (~{wan_num_frames / wan_fps:.1f}s at {wan_fps}fps)"
        )
        logger.info(f"  Steps: {wan_steps}")
        logger.info(f"  Text guidance (scale_t): {wan_guidance_scale}")
        logger.info(f"  Audio guidance (scale_a): {wan_audio_scale}")
        logger.info(f"  Offload: {wan_offload_mode}")
        logger.info("-" * 60)

        start = time.time()

        # Determine CPU offload setting
        enable_cpu_offload = wan_offload_mode in ("model", "sequential")

        # Load HuMo pipeline
        self._pipeline = WanVideoPipeline.from_pretrained(
            humo_path=humo_path,
            wan_path=wan_path,
            whisper_path=whisper_path or None,
            humo_variant=humo_variant,
            dtype=self.config.get_dtype(),
            enable_cpu_offload=enable_cpu_offload,
        )

        load_time = time.time() - start

        logger.info(f"Wan/HuMo pipeline loaded in {load_time:.1f}s")
        logger.info(f"  Mode: {self._pipeline.mode.upper()}")
        logger.info(f"  Device: {self._pipeline.device}")
        logger.info(f"  Dtype: {self._pipeline.dtype}")
        logger.info("=" * 60)

        return LoadResult(
            pipeline=self._pipeline,
            load_time=load_time,
            mode=f"wan_{self._pipeline.mode}",
        )

    def load_api_encoder(self) -> LoadResult:
        """
        Load encoder using remote API backend.

        Returns:
            LoadResult with encoder set
        """
        from llm_dit.backends.api import APIBackend, APIBackendConfig
        from llm_dit.encoders import ZImageTextEncoder
        from llm_dit.templates import TemplateRegistry

        templates_dir = self._resolve_templates_dir()

        logger.info("=" * 60)
        logger.info("LOADING API ENCODER")
        logger.info("=" * 60)
        logger.info(f"  API URL: {self.config.api_url}")
        logger.info(f"  Model: {self.config.api_model}")
        logger.info(f"  Templates: {templates_dir}")
        logger.info("-" * 60)

        start = time.time()

        # Create API backend
        api_config = APIBackendConfig(
            base_url=self.config.api_url,
            model_id=self.config.api_model,
            encoding_format="base64",
            hidden_layer=self.config.hidden_layer,
        )
        backend = APIBackend(api_config)
        logger.info(f"  Hidden layer: {self.config.hidden_layer}")

        # Load templates
        templates = None
        if templates_dir:
            templates = TemplateRegistry.from_directory(templates_dir)
            logger.info(f"  Loaded {len(templates)} templates")

        # Create encoder with API backend
        self._encoder = ZImageTextEncoder(
            backend=backend,
            templates=templates,
        )

        load_time = time.time() - start

        logger.info(f"API encoder ready in {load_time:.1f}s")
        logger.info("=" * 60)

        return LoadResult(
            encoder=self._encoder,
            load_time=load_time,
            mode="api_encoder",
        )

    def load_api_pipeline(self) -> LoadResult:
        """
        Load pipeline with API encoder + local DiT/VAE.

        Returns:
            LoadResult with pipeline set
        """
        from llm_dit.backends.api import APIBackend, APIBackendConfig
        from llm_dit.encoders import ZImageTextEncoder
        from llm_dit.pipelines import ZImagePipeline
        from llm_dit.templates import TemplateRegistry

        templates_dir = self._resolve_templates_dir()

        logger.info("=" * 60)
        logger.info("LOADING DISTRIBUTED PIPELINE")
        logger.info("=" * 60)
        logger.info(f"  API URL: {self.config.api_url}")
        logger.info(f"  API Model: {self.config.api_model}")
        logger.info(f"  Local Model: {self.config.model_path}")
        logger.info(f"  DiT device: {self.config.dit_device_resolved}")
        logger.info(f"  VAE device: {self.config.vae_device_resolved}")
        logger.info(f"  Templates: {templates_dir}")
        logger.info("-" * 60)

        start = time.time()

        # Create API backend
        api_config = APIBackendConfig(
            base_url=self.config.api_url,
            model_id=self.config.api_model,
            encoding_format="base64",
            hidden_layer=self.config.hidden_layer,
        )
        backend = APIBackend(api_config)
        logger.info(f"  Hidden layer: {self.config.hidden_layer}")

        # Load templates
        templates = None
        if templates_dir:
            templates = TemplateRegistry.from_directory(templates_dir)

        # Create encoder with API backend
        api_encoder = ZImageTextEncoder(
            backend=backend,
            templates=templates,
        )

        # Build DyPE config if enabled
        dype_config = build_dype_config(self.config)
        if dype_config is not None:
            logger.info(f"  DyPE: enabled (method={dype_config.method})")

        # Load generator-only pipeline
        self._pipeline = ZImagePipeline.from_pretrained_generator_only(
            self.config.model_path,
            dtype=self.config.get_dtype(),
            enable_cpu_offload=self.config.cpu_offload,
            dit_device=self.config.dit_device_resolved,
            vae_device=self.config.vae_device_resolved,
            dype_config=dype_config,
        )

        # Apply optimizations
        self._apply_optimizations(self._pipeline)

        # Attach API encoder
        self._pipeline.encoder = api_encoder
        self._encoder = api_encoder

        # Load LoRAs
        self._load_loras(self._pipeline)

        load_time = time.time() - start

        logger.info(f"Distributed pipeline ready in {load_time:.1f}s")
        logger.info("=" * 60)

        return LoadResult(
            pipeline=self._pipeline,
            encoder=self._encoder,
            load_time=load_time,
            mode="distributed",
            dit_device=str(next(self._pipeline.transformer.parameters()).device)
            if self._pipeline.transformer
            else None,
            vae_device=str(next(self._pipeline.vae.parameters()).device)
            if self._pipeline.vae
            else None,
        )

    def auto_load(self, encoder_only: bool = False, use_api: bool = False) -> LoadResult:
        """
        Automatically determine and load the appropriate configuration.

        Args:
            encoder_only: If True, only load encoder
            use_api: If True, prefer API backend when api_url is set

        Returns:
            LoadResult with loaded components
        """
        # Check for edit_only mode (Qwen-Image-Edit standalone)
        edit_only = getattr(self.config, "qwen_image_edit_only", False)
        has_edit_model = bool(getattr(self.config, "qwen_image_edit_model_path", ""))

        # API encoder only (no local model, and not edit_only mode)
        if (
            self.config.api_url
            and not self.config.model_path
            and not (edit_only and has_edit_model)
        ):
            return self.load_api_encoder()

        # Distributed mode (API encoder + local DiT/VAE)
        if self.config.api_url and self.config.model_path and use_api:
            return self.load_api_pipeline()

        # Encoder only
        if encoder_only:
            return self.load_encoder()

        # Full pipeline (including edit_only mode)
        return self.load_pipeline()
