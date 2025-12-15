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
from typing import Optional

import torch

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
                if hasattr(pipeline.transformer, 'set_attention_backend'):
                    pipeline.transformer.set_attention_backend("flash")
                    logger.info("  Flash Attention enabled")
                else:
                    logger.warning("  Transformer does not support attention backend selection")
            except Exception as e:
                logger.warning(f"  Failed to enable Flash Attention: {e}")

        # Attention backend (if specified)
        if self.config.attention_backend and self.config.attention_backend != "auto":
            logger.info(f"Setting attention backend to {self.config.attention_backend}...")
            try:
                if hasattr(pipeline.transformer, 'set_attention_backend'):
                    pipeline.transformer.set_attention_backend(self.config.attention_backend)
                    logger.info(f"  Attention backend set to {self.config.attention_backend}")
            except Exception as e:
                logger.warning(f"  Failed to set attention backend: {e}")

        # torch.compile
        if self.config.compile:
            logger.info("Compiling transformer with torch.compile...")
            try:
                pipeline.transformer = torch.compile(
                    pipeline.transformer,
                    mode="reduce-overhead"
                )
                logger.info("  Transformer compiled (first run will be slow)")
            except Exception as e:
                logger.warning(f"  Failed to compile: {e}")

        # Tiled VAE for large images
        if self.config.tiled_vae:
            logger.info(f"Enabling tiled VAE (tile_size={self.config.tile_size}, overlap={self.config.tile_overlap})...")
            try:
                if hasattr(pipeline, 'enable_tiled_vae'):
                    pipeline.enable_tiled_vae(
                        tile_size=self.config.tile_size,
                        tile_overlap=self.config.tile_overlap,
                    )
                    logger.info("  Tiled VAE enabled")
                elif hasattr(pipeline, 'vae') and hasattr(pipeline.vae, 'enable_tiling'):
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
        logger.info(f"  Dtype: {self.config.torch_dtype}")
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
            torch_dtype=self.config.get_torch_dtype(),
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
        from llm_dit.pipelines import ZImagePipeline

        templates_dir = self._resolve_templates_dir()

        logger.info("=" * 60)
        logger.info("LOADING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"  Model: {self.config.model_path}")
        logger.info(f"  Encoder device: {self.config.encoder_device_resolved}")
        logger.info(f"  DiT device: {self.config.dit_device_resolved}")
        logger.info(f"  VAE device: {self.config.vae_device_resolved}")
        logger.info(f"  Dtype: {self.config.torch_dtype}")
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
        logger.info(f"  Scheduler shift: {self.config.shift}")
        if self.config.lora_paths:
            logger.info(f"  LoRAs: {len(self.config.lora_paths)}")
        logger.info("-" * 60)

        start = time.time()

        self._pipeline = ZImagePipeline.from_pretrained(
            self.config.model_path,
            text_encoder_path=self.config.text_encoder_path,
            templates_dir=templates_dir,
            torch_dtype=self.config.get_torch_dtype(),
            encoder_device=self.config.encoder_device_resolved,
            dit_device=self.config.dit_device_resolved,
            vae_device=self.config.vae_device_resolved,
            enable_cache=self.config.embedding_cache,
            cache_size=self.config.cache_size,
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
            logger.info(f"  transformer.device: {next(self._pipeline.transformer.parameters()).device}")
        if self._pipeline.vae is not None:
            logger.info(f"  vae.device: {next(self._pipeline.vae.parameters()).device}")
        logger.info("=" * 60)

        return LoadResult(
            pipeline=self._pipeline,
            encoder=self._encoder,
            load_time=load_time,
            mode="full",
            encoder_device=str(self._pipeline.encoder.device) if self._pipeline.encoder is not None else None,
            dit_device=str(next(self._pipeline.transformer.parameters()).device) if self._pipeline.transformer is not None else None,
            vae_device=str(next(self._pipeline.vae.parameters()).device) if self._pipeline.vae is not None else None,
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

        # Load generator-only pipeline
        self._pipeline = ZImagePipeline.from_pretrained_generator_only(
            self.config.model_path,
            torch_dtype=self.config.get_torch_dtype(),
            enable_cpu_offload=self.config.cpu_offload,
            dit_device=self.config.dit_device_resolved,
            vae_device=self.config.vae_device_resolved,
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
            dit_device=str(next(self._pipeline.transformer.parameters()).device) if self._pipeline.transformer else None,
            vae_device=str(next(self._pipeline.vae.parameters()).device) if self._pipeline.vae else None,
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
        # API encoder only (no local model)
        if self.config.api_url and not self.config.model_path:
            return self.load_api_encoder()

        # Distributed mode (API encoder + local DiT/VAE)
        if self.config.api_url and self.config.model_path and use_api:
            return self.load_api_pipeline()

        # Encoder only
        if encoder_only:
            return self.load_encoder()

        # Full pipeline
        return self.load_pipeline()
