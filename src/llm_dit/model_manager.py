"""Unified model lifecycle management for all pipelines.

Consolidates model loading, unloading, and VRAM management that was
previously scattered across startup.py and server.py into a single
thread-safe manager.

Usage:
    from llm_dit.model_manager import ModelManager
    from llm_dit.cli import load_runtime_config

    config = load_runtime_config(args)
    manager = ModelManager(config)

    # Load a pipeline
    result = manager.load("flux2")

    # Check if loaded
    if manager.is_loaded("flux2"):
        pipeline = manager.get_pipeline("flux2")

    # Unload to free VRAM
    manager.unload("flux2")

    # Unload everything except one
    manager.unload_all_except("zimage")
"""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch

from llm_dit.utils.memory import cleanup_memory

if TYPE_CHECKING:
    from llm_dit.cli import RuntimeConfig
    from llm_dit.config import DyPEConfig as DyPEConfigType

logger = logging.getLogger(__name__)


# Pipeline identifiers (canonical names used internally)
PIPELINE_IDS = {"zimage", "flux2", "ltx2", "qwen_image", "qwen_image_t2i"}

# Aliases that map to canonical IDs
PIPELINE_ALIASES = {
    "z-image": "zimage",
    "qwen-image": "qwen_image",
    "qwenimage-edit": "qwen_image",
    "qwenimage-t2i": "qwen_image_t2i",
    "qwen-image-t2i": "qwen_image_t2i",
}

# Fields that can be hot-reloaded without restarting
HOT_RELOAD_SAFE = {
    # Scheduler params
    "shift",
    "shift_terminal",
    "dynamic_shift",
    "d_noise",
    # Generation defaults
    "height",
    "width",
    "steps",
    "guidance_scale",
    "cfg_normalization",
    "cfg_truncation",
    "cfg_norm_mode",
    # Prompt handling
    "long_prompt_mode",
    "hidden_layer",
    "layer_weights",
    "enable_thinking",
    "default_template",
    "system_prompt",
    "thinking_content",
    "assistant_content",
    # DyPE feature params
    "dype_enabled",
    "dype_method",
    "dype_scale",
    "dype_exponent",
    "dype_start_sigma",
    "dype_base_shift",
    "dype_max_shift",
    "dype_base_resolution",
    "dype_anisotropic",
    "dype_multipass",
    "dype_pass2_strength",
    "dype_pass3_strength",
    "dype_frequency_modulation",
    # SLG feature params
    "slg_scale",
    "slg_layers",
    "slg_start",
    "slg_stop",
    # FMTT feature params
    "fmtt_scale",
    "fmtt_start",
    "fmtt_stop",
    "fmtt_normalize",
    "fmtt_decode_scale",
    "fmtt_siglip_model",
    "fmtt_siglip_device",
    # Cache settings
    "embedding_cache",
    "cache_size",
    # Tiled VAE (can change between generations)
    "tiled_vae",
    "tile_size",
    "tile_overlap",
    # Seed
    "seed",
    "negative_prompt",
}

# Fields that require server restart (model reload)
REQUIRES_RESTART = {
    # Model paths
    "model_path",
    "text_encoder_path",
    "templates_dir",
    "qwen_image_model_path",
    "qwen_image_edit_model_path",
    # Device placement
    "encoder_device",
    "dit_device",
    "vae_device",
    # Quantization
    "quantization",
    "dtype",
    "qwen_image_quantize_text_encoder",
    "qwen_image_quantize_transformer",
    # Memory management
    "cpu_offload",
    "qwen_image_cpu_offload",
    # Attention backend
    "attention_backend",
    "flash_attn",
    # Compilation
    "compile",
    "compile_mode",
    # Audio models
    "audio_enabled",
    "audio_vae_path",
    "vocoder_path",
}


@dataclass
class LoadResult:
    """Result of loading a pipeline or encoder."""

    pipeline: Any = None
    encoder: Any = None
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


def _resolve_pipeline_id(pipeline_id: str) -> str:
    """Resolve aliases to canonical pipeline IDs."""
    canonical = PIPELINE_ALIASES.get(pipeline_id.lower(), pipeline_id.lower())
    if canonical not in PIPELINE_IDS:
        raise ValueError(
            f"Unknown pipeline: {pipeline_id}. "
            f"Available: {sorted(PIPELINE_IDS)}"
        )
    return canonical


class ModelManager:
    """Unified model lifecycle: load, unload, status for all pipelines.

    Replaces the ~15 global variables and scattered load/unload functions
    that were previously in server.py and startup.py.

    Thread-safe: uses per-pipeline locks to prevent concurrent loading
    (which would cause OOM on 24GB GPUs).
    """

    def __init__(self, config: "RuntimeConfig"):
        self.config = config
        self._pipelines: dict[str, Any] = {}
        self._locks: dict[str, threading.Lock] = {
            pid: threading.Lock() for pid in PIPELINE_IDS
        }
        self._loading_in_progress: dict[str, bool] = {
            pid: False for pid in PIPELINE_IDS
        }
        # Z-Image stores encoder separately for encoder-only mode
        self._encoder: Any = None
        self._encoder_only_mode: bool = False
        # LTX-2 cached Gemma3 encoder (persists between generations)
        self._ltx2_encoder: Any = None
        # LTX-2 cached transformer state dict + config (bf16 pinned, for fast reconstruction)
        self._ltx2_transformer_cache: Optional[dict] = None
        # LTX-2 cached VAE decoder (pinned memory, shuttled to GPU per generation)
        self._ltx2_vae: Any = None
        # LTX-2 cached audio decoder and vocoder (pinned memory, shuttled to GPU)
        self._ltx2_audio_decoder: Any = None
        self._ltx2_vocoder: Any = None

    # -- public API --

    def load(self, pipeline_id: str) -> LoadResult:
        """Load a pipeline by ID. Thread-safe with automatic VRAM management.

        Unloads other pipelines first to free VRAM, then loads the requested one.

        Args:
            pipeline_id: Pipeline identifier (e.g., "flux2", "zimage", "ltx2")

        Returns:
            LoadResult with loaded components
        """
        pid = _resolve_pipeline_id(pipeline_id)

        # Fast path: already loaded
        if self.is_loaded(pid):
            return LoadResult(
                pipeline=self._pipelines.get(pid),
                mode=f"{pid}_already_loaded",
            )

        with self._locks[pid]:
            # Double-check inside lock
            if self.is_loaded(pid):
                return LoadResult(
                    pipeline=self._pipelines.get(pid),
                    mode=f"{pid}_already_loaded",
                )

            if self._loading_in_progress.get(pid, False):
                raise ValueError(f"{pid} loading already in progress, please wait")

            self._loading_in_progress[pid] = True
            try:
                # Unload other pipelines first to free VRAM
                self.unload_all_except(pid)

                # Dispatch to pipeline-specific loader
                loader = {
                    "zimage": self._load_zimage,
                    "flux2": self._load_flux2,
                    "ltx2": self._load_ltx2,
                    "qwen_image": self._load_qwen_image,
                    "qwen_image_t2i": self._load_qwen_image_t2i,
                }.get(pid)

                if loader is None:
                    raise ValueError(f"No loader implemented for pipeline: {pid}")

                result = loader()
                return result
            except Exception:
                # Clean up partial state on failure
                self._pipelines.pop(pid, None)
                cleanup_memory()
                raise
            finally:
                self._loading_in_progress[pid] = False

    def unload(self, pipeline_id: str) -> bool:
        """Unload a pipeline to free VRAM.

        Returns True if the pipeline was loaded and is now unloaded.
        """
        pid = _resolve_pipeline_id(pipeline_id)
        return self._unload_pipeline(pid)

    def unload_all_except(self, keep: Optional[str] = None) -> list[str]:
        """Unload all pipelines except the specified one.

        Args:
            keep: Pipeline ID to keep loaded (None = unload all)

        Returns:
            List of pipeline names that were unloaded
        """
        keep_pid = _resolve_pipeline_id(keep) if keep else None
        unloaded = []

        for pid in PIPELINE_IDS:
            if pid == keep_pid:
                continue
            if self._unload_pipeline(pid):
                unloaded.append(pid)

        if unloaded:
            cleanup_memory("unload_all")
            logger.info(f"[VRAM] Unloaded pipelines: {', '.join(unloaded)}")

        return unloaded

    def is_loaded(self, pipeline_id: str) -> bool:
        """Check if a pipeline is currently loaded."""
        pid = _resolve_pipeline_id(pipeline_id)
        pipeline = self._pipelines.get(pid)
        if pipeline is None:
            return False
        # FLUX.2 stores as dict, check it's populated
        if isinstance(pipeline, dict):
            return bool(pipeline)
        return True

    def get_pipeline(self, pipeline_id: str) -> Any:
        """Get a loaded pipeline, or None if not loaded."""
        pid = _resolve_pipeline_id(pipeline_id)
        return self._pipelines.get(pid)

    @property
    def encoder(self) -> Any:
        """Get the Z-Image encoder (for encoder-only mode)."""
        return self._encoder

    @encoder.setter
    def encoder(self, value: Any) -> None:
        self._encoder = value

    @property
    def encoder_only_mode(self) -> bool:
        return self._encoder_only_mode

    @encoder_only_mode.setter
    def encoder_only_mode(self, value: bool) -> None:
        self._encoder_only_mode = value

    @property
    def ltx2_encoder(self) -> Any:
        """Get cached LTX-2 Gemma3 encoder (None if not loaded)."""
        return self._ltx2_encoder

    @property
    def ltx2_transformer_cache(self) -> Optional[dict]:
        """Get cached LTX-2 transformer data (None if not loaded).

        Returns dict with "config" (model config dict) and "state_dict"
        (bf16 pinned tensors) ready for fast model reconstruction.
        """
        return self._ltx2_transformer_cache

    @property
    def ltx2_vae(self) -> Any:
        """Get cached LTX-2 VAE decoder (None if not loaded)."""
        return self._ltx2_vae

    @property
    def ltx2_audio_decoder(self) -> Any:
        """Get cached LTX-2 audio decoder (None if not loaded)."""
        return self._ltx2_audio_decoder

    @property
    def ltx2_vocoder(self) -> Any:
        """Get cached LTX-2 vocoder (None if not loaded)."""
        return self._ltx2_vocoder

    def get_vram_status(self) -> dict:
        """Get current VRAM usage and loaded models status."""
        status = {
            "cuda_available": torch.cuda.is_available(),
            "models_loaded": {
                "zimage_pipeline": self.is_loaded("zimage"),
                "zimage_encoder": self._encoder is not None,
                "qwen_image_pipeline": self.is_loaded("qwen_image"),
                "qwen_image_edit": (
                    self.is_loaded("qwen_image")
                    and getattr(self._pipelines.get("qwen_image"), "edit_pipe", None) is not None
                ),
                "qwen_image_t2i_pipeline": self.is_loaded("qwen_image_t2i"),
                "ltx2_pipeline": self.is_loaded("ltx2"),
                "flux2_pipeline": self.is_loaded("flux2"),
            },
            "vram": None,
        }

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            status["vram"] = {
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "total_gb": round(total, 2),
                "free_gb": round(total - allocated, 2),
            }

        return status

    # -- pipeline-specific loading (private) --

    def _load_zimage(self) -> LoadResult:
        """Load Z-Image pipeline (encoder + DiT + VAE)."""
        config = self.config

        # Use zimage_model_path if set, fall back to legacy model_path
        model_path = config.zimage_model_path or config.model_path
        if not model_path:
            raise ValueError(
                "Z-Image model_path not configured. "
                "Set [zimage].model_path in config.toml"
            )

        logger.info(f"[Z-Image] Loading pipeline from {model_path}...")
        logger.info(f"  Variant: {config.zimage_variant}")

        # Determine encoder device - use CPU when cpu_offload is enabled
        encoder_device = config.encoder_device
        if getattr(config, "cpu_offload", False):
            encoder_device = "cpu"
            logger.info("  CPU offload enabled - placing encoder on CPU")

        start = time.time()
        from llm_dit.pipelines import ZImagePipeline

        zimage_pipeline = ZImagePipeline.from_pretrained(
            model_path,
            text_encoder_path=config.text_encoder_path,
            templates_dir=config.templates_dir,
            dtype=torch.bfloat16,
            encoder_device=encoder_device,
            dit_device=config.dit_device,
            vae_device=config.vae_device,
            quantization=config.quantization,
            attention_backend=getattr(config, "attention_backend", "auto"),
            use_custom_scheduler=True,
        )

        load_time = time.time() - start
        logger.info(f"  Pipeline loaded in {load_time:.1f}s")

        # Apply torch.compile
        if getattr(config, "compile", False):
            compile_mode = getattr(config, "compile_mode", "default")
            logger.info(f"  Compiling transformer (mode={compile_mode})...")
            try:
                zimage_pipeline.transformer = torch.compile(
                    zimage_pipeline.transformer, mode=compile_mode
                )
                logger.info("  Transformer compiled")
            except Exception as e:
                logger.warning(f"  Failed to compile: {e}")

        self._pipelines["zimage"] = zimage_pipeline
        self._encoder = getattr(zimage_pipeline, "encoder", None)
        self._encoder_only_mode = False

        logger.info("[Z-Image] Pipeline loaded successfully")
        return LoadResult(
            pipeline=zimage_pipeline,
            encoder=self._encoder,
            load_time=load_time,
            mode="full",
        )

    def reload_flux2(self, model_name: str) -> LoadResult:
        """Unload current FLUX.2 pipeline and reload with a different model.

        Thread-safe: acquires the flux2 lock.

        Args:
            model_name: The model variant to load (e.g., "klein-base-9b").

        Returns:
            LoadResult with newly loaded pipeline.
        """
        with self._locks["flux2"]:
            # Unload current
            self._unload_flux2()
            cleanup_memory("unload_flux2")

            # Load with the requested model_name
            self._loading_in_progress["flux2"] = True
            try:
                result = self._load_flux2(model_name_override=model_name)
                return result
            except Exception:
                self._pipelines.pop("flux2", None)
                cleanup_memory()
                raise
            finally:
                self._loading_in_progress["flux2"] = False

    def reload_zimage(self) -> LoadResult:
        """Unload Z-Image pipeline and reload fresh (clears fused LoRAs).

        Thread-safe: acquires the zimage lock.
        """
        with self._locks["zimage"]:
            self._unload_zimage()
            cleanup_memory("unload_zimage")

            self._loading_in_progress["zimage"] = True
            try:
                result = self._load_zimage()
                return result
            except Exception:
                self._pipelines.pop("zimage", None)
                cleanup_memory()
                raise
            finally:
                self._loading_in_progress["zimage"] = False

    def _load_flux2(self, model_name_override: Optional[str] = None) -> LoadResult:
        """Load FLUX.2 Klein pipeline with 3-stage loading.

        Stage 1: Load encoder, offload to CPU with pinned memory
        Stage 2: Load transformer (with optional quantization + compile)
        Stage 3: Load VAE (with optional compile)

        This pattern keeps peak VRAM manageable on 24GB GPUs by loading
        one component at a time and using pinned memory for fast DMA
        shuttle of the encoder between CPU and GPU.
        """
        config = self.config

        model_path = getattr(config, "flux2_model_path", None)
        if not model_path:
            raise ValueError(
                "FLUX.2 model_path not configured. "
                "Set flux2.model_path in config.toml"
            )

        model_name = model_name_override or getattr(config, "flux2_model_name", "klein-9b")
        block_offload = getattr(config, "flux2_block_offload", False)
        compile_transformer = getattr(config, "flux2_compile", False)
        compile_vae_flag = getattr(config, "flux2_compile_vae", False)
        compile_mode = getattr(config, "flux2_compile_mode", "max-autotune-no-cudagraphs")
        compile_dynamic = getattr(config, "flux2_compile_dynamic", False)
        encoder_path = getattr(config, "flux2_encoder_path", None)
        vae_path = getattr(config, "flux2_vae_path", None)

        # Resolve quantization from unified config
        quant_config = config.get_pipeline_quant_config("flux2")
        quantization = quant_config.transformer.method if quant_config else "none"

        # Validate incompatible settings before loading anything
        if compile_transformer and block_offload:
            raise ValueError(
                "compile=true is incompatible with block_offload=true. "
                "Set block_offload=false when using compile=true."
            )

        if compile_transformer and quantization == "none":
            logger.warning(
                "[FLUX.2] compile=true with quantization='none' may OOM on 24GB GPUs. "
                "FP8 models use fp8-cast (~4.5GB) by default. "
                "For bf16 models, set [flux2].quantization = 'fp8' to reduce VRAM."
            )

        logger.info(
            f"[FLUX.2] Loading pipeline from {model_path} "
            f"(quantization={quantization}, compile={compile_transformer})"
        )

        loaded_encoder = None
        loaded_transformer = None
        loaded_vae = None

        try:
            from llm_dit.encoders.qwen3_unified import Qwen3UnifiedEncoder
            from llm_dit.models.flux2.constants import FLUX2_MODEL_INFO, get_encoder_preset
            from llm_dit.models.flux2.loader import load_flux2_transformer, load_flux2_vae

            model_path_obj = Path(model_path).expanduser()
            if not model_path_obj.exists():
                raise ValueError(f"FLUX.2 model path does not exist: {model_path}")

            # Stage 1: Load encoder
            model_info = FLUX2_MODEL_INFO.get(model_name.lower(), {})
            preset = get_encoder_preset(model_name)
            text_encoder_spec = encoder_path or model_info.get("text_encoder", "Qwen/Qwen3-8B")

            logger.info(f"[FLUX.2] Stage 1: Loading encoder from {text_encoder_spec}")
            start = time.time()
            loaded_encoder = Qwen3UnifiedEncoder.from_preset(
                preset, model_path=text_encoder_spec, device="cuda"
            )
            # Offload to CPU with pinned memory for fast GPU shuttle via DMA
            loaded_encoder.offload_to_pinned()
            logger.info("[FLUX.2] Encoder loaded and offloaded to CPU with pinned memory")

            # Stage 2: Load transformer
            logger.info(f"[FLUX.2] Stage 2: Loading transformer ({model_name})")
            loaded_transformer = load_flux2_transformer(
                model_name,
                device="cuda",
                model_path=model_path,
                block_offload=block_offload,
                quantize_to=quantization,
            )

            if compile_transformer and not block_offload:
                # dynamic=True generates shape-generic kernels that handle
                # varying sequence lengths without retracing (~90s savings per
                # new resolution). When dynamic, use fullgraph=False as a safety
                # measure for data-dependent branches.
                use_fullgraph = not compile_dynamic
                logger.info(
                    f"[FLUX.2] Wrapping transformer with torch.compile "
                    f"(mode={compile_mode}, fullgraph={use_fullgraph}, dynamic={compile_dynamic}) "
                    "-- actual compilation happens on first forward pass"
                )
                loaded_transformer = torch.compile(
                    loaded_transformer,
                    mode=compile_mode,
                    fullgraph=use_fullgraph,
                    dynamic=compile_dynamic,
                )

            # Stage 3: Load VAE
            logger.info("[FLUX.2] Stage 3: Loading VAE")
            loaded_vae = load_flux2_vae(model_name, device="cuda", vae_path=vae_path)

            if compile_vae_flag:
                logger.info(
                    f"[FLUX.2] Wrapping VAE decoder with torch.compile (mode={compile_mode}, fullgraph=True) "
                    "-- actual compilation happens on first decode"
                )
                loaded_vae.decode = torch.compile(
                    loaded_vae.decode, mode=compile_mode, fullgraph=True,
                )

            # Offload VAE to CPU with pinned memory -- shuttle to GPU only when needed
            loaded_vae.offload_to_pinned()

            load_time = time.time() - start

            # Store persistent model references (only after all three succeed)
            from llm_dit.models.flux2.constants import supports_kv_cache as _supports_kv
            self._pipelines["flux2"] = {
                "encoder": loaded_encoder,
                "transformer": loaded_transformer,
                "vae": loaded_vae,
                "model_name": model_name,
                "use_kv_cache": _supports_kv(model_name),
            }

            logger.info(f"[FLUX.2] Pipeline loaded in {load_time:.1f}s")
            return LoadResult(
                pipeline=self._pipelines["flux2"],
                load_time=load_time,
                mode="flux2_full",
            )
        except Exception:
            # Clean up any partially loaded models
            del loaded_encoder, loaded_transformer, loaded_vae
            cleanup_memory("load_error")
            raise

    def _preload_ltx2_transformer(self, model_path: Path, ltx2_cfg: Any) -> dict:
        """Load transformer weights from disk and cache as pinned tensors.

        Loads the full transformer once (handling both regular and FP8 checkpoint
        formats), extracts the state dict, pins all tensors for fast DMA
        transfer, then discards the model object.

        For FP8 files, uses fp8-cast (official approach): keeps fp8 weights as-is,
        patches forwards on reconstruction. Cache dict includes 'fp8_cast' flag.

        Returns:
            Dict with "config" (model config dict for create_model_from_config),
            "state_dict" (pinned tensors ready for load_state_dict),
            "video_only" (bool), and "fp8_cast" (bool).
        """
        transformer_file = ltx2_cfg.transformer_file if ltx2_cfg else ""

        # Resolve transformer path
        if transformer_file:
            tf_path = model_path / transformer_file
            if not tf_path.exists():
                logger.warning(
                    f"transformer_file '{transformer_file}' not found at {tf_path}, "
                    "falling back to transformer/ directory"
                )
                tf_path = model_path / "transformer"
        else:
            tf_path = model_path / "transformer"

        # Load model to CPU (handles key mapping, etc.)
        # When audio_enabled, load full AV model (video_only=False) so audio weights
        # are included in the cached state dict.
        audio_enabled = ltx2_cfg.audio_enabled if ltx2_cfg else False
        video_only = not audio_enabled
        is_fp8_file = tf_path.is_file() and "fp8" in tf_path.name.lower()
        if is_fp8_file:
            # FP8-cast: keep fp8 weights as-is, patch forwards on reconstruct
            from llm_dit.models.ltx2 import load_ltx2_transformer_fp8_cast
            model = load_ltx2_transformer_fp8_cast(
                tf_path, dtype=torch.bfloat16, device="cpu", video_only=video_only
            )
        else:
            from llm_dit.models.ltx2 import load_ltx2_transformer
            model = load_ltx2_transformer(
                tf_path, dtype=torch.bfloat16, device="cpu", video_only=video_only
            )

        # Load config for model reconstruction
        from llm_dit.models.ltx2.loader import load_config
        config = load_config(tf_path)

        # Extract weight_scales from model (plain attrs on nn.Linear, not in state_dict)
        weight_scales: dict[str, torch.Tensor] = {}
        if is_fp8_file:
            for name, module in model.named_modules():
                if hasattr(module, "_weight_scale"):
                    weight_scales[f"{name}.weight"] = module._weight_scale.pin_memory()
            if weight_scales:
                logger.info(f"  Cached {len(weight_scales)} weight_scale tensors for FP8 dequantization")

        # Extract state dict and pin memory for fast DMA transfers
        sd = {}
        for k, v in model.state_dict().items():
            sd[k] = v.pin_memory()

        param_count = sum(v.numel() for v in sd.values())
        mem_gb = sum(v.nbytes for v in sd.values()) / 1e9
        logger.info(
            f"  Cached {len(sd)} tensors "
            f"({param_count / 1e9:.2f}B params, {mem_gb:.1f}GB pinned)"
        )

        del model
        return {
            "config": config, "state_dict": sd, "video_only": video_only,
            "fp8_cast": is_fp8_file, "weight_scales": weight_scales,
        }

    def _preload_ltx2_vae(self, model_path: Path) -> Any:
        """Load VAE decoder and cache on CPU with pinned memory.

        The VAE is small (~2GB) with no quantization or LoRA mutation,
        so we cache the full model object and shuttle it to GPU per generation.
        """
        from llm_dit.models.ltx2.vae import load_ltx2_vae_decoder

        # V2.3: standalone file at model root; V1: vae/ subdirectory
        v23_path = model_path / "ltx-2.3-video-vae.safetensors"
        vae_path = v23_path if v23_path.exists() else model_path / "vae"

        vae = load_ltx2_vae_decoder(
            vae_path, dtype=torch.bfloat16, device="cpu"
        )
        vae.offload_to_pinned()
        return vae

    def _preload_ltx2_audio_decoder(self, model_path: Path, audio_vae_path: str = "") -> Any:
        """Load audio decoder and cache on CPU with pinned memory (~102MB)."""
        from llm_dit.models.ltx2.audio_vae.loader import load_audio_decoder

        if audio_vae_path:
            path = Path(audio_vae_path)
        else:
            # V2.3: standalone file at model root; V1: audio_vae/ subdirectory
            v23_path = model_path / "ltx-2.3-audio-vae.safetensors"
            path = v23_path if v23_path.exists() else model_path / "audio_vae"
        decoder = load_audio_decoder(path, dtype=torch.bfloat16, device="cpu")
        decoder.offload_to_pinned()
        return decoder

    def _preload_ltx2_vocoder(self, model_path: Path, vocoder_path: str = "") -> Any:
        """Load vocoder and cache on CPU with pinned memory (~107MB)."""
        from llm_dit.models.ltx2.audio_vae.loader import load_vocoder

        if vocoder_path:
            path = Path(vocoder_path)
        else:
            # V2.3: standalone file at model root; V1: vocoder/ subdirectory
            v23_path = model_path / "ltx-2.3-vocoder.safetensors"
            path = v23_path if v23_path.exists() else model_path / "vocoder"
        vocoder = load_vocoder(path, dtype=torch.bfloat16, device="cpu")
        vocoder.offload_to_pinned()
        return vocoder

    def _load_ltx2(self) -> LoadResult:
        """Validate LTX-2 configuration and pre-load Gemma3 encoder.

        Checks model_path exists and required files are present, then
        pre-loads the Gemma3 encoder and offloads to pinned CPU memory
        for fast GPU shuttle on each generation.
        """
        ltx2_cfg = self.config.ltx2 if self.config else None
        model_path_str = ltx2_cfg.model_path if ltx2_cfg else ""
        if not model_path_str:
            raise ValueError("LTX-2 not configured. Set ltx2.model_path in config.toml")

        model_path = Path(model_path_str).expanduser()
        if not model_path.exists():
            raise ValueError(f"LTX-2 model path not found: {model_path}")

        # Validate required directories/files for two-stage pipeline
        missing = []
        transformer_file = ltx2_cfg.transformer_file if ltx2_cfg else ""
        if transformer_file and not (model_path / transformer_file).exists():
            if not (model_path / "transformer").exists():
                missing.append(f"{transformer_file} (or transformer/)")
        elif not transformer_file and not (model_path / "transformer").exists():
            missing.append("transformer/")
        for subdir in ["text_encoder", "vae"]:
            if not (model_path / subdir).exists():
                missing.append(f"{subdir}/")

        upsampler_file = ltx2_cfg.spatial_upsampler_file if ltx2_cfg else ""
        if upsampler_file and not (model_path / upsampler_file).exists():
            missing.append(f"upsampler: {upsampler_file}")

        if missing:
            logger.warning(f"[LTX-2] Missing files (may cause errors): {', '.join(missing)}")

        # Pre-load and cache Gemma3 encoder for persistence between generations
        start = time.time()
        gemma_variant = ltx2_cfg.gemma_variant if ltx2_cfg else "bf16"
        text_encoder_path = str(model_path / "text_encoder")
        encoder_model_id = ltx2_cfg.encoder_model_id if ltx2_cfg else None
        if encoder_model_id:
            text_encoder_path = encoder_model_id

        max_seq_len = ltx2_cfg.max_sequence_length if ltx2_cfg else 512
        logger.info(f"[LTX-2] Pre-loading Gemma3 encoder (variant={gemma_variant}, max_seq_len={max_seq_len})...")

        connectors_file = ltx2_cfg.connectors_file if ltx2_cfg else "ltx-2.3-connectors.safetensors"
        if gemma_variant != "bf16":
            from llm_dit.encoders.gemma3_variants import create_gemma3_encoder
            self._ltx2_encoder = create_gemma3_encoder(
                variant=gemma_variant,
                model_path=str(model_path),
                text_encoder_path=text_encoder_path,
                device="cpu",  # Load to CPU, shuttle to GPU per-request
                dtype=torch.bfloat16,
                max_sequence_length=max_seq_len,
                connectors_file=connectors_file,
            )
        else:
            from llm_dit.encoders.gemma3 import Gemma3Encoder
            connectors_path = str(model_path / connectors_file)
            self._ltx2_encoder = Gemma3Encoder(
                model_id=text_encoder_path,
                device="cpu",
                dtype=torch.bfloat16,
                max_sequence_length=max_seq_len,
                connectors_path=connectors_path,
            )
            self._ltx2_encoder._load_model()

        # Offload to pinned memory for fast GPU shuttle
        self._ltx2_encoder.offload_to_pinned()
        encoder_time = time.time() - start
        logger.info(f"[LTX-2] Encoder pre-loaded and pinned in {encoder_time:.1f}s")

        # Pre-load transformer weights into cache dict (pinned memory for fast DMA)
        tf_start = time.time()
        logger.info("[LTX-2] Pre-loading transformer weights for caching...")
        self._ltx2_transformer_cache = self._preload_ltx2_transformer(model_path, ltx2_cfg)
        tf_time = time.time() - tf_start
        logger.info(f"[LTX-2] Transformer cached in {tf_time:.1f}s")

        # Pre-load VAE decoder (small, pinned for shuttle)
        logger.info("[LTX-2] Pre-loading VAE decoder for caching...")
        vae_start = time.time()
        self._ltx2_vae = self._preload_ltx2_vae(model_path)
        vae_time = time.time() - vae_start
        logger.info(f"[LTX-2] VAE cached in {vae_time:.1f}s")

        # Pre-load audio models when audio_enabled (optional, ~209MB total)
        audio_time = 0.0
        if ltx2_cfg and ltx2_cfg.audio_enabled:
            # Resolve audio paths: V2.3 standalone files or V1 subdirectories
            if ltx2_cfg.audio_vae_path:
                audio_vae_resolved = Path(ltx2_cfg.audio_vae_path)
            else:
                v23_audio = model_path / "ltx-2.3-audio-vae.safetensors"
                audio_vae_resolved = v23_audio if v23_audio.exists() else model_path / "audio_vae"
            if ltx2_cfg.vocoder_path:
                vocoder_resolved = Path(ltx2_cfg.vocoder_path)
            else:
                v23_vocoder = model_path / "ltx-2.3-vocoder.safetensors"
                vocoder_resolved = v23_vocoder if v23_vocoder.exists() else model_path / "vocoder"

            if audio_vae_resolved.exists() and vocoder_resolved.exists():
                logger.info("[LTX-2] Pre-loading audio models for caching...")
                audio_start = time.time()
                self._ltx2_audio_decoder = self._preload_ltx2_audio_decoder(
                    model_path, ltx2_cfg.audio_vae_path,
                )
                self._ltx2_vocoder = self._preload_ltx2_vocoder(
                    model_path, ltx2_cfg.vocoder_path,
                )
                audio_time = time.time() - audio_start
                logger.info(f"[LTX-2] Audio models cached in {audio_time:.1f}s")
            else:
                logger.warning(
                    "[LTX-2] audio_enabled=True but audio models not found. "
                    f"audio_vae={audio_vae_resolved.exists()}, vocoder={vocoder_resolved.exists()}. "
                    "Audio unavailable."
                )

        total_time = time.time() - start
        logger.info(
            f"[LTX-2] All components pre-loaded in {total_time:.1f}s "
            f"(encoder={encoder_time:.1f}s, transformer={tf_time:.1f}s, vae={vae_time:.1f}s"
            f"{f', audio={audio_time:.1f}s' if audio_time > 0 else ''})"
        )

        # Store config dict as sentinel (not None) so is_loaded() returns True
        self._pipelines["ltx2"] = {"model_path": str(model_path)}
        logger.info(f"[LTX-2] Two-stage pipeline validated: {model_path}")
        return LoadResult(mode="ltx2_validated")

    def _load_qwen_image(self) -> LoadResult:
        """Load Qwen-Image pipeline (edit-only mode for on-demand loading)."""
        config = self.config

        if not config.qwen_image_model_path:
            raise ValueError(
                "Qwen-Image model_path not configured. "
                "Set qwen_image.model_path in config.toml"
            )

        from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

        quant_te = getattr(config, "qwen_image_quantize_text_encoder", "none")
        quant_tf = getattr(config, "qwen_image_quantize_transformer", "none")
        quant_te = quant_te if quant_te != "none" else None
        quant_tf = quant_tf if quant_tf != "none" else None

        logger.info("[Qwen-Image] Loading pipeline in edit-only mode...")
        start = time.time()

        qwen_pipeline = QwenImageDiffusersPipeline.from_pretrained(
            config.qwen_image_model_path,
            edit_model_path=config.qwen_image_edit_model_path or None,
            cpu_offload=True,
            edit_only=True,
            quantize_text_encoder=quant_te,
            quantize_transformer=quant_tf,
        )

        load_time = time.time() - start
        self._pipelines["qwen_image"] = qwen_pipeline
        logger.info(f"[Qwen-Image] Edit pipeline loaded in {load_time:.1f}s")

        return LoadResult(
            pipeline=qwen_pipeline,
            load_time=load_time,
            mode="qwen_image_edit",
        )

    def _load_qwen_image_t2i(self) -> LoadResult:
        """Load Qwen-Image T2I pipeline (text-to-image)."""
        config = self.config

        if not config.qwen_image_model_path:
            raise ValueError(
                "Qwen-Image model_path not configured. "
                "Set qwen_image.model_path in config.toml"
            )

        from llm_dit.pipelines.qwen_image_2512 import QwenImage2512Pipeline

        quant_transformer = config.get_qwen_image_quantize_transformer()
        quant_text_encoder = config.qwen_image_quantize_text_encoder
        quant_text_encoder_opt = quant_text_encoder if quant_text_encoder != "none" else None

        logger.info("[Qwen-Image T2I] Loading pipeline...")
        start = time.time()

        t2i_pipeline = QwenImage2512Pipeline.from_pretrained(
            config.qwen_image_model_path,
            quantize_transformer=quant_transformer,
            quantize_text_encoder=quant_text_encoder_opt,
            cpu_offload=config.qwen_image_cpu_offload,
        )

        load_time = time.time() - start
        self._pipelines["qwen_image_t2i"] = t2i_pipeline
        logger.info(f"[Qwen-Image T2I] Pipeline loaded in {load_time:.1f}s")

        return LoadResult(
            pipeline=t2i_pipeline,
            load_time=load_time,
            mode="qwen_image_t2i",
        )

    # -- shared optimization helpers --

    def _apply_optimizations(self, pipeline: Any) -> None:
        """Apply flash attention, compile, attention backend to a pipeline."""
        config = self.config

        # Flash Attention
        if config.flash_attn:
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
        if config.attention_backend and config.attention_backend != "auto":
            try:
                from llm_dit.utils.attention import (
                    get_available_backends,
                    set_attention_backend,
                )

                available = get_available_backends()
                if config.attention_backend in available:
                    set_attention_backend(config.attention_backend)
                    logger.info(f"Global attention backend: {config.attention_backend}")
                else:
                    logger.warning(
                        f"Requested attention backend '{config.attention_backend}' not available. "
                        f"Available: {available}. Using auto-detection."
                    )
            except Exception as e:
                logger.warning(f"Failed to set global attention backend: {e}")

            # Also set diffusers-specific attention backend
            backend_map = {
                "sdpa": "native",
                "flash_attn_2": "flash",
                "flash_attn_3": "flash",
                "xformers": "xformers",
                "sage": "sage",
            }
            diffusers_backend = backend_map.get(
                config.attention_backend, config.attention_backend
            )
            try:
                if hasattr(pipeline.transformer, "set_attention_backend"):
                    pipeline.transformer.set_attention_backend(diffusers_backend)
                    logger.info(f"  Diffusers attention backend: {diffusers_backend}")
            except Exception as e:
                logger.warning(f"  Failed to set diffusers attention backend: {e}")

        # torch.compile
        if config.compile:
            compile_mode = getattr(config, "compile_mode", "default")
            logger.info(f"Compiling transformer (mode={compile_mode})...")
            try:
                pipeline.transformer = torch.compile(
                    pipeline.transformer, mode=compile_mode
                )
                logger.info("  Transformer compiled")
            except Exception as e:
                logger.warning(f"  Failed to compile: {e}")

        # Tiled VAE
        if config.tiled_vae:
            try:
                if hasattr(pipeline, "enable_tiled_vae"):
                    pipeline.enable_tiled_vae(
                        tile_size=config.tile_size,
                        tile_overlap=config.tile_overlap,
                    )
                    logger.info("  Tiled VAE enabled")
                elif hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_tiling"):
                    pipeline.vae.enable_tiling()
                    logger.info("  VAE tiling enabled (basic)")
            except Exception as e:
                logger.warning(f"  Failed to enable tiled VAE: {e}")

    # -- private helpers --

    def _unload_pipeline(self, pid: str) -> bool:
        """Unload a specific pipeline and free its VRAM."""
        if pid == "zimage":
            return self._unload_zimage()
        elif pid == "flux2":
            return self._unload_flux2()
        elif pid == "ltx2":
            return self._unload_ltx2()
        elif pid == "qwen_image":
            return self._unload_qwen_image()
        elif pid == "qwen_image_t2i":
            return self._unload_qwen_image_t2i()
        return False

    def _unload_zimage(self) -> bool:
        """Unload Z-Image pipeline with proper VRAM cleanup."""
        zimage = self._pipelines.get("zimage")
        unloaded = False

        if zimage is not None:
            logger.info("[VRAM] Unloading Z-Image pipeline...")
            try:
                if hasattr(zimage, "transformer") and zimage.transformer is not None:
                    zimage.transformer.to("cpu")
                if hasattr(zimage, "vae") and zimage.vae is not None:
                    zimage.vae.to("cpu")
            except Exception as e:
                logger.warning(f"[VRAM] Error moving Z-Image to CPU: {e}")
            del zimage
            self._pipelines.pop("zimage", None)
            unloaded = True

        if self._encoder is not None:
            logger.info("[VRAM] Unloading Z-Image encoder...")
            try:
                if hasattr(self._encoder, "backend") and self._encoder.backend is not None:
                    if hasattr(self._encoder.backend, "model") and self._encoder.backend.model is not None:
                        self._encoder.backend.model.to("cpu")
            except Exception as e:
                logger.warning(f"[VRAM] Error moving encoder to CPU: {e}")
            del self._encoder
            self._encoder = None
            unloaded = True

        if unloaded:
            # Clear torch.compile cache
            try:
                import torch._dynamo
                torch._dynamo.reset()
                logger.info("[VRAM] Cleared torch.compile cache")
            except Exception as e:
                logger.warning(f"[VRAM] Could not clear compile cache: {e}")

            cleanup_memory("unload_zimage")

        return unloaded

    def _unload_flux2(self) -> bool:
        """Unload FLUX.2 pipeline (dict of encoder/transformer/vae)."""
        flux2 = self._pipelines.get("flux2")
        if flux2 is None:
            return False

        if isinstance(flux2, dict):
            for key in list(flux2.keys()):
                del flux2[key]
        self._pipelines.pop("flux2", None)

        # Clear torch.compile cache (compiled kernels hold CUDA memory)
        try:
            import torch._dynamo
            torch._dynamo.reset()
            logger.info("[FLUX.2] Cleared torch.compile cache")
        except Exception as e:
            logger.warning(f"[FLUX.2] Could not clear compile cache: {e}")

        cleanup_memory("unload_flux2")
        return True

    def _unload_ltx2(self) -> bool:
        """Clean up VRAM and cached components after LTX-2 operations."""
        has_anything = (
            "ltx2" in self._pipelines
            or self._ltx2_encoder is not None
            or self._ltx2_transformer_cache is not None
            or self._ltx2_vae is not None
            or self._ltx2_audio_decoder is not None
            or self._ltx2_vocoder is not None
        )
        if not has_anything:
            return False

        logger.info("[VRAM] Running LTX-2 memory cleanup...")

        # Release cached encoder
        if self._ltx2_encoder is not None:
            del self._ltx2_encoder
            self._ltx2_encoder = None
            logger.info("[VRAM] LTX-2 encoder cache released")

        # Release cached transformer state dict
        if self._ltx2_transformer_cache is not None:
            del self._ltx2_transformer_cache
            self._ltx2_transformer_cache = None
            logger.info("[VRAM] LTX-2 transformer cache released")

        # Release cached VAE
        if self._ltx2_vae is not None:
            del self._ltx2_vae
            self._ltx2_vae = None
            logger.info("[VRAM] LTX-2 VAE cache released")

        # Release cached audio decoder
        if self._ltx2_audio_decoder is not None:
            del self._ltx2_audio_decoder
            self._ltx2_audio_decoder = None
            logger.info("[VRAM] LTX-2 audio decoder cache released")

        # Release cached vocoder
        if self._ltx2_vocoder is not None:
            del self._ltx2_vocoder
            self._ltx2_vocoder = None
            logger.info("[VRAM] LTX-2 vocoder cache released")

        self._pipelines.pop("ltx2", None)
        cleanup_memory("unload_ltx2")
        return True

    def _unload_qwen_image(self) -> bool:
        """Unload Qwen-Image pipeline."""
        qwen = self._pipelines.get("qwen_image")
        if qwen is None:
            return False

        logger.info("[VRAM] Unloading Qwen-Image pipeline...")
        del qwen
        self._pipelines.pop("qwen_image", None)
        cleanup_memory("unload_qwen_image")
        return True

    def _unload_qwen_image_t2i(self) -> bool:
        """Unload Qwen-Image T2I pipeline."""
        t2i = self._pipelines.get("qwen_image_t2i")
        if t2i is None:
            return False

        logger.info("[VRAM] Unloading Qwen-Image T2I pipeline...")
        del t2i
        self._pipelines.pop("qwen_image_t2i", None)
        cleanup_memory("unload_qwen_image_t2i")
        return True

    def _unload_generic(self, pid: str, display_name: str) -> bool:
        """Generic unload for pipelines without special cleanup needs."""
        pipeline = self._pipelines.get(pid)
        if pipeline is None:
            return False

        logger.info(f"[VRAM] Unloading {display_name} pipeline...")
        del pipeline
        self._pipelines.pop(pid, None)
        cleanup_memory(f"unload_{pid}")
        return True

    # -- PipelineLoader compatibility (used by scripts/generate.py) --

    def load_pipeline_from_loader(
        self,
        encoder_only: bool = False,
        use_api: bool = False,
    ) -> LoadResult:
        """Load pipeline using PipelineLoader-compatible logic.

        This preserves the auto_load() behavior from the old PipelineLoader
        class, used by scripts/generate.py and experiments.

        Args:
            encoder_only: If True, only load encoder
            use_api: If True, prefer API backend when api_url is set

        Returns:
            LoadResult with loaded components
        """
        config = self.config
        model_type = getattr(config, "model_type", "zimage")

        # Route to correct pipeline based on model_type
        if model_type == "qwenimage-t2i":
            logger.info("[Qwen-Image T2I] On-demand mode")
            return LoadResult(mode="qwenimage-t2i_ondemand")
        elif model_type == "qwenimage-edit":
            logger.info("[Qwen-Image Edit] On-demand mode")
            return LoadResult(mode="qwenimage-edit_ondemand")
        elif model_type == "ltx2":
            logger.info("[LTX-2] On-demand mode (loaded via /api/ltx2/generate)")
            return LoadResult(pipeline=None, encoder=None, mode="ltx2_ondemand")
        # Z-Image paths
        edit_only = getattr(config, "qwen_image_edit_only", False)
        has_edit_model = bool(getattr(config, "qwen_image_edit_model_path", ""))

        # API encoder only
        if (
            config.api_url
            and not config.model_path
            and not (edit_only and has_edit_model)
        ):
            return self._load_api_encoder()

        # Distributed mode (API encoder + local DiT/VAE)
        if config.api_url and config.model_path and use_api:
            return self._load_api_pipeline()

        # Encoder only
        if encoder_only:
            return self._load_encoder_only()

        # Full Z-Image pipeline
        return self._load_zimage_full()

    def _load_zimage_full(self) -> LoadResult:
        """Full Z-Image pipeline load (PipelineLoader compatibility)."""
        from llm_dit.pipelines import ZImagePipeline

        config = self.config
        templates_dir = self._resolve_templates_dir()

        logger.info("=" * 60)
        logger.info("LOADING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"  Model: {config.model_path}")
        logger.info(f"  Encoder device: {config.encoder_device_resolved}")
        logger.info(f"  DiT device: {config.dit_device_resolved}")
        logger.info(f"  VAE device: {config.vae_device_resolved}")
        logger.info(f"  Dtype: {config.dtype}")
        logger.info(f"  Quantization: {config.quantization}")

        dype_config = build_dype_config(config)
        start = time.time()

        zimage_pipeline = ZImagePipeline.from_pretrained(
            config.model_path,
            text_encoder_path=config.text_encoder_path,
            templates_dir=templates_dir,
            dtype=config.get_dtype(),
            encoder_device=config.encoder_device_resolved,
            dit_device=config.dit_device_resolved,
            vae_device=config.vae_device_resolved,
            quantization=config.quantization,
            enable_cache=config.embedding_cache,
            cache_size=config.cache_size,
            dype_config=dype_config,
        )

        load_time = time.time() - start
        logger.info(f"Pipeline loaded in {load_time:.1f}s")

        self._apply_optimizations(zimage_pipeline)

        self._pipelines["zimage"] = zimage_pipeline
        self._encoder = zimage_pipeline.encoder

        return LoadResult(
            pipeline=zimage_pipeline,
            encoder=self._encoder,
            load_time=load_time,
            mode="full",
            encoder_device=str(zimage_pipeline.encoder.device)
            if zimage_pipeline.encoder is not None
            else None,
            dit_device=str(next(zimage_pipeline.transformer.parameters()).device)
            if zimage_pipeline.transformer is not None
            else None,
            vae_device=str(next(zimage_pipeline.vae.parameters()).device)
            if zimage_pipeline.vae is not None
            else None,
        )

    def _load_encoder_only(self) -> LoadResult:
        """Load only the text encoder (PipelineLoader compatibility)."""
        from llm_dit.encoders import ZImageTextEncoder

        config = self.config
        templates_dir = self._resolve_templates_dir()

        logger.info("=" * 60)
        logger.info("LOADING ENCODER")
        logger.info("=" * 60)

        start = time.time()

        self._encoder = ZImageTextEncoder.from_pretrained(
            config.model_path,
            templates_dir=templates_dir,
            device_map=config.encoder_device_resolved,
            dtype=config.get_dtype(),
            quantization=config.quantization,
            enable_cache=config.embedding_cache,
            cache_size=config.cache_size,
        )

        load_time = time.time() - start
        self._encoder_only_mode = True
        logger.info(f"Encoder loaded in {load_time:.1f}s")

        return LoadResult(
            encoder=self._encoder,
            load_time=load_time,
            mode="encoder_only",
            encoder_device=str(self._encoder.device),
        )

    def _load_api_encoder(self) -> LoadResult:
        """Load encoder using remote API backend (PipelineLoader compatibility)."""
        from llm_dit.backends.api import APIBackend, APIBackendConfig
        from llm_dit.encoders import ZImageTextEncoder
        from llm_dit.templates import TemplateRegistry

        config = self.config
        templates_dir = self._resolve_templates_dir()

        logger.info("=" * 60)
        logger.info("LOADING API ENCODER")
        logger.info("=" * 60)

        start = time.time()

        api_config = APIBackendConfig(
            base_url=config.api_url,
            model_id=config.api_model,
            encoding_format="base64",
            hidden_layer=config.hidden_layer,
        )
        backend = APIBackend(api_config)

        templates = None
        if templates_dir:
            templates = TemplateRegistry.from_directory(templates_dir)

        self._encoder = ZImageTextEncoder(backend=backend, templates=templates)
        load_time = time.time() - start
        self._encoder_only_mode = True

        logger.info(f"API encoder ready in {load_time:.1f}s")
        return LoadResult(
            encoder=self._encoder,
            load_time=load_time,
            mode="api_encoder",
        )

    def _load_api_pipeline(self) -> LoadResult:
        """Load pipeline with API encoder + local DiT/VAE (PipelineLoader compatibility)."""
        from llm_dit.backends.api import APIBackend, APIBackendConfig
        from llm_dit.encoders import ZImageTextEncoder
        from llm_dit.pipelines import ZImagePipeline
        from llm_dit.templates import TemplateRegistry

        config = self.config
        templates_dir = self._resolve_templates_dir()

        logger.info("=" * 60)
        logger.info("LOADING DISTRIBUTED PIPELINE")
        logger.info("=" * 60)

        start = time.time()

        api_config = APIBackendConfig(
            base_url=config.api_url,
            model_id=config.api_model,
            encoding_format="base64",
            hidden_layer=config.hidden_layer,
        )
        backend = APIBackend(api_config)

        templates = None
        if templates_dir:
            templates = TemplateRegistry.from_directory(templates_dir)

        api_encoder = ZImageTextEncoder(backend=backend, templates=templates)

        dype_config = build_dype_config(config)

        zimage_pipeline = ZImagePipeline.from_pretrained_generator_only(
            config.model_path,
            dtype=config.get_dtype(),
            enable_cpu_offload=config.cpu_offload,
            dit_device=config.dit_device_resolved,
            vae_device=config.vae_device_resolved,
            dype_config=dype_config,
        )

        self._apply_optimizations(zimage_pipeline)
        zimage_pipeline.encoder = api_encoder
        self._encoder = api_encoder
        self._pipelines["zimage"] = zimage_pipeline

        load_time = time.time() - start

        logger.info(f"Distributed pipeline ready in {load_time:.1f}s")
        return LoadResult(
            pipeline=zimage_pipeline,
            encoder=api_encoder,
            load_time=load_time,
            mode="distributed",
            dit_device=str(next(zimage_pipeline.transformer.parameters()).device)
            if zimage_pipeline.transformer
            else None,
            vae_device=str(next(zimage_pipeline.vae.parameters()).device)
            if zimage_pipeline.vae
            else None,
        )

    def _load_loras(self, pipeline: Any) -> None:
        """No-op. LoRAs are loaded per-request, not at startup.

        Kept for PipelineLoader backward compatibility.
        """

    def _resolve_templates_dir(self) -> Optional[str]:
        """Find templates directory."""
        config = self.config
        if config.templates_dir:
            return config.templates_dir

        candidates = [
            Path.cwd() / "templates" / "z_image",
            Path(__file__).parent.parent / "templates" / "z_image",
        ]
        for path in candidates:
            if path.exists():
                return str(path)
        return None


# -- Backward compatibility shim --
# startup.py's PipelineLoader is replaced by ModelManager.
# This shim maintains import compatibility for scripts.

class PipelineLoader:
    """Backward-compatible wrapper around ModelManager.

    Preserves the original PipelineLoader interface, including internal
    attributes (_pipeline, _encoder) and methods (_apply_optimizations,
    _load_loras, _resolve_templates_dir) that existing tests depend on.

    Deprecated: use ModelManager directly for new code.
    """

    def __init__(self, config: "RuntimeConfig"):
        self._manager = ModelManager(config)
        self.config = config
        # Expose internal state for test compatibility
        self._pipeline = None
        self._encoder = None

    @property
    def pipeline(self):
        return self._pipeline or self._manager.get_pipeline("zimage")

    @property
    def encoder(self):
        return self._encoder or self._manager.encoder

    def _resolve_templates_dir(self) -> Optional[str]:
        """Delegate to ModelManager."""
        return self._manager._resolve_templates_dir()

    def _apply_optimizations(self, pipeline: Any) -> None:
        """Delegate to ModelManager."""
        return self._manager._apply_optimizations(pipeline)

    def _load_loras(self, pipeline: Any) -> None:
        """Delegate to ModelManager."""
        return self._manager._load_loras(pipeline)

    def load_encoder(self) -> LoadResult:
        result = self._manager._load_encoder_only()
        self._encoder = result.encoder
        return result

    def load_pipeline(self) -> LoadResult:
        result = self._manager.load_pipeline_from_loader()
        self._pipeline = result.pipeline
        self._encoder = result.encoder
        return result

    def load_api_encoder(self) -> LoadResult:
        result = self._manager._load_api_encoder()
        self._encoder = result.encoder
        return result

    def load_api_pipeline(self) -> LoadResult:
        result = self._manager._load_api_pipeline()
        self._pipeline = result.pipeline
        self._encoder = result.encoder
        return result

    def auto_load(
        self,
        encoder_only: bool = False,
        use_api: bool = False,
    ) -> LoadResult:
        """Auto-load the appropriate pipeline based on config.

        Routes through PipelineLoader's own methods (load_encoder,
        load_pipeline, load_api_encoder, load_api_pipeline) so that
        tests can mock individual methods on the instance.
        """
        config = self.config
        model_type = getattr(config, "model_type", "zimage")

        # Non-zimage model types go through load_pipeline
        if model_type not in ("zimage", ""):
            return self.load_pipeline()

        # Z-Image routing
        edit_only = getattr(config, "qwen_image_edit_only", False)
        has_edit_model = bool(getattr(config, "qwen_image_edit_model_path", ""))

        # API encoder only
        if (
            config.api_url
            and not config.model_path
            and not (edit_only and has_edit_model)
        ):
            return self.load_api_encoder()

        # Distributed mode
        if config.api_url and config.model_path and use_api:
            return self.load_api_pipeline()

        # Encoder only
        if encoder_only:
            return self.load_encoder()

        # Full pipeline
        return self.load_pipeline()
