"""
Wan Video Pipeline for text-to-video generation.

Last Updated: 2026-01-13

**Wan T2V**:
    pipe = WanVideoPipeline.from_wan_pretrained("~/Storage/Wan2.1-T2V-1.3B")
    video = pipe(prompt="A cat sleeping on a windowsill")

**HuMo (audio-conditioned)**:
    pipe = WanVideoPipeline.from_pretrained(
        humo_path="~/Storage/HuMo",
        wan_path="~/Storage/Wan2.1-T2V-1.3B",
    )
    video = pipe(prompt="A person dancing", audio="music.wav")

Architectures:
- Wan DiT: Text-to-video transformer (1.3B or 14B)
- HuMo: Wan + audio cross-attention (17B or 1.7B)
- VAE: Wan 2.1 video VAE (16 latent channels)
- Text encoder: UMT5-XXL (4096 dim)
- Audio encoder: Whisper-large-v3 (lazy-loaded for HuMo)
"""

import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class VideoOutput:
    """Output from Wan video generation."""

    frames: np.ndarray  # [B, F, H, W, C] uint8 video frames
    audio: Optional[np.ndarray] = None  # [samples] float audio waveform
    fps: float = 25.0  # Output framerate (HuMo trained at 25 FPS)
    audio_sample_rate: int = 16000  # Whisper sample rate


class ProgressCallback:
    """
    Progress callback for video generation with tqdm-style output.

    Tracks step progress and performance metrics (it/s, ETA).

    Usage:
        callback = ProgressCallback(total_steps=50)
        pipeline(prompt="...", callback=callback)
        callback.close()
    """

    def __init__(
        self,
        total_steps: int = 50,
        desc: str = "Generating",
        disable: bool = False,
    ):
        self.total_steps = total_steps if total_steps else 50
        self.desc = desc
        self.disable = disable
        self.current_step = 0
        self.start_time: Optional[float] = None
        self.step_times: list[float] = []
        self._last_step_time: Optional[float] = None

    def _format_time(self, seconds: float) -> str:
        """Format seconds into human readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}:{secs:02d}"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}:{mins:02d}:00"

    def __call__(self, step: int, timestep: float, **kwargs) -> None:
        """
        Called at end of each diffusion step.

        Args:
            step: Current step index (0-based).
            timestep: Current diffusion timestep.
            **kwargs: Additional info.
        """
        if self.disable:
            return

        now = time.time()

        # Initialize on first call
        if self.start_time is None:
            self.start_time = now
            self._last_step_time = now

        # Track step time
        if self._last_step_time is not None:
            step_time = now - self._last_step_time
            self.step_times.append(step_time)
        self._last_step_time = now

        self.current_step = step + 1  # 1-indexed for display

        # Calculate metrics
        elapsed = now - self.start_time
        avg_step_time = elapsed / self.current_step if self.current_step > 0 else 0
        its = 1.0 / avg_step_time if avg_step_time > 0 else 0
        remaining_steps = self.total_steps - self.current_step
        eta = avg_step_time * remaining_steps

        # Build progress bar string
        bar_width = 30
        filled = int(bar_width * self.current_step / self.total_steps)
        bar = "=" * filled + ">" + " " * (bar_width - filled - 1)

        # Print progress line
        status = (
            f"\r{self.desc}: [{bar}] {self.current_step}/{self.total_steps} "
            f"[{self._format_time(elapsed)}<{self._format_time(eta)}, {its:.2f}it/s]"
        )
        print(status, end="", flush=True)

        # Newline on completion
        if self.current_step >= self.total_steps:
            print()

    def close(self) -> None:
        """Close progress bar (prints newline if needed)."""
        if self.current_step > 0 and self.current_step < self.total_steps:
            print()

    def get_stats(self) -> dict:
        """Get performance statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        avg_step_time = elapsed / self.current_step if self.current_step > 0 else 0
        return {
            "elapsed": elapsed,
            "avg_step_time": avg_step_time,
            "its": 1.0 / avg_step_time if avg_step_time > 0 else 0,
            "step_times": self.step_times,
            "total_steps": self.current_step,
        }


@dataclass
class WanConfig:
    """Configuration for Wan/HuMo video pipeline."""

    # Model type: "wan" for Wan DiT, "humo" for HuMo with audio
    model_type: str = "wan"  # "wan" or "humo"

    # Model paths
    wan_path: str = ""  # Path to Wan checkpoint (for DiT, VAE, text encoder)
    humo_path: str = ""  # Path to HuMo transformer weights (only for humo mode)
    whisper_path: str = ""  # Path to Whisper (optional, for audio)

    # Model variant
    wan_variant: str = "t2v-1.3b"  # "t2v-1.3b" or "t2v-14b" or "i2v-14b"
    humo_variant: str = "17B"  # "17B" or "1.7B" (only for humo mode)

    # Generation defaults
    num_frames: int = 17  # Default for testing (Wan supports up to 97 at 25fps)
    height: int = 480
    width: int = 832
    fps: float = 25.0  # Wan trained at 25 FPS
    num_inference_steps: int = 50

    # Guidance scales
    guidance_scale: float = 5.0  # Text guidance (scale_t)
    audio_scale: float = 0.0  # Audio guidance (scale_a), 0 = T2V mode

    # Memory
    enable_cpu_offload: bool = True
    dtype: str = "bfloat16"


class WanVideoPipeline:
    """
    Wan/HuMo video generation pipeline.

    Uses HuMo transformer as base, supporting both T2V and audio-conditioned modes.
    Audio conditioning is controlled via audio_scale parameter at runtime.

    Architecture:
    - transformer: HuMo-17B or HuMo-1.7B (DiT with audio cross-attention)
    - vae: Wan VAE for video encoding/decoding
    - text_encoder: UMT5-XXL for text conditioning
    - whisper: Whisper-large-v3 encoder for audio (lazy-loaded)
    """

    def __init__(
        self,
        transformer: Optional[nn.Module] = None,
        vae: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        config: Optional[WanConfig] = None,
    ):
        """
        Initialize pipeline with components.

        Typically use from_pretrained() instead of direct construction.
        """
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.config = config or WanConfig()

        # Lazy-loaded audio components
        self._whisper_encoder = None
        self._audio_processor = None

        # Device tracking
        self._device = torch.device("cpu")
        self._dtype = torch.bfloat16

    @classmethod
    def from_pretrained(
        cls,
        humo_path: str,
        wan_path: str,
        whisper_path: Optional[str] = None,
        humo_variant: str = "17B",
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        enable_cpu_offload: bool = True,
        **kwargs,
    ) -> "WanVideoPipeline":
        """
        Load pipeline from pretrained weights.

        Args:
            humo_path: Path to HuMo weights (e.g., ~/Storage/HuMo)
            wan_path: Path to Wan2.1-T2V-1.3B (for VAE/text encoder)
            whisper_path: Path to Whisper (optional, lazy-loads if None)
            humo_variant: "17B" or "1.7B"
            torch_dtype: Model dtype (bfloat16 recommended)
            device: Target device
            enable_cpu_offload: Enable CPU offload for memory efficiency
            **kwargs: Additional arguments

        Returns:
            Initialized WanVideoPipeline
        """
        # Expand paths
        humo_path = str(Path(humo_path).expanduser())
        wan_path = str(Path(wan_path).expanduser())
        if whisper_path:
            whisper_path = str(Path(whisper_path).expanduser())

        logger.info("=" * 60)
        logger.info("LOADING WAN/HUMO VIDEO PIPELINE")
        logger.info("=" * 60)
        logger.info(f"  HuMo path: {humo_path}")
        logger.info(f"  HuMo variant: {humo_variant}")
        logger.info(f"  Wan path: {wan_path}")
        logger.info(f"  Whisper path: {whisper_path or 'lazy-load'}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Dtype: {torch_dtype}")
        logger.info(f"  CPU offload: {enable_cpu_offload}")
        logger.info("-" * 60)

        start_time = time.time()

        # Build config
        config = WanConfig(
            model_type="humo",
            wan_path=wan_path,
            humo_path=humo_path,
            whisper_path=whisper_path or "",
            humo_variant=humo_variant,
            enable_cpu_offload=enable_cpu_offload,
        )

        # Load components
        transformer = cls._load_humo_transformer(humo_path, humo_variant, torch_dtype, device, enable_cpu_offload)
        vae = cls._load_wan_vae(wan_path, torch_dtype, device, enable_cpu_offload)
        text_encoder, tokenizer = cls._load_text_encoder(wan_path, torch_dtype, device, enable_cpu_offload)
        scheduler = cls._create_scheduler()

        load_time = time.time() - start_time
        logger.info(f"Pipeline loaded in {load_time:.1f}s")
        logger.info("=" * 60)

        instance = cls(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            config=config,
        )
        instance._device = torch.device(device)
        instance._dtype = torch_dtype

        return instance

    @classmethod
    def from_wan_pretrained(
        cls,
        wan_path: str,
        wan_variant: str = "t2v-1.3b",
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        enable_cpu_offload: bool = True,
        **kwargs,
    ) -> "WanVideoPipeline":
        """
        Load Wan pipeline for text-to-video generation.

        This is the recommended method for T2V generation.
        Faster loading, smaller memory footprint than HuMo.

        Args:
            wan_path: Path to Wan2.1-T2V checkpoint directory
            wan_variant: Model variant: "t2v-1.3b", "t2v-14b", or "i2v-14b"
            torch_dtype: Model dtype (bfloat16 recommended)
            device: Target device
            enable_cpu_offload: Enable CPU offload for memory efficiency
            **kwargs: Additional arguments

        Returns:
            Initialized WanVideoPipeline in Wan-only mode

        Example:
            pipe = WanVideoPipeline.from_wan_pretrained("~/Storage/Wan2.1-T2V-1.3B")
            video = pipe(prompt="A cat sleeping on a sunny windowsill")
            pipe.save_video(video, "cat.mp4")
        """
        # Expand path
        wan_path = str(Path(wan_path).expanduser())

        logger.info("=" * 60)
        logger.info("LOADING WAN VIDEO PIPELINE")
        logger.info("=" * 60)
        logger.info(f"  Wan path: {wan_path}")
        logger.info(f"  Variant: {wan_variant}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Dtype: {torch_dtype}")
        logger.info(f"  CPU offload: {enable_cpu_offload}")
        logger.info("-" * 60)

        start_time = time.time()

        # Build config
        config = WanConfig(
            model_type="wan",
            wan_path=wan_path,
            wan_variant=wan_variant,
            enable_cpu_offload=enable_cpu_offload,
        )

        # Load components
        transformer = cls._load_wan_transformer(wan_path, wan_variant, torch_dtype, device, enable_cpu_offload)
        vae = cls._load_wan_vae(wan_path, torch_dtype, device, enable_cpu_offload)
        text_encoder, tokenizer = cls._load_text_encoder(wan_path, torch_dtype, device, enable_cpu_offload)
        scheduler = cls._create_scheduler()

        load_time = time.time() - start_time
        logger.info(f"Pipeline loaded in {load_time:.1f}s")
        logger.info("=" * 60)

        instance = cls(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            config=config,
        )
        instance._device = torch.device(device)
        instance._dtype = torch_dtype

        return instance

    @staticmethod
    def _load_wan_transformer(
        wan_path: str,
        variant: str,
        dtype: torch.dtype,
        device: str,
        cpu_offload: bool,
    ) -> nn.Module:
        """Load Wan DiT transformer."""
        from llm_dit.models.wan_dit import WanDiT

        # Map variant string to config name
        config_map = {
            "t2v-1.3b": "wan2.1-t2v-1.3b",
            "t2v-14b": "wan2.1-t2v-14b",
            "i2v-14b": "wan2.1-i2v-14b",
            "1.3b": "wan2.1-t2v-1.3b",  # Alias
            "14b": "wan2.1-t2v-14b",  # Alias
        }
        config_name = config_map.get(variant.lower(), variant)

        # Find weights file
        wan_path = Path(wan_path)
        weights_file = wan_path / "diffusion_pytorch_model.safetensors"

        if not weights_file.exists():
            raise FileNotFoundError(
                f"Wan DiT weights not found at {weights_file}. "
                f"Download with: huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir {wan_path}"
            )

        logger.info(f"Loading Wan DiT ({config_name}) from {weights_file}")

        # Load model
        transformer = WanDiT.from_pretrained(
            str(weights_file),
            config_name=config_name,
            device="cpu" if cpu_offload else device,
            dtype=dtype,
        )

        if not cpu_offload:
            transformer = transformer.to(device)

        params_b = sum(p.numel() for p in transformer.parameters()) / 1e9
        logger.info(f"  Wan DiT loaded: {params_b:.2f}B params")

        return transformer

    @staticmethod
    def _load_humo_transformer(
        humo_path: str,
        variant: str,
        dtype: torch.dtype,
        device: str,
        cpu_offload: bool,
    ) -> nn.Module:
        """Load HuMo transformer weights."""
        variant_path = Path(humo_path) / f"HuMo-{variant}"
        logger.info(f"Loading HuMo-{variant} transformer from {variant_path}")

        # Check for weights in multiple formats
        index_file = variant_path / "humo.safetensors.index.json"
        ema_pth_file = variant_path / "ema.pth"
        single_safetensors = variant_path / "humo.safetensors"

        if not (index_file.exists() or ema_pth_file.exists() or single_safetensors.exists()):
            raise FileNotFoundError(
                f"HuMo weights not found at {variant_path}. "
                f"Expected one of: {index_file.name}, {ema_pth_file.name}, {single_safetensors.name}. "
                f"Download with: huggingface-cli download bytedance-research/HuMo --local-dir {humo_path}"
            )

        # Determine architecture from variant
        if variant == "17B":
            # HuMo-17B: 40 blocks, hidden=5120, 40 heads
            # 36 input channels (noise 16 + image 16 + audio 4) for I2V/audio modes
            num_layers = 40
            hidden_size = 5120
            num_heads = 40
            ffn_dim = 13824
            patch_in_channels = 36
        else:
            # HuMo-1.7B: 30 blocks, hidden=1536, 24 heads
            # 16 input channels (noise only) for T2V mode
            num_layers = 30
            hidden_size = 1536
            num_heads = 24  # head_dim=64
            ffn_dim = 8960
            patch_in_channels = 16

        # Create transformer model
        from llm_dit.models.humo_transformer import HuMoTransformer

        transformer = HuMoTransformer(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            patch_in_channels=patch_in_channels,
        )

        # Load weights based on available format
        if index_file.exists():
            # Sharded safetensors (HuMo-17B)
            import json
            from safetensors import safe_open

            with open(index_file) as f:
                index = json.load(f)

            weight_map = index.get("weight_map", {})
            shard_files = set(weight_map.values())
            logger.info(f"  Loading {len(shard_files)} shards...")

            state_dict = {}
            for shard_file in sorted(shard_files):
                shard_path = variant_path / shard_file
                logger.info(f"    Loading {shard_file}...")
                with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)

        elif single_safetensors.exists():
            # Single safetensors file
            from safetensors.torch import load_file as load_safetensors
            logger.info(f"  Loading {single_safetensors.name}...")
            state_dict = load_safetensors(str(single_safetensors))

        else:
            # PyTorch .pth file (HuMo-1.7B ema.pth)
            logger.info(f"  Loading {ema_pth_file.name}...")
            state_dict = torch.load(str(ema_pth_file), map_location="cpu", weights_only=True)

        # Load state dict
        missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"  Missing keys: {len(missing)}")
        if unexpected:
            logger.warning(f"  Unexpected keys: {len(unexpected)}")

        transformer = transformer.to(dtype)
        if not cpu_offload:
            transformer = transformer.to(device)

        logger.info(f"  HuMo transformer loaded: {num_layers} layers, {hidden_size} hidden, {num_heads} heads")

        return transformer

    @staticmethod
    def _load_wan_vae(
        wan_path: str,
        dtype: torch.dtype,
        device: str,
        cpu_offload: bool,
    ) -> nn.Module:
        """Load Wan VAE for video encoding/decoding."""
        from safetensors.torch import load_file as load_safetensors

        vae_path = Path(wan_path) / "Wan2.1_VAE.safetensors"

        if not vae_path.exists():
            raise FileNotFoundError(
                f"Wan VAE not found at {vae_path}. "
                f"Download and convert with:\n"
                f"  huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir {wan_path}\n"
                f"  uv run python scripts/convert_to_safetensors.py {wan_path} --recursive"
            )

        logger.info(f"Loading Wan VAE from {vae_path}")
        state_dict = load_safetensors(str(vae_path))

        # Create VAE model
        from llm_dit.models.wan_vae import WanVAE

        vae = WanVAE()
        # Checkpoint keys have no 'model.' prefix - load into inner VideoVAE
        vae.model.load_state_dict(state_dict)
        vae = vae.to(dtype)

        if not cpu_offload:
            vae = vae.to(device)

        logger.info("  Wan VAE loaded")
        return vae

    @staticmethod
    def _load_text_encoder(
        wan_path: str,
        dtype: torch.dtype,
        device: str,
        cpu_offload: bool,
    ):
        """Load UMT5-XXL text encoder and tokenizer."""
        from ..models.wan_text_encoder import WanTextEncoder

        tokenizer_path = Path(wan_path) / "google" / "umt5-xxl"
        weights_path = Path(wan_path) / "models_t5_umt5-xxl-enc-bf16.safetensors"

        logger.info(f"Loading UMT5-XXL text encoder from {weights_path}")

        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_path}. "
                f"Download with: huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir {wan_path}"
            )

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Text encoder weights not found at {weights_path}. "
                f"Download and convert with:\n"
                f"  huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir {wan_path}\n"
                f"  uv run python scripts/convert_to_safetensors.py {wan_path} --recursive"
            )

        # Create encoder with tokenizer
        text_encoder = WanTextEncoder(max_length=512, dtype=dtype)
        text_encoder.load_tokenizer(str(tokenizer_path))
        text_encoder.load_weights(str(weights_path))

        # Always convert to target dtype (weights are stored in fp32)
        text_encoder.model = text_encoder.model.to(dtype=dtype)

        if not cpu_offload:
            text_encoder.model = text_encoder.model.to(device=device)

        logger.info("  UMT5-XXL text encoder loaded")
        # Return encoder only (tokenizer is integrated)
        return text_encoder, text_encoder.tokenizer

    @staticmethod
    def _create_scheduler():
        """Create Wan-specific flow match scheduler."""
        from llm_dit.schedulers.flow_match import FlowMatchScheduler

        # DiffSynth-Engine uses sigma_min=0.001, sigma_max=0.999
        # NOT 0.0 and 1.0 - the boundaries prevent numerical instability
        # Formula: sigma' = shift * sigma / (1 + (shift - 1) * sigma)
        scheduler = FlowMatchScheduler(
            num_train_timesteps=1000,
            shift=5.0,
            sigma_min=0.001,
            sigma_max=0.999,
        )
        return scheduler

    def load_whisper(self, whisper_path: Optional[str] = None) -> None:
        """
        Load Whisper encoder for audio conditioning.

        Called lazily when audio is first used.

        Args:
            whisper_path: Path to Whisper weights, or None to use config path
        """
        if self._whisper_encoder is not None:
            return  # Already loaded

        path = whisper_path or self.config.whisper_path
        if not path:
            path = "openai/whisper-large-v3"  # Default to HuggingFace

        logger.info(f"Loading Whisper encoder from {path}")

        from transformers import WhisperModel, WhisperProcessor

        self._audio_processor = WhisperProcessor.from_pretrained(path)
        whisper = WhisperModel.from_pretrained(path, torch_dtype=self._dtype)
        self._whisper_encoder = whisper.encoder

        if not self.config.enable_cpu_offload:
            self._whisper_encoder = self._whisper_encoder.to(self._device)

        logger.info("  Whisper encoder loaded")

    def encode_audio(self, audio: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Encode audio using Whisper.

        Args:
            audio: Audio file path, waveform array, or tensor

        Returns:
            Audio embeddings [B, seq_len, 1280]
        """
        # Lazy load Whisper
        if self._whisper_encoder is None:
            self.load_whisper()

        # Load audio if path
        if isinstance(audio, str):
            import torchaudio
            waveform, sr = torchaudio.load(audio)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            audio = waveform.squeeze().numpy()

        # Process with Whisper processor
        inputs = self._audio_processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )

        # Encode
        input_features = inputs.input_features.to(self._device, self._dtype)
        with torch.no_grad():
            encoder_output = self._whisper_encoder(input_features)
            audio_embeds = encoder_output.last_hidden_state

        return audio_embeds

    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        audio: Optional[Union[str, np.ndarray, torch.Tensor]] = None,
        image: Optional[Union[torch.Tensor, np.ndarray, "PIL.Image.Image"]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        audio_scale: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        callback: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> VideoOutput:
        """
        Generate video from text prompt and optional audio/image.

        Args:
            prompt: Text prompt describing desired video
            negative_prompt: Negative prompt for CFG
            audio: Audio for conditioning (file path, waveform, or None for T2V)
            image: Reference image for TIA mode (optional)
            height: Video height (default: 720)
            width: Video width (default: 1280)
            num_frames: Number of frames (default: 97)
            num_inference_steps: Diffusion steps (default: 50)
            guidance_scale: Text guidance scale_t (default: 5.0)
            audio_scale: Audio guidance scale_a (default: 0.0, set >0 for audio mode)
            generator: Torch generator for reproducibility
            seed: Random seed
            callback: Progress callback
            **kwargs: Additional arguments

        Returns:
            VideoOutput with generated frames
        """
        # Use defaults from config
        height = height or self.config.height
        width = width or self.config.width
        num_frames = num_frames or self.config.num_frames
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        guidance_scale = guidance_scale if guidance_scale is not None else self.config.guidance_scale
        audio_scale = audio_scale if audio_scale is not None else self.config.audio_scale

        # Auto-enable audio mode if audio provided
        if audio is not None and audio_scale == 0.0:
            audio_scale = 1.0
            logger.info("Audio provided, setting audio_scale=1.0")

        # Create generator from seed
        if seed is not None and generator is None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        # Determine mode
        if audio is not None:
            mode = "TIA" if image is not None else "TA"
        else:
            mode = "T2V"

        logger.info(
            f"Generating video: {width}x{height}, {num_frames} frames, "
            f"{num_inference_steps} steps, mode={mode}"
        )
        logger.info(f"  scale_t={guidance_scale}, scale_a={audio_scale}")

        # Encode text (embeddings with padding positions zeroed)
        text_embeds = self._encode_text(prompt)
        if negative_prompt:
            negative_embeds = self._encode_text(negative_prompt)
        else:
            negative_embeds = self._encode_text("")

        # Encode audio if provided
        audio_embeds = None
        if audio is not None:
            audio_embeds = self.encode_audio(audio)

        # Encode image if provided
        image_latents = None
        if image is not None:
            image_latents = self._encode_image(image)

        # Run diffusion
        latents = self._diffusion_loop(
            text_embeds=text_embeds,
            negative_embeds=negative_embeds,
            audio_embeds=audio_embeds,
            image_latents=image_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            audio_scale=audio_scale,
            generator=generator,
            callback=callback,
        )

        # Decode latents to video
        frames = self._decode_latents(latents)

        # Trim to requested frame count (VAE may output slightly more due to causal conv)
        if frames.shape[1] > num_frames:
            frames = frames[:, :num_frames]

        return VideoOutput(
            frames=frames,
            fps=self.config.fps,
        )

    def _encode_text(self, prompt: str) -> torch.Tensor:
        """
        Encode text prompt with UMT5.

        Returns:
            text_embeds: [B, S, text_dim] with zeros after actual tokens

        Note:
            Following DiffSynth-Studio, we zero out padding embeddings but don't use
            explicit attention masks. This is sufficient because V=0 means the
            softmax-weighted sum contributes zero from padding positions.
        """
        # Move encoder to device if needed
        if self.config.enable_cpu_offload:
            self.text_encoder.model = self.text_encoder.model.to(self._device)

        # Encode using WanTextEncoder interface
        text_embeds, attention_mask = self.text_encoder.encode(prompt)

        # Zero out embeddings after actual token count
        # This matches DiffSynth-Studio behavior which zeros padding positions
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            text_embeds[i, v:] = 0

        # Offload if needed
        if self.config.enable_cpu_offload:
            self.text_encoder.model = self.text_encoder.model.to("cpu")
            torch.cuda.empty_cache()

        return text_embeds

    def _encode_image(self, image) -> torch.Tensor:
        """Encode reference image with VAE for Image-to-Video conditioning."""
        # Convert to tensor if needed
        if not isinstance(image, torch.Tensor):
            from PIL import Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            # Assume PIL Image
            image = torch.from_numpy(np.array(image)).float() / 255.0
            # Create 5D tensor: [B, C, T, H, W] with T=1 for single image
            image = image.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]
        elif image.ndim == 4:
            # 4D tensor [B, C, H, W] -> add temporal dim -> [B, C, 1, H, W]
            image = image.unsqueeze(2)

        image = image.to(self._device, self._dtype)

        # Move VAE to device if needed
        if self.config.enable_cpu_offload:
            self.vae = self.vae.to(self._device)

        with torch.no_grad():
            latents = self.vae.encode(image)

        # Offload if needed
        if self.config.enable_cpu_offload:
            self.vae = self.vae.to("cpu")
            torch.cuda.empty_cache()

        return latents

    def _prepare_transformer_input(
        self,
        noise_latents: torch.Tensor,
        image_latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prepare input for HuMo transformer.

        Input channels depend on variant:
        - HuMo-17B: 36 channels (noise 16 + image 16 + extra 4)
        - HuMo-1.7B: 16 channels (noise only, T2V mode)

        Args:
            noise_latents: [B, 16, T, H, W] noise/denoising latents
            image_latents: [B, 16, T, H, W] image conditioning or None

        Returns:
            Transformer input with variant-appropriate channels
        """
        # HuMo-1.7B only supports 16-channel input (T2V mode only)
        if self.config.humo_variant == "1.7B":
            return noise_latents

        # HuMo-17B: 36 channels for I2V/TIA modes
        B, C, T, H, W = noise_latents.shape

        # Image conditioning: use provided latents or zeros for T2V mode
        if image_latents is None:
            image_cond = torch.zeros_like(noise_latents)
        else:
            # Ensure image latents match temporal dimension
            if image_latents.shape[2] != T:
                # Repeat first frame for all timesteps (image-to-video)
                image_cond = image_latents[:, :, :1, :, :].expand(-1, -1, T, -1, -1)
            else:
                image_cond = image_latents

        # Extra conditioning: 4 channels (typically mask/padding info)
        # For now, zeros - can be extended for more complex conditioning
        extra_cond = torch.zeros(
            B, 4, T, H, W,
            device=noise_latents.device,
            dtype=noise_latents.dtype
        )

        # Concatenate: [noise, image, extra] -> [B, 36, T, H, W]
        return torch.cat([noise_latents, image_cond, extra_cond], dim=1)

    def _diffusion_loop(
        self,
        text_embeds: torch.Tensor,
        negative_embeds: torch.Tensor,
        audio_embeds: Optional[torch.Tensor],
        image_latents: Optional[torch.Tensor],
        height: int,
        width: int,
        num_frames: int,
        num_inference_steps: int,
        guidance_scale: float,
        audio_scale: float,
        generator: Optional[torch.Generator],
        callback: Optional[ProgressCallback],
    ) -> torch.Tensor:
        """Run the diffusion denoising loop."""
        # Calculate latent dimensions
        # Wan VAE: 8x spatial downscale, 4x temporal compression
        # Formula: latent_frames = (num_frames - 1) // 4 + 1
        # Example: 17 frames -> 5 latent frames -> decode back to 17 frames
        latent_height = height // 8
        latent_width = width // 8
        latent_frames = (num_frames - 1) // 4 + 1

        # Initialize latents (16 channels for noise)
        # Generate on CPU for reproducibility, then move to device
        latents = torch.randn(
            (1, 16, latent_frames, latent_height, latent_width),
            generator=generator,
            device="cpu",
            dtype=self._dtype,
        ).to(self._device)

        # Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # Move transformer to device if needed
        if self.config.enable_cpu_offload:
            self.transformer = self.transformer.to(self._device)

        # Check if using Wan or HuMo mode
        is_wan_mode = self.config.model_type == "wan"

        # Denoising loop
        for i, t in enumerate(timesteps):
            # Prepare noise latents for CFG (duplicate if using guidance)
            if guidance_scale > 1.0:
                latent_for_cfg = torch.cat([latents, latents])
            else:
                latent_for_cfg = latents

            # Prepare transformer input based on mode
            if is_wan_mode:
                # Wan: use 16-channel noise latents directly
                transformer_input = latent_for_cfg
            else:
                # HuMo: assemble 36-channel input
                if guidance_scale > 1.0:
                    img_for_cfg = torch.cat([image_latents, image_latents]) if image_latents is not None else None
                else:
                    img_for_cfg = image_latents
                transformer_input = self._prepare_transformer_input(latent_for_cfg, img_for_cfg)

            timestep = t.expand(transformer_input.shape[0]).to(device=self._device, dtype=self._dtype)

            # Prepare text embeddings for CFG
            # Note: We zero out padding embeddings in _encode_text() which is sufficient
            # for cross-attention (V=0 means zero contribution). Explicit attention masks
            # are NOT used because the model wasn't trained with them.
            if guidance_scale > 1.0:
                text_input = torch.cat([negative_embeds, text_embeds])
            else:
                text_input = text_embeds

            # Forward pass through transformer
            with torch.no_grad():
                if is_wan_mode:
                    # Wan forward (16 channels)
                    # Note: encoder_attention_mask=None because zeroed embeddings are sufficient
                    noise_pred = self.transformer(
                        hidden_states=transformer_input,
                        timestep=timestep,
                        encoder_hidden_states=text_input,
                    )
                else:
                    # HuMo forward (36 channels, audio optional)
                    audio_input = None
                    if audio_embeds is not None and audio_scale > 0:
                        if guidance_scale > 1.0:
                            audio_input = torch.cat([audio_embeds, audio_embeds])
                        else:
                            audio_input = audio_embeds

                    noise_pred = self.transformer(
                        hidden_states=transformer_input,
                        timestep=timestep,
                        encoder_hidden_states=text_input,
                        audio_hidden_states=audio_input,
                        audio_scale=audio_scale,
                    )

            # CFG
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # Callback
            if callback is not None:
                callback(i, t.item())

        # Offload transformer if needed
        if self.config.enable_cpu_offload:
            self.transformer = self.transformer.to("cpu")
            torch.cuda.empty_cache()

        return latents

    def _decode_latents(self, latents: torch.Tensor) -> np.ndarray:
        """Decode latents to video frames."""
        # Move VAE to device if needed
        if self.config.enable_cpu_offload:
            self.vae = self.vae.to(self._device)

        with torch.no_grad():
            video = self.vae.decode(latents)

        # Offload VAE if needed
        if self.config.enable_cpu_offload:
            self.vae = self.vae.to("cpu")
            torch.cuda.empty_cache()

        # Convert to numpy uint8
        # VAE outputs [-1, 1] range - remap to [0, 1] first
        video = video.cpu().float().numpy()
        video = (video + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        video = (video * 255).clip(0, 255).astype(np.uint8)

        # Reshape: [1, 3, F, H, W] -> [1, F, H, W, 3]
        if video.ndim == 5:
            video = video.transpose(0, 2, 3, 4, 1)

        return video

    def save_video(
        self,
        output: Union[VideoOutput, np.ndarray],
        output_path: str,
        audio: Optional[np.ndarray] = None,
        fps: Optional[float] = None,
        audio_sample_rate: int = 16000,
    ) -> str:
        """
        Save video output to file.

        Args:
            output: VideoOutput or video frames array
            output_path: Path to save video
            audio: Audio waveform (if output is array)
            fps: Framerate (if output is array)
            audio_sample_rate: Audio sample rate

        Returns:
            Path to saved video
        """
        if isinstance(output, VideoOutput):
            frames = output.frames
            audio = output.audio
            fps = output.fps
            audio_sample_rate = output.audio_sample_rate
        else:
            frames = output
            fps = fps or 25.0

        # Handle batch dimension: [B, F, H, W, C] -> [F, H, W, C]
        if frames.ndim == 5:
            frames = frames[0]

        # Ensure uint8
        if frames.dtype != np.uint8:
            if frames.max() <= 1.0:
                frames = (frames * 255).clip(0, 255).astype(np.uint8)
            else:
                frames = frames.clip(0, 255).astype(np.uint8)

        saved = False

        # Option 1: PyAV (supports audio)
        try:
            import av
            from fractions import Fraction

            num_frames, height, width, _ = frames.shape
            container = av.open(output_path, mode="w")

            # Video stream
            video_stream = container.add_stream("libx264", rate=int(fps))
            video_stream.width = width
            video_stream.height = height
            video_stream.pix_fmt = "yuv420p"

            # Audio stream (if audio provided)
            audio_stream = None
            if audio is not None:
                audio_stream = container.add_stream("aac", rate=audio_sample_rate)
                audio_stream.codec_context.sample_rate = audio_sample_rate
                audio_stream.codec_context.layout = "stereo"
                audio_stream.codec_context.time_base = Fraction(1, audio_sample_rate)

            # Write video frames
            for frame_array in frames:
                frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
                for packet in video_stream.encode(frame):
                    container.mux(packet)

            # Flush video encoder
            for packet in video_stream.encode():
                container.mux(packet)

            # Write audio if provided
            if audio is not None and audio_stream is not None:
                audio_data = audio
                if isinstance(audio_data, np.ndarray):
                    audio_tensor = torch.from_numpy(audio_data)
                else:
                    audio_tensor = audio_data

                if audio_tensor.ndim == 1:
                    audio_tensor = audio_tensor.unsqueeze(1)
                if audio_tensor.shape[1] != 2 and audio_tensor.shape[0] == 2:
                    audio_tensor = audio_tensor.T
                if audio_tensor.shape[1] == 1:
                    audio_tensor = audio_tensor.repeat(1, 2)

                if audio_tensor.dtype != torch.int16:
                    audio_tensor = torch.clip(audio_tensor.float(), -1.0, 1.0)
                    audio_tensor = (audio_tensor * 32767.0).to(torch.int16)

                audio_np = audio_tensor.contiguous().reshape(1, -1).cpu().numpy()
                audio_frame = av.AudioFrame.from_ndarray(
                    audio_np, format="s16", layout="stereo"
                )
                audio_frame.sample_rate = audio_sample_rate

                resampler = av.audio.resampler.AudioResampler(
                    format=audio_stream.codec_context.format or "fltp",
                    layout=audio_stream.codec_context.layout or "stereo",
                    rate=audio_sample_rate,
                )
                audio_pts = 0
                for resampled_frame in resampler.resample(audio_frame):
                    if resampled_frame.pts is None:
                        resampled_frame.pts = audio_pts
                    audio_pts += resampled_frame.samples
                    resampled_frame.sample_rate = audio_sample_rate
                    for packet in audio_stream.encode(resampled_frame):
                        container.mux(packet)

                for packet in audio_stream.encode():
                    container.mux(packet)

                logger.info(f"Video+Audio saved with PyAV: {output_path}")
            else:
                logger.info(f"Video saved with PyAV: {output_path}")

            container.close()
            saved = True
        except ImportError:
            logger.debug("PyAV not available, trying alternatives...")
        except Exception as e:
            logger.warning(f"PyAV failed: {e}")

        # Option 2: torchvision
        if not saved:
            try:
                import torchvision.io as tvio

                video_tensor = torch.from_numpy(frames)
                tvio.write_video(output_path, video_tensor, fps=fps)
                saved = True
                logger.info(f"Video saved with torchvision: {output_path}")
            except Exception as e:
                logger.debug(f"torchvision.io.write_video failed: {e}")

        # Option 3: imageio (matching DiffSynth-Engine approach)
        if not saved:
            try:
                import imageio.v3 as iio

                # Use FFMPEG plugin explicitly with libx264 codec (matches DiffSynth-Engine)
                codec = "libvpx-vp9" if output_path.endswith(".webm") else "libx264"
                with iio.imopen(output_path, "w", plugin="FFMPEG") as writer:
                    writer.write(frames, fps=fps, codec=codec)
                saved = True
                logger.info(f"Video saved with imageio (FFMPEG): {output_path}")
            except Exception as e:
                logger.debug(f"imageio failed: {e}")

        # Option 4: OpenCV
        if not saved:
            try:
                import cv2

                h, w = frames.shape[1:3]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                for frame in frames:
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                out.release()
                saved = True
                logger.info(f"Video saved with OpenCV: {output_path}")
            except Exception as e:
                logger.debug(f"OpenCV failed: {e}")

        if not saved:
            raise RuntimeError(
                "No video encoder available. Install one of:\n"
                "  pip install av          (recommended, supports audio)\n"
                "  pip install torchvision\n"
                "  pip install imageio[ffmpeg]\n"
                "  pip install opencv-python"
            )

        return output_path

    def offload(self) -> None:
        """Offload all models to CPU."""
        if self.transformer is not None:
            self.transformer = self.transformer.to("cpu")
        if self.vae is not None:
            self.vae = self.vae.to("cpu")
        if self.text_encoder is not None:
            self.text_encoder.model = self.text_encoder.model.to("cpu")
        if self._whisper_encoder is not None:
            self._whisper_encoder = self._whisper_encoder.to("cpu")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("WanVideoPipeline offloaded to CPU")

    def to(self, device: Union[str, torch.device]) -> "WanVideoPipeline":
        """Move pipeline to device."""
        device = torch.device(device)
        if self.transformer is not None:
            self.transformer = self.transformer.to(device)
        if self.vae is not None:
            self.vae = self.vae.to(device)
        if self.text_encoder is not None:
            self.text_encoder.model = self.text_encoder.model.to(device)
        if self._whisper_encoder is not None:
            self._whisper_encoder = self._whisper_encoder.to(device)
        self._device = device
        return self

    @property
    def device(self) -> torch.device:
        """Get pipeline device."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Get pipeline dtype."""
        return self._dtype

    @property
    def mode(self) -> str:
        """Get pipeline mode based on config."""
        if self.config.audio_scale > 0:
            return "audio"
        return "t2v"

    def estimate_memory(self) -> Dict[str, float]:
        """Estimate memory usage in GB."""
        variant = self.config.humo_variant
        if variant == "17B":
            transformer_gb = 34.0  # 17B params in bf16
        else:
            transformer_gb = 3.4  # 1.7B params in bf16

        return {
            "transformer": transformer_gb,
            "vae": 1.0,
            "text_encoder": 12.0,  # UMT5-XXL in bf16
            "whisper": 3.0,  # Whisper-large-v3
            "peak_with_offload": max(transformer_gb, 12.0) + 2.0,  # Largest component + overhead
            "peak_no_offload": transformer_gb + 12.0 + 1.0 + 3.0 + 5.0,  # All + activations
        }
