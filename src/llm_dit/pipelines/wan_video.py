"""
Wan Video Pipeline with HuMo audio conditioning support.

Last Updated: 2026-01-11

This pipeline uses HuMo's transformer as the base, which supports:
- Text-to-video (T2V): scale_a=0, text prompt only
- Text-Audio (TA): scale_a>0, audio-synchronized video
- Text-Image-Audio (TIA): scale_a>0, audio + reference image

Architecture:
- Transformer: HuMo-17B or HuMo-1.7B (DiT with audio cross-attention)
- VAE: From Wan2.1-T2V-1.3B
- Text encoder: UMT5-XXL from Wan2.1-T2V-1.3B
- Audio encoder: Whisper-large-v3 (lazy-loaded)

Example:
    from llm_dit.pipelines.wan_video import WanVideoPipeline

    # Load pipeline
    pipe = WanVideoPipeline.from_pretrained(
        humo_path="~/Storage/HuMo/HuMo-17B",
        wan_path="~/Storage/Wan2.1-T2V-1.3B",
    )

    # T2V mode (no audio)
    video = pipe(prompt="A woman dancing gracefully")

    # TA mode (audio-conditioned)
    video = pipe(
        prompt="A person dancing",
        audio="music.wav",
        audio_scale=1.0,
    )

    pipe.save_video(video, "output.mp4")
"""

import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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

    # Model paths
    humo_path: str = ""  # Path to HuMo transformer weights
    wan_path: str = ""  # Path to Wan2.1-T2V for VAE/text encoder
    whisper_path: str = ""  # Path to Whisper (optional, for audio)

    # Model variant
    humo_variant: str = "17B"  # "17B" or "1.7B"

    # Generation defaults
    num_frames: int = 97  # HuMo trained at 97 frames
    height: int = 720
    width: int = 1280
    fps: float = 25.0  # HuMo trained at 25 FPS
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
            humo_path=humo_path,
            wan_path=wan_path,
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

        # Check for safetensors index
        index_file = variant_path / "humo.safetensors.index.json"
        if not index_file.exists():
            raise FileNotFoundError(
                f"HuMo weights not found at {variant_path}. "
                f"Expected {index_file}. Download with: "
                f"huggingface-cli download bytedance-research/HuMo --local-dir {humo_path}"
            )

        # Load sharded weights
        import json
        from safetensors import safe_open

        with open(index_file) as f:
            index = json.load(f)

        weight_map = index.get("weight_map", {})
        shard_files = set(weight_map.values())

        logger.info(f"  Loading {len(shard_files)} shards...")

        # Determine architecture from variant
        if variant == "17B":
            # HuMo-17B: 40 blocks, hidden=5120, 40 heads
            num_layers = 40
            hidden_size = 5120
            num_heads = 40
            ffn_dim = 13824
        else:
            # HuMo-1.7B: 30 blocks, hidden=1536, 12 heads (matches Wan 1.3B)
            num_layers = 30
            hidden_size = 1536
            num_heads = 12
            ffn_dim = 8960

        # Create transformer model
        from llm_dit.models.humo_transformer import HuMoTransformer

        transformer = HuMoTransformer(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
        )

        # Load weights from shards
        state_dict = {}
        for shard_file in sorted(shard_files):
            shard_path = variant_path / shard_file
            logger.info(f"    Loading {shard_file}...")
            with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)

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
        vae_path = Path(wan_path) / "Wan2.1_VAE.pth"
        logger.info(f"Loading Wan VAE from {vae_path}")

        if not vae_path.exists():
            raise FileNotFoundError(
                f"Wan VAE not found at {vae_path}. "
                f"Download with: huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir {wan_path}"
            )

        # Load VAE state dict
        state_dict = torch.load(vae_path, map_location="cpu", weights_only=True)

        # Create VAE model
        from llm_dit.models.wan_vae import WanVAE

        vae = WanVAE()
        vae.load_state_dict(state_dict)
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
        encoder_path = Path(wan_path) / "google" / "umt5-xxl"
        logger.info(f"Loading UMT5-XXL text encoder from {encoder_path}")

        if not encoder_path.exists():
            raise FileNotFoundError(
                f"Text encoder not found at {encoder_path}. "
                f"Download with: huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir {wan_path}"
            )

        from transformers import AutoTokenizer, T5EncoderModel

        tokenizer = AutoTokenizer.from_pretrained(str(encoder_path))

        # Check for model weights
        model_file = encoder_path / "model.safetensors"
        if not model_file.exists():
            # Try to load from HuggingFace if local weights missing
            logger.info("  Local weights not found, loading from google/umt5-xxl...")
            text_encoder = T5EncoderModel.from_pretrained(
                "google/umt5-xxl",
                torch_dtype=dtype,
            )
        else:
            text_encoder = T5EncoderModel.from_pretrained(
                str(encoder_path),
                torch_dtype=dtype,
            )

        if not cpu_offload:
            text_encoder = text_encoder.to(device)

        logger.info("  UMT5-XXL text encoder loaded")
        return text_encoder, tokenizer

    @staticmethod
    def _create_scheduler():
        """Create diffusion scheduler."""
        from diffusers import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=3.0,
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

        # Encode text
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

        return VideoOutput(
            frames=frames,
            fps=self.config.fps,
        )

    def _encode_text(self, prompt: str) -> torch.Tensor:
        """Encode text prompt with UMT5."""
        inputs = self.tokenizer(
            prompt,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs.input_ids.to(self._device)
        attention_mask = inputs.attention_mask.to(self._device)

        # Move encoder to device if needed
        if self.config.enable_cpu_offload:
            self.text_encoder = self.text_encoder.to(self._device)

        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            text_embeds = outputs.last_hidden_state

        # Offload if needed
        if self.config.enable_cpu_offload:
            self.text_encoder = self.text_encoder.to("cpu")
            torch.cuda.empty_cache()

        return text_embeds

    def _encode_image(self, image) -> torch.Tensor:
        """Encode reference image with VAE."""
        # Convert to tensor if needed
        if not isinstance(image, torch.Tensor):
            from PIL import Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            # Assume PIL Image
            image = torch.from_numpy(np.array(image)).float() / 255.0
            image = image.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

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
        # Wan VAE: spatial 8x downscale, temporal 4x downscale
        latent_height = height // 8
        latent_width = width // 8
        latent_frames = (num_frames - 1) // 4 + 1  # Temporal compression

        # Initialize latents
        latents = torch.randn(
            (1, 16, latent_frames, latent_height, latent_width),
            generator=generator,
            device=self._device,
            dtype=self._dtype,
        )

        # Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # Move transformer to device if needed
        if self.config.enable_cpu_offload:
            self.transformer = self.transformer.to(self._device)

        # Denoising loop
        for i, t in enumerate(timesteps):
            # Prepare model input
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            timestep = t.expand(latent_model_input.shape[0]).to(self._device)

            # Prepare text embeddings for CFG
            if guidance_scale > 1.0:
                text_input = torch.cat([negative_embeds, text_embeds])
            else:
                text_input = text_embeds

            # Prepare audio embeddings
            audio_input = None
            if audio_embeds is not None and audio_scale > 0:
                if guidance_scale > 1.0:
                    # Duplicate for CFG
                    audio_input = torch.cat([audio_embeds, audio_embeds])
                else:
                    audio_input = audio_embeds

            # Forward pass
            with torch.no_grad():
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
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
        video = video.cpu().float().numpy()
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

        # Option 3: imageio
        if not saved:
            try:
                import imageio.v3 as iio

                iio.imwrite(output_path, frames, fps=fps)
                saved = True
                logger.info(f"Video saved with imageio: {output_path}")
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
            self.text_encoder = self.text_encoder.to("cpu")
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
            self.text_encoder = self.text_encoder.to(device)
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
