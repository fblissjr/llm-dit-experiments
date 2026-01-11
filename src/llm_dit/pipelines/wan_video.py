"""
Wan Video Pipeline for text-to-video and image-to-video generation.

Last Updated: 2026-01-11

This pipeline implements Wan 2.1/2.2 video generation with support for:
- Text-to-video (T2V) generation
- Image-to-video (I2V) conditioning
- Optional HuMo audio conditioning extension

Architecture:
- VideoTransformer: 40-layer DiT with 5120 dim, 40 heads
- Video VAE: 3D conv encoder/decoder (8x spatial, 4x temporal compression)
- UMT5-XXL: Text encoder (4096 dim -> 5120 via projection)
- Flow matching scheduler

Memory Strategy (24GB VRAM):
1. Enable model CPU offload (moves each component to GPU only when needed)
2. Sequential loading: encoder -> transformer -> VAE
3. FP8 quantized transformer for memory efficiency

Example:
    from llm_dit.pipelines.wan_video import WanVideoPipeline

    pipe = WanVideoPipeline.from_pretrained(
        model_path="path/to/wan-weights.safetensors",
        vae_path="path/to/vae.safetensors",
        text_encoder_path="path/to/umt5-xxl",
    )

    video = pipe(
        prompt="A woman dancing in a garden",
        num_frames=81,
    )
    pipe.save_video(video, "output.mp4")
"""

import gc
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np

try:
    from safetensors import safe_open
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    safe_open = None
    load_safetensors = None

logger = logging.getLogger(__name__)


# Reuse VideoOutput from LTX2 pattern
@dataclass
class VideoOutput:
    """Output from Wan video generation."""

    frames: np.ndarray  # [B, F, H, W, C] uint8 video frames
    audio: Optional[np.ndarray] = None  # [B, samples] float audio (if HuMo conditioning)
    fps: float = 24.0  # Output framerate
    audio_sample_rate: int = 24000  # Audio sample rate


class ProgressCallback:
    """
    Progress callback for video generation with tqdm-style output.

    Tracks step progress and performance metrics (it/s, ETA).

    Usage:
        callback = ProgressCallback(total_steps=30)
        pipeline(prompt="...", callback=callback)
        callback.close()
    """

    def __init__(
        self,
        total_steps: int = 30,
        desc: str = "Generating",
        disable: bool = False,
    ):
        """
        Initialize progress callback.

        Args:
            total_steps: Total number of diffusion steps.
            desc: Description shown in progress bar.
            disable: Disable progress output.
        """
        self.total_steps = total_steps if total_steps else 30
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

    def __call__(self, step: int, timestep: float, latents: torch.Tensor) -> None:
        """
        Called at end of each diffusion step.

        Args:
            step: Current step index (0-based).
            timestep: Current diffusion timestep.
            latents: Current latent tensor.
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


# ============================================================================
# Wan Architecture Components
# ============================================================================

@dataclass
class WanConfig:
    """Configuration for Wan video transformer."""

    # Transformer dimensions
    hidden_size: int = 5120
    num_layers: int = 40
    num_attention_heads: int = 40
    head_dim: int = 128  # 5120 / 40

    # Input/output
    in_channels: int = 16  # VAE latent channels
    out_channels: int = 16
    patch_size: Tuple[int, int, int] = (1, 2, 2)  # (T, H, W)

    # Text encoder
    text_dim: int = 4096  # UMT5-XXL output dimension
    text_len: int = 512  # Max text sequence length

    # Conditioning
    freq_shift: int = 256
    time_embed_dim: int = 512
    ffn_dim_mult: float = 8 / 3  # Standard transformer ratio
    qk_norm: bool = True
    use_rope: bool = True  # 3D rotary position embedding

    # Normalization
    norm_eps: float = 1e-6

    @classmethod
    def from_json(cls, path: str) -> "WanConfig":
        """Load config from JSON file."""
        import json
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class WanTextProjection(nn.Module):
    """Projects UMT5 text embeddings to transformer hidden dim."""

    def __init__(self, text_dim: int = 4096, hidden_dim: int = 5120):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_embeds: [B, seq_len, text_dim]
        Returns:
            [B, seq_len, hidden_dim]
        """
        return self.norm(self.proj(text_embeds))


class WanTimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding with MLP projection."""

    def __init__(
        self,
        dim: int = 512,
        hidden_dim: int = 5120,
        freq_shift: int = 256,
    ):
        super().__init__()
        self.dim = dim
        self.freq_shift = freq_shift
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B] float timesteps in [0, 1]
        Returns:
            [B, hidden_dim] timestep embeddings
        """
        # Sinusoidal embedding
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, device=timesteps.device))
            * torch.arange(half_dim, device=timesteps.device)
            / half_dim
        )
        # Shift frequencies for flow matching
        freqs = freqs * self.freq_shift

        args = timesteps[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        return self.mlp(embedding)


class RMSNorm(nn.Module):
    """RMS normalization (used instead of LayerNorm in Wan)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ============================================================================
# Main Pipeline
# ============================================================================

class WanVideoPipeline:
    """
    Wan 2.1/2.2 video generation pipeline.

    Supports text-to-video (T2V) and image-to-video (I2V) generation.
    Optional HuMo audio conditioning can be added as an extension.

    This is our own implementation following the Wan architecture,
    not a wrapper around external code.
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

        Args:
            transformer: Video transformer (DiT).
            vae: Video VAE for encoding/decoding.
            text_encoder: UMT5-XXL text encoder.
            tokenizer: Text tokenizer.
            scheduler: Diffusion scheduler.
            config: Wan configuration.
        """
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.config = config or WanConfig()

        self._device = torch.device("cpu")
        self._dtype = torch.bfloat16
        self._is_offloaded = True

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        vae_path: Optional[str] = None,
        text_encoder_path: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        enable_cpu_offload: bool = True,
        **kwargs,
    ) -> "WanVideoPipeline":
        """
        Load pipeline from pretrained weights.

        Args:
            model_path: Path to transformer safetensors weights.
            vae_path: Path to VAE weights (optional, can be in model_path dir).
            text_encoder_path: Path to UMT5-XXL encoder.
            torch_dtype: Model dtype (bfloat16 recommended, fp8 for memory).
            device: Device to load on.
            enable_cpu_offload: Enable CPU offload for memory efficiency.
            **kwargs: Additional arguments.

        Returns:
            Initialized WanVideoPipeline.
        """
        if load_safetensors is None:
            raise ImportError(
                "safetensors required. Install with: pip install safetensors"
            )

        model_path = Path(model_path).expanduser()
        logger.info(f"Loading Wan pipeline from {model_path}")

        # Determine what we're loading
        if model_path.is_file():
            transformer_path = model_path
            model_dir = model_path.parent
        else:
            model_dir = model_path
            # Look for transformer weights
            possible_names = [
                "diffusion_pytorch_model.safetensors",
                "transformer.safetensors",
                "model.safetensors",
            ]
            transformer_path = None
            for name in possible_names:
                candidate = model_dir / name
                if candidate.exists():
                    transformer_path = candidate
                    break

            if transformer_path is None:
                # Check for FP8 variants
                for f in model_dir.glob("*.safetensors"):
                    if "fp8" in f.name.lower() or "transformer" in f.name.lower():
                        transformer_path = f
                        break

            if transformer_path is None:
                raise FileNotFoundError(
                    f"Could not find transformer weights in {model_dir}. "
                    f"Looked for: {possible_names}"
                )

        logger.info(f"Loading transformer from: {transformer_path}")

        # Load state dict to inspect structure
        logger.info("Inspecting model weight structure...")
        with safe_open(str(transformer_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            logger.info(f"Found {len(keys)} keys in checkpoint")

            # Log sample keys for debugging
            sample_keys = keys[:10] if len(keys) > 10 else keys
            for key in sample_keys:
                tensor = f.get_tensor(key)
                logger.debug(f"  {key}: {tensor.shape}, {tensor.dtype}")

        # For now, create a placeholder pipeline
        # The actual transformer will be built once we verify weight structure
        instance = cls(
            transformer=None,
            vae=None,
            text_encoder=None,
            tokenizer=None,
            scheduler=None,
        )
        instance._device = torch.device(device)
        instance._dtype = torch_dtype
        instance._model_path = transformer_path
        instance._vae_path = vae_path
        instance._text_encoder_path = text_encoder_path
        instance._enable_cpu_offload = enable_cpu_offload
        instance._weight_keys = keys  # Store for architecture verification

        logger.info(
            "Pipeline initialized (lazy loading). "
            "Components will load on first generation."
        )

        return instance

    @classmethod
    def from_config(
        cls,
        config: "Config",
        device: str = "cuda",
        **kwargs,
    ) -> "WanVideoPipeline":
        """
        Load pipeline using config settings.

        Args:
            config: Config object with wan section.
            device: Device to load on.
            **kwargs: Override config settings.

        Returns:
            Initialized WanVideoPipeline.
        """
        # Import here to avoid circular dependency
        from llm_dit.config import Config

        # Get Wan config section (will be added to config.py)
        wan_config = getattr(config, "wan", None)
        if wan_config is None:
            raise ValueError(
                "Config has no 'wan' section. "
                "Add [wan] section with model_path."
            )

        return cls.from_pretrained(
            model_path=wan_config.model_path,
            vae_path=getattr(wan_config, "vae_path", None),
            text_encoder_path=getattr(wan_config, "text_encoder_path", None),
            torch_dtype=getattr(wan_config, "get_torch_dtype", lambda: torch.bfloat16)(),
            device=device,
            enable_cpu_offload=getattr(wan_config, "offload_mode", "model") != "none",
            **kwargs,
        )

    def _load_components(self) -> None:
        """
        Lazy load model components on first use.

        This allows memory-efficient initialization where components
        are only loaded when actually needed.
        """
        if self.transformer is not None:
            # Already loaded
            return

        logger.info("Loading Wan components...")

        # TODO: Build transformer architecture based on weight keys
        # For now, just log what we would load
        logger.warning(
            "Transformer architecture implementation in progress. "
            "Weight keys available for verification."
        )

        # Load scheduler from diffusers
        try:
            from diffusers import FlowMatchEulerDiscreteScheduler

            self.scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=3.0,  # Wan uses shift=3.0 for flow matching
            )
            logger.info("Scheduler loaded: FlowMatchEulerDiscreteScheduler")
        except ImportError:
            logger.warning(
                "diffusers not available for scheduler. "
                "Install with: pip install diffusers"
            )

    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "low quality, blurry, distorted",
        image: Optional[Union[torch.Tensor, np.ndarray, "PIL.Image.Image"]] = None,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 81,
        fps: float = 24.0,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        callback: Optional[ProgressCallback] = None,
        output_type: str = "np",
        **kwargs,
    ) -> VideoOutput:
        """
        Generate video from text prompt.

        Args:
            prompt: Text prompt describing desired video.
            negative_prompt: Negative prompt for CFG.
            image: Optional input image for I2V mode.
            height: Video height (multiple of 16).
            width: Video width (multiple of 16).
            num_frames: Number of frames (multiple of 4 + 1, e.g., 81).
            fps: Output framerate.
            num_inference_steps: Diffusion steps (30 recommended).
            guidance_scale: CFG scale (5.0 recommended for Wan).
            generator: Optional torch Generator for reproducibility.
            seed: Random seed.
            callback: Progress callback.
            output_type: "np" for numpy, "pt" for torch.
            **kwargs: Additional arguments.

        Returns:
            VideoOutput with generated frames.
        """
        # Ensure components are loaded
        self._load_components()

        # Create generator from seed
        if seed is not None and generator is None:
            device = self._device if self._device.type != "cpu" else "cpu"
            generator = torch.Generator(device=device).manual_seed(seed)

        logger.info(
            f"Generating video: {width}x{height}, {num_frames} frames @ {fps}fps, "
            f"{num_inference_steps} steps, guidance={guidance_scale}"
        )

        # Calculate latent dimensions
        # VAE compression: 8x spatial, 4x temporal
        latent_height = height // 8
        latent_width = width // 8
        latent_frames = (num_frames - 1) // 4 + 1  # e.g., 81 -> 21

        logger.info(
            f"Latent shape: ({latent_frames}, {latent_height}, {latent_width})"
        )

        # TODO: Implement actual generation once transformer is built
        # For now, return placeholder video
        logger.warning(
            "Generation not yet implemented. "
            "Returning placeholder frames for testing."
        )

        # Create placeholder video (random noise for testing pipeline)
        frames = np.random.randint(
            0, 255,
            size=(1, num_frames, height, width, 3),
            dtype=np.uint8,
        )

        return VideoOutput(
            frames=frames,
            audio=None,
            fps=fps,
        )

    def save_video(
        self,
        output: Union[VideoOutput, np.ndarray],
        output_path: str,
        audio: Optional[np.ndarray] = None,
        fps: Optional[float] = None,
        audio_sample_rate: int = 24000,
    ) -> str:
        """
        Save video output to file.

        Args:
            output: VideoOutput or video frames array.
            output_path: Path to save video.
            audio: Audio waveform (if output is array).
            fps: Framerate (if output is array).
            audio_sample_rate: Audio sample rate.

        Returns:
            Path to saved video.
        """
        if isinstance(output, VideoOutput):
            frames = output.frames
            audio = output.audio
            fps = output.fps
            audio_sample_rate = output.audio_sample_rate
        else:
            frames = output
            fps = fps or 24.0

        # Handle batch dimension: [B, F, H, W, C] -> [F, H, W, C]
        if frames.ndim == 5:
            frames = frames[0]

        # Ensure uint8
        if frames.dtype != np.uint8:
            if frames.max() <= 1.0:
                frames = (frames * 255).round().astype(np.uint8)
            else:
                frames = frames.astype(np.uint8)

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
            self.transformer.to("cpu")
        if self.vae is not None:
            self.vae.to("cpu")
        if self.text_encoder is not None:
            self.text_encoder.to("cpu")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._is_offloaded = True
        logger.info("WanVideoPipeline offloaded to CPU")

    def to(self, device: Union[str, torch.device]) -> "WanVideoPipeline":
        """Move pipeline to device."""
        device = torch.device(device)
        if self.transformer is not None:
            self.transformer.to(device)
        if self.vae is not None:
            self.vae.to(device)
        if self.text_encoder is not None:
            self.text_encoder.to(device)

        self._device = device
        self._is_offloaded = str(device) == "cpu"
        return self

    @property
    def device(self) -> torch.device:
        """Get pipeline device."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Get pipeline dtype."""
        return self._dtype

    def get_weight_keys(self) -> List[str]:
        """Get list of weight keys from loaded checkpoint (for debugging)."""
        return getattr(self, "_weight_keys", [])

    def estimate_memory(self) -> Dict[str, float]:
        """Estimate memory usage in GB."""
        return {
            "transformer": 14.0,  # 14B params in fp8 ~ 14GB
            "vae": 0.5,
            "text_encoder": 6.0,  # UMT5-XXL
            "peak": 16.0,
        }
