"""
LTX-2 Pipeline for text-to-video generation.

Last Updated: 2026-01-09

This pipeline wraps the diffusers LTX2Pipeline with our config integration
and memory management patterns optimized for RTX 4090 (24GB VRAM).

Key Features:
- Uses diffusers' LTX2Pipeline under the hood for proven stability
- Integrates with LTX2Config for settings management
- CPU offloading via enable_model_cpu_offload() for memory-constrained GPUs
- LoRA support via diffusers' LTXVideoLoraLoaderMixin
- Both video and audio output

Memory Strategy (24GB VRAM):
1. Enable model CPU offload (moves each component to GPU only when needed)
2. Sequential loading: encoder -> transformer -> VAE
3. Tiled VAE decode for large videos
4. FP8 quantized transformer (native on RTX 4090 SM89)

Example:
    from llm_dit.pipelines.ltx2 import LTX2Pipeline
    from llm_dit.config import Config

    config = Config.load("config.toml", profile="rtx4090")
    pipe = LTX2Pipeline.from_config(config)

    video, audio = pipe(
        prompt="A cat walking through a sunny garden",
        negative_prompt="blurry, distorted",
    )
    pipe.save_video(video, audio, "output.mp4")
"""

import gc
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import torch
import numpy as np

logger = logging.getLogger(__name__)


class ProgressCallback:
    """
    Progress callback for diffusers pipeline with tqdm-style output.

    Tracks step progress and performance metrics (it/s, ETA).

    Usage:
        callback = ProgressCallback(total_steps=12)
        pipeline(prompt="...", callback_on_step_end=callback)
        callback.close()
    """

    def __init__(
        self,
        total_steps: int,
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
        self.total_steps = total_steps
        self.desc = desc
        self.disable = disable
        self.current_step = 0
        self.start_time: Optional[float] = None
        self.step_times: list[float] = []
        self._last_step_time: Optional[float] = None
        self._pbar = None

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

    def __call__(self, pipe, step: int, timestep: int, callback_kwargs: dict) -> dict:
        """
        Called at end of each diffusion step.

        Args:
            pipe: Pipeline instance.
            step: Current step index (0-based).
            timestep: Current diffusion timestep.
            callback_kwargs: Additional callback arguments.

        Returns:
            callback_kwargs dict (unchanged).
        """
        if self.disable:
            return callback_kwargs

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
        pct = (self.current_step / self.total_steps) * 100
        bar_width = 30
        filled = int(bar_width * self.current_step / self.total_steps)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Print progress line (carriage return to overwrite)
        status = (
            f"\r{self.desc}: |{bar}| {self.current_step}/{self.total_steps} "
            f"[{self._format_time(elapsed)}<{self._format_time(eta)}, {its:.2f}it/s]"
        )
        print(status, end="", flush=True)

        # Newline on completion
        if self.current_step >= self.total_steps:
            print()  # Final newline

        return callback_kwargs

    def close(self) -> None:
        """Close progress bar (prints newline if needed)."""
        if self.current_step > 0 and self.current_step < self.total_steps:
            print()  # Ensure newline if interrupted

    def get_stats(self) -> dict:
        """
        Get performance statistics.

        Returns:
            Dict with elapsed, avg_step_time, its, step_times.
        """
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
class VideoOutput:
    """Output from LTX2 video generation."""

    frames: np.ndarray  # [B, F, H, W, C] uint8 video frames
    audio: Optional[np.ndarray]  # [B, samples] float audio waveform
    fps: float  # Output framerate
    audio_sample_rate: int  # Audio sample rate (typically 24000)


class LTX2Pipeline:
    """
    LTX-2 text-to-video generation pipeline.

    Wraps diffusers' LTX2Pipeline with our config system and memory
    management patterns for 24GB VRAM constraint.

    Supports:
    - Text-to-video generation
    - Image-to-video (I2V) with input image conditioning
    - Audio generation (joint video+audio)
    - LoRA fine-tuning
    - CPU offloading for memory efficiency
    """

    def __init__(
        self,
        pipe,  # diffusers LTX2Pipeline
        config: Optional["LTX2Config"] = None,
    ):
        """
        Initialize pipeline wrapper.

        Args:
            pipe: Diffusers LTX2Pipeline instance.
            config: Optional LTX2Config for generation defaults.
        """
        self._pipe = pipe
        self._config = config
        self._is_offloaded = False

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        enable_cpu_offload: bool = True,
        variant: Optional[str] = None,
        **kwargs,
    ) -> "LTX2Pipeline":
        """
        Load pipeline from pretrained model directory.

        Args:
            model_path: Path to model directory or HuggingFace model ID.
            torch_dtype: Model dtype (bfloat16 recommended).
            device: Device to load on ("cuda", "cpu").
            enable_cpu_offload: Enable model CPU offload for memory efficiency.
            variant: Model variant (e.g., "fp8" for FP8 quantized).
            **kwargs: Additional arguments for diffusers pipeline.

        Returns:
            Initialized LTX2Pipeline.
        """
        try:
            from diffusers import LTX2Pipeline as DiffusersLTX2Pipeline
        except ImportError:
            raise ImportError(
                "diffusers with LTX2Pipeline support required. "
                "Install with: pip install diffusers>=0.32.0"
            )

        logger.info(f"Loading LTX-2 pipeline from {model_path}")

        # Check if local text encoder is float32 (too big for 24GB)
        text_encoder_path = Path(model_path).expanduser() / "text_encoder"
        use_hf_encoder = False
        hf_encoder_id = "google/gemma-3-12b-it-qat-q4_0-unquantized"

        if text_encoder_path.exists():
            config_path = text_encoder_path / "config.json"
            if config_path.exists():
                import json
                with open(config_path) as f:
                    te_config = json.load(f)
                    if te_config.get("dtype") == "float32":
                        logger.warning(
                            f"Local text encoder is float32 (~50GB) - too large for 24GB VRAM. "
                            f"Loading Q4 version from HuggingFace instead: {hf_encoder_id}"
                        )
                        use_hf_encoder = True

        load_kwargs = {"torch_dtype": torch_dtype, "variant": variant, **kwargs}

        if use_hf_encoder:
            # Load text encoder in bfloat16 WITHOUT quantization
            # Let diffusers' sequential CPU offload handle memory management
            # Each layer is ~600MB, sequential offload loads one at a time
            try:
                from transformers import Gemma3ForConditionalGeneration, AutoTokenizer

                encoder_id = "google/gemma-3-12b-it"
                logger.info(f"Loading text encoder (bfloat16, no quant): {encoder_id}")
                logger.info("Sequential CPU offload will move layers one at a time (~600MB each)")

                text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
                    encoder_id,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    # No device_map - let diffusers handle placement
                )
                tokenizer = AutoTokenizer.from_pretrained(encoder_id)

                load_kwargs["text_encoder"] = text_encoder
                load_kwargs["tokenizer"] = tokenizer
                logger.info("Text encoder loaded to CPU (bfloat16)")
            except Exception as e:
                logger.error(f"Failed to load text encoder: {e}")
                raise

        pipe = DiffusersLTX2Pipeline.from_pretrained(model_path, **load_kwargs)

        # Enable CPU offload for memory efficiency
        if enable_cpu_offload and device != "cpu":
            # Use sequential offload - more aggressive, moves each layer individually
            # Slower (~3-5x) but required for 24GB with 12B encoder + 19B transformer
            logger.info("Enabling SEQUENTIAL CPU offload (layer-by-layer) for memory efficiency")
            pipe.enable_sequential_cpu_offload()
        elif device != "cpu":
            pipe.to(device)

        # Log memory status after loading
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"GPU memory after load: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

        return cls(pipe=pipe)

    @classmethod
    def from_single_file(
        cls,
        checkpoint_path: str,
        encoder_model_id: str = "google/gemma-3-12b-it-qat-q4_0-unquantized",
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        enable_cpu_offload: bool = True,
        **kwargs,
    ) -> "LTX2Pipeline":
        """
        Load pipeline from a single safetensors checkpoint.

        The LTX-2 checkpoint contains only transformer/VAE/vocoder weights.
        The Gemma 3-12B text encoder must be loaded separately and passed
        to diffusers.

        Args:
            checkpoint_path: Path to .safetensors checkpoint file.
            encoder_model_id: HuggingFace model ID for Gemma 3 text encoder.
                Defaults to the Q4 QAT quantized variant for memory efficiency.
            torch_dtype: Model dtype for transformer/VAE.
            device: Device to load on.
            enable_cpu_offload: Enable CPU offload for memory-constrained GPUs.
            **kwargs: Additional arguments passed to diffusers.

        Returns:
            Initialized LTX2Pipeline.
        """
        try:
            from diffusers import LTX2Pipeline as DiffusersLTX2Pipeline
        except ImportError:
            raise ImportError(
                "diffusers with LTX2Pipeline support required. "
                "Install with: pip install diffusers>=0.32.0"
            )

        try:
            from transformers import Gemma3ForConditionalGeneration, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers with Gemma3 support required. "
                "Install with: pip install transformers>=4.44.0"
            )

        # Load Gemma 3 text encoder and tokenizer separately
        # (not included in LTX-2 safetensors checkpoint)
        logger.info(f"Loading Gemma 3 text encoder: {encoder_model_id}")

        # Handle local paths - LTX-2 directory structure has text_encoder/ and tokenizer/ siblings
        encoder_path = Path(encoder_model_id).expanduser()
        if encoder_path.exists() and encoder_path.is_dir():
            # Local directory - check for sibling tokenizer/ directory
            tokenizer_path = encoder_path.parent / "tokenizer"
            if tokenizer_path.exists():
                logger.info(f"Using local tokenizer: {tokenizer_path}")
                tokenizer_source = str(tokenizer_path)
            else:
                # Tokenizer might be in same directory as encoder
                tokenizer_source = str(encoder_path)
            encoder_source = str(encoder_path)
        else:
            # HuggingFace model ID - both encoder and tokenizer from same ID
            encoder_source = encoder_model_id
            tokenizer_source = encoder_model_id

        text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
            encoder_source,
            torch_dtype=torch_dtype,
            # Start on CPU when offloading, diffusers will manage device placement
            device_map="cpu" if enable_cpu_offload else "auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

        logger.info(f"Loading LTX-2 transformer from checkpoint: {checkpoint_path}")

        # Pass pre-loaded encoder to diffusers
        pipe = DiffusersLTX2Pipeline.from_single_file(
            checkpoint_path,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            torch_dtype=torch_dtype,
            **kwargs,
        )

        if enable_cpu_offload and device != "cpu":
            pipe.enable_model_cpu_offload()
        elif device != "cpu":
            pipe.to(device)

        return cls(pipe=pipe)

    @classmethod
    def from_config(
        cls,
        config: "Config",
        device: str = "cuda",
        **kwargs,
    ) -> "LTX2Pipeline":
        """
        Load pipeline using LTX2Config settings.

        Args:
            config: Config object with ltx2 section.
            device: Device to load on.
            **kwargs: Override config settings.

        Returns:
            Initialized LTX2Pipeline.
        """
        from llm_dit.config import Config, LTX2Config

        ltx2_config: LTX2Config = config.ltx2

        # Validate config
        ltx2_config.validate()

        # Get model path
        model_path = ltx2_config.model_path
        if not model_path:
            raise ValueError("LTX2Config.model_path is required")

        model_dir = Path(model_path).expanduser()

        # Check if this is a full HuggingFace directory (has model_index.json)
        # If so, prefer from_pretrained() which reads configs properly
        model_index_path = model_dir / "model_index.json"
        if model_index_path.exists():
            logger.info(f"Found model_index.json - using from_pretrained() for proper config loading")
            pipe = cls.from_pretrained(
                str(model_dir),
                torch_dtype=ltx2_config.get_torch_dtype(),
                device=device,
                enable_cpu_offload=(ltx2_config.offload_mode != "none"),
                **kwargs,
            )
        else:
            # No model_index.json - try single file loading
            # Auto-detect local text_encoder if available
            encoder_model_id = ltx2_config.encoder_model_id
            local_encoder_path = model_dir / "text_encoder"
            if local_encoder_path.exists() and local_encoder_path.is_dir():
                logger.info(f"Found local text encoder: {local_encoder_path}")
                encoder_model_id = str(local_encoder_path)

            transformer_path = ltx2_config.get_model_file_path()

            if os.path.isfile(transformer_path):
                # Single file loading - Gemma 3 encoder loaded separately
                pipe = cls.from_single_file(
                    transformer_path,
                    encoder_model_id=encoder_model_id,
                    torch_dtype=ltx2_config.get_torch_dtype(),
                    device=device,
                    enable_cpu_offload=(ltx2_config.offload_mode != "none"),
                    **kwargs,
                )
            else:
                # Try as HuggingFace model ID
                pipe = cls.from_pretrained(
                    model_path,
                    torch_dtype=ltx2_config.get_torch_dtype(),
                    device=device,
                    enable_cpu_offload=(ltx2_config.offload_mode != "none"),
                    **kwargs,
                )

        pipe._config = ltx2_config

        # Load LoRA if configured
        if ltx2_config.lora_path:
            pipe.load_lora(ltx2_config.lora_path, scale=ltx2_config.lora_scale)

        return pipe

    def load_lora(
        self,
        lora_path: str,
        scale: float = 1.0,
        adapter_name: str = "default",
    ) -> None:
        """
        Load LoRA weights.

        Args:
            lora_path: Path to LoRA safetensors file.
            scale: LoRA blend scale (0.0-1.0).
            adapter_name: Name for the adapter.
        """
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA file not found: {lora_path}")

        logger.info(f"Loading LoRA from {lora_path} with scale={scale}")
        self._pipe.load_lora_weights(
            lora_path,
            adapter_name=adapter_name,
        )
        self._pipe.set_adapters([adapter_name], [scale])

    def unload_lora(self) -> None:
        """Unload all LoRA adapters."""
        self._pipe.unload_lora_weights()
        logger.info("LoRA weights unloaded")

    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "worst quality, blurry, distorted, inconsistent motion",
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        fps: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        enable_audio: Optional[bool] = None,
        output_type: str = "np",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable] = None,
        **kwargs,
    ) -> Union[VideoOutput, Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Generate video from text prompt.

        Args:
            prompt: Text prompt describing desired video.
            negative_prompt: Negative prompt for CFG.
            height: Video height (multiple of 32).
            width: Video width (multiple of 32).
            num_frames: Number of frames to generate.
            fps: Output framerate.
            num_inference_steps: Diffusion steps.
            guidance_scale: CFG scale (3.0-4.0 recommended).
            generator: Optional torch Generator for reproducibility.
            seed: Random seed (creates generator if provided).
            enable_audio: Generate audio (defaults to config setting).
            output_type: "np" for numpy, "pt" for torch, "pil" for PIL.
            return_dict: Return VideoOutput dataclass.
            callback_on_step_end: Callback for progress tracking.
            **kwargs: Additional arguments for diffusers pipeline.

        Returns:
            VideoOutput with frames and optional audio, or tuple.
        """
        # Use config defaults if not specified
        if self._config is not None:
            height = height or self._config.height
            width = width or self._config.width
            num_frames = num_frames or self._config.num_frames
            fps = fps or float(self._config.fps)
            num_inference_steps = num_inference_steps or self._config.get_total_steps()
            guidance_scale = guidance_scale or self._config.guidance_scale
            if enable_audio is None:
                enable_audio = self._config.audio_enabled
        else:
            # Sensible defaults
            height = height or 768
            width = width or 512
            num_frames = num_frames or 33
            fps = fps or 24.0
            num_inference_steps = num_inference_steps or 12
            guidance_scale = guidance_scale or 3.5
            enable_audio = enable_audio if enable_audio is not None else False

        # Create generator from seed if provided
        if seed is not None and generator is None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        logger.info(
            f"Generating video: {width}x{height}, {num_frames} frames @ {fps}fps, "
            f"{num_inference_steps} steps, guidance={guidance_scale}"
        )

        # Call diffusers pipeline
        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=fps,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type=output_type,
            return_dict=False,
            callback_on_step_end=callback_on_step_end,
            **kwargs,
        )

        # Unpack results (video, audio)
        video, audio = result

        # Get audio sample rate
        audio_sample_rate = getattr(
            self._pipe.vocoder.config if hasattr(self._pipe, "vocoder") else None,
            "output_sampling_rate",
            24000,
        )

        if return_dict:
            return VideoOutput(
                frames=video,
                audio=audio if enable_audio else None,
                fps=fps,
                audio_sample_rate=audio_sample_rate,
            )
        else:
            return video, audio if enable_audio else None

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
        try:
            from diffusers.pipelines.ltx2.export_utils import encode_video
        except ImportError:
            raise ImportError(
                "diffusers export_utils required. "
                "Install with: pip install diffusers>=0.32.0"
            )

        if isinstance(output, VideoOutput):
            frames = output.frames
            audio = output.audio
            fps = output.fps
            audio_sample_rate = output.audio_sample_rate
        else:
            frames = output
            fps = fps or 24.0

        # Ensure uint8
        if frames.dtype != np.uint8:
            frames = (frames * 255).round().astype(np.uint8)

        # Convert to torch for export
        video_tensor = torch.from_numpy(frames)

        # Handle batch dimension
        if video_tensor.ndim == 5:
            video_tensor = video_tensor[0]  # Take first batch

        # Prepare audio
        audio_tensor = None
        if audio is not None:
            audio_tensor = torch.from_numpy(audio).float()
            if audio_tensor.ndim == 2:
                audio_tensor = audio_tensor[0]  # Take first batch

        # Save video
        encode_video(
            video_tensor,
            fps=fps,
            audio=audio_tensor,
            audio_sample_rate=audio_sample_rate,
            output_path=output_path,
        )

        logger.info(f"Video saved to {output_path}")
        return output_path

    def offload(self) -> None:
        """Offload all models to CPU."""
        if hasattr(self._pipe, "to"):
            self._pipe.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._is_offloaded = True
        logger.info("LTX2Pipeline offloaded to CPU")

    def to(self, device: torch.device) -> "LTX2Pipeline":
        """Move pipeline to device."""
        self._pipe.to(device)
        self._is_offloaded = (str(device) == "cpu")
        return self

    @property
    def device(self) -> torch.device:
        """Get pipeline device."""
        if hasattr(self._pipe, "device"):
            return self._pipe.device
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        """Get pipeline dtype."""
        if hasattr(self._pipe, "dtype"):
            return self._pipe.dtype
        return torch.bfloat16

    def estimate_memory(self) -> dict:
        """Estimate memory usage based on config."""
        if self._config is not None:
            return self._config.estimate_vram_usage()
        return {
            "transformer": 19.0,
            "encoder": 6.0,
            "vae": 2.0,
            "peak": 21.0,
        }
