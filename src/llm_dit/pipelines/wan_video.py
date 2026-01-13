"""
Wan 2.1 Text-to-Video Pipeline.

Simple T2V implementation following DiffSynth-Engine.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class WanConfig:
    """Wan pipeline configuration."""

    model_path: str
    t5_path: Optional[str] = None
    vae_path: Optional[str] = None
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    fps: int = 16  # Wan default
    cfg_scale: float = 5.0
    num_inference_steps: int = 50
    shift: float = 5.0  # Flow matching shift


class WanVideoPipeline:
    """Wan 2.1 Text-to-Video Pipeline.

    Simple T2V following DiffSynth-Engine exactly.
    """

    def __init__(
        self,
        config: WanConfig,
        tokenizer,
        text_encoder: nn.Module,
        dit: nn.Module,
        vae: nn.Module,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.dit = dit
        self.vae = vae

        # VAE properties
        self.z_dim = 16  # Wan 2.1 latent channels
        self.upsampling_factor = 8  # Spatial downsampling

        # Scheduler
        self.sigmas = None
        self.timesteps = None

    @property
    def device(self) -> torch.device:
        return torch.device(self.config.device)

    @property
    def dtype(self) -> torch.dtype:
        return self.config.dtype

    # =========================================================================
    # Model Loading
    # =========================================================================

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "WanVideoPipeline":
        """Load Wan T2V pipeline from pretrained weights."""
        from safetensors.torch import load_file

        model_path = Path(model_path).expanduser()
        device = kwargs.get("device", "cuda")
        dtype = kwargs.get("torch_dtype", torch.bfloat16)

        config = WanConfig(
            model_path=str(model_path),
            device=device,
            dtype=dtype,
        )

        # Load DiT using from_pretrained
        dit_path = model_path / "diffusion_pytorch_model.safetensors"
        if not dit_path.exists():
            dit_path = model_path / "dit.safetensors"
        logger.info(f"Loading DiT from {dit_path}")

        from llm_dit.models.wan_dit import WanDiT

        # Determine config from path
        config_name = "wan2.1-t2v-1.3b"  # Default, could be detected from path
        if "14B" in str(model_path) or "14b" in str(model_path):
            config_name = "wan2.1-t2v-14b"
        dit = WanDiT.from_pretrained(str(dit_path), config_name=config_name, device=device, dtype=dtype)
        dit.eval()

        # Load VAE (check multiple possible locations)
        vae_path = kwargs.get("vae_path")
        if vae_path is None:
            for candidate in [
                model_path / "Wan2.1_VAE.safetensors",
                model_path / "vae.safetensors",
                model_path.parent / "Wan2.1-VAE" / "vae.safetensors",
            ]:
                if candidate.exists():
                    vae_path = candidate
                    break
        if vae_path is None or not Path(vae_path).exists():
            raise FileNotFoundError(f"VAE not found. Searched in {model_path}")
        logger.info(f"Loading VAE from {vae_path}")

        from llm_dit.models.wan_vae import WanVAE

        vae = WanVAE(dtype=dtype)
        vae_state = load_file(str(vae_path))
        vae.model.load_state_dict(vae_state, strict=False)
        vae = vae.to(device=device, dtype=dtype)
        vae.eval()

        # Load Text Encoder (check multiple possible locations)
        t5_path = kwargs.get("t5_path")
        if t5_path is None:
            for candidate in [
                model_path / "models_t5_umt5-xxl-enc-bf16.safetensors",
                model_path / "umt5.safetensors",
                model_path.parent / "Wan2.1-UMT5" / "umt5.safetensors",
                model_path / "models_t5_umt5-xxl-enc" / "model.safetensors",
            ]:
                if candidate.exists():
                    t5_path = candidate
                    break
        if t5_path is None or not Path(t5_path).exists():
            raise FileNotFoundError(f"T5 encoder not found. Searched in {model_path}")
        logger.info(f"Loading T5 from {t5_path}")

        from llm_dit.models.wan_text_encoder import WanTextEncoder

        text_encoder = WanTextEncoder(dtype=dtype)
        text_encoder.load_weights(str(t5_path))

        # Load tokenizer (check multiple possible locations)
        tokenizer_path = kwargs.get("tokenizer_path")
        if tokenizer_path is None:
            for candidate in [
                model_path / "google" / "umt5-xxl",  # Wan2.1 bundled tokenizer
                model_path / "google",
                model_path / "models_t5_umt5-xxl-enc",
                model_path.parent / "Wan2.1-UMT5",
            ]:
                if candidate.exists() and (candidate / "tokenizer.json").exists():
                    tokenizer_path = candidate
                    break
        if tokenizer_path and Path(tokenizer_path).exists():
            text_encoder.load_tokenizer(str(tokenizer_path))
        else:
            # Fall back to HuggingFace hub
            logger.info("Tokenizer not found locally, loading from google/umt5-xxl")
            text_encoder.load_tokenizer("google/umt5-xxl")

        # Keep text encoder on CPU for memory efficiency
        # It will be moved to GPU temporarily during encode_prompt
        text_encoder.model = text_encoder.model.to(dtype=dtype)
        text_encoder.eval()

        return cls(
            config=config,
            tokenizer=text_encoder.tokenizer,
            text_encoder=text_encoder,
            dit=dit,
            vae=vae,
        )

    # =========================================================================
    # Text Encoding
    # =========================================================================

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode text prompt - matches DiffSynth-Studio pattern."""
        # Move text encoder to GPU temporarily
        self.text_encoder.model = self.text_encoder.model.to(self.device)

        # Text encoder handles tokenization internally
        prompt_emb, mask = self.text_encoder.encode(prompt)

        # Zero padding positions (DiffSynth-Studio pattern)
        prompt_emb = prompt_emb.masked_fill(mask.unsqueeze(-1).expand_as(prompt_emb) == 0, 0)

        # Move back to CPU to free GPU memory
        self.text_encoder.model = self.text_encoder.model.to("cpu")
        torch.cuda.empty_cache()

        return prompt_emb

    # =========================================================================
    # Scheduler
    # =========================================================================

    def set_timesteps(self, num_inference_steps: int):
        """Set scheduler timesteps - matches DiffSynth-Studio."""
        shift = self.config.shift
        sigma_min = 0.001
        sigma_max = 0.999

        # Linear spacing - DiffSynth-Studio pattern: n+1 points, drop last
        # This gives different step distribution than linspace(max, min, n)
        sigmas = torch.linspace(sigma_max, sigma_min, num_inference_steps + 1)[:-1]

        # Apply shift
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        # Timesteps computed BEFORE appending 0 (DiffSynth-Studio pattern)
        self.timesteps = (sigmas * 1000).to(self.device)

        # Append 0 at end for final step
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
        self.sigmas = sigmas.to(self.device)

    def step(self, latents: torch.Tensor, noise_pred: torch.Tensor, step_idx: int) -> torch.Tensor:
        """Euler step - matches DiffSynth-Studio exactly."""
        sigma = self.sigmas[step_idx]
        sigma_next = self.sigmas[step_idx + 1]
        # No dtype conversion - DiffSynth-Studio does arithmetic in original dtype
        return latents + noise_pred * (sigma_next - sigma)

    # =========================================================================
    # Noise Prediction
    # =========================================================================

    def predict_noise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Run DiT forward pass."""
        return self.dit(
            hidden_states=latents.to(self.dtype),
            timestep=timestep.to(self.dtype),
            encoder_hidden_states=context.to(self.dtype),
        )

    def predict_noise_with_cfg(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_emb_posi: torch.Tensor,
        prompt_emb_nega: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        """Classifier-free guidance."""
        if cfg_scale <= 1.0:
            return self.predict_noise(latents, timestep, prompt_emb_posi)

        # Batched CFG
        latents_batch = torch.cat([latents, latents], dim=0)
        context_batch = torch.cat([prompt_emb_posi, prompt_emb_nega], dim=0)
        timestep_batch = timestep

        noise_pred = self.predict_noise(latents_batch, timestep_batch, context_batch)

        noise_pred_posi, noise_pred_nega = noise_pred.chunk(2)
        noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)

        return noise_pred

    # =========================================================================
    # VAE
    # =========================================================================

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video to latents."""
        return self.vae.encode(video.to(self.dtype))

    def decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to video in float32 for temporal stability.

        The VAE uses causal convolutions with feat_cache passed between frames.
        Running in bfloat16 causes quantization noise to accumulate frame-over-frame,
        causing flickering. Float32 decode prevents this drift.

        This matches DiffSynth-Engine which defaults VAE to float32.
        """
        # Store original dtype (both model weights and autocast control)
        original_dtype = self.vae.dtype
        device = latents.device

        # Convert VAE to float32 - both weights AND autocast dtype
        self.vae.to(dtype=torch.float32)
        self.vae.dtype = torch.float32  # Controls internal autocast

        # Run decode in float32
        video = self.vae.decode(latents.float())

        # Restore VAE to original dtype to save VRAM
        self.vae.to(dtype=original_dtype)
        self.vae.dtype = original_dtype

        return video

    # =========================================================================
    # Main Generation
    # =========================================================================

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate video from text prompt.

        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            height: Video height (must be divisible by 16)
            width: Video width (must be divisible by 16)
            num_frames: Number of frames (must be 4N+1)
            num_inference_steps: Denoising steps
            cfg_scale: CFG scale
            seed: Random seed

        Returns:
            Video frames as numpy array [B, F, H, W, C] uint8
        """
        # Defaults
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        cfg_scale = cfg_scale or self.config.cfg_scale

        # Validation
        assert height % 16 == 0, "height must be divisible by 16"
        assert width % 16 == 0, "width must be divisible by 16"
        assert (num_frames - 1) % 4 == 0, "num_frames must be 4N+1"

        # Latent dimensions
        latent_frames = (num_frames - 1) // 4 + 1
        latent_height = height // self.upsampling_factor
        latent_width = width // self.upsampling_factor

        # Initialize noise
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        latents = torch.randn(
            (1, self.z_dim, latent_frames, latent_height, latent_width),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )

        # Encode prompts
        logger.info("Encoding prompts...")
        prompt_emb_posi = self.encode_prompt(prompt)
        prompt_emb_nega = self.encode_prompt(negative_prompt)

        # Setup scheduler
        self.set_timesteps(num_inference_steps)

        # Denoise
        logger.info(f"Denoising {num_inference_steps} steps...")
        for i, timestep in enumerate(tqdm(self.timesteps, desc="Denoising")):
            timestep = timestep.unsqueeze(0).to(self.device)

            noise_pred = self.predict_noise_with_cfg(
                latents=latents,
                timestep=timestep,
                prompt_emb_posi=prompt_emb_posi,
                prompt_emb_nega=prompt_emb_nega,
                cfg_scale=cfg_scale,
            )

            latents = self.step(latents, noise_pred, i)

        # Decode
        logger.info("Decoding video...")
        video = self.decode_video(latents)

        # Trim to requested frame count (nuclear fix produces extra frames)
        if video.shape[2] > num_frames:
            video = video[:, :, :num_frames, :, :]

        # To numpy uint8
        video = video.float().cpu()
        video = (video + 1) / 2  # [-1, 1] -> [0, 1]
        video = video.clamp(0, 1)
        video = (video * 255).round().to(torch.uint8)
        video = video.permute(0, 2, 3, 4, 1).numpy()  # [B, C, F, H, W] -> [B, F, H, W, C]

        return video

    # =========================================================================
    # Video Saving
    # =========================================================================

    def save_video(
        self,
        video: np.ndarray,
        path: str,
        fps: Optional[int] = None,
    ) -> str:
        """Save video to file.

        Args:
            video: Video frames [B, F, H, W, C] or [F, H, W, C]
            path: Output path
            fps: Frame rate (default: config.fps)

        Returns:
            Path to saved video
        """
        fps = fps or self.config.fps

        # Handle batch dimension
        if video.ndim == 5:
            video = video[0]  # Take first batch

        try:
            import imageio.v3 as iio

            codec = "libvpx-vp9" if path.endswith(".webm") else "libx264"
            with iio.imopen(path, "w", plugin="FFMPEG") as writer:
                writer.write(video, fps=fps, codec=codec)
            logger.info(f"Saved video to {path}")
        except Exception as e:
            logger.warning(f"imageio failed: {e}, trying torchvision")
            import torchvision.io as tvio

            video_tensor = torch.from_numpy(video)
            tvio.write_video(path, video_tensor, fps=fps)
            logger.info(f"Saved video to {path}")

        return path

    # =========================================================================
    # Memory Management
    # =========================================================================

    def to(self, device: str) -> "WanVideoPipeline":
        """Move pipeline to device."""
        self.config.device = device
        self.dit = self.dit.to(device)
        self.vae = self.vae.to(device)
        self.text_encoder = self.text_encoder.to(device)
        return self

    def offload(self) -> None:
        """Offload models to CPU."""
        import gc

        self.dit = self.dit.to("cpu")
        self.vae = self.vae.to("cpu")
        self.text_encoder = self.text_encoder.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
