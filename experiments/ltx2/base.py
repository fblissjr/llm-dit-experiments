"""
LTX-2 Experiment Base Class.

Last Updated: 2026-01-17

Provides shared infrastructure for all LTX-2 experiments:
- Model loading with memory tracking
- Encoding with 8-bit quantization and layer masking
- Video generation with configurable options
- Scoring with SigLIP
- Output saving with metadata
- Memory cleanup between iterations

Experiments should inherit from LTX2ExperimentBase and override:
- setup(): Experiment-specific initialization
- run_iteration(config): Single experiment iteration
- aggregate_results(results): Combine iteration results

Usage:
    from experiments.ltx2.base import LTX2ExperimentBase

    class MyExperiment(LTX2ExperimentBase):
        def __init__(self):
            super().__init__("my_experiment")

        def setup(self):
            self.load_model()
            self.load_encoder()

        def run_iteration(self, config: dict) -> dict:
            embeds = self.encode(config["prompt"])
            video = self.generate_video(embeds)
            score = self.score_video(video, config["prompt"])
            return {"score": score, **config}

    # Run experiment
    exp = MyExperiment()
    results = exp.run([
        {"prompt": "A cat sleeping"},
        {"prompt": "A dog running"},
    ])
"""

import json
import logging
import math
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm

# LTX-2 specific imports
from llm_dit.models.ltx2 import Modality

# Core library imports
from llm_dit.utils.memory import MemoryTracker, cleanup_memory, log_memory_usage
from llm_dit.utils.metrics import SigLIPScorer
from llm_dit.data import get_all_prompts

logger = logging.getLogger(__name__)


class LTX2ExperimentBase(ABC):
    """
    Shared infrastructure for all LTX-2 experiments.

    Handles model loading, encoding, generation, scoring, and output saving.
    Experiments focus ONLY on their specific logic.

    Attributes:
        experiment_name: Name used for output directories
        output_dir: Path to output directory
        model: LTX2Transformer (lazy loaded)
        encoder: Gemma3Encoder (lazy loaded)
        scorer: SigLIPScorer (lazy loaded)
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "outputs",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize experiment base.

        Args:
            experiment_name: Name for this experiment (used in output paths)
            output_dir: Base output directory
            device: Compute device
            dtype: Model dtype
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.device = device
        self.dtype = dtype

        # Lazy-loaded components
        self.model = None
        self.encoder = None
        self.scorer = None
        self.vae = None
        self.pipeline = None
        self.connectors = None  # LTX2TextConnectors for pure PyTorch path

        # Experiment state
        self.run_timestamp = None
        self.run_dir = None

    # =========================================================================
    # Core Infrastructure (shared)
    # =========================================================================

    def load_model(
        self,
        model_path: str = "models/LTX-2/transformer/",
        use_pure_pytorch: bool = True,
        use_group_offloading: bool = False,
        num_blocks_per_group: int = 1,
    ) -> None:
        """
        Load LTX2 transformer with memory tracking.

        Args:
            model_path: Path to model weights
            use_pure_pytorch: If True, use our pure PyTorch model.
                             If False, use diffusers model.
            use_group_offloading: If True and use_pure_pytorch=False, apply
                                 group offloading for memory efficiency (~5GB VRAM).
                                 Streams transformer blocks from CPU during generation.
            num_blocks_per_group: Blocks per offload group (1=min VRAM, higher=faster).
        """
        with MemoryTracker("Load transformer"):
            if use_pure_pytorch:
                from llm_dit.models import load_ltx2_transformer
                self.model = load_ltx2_transformer(
                    model_path,
                    dtype=self.dtype,
                    device="cpu",  # Load to CPU first
                )
                self.model = self.model.to(self.device)
            else:
                # Use diffusers model via pipeline
                from diffusers import LTX2Pipeline
                self.pipeline = LTX2Pipeline.from_pretrained(
                    "models/LTX-2",
                    torch_dtype=self.dtype,
                    text_encoder=None,  # We handle encoding separately
                    tokenizer=None,
                )

                if use_group_offloading:
                    # Apply group offloading for memory-constrained GPUs (e.g., RTX 4090 24GB)
                    from diffusers.hooks import apply_group_offloading
                    import torch

                    apply_group_offloading(
                        self.pipeline.transformer,
                        onload_device=torch.device("cuda"),
                        offload_device=torch.device("cpu"),
                        offload_type="block_level",
                        num_blocks_per_group=num_blocks_per_group,
                        use_stream=True,
                        non_blocking=True,
                    )

                    # Keep VAE and connectors on GPU (small enough)
                    self.pipeline.vae.to("cuda")
                    if self.pipeline.connectors is not None:
                        self.pipeline.connectors.to("cuda")
                else:
                    self.pipeline.to(self.device)

                self.model = self.pipeline.transformer

        log_memory_usage("After loading transformer")

    def load_encoder(
        self,
        model_path: str = "models/LTX-2/text_encoder/",
        use_8bit: bool = True,
    ) -> None:
        """
        Load Gemma3 text encoder (8-bit by default for memory).

        Args:
            model_path: Path to text encoder
            use_8bit: Use 8-bit quantization (recommended for RTX 4090)
        """
        with MemoryTracker("Load encoder"):
            from llm_dit.encoders import Gemma3Encoder
            self.encoder = Gemma3Encoder(
                model_id=model_path,  # Gemma3Encoder expects model_id
                load_in_8bit=use_8bit,
                device=self.device,
            )

        log_memory_usage("After loading encoder")

    def load_vae(self, model_path: str = "models/LTX-2/vae/") -> None:
        """Load VAE for decoding latents to video."""
        with MemoryTracker("Load VAE"):
            from diffusers import AutoencoderKLLTXVideo
            self.vae = AutoencoderKLLTXVideo.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
            )
            self.vae = self.vae.to(self.device)

        log_memory_usage("After loading VAE")

    def load_connectors(self, model_path: str = "models/LTX-2/connectors/") -> None:
        """
        Load LTX2TextConnectors for pure PyTorch path.

        The connectors process packed text embeddings through:
        1. text_proj_in: Linear(188160 -> 3840)
        2. video_connector: 2-block transformer with 128 thinking tokens
        3. (optional) audio_connector for audio generation

        Args:
            model_path: Path to connector weights directory
        """
        with MemoryTracker("Load connectors"):
            from llm_dit.models import load_ltx2_connectors
            self.connectors = load_ltx2_connectors(
                model_path,
                device=self.device,
                dtype=self.dtype,
            )

        log_memory_usage("After loading connectors")

    def load_scorer(self) -> None:
        """Load SigLIP scorer for text-video alignment."""
        if self.scorer is not None:
            return

        with MemoryTracker("Load scorer"):
            self.scorer = SigLIPScorer(device=self.device, dtype=self.dtype)

    def encode_packed(
        self,
        prompt: str,
        active_layers: Optional[List[int]] = None,
        masking_mode: str = "soft",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prompt to packed multi-layer format [B, T, 188160].

        This format is suitable for passing to the connectors' text_proj_in,
        which projects to [B, T, 3840] before the video_connector transformer.

        Args:
            prompt: Text prompt
            active_layers: Optional list of layer indices to keep active (0-48).
                          Used for layer ablation experiments.
            masking_mode: How to handle inactive layers ("soft", "zero", "weighted").

        Returns:
            Tuple of:
                - packed_embeds: [B, T, 188160] packed multi-layer embeddings
                - attention_mask: [B, T] attention mask
        """
        if self.encoder is None:
            raise RuntimeError("Encoder not loaded. Call load_encoder() first.")

        if active_layers is not None:
            # Layer ablation: keep only specified layers active
            result = self.encoder.encode_with_layer_masking(
                prompt,
                active_layers=active_layers,
                masking_mode=masking_mode,
                return_packed=True,
            )
            return result['prompt_embeds'], result['attention_mask']
        else:
            # Standard encoding - get packed format
            result = self.encoder.encode_multilayer(
                prompt,
                layer_indices=None,  # All layers
                return_projected=False,
            )
            from llm_dit.encoders.gemma3 import pack_text_embeds
            packed = pack_text_embeds(
                result['layer_stack'],
                sequence_length=result['seq_lengths'][0],
                device=self.encoder.device,
            )
            return packed, result['attention_mask']

    def encode(
        self,
        prompt: str,
        layer_weights: Optional[torch.Tensor] = None,
        active_layers: Optional[List[int]] = None,
        masking_mode: str = "soft",
        return_packed: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Encode prompt to pipeline-ready embeddings.

        Args:
            prompt: Text prompt
            layer_weights: Optional per-layer weights [49] for weighted blending.
                          Layers with 0 weight are excluded.
            active_layers: Optional list of layer indices to keep active (0-48).
                          Used for layer ablation experiments.
            masking_mode: How to handle inactive layers ("soft", "zero", "weighted").
                         Only used when active_layers is set.
            return_packed: If True, return pipeline-ready packed embeddings.
            **kwargs: Additional encoder arguments

        Returns:
            Prompt embeddings tensor ready for pipeline (if return_packed=True)
            or raw encoding output
        """
        if self.encoder is None:
            raise RuntimeError("Encoder not loaded. Call load_encoder() first.")

        if active_layers is not None:
            # Layer ablation: keep only specified layers active
            result = self.encoder.encode_with_layer_masking(
                prompt,
                active_layers=active_layers,
                masking_mode=masking_mode,
                return_packed=return_packed,
            )
            return result['prompt_embeds'] if return_packed else result

        elif layer_weights is not None:
            # Weighted blending: apply per-layer weights
            # Convert weights to active layers list (non-zero weights)
            active = [i for i in range(len(layer_weights)) if layer_weights[i] > 0]
            result = self.encoder.encode_multilayer(
                prompt,
                layer_indices=active if len(active) < 49 else None,
                return_projected=False,
            )

            # Apply weights to layer stack
            hidden_states = result['layer_stack']
            weights = layer_weights.to(hidden_states.device).view(1, 1, 1, -1)
            if len(active) < 49:
                # Map weights to selected layers
                weights = layer_weights[active].to(hidden_states.device).view(1, 1, 1, -1)
            hidden_states = hidden_states * weights

            if return_packed:
                from llm_dit.encoders.gemma3 import pack_text_embeds
                return pack_text_embeds(
                    hidden_states,
                    sequence_length=result['seq_lengths'][0],
                    device=self.encoder.device,
                )
            return hidden_states

        else:
            # Standard encoding
            result = self.encoder.encode(prompt, return_padded=True, **kwargs)
            if return_packed:
                # Return padded embeddings for pipeline
                return result.padded_embeddings
            return result

    def _create_position_indices(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Create 3D position indices [B, 3, T] for video (temporal, height, width).

        LTX-2 VAE compression: 32x spatial, 8x temporal.
        Reference: coderef/LTX-2/ltx-core/components/patchifiers.py

        Args:
            batch_size: Batch size
            num_frames: Number of video frames
            height: Video height in pixels
            width: Video width in pixels

        Returns:
            Position indices tensor [B, 3, T] where T = t_latent * h_latent * w_latent
        """
        # Compute latent dimensions with LTX-2's compression ratios
        t_latent = (num_frames - 1) // 8 + 1
        h_latent = height // 32
        w_latent = width // 32

        # Create meshgrid of position indices
        t_indices = torch.arange(t_latent, device=self.device)
        h_indices = torch.arange(h_latent, device=self.device)
        w_indices = torch.arange(w_latent, device=self.device)

        # Create 3D grid: [t_latent, h_latent, w_latent]
        # Order is (t, h, w) matching the official implementation
        grid_t, grid_h, grid_w = torch.meshgrid(t_indices, h_indices, w_indices, indexing='ij')

        # Flatten and stack to [3, T]
        positions = torch.stack([
            grid_t.flatten(),
            grid_h.flatten(),
            grid_w.flatten(),
        ], dim=0)  # [3, T]

        # Expand for batch: [B, 3, T]
        positions = positions.unsqueeze(0).expand(batch_size, -1, -1)

        return positions

    def _create_video_modality(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        positions: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> Modality:
        """
        Create Modality dataclass for transformer input.

        Bundles the latent tokens, timestep embeddings, positional information,
        and text conditioning for the diffusion transformer.

        Args:
            latent: [B, T, D] latent tokens (D=128 for LTX-2)
            timestep: [B, T] per-token timesteps
            positions: [B, 3, T] position indices
            prompt_embeds: [B, seq_len, context_dim] text embeddings

        Returns:
            Modality dataclass ready for transformer forward pass
        """
        return Modality(
            latent=latent,
            timesteps=timestep,
            positions=positions,
            context=prompt_embeds,
            enabled=True,
            context_mask=None,
        )

    def generate_video(
        self,
        prompt_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_frames: int = 33,
        height: int = 512,
        width: int = 768,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        seed: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate video from embeddings.

        Args:
            prompt_embeds: Text embeddings from encode() or encode_packed()
                          - If [B, T, 188160]: passes through connectors (if loaded)
                          - If [B, T, 3840]: passes directly to DiT caption_projection
            attention_mask: Optional attention mask [B, T] for connector processing
            num_frames: Number of video frames
            height: Video height
            width: Video width
            num_inference_steps: Diffusion steps
            guidance_scale: CFG scale
            seed: Random seed (None for random)
            **kwargs: Additional generation arguments

        Returns:
            Video tensor [F, H, W, C]
        """
        if self.pipeline is None and self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Use pipeline if available, otherwise use pure PyTorch model
        if self.pipeline is not None:
            # Build pipeline arguments
            pipeline_kwargs = {
                "prompt_embeds": prompt_embeds,
                "num_frames": num_frames,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
                **kwargs,
            }
            # Add attention mask if provided (required when using prompt_embeds)
            if attention_mask is not None:
                pipeline_kwargs["prompt_attention_mask"] = attention_mask

            output = self.pipeline(**pipeline_kwargs)
            return output.frames[0]
        else:
            # Pure PyTorch generation loop
            # Reference: coderef/LTX-2/ltx-core/components/ for scheduler & diffusion steps

            if self.vae is None:
                raise RuntimeError("VAE not loaded. Call load_vae() first for pure PyTorch mode.")

            # =====================================================================
            # Step 0: Process embeddings through connectors if available
            # =====================================================================
            # Check if embeddings are packed format [B, T, 188160]
            embed_dim = prompt_embeds.shape[-1]
            if embed_dim == 188160 and self.connectors is not None:
                # Process through connectors: text_proj_in + video_connector
                if attention_mask is None:
                    # Create all-ones mask if not provided
                    attention_mask = torch.ones(
                        prompt_embeds.shape[0], prompt_embeds.shape[1],
                        device=prompt_embeds.device, dtype=prompt_embeds.dtype
                    )
                video_embeds, _, new_mask = self.connectors(
                    prompt_embeds,
                    attention_mask,
                    additive_mask=False,
                )
                prompt_embeds = video_embeds
            elif embed_dim == 188160 and self.connectors is None:
                raise RuntimeError(
                    "Packed embeddings [B, T, 188160] require connectors. "
                    "Call load_connectors() first, or use encode() instead of encode_packed()."
                )

            # =====================================================================
            # Step 1: Compute latent dimensions (32x spatial, 8x temporal compression)
            # =====================================================================
            t_latent = (num_frames - 1) // 8 + 1
            h_latent = height // 32
            w_latent = width // 32
            num_tokens = t_latent * h_latent * w_latent

            # =====================================================================
            # Step 2: Initialize noise [B, T, D] where D=128 (VAE latent channels)
            # =====================================================================
            latents = torch.randn(
                (1, num_tokens, 128),
                generator=generator,
                device=self.device,
                dtype=self.dtype,
            )

            # =====================================================================
            # Step 3: Create position indices
            # =====================================================================
            positions = self._create_position_indices(1, num_frames, height, width)

            # =====================================================================
            # Step 4: Set up sigma schedule with dynamic shift
            # LTX-2 uses resolution-dependent shift for better results
            # Reference: coderef/LTX-2/ltx-core/components/schedulers.py
            # =====================================================================
            video_seq_len = num_tokens
            base_seq_len, max_seq_len = 1024, 4096
            base_shift, max_shift = 0.95, 2.05

            # Linear interpolation for shift based on resolution
            m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
            mu = base_shift + m * (video_seq_len - base_seq_len)
            mu = max(min(mu, max_shift), base_shift)  # Clamp to valid range

            # Linear sigmas with exponential shift (time warping)
            sigmas = torch.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
            exp_mu = math.exp(mu)
            sigmas = exp_mu / (exp_mu + (1.0 / sigmas - 1.0))
            sigmas = sigmas.to(self.device, dtype=self.dtype)

            # =====================================================================
            # Step 5: Denoising loop (Euler method with velocity prediction)
            # Reference: coderef/LTX-2/ltx-core/components/diffusion_steps.py
            # =====================================================================
            for i in tqdm(range(len(sigmas)), desc="Denoising"):
                sigma = sigmas[i]
                sigma_next = sigmas[i + 1] if i + 1 < len(sigmas) else torch.tensor(0.0, device=self.device, dtype=self.dtype)

                # Timestep for model in [0, 1000] range (LTX-2 convention)
                timestep = (sigma * 1000).expand(1, num_tokens)

                # Classifier-free guidance: compute both conditional and unconditional
                if guidance_scale > 1.0:
                    # Unconditional pass (zero embeddings)
                    uncond_embeds = torch.zeros_like(prompt_embeds)
                    uncond_modality = self._create_video_modality(latents, timestep, positions, uncond_embeds)
                    velocity_uncond, _ = self.model(video=uncond_modality)

                    # Conditional pass
                    cond_modality = self._create_video_modality(latents, timestep, positions, prompt_embeds)
                    velocity_cond, _ = self.model(video=cond_modality)

                    # CFG blend: LTX-2 uses cond + (scale - 1) * (cond - uncond)
                    # This is equivalent to: uncond + scale * (cond - uncond)
                    velocity = velocity_cond + (guidance_scale - 1.0) * (velocity_cond - velocity_uncond)
                else:
                    modality = self._create_video_modality(latents, timestep, positions, prompt_embeds)
                    velocity, _ = self.model(video=modality)

                # Euler step: x_{t-1} = x_t + v * dt where dt = sigma_next - sigma
                # Note: dt is negative (moving toward clean), so this subtracts noise
                dt = sigma_next - sigma
                latents = latents + velocity * dt

            # =====================================================================
            # Step 6: Reshape latents for VAE decode
            # From [B, T, D] to [B, D, T_lat, H_lat, W_lat]
            # =====================================================================
            latents = latents.transpose(1, 2)  # [B, D, T]
            latents = latents.reshape(1, 128, t_latent, h_latent, w_latent)

            # =====================================================================
            # Step 7: CRITICAL - Denormalize latents before VAE decode
            # The VAE has learned normalization parameters that must be reversed
            # Reference: diffusers/pipelines/ltx2/pipeline_ltx2.py:588-601
            # =====================================================================
            latents_mean = self.vae.latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
            latents_std = self.vae.latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
            scaling_factor = self.vae.config.scaling_factor
            latents = latents * latents_std / scaling_factor + latents_mean

            # =====================================================================
            # Step 8: VAE decode to pixel space
            # =====================================================================
            with torch.no_grad():
                video = self.vae.decode(latents).sample

            # =====================================================================
            # Step 9: Convert to uint8 [F, H, W, C] format for export_to_video
            # =====================================================================
            video = video.squeeze(0).permute(1, 2, 3, 0)  # [B, C, T, H, W] -> [T, H, W, C]
            video = ((video + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)

            return video

    def score_video(
        self,
        video: torch.Tensor,
        prompt: str,
        sample_frames: int = 8,
    ) -> float:
        """
        Score video-text alignment using SigLIP.

        Args:
            video: Video tensor [F, H, W, C]
            prompt: Text prompt
            sample_frames: Number of frames to sample

        Returns:
            Mean alignment score across frames
        """
        self.load_scorer()
        _, mean_score = self.scorer.score_video(video, prompt, sample_rate=len(video) // sample_frames)
        return mean_score

    def save_video(
        self,
        video: torch.Tensor,
        name: str,
        prompt: str,
        metadata: Optional[Dict] = None,
    ) -> Path:
        """
        Save video with metadata.

        Args:
            video: Video tensor [F, H, W, C]
            name: Filename (without extension)
            prompt: Text prompt
            metadata: Additional metadata to save

        Returns:
            Path to saved video
        """
        if self.run_dir is None:
            self._init_run_dir()

        video_path = self.run_dir / f"{name}.mp4"
        meta_path = self.run_dir / f"{name}.json"

        # Save video
        try:
            from diffusers.utils import export_to_video
            export_to_video(video, str(video_path))
        except ImportError:
            # Fallback: save as numpy
            import numpy as np
            np.save(str(video_path).replace('.mp4', '.npy'), video.cpu().numpy())

        # Save metadata
        meta = {
            "prompt": prompt,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {}),
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        return video_path

    def cleanup(self) -> None:
        """Clean up memory between iterations."""
        cleanup_memory()

    # =========================================================================
    # Batch Operations (for memory-optimized experiments)
    # =========================================================================

    def encode_batch(
        self,
        prompts: List[str],
        configs: Optional[List[Dict[str, Any]]] = None,
        cache_to_cpu: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Batch encode all prompts before generation phase.

        Use this in setup() for memory-optimized experiments that need to:
        1. Load encoder, encode all prompts
        2. Offload encoder
        3. Load transformer, generate all videos

        Args:
            prompts: List of text prompts
            configs: Optional configs with 'layer_weights' or 'layer_mask' per prompt.
                     If None, encodes with default settings.
                     If provided, must match prompts length or be single config for all.
            cache_to_cpu: Move embeddings to CPU after encoding (frees GPU memory)

        Returns:
            Dict mapping prompt index (or (config_idx, prompt_idx) tuple) to embeddings

        Example:
            def setup(self):
                self.load_encoder()
                self.embeddings_cache = self.encode_batch(
                    prompts=self.prompts,
                    configs=self.layer_configs,
                )
                self.offload_encoder()
                self.load_model()
        """
        if self.encoder is None:
            raise RuntimeError("Encoder not loaded. Call load_encoder() first.")

        cache = {}

        # Handle configs
        if configs is None:
            # No configs - encode each prompt once
            for i, prompt in enumerate(tqdm(prompts, desc="Encoding prompts")):
                embeds = self.encode(prompt)
                if cache_to_cpu:
                    embeds = embeds.cpu()
                cache[i] = embeds
        elif len(configs) == 1:
            # Single config for all prompts
            config = configs[0]
            layer_weights = config.get("layer_weights")
            layer_mask = config.get("layer_mask")
            for i, prompt in enumerate(tqdm(prompts, desc="Encoding prompts")):
                embeds = self.encode(prompt, layer_weights=layer_weights, layer_mask=layer_mask)
                if cache_to_cpu:
                    embeds = embeds.cpu()
                cache[i] = embeds
        elif len(configs) == len(prompts):
            # One config per prompt
            for i, (prompt, config) in enumerate(tqdm(
                zip(prompts, configs), total=len(prompts), desc="Encoding prompts"
            )):
                layer_weights = config.get("layer_weights")
                layer_mask = config.get("layer_mask")
                embeds = self.encode(prompt, layer_weights=layer_weights, layer_mask=layer_mask)
                if cache_to_cpu:
                    embeds = embeds.cpu()
                cache[i] = embeds
        else:
            # Multiple configs × multiple prompts (full sweep)
            for ci, config in enumerate(configs):
                layer_weights = config.get("layer_weights")
                layer_mask = config.get("layer_mask")
                for pi, prompt in enumerate(tqdm(
                    prompts, desc=f"Encoding config {ci+1}/{len(configs)}"
                )):
                    embeds = self.encode(prompt, layer_weights=layer_weights, layer_mask=layer_mask)
                    if cache_to_cpu:
                        embeds = embeds.cpu()
                    cache[(ci, pi)] = embeds

        self._embeddings_cache = cache
        logger.info(f"Cached {len(cache)} embeddings")
        return cache

    def get_cached_embeds(
        self,
        key: Union[int, tuple],
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Retrieve cached embeddings from encode_batch().

        Args:
            key: Index or (config_idx, prompt_idx) tuple
            device: Move to this device (default: self.device)

        Returns:
            Embeddings tensor on specified device
        """
        if not hasattr(self, '_embeddings_cache') or self._embeddings_cache is None:
            raise RuntimeError("No embeddings cache. Call encode_batch() first.")

        embeds = self._embeddings_cache[key]
        target_device = device or self.device
        if embeds.device != torch.device(target_device):
            embeds = embeds.to(target_device)
        return embeds

    def offload_encoder(self) -> None:
        """Offload encoder to CPU to free GPU memory for generation.

        Note: 8-bit quantized models cannot be moved between devices,
        so we delete them entirely instead.
        """
        if self.encoder is not None:
            try:
                self.encoder.to("cpu")
            except ValueError as e:
                # 8-bit models can't be moved - delete instead
                if "8-bit" in str(e) or "bitsandbytes" in str(e):
                    logger.info("8-bit model detected, deleting encoder instead of moving to CPU")
                    del self.encoder
                    self.encoder = None
                else:
                    raise
            cleanup_memory()
            log_memory_usage("After offloading encoder")

    def offload_scorer(self) -> None:
        """Offload scorer to CPU."""
        if self.scorer is not None:
            self.scorer.offload()

    # =========================================================================
    # Router Integration (per-token layer routing)
    # =========================================================================

    def load_router(
        self,
        checkpoint_path: Optional[str] = None,
        routing_mode: str = "soft",
        temperature: float = 1.0,
        top_k: int = 8,
    ) -> "TokenLayerRouter":
        """
        Load TokenLayerRouter for per-token layer routing.

        Args:
            checkpoint_path: Path to trained router checkpoint (if None, uses uniform init)
            routing_mode: "soft" (differentiable), "topk" (sparse), or "gumbel"
            temperature: Softmax temperature (lower = sharper selection)
            top_k: Number of layers for topk mode

        Returns:
            TokenLayerRouter instance
        """
        from llm_dit.router import TokenLayerRouter

        router = TokenLayerRouter(
            hidden_dim=3840,  # Gemma-2 9B hidden dim
            num_layers=49,     # Gemma-2 9B layers
            bottleneck_dim=64,
            temperature=temperature,
            routing_mode=routing_mode,
            top_k=top_k,
            init_uniform=checkpoint_path is None,  # Uniform if no checkpoint
        )

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if "router_state_dict" in checkpoint:
                router.load_state_dict(checkpoint["router_state_dict"])
            else:
                router.load_state_dict(checkpoint)
            logger.info(f"Loaded router from {checkpoint_path}")

        router = router.to(self.device)
        router.eval()  # Set to inference mode
        return router

    def encode_with_router(
        self,
        prompt: str,
        router: "TokenLayerRouter",
        router_input_mode: str = "mean",
        return_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Encode prompt with per-token layer routing.

        Instead of uniform layer blending, uses the router to predict
        per-token layer weights for dynamic layer selection.

        Flow:
            1. Get multi-layer hidden states [B, T, D, L] from Gemma
            2. Extract router input (e.g., mean across layers)
            3. Router predicts per-token weights [B, T, L]
            4. Weighted sum: [B, T, D, L] x [B, T, L] -> [B, T, D]
            5. Normalize and project for DiT

        Args:
            prompt: Text prompt
            router: TokenLayerRouter instance
            router_input_mode: How to extract router input from layer stack.
                Options: "layer_0", "layer_24", "layer_47", "layer_48", "mean"
            return_stats: If True, also return routing statistics

        Returns:
            prompt_embeds: [B, seq_len, 4096] embeddings ready for DiT
            stats (optional): Dict with routing statistics if return_stats=True
        """
        if self.encoder is None:
            raise RuntimeError("Encoder not loaded. Call load_encoder() first.")

        from llm_dit.router import extract_router_input

        # Step 1: Get multi-layer hidden states from Gemma
        result = self.encoder.encode_multilayer(
            prompt,
            layer_indices=None,  # All 49 layers
            return_projected=False,
        )

        layer_stack = result['layer_stack']  # [B, T, D, L]
        attention_mask = result['attention_mask']  # [B, T]
        seq_length = result['seq_lengths'][0]

        # Step 2: Extract router input based on mode
        router_input = extract_router_input(layer_stack, mode=router_input_mode)

        # Step 3: Get per-token layer weights from router
        with torch.no_grad():
            layer_weights = router(router_input, attention_mask)  # [B, T, L]

        # Step 4: Apply per-token weighted sum across layers
        # layer_stack: [B, T, D, L]
        # layer_weights: [B, T, L]
        # result: [B, T, D] where D=3840
        weighted_hidden = torch.einsum('btdl,btl->btd', layer_stack, layer_weights)

        # Step 5: Normalize (match LTX-2 pipeline behavior)
        # Each position vector is L2-normalized
        eps = 1e-6
        prompt_embeds = weighted_hidden / (weighted_hidden.norm(dim=-1, keepdim=True) + eps)

        # Apply scale factor (8.0 to match LTX-2 pack_text_embeds)
        prompt_embeds = prompt_embeds * 8.0

        # Note: Output is [B, T, 3840] - same as standard encode() output
        # Pipeline connectors handle the final projection to 4096 for DiT

        if return_stats:
            stats = router.get_routing_stats(layer_weights)
            return prompt_embeds, stats
        return prompt_embeds

    # =========================================================================
    # Extension Points (experiments override)
    # =========================================================================

    def setup(self) -> None:
        """
        Override: Experiment-specific setup.

        Called once before running iterations. Load models, prepare data, etc.
        """
        pass

    @abstractmethod
    def run_iteration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override: Single experiment iteration.

        Args:
            config: Configuration for this iteration

        Returns:
            Results dictionary for this iteration
        """
        raise NotImplementedError

    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Override: Combine iteration results.

        Default: Return list of all results.

        Args:
            results: List of iteration results

        Returns:
            Aggregated results dictionary
        """
        return {"results": results}

    # =========================================================================
    # Runner (shared)
    # =========================================================================

    def run(
        self,
        configs: List[Dict[str, Any]],
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Run experiment with all configurations.

        Args:
            configs: List of configuration dicts for each iteration
            save_results: Whether to save results to JSON

        Returns:
            Aggregated results dictionary
        """
        self._init_run_dir()

        logger.info(f"Running {self.experiment_name} with {len(configs)} configurations")

        # Setup
        self.setup()

        # Run iterations
        results = []
        for config in tqdm(configs, desc=self.experiment_name):
            try:
                result = self.run_iteration(config)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in iteration: {e}")
                results.append({"error": str(e), "config": config})
            finally:
                self.cleanup()

        # Aggregate
        aggregated = self.aggregate_results(results)

        # Save results
        if save_results:
            results_path = self.run_dir / "results.json"
            with open(results_path, 'w') as f:
                json.dump(aggregated, f, indent=2, default=str)
            logger.info(f"Results saved to {results_path}")

        return aggregated

    def _init_run_dir(self) -> None:
        """Initialize run directory with timestamp."""
        if self.run_timestamp is None:
            self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.run_dir = self.output_dir / self.run_timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Example Experiments
# =============================================================================

class LayerBlendSweep(LTX2ExperimentBase):
    """
    Example: Sweep over different layer weight configurations.

    Tests how different layer blending strategies affect generation quality.

    Uses memory-optimized two-phase pattern:
    1. setup(): Load encoder, batch encode ALL (config × prompt) combinations
    2. run_iteration(): Load from cache, generate, score

    This keeps encoder and transformer from competing for GPU memory.
    """

    # Layer blend configurations (module-level for reuse)
    BLEND_CONFIGS = {
        "baseline": {"description": "All 49 layers, uniform", "layers": list(range(49))},
        "late_heavy": {"description": "Upweight layers 40-47", "layers": list(range(49)),
                      "weights": {i: (2.0 if 40 <= i <= 47 else 1.0) for i in range(49)}},
        "late_only": {"description": "Only layers 40-48", "layers": list(range(40, 49))},
        "top_contributors": {"description": "Only layers 43-47", "layers": list(range(43, 48))},
    }

    def __init__(self, output_dir: str = "outputs", quick: bool = False):
        super().__init__("layer_blend_sweep", output_dir)
        self.quick = quick

    def setup(self) -> None:
        """
        Two-phase setup: encode all, then load model.

        Phase 1: Encoder on GPU
        - Load encoder (8-bit)
        - Batch encode all prompts with all layer configs
        - Cache to CPU

        Phase 2: Model on GPU
        - Offload encoder
        - Load transformer pipeline
        """
        # Phase 1: Encoding
        self.load_encoder()
        self.prompts = get_all_prompts(quick=self.quick)

        # Build layer weight configs
        configs = []
        config_names = ["baseline", "late_heavy"] if self.quick else list(self.BLEND_CONFIGS.keys())

        for name in config_names:
            cfg = self.BLEND_CONFIGS[name]
            # Build layer weights tensor
            import numpy as np
            weights = np.zeros(49)
            if "weights" in cfg:
                for i, w in cfg["weights"].items():
                    weights[i] = w
            else:
                for i in cfg["layers"]:
                    weights[i] = 1.0
            weights = weights / weights.sum()  # Normalize

            configs.append({
                "name": name,
                "layer_weights": torch.tensor(weights, dtype=torch.float32),
            })

        self.config_names = config_names
        self.prompt_names = list(self.prompts.keys())

        # Batch encode: configs × prompts
        self.encode_batch(
            prompts=[self.prompts[n] for n in self.prompt_names],
            configs=configs,
        )

        # Phase 2: Generation
        self.offload_encoder()
        self.load_model(use_pure_pytorch=False)

    def run_iteration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate video from cached embeddings."""
        config_idx = config["config_idx"]
        prompt_idx = config["prompt_idx"]
        config_name = self.config_names[config_idx]
        prompt_name = self.prompt_names[prompt_idx]
        prompt = self.prompts[prompt_name]

        # Get cached embeddings
        embeds = self.get_cached_embeds((config_idx, prompt_idx))

        # Generate
        video = self.generate_video(embeds, seed=42)

        # Score
        score = self.score_video(video, prompt)

        # Save
        self.save_video(
            video,
            f"{config_name}_{prompt_name}",
            prompt,
            {"config": config_name, "score": score},
        )

        return {"config": config_name, "prompt": prompt_name, "score": score}

    def get_run_configs(self) -> List[Dict[str, Any]]:
        """Generate all (config, prompt) combinations for run()."""
        configs = []
        for ci in range(len(self.config_names)):
            for pi in range(len(self.prompt_names)):
                configs.append({"config_idx": ci, "prompt_idx": pi})
        return configs

    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Group results by config and compute averages."""
        from collections import defaultdict
        by_config = defaultdict(list)
        for r in results:
            if "error" not in r:
                by_config[r["config"]].append(r["score"])

        summary = {}
        for config, scores in by_config.items():
            summary[config] = {
                "mean_score": sum(scores) / len(scores) if scores else 0,
                "n": len(scores),
            }
        return {"by_config": summary, "all_results": results}


class QuickTest(LTX2ExperimentBase):
    """
    Minimal test to verify infrastructure works.

    Just generates one video with default settings.
    """

    def __init__(self):
        super().__init__("quick_test")

    def setup(self) -> None:
        # Minimal setup for quick test
        pass

    def run_iteration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Just verify the config passes through
        return {"status": "ok", **config}
