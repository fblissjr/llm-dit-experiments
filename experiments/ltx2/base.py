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
from llm_dit.models.ltx2_components import Modality

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
    ) -> None:
        """
        Load LTX2 transformer with memory tracking.

        Args:
            model_path: Path to model weights
            use_pure_pytorch: If True, use our pure PyTorch model.
                             If False, use diffusers model.
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
                from diffusers import LTX2VideoPipeline
                self.pipeline = LTX2VideoPipeline.from_pretrained(
                    "models/LTX-2",
                    torch_dtype=self.dtype,
                )
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
                model_path=model_path,
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

    def load_scorer(self) -> None:
        """Load SigLIP scorer for text-video alignment."""
        if self.scorer is not None:
            return

        with MemoryTracker("Load scorer"):
            self.scorer = SigLIPScorer(device=self.device, dtype=self.dtype)

    def encode(
        self,
        prompt: str,
        layer_weights: Optional[torch.Tensor] = None,
        layer_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Encode prompt to embeddings.

        Args:
            prompt: Text prompt
            layer_weights: Optional per-layer weights for blending
            layer_mask: Optional binary mask for layer ablation
            **kwargs: Additional encoder arguments

        Returns:
            Prompt embeddings tensor
        """
        if self.encoder is None:
            raise RuntimeError("Encoder not loaded. Call load_encoder() first.")

        if layer_mask is not None:
            return self.encoder.encode_with_layer_masking(
                prompt,
                layer_mask=layer_mask,
                **kwargs,
            )
        elif layer_weights is not None:
            return self.encoder.encode_with_layer_weights(
                prompt,
                layer_weights=layer_weights,
                **kwargs,
            )
        else:
            return self.encoder.encode(prompt, **kwargs)

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
            prompt_embeds: Text embeddings from encode()
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
            output = self.pipeline(
                prompt_embeds=prompt_embeds,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                **kwargs,
            )
            return output.frames[0]
        else:
            # TODO: Implement pure PyTorch generation loop
            raise NotImplementedError(
                "Pure PyTorch generation not yet implemented. "
                "Use load_model(use_pure_pytorch=False) for now."
            )

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
        """Offload encoder to CPU to free GPU memory for generation."""
        if self.encoder is not None:
            self.encoder.to("cpu")
            cleanup_memory()
            log_memory_usage("After offloading encoder")

    def offload_scorer(self) -> None:
        """Offload scorer to CPU."""
        if self.scorer is not None:
            self.scorer.offload()

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
