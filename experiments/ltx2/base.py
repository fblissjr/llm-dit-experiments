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
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from tqdm import tqdm

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
    """

    def __init__(self, output_dir: str = "outputs"):
        super().__init__("layer_blend_sweep", output_dir)

    def setup(self) -> None:
        self.load_model(use_pure_pytorch=False)  # Use diffusers for now
        self.load_encoder()
        self.prompts = get_all_prompts(quick=True)

    def run_iteration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        layer_weights = config["layer_weights"]
        prompt_name = config["prompt_name"]
        prompt = self.prompts[prompt_name]

        # Encode with layer weights
        embeds = self.encode(prompt, layer_weights=layer_weights)

        # Offload encoder before generation
        self.offload_encoder()

        # Generate video
        video = self.generate_video(embeds, seed=42)

        # Score
        score = self.score_video(video, prompt)

        # Save
        self.save_video(
            video,
            f"{config['name']}_{prompt_name}",
            prompt,
            {"layer_weights": layer_weights.tolist(), "score": score},
        )

        return {
            "name": config["name"],
            "prompt_name": prompt_name,
            "score": score,
        }


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
