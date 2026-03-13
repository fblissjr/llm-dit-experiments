"""
Shared Experiment Infrastructure.

Last Updated: 2026-01-18

Provides pipeline-agnostic base classes and dataclasses for all experiments:
- ExperimentConfig: Configuration for a single experiment iteration
- ExperimentResult: Result from a single experiment iteration
- ExperimentRunnerBase: Abstract base class for experiment runners

All experiment runners should inherit from ExperimentRunnerBase to get:
- Consistent output paths: experiments/results/{pipeline}/{experiment}_{timestamp}/
- Standard directory structure (videos/, images/, metadata/, tensors/)
- Auto-discovery compatibility with compare/discovery.py
- Shared dataclasses for configs and results

Usage:
    from experiments.base import ExperimentRunnerBase, ExperimentConfig, ExperimentResult

    class MyExperiment(ExperimentRunnerBase):
        def __init__(self):
            super().__init__("my_experiment", pipeline="ltx2")

        def load_pipeline(self):
            # Load your models here
            pass

        def run_single(self, config: ExperimentConfig) -> ExperimentResult:
            # Run a single experiment iteration
            pass
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class ExperimentConfig:
    """
    Pipeline-agnostic experiment configuration.

    Used by all experiment runners regardless of pipeline (LTX-2, Z-Image, etc.).
    Pipeline-specific parameters go in the `extra` dict.
    """

    experiment_name: str
    prompt_id: str
    prompt_text: str
    seed: int
    variable_name: str
    variable_value: Any
    # Common video/image params
    width: int = 1024
    height: int = 1024
    num_frames: int = 33  # For video pipelines (ignored for image)
    num_inference_steps: int = 50
    guidance_scale: float = 3.0
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # Pipeline-specific params stored here
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling non-JSON-serializable values."""
        result = asdict(self)
        # Handle variable_value that might be a tensor or other non-serializable type
        if hasattr(result["variable_value"], "tolist"):
            result["variable_value"] = result["variable_value"].tolist()
        elif hasattr(result["variable_value"], "__dict__"):
            result["variable_value"] = str(result["variable_value"])
        return result


@dataclass
class ExperimentResult:
    """
    Pipeline-agnostic experiment result.

    Metrics (siglip_score, image_reward) are typically computed post-generation
    in a batch to allow memory-efficient pipeline offloading.
    """

    config: ExperimentConfig
    output_path: str
    generation_time_seconds: float
    # Metrics (computed post-generation)
    siglip_score: Optional[float] = None
    image_reward: Optional[float] = None
    # Error handling
    error: Optional[str] = None
    # Pipeline-specific results stored here
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "config": self.config.to_dict(),
            "output_path": self.output_path,
            "generation_time_seconds": self.generation_time_seconds,
            "siglip_score": self.siglip_score,
            "image_reward": self.image_reward,
            "error": self.error,
            "extra": self.extra,
        }
        return result


# =============================================================================
# Base Class
# =============================================================================


class ExperimentRunnerBase(ABC):
    """
    Base class for all experiment runners across pipelines.

    Provides:
    - Consistent output path pattern: experiments/results/{pipeline}/{experiment}_{timestamp}/
    - Standard directory structure (videos/, images/, metadata/, tensors/)
    - Metadata saving and summary generation
    - Discovery compatibility (works with compare/discovery.py)

    Subclasses must implement:
    - load_pipeline(): Load models/encoders needed for the experiment
    - run_single(config): Execute a single experiment configuration

    Example:
        class MyLTX2Experiment(ExperimentRunnerBase):
            def __init__(self):
                super().__init__("layer_ablation", pipeline="ltx2")

            def load_pipeline(self):
                self.encoder = load_encoder()
                self.model = load_model()

            def run_single(self, config):
                embeds = self.encoder.encode(config.prompt_text)
                video = self.model.generate(embeds)
                return ExperimentResult(config=config, output_path=str(video_path), ...)
    """

    # Known pipelines for validation and discovery
    KNOWN_PIPELINES = ["ltx2", "z_image", "qwen3_vl"]

    def __init__(
        self,
        experiment_name: str,
        pipeline: str,
        output_base: str = "experiments/results",
        dry_run: bool = False,
        create_dirs: bool = True,
    ):
        """
        Initialize experiment runner.

        Args:
            experiment_name: Name for this experiment (e.g., "layer_ablation")
            pipeline: Pipeline identifier (e.g., "ltx2", "z_image")
            output_base: Base output directory
            dry_run: If True, skip actual generation (for testing)
            create_dirs: If True, create output directories on init
        """
        self.experiment_name = experiment_name
        self.pipeline = pipeline
        self.dry_run = dry_run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Validate pipeline
        if pipeline not in self.KNOWN_PIPELINES:
            logger.warning(
                f"Unknown pipeline '{pipeline}'. Known pipelines: {self.KNOWN_PIPELINES}"
            )

        # Output: experiments/results/{pipeline}/{experiment}_{timestamp}/
        self.output_dir = (
            Path(output_base) / pipeline / f"{experiment_name}_{self.timestamp}"
        )

        if create_dirs and not dry_run:
            self._init_output_dirs()

    def _init_output_dirs(self) -> None:
        """Create standard output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create standard subdirectories based on pipeline type
        if self.pipeline == "ltx2":
            # Video pipelines
            (self.output_dir / "videos").mkdir(exist_ok=True)
            (self.output_dir / "frames").mkdir(exist_ok=True)  # For frame extraction
        else:
            # Image pipelines
            (self.output_dir / "images").mkdir(exist_ok=True)

        # Common directories
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        (self.output_dir / "tensors").mkdir(exist_ok=True)  # For .pt, .npz files

        logger.info(f"Output directory: {self.output_dir}")

    @property
    def videos_dir(self) -> Path:
        """Path to videos output directory."""
        return self.output_dir / "videos"

    @property
    def images_dir(self) -> Path:
        """Path to images output directory."""
        return self.output_dir / "images"

    @property
    def metadata_dir(self) -> Path:
        """Path to metadata directory."""
        return self.output_dir / "metadata"

    @property
    def tensors_dir(self) -> Path:
        """Path to tensors directory."""
        return self.output_dir / "tensors"

    @abstractmethod
    def load_pipeline(self) -> None:
        """
        Load models/encoders needed for the experiment.

        This is called before running experiments. Subclasses should load
        all necessary models here.
        """
        raise NotImplementedError

    @abstractmethod
    def run_single(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Run a single experiment configuration.

        Args:
            config: Configuration for this iteration

        Returns:
            ExperimentResult with output path, timing, and any errors
        """
        raise NotImplementedError

    def save_metadata(self, result: ExperimentResult, filename: Optional[str] = None) -> Path:
        """
        Save result metadata to JSON file.

        Args:
            result: Experiment result to save
            filename: Optional filename (default: derived from output_path)

        Returns:
            Path to saved metadata file
        """
        if filename is None:
            # Derive from output path
            output_path = Path(result.output_path)
            filename = output_path.stem + ".json"

        metadata_path = self.metadata_dir / filename

        with open(metadata_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        return metadata_path

    def save_summary(
        self, results: List[ExperimentResult], extra_metadata: Optional[Dict] = None
    ) -> Path:
        """
        Save experiment summary with aggregated results.

        Args:
            results: List of all experiment results
            extra_metadata: Additional metadata to include

        Returns:
            Path to saved summary file
        """
        successful = [r for r in results if r.error is None]
        failed = [r for r in results if r.error is not None]

        # Compute metric statistics
        siglip_scores = [r.siglip_score for r in successful if r.siglip_score is not None]
        image_reward_scores = [r.image_reward for r in successful if r.image_reward is not None]

        summary = {
            "experiment": self.experiment_name,
            "pipeline": self.pipeline,
            "timestamp": self.timestamp,
            "output_dir": str(self.output_dir),
            "total_runs": len(results),
            "successful_runs": len(successful),
            "failed_runs": len(failed),
            "total_time_seconds": sum(r.generation_time_seconds for r in results),
        }

        # Add metric stats if available
        if siglip_scores:
            summary["siglip_score"] = {
                "mean": sum(siglip_scores) / len(siglip_scores),
                "min": min(siglip_scores),
                "max": max(siglip_scores),
                "count": len(siglip_scores),
            }

        if image_reward_scores:
            summary["image_reward"] = {
                "mean": sum(image_reward_scores) / len(image_reward_scores),
                "min": min(image_reward_scores),
                "max": max(image_reward_scores),
                "count": len(image_reward_scores),
            }

        # Add extra metadata
        if extra_metadata:
            summary["extra"] = extra_metadata

        # Add all results
        summary["results"] = [r.to_dict() for r in results]

        summary_path = self.output_dir / "results.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Summary saved to {summary_path}")
        return summary_path

    def run_experiment(
        self,
        configs: List[ExperimentConfig],
        save_individual_metadata: bool = True,
        save_summary: bool = True,
    ) -> List[ExperimentResult]:
        """
        Run full experiment across all configurations.

        Args:
            configs: List of experiment configurations
            save_individual_metadata: Save metadata for each result
            save_summary: Save summary at the end

        Returns:
            List of all experiment results
        """
        logger.info(
            f"Running {self.experiment_name} ({self.pipeline}) with {len(configs)} configurations"
        )

        if not self.dry_run:
            self.load_pipeline()

        results = []
        from tqdm import tqdm

        for i, config in enumerate(tqdm(configs, desc=self.experiment_name)):
            logger.debug(f"Progress: {i + 1}/{len(configs)}")

            try:
                result = self.run_single(config)
            except Exception as e:
                logger.error(f"Error in iteration: {e}")
                result = ExperimentResult(
                    config=config,
                    output_path="",
                    generation_time_seconds=0.0,
                    error=str(e),
                )

            results.append(result)

            # Save individual metadata
            if save_individual_metadata and not self.dry_run and result.output_path:
                self.save_metadata(result)

        # Save summary
        if save_summary and not self.dry_run:
            self.save_summary(results)

        return results
