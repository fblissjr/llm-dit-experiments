"""
Tests for shared experiment infrastructure.

Last Updated: 2026-01-18

Tests for experiments/base.py - the pipeline-agnostic base classes and dataclasses.
These tests use GENERIC values, not pipeline-specific (LTX-2, etc.) values.

Run with: uv run pytest tests/unit/test_experiment_base.py -v
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest  # noqa: F401 (used by pytest framework)

from experiments.base import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunnerBase,
)


# =============================================================================
# ExperimentConfig Tests
# =============================================================================


class TestExperimentConfig:
    """Tests for the ExperimentConfig dataclass."""

    def test_instantiation_with_required_fields(self):
        """Test ExperimentConfig can be created with only required fields."""
        config = ExperimentConfig(
            experiment_name="test_experiment",
            prompt_id="prompt_001",
            prompt_text="A test prompt",
            seed=42,
            variable_name="test_var",
            variable_value="test_value",
        )

        assert config.experiment_name == "test_experiment"
        assert config.prompt_id == "prompt_001"
        assert config.prompt_text == "A test prompt"
        assert config.seed == 42
        assert config.variable_name == "test_var"
        assert config.variable_value == "test_value"

    def test_default_values(self):
        """Test ExperimentConfig uses correct defaults for optional fields."""
        config = ExperimentConfig(
            experiment_name="test",
            prompt_id="p1",
            prompt_text="Test",
            seed=1,
            variable_name="v",
            variable_value=1,
        )

        # Generic defaults (not pipeline-specific)
        assert config.width == 1024
        assert config.height == 1024
        assert config.num_frames == 33
        assert config.num_inference_steps == 50
        assert config.guidance_scale == 3.0
        assert config.extra == {}

    def test_timestamp_auto_generation(self):
        """Test timestamp is auto-generated if not provided."""
        before = datetime.now().isoformat()
        config = ExperimentConfig(
            experiment_name="test",
            prompt_id="p1",
            prompt_text="Test",
            seed=1,
            variable_name="v",
            variable_value=1,
        )
        after = datetime.now().isoformat()

        # Timestamp should be between before and after
        assert config.timestamp >= before
        assert config.timestamp <= after

    def test_timestamp_explicit(self):
        """Test explicit timestamp overrides auto-generation."""
        explicit_ts = "2026-01-15T12:00:00"
        config = ExperimentConfig(
            experiment_name="test",
            prompt_id="p1",
            prompt_text="Test",
            seed=1,
            variable_name="v",
            variable_value=1,
            timestamp=explicit_ts,
        )

        assert config.timestamp == explicit_ts

    def test_extra_dict_for_extensions(self):
        """Test extra dict can hold pipeline-specific parameters."""
        config = ExperimentConfig(
            experiment_name="test",
            prompt_id="p1",
            prompt_text="Test",
            seed=1,
            variable_name="v",
            variable_value=1,
            extra={
                "ltx2_specific": "value",
                "layer_weights": [1.0, 2.0, 3.0],
                "nested": {"key": "value"},
            },
        )

        assert config.extra["ltx2_specific"] == "value"
        assert config.extra["layer_weights"] == [1.0, 2.0, 3.0]
        assert config.extra["nested"]["key"] == "value"

    def test_to_dict_basic(self):
        """Test to_dict() returns correct dictionary."""
        config = ExperimentConfig(
            experiment_name="test",
            prompt_id="p1",
            prompt_text="Test prompt",
            seed=42,
            variable_name="var",
            variable_value="val",
            width=512,
            height=768,
        )

        d = config.to_dict()

        assert d["experiment_name"] == "test"
        assert d["prompt_id"] == "p1"
        assert d["prompt_text"] == "Test prompt"
        assert d["seed"] == 42
        assert d["variable_name"] == "var"
        assert d["variable_value"] == "val"
        assert d["width"] == 512
        assert d["height"] == 768

    def test_to_dict_handles_tensor_value(self):
        """Test to_dict() converts tensor variable_value to list."""
        import torch

        config = ExperimentConfig(
            experiment_name="test",
            prompt_id="p1",
            prompt_text="Test",
            seed=1,
            variable_name="weights",
            variable_value=torch.tensor([1.0, 2.0, 3.0]),
        )

        d = config.to_dict()
        assert d["variable_value"] == [1.0, 2.0, 3.0]
        assert isinstance(d["variable_value"], list)

    def test_to_dict_handles_object_value(self):
        """Test to_dict() converts object variable_value to string."""

        class CustomObject:
            def __init__(self):
                self.x = 1

            def __str__(self):
                return "CustomObject(x=1)"

        config = ExperimentConfig(
            experiment_name="test",
            prompt_id="p1",
            prompt_text="Test",
            seed=1,
            variable_name="obj",
            variable_value=CustomObject(),
        )

        d = config.to_dict()
        assert d["variable_value"] == "CustomObject(x=1)"
        assert isinstance(d["variable_value"], str)

    def test_to_dict_is_json_serializable(self):
        """Test to_dict() output can be serialized to JSON."""
        config = ExperimentConfig(
            experiment_name="test",
            prompt_id="p1",
            prompt_text="Test",
            seed=1,
            variable_name="v",
            variable_value={"nested": [1, 2, 3]},
            extra={"more": "data"},
        )

        d = config.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["experiment_name"] == "test"
        assert parsed["variable_value"] == {"nested": [1, 2, 3]}


# =============================================================================
# ExperimentResult Tests
# =============================================================================


class TestExperimentResult:
    """Tests for the ExperimentResult dataclass."""

    def test_instantiation_from_config(self):
        """Test ExperimentResult can be created from a config."""
        config = ExperimentConfig(
            experiment_name="test",
            prompt_id="p1",
            prompt_text="Test",
            seed=42,
            variable_name="v",
            variable_value=1,
        )

        result = ExperimentResult(
            config=config,
            output_path="/path/to/output.mp4",
            generation_time_seconds=12.5,
        )

        assert result.config is config
        assert result.output_path == "/path/to/output.mp4"
        assert result.generation_time_seconds == 12.5

    def test_optional_metric_fields(self):
        """Test optional metric fields default to None."""
        config = ExperimentConfig(
            experiment_name="test",
            prompt_id="p1",
            prompt_text="Test",
            seed=1,
            variable_name="v",
            variable_value=1,
        )

        result = ExperimentResult(
            config=config,
            output_path="/out.mp4",
            generation_time_seconds=5.0,
        )

        assert result.siglip_score is None
        assert result.image_reward is None
        assert result.error is None
        assert result.extra == {}

    def test_metric_fields_populated(self):
        """Test metric fields can be populated."""
        config = ExperimentConfig(
            experiment_name="test",
            prompt_id="p1",
            prompt_text="Test",
            seed=1,
            variable_name="v",
            variable_value=1,
        )

        result = ExperimentResult(
            config=config,
            output_path="/out.mp4",
            generation_time_seconds=5.0,
            siglip_score=0.85,
            image_reward=1.23,
        )

        assert result.siglip_score == 0.85
        assert result.image_reward == 1.23

    def test_error_handling_field(self):
        """Test error field captures failure information."""
        config = ExperimentConfig(
            experiment_name="test",
            prompt_id="p1",
            prompt_text="Test",
            seed=1,
            variable_name="v",
            variable_value=1,
        )

        result = ExperimentResult(
            config=config,
            output_path="",
            generation_time_seconds=0.5,
            error="CUDA out of memory",
        )

        assert result.error == "CUDA out of memory"
        assert result.output_path == ""

    def test_extra_dict_for_pipeline_results(self):
        """Test extra dict stores pipeline-specific results."""
        config = ExperimentConfig(
            experiment_name="test",
            prompt_id="p1",
            prompt_text="Test",
            seed=1,
            variable_name="v",
            variable_value=1,
        )

        result = ExperimentResult(
            config=config,
            output_path="/out.mp4",
            generation_time_seconds=10.0,
            extra={
                "peak_memory_gb": 18.5,
                "num_tokens_generated": 1920,
                "routing_entropy": 0.73,
            },
        )

        assert result.extra["peak_memory_gb"] == 18.5
        assert result.extra["num_tokens_generated"] == 1920

    def test_to_dict_basic(self):
        """Test to_dict() returns correct dictionary structure."""
        config = ExperimentConfig(
            experiment_name="test",
            prompt_id="p1",
            prompt_text="Test",
            seed=42,
            variable_name="v",
            variable_value="val",
        )

        result = ExperimentResult(
            config=config,
            output_path="/path/output.mp4",
            generation_time_seconds=7.5,
            siglip_score=0.9,
        )

        d = result.to_dict()

        assert d["output_path"] == "/path/output.mp4"
        assert d["generation_time_seconds"] == 7.5
        assert d["siglip_score"] == 0.9
        assert d["image_reward"] is None
        assert d["error"] is None
        assert "config" in d
        assert d["config"]["experiment_name"] == "test"

    def test_to_dict_is_json_serializable(self):
        """Test to_dict() output can be serialized to JSON."""
        config = ExperimentConfig(
            experiment_name="test",
            prompt_id="p1",
            prompt_text="Test",
            seed=1,
            variable_name="v",
            variable_value=1,
        )

        result = ExperimentResult(
            config=config,
            output_path="/out.mp4",
            generation_time_seconds=5.0,
            siglip_score=0.8,
            extra={"nested": [1, 2, 3]},
        )

        d = result.to_dict()
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

        parsed = json.loads(json_str)
        assert parsed["siglip_score"] == 0.8


# =============================================================================
# ExperimentRunnerBase Tests
# =============================================================================


class ConcreteRunner(ExperimentRunnerBase):
    """Concrete implementation for testing the abstract base class."""

    def load_pipeline(self) -> None:
        self.pipeline_loaded = True

    def run_single(self, config: ExperimentConfig) -> ExperimentResult:
        return ExperimentResult(
            config=config,
            output_path=str(self.output_dir / "test_output.mp4"),
            generation_time_seconds=1.0,
        )


class TestExperimentRunnerBase:
    """Tests for the ExperimentRunnerBase abstract class."""

    def test_output_path_pattern(self):
        """Test output path follows: {output_base}/{pipeline}/{experiment}_{timestamp}/"""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ConcreteRunner(
                experiment_name="test_exp",
                pipeline="ltx2",
                output_base=tmpdir,
                dry_run=True,  # Don't create dirs
            )

            # Path should match pattern
            expected_pattern = f"{tmpdir}/ltx2/test_exp_"
            assert str(runner.output_dir).startswith(expected_pattern)

            # Should have timestamp suffix (YYYYMMDD_HHMMSS format)
            suffix = str(runner.output_dir).replace(expected_pattern, "")
            assert len(suffix) == 15  # YYYYMMDD_HHMMSS
            assert suffix[8] == "_"  # Date-time separator

    def test_init_output_dirs_creates_structure(self):
        """Test _init_output_dirs() creates standard directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ConcreteRunner(
                experiment_name="test_exp",
                pipeline="ltx2",
                output_base=tmpdir,
                dry_run=False,
            )

            # Video pipeline dirs
            assert runner.output_dir.exists()
            assert (runner.output_dir / "videos").exists()
            assert (runner.output_dir / "frames").exists()
            assert (runner.output_dir / "metadata").exists()
            assert (runner.output_dir / "tensors").exists()

    def test_init_output_dirs_image_pipeline(self):
        """Test _init_output_dirs() creates images/ for non-video pipelines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ConcreteRunner(
                experiment_name="test_exp",
                pipeline="z_image",  # Image pipeline
                output_base=tmpdir,
                dry_run=False,
            )

            # Image pipeline dirs
            assert (runner.output_dir / "images").exists()
            assert not (runner.output_dir / "videos").exists()

    def test_dry_run_skips_directory_creation(self):
        """Test dry_run=True skips directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ConcreteRunner(
                experiment_name="test_exp",
                pipeline="ltx2",
                output_base=tmpdir,
                dry_run=True,
            )

            assert not runner.output_dir.exists()

    def test_known_pipelines_contains_expected(self):
        """Test KNOWN_PIPELINES contains the expected pipeline identifiers."""
        expected = ["ltx2", "z_image", "wan", "qwen3_vl"]
        assert ExperimentRunnerBase.KNOWN_PIPELINES == expected

    def test_unknown_pipeline_logs_warning(self, caplog):
        """Test unknown pipeline logs a warning but doesn't fail."""
        import logging

        with tempfile.TemporaryDirectory() as tmpdir:
            with caplog.at_level(logging.WARNING):
                runner = ConcreteRunner(
                    experiment_name="test",
                    pipeline="unknown_pipeline",
                    output_base=tmpdir,
                    dry_run=True,
                )

            # Should warn about unknown pipeline
            assert "unknown_pipeline" in caplog.text.lower() or runner.pipeline == "unknown_pipeline"

    def test_directory_properties(self):
        """Test convenience properties return correct paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ConcreteRunner(
                experiment_name="test",
                pipeline="ltx2",
                output_base=tmpdir,
                dry_run=True,
            )

            assert runner.videos_dir == runner.output_dir / "videos"
            assert runner.images_dir == runner.output_dir / "images"
            assert runner.metadata_dir == runner.output_dir / "metadata"
            assert runner.tensors_dir == runner.output_dir / "tensors"

    def test_save_metadata(self):
        """Test save_metadata() writes JSON correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ConcreteRunner(
                experiment_name="test",
                pipeline="ltx2",
                output_base=tmpdir,
                dry_run=False,
            )

            config = ExperimentConfig(
                experiment_name="test",
                prompt_id="p1",
                prompt_text="Test prompt",
                seed=42,
                variable_name="v",
                variable_value="val",
            )

            result = ExperimentResult(
                config=config,
                output_path=str(runner.output_dir / "test.mp4"),
                generation_time_seconds=5.0,
                siglip_score=0.85,
            )

            meta_path = runner.save_metadata(result)

            assert meta_path.exists()
            assert meta_path.suffix == ".json"

            with open(meta_path) as f:
                loaded = json.load(f)
            assert loaded["siglip_score"] == 0.85
            assert loaded["config"]["prompt_text"] == "Test prompt"

    def test_save_summary(self):
        """Test save_summary() writes aggregated results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ConcreteRunner(
                experiment_name="test",
                pipeline="ltx2",
                output_base=tmpdir,
                dry_run=False,
            )

            results = []
            for i in range(3):
                config = ExperimentConfig(
                    experiment_name="test",
                    prompt_id=f"p{i}",
                    prompt_text=f"Prompt {i}",
                    seed=i,
                    variable_name="v",
                    variable_value=i,
                )
                results.append(
                    ExperimentResult(
                        config=config,
                        output_path=f"/out{i}.mp4",
                        generation_time_seconds=float(i + 1),
                        siglip_score=0.8 + i * 0.05,
                    )
                )

            summary_path = runner.save_summary(results)

            assert summary_path.exists()
            assert summary_path.name == "results.json"

            with open(summary_path) as f:
                summary = json.load(f)

            assert summary["experiment"] == "test"
            assert summary["pipeline"] == "ltx2"
            assert summary["total_runs"] == 3
            assert summary["successful_runs"] == 3
            assert "siglip_score" in summary
            assert summary["siglip_score"]["count"] == 3

    def test_run_experiment_calls_methods_correctly(self):
        """Test run_experiment() orchestrates the experiment correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ConcreteRunner(
                experiment_name="test",
                pipeline="ltx2",
                output_base=tmpdir,
                dry_run=False,
            )

            configs = [
                ExperimentConfig(
                    experiment_name="test",
                    prompt_id=f"p{i}",
                    prompt_text=f"Prompt {i}",
                    seed=i,
                    variable_name="v",
                    variable_value=i,
                )
                for i in range(2)
            ]

            results = runner.run_experiment(configs)

            # Pipeline should have been loaded
            assert runner.pipeline_loaded

            # Should have results for each config
            assert len(results) == 2

            # Summary should exist
            assert (runner.output_dir / "results.json").exists()


# =============================================================================
# Discovery Integration Tests
# =============================================================================


class TestDiscoveryIntegration:
    """Tests for experiment discovery compatibility."""

    def _create_valid_experiment(
        self,
        exp_dir: Path,
        media_type: str = "videos",
        ext: str = ".mp4",
    ) -> None:
        """Helper to create a valid experiment directory structure.

        Discovery requires:
        - metadata/*.json files with valid config
        - Corresponding media files that exist
        """
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / media_type).mkdir(exist_ok=True)
        (exp_dir / "metadata").mkdir(exist_ok=True)
        (exp_dir / "results.json").write_text('{"test": true}')

        # Create a dummy media file
        media_file = exp_dir / media_type / f"test_output{ext}"
        media_file.write_bytes(b"dummy media content")

        # Create metadata that references the media file
        metadata = {
            "config": {
                "prompt_id": "test_prompt",
                "variable_name": "test_var",
                "variable_value": "test_val",
                "seed": 42,
            },
            "output_path": str(media_file),
            "siglip_score": 0.85,
        }
        (exp_dir / "metadata" / "test_output.json").write_text(json.dumps(metadata))

    def test_discovers_pipeline_subdirectories(self):
        """Test discover_experiments() finds experiments in pipeline subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)

            # Create pipeline directory structure
            ltx2_dir = results_dir / "ltx2"
            ltx2_dir.mkdir()

            # Create valid experiment directories
            exp1_dir = ltx2_dir / "layer_ablation_20260115_191000"
            exp2_dir = ltx2_dir / "activation_steering_20260115_214600"

            self._create_valid_experiment(exp1_dir, media_type="videos", ext=".mp4")
            self._create_valid_experiment(exp2_dir, media_type="videos", ext=".mp4")

            # Import and run discovery
            from experiments.compare.discovery import discover_experiments

            experiments = discover_experiments(results_dir)

            # Should find both experiments
            assert len(experiments) >= 2
            exp_names = [e.experiment_type for e in experiments]
            assert "layer_ablation" in exp_names
            assert "activation_steering" in exp_names

    def test_backward_compatible_flat_structure(self):
        """Test discover_experiments() handles legacy flat structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)

            # Create legacy flat experiment (not in pipeline subdir)
            legacy_exp = results_dir / "old_experiment_20251210_143022"
            self._create_valid_experiment(legacy_exp, media_type="images", ext=".png")

            from experiments.compare.discovery import discover_experiments

            experiments = discover_experiments(results_dir)

            # Should find legacy experiment
            exp_types = [e.experiment_type for e in experiments]
            assert "old_experiment" in exp_types

    def test_runner_output_discoverable(self):
        """Test that ExperimentRunnerBase output is discoverable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a runner
            runner = ConcreteRunner(
                experiment_name="discoverable_test",
                pipeline="ltx2",
                output_base=tmpdir,
                dry_run=False,
            )

            # Create valid experiment output with proper structure
            # Discovery requires metadata + actual media files
            (runner.output_dir / "results.json").write_text('{"test": true}')

            # Create a dummy video
            video_file = runner.videos_dir / "test_video.mp4"
            video_file.write_bytes(b"dummy video content")

            # Create metadata pointing to the video
            metadata = {
                "config": {
                    "prompt_id": "test",
                    "variable_name": "var",
                    "variable_value": "val",
                    "seed": 1,
                },
                "output_path": str(video_file),
            }
            (runner.metadata_dir / "test_video.json").write_text(json.dumps(metadata))

            # Try to discover it
            from experiments.compare.discovery import discover_experiments

            experiments = discover_experiments(Path(tmpdir))

            # Should find our experiment
            exp_types = [e.experiment_type for e in experiments]
            assert "discoverable_test" in exp_types


# Run with: uv run pytest tests/unit/test_experiment_base.py -v
