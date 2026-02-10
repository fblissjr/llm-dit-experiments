"""
Unit tests for RunManifest metadata dataclass.

Last Updated: 2026-02-10

Run with:
    uv run pytest tests/utils/test_run_manifest.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import orjson
import torch

from tests.utils.run_manifest import RunManifest


class TestRunManifestCreate:
    """Tests for the RunManifest.create() factory."""

    def test_basic_create(self):
        """Factory populates identity and environment fields."""
        m = RunManifest.create(
            test_name="test_smoke",
            prompt="A cat walking",
        )
        assert m.test_name == "test_smoke"
        assert m.prompt == "A cat walking"
        assert m.timestamp  # non-empty ISO string
        assert "python_version" in m.environment
        assert "torch_version" in m.environment

    def test_all_fields(self):
        """All explicit fields are stored correctly."""
        m = RunManifest.create(
            test_name="test_ref",
            prompt="A sunset over the ocean",
            num_frames=121,
            height=512,
            width=768,
            seed=123,
            negative_prompt="blurry, distorted",
            two_stage_config={"stage1_steps": 40, "stage2_steps": 3},
        )
        assert m.num_frames == 121
        assert m.height == 512
        assert m.width == 768
        assert m.seed == 123
        assert m.negative_prompt == "blurry, distorted"
        assert m.two_stage == {"stage1_steps": 40, "stage2_steps": 3}

    def test_defaults(self):
        """Unspecified fields have sensible defaults."""
        m = RunManifest.create(test_name="t", prompt="p")
        assert m.negative_prompt == ""
        assert m.seed == 0
        assert m.num_frames == 0
        assert m.two_stage is None
        assert m.video_path == ""
        assert m.frame_paths == []
        assert m.frame_indices == []
        assert m.video_shape == []
        assert m.pixel_mean == 0.0
        assert m.pixel_std == 0.0
        assert m.stage_timings == {}
        assert m.total_time == 0.0
        assert m.peak_vram_gb == 0.0

    def test_environment_has_cuda_info(self):
        """Environment includes CUDA info when available."""
        m = RunManifest.create(test_name="t", prompt="p")
        if torch.cuda.is_available():
            assert "cuda_version" in m.environment
            assert "gpu_name" in m.environment
            assert "gpu_vram_gb" in m.environment
        else:
            assert m.environment["cuda_available"] is False


class TestRunManifestVideoInfo:
    """Tests for set_video_info()."""

    def test_torch_tensor(self):
        """Computes stats from torch tensor."""
        video = torch.randint(0, 256, (33, 64, 96, 3), dtype=torch.uint8)
        m = RunManifest.create(test_name="t", prompt="p")
        m.set_video_info(video, Path("/some/dir/video.mp4"))

        assert m.video_shape == [33, 64, 96, 3]
        assert m.pixel_mean > 0  # random data, extremely unlikely to be 0
        assert m.pixel_std > 0
        assert m.video_path == "video.mp4"  # relative filename only

    def test_numpy_array(self):
        """Computes stats from numpy array."""
        video = np.random.randint(0, 256, (10, 32, 48, 3), dtype=np.uint8)
        m = RunManifest.create(test_name="t", prompt="p")
        m.set_video_info(video, Path("/tmp/out/result.mp4"))

        assert m.video_shape == [10, 32, 48, 3]
        assert m.video_path == "result.mp4"

    def test_blank_video_has_zero_std(self):
        """A solid-color video has zero pixel std."""
        video = torch.full((5, 8, 8, 3), 128, dtype=torch.uint8)
        m = RunManifest.create(test_name="t", prompt="p")
        m.set_video_info(video, Path("v.mp4"))
        assert m.pixel_std == 0.0
        assert m.pixel_mean == 128.0


class TestRunManifestFrameInfo:
    """Tests for set_frame_info()."""

    def test_relative_paths(self):
        """Paths are stored as filenames only."""
        m = RunManifest.create(test_name="t", prompt="p")
        m.set_frame_info(
            [Path("/a/b/frame_0000.png"), Path("/a/b/frame_0016.png")],
            [0, 16],
        )
        assert m.frame_paths == ["frame_0000.png", "frame_0016.png"]
        assert m.frame_indices == [0, 16]


class TestRunManifestSerialization:
    """Tests for save/load round-trip via orjson."""

    def _populated_manifest(self) -> RunManifest:
        """Create a fully-populated manifest for testing."""
        m = RunManifest.create(
            test_name="test_full",
            prompt="A cat walking through a garden",
            num_frames=33,
            height=512,
            width=768,
            seed=42,
            negative_prompt="worst quality",
            two_stage_config={"stage1_steps": 10, "stage2_steps": 3},
        )
        video = torch.randint(0, 256, (33, 16, 24, 3), dtype=torch.uint8)
        m.set_video_info(video, Path("/out/video.mp4"))
        m.set_frame_info(
            [Path("/out/frame_0000.png"), Path("/out/frame_0032.png")],
            [0, 32],
        )
        m.stage_timings = {"encoding": 5.2, "stage1_denoise": 30.1}
        m.total_time = 45.0
        m.peak_vram_gb = 14.5
        return m

    def test_round_trip(self, tmp_path):
        """Save then load preserves all fields."""
        original = self._populated_manifest()
        path = tmp_path / "manifest.json"
        original.save(path)
        loaded = RunManifest.load(path)

        assert loaded.test_name == original.test_name
        assert loaded.prompt == original.prompt
        assert loaded.negative_prompt == original.negative_prompt
        assert loaded.seed == original.seed
        assert loaded.num_frames == original.num_frames
        assert loaded.height == original.height
        assert loaded.width == original.width
        assert loaded.two_stage == original.two_stage
        assert loaded.video_path == original.video_path
        assert loaded.frame_paths == original.frame_paths
        assert loaded.frame_indices == original.frame_indices
        assert loaded.video_shape == original.video_shape
        assert loaded.stage_timings == original.stage_timings
        assert loaded.total_time == original.total_time
        assert loaded.peak_vram_gb == original.peak_vram_gb

    def test_json_is_readable(self, tmp_path):
        """Saved JSON is human-readable (indented)."""
        m = self._populated_manifest()
        path = tmp_path / "manifest.json"
        m.save(path)

        raw = path.read_bytes()
        data = orjson.loads(raw)
        assert isinstance(data, dict)
        assert data["test_name"] == "test_full"
        assert data["seed"] == 42

        # Should be indented (multi-line)
        text = raw.decode()
        assert "\n" in text

    def test_none_two_stage(self, tmp_path):
        """Single-stage manifest has null two_stage."""
        m = RunManifest.create(test_name="t", prompt="p")
        path = tmp_path / "manifest.json"
        m.save(path)
        loaded = RunManifest.load(path)
        assert loaded.two_stage is None

    def test_empty_lists_preserved(self, tmp_path):
        """Empty frame_paths/indices survive round-trip."""
        m = RunManifest.create(test_name="t", prompt="p")
        path = tmp_path / "manifest.json"
        m.save(path)
        loaded = RunManifest.load(path)
        assert loaded.frame_paths == []
        assert loaded.frame_indices == []
        assert loaded.video_shape == []
