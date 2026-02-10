"""
Unit tests for frame sampling utilities.

Last Updated: 2026-02-10

Run with:
    uv run pytest tests/utils/test_frame_sampler.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.utils.frame_sampler import compute_sample_indices, sample_frames, save_frames


class TestComputeSampleIndices:
    """Tests for evenly-spaced frame index computation."""

    def test_standard_33_frames(self):
        """33 frames with 8 samples should span the full range."""
        indices = compute_sample_indices(33, 8)
        assert len(indices) == 8
        assert indices[0] == 0
        assert indices[-1] == 32
        # Should be sorted
        assert indices == sorted(indices)

    def test_standard_121_frames(self):
        """121 frames with 8 samples should span the full range."""
        indices = compute_sample_indices(121, 8)
        assert len(indices) == 8
        assert indices[0] == 0
        assert indices[-1] == 120

    def test_zero_frames(self):
        """Zero frames returns empty list."""
        assert compute_sample_indices(0) == []
        assert compute_sample_indices(0, 4) == []

    def test_zero_samples(self):
        """Zero samples returns empty list."""
        assert compute_sample_indices(33, 0) == []

    def test_negative_frames(self):
        """Negative frames returns empty list."""
        assert compute_sample_indices(-5, 8) == []

    def test_single_frame(self):
        """Single frame always returns [0]."""
        assert compute_sample_indices(1, 1) == [0]
        assert compute_sample_indices(1, 8) == [0]

    def test_two_frames(self):
        """Two frames returns first and last."""
        assert compute_sample_indices(2, 2) == [0, 1]
        assert compute_sample_indices(2, 8) == [0, 1]

    def test_samples_exceed_frames(self):
        """When num_samples >= total_frames, return all frames."""
        assert compute_sample_indices(5, 8) == [0, 1, 2, 3, 4]
        assert compute_sample_indices(3, 3) == [0, 1, 2]
        assert compute_sample_indices(3, 100) == [0, 1, 2]

    def test_no_duplicates(self):
        """Result never contains duplicate indices."""
        for total in [1, 2, 3, 5, 10, 33, 121]:
            for n in [1, 2, 4, 8, 16]:
                indices = compute_sample_indices(total, n)
                assert len(indices) == len(set(indices)), (
                    f"Duplicates for total={total}, n={n}: {indices}"
                )

    def test_always_sorted(self):
        """Result is always sorted ascending."""
        for total in [1, 5, 33, 121]:
            for n in [1, 4, 8]:
                indices = compute_sample_indices(total, n)
                assert indices == sorted(indices)

    def test_all_in_range(self):
        """All indices are within [0, total_frames - 1]."""
        for total in [1, 5, 33, 121]:
            for n in [1, 4, 8, 16]:
                indices = compute_sample_indices(total, n)
                for i in indices:
                    assert 0 <= i < total, f"Index {i} out of range [0, {total})"

    @pytest.mark.parametrize("total,n,expected_first_last", [
        (33, 8, (0, 32)),
        (121, 8, (0, 120)),
        (9, 4, (0, 8)),
        (100, 2, (0, 99)),
    ])
    def test_first_and_last_guaranteed(self, total, n, expected_first_last):
        """First and last frames are always included."""
        indices = compute_sample_indices(total, n)
        assert indices[0] == expected_first_last[0]
        assert indices[-1] == expected_first_last[1]

    def test_even_spacing(self):
        """Indices should be approximately evenly spaced."""
        indices = compute_sample_indices(100, 5)
        gaps = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
        # All gaps should be within 1 of each other (approximately even)
        assert max(gaps) - min(gaps) <= 1


class TestSampleFrames:
    """Tests for frame extraction from video tensors."""

    def test_torch_tensor_input(self):
        """Accepts [F, H, W, C] torch tensor."""
        video = torch.randint(0, 256, (33, 64, 96, 3), dtype=torch.uint8)
        indices, frames = sample_frames(video, num_samples=4)
        assert len(indices) == 4
        assert len(frames) == 4
        assert all(isinstance(f, np.ndarray) for f in frames)
        assert all(f.shape == (64, 96, 3) for f in frames)

    def test_numpy_input(self):
        """Accepts [F, H, W, C] numpy array."""
        video = np.random.randint(0, 256, (33, 64, 96, 3), dtype=np.uint8)
        indices, frames = sample_frames(video, num_samples=4)
        assert len(indices) == 4
        assert len(frames) == 4
        assert all(f.dtype == np.uint8 for f in frames)

    def test_correct_frames_extracted(self):
        """Extracted frames match the video at the given indices."""
        video = np.arange(10 * 4 * 4 * 3, dtype=np.uint8).reshape(10, 4, 4, 3)
        indices, frames = sample_frames(video, num_samples=3)
        for idx, frame in zip(indices, frames):
            np.testing.assert_array_equal(frame, video[idx])

    def test_default_num_samples(self):
        """Default is 8 samples."""
        video = np.zeros((33, 4, 4, 3), dtype=np.uint8)
        indices, _ = sample_frames(video)
        assert len(indices) == 8


class TestSaveFrames:
    """Tests for PNG frame saving."""

    def test_saves_pngs(self, tmp_path):
        """Creates PNG files with correct naming."""
        frames = [np.zeros((32, 48, 3), dtype=np.uint8) for _ in range(3)]
        indices = [0, 16, 32]
        paths = save_frames(frames, indices, tmp_path)

        assert len(paths) == 3
        assert paths[0].name == "frame_0000.png"
        assert paths[1].name == "frame_0016.png"
        assert paths[2].name == "frame_0032.png"
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 0

    def test_custom_prefix(self, tmp_path):
        """Custom prefix changes filename pattern."""
        frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
        paths = save_frames(frames, [5], tmp_path, prefix="thumb")
        assert paths[0].name == "thumb_0005.png"

    def test_creates_output_dir(self, tmp_path):
        """Creates output directory if it doesn't exist."""
        nested = tmp_path / "a" / "b" / "c"
        frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
        paths = save_frames(frames, [0], nested)
        assert nested.exists()
        assert paths[0].exists()

    def test_saved_pngs_are_valid_images(self, tmp_path):
        """Saved PNGs can be re-read as images."""
        from PIL import Image

        frame = np.random.randint(0, 256, (32, 48, 3), dtype=np.uint8)
        paths = save_frames([frame], [7], tmp_path)

        img = Image.open(paths[0])
        assert img.size == (48, 32)  # PIL uses (width, height)
        reloaded = np.array(img)
        np.testing.assert_array_equal(reloaded, frame)
