"""Automated statistical validation for E2E test outputs.

last updated: 2026-02-12

Provides image and video validation with structured results.
Each check returns a CheckResult with pass/fail, measured value,
and threshold used -- enabling both automated gating and human review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class CheckResult:
    """Result of a single validation check."""

    passed: bool
    name: str
    value: float | str | None = None
    threshold: str = ""
    detail: str = ""


@dataclass
class ValidationResult:
    """Aggregate result of all validation checks for a test output."""

    passed: bool = True
    checks: dict[str, CheckResult] = field(default_factory=dict)

    def add(self, check: CheckResult) -> None:
        self.checks[check.name] = check
        if not check.passed:
            self.passed = False

    def summary(self) -> str:
        """Human-readable summary of all checks."""
        lines = []
        for name, check in self.checks.items():
            status = "PASS" if check.passed else "FAIL"
            detail = f" ({check.detail})" if check.detail else ""
            lines.append(f"  [{status}] {name}: {check.value}{detail}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize for manifest.json."""
        return {
            "passed": self.passed,
            "checks": {
                name: {
                    "passed": c.passed,
                    "value": c.value,
                    "threshold": c.threshold,
                }
                for name, c in self.checks.items()
            },
        }


# ---------------------------------------------------------------------------
# Image validation (FLUX.2, Z-Image, Qwen-Image)
# ---------------------------------------------------------------------------


def validate_image(
    image_path: Path,
    expected_w: int | None = None,
    expected_h: int | None = None,
    std_range: tuple[float, float] = (5.0, 80.0),
) -> ValidationResult:
    """Run all image validation checks.

    Args:
        image_path: Path to output PNG
        expected_w: Expected width (None to skip dimension check)
        expected_h: Expected height (None to skip dimension check)
        std_range: (min, max) acceptable pixel standard deviation
    """
    result = ValidationResult()

    # valid_format: PIL can decode it
    try:
        img = Image.open(image_path)
        img.load()
        result.add(CheckResult(
            passed=True, name="valid_format",
            value=img.format or "unknown", threshold="PIL decodable",
        ))
    except Exception as e:
        result.add(CheckResult(
            passed=False, name="valid_format",
            value=str(e), threshold="PIL decodable",
        ))
        return result  # Can't run further checks

    arr = np.array(img, dtype=np.float32)

    # correct_dimensions
    if expected_w is not None and expected_h is not None:
        actual_w, actual_h = img.size
        dims_ok = actual_w == expected_w and actual_h == expected_h
        result.add(CheckResult(
            passed=dims_ok, name="correct_dimensions",
            value=f"{actual_w}x{actual_h}",
            threshold=f"expected {expected_w}x{expected_h}",
        ))

    # not_noise: pixel std in acceptable range
    pixel_std = float(np.std(arr))
    std_ok = std_range[0] <= pixel_std <= std_range[1]
    result.add(CheckResult(
        passed=std_ok, name="not_noise",
        value=round(pixel_std, 2),
        threshold=f"std in [{std_range[0]}, {std_range[1]}]",
    ))

    # not_blank: mean not near 0 or 255
    pixel_mean = float(np.mean(arr))
    blank_ok = 5.0 < pixel_mean < 250.0
    result.add(CheckResult(
        passed=blank_ok, name="not_blank",
        value=round(pixel_mean, 2),
        threshold="mean in (5, 250)",
    ))

    return result


# ---------------------------------------------------------------------------
# Video validation (LTX-2)
# ---------------------------------------------------------------------------


def validate_video(
    video_path: Path,
    expected_frames: int | None = None,
    expected_w: int | None = None,
    expected_h: int | None = None,
    frozen_threshold: float = 0.1,
) -> ValidationResult:
    """Run all video validation checks.

    Requires the video file to be a playable mp4. Uses frame sampling
    for efficiency (checks first, middle, last frames + adjacent pairs).
    """
    result = ValidationResult()

    # Check file exists and is non-empty
    if not video_path.exists():
        result.add(CheckResult(
            passed=False, name="file_exists",
            value=str(video_path), threshold="exists on disk",
        ))
        return result

    file_size = video_path.stat().st_size
    if file_size < 1024:
        result.add(CheckResult(
            passed=False, name="valid_size",
            value=file_size, threshold=">1024 bytes",
        ))
        return result

    result.add(CheckResult(
        passed=True, name="valid_size",
        value=file_size, threshold=">1024 bytes",
    ))

    # Try to read frames using PIL/imageio
    try:
        import imageio.v3 as iio

        frames = iio.imread(video_path, plugin="pyav")
    except ImportError:
        # imageio not available -- skip frame-level checks
        result.add(CheckResult(
            passed=True, name="frame_analysis",
            value="skipped", threshold="imageio not available",
            detail="Install imageio[pyav] for frame-level video validation",
        ))
        return result
    except Exception as e:
        result.add(CheckResult(
            passed=False, name="readable",
            value=str(e), threshold="decodable video",
        ))
        return result

    num_frames = len(frames)

    # correct_shape
    if expected_frames is not None:
        frames_ok = num_frames == expected_frames
        result.add(CheckResult(
            passed=frames_ok, name="correct_frame_count",
            value=num_frames, threshold=f"expected {expected_frames}",
        ))

    if expected_w is not None and expected_h is not None and num_frames > 0:
        h, w = frames[0].shape[:2]
        dims_ok = w == expected_w and h == expected_h
        result.add(CheckResult(
            passed=dims_ok, name="correct_dimensions",
            value=f"{w}x{h}", threshold=f"expected {expected_w}x{expected_h}",
        ))

    if num_frames < 2:
        return result

    # not_noise: per-frame pixel variance check
    sample_indices = [0, num_frames // 2, num_frames - 1]
    for idx in sample_indices:
        frame = frames[idx].astype(np.float32)
        std_val = float(np.std(frame))
        result.add(CheckResult(
            passed=5.0 <= std_val <= 100.0, name=f"not_noise_frame_{idx}",
            value=round(std_val, 2), threshold="std in [5, 100]",
        ))

    # not_frozen: adjacent frames must differ (MSE above threshold).
    # Default 0.1 catches truly frozen (identical) frames while allowing
    # naturally low inter-frame variation in short clips (9 frames = 0.37s).
    for i in range(min(3, num_frames - 1)):
        f1 = frames[i].astype(np.float32)
        f2 = frames[i + 1].astype(np.float32)
        mse = float(np.mean((f1 - f2) ** 2))
        not_frozen = mse > frozen_threshold
        result.add(CheckResult(
            passed=not_frozen, name=f"not_frozen_{i}_{i+1}",
            value=round(mse, 2), threshold=f"MSE > {frozen_threshold}",
        ))

    return result
