"""Unit tests for scripts/batch_flux2.py -- batch FLUX.2 generation.

Last Updated: 2026-03-30

Tests cover:
- Argument parsing (with config.toml defaults)
- Image collection from directory
- Base64 encoding of image files
- Request body building
- Resume/skip logic
- Output path generation
- Config loading

Run with: uv run pytest tests/unit/test_batch_flux2.py -v
"""

import base64
from pathlib import Path

import pytest

from scripts.batch_flux2 import (
    build_body,
    collect_images,
    create_parser,
    encode_image_b64,
    load_batch_config,
    output_path_for,
    should_skip,
)


# ===========================================================================
# Config Loading
# ===========================================================================


class TestLoadBatchConfig:
    """Verify config.toml reading for server URL and model defaults."""

    def test_reads_server_and_model(self, tmp_path):
        toml = tmp_path / "config.toml"
        toml.write_text(
            '[server]\nhost = "192.168.1.10"\nport = 9000\n\n'
            '[flux2]\ndefault_model = "klein-9b-kv-fp8"\n'
        )
        cfg = load_batch_config(toml)
        assert cfg["server_url"] == "http://192.168.1.10:9000"
        assert cfg["model_name"] == "klein-9b-kv-fp8"

    def test_defaults_when_no_file(self, tmp_path):
        cfg = load_batch_config(tmp_path / "nonexistent.toml")
        assert cfg["server_url"] == "http://127.0.0.1:7860"
        assert cfg["model_name"] == "klein-9b-fp8"

    def test_defaults_when_sections_missing(self, tmp_path):
        toml = tmp_path / "config.toml"
        toml.write_text("# empty config\n")
        cfg = load_batch_config(toml)
        assert cfg["server_url"] == "http://127.0.0.1:7860"
        assert cfg["model_name"] == "klein-9b-fp8"

    def test_partial_server_section(self, tmp_path):
        toml = tmp_path / "config.toml"
        toml.write_text('[server]\nport = 8080\n')
        cfg = load_batch_config(toml)
        assert cfg["server_url"] == "http://127.0.0.1:8080"


# ===========================================================================
# Argument Parsing
# ===========================================================================


class TestArgParsing:
    """Verify argument collection and defaults."""

    def test_minimal_args(self):
        parser = create_parser()
        args = parser.parse_args([
            "--input-dir", "/tmp/images",
            "--prompt", "a cat",
        ])
        assert args.input_dir == "/tmp/images"
        assert args.prompt == "a cat"
        assert args.output_dir == "outputs/batch/"
        # model_name defaults to None (filled from config.toml at runtime)
        assert args.model_name is None
        # server defaults to None (filled from config.toml at runtime)
        assert args.server is None

    def test_all_flags(self):
        parser = create_parser()
        args = parser.parse_args([
            "--input-dir", "/data/in",
            "--output-dir", "/data/out",
            "--prompt", "transform this",
            "--model-name", "klein-9b-kv",
            "--width", "1024",
            "--height", "768",
            "--seed", "42",
            "--match-image-size", "0 (First Image)",
            "--server", "http://localhost:9000",
            "--timeout", "600",
            "--no-resume",
        ])
        assert args.input_dir == "/data/in"
        assert args.output_dir == "/data/out"
        assert args.prompt == "transform this"
        assert args.model_name == "klein-9b-kv"
        assert args.width == 1024
        assert args.height == 768
        assert args.seed == 42
        assert args.match_image_size == "0 (First Image)"
        assert args.server == "http://localhost:9000"
        assert args.timeout == 600
        assert args.no_resume is True

    def test_no_resume_default_false(self):
        parser = create_parser()
        args = parser.parse_args([
            "--input-dir", "/tmp/x",
            "--prompt", "test",
        ])
        assert args.no_resume is False

    def test_config_flag(self):
        parser = create_parser()
        args = parser.parse_args([
            "--config", "/path/to/config.toml",
            "--input-dir", "/tmp/x",
            "--prompt", "test",
        ])
        assert args.config == "/path/to/config.toml"

    def test_config_default(self):
        parser = create_parser()
        args = parser.parse_args([
            "--input-dir", "/tmp/x",
            "--prompt", "test",
        ])
        assert args.config == "config.toml"


# ===========================================================================
# Image Collection
# ===========================================================================


class TestCollectImages:
    """Verify image file discovery and sorting."""

    def test_finds_common_formats(self, tmp_path):
        for name in ["a.png", "b.jpg", "c.jpeg", "d.webp"]:
            (tmp_path / name).write_bytes(b"\x00")
        result = collect_images(tmp_path)
        assert len(result) == 4

    def test_sorted_alphabetically(self, tmp_path):
        for name in ["c.png", "a.png", "b.png"]:
            (tmp_path / name).write_bytes(b"\x00")
        result = collect_images(tmp_path)
        assert [p.name for p in result] == ["a.png", "b.png", "c.png"]

    def test_ignores_non_image_files(self, tmp_path):
        (tmp_path / "readme.txt").write_text("hello")
        (tmp_path / "data.csv").write_text("1,2,3")
        (tmp_path / "real.png").write_bytes(b"\x00")
        result = collect_images(tmp_path)
        assert len(result) == 1
        assert result[0].name == "real.png"

    def test_empty_dir_returns_empty(self, tmp_path):
        result = collect_images(tmp_path)
        assert result == []

    def test_case_insensitive_extensions(self, tmp_path):
        for name in ["photo.PNG", "shot.JPG", "art.Jpeg"]:
            (tmp_path / name).write_bytes(b"\x00")
        result = collect_images(tmp_path)
        assert len(result) == 3


# ===========================================================================
# Base64 Encoding
# ===========================================================================


class TestEncodeImage:
    """Verify file-to-base64 encoding."""

    def test_encodes_file_to_base64(self, tmp_path):
        img = tmp_path / "test.png"
        raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        img.write_bytes(raw)
        result = encode_image_b64(img)
        decoded = base64.b64decode(result)
        assert decoded == raw

    def test_returns_plain_base64_no_prefix(self, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0")
        result = encode_image_b64(img)
        # Should NOT have data:image/... prefix
        assert not result.startswith("data:")
        # Should be valid base64
        base64.b64decode(result)


# ===========================================================================
# Request Body Building
# ===========================================================================


class TestBuildBody:
    """Verify API request body construction."""

    def test_minimal_body(self):
        parser = create_parser()
        args = parser.parse_args([
            "--input-dir", "/tmp/x",
            "--prompt", "a cat",
            "--model-name", "klein-9b-kv-fp8",
        ])
        body = build_body(args, "BASE64DATA")
        assert body["prompt"] == "a cat"
        assert body["model_name"] == "klein-9b-kv-fp8"
        assert body["reference_images"] == ["BASE64DATA"]

    def test_omits_none_values(self):
        parser = create_parser()
        args = parser.parse_args([
            "--input-dir", "/tmp/x",
            "--prompt", "test",
            "--model-name", "klein-9b-kv-fp8",
        ])
        body = build_body(args, "B64")
        assert "width" not in body
        assert "height" not in body
        assert "seed" not in body

    def test_includes_explicit_values(self):
        parser = create_parser()
        args = parser.parse_args([
            "--input-dir", "/tmp/x",
            "--prompt", "test",
            "--model-name", "klein-9b-kv-fp8",
            "--width", "1024",
            "--height", "768",
            "--seed", "42",
            "--match-image-size", "0 (First Image)",
        ])
        body = build_body(args, "B64")
        assert body["width"] == 1024
        assert body["height"] == 768
        assert body["seed"] == 42
        assert body["match_image_size"] == "0 (First Image)"

    def test_excludes_batch_specific_fields(self):
        """Batch-only args (input_dir, output_dir, etc.) must not leak into API body."""
        parser = create_parser()
        args = parser.parse_args([
            "--input-dir", "/tmp/x",
            "--output-dir", "/tmp/y",
            "--prompt", "test",
            "--model-name", "klein-9b-kv-fp8",
            "--server", "http://localhost:9000",
            "--timeout", "600",
            "--no-resume",
        ])
        body = build_body(args, "B64")
        for key in ("input_dir", "output_dir", "server", "timeout", "no_resume", "config"):
            assert key not in body


# ===========================================================================
# Resume / Skip Logic
# ===========================================================================


class TestShouldSkip:
    """Verify resume support -- skip images whose output already exists."""

    def test_skip_when_output_exists(self, tmp_path):
        out = tmp_path / "photo.png"
        out.write_bytes(b"\x89PNG")
        assert should_skip(tmp_path, "photo.png") is True

    def test_no_skip_when_output_missing(self, tmp_path):
        assert should_skip(tmp_path, "photo.png") is False

    def test_no_skip_when_output_empty(self, tmp_path):
        (tmp_path / "photo.png").write_bytes(b"")
        assert should_skip(tmp_path, "photo.png") is False


# ===========================================================================
# Output Path
# ===========================================================================


class TestOutputPath:
    """Verify output filename generation."""

    def test_adds_edited_suffix_and_png(self):
        result = output_path_for(Path("/out"), Path("/in/photo_001.png"))
        assert result == Path("/out/photo_001_edited.png")

    def test_jpg_input_becomes_png_output(self):
        result = output_path_for(Path("/out"), Path("/in/shot.jpg"))
        assert result == Path("/out/shot_edited.png")
