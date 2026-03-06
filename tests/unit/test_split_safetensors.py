"""Tests for LTX-2.3 safetensors split script.

Last Updated: 2026-03-06

Validates the split_safetensors() function that separates bundled
LTX-2.3 fp8 checkpoints into individual component files.

The script splits a single ~28GB bundled file into 5 component files:
  - transformer  (strips "model.diffusion_model." prefix, excludes connectors)
  - connectors   (keeps full prefixes -- connector loader expects them)
  - video-vae    (strips "vae." prefix)
  - audio-vae    (strips "audio_vae." prefix)
  - vocoder      (strips "vocoder." prefix)

Run with: uv run pytest tests/unit/test_split_safetensors.py -v
"""

import logging
import sys
from pathlib import Path

import pytest
import torch

# The split script is not a package; insert the scripts directory so it can be
# imported directly.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from split_ltx23_safetensors import COMPONENT_GROUPS, split_safetensors  # noqa: E402

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXPECTED_COMPONENT_NAMES = {
    "transformer",
    "connectors",
    "video-vae",
    "audio-vae",
    "vocoder",
}

_REQUIRED_GROUP_KEYS = {"name", "output", "prefixes"}


# ---------------------------------------------------------------------------
# Synthetic bundle fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_bundle(tmp_path):
    """Create a small synthetic bundled safetensors file.

    Mirrors the key prefix structure of the real LTX-2.3 fp8 checkpoint so
    that split logic can be exercised without any real model weights.
    """
    from safetensors.torch import save_file

    tensors = {
        # Transformer keys (stripped to bare names)
        "model.diffusion_model.blocks.0.weight": torch.randn(4, 4),
        "model.diffusion_model.norm.weight": torch.randn(4),
        # Connector keys -- excluded from transformer, kept with full prefix
        "model.diffusion_model.video_embeddings_connector.weight": torch.randn(4, 4),
        "model.diffusion_model.audio_embeddings_connector.weight": torch.randn(4, 4),
        # Additional connector prefix
        "text_embedding_projection.weight": torch.randn(4, 4),
        # Video VAE keys
        "vae.encoder.weight": torch.randn(4, 4),
        # Audio VAE keys
        "audio_vae.decoder.weight": torch.randn(4, 4),
        # Vocoder keys
        "vocoder.generator.weight": torch.randn(4, 4),
        # Unclaimed key (no group prefix)
        "some_other_key": torch.randn(4),
    }
    bundle_path = tmp_path / "bundle.safetensors"
    save_file(tensors, str(bundle_path))
    return bundle_path


# ---------------------------------------------------------------------------
# TestComponentGroups
# ---------------------------------------------------------------------------

class TestComponentGroups:
    """Validate COMPONENT_GROUPS structure."""

    def test_five_component_groups(self):
        """There must be exactly 5 component groups."""
        assert len(COMPONENT_GROUPS) == 5

    def test_each_group_has_required_keys(self):
        """Every group dict must have name, output, and prefixes."""
        for group in COMPONENT_GROUPS:
            missing = _REQUIRED_GROUP_KEYS - set(group.keys())
            assert not missing, (
                f"Group '{group.get('name', '?')}' is missing keys: {missing}"
            )

    def test_group_names_are_expected(self):
        """All 5 expected component names must be present."""
        names = {g["name"] for g in COMPONENT_GROUPS}
        assert names == _EXPECTED_COMPONENT_NAMES

    def test_each_group_has_at_least_one_prefix(self):
        """Every group must have at least one prefix to match keys against."""
        for group in COMPONENT_GROUPS:
            assert len(group["prefixes"]) >= 1, (
                f"Group '{group['name']}' has no prefixes"
            )

    def test_transformer_has_exclude_prefixes(self):
        """Transformer group must define exclude_prefixes for connector keys."""
        transformer = next(g for g in COMPONENT_GROUPS if g["name"] == "transformer")
        assert "exclude_prefixes" in transformer
        assert len(transformer["exclude_prefixes"]) >= 1

    def test_connectors_strip_is_none(self):
        """Connector group must have strip=None (keys kept with full prefix)."""
        connectors = next(g for g in COMPONENT_GROUPS if g["name"] == "connectors")
        assert connectors.get("strip") is None

    def test_non_connector_groups_have_strip(self):
        """All groups except connectors must have a non-None strip value."""
        for group in COMPONENT_GROUPS:
            if group["name"] == "connectors":
                continue
            assert group.get("strip") is not None, (
                f"Group '{group['name']}' should have a strip prefix but has None"
            )

    def test_output_filenames_are_safetensors(self):
        """All output filenames must end with .safetensors."""
        for group in COMPONENT_GROUPS:
            assert group["output"].endswith(".safetensors"), (
                f"Group '{group['name']}' output '{group['output']}' is not a .safetensors file"
            )


# ---------------------------------------------------------------------------
# TestSplitSafetensors
# ---------------------------------------------------------------------------

class TestSplitSafetensors:
    """Test split_safetensors() with synthetic data."""

    def test_split_returns_component_counts(self, synthetic_bundle, tmp_path):
        """split_safetensors() returns a dict mapping component name to key count."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        results = split_safetensors(synthetic_bundle, output_dir)
        assert isinstance(results, dict)
        # Every returned key should be a known component name
        for name in results:
            assert name in _EXPECTED_COMPONENT_NAMES

    def test_split_produces_component_files(self, synthetic_bundle, tmp_path):
        """split_safetensors() writes one file per non-empty component."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        results = split_safetensors(synthetic_bundle, output_dir)
        for group in COMPONENT_GROUPS:
            if group["name"] in results:
                output_path = output_dir / group["output"]
                assert output_path.exists(), (
                    f"Expected output file {output_path} for component '{group['name']}'"
                )

    def test_transformer_keys_stripped(self, synthetic_bundle, tmp_path):
        """Transformer keys must have 'model.diffusion_model.' prefix removed."""
        from safetensors import safe_open

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        split_safetensors(synthetic_bundle, output_dir)

        transformer_file = output_dir / "ltx-2.3-transformer-fp8.safetensors"
        assert transformer_file.exists()

        with safe_open(str(transformer_file), framework="pt") as f:
            keys = list(f.keys())

        # No key should start with the original prefix
        for key in keys:
            assert not key.startswith("model.diffusion_model."), (
                f"Transformer key '{key}' still has original prefix"
            )
        # Keys should be the bare names
        assert "blocks.0.weight" in keys or "norm.weight" in keys, (
            f"Expected stripped transformer keys; got: {keys}"
        )

    def test_transformer_excludes_connector_keys(self, synthetic_bundle, tmp_path):
        """Transformer file must NOT contain keys from connector prefixes."""
        from safetensors import safe_open

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        split_safetensors(synthetic_bundle, output_dir)

        transformer_file = output_dir / "ltx-2.3-transformer-fp8.safetensors"
        assert transformer_file.exists()

        with safe_open(str(transformer_file), framework="pt") as f:
            keys = list(f.keys())

        # No connector-derived key should appear in the transformer file
        for key in keys:
            assert "video_embeddings_connector" not in key, (
                f"Connector key '{key}' ended up in transformer file"
            )
            assert "audio_embeddings_connector" not in key, (
                f"Connector key '{key}' ended up in transformer file"
            )

    def test_connectors_keep_full_prefix(self, synthetic_bundle, tmp_path):
        """Connector file keys must retain their original full prefixes (strip=None)."""
        from safetensors import safe_open

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        split_safetensors(synthetic_bundle, output_dir)

        connectors_file = output_dir / "ltx-2.3-connectors.safetensors"
        assert connectors_file.exists()

        with safe_open(str(connectors_file), framework="pt") as f:
            keys = list(f.keys())

        # At least one key should keep the full prefix
        has_full_prefix = any(
            k.startswith("model.diffusion_model.video_embeddings_connector.")
            or k.startswith("model.diffusion_model.audio_embeddings_connector.")
            or k.startswith("text_embedding_projection.")
            for k in keys
        )
        assert has_full_prefix, (
            f"Connector keys should keep full prefixes, got: {keys}"
        )

    def test_vae_keys_stripped(self, synthetic_bundle, tmp_path):
        """Video VAE file keys must have 'vae.' prefix stripped."""
        from safetensors import safe_open

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        split_safetensors(synthetic_bundle, output_dir)

        vae_file = output_dir / "ltx-2.3-video-vae.safetensors"
        assert vae_file.exists()

        with safe_open(str(vae_file), framework="pt") as f:
            keys = list(f.keys())

        for key in keys:
            assert not key.startswith("vae."), (
                f"VAE key '{key}' still has original 'vae.' prefix"
            )
        assert "encoder.weight" in keys, (
            f"Expected 'encoder.weight' in VAE file; got: {keys}"
        )

    def test_audio_vae_keys_stripped(self, synthetic_bundle, tmp_path):
        """Audio VAE file keys must have 'audio_vae.' prefix stripped."""
        from safetensors import safe_open

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        split_safetensors(synthetic_bundle, output_dir)

        audio_vae_file = output_dir / "ltx-2.3-audio-vae.safetensors"
        assert audio_vae_file.exists()

        with safe_open(str(audio_vae_file), framework="pt") as f:
            keys = list(f.keys())

        for key in keys:
            assert not key.startswith("audio_vae."), (
                f"Audio VAE key '{key}' still has original 'audio_vae.' prefix"
            )
        assert "decoder.weight" in keys, (
            f"Expected 'decoder.weight' in audio VAE file; got: {keys}"
        )

    def test_vocoder_keys_stripped(self, synthetic_bundle, tmp_path):
        """Vocoder file keys must have 'vocoder.' prefix stripped."""
        from safetensors import safe_open

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        split_safetensors(synthetic_bundle, output_dir)

        vocoder_file = output_dir / "ltx-2.3-vocoder.safetensors"
        assert vocoder_file.exists()

        with safe_open(str(vocoder_file), framework="pt") as f:
            keys = list(f.keys())

        for key in keys:
            assert not key.startswith("vocoder."), (
                f"Vocoder key '{key}' still has original 'vocoder.' prefix"
            )
        assert "generator.weight" in keys, (
            f"Expected 'generator.weight' in vocoder file; got: {keys}"
        )

    def test_dry_run_no_files_written(self, synthetic_bundle, tmp_path):
        """dry_run=True must not write any files to disk."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        results = split_safetensors(synthetic_bundle, output_dir, dry_run=True)

        # Results dict should still be populated (dry_run reports what would be done)
        assert isinstance(results, dict)
        # But no actual files should exist
        written = list(output_dir.iterdir())
        assert len(written) == 0, (
            f"dry_run=True should write no files, but found: {[f.name for f in written]}"
        )

    def test_unclaimed_keys_detected(self, synthetic_bundle, tmp_path, caplog):
        """Keys not matching any group prefix must be reported as unclaimed."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with caplog.at_level(logging.WARNING):
            split_safetensors(synthetic_bundle, output_dir)

        # The "some_other_key" in our synthetic bundle has no matching prefix
        assert "some_other_key" in caplog.text or "Unclaimed" in caplog.text, (
            "Expected unclaimed key warning; caplog text: " + caplog.text
        )

    def test_empty_bundle_produces_no_outputs(self, tmp_path):
        """A bundle with no matching keys produces an empty results dict."""
        from safetensors.torch import save_file

        # Bundle with only unclaimed keys
        bundle_path = tmp_path / "empty_bundle.safetensors"
        save_file({"unrelated_key": torch.randn(4)}, str(bundle_path))

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        results = split_safetensors(bundle_path, output_dir)

        assert results == {}, f"Expected empty results for unmatched bundle, got: {results}"

    def test_key_count_in_results(self, synthetic_bundle, tmp_path):
        """Results dict values must reflect the number of keys written per component."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        results = split_safetensors(synthetic_bundle, output_dir)

        # Transformer: 2 bare keys (blocks.0.weight, norm.weight)
        assert results.get("transformer") == 2, (
            f"Expected 2 transformer keys, got {results.get('transformer')}"
        )
        # Connectors: 3 keys (video_connector, audio_connector, text_embedding_projection)
        assert results.get("connectors") == 3, (
            f"Expected 3 connector keys, got {results.get('connectors')}"
        )
        # video-vae: 1 key (encoder.weight)
        assert results.get("video-vae") == 1, (
            f"Expected 1 video-vae key, got {results.get('video-vae')}"
        )
        # audio-vae: 1 key (decoder.weight)
        assert results.get("audio-vae") == 1, (
            f"Expected 1 audio-vae key, got {results.get('audio-vae')}"
        )
        # vocoder: 1 key (generator.weight)
        assert results.get("vocoder") == 1, (
            f"Expected 1 vocoder key, got {results.get('vocoder')}"
        )
