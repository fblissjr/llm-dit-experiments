"""Test config factory: loads TOML overlays merged over config.toml.example.

last updated: 2026-02-12

Merges three sources:
  1. config.toml.example (base defaults)
  2. tests/configs/<overlay>.toml (test-specific overrides)
  3. Model paths from config.toml (real machine paths)

The merged result is written to a temp file and parsed through the same
Config.from_toml() -> RuntimeConfig.from_toml_config() path the real server uses.
"""

import copy
import tempfile
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

import tomli_w

from llm_dit.config import Config, RuntimeConfig

# Project root (where config.toml.example lives)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_BASE_CONFIG = _PROJECT_ROOT / "config.toml.example"
_OVERLAY_DIR = _PROJECT_ROOT / "tests" / "configs"
_REAL_CONFIG = _PROJECT_ROOT / "config.toml"

# TOML sections that contain model paths to extract from config.toml
_MODEL_PATH_KEYS = {
    "flux2": ("model_path", "vae_path", "encoder_path"),
    "ltx2": ("model_path", "transformer_file", "encoder_model_id"),
    "zimage": ("model_path",),
    "qwen_image": ("model_path",),
    "encoder": ("model_path",),
}


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Deep-merge overlay into base (overlay wins on conflicts)."""
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _extract_model_paths(real_config: dict) -> dict:
    """Extract model path fields from the real config.toml.

    Only copies path-related fields to avoid overriding test parameters.
    """
    paths = {}
    for section, keys in _MODEL_PATH_KEYS.items():
        if section in real_config:
            section_paths = {}
            for key in keys:
                if key in real_config[section]:
                    val = real_config[section][key]
                    if val:  # Skip empty strings
                        section_paths[key] = val
            if section_paths:
                paths[section] = section_paths
    return paths


def load_test_config(
    overlay_name: str,
    output_dir: Path | None = None,
) -> tuple[RuntimeConfig, Path]:
    """Load a test config by merging: config.toml.example + overlay + real model paths.

    Args:
        overlay_name: Name of overlay file (without .toml extension) in tests/configs/
        output_dir: If provided, save frozen TOML here as config_frozen.toml

    Returns:
        (RuntimeConfig, path_to_frozen_toml) for reproducibility.
    """
    # 1. Load base config
    with open(_BASE_CONFIG, "rb") as f:
        base = tomllib.load(f)

    # 2. Load overlay
    overlay_path = _OVERLAY_DIR / f"{overlay_name}.toml"
    if not overlay_path.exists():
        available = [p.stem for p in _OVERLAY_DIR.glob("*.toml")]
        raise FileNotFoundError(
            f"Test overlay '{overlay_name}' not found at {overlay_path}. "
            f"Available: {available}"
        )
    with open(overlay_path, "rb") as f:
        overlay = tomllib.load(f)

    # 3. Load model paths from real config (if it exists)
    model_paths = {}
    if _REAL_CONFIG.exists():
        with open(_REAL_CONFIG, "rb") as f:
            real = tomllib.load(f)
        model_paths = _extract_model_paths(real)

    # 4. Merge: base + overlay + model_paths
    merged = _deep_merge(base, overlay)
    merged = _deep_merge(merged, model_paths)

    # 5. Write merged TOML to temp file (or output_dir)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        frozen_path = output_dir / "config_frozen.toml"
    else:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".toml", prefix="test_config_", delete=False
        )
        frozen_path = Path(tmp.name)
        tmp.close()

    with open(frozen_path, "wb") as f:
        tomli_w.dump(merged, f)

    # 6. Parse through the real config pipeline
    config = Config.from_toml(frozen_path)
    runtime_config = RuntimeConfig.from_toml_config(config)

    return runtime_config, frozen_path
