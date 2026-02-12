"""Canonical test parameter constants for all pipelines.

last updated: 2026-02-12

Single source of truth for generation parameters used across:
- TOML overlays (tests/configs/*.toml) for E2E API tests
- Protocol configs (tests/backends/protocol.py) for integration tests
- Validation tests (tests/unit/test_config_consistency.py) for drift detection

See tests/constants/README.md for architecture and how to add new pipelines.
"""

# Tier names used across all pipelines
SMOKE = "smoke"
STANDARD = "standard"
REFERENCE = "reference"

from tests.constants import flux2, ltx2, zimage

_PIPELINE_MODULES = {
    "ltx2": ltx2,
    "flux2": flux2,
    "zimage": zimage,
}


def get_pipeline_constants(pipeline: str):
    """Get the constants module for a pipeline.

    Args:
        pipeline: Pipeline name ("ltx2", "flux2", "zimage")

    Returns:
        The pipeline's constants module.

    Raises:
        KeyError: If pipeline name is not recognized.
    """
    if pipeline not in _PIPELINE_MODULES:
        available = sorted(_PIPELINE_MODULES.keys())
        raise KeyError(f"Unknown pipeline '{pipeline}'. Available: {available}")
    return _PIPELINE_MODULES[pipeline]
