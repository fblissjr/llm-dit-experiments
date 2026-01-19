"""
Pytest configuration for E2E tests.

Last Updated: 2026-01-19

Provides shared fixtures and configuration for end-to-end tests.
This file is designed to be portable - copy to LTX-2 repo along with
tests/backends/ for 1:1 comparison testing.
"""

import gc
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Backend Import Handling
# =============================================================================

# Try multiple import paths for portability
_backends_imported = False
_backend_module = None

try:
    # Primary: Standard package import
    from tests import backends as _backend_module

    _backends_imported = True
except ImportError:
    pass

if not _backends_imported:
    try:
        # Secondary: Relative import (when in LTX-2 repo)
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import backends as _backend_module

        _backends_imported = True
    except ImportError:
        pass

if not _backends_imported:
    logger.warning(
        "Could not import tests.backends module. "
        "Backend-based tests will be skipped."
    )


def get_backend_or_skip():
    """Get backend or skip test if not available."""
    if not _backends_imported:
        pytest.skip("tests.backends module not available")
    backend_name = _backend_module.get_backend_name()
    if backend_name == "none":
        pytest.skip("No video generation backend available")
    return _backend_module.get_backend()


# =============================================================================
# Hardware Detection
# =============================================================================


def has_cuda() -> bool:
    return torch.cuda.is_available()


def get_vram_gb() -> float:
    """Get available VRAM in GB."""
    if not has_cuda():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def has_sufficient_vram(minimum_gb: float = 16.0) -> bool:
    """Check if GPU has sufficient VRAM."""
    return get_vram_gb() >= minimum_gb


def models_available(model_path_str: str = "models/LTX-2") -> bool:
    """Check if LTX-2 models are available."""
    model_path = Path(model_path_str)
    transformer_exists = (model_path / "transformer").exists()
    encoder_exists = (model_path / "text_encoder").exists()
    return transformer_exists and encoder_exists


# =============================================================================
# Pytest Hooks
# =============================================================================


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests based on available hardware and backends."""
    for item in items:
        # Skip GPU tests if CUDA not available
        if "e2e" in item.keywords and not has_cuda():
            item.add_marker(pytest.mark.skip(reason="CUDA not available"))

        # Skip slow tests unless explicitly requested
        if "slow" in item.keywords:
            if not config.getoption("--runslow", default=False):
                item.add_marker(
                    pytest.mark.skip(reason="Need --runslow option to run")
                )


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run slow tests (reference quality tests)",
    )
    parser.addoption(
        "--backend",
        action="store",
        default=None,
        help="Force specific backend (llm_dit or ltx2)",
    )


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end")

    # Set backend override from command line
    backend = config.getoption("--backend")
    if backend:
        os.environ["LLM_DIT_TEST_BACKEND"] = backend


# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def backend_name():
    """Get name of the active backend."""
    if not _backends_imported:
        return "none"
    return _backend_module.get_backend_name()


@pytest.fixture(scope="session")
def backend():
    """Get the video generation backend (session-scoped for efficiency)."""
    return get_backend_or_skip()


@pytest.fixture(scope="module")
def output_base(backend_name) -> Path:
    """Get output base directory for test results."""
    base = Path(f"outputs/tests/baseline/{backend_name}")
    base.mkdir(parents=True, exist_ok=True)
    return base


@pytest.fixture
def output_dir(output_base, request) -> Path:
    """Get timestamped output directory for this specific test."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize test name for filesystem
    test_name = request.node.name.replace("[", "_").replace("]", "").replace("/", "_")
    out_dir = output_base / f"{test_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.fixture(autouse=True)
def cleanup_gpu():
    """Clean up GPU memory before and after each test."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@pytest.fixture
def tmp_output_dir(tmp_path) -> Path:
    """Temporary output directory (cleaned up after test)."""
    output = tmp_path / "output"
    output.mkdir()
    return output


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def smoke_prompt() -> str:
    """Minimal prompt for smoke tests."""
    return "A cat walking"


@pytest.fixture
def reference_prompts() -> dict:
    """Official LTX-2 reference prompts."""
    return {
        "cat_walking": "A cat walking through a sunny garden",
        "sunset": "A beautiful sunset over the ocean with gentle waves",
        "city_night": "A futuristic city at night with neon lights",
    }


@pytest.fixture
def smoke_config():
    """Minimal configuration for smoke tests."""
    if not _backends_imported:
        pytest.skip("Backend module not available")
    return _backend_module.GenerationConfig(
        num_frames=9,
        height=256,
        width=384,
        num_inference_steps=2,
        guidance_scale=1.0,
        seed=10,
        fp8=True,
    )


@pytest.fixture
def short_config():
    """Short configuration for quick but meaningful tests."""
    if not _backends_imported:
        pytest.skip("Backend module not available")
    return _backend_module.GenerationConfig(
        num_frames=33,
        height=384,
        width=512,
        num_inference_steps=10,
        guidance_scale=3.0,
        seed=10,
        fp8=True,
    )


@pytest.fixture
def reference_config():
    """Official LTX-2 reference configuration."""
    if not _backends_imported:
        pytest.skip("Backend module not available")
    return _backend_module.GenerationConfig(
        num_frames=121,
        height=512,
        width=768,
        num_inference_steps=40,
        guidance_scale=4.0,
        seed=10,
        fp8=True,
    )
