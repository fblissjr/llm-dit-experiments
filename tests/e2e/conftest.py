"""
Pytest configuration for E2E tests.

Last Updated: 2026-01-19

Provides shared fixtures and configuration for end-to-end tests.
This file is designed to be portable - copy to LTX-2 repo along with
tests/backends/ for 1:1 comparison testing.

Logging Structure:
    outputs/tests/baseline/{backend}/{test_name}_{timestamp}/
    ├── video.mp4           # Generated video
    ├── metadata.json       # Config, stats, params
    ├── generation.log      # INFO+ generation progress
    ├── debug.log           # DEBUG+ full trace
    └── errors.log          # WARNING+ issues only

    outputs/tests/runs/{timestamp}/
    ├── session.log         # Full session log (all tests)
    ├── summary.json        # Test results summary
    └── environment.json    # GPU, backend, versions
"""

import gc
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytest
import torch

# =============================================================================
# Session-Level Logging Setup
# =============================================================================

_session_timestamp: Optional[str] = None
_session_log_dir: Optional[Path] = None
_session_file_handler: Optional[logging.FileHandler] = None


def _get_session_timestamp() -> str:
    """Get or create session timestamp (consistent across all tests)."""
    global _session_timestamp
    if _session_timestamp is None:
        _session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _session_timestamp


def _get_session_log_dir() -> Path:
    """Get or create session log directory."""
    global _session_log_dir
    if _session_log_dir is None:
        timestamp = _get_session_timestamp()
        _session_log_dir = Path(f"outputs/tests/runs/{timestamp}")
        _session_log_dir.mkdir(parents=True, exist_ok=True)
    return _session_log_dir


def _setup_session_logging():
    """Setup session-level file logging."""
    global _session_file_handler

    if _session_file_handler is not None:
        return  # Already setup

    log_dir = _get_session_log_dir()
    session_log_path = log_dir / "session.log"

    # Create file handler for session log
    _session_file_handler = logging.FileHandler(session_log_path, mode="w")
    _session_file_handler.setLevel(logging.DEBUG)
    _session_file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)8s] %(name)s: %(message)s")
    )

    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(_session_file_handler)
    root_logger.setLevel(logging.DEBUG)

    # Also capture warnings
    logging.captureWarnings(True)


# Setup session logging immediately on import
_setup_session_logging()
logger = logging.getLogger(__name__)


# =============================================================================
# Backend Import Handling
# =============================================================================

_backends_imported = False
_backend_module = None

try:
    from tests import backends as _backend_module
    _backends_imported = True
except ImportError:
    pass

if not _backends_imported:
    try:
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


def _get_environment_info() -> dict:
    """Collect environment information for reproducibility."""
    info = {
        "timestamp": _get_session_timestamp(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": has_cuda(),
    }

    if has_cuda():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_vram_gb"] = round(get_vram_gb(), 2)

    if _backends_imported:
        info["backend"] = _backend_module.get_backend_name()

    info["models_available"] = models_available()

    return info


# =============================================================================
# Pytest Hooks
# =============================================================================


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests based on available hardware and backends."""
    for item in items:
        if "e2e" in item.keywords and not has_cuda():
            item.add_marker(pytest.mark.skip(reason="CUDA not available"))

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

    backend = config.getoption("--backend")
    if backend:
        os.environ["LLM_DIT_TEST_BACKEND"] = backend

    # Save environment info at session start
    log_dir = _get_session_log_dir()
    env_info = _get_environment_info()
    with open(log_dir / "environment.json", "w") as f:
        json.dump(env_info, f, indent=2)

    logger.info(f"Test session started: {_get_session_timestamp()}")
    logger.info(f"Session logs: {log_dir}")
    logger.info(f"Backend: {env_info.get('backend', 'unknown')}")
    if has_cuda():
        logger.info(f"GPU: {env_info.get('gpu_name')} ({env_info.get('gpu_vram_gb')}GB)")


def pytest_sessionfinish(session, exitstatus):
    """Save session summary at end."""
    log_dir = _get_session_log_dir()

    summary = {
        "timestamp": _get_session_timestamp(),
        "exit_status": exitstatus,
        "total_tests": session.testscollected,
        "passed": session.testscollected - session.testsfailed - getattr(session, "testsskipped", 0),
        "failed": session.testsfailed,
    }

    with open(log_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Session complete: {summary['passed']} passed, {summary['failed']} failed")


# =============================================================================
# Per-Test Logging
# =============================================================================


class TestLogHandler:
    """Manages per-test log files in the output directory."""

    def __init__(self, output_dir: Path, test_name: str):
        self.output_dir = output_dir
        self.test_name = test_name
        self.handlers: list[logging.Handler] = []

    def setup(self):
        """Setup log handlers for this test."""
        root_logger = logging.getLogger()

        # generation.log - INFO and above (progress, key events)
        gen_handler = logging.FileHandler(self.output_dir / "generation.log", mode="w")
        gen_handler.setLevel(logging.INFO)
        gen_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)8s] %(name)s: %(message)s")
        )
        root_logger.addHandler(gen_handler)
        self.handlers.append(gen_handler)

        # debug.log - DEBUG and above (full trace)
        debug_handler = logging.FileHandler(self.output_dir / "debug.log", mode="w")
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)8s] %(name)s (%(filename)s:%(lineno)d): %(message)s")
        )
        root_logger.addHandler(debug_handler)
        self.handlers.append(debug_handler)

        # errors.log - WARNING and above (issues only)
        error_handler = logging.FileHandler(self.output_dir / "errors.log", mode="w")
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)8s] %(name)s (%(filename)s:%(lineno)d): %(message)s")
        )
        root_logger.addHandler(error_handler)
        self.handlers.append(error_handler)

        logger.info(f"Test started: {self.test_name}")
        logger.info(f"Output dir: {self.output_dir}")

    def teardown(self):
        """Remove log handlers for this test."""
        root_logger = logging.getLogger()
        for handler in self.handlers:
            handler.flush()
            handler.close()
            root_logger.removeHandler(handler)
        self.handlers.clear()


# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def session_log_dir() -> Path:
    """Get session log directory."""
    return _get_session_log_dir()


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
    timestamp = _get_session_timestamp()
    base = Path(f"outputs/tests/baseline/{backend_name}/{timestamp}")
    base.mkdir(parents=True, exist_ok=True)
    return base


@pytest.fixture
def output_dir(output_base, request) -> Path:
    """Get output directory for this specific test with per-test logging."""
    # Sanitize test name for filesystem
    test_name = request.node.name.replace("[", "_").replace("]", "").replace("/", "_")
    out_dir = output_base / test_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup per-test logging
    log_handler = TestLogHandler(out_dir, test_name)
    log_handler.setup()

    yield out_dir

    # Teardown per-test logging
    log_handler.teardown()


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
    """Minimal configuration for smoke tests (~30s, 14GB VRAM)."""
    if not _backends_imported:
        pytest.skip("Backend module not available")
    # Import from single source of truth
    return _backend_module.SMOKE_CONFIG


@pytest.fixture
def short_config():
    """Short configuration for quick but meaningful tests (~2min, 16GB VRAM)."""
    if not _backends_imported:
        pytest.skip("Backend module not available")
    return _backend_module.SHORT_CONFIG


@pytest.fixture
def reference_config():
    """Official LTX-2 reference configuration (~10min, 20GB VRAM)."""
    if not _backends_imported:
        pytest.skip("Backend module not available")
    return _backend_module.REFERENCE_CONFIG
