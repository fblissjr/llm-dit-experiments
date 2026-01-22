"""
Pytest configuration for E2E tests.

Last Updated: 2026-01-22

Provides shared fixtures and configuration for end-to-end tests.
This file is designed to be portable - copy to LTX-2 repo along with
tests/backends/ for 1:1 comparison testing.

Output Structure (consolidated - one directory per test):
    outputs/tests/runs/{backend}_{test_name}_{timestamp}/
    ├── video.mp4           # Generated video
    ├── metadata.json       # Config, stats, params
    ├── generation.log      # INFO+ generation progress
    ├── debug.log           # DEBUG+ full trace
    ├── errors.log          # WARNING+ issues only
    ├── summary.json        # Test results summary (written at session end)
    └── environment.json    # GPU, backend, versions (written at session end)
"""

import gc
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytest
import torch

# =============================================================================
# Session-Level State
# =============================================================================

_session_timestamp: Optional[str] = None
_session_environment_info: Optional[dict] = None
_test_output_dirs: list[Path] = []  # Track all test output directories


def _get_session_timestamp() -> str:
    """Get or create session timestamp (consistent across all tests)."""
    global _session_timestamp
    if _session_timestamp is None:
        _session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _session_timestamp


def _register_test_output_dir(output_dir: Path) -> None:
    """Register a test output directory for session file writing."""
    global _test_output_dirs
    if output_dir not in _test_output_dirs:
        _test_output_dirs.append(output_dir)


def _get_test_output_dirs() -> list[Path]:
    """Get all registered test output directories."""
    return _test_output_dirs


# Configure root logger at import (console only, files added per-test)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
)
logging.captureWarnings(True)
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
    logger.warning("Could not import tests.backends module. Backend-based tests will be skipped.")


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
    """Collect and cache environment information for reproducibility."""
    global _session_environment_info
    if _session_environment_info is not None:
        return _session_environment_info

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

    _session_environment_info = info
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
                item.add_marker(pytest.mark.skip(reason="Need --runslow option to run"))


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

    # Collect environment info (written to test output dirs at session end)
    env_info = _get_environment_info()

    logger.info(f"Test session started: {_get_session_timestamp()}")
    logger.info(f"Backend: {env_info.get('backend', 'unknown')}")
    if has_cuda():
        logger.info(f"GPU: {env_info.get('gpu_name')} ({env_info.get('gpu_vram_gb')}GB)")


def pytest_sessionfinish(session, exitstatus):
    """Save session summary and environment to test output directories."""
    timestamp = _get_session_timestamp()
    test_output_dirs = _get_test_output_dirs()

    summary = {
        "timestamp": timestamp,
        "exit_status": exitstatus,
        "total_tests": session.testscollected,
        "passed": session.testscollected
        - session.testsfailed
        - getattr(session, "testsskipped", 0),
        "failed": session.testsfailed,
    }

    env_info = _get_environment_info()

    # Write summary.json and environment.json to each test output directory
    for output_dir in test_output_dirs:
        if output_dir.exists():
            with open(output_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            with open(output_dir / "environment.json", "w") as f:
                json.dump(env_info, f, indent=2)

    logger.info(f"Session complete: {summary['passed']} passed, {summary['failed']} failed")

    # Print clear output summary to console
    print("\n" + "=" * 60)
    print("OUTPUT SUMMARY")
    print("=" * 60)

    runs_dir = Path("outputs/tests/runs")

    if test_output_dirs:
        print(f"Test outputs:   {runs_dir}/")
        for test_dir in test_output_dirs:
            files = list(test_dir.iterdir()) if test_dir.exists() else []
            # Simple formatter to show first few files
            file_list = ", ".join(f.name for f in sorted(files)[:5])
            if len(files) > 5:
                file_list += f", ... (+{len(files) - 5} more)"

            print(f"  └── {test_dir.name}/")
            if files:
                print(f"      {file_list}")
            else:
                print(f"      (empty - test may have failed before output)")
    else:
        print(f"Test outputs:   (none created in {runs_dir})")

    print("=" * 60)


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
            logging.Formatter(
                "%(asctime)s [%(levelname)8s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
            )
        )
        root_logger.addHandler(debug_handler)
        self.handlers.append(debug_handler)

        # errors.log - WARNING and above (issues only)
        error_handler = logging.FileHandler(self.output_dir / "errors.log", mode="w")
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)8s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
            )
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
def backend_name():
    """Get name of the active backend."""
    if not _backends_imported:
        return "none"
    return _backend_module.get_backend_name()


@pytest.fixture(scope="session")
def backend():
    """Get the video generation backend (session-scoped for efficiency)."""
    return get_backend_or_skip()


@pytest.fixture
def output_dir(backend_name, request) -> Path:
    """Get output directory for this specific test.

    All test outputs go to: outputs/tests/runs/{backend}_{test}_{timestamp}/
    Session files (summary.json, environment.json) are written here at session end.
    """
    # 1. Get components
    timestamp = _get_session_timestamp()
    # Sanitize test name (handle pytest parametrization like [1-2])
    test_name = request.node.name.replace("[", "_").replace("]", "").replace("/", "_")

    # 2. Construct the flat path: outputs/tests/runs/{backend}_{test}_{time}
    dir_name = f"{backend_name}_{test_name}_{timestamp}"
    out_dir = Path("outputs/tests/runs") / dir_name

    # 3. Create directory and register for session file writing
    out_dir.mkdir(parents=True, exist_ok=True)
    _register_test_output_dir(out_dir)

    # 4. Setup per-test logging (generation.log, debug.log, errors.log)
    log_handler = TestLogHandler(out_dir, test_name)
    log_handler.setup()

    yield out_dir

    # 5. Teardown
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
