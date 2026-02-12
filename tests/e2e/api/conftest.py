"""Pytest configuration for API E2E tests.

last updated: 2026-02-12

Provides TestClient-based fixtures that exercise the full FastAPI stack:
Pydantic validation -> router logic -> dependency injection -> ModelManager -> pipeline.

Config loaded from TOML overlays using the same Config.from_toml() ->
RuntimeConfig.from_toml_config() path the real server uses.

Output structure:
    outputs/tests/runs/api_{pipeline}_{test_name}_{timestamp}/
"""

import gc
import logging
import time
from datetime import datetime
from pathlib import Path

import pytest
import torch
from starlette.testclient import TestClient

from tests.e2e.api.config_factory import load_test_config
from tests.e2e.api.run_recorder import RunRecorder

logger = logging.getLogger(__name__)

# Track all output dirs for session-end summary
_test_output_dirs: list[Path] = []
_session_timestamp: str | None = None


def _get_session_timestamp() -> str:
    global _session_timestamp
    if _session_timestamp is None:
        _session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _session_timestamp


# ---------------------------------------------------------------------------
# Log handler (adapted from tests/integration/pipeline/conftest.py)
# ---------------------------------------------------------------------------


class TestLogHandler:
    """Manages per-test log files in the output directory."""

    def __init__(self, output_dir: Path, test_name: str):
        self.output_dir = output_dir
        self.test_name = test_name
        self.handlers: list[logging.Handler] = []

    def setup(self):
        root_logger = logging.getLogger()

        gen_handler = logging.FileHandler(self.output_dir / "generation.log", mode="w")
        gen_handler.setLevel(logging.INFO)
        gen_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)8s] %(name)s: %(message)s")
        )
        root_logger.addHandler(gen_handler)
        self.handlers.append(gen_handler)

        debug_handler = logging.FileHandler(self.output_dir / "debug.log", mode="w")
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)8s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
            )
        )
        root_logger.addHandler(debug_handler)
        self.handlers.append(debug_handler)

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
        root_logger = logging.getLogger()
        for handler in self.handlers:
            handler.flush()
            handler.close()
            root_logger.removeHandler(handler)
        self.handlers.clear()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config_overlay():
    """Override in test modules to select the TOML overlay.

    Example in test_flux2_smoke.py:
        @pytest.fixture(scope="module")
        def config_overlay():
            return "flux2_smoke"
    """
    raise NotImplementedError(
        "Test module must define a config_overlay fixture returning "
        "the overlay name (e.g., 'flux2_smoke')"
    )


@pytest.fixture(scope="module")
def pipeline_config(config_overlay):
    """Load RuntimeConfig from the module's TOML overlay.

    Returns (RuntimeConfig, path_to_frozen_toml).
    """
    return load_test_config(config_overlay)


@pytest.fixture(scope="module")
def api_client(pipeline_config):
    """TestClient with ModelManager + config on app.state.

    Session-scoped: models load once, shared across all tests in the module.
    Mirrors the initialization in server.py:main().
    """
    runtime_config, config_path = pipeline_config

    # Import and set up the FastAPI app
    import web.server as srv
    from web.server import app, _register_routers

    # Register routers (guard: only if not already registered)
    if len(app.routes) <= 2:  # Only default routes (root + openapi)
        _register_routers()

    # Initialize ModelManager (same as server.py main())
    from llm_dit.model_manager import ModelManager

    manager = ModelManager(runtime_config)

    # Set app.state for DI (ConfigDep, ManagerDep)
    app.state.runtime_config = runtime_config
    app.state.model_manager = manager

    # Module-level globals some routers still access
    srv.runtime_config = runtime_config
    srv.model_manager = manager
    srv.encoder_only_mode = False
    srv.generation_history = []
    srv.server_start_time = time.time()

    with TestClient(app) as client:
        yield client

    # Cleanup: unload all pipelines
    manager.unload_all_except(keep=None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def run_recorder(api_client, pipeline_config, config_overlay, request):
    """Per-test RunRecorder: creates output dir, captures metadata.

    Yields a RunRecorder that wraps the TestClient with full metadata capture.
    On teardown, writes manifest.json.
    """
    _, config_path = pipeline_config
    timestamp = _get_session_timestamp()
    test_name = request.node.name.replace("[", "_").replace("]", "").replace("/", "_")

    # Determine pipeline from overlay name
    pipeline = config_overlay.split("_")[0]

    # Output directory
    dir_name = f"api_{pipeline}_{test_name}_{timestamp}"
    output_dir = Path("outputs/tests/runs") / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    _test_output_dirs.append(output_dir)

    # Per-test logging
    log_handler = TestLogHandler(output_dir, test_name)
    log_handler.setup()

    recorder = RunRecorder(
        client=api_client,
        output_dir=output_dir,
        config_path=config_path,
        pipeline=pipeline,
        test_name=test_name,
        config_overlay=config_overlay,
    )

    # Capture server context before the test
    recorder.capture_context()

    yield recorder

    # Finalize: write manifest
    recorder.finalize()
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


# ---------------------------------------------------------------------------
# Session hooks
# ---------------------------------------------------------------------------


def pytest_sessionfinish(session, exitstatus):
    """Print output summary at session end."""
    print("\n" + "=" * 60)
    print("API E2E OUTPUT SUMMARY")
    print("=" * 60)

    runs_dir = Path("outputs/tests/runs")

    if _test_output_dirs:
        print(f"Test outputs:   {runs_dir}/")
        for test_dir in _test_output_dirs:
            files = list(test_dir.iterdir()) if test_dir.exists() else []
            file_list = ", ".join(f.name for f in sorted(files)[:5])
            if len(files) > 5:
                file_list += f", ... (+{len(files) - 5} more)"
            print(f"  -- {test_dir.name}/")
            if files:
                print(f"      {file_list}")
    else:
        print(f"Test outputs:   (none created in {runs_dir})")

    print("=" * 60)
