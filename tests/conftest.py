"""
Shared pytest fixtures and configuration for llm-dit-experiments.

Environment Variables:
    Z_IMAGE_MODEL_PATH: Path to Z-Image model directory
    HEYLOOKITSANLLM_URL: URL for heylookitsanllm API server (e.g., http://localhost:8080)
    TEST_LORA_PATH: Path to test LoRA weights

Example Usage:
    # Run unit tests only (no GPU needed)
    pytest -m unit

    # Run integration tests with model
    Z_IMAGE_MODEL_PATH=/path/to/model pytest -m integration

    # Run API tests (requires heylookitsanllm running)
    HEYLOOKITSANLLM_URL=http://localhost:8080 pytest -m requires_api

    # Run all tests on Linux with GPU
    Z_IMAGE_MODEL_PATH=/path/to/model pytest
"""

import os
import sys
from pathlib import Path

import pytest
import torch


# Environment detection
def is_mac():
    return sys.platform == "darwin"


def is_linux():
    return sys.platform == "linux"


def has_cuda():
    return torch.cuda.is_available()


def has_mps():
    return torch.backends.mps.is_available()


# Skip decorators for use in tests
skip_on_mac = pytest.mark.skipif(is_mac(), reason="Linux-only test")
skip_on_linux = pytest.mark.skipif(is_linux(), reason="macOS-only test")
skip_without_cuda = pytest.mark.skipif(not has_cuda(), reason="CUDA required")
skip_without_mps = pytest.mark.skipif(not has_mps(), reason="MPS required")


# Auto-skip GPU tests when CUDA not available
def pytest_collection_modifyitems(config, items):
    """Auto-skip tests based on available hardware."""
    for item in items:
        # Skip GPU tests if CUDA not available
        if "requires_gpu" in item.keywords and not has_cuda():
            item.add_marker(pytest.mark.skip(reason="CUDA GPU not available"))

        # Skip API tests if URL not set
        if "requires_api" in item.keywords and not os.getenv("HEYLOOKITSANLLM_URL"):
            item.add_marker(pytest.mark.skip(reason="HEYLOOKITSANLLM_URL not set"))

        # Skip model tests if path not set
        if "requires_model" in item.keywords and not os.getenv("Z_IMAGE_MODEL_PATH"):
            item.add_marker(pytest.mark.skip(reason="Z_IMAGE_MODEL_PATH not set"))

        # Skip LoRA tests if path not set
        if "requires_lora" in item.keywords and not os.getenv("TEST_LORA_PATH"):
            item.add_marker(pytest.mark.skip(reason="TEST_LORA_PATH not set"))


# Session-scoped fixtures (loaded once per test session)
@pytest.fixture(scope="session")
def z_image_model_path():
    """Path to Z-Image model (from environment or skip)."""
    path = os.getenv("Z_IMAGE_MODEL_PATH")
    if path is None:
        pytest.skip("Z_IMAGE_MODEL_PATH not set")
    if not Path(path).exists():
        pytest.skip(f"Z_IMAGE_MODEL_PATH does not exist: {path}")
    return path


@pytest.fixture(scope="session")
def templates_dir():
    """Path to templates directory."""
    return Path(__file__).parent.parent / "templates" / "z_image"


@pytest.fixture(scope="session")
def api_server_url():
    """URL for heylookitsanllm API (for distributed tests)."""
    url = os.getenv("HEYLOOKITSANLLM_URL")
    if url is None:
        pytest.skip("HEYLOOKITSANLLM_URL not set")
    return url


@pytest.fixture(scope="session")
def lora_path():
    """Path to test LoRA weights."""
    path = os.getenv("TEST_LORA_PATH")
    if path is None:
        pytest.skip("TEST_LORA_PATH not set")
    return path


# Mock fixtures for testing without real backends
@pytest.fixture
def mock_backend():
    """Mock backend for testing without real model."""
    from unittest.mock import MagicMock

    from llm_dit.backends.protocol import EncodingOutput

    backend = MagicMock()
    backend.embedding_dim = 2560
    backend.max_sequence_length = 512
    backend.device = torch.device("cpu")
    backend.dtype = torch.bfloat16

    def mock_encode(texts, return_padded=False):
        # Return realistic-shaped embeddings
        embeddings = [torch.randn(len(t.split()) + 10, 2560) for t in texts]
        masks = [torch.ones(e.shape[0], dtype=torch.bool) for e in embeddings]
        return EncodingOutput(embeddings=embeddings, attention_masks=masks)

    backend.encode = mock_encode
    return backend


@pytest.fixture
def mock_api_response():
    """Mock httpx response for API backend testing."""
    import base64

    import numpy as np

    # Create fake hidden states matching Qwen3-4B output
    seq_len = 50
    hidden_dim = 2560
    hidden_states = np.random.randn(seq_len, hidden_dim).astype(np.float32)
    encoded = base64.b64encode(hidden_states.tobytes()).decode("ascii")

    return {
        "hidden_states": encoded,
        "shape": [seq_len, hidden_dim],
        "dtype": "float32",
        "encoding_format": "base64",
    }


@pytest.fixture
def sample_prompts():
    """Sample prompts for testing."""
    return [
        "A cat sleeping in sunlight",
        "A mountain landscape at sunset",
        "A portrait of an old man with a beard",
        "An anime character with blue hair",
        "A futuristic city at night",
    ]


@pytest.fixture
def sample_formatted_prompt():
    """Pre-formatted prompt in Qwen3 chat template format."""
    return """<|im_start|>user
A cat sleeping in sunlight<|im_end|>
<|im_start|>assistant
<think>
Soft golden light, peaceful expression, curled up position.
</think>

"""


# Temporary directory fixtures
@pytest.fixture
def output_dir(tmp_path):
    """Temporary output directory for generated images."""
    output = tmp_path / "output"
    output.mkdir()
    return output


@pytest.fixture
def test_config_file(tmp_path):
    """Create a temporary test config file."""
    config_path = tmp_path / "test_config.toml"
    config_path.write_text(
        """
[default]
model_path = "/test/path"
templates_dir = "templates/z_image"

[default.encoder]
device = "cpu"
torch_dtype = "bfloat16"
quantization = "none"
cpu_offload = false
max_length = 512

[default.pipeline]
device = "cpu"
torch_dtype = "bfloat16"

[default.generation]
height = 512
width = 512
num_inference_steps = 4
guidance_scale = 0.0
enable_thinking = true

[default.scheduler]
shift = 3.0

[default.optimization]
flash_attn = false
compile = false
cpu_offload = false

[default.pytorch]
attention_backend = "auto"
use_custom_scheduler = false
tiled_vae = false
tile_size = 512
tile_overlap = 64

[default.lora]
paths = []
scales = []

[low_vram]
model_path = "/test/path"

[low_vram.encoder]
device = "cuda"
quantization = "8bit"
cpu_offload = true
"""
    )
    return config_path
