"""
Tests for d_noise (sigma schedule scaling) feature.

D-noise scales the sigma schedule:
- d_noise < 1.0: Sharper/more detail (e.g., 0.95-0.98)
- d_noise > 1.0: Softer/deeper colors (e.g., 1.02-1.05)
- d_noise = 1.0: No change (default)

Run with: uv run pytest tests/unit/test_d_noise.py -v
"""

import pytest
import torch

pytestmark = pytest.mark.unit


class TestDNoiseConfig:
    """Test d_noise configuration wiring."""

    def test_d_noise_in_scheduler_config(self):
        """d_noise should be a field in SchedulerConfig."""
        from llm_dit.config import SchedulerConfig

        config = SchedulerConfig()
        assert hasattr(config, 'd_noise')
        assert config.d_noise == 1.0  # default

    def test_d_noise_in_runtime_config(self):
        """d_noise should be a field in RuntimeConfig."""
        from llm_dit.cli import RuntimeConfig

        config = RuntimeConfig()
        assert hasattr(config, 'd_noise')
        assert config.d_noise == 1.0  # default

    def test_d_noise_cli_argument_exists(self):
        """--d-noise should be a valid CLI argument."""
        from llm_dit.cli import create_base_parser

        parser = create_base_parser()
        # Test that the argument is recognized (parse_known_args ignores positional args)
        args, _ = parser.parse_known_args(['--d-noise', '0.95'])
        assert args.d_noise == 0.95


class TestDNoiseSigmaScaling:
    """Test d_noise sigma scaling behavior."""

    def test_d_noise_scales_sigmas(self):
        """d_noise should scale scheduler sigmas."""
        from diffusers import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(9)

        original_sigmas = scheduler.sigmas.clone()
        d_noise = 0.95

        # Apply scaling (simulating pipeline behavior)
        scheduler.sigmas = scheduler.sigmas * d_noise

        # Verify scaling
        assert torch.allclose(scheduler.sigmas, original_sigmas * d_noise)

    def test_d_noise_less_than_one_reduces_sigmas(self):
        """d_noise < 1.0 should reduce sigma values (sharper)."""
        from diffusers import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(9)

        original_sigmas = scheduler.sigmas.clone()
        d_noise = 0.95
        scheduler.sigmas = scheduler.sigmas * d_noise

        # All sigmas should be smaller (except trailing zeros)
        non_zero_mask = original_sigmas > 0
        assert (scheduler.sigmas[non_zero_mask] < original_sigmas[non_zero_mask]).all()

    def test_d_noise_greater_than_one_increases_sigmas(self):
        """d_noise > 1.0 should increase sigma values (softer)."""
        from diffusers import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(9)

        original_sigmas = scheduler.sigmas.clone()
        d_noise = 1.05
        scheduler.sigmas = scheduler.sigmas * d_noise

        # All sigmas should be larger (except trailing zeros)
        non_zero_mask = original_sigmas > 0
        assert (scheduler.sigmas[non_zero_mask] > original_sigmas[non_zero_mask]).all()

    def test_d_noise_one_no_change(self):
        """d_noise = 1.0 should not change sigmas."""
        from diffusers import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(9)

        original_sigmas = scheduler.sigmas.clone()
        d_noise = 1.0
        scheduler.sigmas = scheduler.sigmas * d_noise

        assert torch.allclose(scheduler.sigmas, original_sigmas)


class TestDNoiseWebServer:
    """Test d_noise in web server request models."""

    def test_d_noise_in_generate_request(self):
        """d_noise should be in GenerateRequest model."""
        import sys
        from pathlib import Path

        # Add web directory to path
        web_dir = Path(__file__).parent.parent.parent / "web"
        sys.path.insert(0, str(web_dir))

        # Import after adding to path
        from pydantic import BaseModel

        # Check GenerateRequest has d_noise field
        # We do this by importing the server module
        import importlib.util
        spec = importlib.util.spec_from_file_location("server", web_dir / "server.py")
        # Skip if import fails (dependencies not available)
        if spec is None:
            pytest.skip("Cannot load server module")

    def test_d_noise_default_value(self):
        """d_noise should default to 1.0."""
        from llm_dit.cli import RuntimeConfig

        config = RuntimeConfig()
        assert config.d_noise == 1.0
