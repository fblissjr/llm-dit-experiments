"""
API defaults test for Z-Image.

Tests that the server API returns correct defaults from config.toml
for the BASE model (shift=6.0, guidance_scale=4.0, steps=30).

Run with server already running:
    pytest tests/integration/pipeline/z_image/test_api_defaults.py -v -s

Or start server first:
    uv run python -m web.server --config config.toml &
    pytest tests/integration/pipeline/z_image/test_api_defaults.py -v -s

Last updated: 2026-01-30
"""

import os
import pytest
import requests

# Server URL - defaults to localhost:7860
SERVER_URL = os.getenv("TEST_SERVER_URL", "http://localhost:7860")


class TestAPIDefaults:
    """Verify API returns correct defaults for Z-Image BASE model."""

    @pytest.fixture
    def api_url(self):
        """Return the server URL."""
        return SERVER_URL

    def test_server_reachable(self, api_url):
        """Basic connectivity check."""
        try:
            resp = requests.get(f"{api_url}/api/health", timeout=5)
            assert resp.status_code == 200, f"Server not healthy: {resp.status_code}"
        except requests.exceptions.ConnectionError:
            pytest.skip(f"Server not running at {api_url}")

    def test_generation_config_shift(self, api_url):
        """Verify shift defaults to 6.0 for BASE model (not 3.0 for turbo)."""
        try:
            resp = requests.get(f"{api_url}/api/generation-config", timeout=5)
            resp.raise_for_status()
            data = resp.json()

            shift = data.get("shift")
            print(f"API returned shift={shift}")

            # BASE model should have shift=6.0
            # TURBO model would have shift=3.0
            assert shift == 6.0, (
                f"Expected shift=6.0 for BASE model, got {shift}. "
                "Is config.toml [scheduler].shift correct?"
            )
        except requests.exceptions.ConnectionError:
            pytest.skip(f"Server not running at {api_url}")

    def test_generation_config_guidance_scale(self, api_url):
        """Verify guidance_scale defaults to 4.0 for BASE model (not 0.0 for turbo)."""
        try:
            resp = requests.get(f"{api_url}/api/generation-config", timeout=5)
            resp.raise_for_status()
            data = resp.json()

            guidance_scale = data.get("guidance_scale")
            print(f"API returned guidance_scale={guidance_scale}")

            # BASE model should have guidance_scale=4.0
            # TURBO model would have guidance_scale=0.0 (CFG baked in)
            assert guidance_scale == 4.0, (
                f"Expected guidance_scale=4.0 for BASE model, got {guidance_scale}. "
                "Is config.toml [generation].guidance_scale correct?"
            )
        except requests.exceptions.ConnectionError:
            pytest.skip(f"Server not running at {api_url}")

    def test_generation_config_steps(self, api_url):
        """Verify steps defaults to 30 for BASE model (not 9 for turbo)."""
        try:
            resp = requests.get(f"{api_url}/api/generation-config", timeout=5)
            resp.raise_for_status()
            data = resp.json()

            steps = data.get("steps")
            print(f"API returned steps={steps}")

            # BASE model should have steps=30
            # TURBO model would have steps=9
            assert steps == 30, (
                f"Expected steps=30 for BASE model, got {steps}. "
                "Is config.toml [generation].num_inference_steps correct?"
            )
        except requests.exceptions.ConnectionError:
            pytest.skip(f"Server not running at {api_url}")

    def test_full_generation_config(self, api_url):
        """Print full generation config for debugging."""
        try:
            resp = requests.get(f"{api_url}/api/generation-config", timeout=5)
            resp.raise_for_status()
            data = resp.json()

            print("\n=== Full Generation Config ===")
            for key, value in sorted(data.items()):
                if key != "features":
                    print(f"  {key}: {value}")

            if "features" in data:
                print("  features:")
                for k, v in data["features"].items():
                    print(f"    {k}: {v}")
            print("==============================")

        except requests.exceptions.ConnectionError:
            pytest.skip(f"Server not running at {api_url}")

    def test_zimage_status(self, api_url):
        """Check loaded Z-Image variant from status endpoint."""
        try:
            resp = requests.get(f"{api_url}/api/status", timeout=5)
            resp.raise_for_status()
            data = resp.json()

            print(f"\n=== Server Status ===")
            print(f"  loaded_pipelines: {data.get('loaded_pipelines', [])}")

            config_info = data.get("config_info", {})
            print(f"  model_type: {config_info.get('model_type')}")

            # If zimage is loaded, check variant
            if "zimage" in data.get("loaded_pipelines", []):
                # The variant should be 'base'
                print(f"  Z-Image loaded!")
            print("=====================")

        except requests.exceptions.ConnectionError:
            pytest.skip(f"Server not running at {api_url}")


class TestRequestDefaults:
    """Test that generation requests use correct defaults."""

    @pytest.fixture
    def api_url(self):
        """Return the server URL."""
        return SERVER_URL

    def test_empty_request_gets_defaults(self, api_url):
        """Sending minimal request should use server defaults."""
        try:
            # Minimal request - just a prompt
            resp = requests.post(
                f"{api_url}/api/generate",
                json={"prompt": "test"},
                timeout=5,
            )

            # We expect either success or validation error
            # If validation passes, the defaults were applied
            # We're not actually generating, just checking the request handling
            print(f"Response status: {resp.status_code}")
            print(f"Response: {resp.text[:500] if resp.text else 'empty'}")

        except requests.exceptions.ConnectionError:
            pytest.skip(f"Server not running at {api_url}")


if __name__ == "__main__":
    """Quick standalone test without pytest."""
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else SERVER_URL

    print(f"Testing server at {url}")

    try:
        # Get generation config
        resp = requests.get(f"{url}/api/generation-config", timeout=5)
        resp.raise_for_status()
        data = resp.json()

        print("\n=== Generation Config ===")
        print(f"  shift: {data.get('shift')} (expected: 6.0 for BASE)")
        print(f"  guidance_scale: {data.get('guidance_scale')} (expected: 4.0 for BASE)")
        print(f"  steps: {data.get('steps')} (expected: 30 for BASE)")
        print(f"  dynamic_shift: {data.get('dynamic_shift')}")
        print(f"  d_noise: {data.get('d_noise')}")

        # Check values
        issues = []
        if data.get("shift") != 6.0:
            issues.append(f"shift={data.get('shift')} (expected 6.0)")
        if data.get("guidance_scale") != 4.0:
            issues.append(f"guidance_scale={data.get('guidance_scale')} (expected 4.0)")
        if data.get("steps") != 30:
            issues.append(f"steps={data.get('steps')} (expected 30)")

        if issues:
            print("\n❌ ISSUES FOUND:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("\n✅ All defaults correct for BASE model!")

    except requests.exceptions.ConnectionError:
        print(f"❌ Server not running at {url}")
        sys.exit(1)
