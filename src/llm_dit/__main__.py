"""
Entry point for running llm_dit as a module.

Last Updated: 2026-01-09

This allows running the package via:
    python -m llm_dit [args]

Which is equivalent to:
    python web/server.py [args]

The web server is the main entry point for llm_dit, providing both the
web UI and CLI functionality for all model types (Z-Image, Qwen-Image, LTX-2).
"""

import sys
from pathlib import Path

# Add web directory to path so we can import server
_project_root = Path(__file__).parent.parent.parent
_web_dir = _project_root / "web"
if str(_web_dir) not in sys.path:
    sys.path.insert(0, str(_web_dir))


def main():
    """Run the llm_dit web server / CLI."""
    # Import here to avoid circular imports and ensure path is set
    from server import main as server_main
    return server_main()


if __name__ == "__main__":
    sys.exit(main() or 0)
