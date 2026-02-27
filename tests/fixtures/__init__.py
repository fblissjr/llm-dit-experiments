"""
Test fixtures for llm-dit-experiments.

Last Updated: 2026-02-01

Provides shared test data:
- prompts/: Test prompts organized by model/pipeline
- configs/: Test configuration presets (uses same system as production)
- (future: images/, videos/ for I2V testing)
"""

from pathlib import Path

FIXTURES_DIR = Path(__file__).parent
