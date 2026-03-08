"""
Unit tests for _resolve_quantize() helper in LTX-2 generation.

Last Updated: 2026-02-15

The _resolve_quantize() helper normalizes quantize string shorthands
(e.g., "fp8" -> "fp8-weight-only") and returns a (should_quantize, precision)
tuple used by both generate_video_with_offloading() and generate_video_two_stage().

Run with: uv run pytest tests/unit/test_ltx2_resolve_quantize.py -v
"""

import pytest

from llm_dit.pipelines.generate import _resolve_quantize
from llm_dit.quantization import QUANT_ALIASES


class TestResolveQuantize:
    """Tests for the _resolve_quantize() normalization helper."""

    @pytest.mark.parametrize("value", [None, "", "none"])
    def test_none_variations_return_false(self, value):
        """None, empty string, and 'none' all disable quantization."""
        should_quantize, precision = _resolve_quantize(value)
        assert should_quantize is False
        assert precision == "none"

    def test_fp8_alias_expands_to_dynamic(self):
        """'fp8' shorthand expands to 'fp8-dynamic'."""
        should_quantize, precision = _resolve_quantize("fp8")
        assert should_quantize is True
        assert precision == "fp8-dynamic"

    @pytest.mark.parametrize("method", ["fp8-dynamic", "fp8-weight-only", "int8", "int4"])
    def test_known_methods_pass_through(self, method):
        """Known full method names pass through unchanged."""
        should_quantize, precision = _resolve_quantize(method)
        assert should_quantize is True
        assert precision == method

    def test_unknown_method_passes_through(self):
        """Unknown methods pass through (validation happens downstream)."""
        should_quantize, precision = _resolve_quantize("future-method-v3")
        assert should_quantize is True
        assert precision == "future-method-v3"

    def test_quant_aliases_contains_fp8(self):
        """QUANT_ALIASES maps 'fp8' to 'fp8-dynamic'."""
        assert "fp8" in QUANT_ALIASES
        assert QUANT_ALIASES["fp8"] == "fp8-dynamic"

    def test_gguf_not_in_quant_aliases(self):
        """GGUF support was removed -- verify it's not in aliases."""
        assert "gguf" not in QUANT_ALIASES
