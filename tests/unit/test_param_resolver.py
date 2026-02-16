"""
Unit tests for web/param_resolver.py -- unified parameter resolution.

Tests the precedence rule: client-sent > config.toml > schema default.

Run with: uv run pytest tests/unit/test_param_resolver.py -v
"""

from typing import Optional

import pytest
from pydantic import BaseModel

from web.param_resolver import csv_to_int_list, resolve_param

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Test fixtures: minimal Pydantic models that mimic real request schemas
# ---------------------------------------------------------------------------


class FakeRequest(BaseModel):
    """Mimics a generation request with various field types."""

    steps: int = 40
    guidance_scale: float = 3.0
    stg_scale: float = 1.0
    negative_prompt: str = "bad quality"
    stage1_steps: Optional[int] = None
    stg_blocks: Optional[list[int]] = None
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# resolve_param tests
# ---------------------------------------------------------------------------


class TestResolveParam:
    """Tests for the resolve_param() function."""

    def test_client_explicit_wins_over_config(self):
        """When client sends a value, it always wins over config.toml."""
        request = FakeRequest.model_validate({"steps": 20})
        result = resolve_param(request, "steps", config_value=50)
        assert result == 20

    def test_config_wins_over_schema_default(self):
        """When client omits a field, config.toml value is used."""
        request = FakeRequest.model_validate({})
        result = resolve_param(request, "steps", config_value=50)
        assert result == 50

    def test_falsy_zero_int_preserved(self):
        """Client sending 0 must NOT fall through to config."""
        request = FakeRequest.model_validate({"steps": 0})
        result = resolve_param(request, "steps", config_value=50)
        assert result == 0

    def test_falsy_zero_float_preserved(self):
        """Client sending 0.0 must NOT fall through to config."""
        request = FakeRequest.model_validate({"stg_scale": 0.0})
        result = resolve_param(request, "stg_scale", config_value=1.0)
        assert result == 0.0

    def test_falsy_empty_string_preserved(self):
        """Client sending "" must NOT fall through to config."""
        request = FakeRequest.model_validate({"negative_prompt": ""})
        result = resolve_param(request, "negative_prompt", config_value="default neg")
        assert result == ""

    def test_skip_none_true_falls_through(self):
        """With skip_none=True, client sending None uses config value."""
        request = FakeRequest.model_validate({"stage1_steps": None})
        result = resolve_param(request, "stage1_steps", config_value=40, skip_none=True)
        assert result == 40

    def test_skip_none_false_keeps_none(self):
        """With skip_none=False (default), client sending None is preserved."""
        request = FakeRequest.model_validate({"seed": None})
        result = resolve_param(request, "seed", config_value=42)
        assert result is None

    def test_optional_field_omitted_uses_config(self):
        """Optional field not in request uses config value."""
        request = FakeRequest.model_validate({})
        result = resolve_param(request, "stage1_steps", config_value=40, skip_none=True)
        assert result == 40

    def test_config_none_graceful(self):
        """When config value is None and client omitted field, returns None."""
        request = FakeRequest.model_validate({})
        result = resolve_param(request, "stage1_steps", config_value=None)
        assert result is None

    def test_list_field_client_wins(self):
        """Client sending a list wins over config."""
        request = FakeRequest.model_validate({"stg_blocks": [10, 20]})
        result = resolve_param(request, "stg_blocks", config_value=[29], skip_none=True)
        assert result == [10, 20]

    def test_list_field_omitted_uses_config(self):
        """Omitted list field falls through to config."""
        request = FakeRequest.model_validate({})
        result = resolve_param(request, "stg_blocks", config_value=[29, 30], skip_none=True)
        assert result == [29, 30]

    def test_multiple_fields_independent(self):
        """Each field resolves independently -- some from client, some from config."""
        request = FakeRequest.model_validate({"steps": 20, "guidance_scale": 5.0})
        assert resolve_param(request, "steps", config_value=50) == 20
        assert resolve_param(request, "guidance_scale", config_value=3.5) == 5.0
        assert resolve_param(request, "stg_scale", config_value=0.5) == 0.5  # from config


# ---------------------------------------------------------------------------
# csv_to_int_list tests
# ---------------------------------------------------------------------------


class TestCsvToIntList:
    """Tests for csv_to_int_list() converter."""

    def test_single_value(self):
        assert csv_to_int_list("29") == [29]

    def test_multiple_values(self):
        assert csv_to_int_list("29,30") == [29, 30]

    def test_whitespace_handling(self):
        assert csv_to_int_list(" 29 , 30 , 31 ") == [29, 30, 31]

    def test_empty_string(self):
        assert csv_to_int_list("") == []

    def test_whitespace_only(self):
        assert csv_to_int_list("   ") == []

    def test_single_value_with_whitespace(self):
        assert csv_to_int_list("  29  ") == [29]
