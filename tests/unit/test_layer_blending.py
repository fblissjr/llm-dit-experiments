"""
Unit tests for multi-layer blending functionality.

Tests the layer_weights parsing, CLI integration, and config handling.
These tests run on any platform without GPU or model files.
"""

import pytest

from llm_dit.cli import parse_layer_weights, RuntimeConfig

pytestmark = pytest.mark.unit


class TestParseLayerWeights:
    """Test parse_layer_weights function."""

    def test_two_layer_blend(self):
        """Test parsing two-layer blend string."""
        result = parse_layer_weights("-2:0.7,-6:0.3")
        assert result == {-2: 0.7, -6: 0.3}

    def test_three_layer_blend(self):
        """Test parsing three-layer blend string."""
        result = parse_layer_weights("-1:0.33,-2:0.34,-3:0.33")
        assert result == {-1: 0.33, -2: 0.34, -3: 0.33}

    def test_equal_weights(self):
        """Test parsing equal weight blend."""
        result = parse_layer_weights("-2:0.5,-5:0.5")
        assert result == {-2: 0.5, -5: 0.5}

    def test_whitespace_handling(self):
        """Test that whitespace is handled correctly."""
        result = parse_layer_weights("-2:0.7, -6:0.3")
        assert result == {-2: 0.7, -6: 0.3}

    def test_single_layer(self):
        """Test parsing a single layer (degenerate case)."""
        result = parse_layer_weights("-2:1.0")
        assert result == {-2: 1.0}

    def test_deep_layers(self):
        """Test parsing deep (middle) layers."""
        result = parse_layer_weights("-19:0.5,-2:0.5")
        assert result == {-19: 0.5, -2: 0.5}

    def test_invalid_format_no_colon(self):
        """Test that missing colon raises ValueError."""
        with pytest.raises(ValueError, match="Invalid layer weight format"):
            parse_layer_weights("-2,0.7")

    def test_invalid_layer_not_int(self):
        """Test that non-integer layer raises ValueError."""
        with pytest.raises(ValueError, match="Layer must be int"):
            parse_layer_weights("abc:0.7")

    def test_invalid_weight_not_float(self):
        """Test that non-float weight raises ValueError."""
        with pytest.raises(ValueError, match="weight must be float"):
            parse_layer_weights("-2:abc")


class TestRuntimeConfigLayerWeights:
    """Test RuntimeConfig layer_weights field."""

    def test_default_layer_weights_is_none(self):
        """Test that layer_weights defaults to None."""
        config = RuntimeConfig()
        assert config.layer_weights is None

    def test_layer_weights_can_be_set(self):
        """Test that layer_weights can be set to a dict."""
        config = RuntimeConfig()
        config.layer_weights = {-2: 0.7, -6: 0.3}
        assert config.layer_weights == {-2: 0.7, -6: 0.3}

    def test_hidden_layer_default(self):
        """Test that hidden_layer has correct default."""
        config = RuntimeConfig()
        assert config.hidden_layer == -2


class TestLayerWeightsIntegration:
    """Integration tests for layer_weights in the pipeline context."""

    def test_layer_weights_overrides_hidden_layer_semantics(self):
        """Test that when layer_weights is set, it should override hidden_layer."""
        # This is a semantic test - in actual usage, if layer_weights is provided,
        # the pipeline will use encode_blended() instead of encode()
        config = RuntimeConfig()
        config.hidden_layer = -2
        config.layer_weights = {-2: 0.7, -6: 0.3}

        # Both should be set - the pipeline chooses which to use
        assert config.hidden_layer == -2
        assert config.layer_weights == {-2: 0.7, -6: 0.3}

    def test_weights_normalization_examples(self):
        """Test that weights don't need to sum to 1.0 (backend normalizes)."""
        # The backend normalizes weights, so we can test various sum values
        result = parse_layer_weights("-2:7,-6:3")  # Sums to 10
        assert result == {-2: 7.0, -6: 3.0}

        result = parse_layer_weights("-2:1,-6:1")  # Sums to 2
        assert result == {-2: 1.0, -6: 1.0}
