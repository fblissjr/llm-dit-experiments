"""
Unit tests for prompt rewriter utilities.

Tests language detection, response cleaning, and rewriter configuration.
"""

import pytest

pytestmark = pytest.mark.unit


class TestLanguageDetection:
    """Test automatic language detection."""

    def test_import(self):
        from llm_dit.utils.prompt_rewriter import detect_language
        assert detect_language is not None

    def test_english_detection(self):
        from llm_dit.utils.prompt_rewriter import detect_language

        assert detect_language("A cat sleeping in sunlight") == "en"
        assert detect_language("beautiful sunset over mountains") == "en"
        assert detect_language("portrait of a young woman") == "en"

    def test_chinese_detection(self):
        from llm_dit.utils.prompt_rewriter import detect_language

        assert detect_language("一只猫在阳光下睡觉") == "zh"
        assert detect_language("美丽的山间日落") == "zh"
        assert detect_language("年轻女性的肖像") == "zh"

    def test_mixed_defaults_to_chinese(self):
        """Mixed text with Chinese characters should detect as Chinese."""
        from llm_dit.utils.prompt_rewriter import detect_language

        # Mixed with Chinese character triggers zh
        assert detect_language("A cat 猫 sleeping") == "zh"

    def test_empty_string(self):
        from llm_dit.utils.prompt_rewriter import detect_language

        assert detect_language("") == "en"

    def test_numbers_only(self):
        from llm_dit.utils.prompt_rewriter import detect_language

        assert detect_language("12345") == "en"


class TestSystemPrompts:
    """Test system prompt constants."""

    def test_english_prompt_exists(self):
        from llm_dit.utils.prompt_rewriter import ENGLISH_SYSTEM_PROMPT

        assert ENGLISH_SYSTEM_PROMPT is not None
        assert len(ENGLISH_SYSTEM_PROMPT) > 1000  # Should be substantial
        assert "Image Prompt Rewriting Expert" in ENGLISH_SYSTEM_PROMPT
        assert "portrait" in ENGLISH_SYSTEM_PROMPT.lower()

    def test_chinese_prompt_exists(self):
        from llm_dit.utils.prompt_rewriter import CHINESE_SYSTEM_PROMPT

        assert CHINESE_SYSTEM_PROMPT is not None
        assert len(CHINESE_SYSTEM_PROMPT) > 500  # Should be substantial
        assert "图像" in CHINESE_SYSTEM_PROMPT

    def test_negative_prompts_exist(self):
        from llm_dit.utils.prompt_rewriter import (
            DEFAULT_NEGATIVE_PROMPT,
            DEFAULT_NEGATIVE_PROMPT_EN,
        )

        assert DEFAULT_NEGATIVE_PROMPT is not None
        assert DEFAULT_NEGATIVE_PROMPT_EN is not None
        # Chinese version
        assert "低分辨率" in DEFAULT_NEGATIVE_PROMPT
        # English version
        assert "Low resolution" in DEFAULT_NEGATIVE_PROMPT_EN


class TestPromptRewriterInit:
    """Test PromptRewriter initialization."""

    def test_init_without_backend(self):
        from llm_dit.utils.prompt_rewriter import PromptRewriter

        rewriter = PromptRewriter()
        assert rewriter.backend is None
        assert rewriter.api_url is None

    def test_init_with_api_url(self):
        from llm_dit.utils.prompt_rewriter import PromptRewriter

        rewriter = PromptRewriter(
            api_url="http://localhost:8080/v1",
            api_model="test-model",
        )
        assert rewriter.api_url == "http://localhost:8080/v1"
        assert rewriter.api_model == "test-model"

    def test_from_api_classmethod(self):
        from llm_dit.utils.prompt_rewriter import PromptRewriter

        rewriter = PromptRewriter.from_api(
            api_url="http://localhost:8080/v1",
            model="custom-model",
            timeout=60.0,
        )
        assert rewriter.api_url == "http://localhost:8080/v1"
        assert rewriter.api_model == "custom-model"
        assert rewriter.timeout == 60.0

    def test_set_api(self):
        from llm_dit.utils.prompt_rewriter import PromptRewriter

        rewriter = PromptRewriter()
        rewriter.set_api("http://example.com/v1", "new-model")

        assert rewriter.api_url == "http://example.com/v1"
        assert rewriter.api_model == "new-model"


class TestPromptRewriterCleanResponse:
    """Test response cleaning logic."""

    def test_strips_whitespace(self):
        from llm_dit.utils.prompt_rewriter import PromptRewriter

        rewriter = PromptRewriter()
        result = rewriter._clean_response("  hello world  ")
        assert result == "hello world"

    def test_removes_newlines(self):
        from llm_dit.utils.prompt_rewriter import PromptRewriter

        rewriter = PromptRewriter()
        result = rewriter._clean_response("hello\nworld\n\ntest")
        assert result == "hello world test"

    def test_collapses_multiple_spaces(self):
        from llm_dit.utils.prompt_rewriter import PromptRewriter

        rewriter = PromptRewriter()
        result = rewriter._clean_response("hello    world")
        assert result == "hello world"


class TestPromptRewriterGetNegativePrompt:
    """Test get_negative_prompt method."""

    def test_default_returns_chinese(self):
        from llm_dit.utils.prompt_rewriter import PromptRewriter

        rewriter = PromptRewriter()
        neg = rewriter.get_negative_prompt()
        assert "低分辨率" in neg

    def test_chinese_explicit(self):
        from llm_dit.utils.prompt_rewriter import PromptRewriter

        rewriter = PromptRewriter()
        neg = rewriter.get_negative_prompt(language="zh")
        assert "低分辨率" in neg

    def test_english_explicit(self):
        from llm_dit.utils.prompt_rewriter import PromptRewriter

        rewriter = PromptRewriter()
        neg = rewriter.get_negative_prompt(language="en")
        assert "Low resolution" in neg


class TestPromptRewriterRewrite:
    """Test rewrite method (without actual API calls)."""

    def test_rewrite_empty_prompt(self):
        """Empty prompt should be returned as-is."""
        from llm_dit.utils.prompt_rewriter import PromptRewriter

        rewriter = PromptRewriter()
        result = rewriter.rewrite("")
        assert result == ""

    def test_rewrite_whitespace_prompt(self):
        """Whitespace-only prompt should be returned as-is."""
        from llm_dit.utils.prompt_rewriter import PromptRewriter

        rewriter = PromptRewriter()
        result = rewriter.rewrite("   ")
        assert result == "   "

    def test_rewrite_no_backend_raises(self):
        """Rewriting without backend should raise RuntimeError."""
        from llm_dit.utils.prompt_rewriter import PromptRewriter

        rewriter = PromptRewriter()

        with pytest.raises(RuntimeError, match="No LLM backend configured"):
            rewriter.rewrite("a cat sleeping")


class TestCreateRewriterFromConfig:
    """Test factory function."""

    def test_create_with_api_url(self):
        from llm_dit.utils.prompt_rewriter import create_rewriter_from_config

        rewriter = create_rewriter_from_config(
            api_url="http://localhost:8080/v1",
            api_model="test-model",
            timeout=30.0,
        )

        assert rewriter.api_url == "http://localhost:8080/v1"
        assert rewriter.api_model == "test-model"
        assert rewriter.timeout == 30.0

    def test_create_without_url(self):
        from llm_dit.utils.prompt_rewriter import create_rewriter_from_config

        rewriter = create_rewriter_from_config()
        assert rewriter.api_url is None


class TestLLMBackendProtocol:
    """Test LLMBackend protocol interface."""

    def test_protocol_exists(self):
        from llm_dit.utils.prompt_rewriter import LLMBackend
        assert LLMBackend is not None

    def test_mock_backend_works(self):
        from llm_dit.utils.prompt_rewriter import PromptRewriter

        class MockBackend:
            def complete(self, prompt, system_prompt=None, max_tokens=1024, temperature=0.7):
                return "A beautiful orange tabby cat lies peacefully in a warm sunbeam."

        rewriter = PromptRewriter(backend=MockBackend())
        result = rewriter.rewrite("cat in sun")

        assert "cat" in result.lower()
        assert "sunbeam" in result.lower()
