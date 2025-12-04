"""
Unit tests for conversation types and Qwen3 chat template formatting.

Tests the Qwen3Formatter against the exact patterns documented in
ComfyUI-QwenImageWanBridge/nodes/docs/z_image_encoder.md.

These tests run on any platform without GPU or model files.
"""

import pytest

from llm_dit.conversation import Conversation, Message, Role
from llm_dit.conversation.formatter import Qwen3Formatter, format_prompt

pytestmark = pytest.mark.unit


class TestRole:
    """Test Role enum."""

    def test_role_values(self):
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"


class TestMessage:
    """Test Message dataclass."""

    def test_create_system_message(self):
        msg = Message.system("You are helpful")
        assert msg.role == Role.SYSTEM
        assert msg.content == "You are helpful"
        assert msg.thinking is None

    def test_create_user_message(self):
        msg = Message.user("A cat sleeping")
        assert msg.role == Role.USER
        assert msg.content == "A cat sleeping"

    def test_create_assistant_message(self):
        msg = Message.assistant("Here is your image")
        assert msg.role == Role.ASSISTANT
        assert msg.content == "Here is your image"
        assert msg.thinking is None

    def test_create_assistant_with_thinking(self):
        msg = Message.assistant("Here it is", thinking="Let me think...")
        assert msg.role == Role.ASSISTANT
        assert msg.content == "Here it is"
        assert msg.thinking == "Let me think..."

    def test_serialization_roundtrip(self):
        msg = Message.assistant("Hello", thinking="Thinking...")
        data = msg.to_dict()
        restored = Message.from_dict(data)

        assert restored.role == msg.role
        assert restored.content == msg.content
        assert restored.thinking == msg.thinking


class TestConversation:
    """Test Conversation dataclass."""

    def test_simple_user_only(self):
        conv = Conversation.simple("A cat")
        # Should have user + assistant messages
        assert len(conv.messages) == 2
        assert conv.messages[0].role == Role.USER
        assert conv.messages[1].role == Role.ASSISTANT

    def test_simple_with_system(self):
        conv = Conversation.simple("A cat", system_prompt="You are a painter")
        assert len(conv.messages) == 3
        assert conv.messages[0].role == Role.SYSTEM
        assert conv.messages[0].content == "You are a painter"

    def test_simple_with_thinking(self):
        conv = Conversation.simple(
            "A cat", thinking_content="Orange fur", enable_thinking=True
        )
        assert conv.enable_thinking is True
        assert conv.messages[-1].thinking == "Orange fur"

    def test_simple_with_assistant_content(self):
        conv = Conversation.simple("A cat", assistant_content="Here is your cat:")
        assert conv.messages[-1].content == "Here is your cat:"

    def test_copy_is_independent(self):
        conv = Conversation.simple("test")
        copy = conv.copy()
        copy.add_user("extra message")

        # Original should be unchanged
        assert len(conv.messages) != len(copy.messages)

    def test_add_messages(self):
        conv = Conversation()
        conv.add_system("System prompt")
        conv.add_user("User message")
        conv.add_assistant("Response", thinking="Thinking...")

        assert len(conv.messages) == 3
        assert conv.messages[2].thinking == "Thinking..."


class TestQwen3Formatter:
    """Test Qwen3Formatter against ComfyUI reference patterns."""

    @pytest.fixture
    def formatter(self):
        return Qwen3Formatter()

    # Test cases matching z_image_encoder.md "Formatted Prompt Examples"

    def test_minimal_no_think_block(self, formatter):
        """Test 1: Minimal (matches diffusers default)"""
        conv = Conversation.simple("A cat sleeping", enable_thinking=False)
        formatted = formatter.format(conv)

        expected = (
            "<|im_start|>user\n"
            "A cat sleeping<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        assert formatted == expected

    def test_with_system_prompt(self, formatter):
        """Test 2: With System Prompt"""
        conv = Conversation.simple(
            "A cat sleeping",
            system_prompt="Generate a photorealistic image.",
            enable_thinking=False,
        )
        formatted = formatter.format(conv)

        expected = (
            "<|im_start|>system\n"
            "Generate a photorealistic image.<|im_end|>\n"
            "<|im_start|>user\n"
            "A cat sleeping<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        assert formatted == expected

    def test_with_think_block_empty(self, formatter):
        """Test 3: With Think Block (empty)"""
        conv = Conversation.simple("A cat sleeping", enable_thinking=True)
        formatted = formatter.format(conv)

        expected = (
            "<|im_start|>user\n"
            "A cat sleeping<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n"
            "\n"
            "</think>\n"
            "\n"
        )
        assert formatted == expected

    def test_with_think_block_and_content(self, formatter):
        """Test 4: With Think Block + Thinking Content"""
        conv = Conversation.simple(
            "A cat sleeping",
            thinking_content="Soft lighting, peaceful mood, curled up position.",
            enable_thinking=True,
        )
        formatted = formatter.format(conv)

        expected = (
            "<|im_start|>user\n"
            "A cat sleeping<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n"
            "Soft lighting, peaceful mood, curled up position.\n"
            "</think>\n"
            "\n"
        )
        assert formatted == expected

    def test_with_think_and_assistant_content(self, formatter):
        """Test 5: With Think Block + Thinking + Assistant Content"""
        conv = Conversation.simple(
            "A cat sleeping",
            thinking_content="Soft lighting, peaceful mood.",
            assistant_content="Creating a cozy scene...",
            enable_thinking=True,
        )
        formatted = formatter.format(conv)

        expected = (
            "<|im_start|>user\n"
            "A cat sleeping<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n"
            "Soft lighting, peaceful mood.\n"
            "</think>\n"
            "\n"
            "Creating a cozy scene...<|im_end|>"
        )
        assert formatted == expected

    def test_full_example(self, formatter):
        """Test 6: Full Example (System + Think + Assistant)"""
        conv = Conversation.simple(
            "A cat sleeping on a windowsill",
            system_prompt="You are an expert photographer.",
            thinking_content="Golden hour light, shallow depth of field.",
            assistant_content="Capturing the peaceful moment...",
            enable_thinking=True,
        )
        formatted = formatter.format(conv)

        expected = (
            "<|im_start|>system\n"
            "You are an expert photographer.<|im_end|>\n"
            "<|im_start|>user\n"
            "A cat sleeping on a windowsill<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n"
            "Golden hour light, shallow depth of field.\n"
            "</think>\n"
            "\n"
            "Capturing the peaceful moment...<|im_end|>"
        )
        assert formatted == expected

    def test_is_final_omits_end_token(self, formatter):
        """Test that is_final=True omits closing token when assistant is empty."""
        conv = Conversation.simple("A cat", enable_thinking=True)
        conv.is_final = True
        formatted = formatter.format(conv)

        # Should NOT end with <|im_end|> when assistant content is empty
        assert not formatted.rstrip().endswith("<|im_end|>")

    def test_is_final_with_content_keeps_end_token(self, formatter):
        """Test that is_final=True keeps closing token when assistant has content."""
        conv = Conversation.simple(
            "A cat", assistant_content="Here it is:", enable_thinking=True
        )
        conv.is_final = True
        formatted = formatter.format(conv)

        # Should end with <|im_end|> when assistant content is provided
        assert formatted.rstrip().endswith("<|im_end|>")


class TestFormatPromptHelper:
    """Test format_prompt convenience function."""

    def test_basic_usage(self):
        formatted = format_prompt("A cat")
        assert "<|im_start|>user" in formatted
        assert "A cat" in formatted

    def test_with_all_parameters(self):
        formatted = format_prompt(
            user_prompt="A cat",
            system_prompt="You are helpful",
            thinking_content="Fluffy cat",
            assistant_content="Here it is",
            enable_thinking=True,
            is_final=True,
        )

        assert "<|im_start|>system" in formatted
        assert "<think>" in formatted
        assert "Fluffy cat" in formatted
        assert "Here it is" in formatted
        assert formatted.endswith("<|im_end|>")

    def test_matches_formatter_output(self):
        """format_prompt should produce same output as Qwen3Formatter."""
        formatter = Qwen3Formatter()

        formatted1 = format_prompt(
            "A cat", system_prompt="Test", thinking_content="Think", enable_thinking=True
        )

        conv = Conversation.simple(
            "A cat", system_prompt="Test", thinking_content="Think", enable_thinking=True
        )
        formatted2 = formatter.format(conv)

        assert formatted1 == formatted2
