"""
Conversation types for structured prompt building.

These types support the Qwen3-4B chat template format used by Z-Image,
with content-driven thinking block behavior.

Format Reference (Official HuggingFace Space):
- Default (no thinking): <|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n
- With thinking: <|im_start|>assistant\\n<think>\\n{content}\\n</think>\\n\\n
- Empty think block: <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n

The thinking block is content-driven:
- If thinking_content is provided -> add think block with content
- If force_think_block=True -> add empty think block
- Otherwise -> no think block (matches official HF Space default)
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def clean_content(text: str) -> str:
    """
    Clean whitespace and newlines from user-provided content.

    - Strips leading/trailing whitespace
    - Normalizes multiple newlines to single newlines
    - Normalizes multiple spaces to single spaces
    """
    if not text:
        return text
    # Strip leading/trailing whitespace
    text = text.strip()
    # Normalize multiple newlines to single
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalize multiple spaces (but not newlines) to single
    text = re.sub(r"[^\S\n]+", " ", text)
    return text


def strip_quotes(text: str) -> str:
    """
    Strip double quotes from JSON-type prompts.

    Z-Image training data treats " characters as text to render in images.
    This helper removes them for prompts that may come from JSON sources.
    """
    if not text:
        return text
    # Remove all double quotes
    return text.replace('"', "")


class Role(Enum):
    """Message roles in Qwen3 chat template."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """
    A single message in a conversation.

    Attributes:
        role: The speaker (system, user, assistant)
        content: The message content
        thinking: Optional thinking content for assistant messages.
                 Only used when role=ASSISTANT and conversation.enable_thinking=True.
    """

    role: Role
    content: str
    thinking: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        result = {"role": self.role.value, "content": self.content}
        if self.thinking is not None:
            result["thinking"] = self.thinking
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Deserialize from dict."""
        return cls(
            role=Role(data["role"]),
            content=data["content"],
            thinking=data.get("thinking"),
        )

    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(Role.SYSTEM, content)

    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(Role.USER, content)

    @classmethod
    def assistant(cls, content: str = "", thinking: str | None = None) -> "Message":
        """Create an assistant message with optional thinking."""
        return cls(Role.ASSISTANT, content, thinking)


@dataclass
class Conversation:
    """
    A conversation for Z-Image encoding.

    Supports multi-turn conversations with optional thinking blocks.
    By default, no thinking block is added (matching official HF Space).

    Attributes:
        messages: List of messages in the conversation
        enable_thinking: Whether to include <think></think> blocks.
                        Default is False to match official HF Space.
        is_final: If True, last assistant message omits closing <|im_end|>
                 (model is "generating"). If False, message is complete.

    Example (default - matches official):
        conv = Conversation.simple("A cat sleeping")
        # -> <|im_start|>user\\nA cat sleeping<|im_end|>\\n<|im_start|>assistant\\n

    Example (with thinking content):
        conv = Conversation.simple("A cat", thinking_content="Orange fur")
        # -> ...assistant\\n<think>\\nOrange fur\\n</think>\\n\\n

    Example (force empty think block):
        conv = Conversation.simple("A cat", force_think_block=True)
        # -> ...assistant\\n<think>\\n\\n</think>\\n\\n
    """

    messages: list[Message] = field(default_factory=list)
    enable_thinking: bool = False  # Default False to match official HF Space
    is_final: bool = True  # Last message omits <|im_end|> if True

    def add_system(self, content: str) -> "Conversation":
        """Add a system message. Should be first if used."""
        self.messages.append(Message.system(content))
        return self

    def add_user(self, content: str) -> "Conversation":
        """Add a user message."""
        self.messages.append(Message.user(content))
        return self

    def add_assistant(
        self, content: str = "", thinking: str | None = None
    ) -> "Conversation":
        """
        Add an assistant message with optional thinking.

        Args:
            content: The assistant's response (after thinking)
            thinking: Content inside <think></think> block.
                     Only used if enable_thinking=True.

        Returns:
            Self for chaining
        """
        self.messages.append(Message.assistant(content, thinking))
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (JSON-compatible)."""
        return {
            "enable_thinking": self.enable_thinking,
            "is_final": self.is_final,
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Conversation":
        """Deserialize from dict."""
        conv = cls(
            enable_thinking=data.get("enable_thinking", True),
            is_final=data.get("is_final", True),
        )
        for msg_data in data.get("messages", []):
            conv.messages.append(Message.from_dict(msg_data))
        return conv

    def copy(self) -> "Conversation":
        """Create a deep copy of the conversation."""
        return Conversation.from_dict(self.to_dict())

    @classmethod
    def simple(
        cls,
        user_prompt: str,
        system_prompt: str = "",
        thinking_content: str = "",
        assistant_content: str = "",
        force_think_block: bool = False,
        clean_whitespace: bool = True,
        remove_quotes: bool = False,
    ) -> "Conversation":
        """
        Create a simple single-turn conversation.

        This is the most common pattern for Z-Image generation.
        Uses content-driven logic for thinking blocks:
        - If thinking_content is provided -> add think block with content
        - If force_think_block=True -> add empty think block
        - Otherwise -> no think block (matches official HF Space)

        Args:
            user_prompt: The user's generation request
            system_prompt: Optional system prompt (omitted if empty)
            thinking_content: Content inside <think></think> (triggers think block)
            assistant_content: Content after </think>
            force_think_block: If True, add empty think block even without content
            clean_whitespace: If True (default), clean extra whitespace/newlines
            remove_quotes: If True, strip " characters (for JSON-type prompts)

        Returns:
            Conversation ready for formatting

        Examples:
            # Match official HF Space (no think block):
            conv = Conversation.simple("A cat sleeping")

            # With thinking content (automatically adds think block):
            conv = Conversation.simple("A cat", thinking_content="Orange fur")

            # Force empty think block:
            conv = Conversation.simple("A cat", force_think_block=True)

            # Full control:
            conv = Conversation.simple(
                "A cat",
                system_prompt="You are a photographer.",
                thinking_content="Soft lighting.",
                assistant_content="Here is your image:",
            )

            # JSON prompt with quotes stripped:
            conv = Conversation.simple('"A cat sleeping"', remove_quotes=True)
        """
        # Apply content cleaning
        if clean_whitespace:
            user_prompt = clean_content(user_prompt)
            system_prompt = clean_content(system_prompt)
            thinking_content = clean_content(thinking_content)
            assistant_content = clean_content(assistant_content)

        # Strip quotes if requested (for JSON-type prompts)
        if remove_quotes:
            user_prompt = strip_quotes(user_prompt)
            system_prompt = strip_quotes(system_prompt)
            thinking_content = strip_quotes(thinking_content)
            assistant_content = strip_quotes(assistant_content)

        # Content-driven: enable thinking if content provided OR forced
        has_thinking = bool(thinking_content) or force_think_block
        conv = cls(enable_thinking=has_thinking)

        if system_prompt:
            conv.add_system(system_prompt)
        conv.add_user(user_prompt)
        conv.add_assistant(assistant_content, thinking_content if has_thinking else None)
        return conv
