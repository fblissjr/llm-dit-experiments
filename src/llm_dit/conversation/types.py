"""
Conversation types for structured prompt building.

These types mirror the conversation handling in ComfyUI ZImageTurnBuilder,
allowing multi-turn conversations with thinking blocks.

Based on: ComfyUI-QwenImageWanBridge/nodes/z_image_encoder.py format_conversation()
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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

    Supports multi-turn conversations with thinking blocks, matching
    the ComfyUI ZImageTurnBuilder behavior.

    Attributes:
        messages: List of messages in the conversation
        enable_thinking: Whether to include <think></think> blocks
        is_final: If True, last assistant message omits closing <|im_end|>
                 (model is "generating"). If False, message is complete.

    Example:
        conv = Conversation()
        conv.add_system("You are a helpful assistant.")
        conv.add_user("Generate a cat image.")
        conv.add_assistant(thinking="I should create a fluffy cat...")

        formatted = Qwen3Formatter().format(conv)
    """

    messages: list[Message] = field(default_factory=list)
    enable_thinking: bool = True
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
        enable_thinking: bool = True,
    ) -> "Conversation":
        """
        Create a simple single-turn conversation.

        This is the most common pattern for Z-Image generation.

        Args:
            user_prompt: The user's generation request
            system_prompt: Optional system prompt (template)
            thinking_content: Content inside <think></think>
            assistant_content: Content after </think>
            enable_thinking: Whether to include thinking block

        Returns:
            Conversation ready for formatting
        """
        conv = cls(enable_thinking=enable_thinking)
        if system_prompt:
            conv.add_system(system_prompt)
        conv.add_user(user_prompt)
        conv.add_assistant(assistant_content, thinking_content if enable_thinking else None)
        return conv
