"""
Conversation management for LLM-DiT encoding.

Provides:
- Message and Conversation dataclasses for structured prompts
- Qwen3Formatter for chat template formatting
- format_prompt() convenience function
"""

from llm_dit.conversation.types import Message, Conversation, Role
from llm_dit.conversation.formatter import Qwen3Formatter


def format_prompt(
    user_prompt: str,
    system_prompt: str = "",
    thinking_content: str = "",
    assistant_content: str = "",
    enable_thinking: bool = True,
) -> str:
    """
    Format a prompt using Qwen3 chat template.

    Convenience function that creates a Conversation and formats it.

    Args:
        user_prompt: The user's generation request
        system_prompt: Optional system prompt
        thinking_content: Content inside <think></think>
        assistant_content: Content after </think>
        enable_thinking: Whether to include thinking block

    Returns:
        Formatted prompt string ready for tokenization

    Example:
        formatted = format_prompt(
            user_prompt="A cat sleeping",
            system_prompt="Generate photorealistic images.",
            enable_thinking=True,
        )
    """
    conv = Conversation.simple(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        thinking_content=thinking_content,
        assistant_content=assistant_content,
        enable_thinking=enable_thinking,
    )
    formatter = Qwen3Formatter()
    return formatter.format(conv)


__all__ = ["Message", "Conversation", "Role", "Qwen3Formatter", "format_prompt"]
