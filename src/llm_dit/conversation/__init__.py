"""
Conversation management for LLM-DiT encoding.

Provides:
- Message and Conversation dataclasses for structured prompts
- Qwen3Formatter for chat template formatting
- format_prompt() convenience function
- clean_content() and strip_quotes() helpers
"""

from llm_dit.conversation.types import (
    Message,
    Conversation,
    Role,
    clean_content,
    strip_quotes,
)
from llm_dit.conversation.formatter import Qwen3Formatter, format_prompt


__all__ = [
    "Message",
    "Conversation",
    "Role",
    "Qwen3Formatter",
    "format_prompt",
    "clean_content",
    "strip_quotes",
]
