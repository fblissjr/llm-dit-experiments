"""
Conversation management for LLM-DiT encoding.

Provides:
- Message and Conversation dataclasses for structured prompts
- Qwen3Formatter for chat template formatting
"""

from llm_dit.conversation.types import Message, Conversation, Role
from llm_dit.conversation.formatter import Qwen3Formatter

__all__ = ["Message", "Conversation", "Role", "Qwen3Formatter"]
