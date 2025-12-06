"""
Qwen3 chat template formatter.

Formats conversations into the exact token format expected by Qwen3-4B,
matching the official Z-Image HuggingFace Space implementation.

Key format details:
- System: <|im_start|>system\\n{content}<|im_end|>
- User: <|im_start|>user\\n{content}<|im_end|>
- Assistant (no thinking): <|im_start|>assistant\\n
- Assistant (with thinking): <|im_start|>assistant\\n<think>\\n{thinking}\\n</think>\\n\\n{content}
- Final message: Omits <|im_end|> if is_final=True (model is generating)

Content-driven behavior:
- Default: No thinking block (matches official HF Space)
- If thinking_content provided: Add think block with content
- If force_think_block=True: Add empty think block

Reference:
- Official HF Space uses tokenizer.apply_chat_template(enable_thinking=True) -> NO think block
- Qwen3 tokenizer enable_thinking=False adds empty <think>\\n\\n</think>\\n\\n
- We format manually for full control over all four prompt components
"""

from llm_dit.conversation.types import Conversation, Message, Role


class Qwen3Formatter:
    """
    Format conversations for Qwen3-4B chat template.

    This formatter produces the exact token format expected by the Z-Image
    text encoder, including proper handling of thinking blocks.

    Example output (with thinking):
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        Generate a cat image.<|im_end|>
        <|im_start|>assistant
        <think>
        I should create a fluffy cat...
        </think>

        [assistant content here]

    Note: Final assistant message omits <|im_end|> when is_final=True,
    as the model is expected to continue generating.
    """

    # Special tokens
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    THINK_START = "<think>"
    THINK_END = "</think>"

    def format(self, conversation: Conversation) -> str:
        """
        Format a conversation to Qwen3 chat template string.

        Args:
            conversation: The conversation to format

        Returns:
            Formatted string with all chat template tokens
        """
        if not conversation.messages:
            return ""

        parts = []
        num_messages = len(conversation.messages)

        for i, msg in enumerate(conversation.messages):
            is_last = i == num_messages - 1
            skip_end = is_last and conversation.is_final

            formatted = self._format_message(
                msg,
                enable_thinking=conversation.enable_thinking,
                skip_end=skip_end,
            )
            parts.append(formatted)

        return "\n".join(parts)

    def _format_message(
        self,
        msg: Message,
        enable_thinking: bool = True,
        skip_end: bool = False,
    ) -> str:
        """
        Format a single message.

        Args:
            msg: The message to format
            enable_thinking: Whether to include thinking block for assistant
            skip_end: If True, omit closing <|im_end|> (only if content is empty)

        Returns:
            Formatted message string

        Note on <|im_end|> handling:
        - When assistant_content is provided, always add <|im_end|> (complete message)
        - When assistant_content is empty AND is_final=True, omit <|im_end|> (model generating)
        - When assistant_content is empty AND is_final=False, add <|im_end|> (more turns)

        This matches the exact patterns from ComfyUI-QwenImageWanBridge.
        """
        role = msg.role.value
        content = msg.content

        if msg.role == Role.ASSISTANT and enable_thinking:
            # Assistant message with thinking block
            thinking = msg.thinking if msg.thinking else ""
            inner = f"{self.THINK_START}\n{thinking}\n{self.THINK_END}\n\n{content}"

            # Add closing tag if content is provided OR not skip_end
            # Only omit when: content is empty AND skip_end=True
            if content or not skip_end:
                return f"{self.IM_START}{role}\n{inner}{self.IM_END}"
            else:
                return f"{self.IM_START}{role}\n{inner}"
        else:
            # System, user, or assistant without thinking
            # Same logic: close if content or not final
            if content or not skip_end:
                return f"{self.IM_START}{role}\n{content}{self.IM_END}"
            else:
                return f"{self.IM_START}{role}\n{content}"

    def format_simple(
        self,
        user_prompt: str,
        system_prompt: str = "",
        thinking_content: str = "",
        assistant_content: str = "",
        force_think_block: bool = False,
        is_final: bool = True,
        clean_whitespace: bool = True,
        remove_quotes: bool = False,
    ) -> str:
        """
        Format a simple single-turn conversation.

        Convenience method for the most common use case.
        Uses content-driven logic for thinking blocks.

        Args:
            user_prompt: The user's request
            system_prompt: Optional system prompt (omitted if empty)
            thinking_content: Content inside <think></think> (triggers think block)
            assistant_content: Content after </think>
            force_think_block: If True, add empty think block even without content
            is_final: Whether to omit final <|im_end|>
            clean_whitespace: If True (default), clean extra whitespace/newlines
            remove_quotes: If True, strip " characters (for JSON-type prompts)

        Returns:
            Formatted string
        """
        conv = Conversation.simple(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            thinking_content=thinking_content,
            assistant_content=assistant_content,
            force_think_block=force_think_block,
            clean_whitespace=clean_whitespace,
            remove_quotes=remove_quotes,
        )
        conv.is_final = is_final
        return self.format(conv)


# Default formatter instance
default_formatter = Qwen3Formatter()


def format_prompt(
    user_prompt: str,
    system_prompt: str = "",
    thinking_content: str = "",
    assistant_content: str = "",
    force_think_block: bool = False,
    is_final: bool = True,
    clean_whitespace: bool = True,
    remove_quotes: bool = False,
) -> str:
    """
    Convenience function to format a simple prompt.

    Uses content-driven logic: thinking block is added only if
    thinking_content is provided or force_think_block=True.

    Args:
        user_prompt: The user's request
        system_prompt: Optional system prompt
        thinking_content: Content inside <think></think> (triggers think block)
        assistant_content: Content after </think>
        force_think_block: If True, add empty think block even without content
        is_final: Whether to omit final <|im_end|>
        clean_whitespace: If True (default), clean extra whitespace/newlines
        remove_quotes: If True, strip " characters (for JSON-type prompts)

    Example (matches official HF Space - no think block):
        formatted = format_prompt("A cat sleeping")

    Example (with thinking content):
        formatted = format_prompt(
            user_prompt="A cat sleeping on a windowsill",
            system_prompt="Generate photorealistic images.",
            thinking_content="Natural lighting, soft shadows...",
        )

    Example (force empty think block):
        formatted = format_prompt("A cat", force_think_block=True)

    Example (strip quotes from JSON prompt):
        formatted = format_prompt('"A cat sleeping"', remove_quotes=True)
    """
    return default_formatter.format_simple(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        thinking_content=thinking_content,
        assistant_content=assistant_content,
        force_think_block=force_think_block,
        is_final=is_final,
        clean_whitespace=clean_whitespace,
        remove_quotes=remove_quotes,
    )
