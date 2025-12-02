"""
Qwen3 chat template formatter.

Formats conversations into the exact token format expected by Qwen3-4B.
Based on: ComfyUI-QwenImageWanBridge/nodes/z_image_encoder.py _format_prompt()

Key format details:
- System: <|im_start|>system\\n{content}<|im_end|>
- User: <|im_start|>user\\n{content}<|im_end|>
- Assistant (with thinking): <|im_start|>assistant\\n<think>\\n{thinking}\\n</think>\\n\\n{content}
- Final message: Omits <|im_end|> if is_final=True (model is generating)

Reference:
- DiffSynth uses tokenizer.apply_chat_template() with enable_thinking=True
- We format manually for full control over thinking content
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
            skip_end: If True, omit closing <|im_end|>

        Returns:
            Formatted message string
        """
        role = msg.role.value
        content = msg.content

        if msg.role == Role.ASSISTANT and enable_thinking:
            # Assistant message with thinking block
            thinking = msg.thinking if msg.thinking else ""
            inner = f"{self.THINK_START}\n{thinking}\n{self.THINK_END}\n\n{content}"

            if skip_end:
                return f"{self.IM_START}{role}\n{inner}"
            else:
                return f"{self.IM_START}{role}\n{inner}{self.IM_END}"
        else:
            # System, user, or assistant without thinking
            if skip_end:
                return f"{self.IM_START}{role}\n{content}"
            else:
                return f"{self.IM_START}{role}\n{content}{self.IM_END}"

    def format_simple(
        self,
        user_prompt: str,
        system_prompt: str = "",
        thinking_content: str = "",
        assistant_content: str = "",
        enable_thinking: bool = True,
        is_final: bool = True,
    ) -> str:
        """
        Format a simple single-turn conversation.

        Convenience method for the most common use case.

        Args:
            user_prompt: The user's request
            system_prompt: Optional system prompt
            thinking_content: Content inside <think></think>
            assistant_content: Content after </think>
            enable_thinking: Whether to include thinking block
            is_final: Whether to omit final <|im_end|>

        Returns:
            Formatted string
        """
        conv = Conversation.simple(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            thinking_content=thinking_content,
            assistant_content=assistant_content,
            enable_thinking=enable_thinking,
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
    enable_thinking: bool = True,
    is_final: bool = True,
) -> str:
    """
    Convenience function to format a simple prompt.

    Example:
        formatted = format_prompt(
            user_prompt="A cat sleeping on a windowsill",
            system_prompt="Generate photorealistic images.",
            thinking_content="Natural lighting, soft shadows...",
        )
    """
    return default_formatter.format_simple(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        thinking_content=thinking_content,
        assistant_content=assistant_content,
        enable_thinking=enable_thinking,
        is_final=is_final,
    )
