"""
Z-Image text encoder with template and conversation support.

This encoder provides a high-level interface that:
1. Accepts simple prompts or full Conversation objects
2. Applies templates (system prompts, thinking blocks)
3. Formats using Qwen3 chat template
4. Encodes via the configured backend

Example:
    encoder = ZImageTextEncoder.from_pretrained("/path/to/z-image")

    # Simple usage
    output = encoder.encode("A cat sleeping in sunlight")

    # With template
    output = encoder.encode(
        "A cat sleeping in sunlight",
        template="photorealistic",
    )

    # With full conversation control
    conv = Conversation.simple(
        user_prompt="A cat sleeping",
        system_prompt="Generate natural images.",
        thinking_content="Focus on soft lighting...",
        enable_thinking=True,
    )
    output = encoder.encode(conv)
"""

import logging
from pathlib import Path
from typing import List, Union

import torch

from llm_dit.backends import BackendConfig, EncodingOutput
from llm_dit.backends.transformers import TransformersBackend
from llm_dit.conversation import Conversation, Qwen3Formatter
from llm_dit.templates import Template, TemplateRegistry

logger = logging.getLogger(__name__)


class ZImageTextEncoder:
    """
    High-level text encoder for Z-Image with template support.

    This class provides a user-friendly interface for encoding prompts,
    handling all the complexity of template application, conversation
    formatting, and backend encoding.

    Attributes:
        backend: The underlying TextEncoderBackend
        templates: TemplateRegistry for loading templates
        formatter: Qwen3Formatter for chat template formatting
        default_template: Optional default template name
    """

    def __init__(
        self,
        backend: TransformersBackend,
        templates: TemplateRegistry | None = None,
        default_template: str | None = None,
    ):
        """
        Initialize the encoder.

        Args:
            backend: A TextEncoderBackend instance
            templates: Optional TemplateRegistry for template support
            default_template: Optional default template name to use
        """
        self.backend = backend
        self.templates = templates
        self.formatter = Qwen3Formatter()
        self.default_template = default_template

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        templates_dir: str | Path | None = None,
        default_template: str | None = None,
        **kwargs,
    ) -> "ZImageTextEncoder":
        """
        Load encoder from pretrained model.

        Args:
            model_path: Path to Z-Image model or HuggingFace ID
            templates_dir: Optional path to templates directory
            default_template: Optional default template name
            **kwargs: Additional arguments for TransformersBackend

        Returns:
            Initialized ZImageTextEncoder

        Example:
            encoder = ZImageTextEncoder.from_pretrained(
                "/path/to/z-image",
                templates_dir="templates/z_image",
                default_template="photorealistic",
            )
        """
        # Load backend
        backend = TransformersBackend.from_pretrained(model_path, **kwargs)

        # Load templates if directory provided
        templates = None
        if templates_dir is not None:
            templates_path = Path(templates_dir)
            if templates_path.exists():
                templates = TemplateRegistry.from_directory(templates_path)
                logger.info(f"Loaded {len(templates)} templates from {templates_path}")
            else:
                logger.warning(f"Templates directory not found: {templates_path}")

        return cls(backend, templates, default_template)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension (2560 for Qwen3-4B)."""
        return self.backend.embedding_dim

    @property
    def max_sequence_length(self) -> int:
        """Return max sequence length (512 for Z-Image)."""
        return self.backend.max_sequence_length

    @property
    def device(self) -> torch.device:
        """Return current device."""
        return self.backend.device

    @property
    def dtype(self) -> torch.dtype:
        """Return model dtype."""
        return self.backend.dtype

    def get_template(self, name: str) -> Template | None:
        """Get a template by name."""
        if self.templates is None:
            return None
        return self.templates.get(name)

    def list_templates(self) -> List[str]:
        """List available template names."""
        if self.templates is None:
            return []
        return list(self.templates.keys())

    def encode(
        self,
        prompt: Union[str, Conversation],
        template: str | Template | None = None,
        system_prompt: str | None = None,
        thinking_content: str | None = None,
        assistant_content: str | None = None,
        enable_thinking: bool = False,  # Default False to match diffusers/ComfyUI
        return_padded: bool = False,
    ) -> EncodingOutput:
        """
        Encode a prompt to embeddings.

        This method accepts either:
        1. A simple string prompt (will be wrapped in a Conversation)
        2. A Conversation object (used directly)

        When a string is provided, you can customize the encoding with:
        - template: Apply a named template or Template object
        - system_prompt: Override the system prompt
        - thinking_content: Content for the thinking block
        - assistant_content: Content after thinking
        - enable_thinking: Whether to include <think></think>

        Args:
            prompt: The prompt string or Conversation object
            template: Optional template name or Template object
            system_prompt: Override system prompt (ignored if Conversation)
            thinking_content: Thinking block content (ignored if Conversation)
            assistant_content: Assistant content after thinking (ignored if Conversation)
            enable_thinking: Whether to use thinking tags (ignored if Conversation)
            return_padded: Also return padded batch tensors

        Returns:
            EncodingOutput with variable-length embeddings

        Example:
            # Simple prompt
            output = encoder.encode("A beautiful sunset")

            # With template
            output = encoder.encode("A cat", template="photorealistic")

            # Full control
            output = encoder.encode(
                "A cat sleeping",
                system_prompt="Generate natural images with soft lighting.",
                thinking_content="I'll focus on warm afternoon light...",
                enable_thinking=True,
            )

            # Using Conversation directly
            conv = Conversation.simple("A cat", system_prompt="Natural images.")
            output = encoder.encode(conv)
        """
        # If already a Conversation, use it directly
        if isinstance(prompt, Conversation):
            conversation = prompt
        else:
            # Build Conversation from parameters
            conversation = self._build_conversation(
                prompt=prompt,
                template=template,
                system_prompt=system_prompt,
                thinking_content=thinking_content,
                assistant_content=assistant_content,
                enable_thinking=enable_thinking,
            )

        # Format to chat template string
        formatted = self.formatter.format(conversation)
        logger.debug(f"Formatted prompt ({len(formatted)} chars)")

        # Encode via backend
        return self.backend.encode([formatted], return_padded=return_padded)

    def encode_batch(
        self,
        prompts: List[Union[str, Conversation]],
        template: str | Template | None = None,
        enable_thinking: bool = False,  # Default False to match diffusers/ComfyUI
        return_padded: bool = False,
    ) -> EncodingOutput:
        """
        Encode a batch of prompts.

        For batch encoding, templates can be specified but per-prompt
        customization is not supported. Use Conversation objects for
        full control over individual prompts.

        Args:
            prompts: List of prompt strings or Conversation objects
            template: Optional template to apply to string prompts
            enable_thinking: Whether to use thinking tags for string prompts
            return_padded: Also return padded batch tensors

        Returns:
            EncodingOutput with embeddings for each prompt
        """
        formatted_list = []

        for prompt in prompts:
            if isinstance(prompt, Conversation):
                formatted = self.formatter.format(prompt)
            else:
                conv = self._build_conversation(
                    prompt=prompt,
                    template=template,
                    enable_thinking=enable_thinking,
                )
                formatted = self.formatter.format(conv)
            formatted_list.append(formatted)

        return self.backend.encode(formatted_list, return_padded=return_padded)

    def _build_conversation(
        self,
        prompt: str,
        template: str | Template | None = None,
        system_prompt: str | None = None,
        thinking_content: str | None = None,
        assistant_content: str | None = None,
        enable_thinking: bool = True,
    ) -> Conversation:
        """Build a Conversation from parameters."""
        # Resolve template
        resolved_template = self._resolve_template(template)

        # Apply template values as defaults
        final_system = system_prompt
        final_thinking = thinking_content
        final_assistant = assistant_content
        final_enable_thinking = enable_thinking

        if resolved_template is not None:
            if final_system is None:
                final_system = resolved_template.content
            if final_thinking is None and resolved_template.thinking_content:
                final_thinking = resolved_template.thinking_content
            if final_assistant is None and resolved_template.assistant_content:
                final_assistant = resolved_template.assistant_content
            if resolved_template.add_think_block is not None:
                final_enable_thinking = resolved_template.add_think_block

        # Create conversation
        return Conversation.simple(
            user_prompt=prompt,
            system_prompt=final_system or "",
            thinking_content=final_thinking or "",
            assistant_content=final_assistant or "",
            enable_thinking=final_enable_thinking,
        )

    def _resolve_template(
        self, template: str | Template | None
    ) -> Template | None:
        """Resolve template from name or use default."""
        # If Template object, use directly
        if isinstance(template, Template):
            return template

        # If string name, look up in registry
        if isinstance(template, str):
            if self.templates is None:
                logger.warning(f"Template '{template}' requested but no registry loaded")
                return None
            resolved = self.templates.get(template)
            if resolved is None:
                logger.warning(f"Template '{template}' not found in registry")
            return resolved

        # If None, try default template
        if template is None and self.default_template is not None:
            return self._resolve_template(self.default_template)

        return None

    def to(self, device: torch.device) -> "ZImageTextEncoder":
        """Move encoder to device."""
        self.backend.to(device)
        return self
