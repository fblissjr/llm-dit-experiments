"""
Z-Image text encoder with template and conversation support.

This encoder provides a high-level interface that:
1. Accepts simple prompts or full Conversation objects
2. Applies templates (system prompts, thinking blocks)
3. Formats using Qwen3 chat template (matching official HF Space)
4. Encodes via the configured backend

Content-driven behavior (matches official HF Space):
- Default: No thinking block
- If thinking_content provided: Add think block with content
- If force_think_block=True: Add empty think block

Example:
    encoder = ZImageTextEncoder.from_pretrained("/path/to/z-image")

    # Simple usage (matches official HF Space - no think block)
    output = encoder.encode("A cat sleeping in sunlight")

    # With thinking content (automatically adds think block)
    output = encoder.encode(
        "A cat sleeping",
        thinking_content="Soft lighting, orange fur...",
    )

    # Force empty think block
    output = encoder.encode("A cat", force_think_block=True)

    # Full control with all components
    output = encoder.encode(
        "A cat sleeping",
        system_prompt="You are a photographer.",
        thinking_content="Focus on soft lighting...",
        assistant_content="Here is your image:",
    )
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
        force_think_block: bool = False,
        return_padded: bool = False,
        remove_quotes: bool = False,
        layer_index: int = -2,
    ) -> EncodingOutput:
        """
        Encode a prompt to embeddings.

        This method accepts either:
        1. A simple string prompt (will be wrapped in a Conversation)
        2. A Conversation object (used directly)

        Uses content-driven logic for thinking blocks:
        - If thinking_content provided: Add think block with content
        - If force_think_block=True: Add empty think block
        - Otherwise: No think block (matches official HF Space)

        Args:
            prompt: The prompt string or Conversation object
            template: Optional template name or Template object
            system_prompt: Override system prompt (ignored if Conversation)
            thinking_content: Thinking block content - triggers think block (ignored if Conversation)
            assistant_content: Assistant content after thinking (ignored if Conversation)
            force_think_block: If True, add empty think block even without content
            return_padded: Also return padded batch tensors
            remove_quotes: If True, strip " characters (for JSON-type prompts)
            layer_index: Which hidden layer to extract (default: -2, penultimate).
                        Useful values: -1 (last), -2 (penultimate), -3, -4.
                        Z-Image uses -2 by default.

        Returns:
            EncodingOutput with variable-length embeddings

        Example:
            # Simple prompt (matches official HF Space - no think block)
            output = encoder.encode("A beautiful sunset")

            # With template
            output = encoder.encode("A cat", template="photorealistic")

            # With thinking content (automatically adds think block)
            output = encoder.encode(
                "A cat sleeping",
                thinking_content="Soft lighting, warm afternoon...",
            )

            # Force empty think block
            output = encoder.encode("A cat", force_think_block=True)

            # Full control
            output = encoder.encode(
                "A cat sleeping",
                system_prompt="Generate natural images.",
                thinking_content="Warm afternoon light...",
                assistant_content="Here is your image:",
            )
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
                force_think_block=force_think_block,
                remove_quotes=remove_quotes,
            )

        # Format to chat template string
        formatted = self.formatter.format(conversation)
        logger.debug(f"Formatted prompt ({len(formatted)} chars)")

        # Encode via backend
        output = self.backend.encode(
            [formatted],
            return_padded=return_padded,
            layer_index=layer_index,
        )

        # Attach formatted prompt for debugging
        output.formatted_prompts = [formatted]
        return output

    def encode_batch(
        self,
        prompts: List[Union[str, Conversation]],
        template: str | Template | None = None,
        force_think_block: bool = False,
        return_padded: bool = False,
        layer_index: int = -2,
    ) -> EncodingOutput:
        """
        Encode a batch of prompts.

        For batch encoding, templates can be specified but per-prompt
        customization is not supported. Use Conversation objects for
        full control over individual prompts.

        Args:
            prompts: List of prompt strings or Conversation objects
            template: Optional template to apply to string prompts
            force_think_block: If True, add empty think block for string prompts
            return_padded: Also return padded batch tensors
            layer_index: Which hidden layer to extract (default: -2, penultimate)

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
                    force_think_block=force_think_block,
                )
                formatted = self.formatter.format(conv)
            formatted_list.append(formatted)

        output = self.backend.encode(
            formatted_list,
            return_padded=return_padded,
            layer_index=layer_index,
        )

        # Attach formatted prompts for debugging
        output.formatted_prompts = formatted_list
        return output

    def encode_blended(
        self,
        prompt: Union[str, Conversation],
        layer_weights: dict[int, float],
        template: str | Template | None = None,
        system_prompt: str | None = None,
        thinking_content: str | None = None,
        assistant_content: str | None = None,
        force_think_block: bool = False,
        return_padded: bool = False,
        remove_quotes: bool = False,
    ) -> EncodingOutput:
        """
        Encode a prompt using a weighted blend of multiple hidden layers.

        This allows combining semantic information from different depths of the
        transformer. For example, deeper layers (-5, -6) may capture more
        structural information while shallower layers (-1, -2) capture
        more semantic information.

        Args:
            prompt: The prompt string or Conversation object
            layer_weights: Dict mapping layer indices to weights, e.g.:
                {-2: 0.7, -5: 0.3} blends 70% penultimate + 30% layer -5
                Weights are normalized to sum to 1.0
            template: Optional template name or Template object
            system_prompt: Override system prompt
            thinking_content: Thinking block content
            assistant_content: Assistant content after thinking
            force_think_block: If True, add empty think block
            return_padded: Also return padded batch tensors
            remove_quotes: If True, strip " characters

        Returns:
            EncodingOutput with blended embeddings

        Example:
            # Blend semantic (-2) and structural (-5) layers
            output = encoder.encode_blended(
                "A cat sleeping in sunlight",
                layer_weights={-2: 0.7, -5: 0.3}
            )
        """
        # Build conversation
        if isinstance(prompt, Conversation):
            conversation = prompt
        else:
            conversation = self._build_conversation(
                prompt=prompt,
                template=template,
                system_prompt=system_prompt,
                thinking_content=thinking_content,
                assistant_content=assistant_content,
                force_think_block=force_think_block,
                remove_quotes=remove_quotes,
            )

        # Format to chat template string
        formatted = self.formatter.format(conversation)
        logger.debug(f"Formatted prompt for blending ({len(formatted)} chars)")

        # Encode with blending via backend
        output = self.backend.encode_blended(
            [formatted],
            layer_weights=layer_weights,
            return_padded=return_padded,
        )

        # Attach formatted prompt for debugging
        output.formatted_prompts = [formatted]
        return output

    def _build_conversation(
        self,
        prompt: str,
        template: str | Template | None = None,
        system_prompt: str | None = None,
        thinking_content: str | None = None,
        assistant_content: str | None = None,
        force_think_block: bool = False,
        remove_quotes: bool = False,
    ) -> Conversation:
        """
        Build a Conversation from parameters.

        Uses content-driven logic: thinking block is added only if
        thinking_content is provided or force_think_block=True.
        """
        # Resolve template
        resolved_template = self._resolve_template(template)

        # Apply template values as defaults
        final_system = system_prompt
        final_thinking = thinking_content
        final_assistant = assistant_content
        final_force_think_block = force_think_block

        if resolved_template is not None:
            if final_system is None:
                final_system = resolved_template.content
            if final_thinking is None and resolved_template.thinking_content:
                final_thinking = resolved_template.thinking_content
            if final_assistant is None and resolved_template.assistant_content:
                final_assistant = resolved_template.assistant_content
            # Template can force think block even without content
            if resolved_template.add_think_block is not None:
                final_force_think_block = resolved_template.add_think_block

        # Create conversation with content-driven logic
        return Conversation.simple(
            user_prompt=prompt,
            system_prompt=final_system or "",
            thinking_content=final_thinking or "",
            assistant_content=final_assistant or "",
            force_think_block=final_force_think_block,
            remove_quotes=remove_quotes,
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
