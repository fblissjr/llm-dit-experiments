"""
Qwen3-VL embedding extraction for vision conditioning.

This module provides the VLEmbeddingExtractor class for extracting vision-conditioned
embeddings from Qwen3-VL that can be used to condition Z-Image generation.

Architecture Insight:
    Qwen3-VL-4B's text model shares architecture with Qwen3-4B (hidden_size=2560).
    The vision encoder projects image features into this 2560-dim space via
    PatchMerger.linear_fc2, enabling zero-shot vision conditioning without training.

Supported Model Variants:
    This module supports two Qwen3-VL-4B variants, both compatible with Z-Image:

    1. Qwen3-VL-4B-Instruct (non-thinking):
       - Does NOT use <think>...</think> blocks
       - Chat template ends with: <|im_start|>assistant\\n
       - Official params: temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5

    2. Qwen3-VL-4B-Thinking:
       - NATIVELY supports <think>...</think> blocks in chat template
       - Chat template ends with: <|im_start|>assistant\\n<think>\\n
       - May preserve visual concepts better in later layers due to "thinking" training

    The model variant is auto-detected from the chat_template at load time.
    See src/llm_dit/constants/__init__.py for all token IDs.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from PIL import Image

from .blending import (
    DEFAULT_TARGET_STD,
    mask_outlier_dimensions,
    normalize_hybrid,
    normalize_per_dimension,
    scale_embeddings,
)

# Import token constants from central location
# Note: THINK_* tokens kept for parsing output from Thinking model variant
from llm_dit.constants import (
    THINK_END_TOKEN_ID,
    THINK_START_TOKEN_ID,
    VISION_END_TOKEN_ID,
    VISION_START_TOKEN_ID,
)

if TYPE_CHECKING:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

logger = logging.getLogger(__name__)


@dataclass
class VLExtractionResult:
    """Result of VL embedding extraction."""

    embeddings: torch.Tensor  # (seq_len, hidden_dim)
    num_tokens: int
    hidden_layer: int
    original_std: float
    scaled_std: float
    scale_factor: float
    token_selection: str  # "all", "image_only", "image_only_no_markers", "text_only"
    text_description: str | None
    full_prompt_with_tokens: str  # Full decoded prompt including all special tokens
    input_token_ids: list[int]  # Raw token IDs for debugging
    normalization_mode: str = "global"  # "global", "per_dim", or "hybrid"
    system_prompt: str | None = None  # System prompt used (if any)
    # Outlier masking fields
    outlier_masking: str = "none"  # "none", "zero", "clamp", or "scale"
    outlier_threshold: float = 10.0  # Std ratio threshold for masking
    masked_dimensions: list[int] | None = None  # Dimensions that were masked
    masked_dim_ratios: dict[int, float] | None = None  # Dimension index -> std ratio
    # Model variant info
    model_variant: str = "instruct"  # "instruct" or "thinking"


class VLEmbeddingExtractor:
    """
    Extract vision-conditioned embeddings from Qwen3-VL.

    This class manages the Qwen3-VL model lifecycle and provides methods
    for extracting embeddings suitable for Z-Image conditioning.

    Supported variants (auto-detected):
        - "instruct": Qwen3-VL-4B-Instruct (no think blocks)
        - "thinking": Qwen3-VL-4B-Thinking (native <think> support)

    Example:
        >>> extractor = VLEmbeddingExtractor.from_pretrained("/path/to/Qwen3-VL-4B")
        >>> print(f"Model variant: {extractor.model_variant}")
        >>> result = extractor.extract(image, text="A house with a red roof")
        >>> print(result.embeddings.shape)  # (seq_len, 2560)

    Memory Management:
        For memory-constrained systems, use the two-stage workflow:
        1. Extract embeddings with `extract()`
        2. Call `unload()` to free VRAM
        3. Load Z-Image pipeline and generate with embeddings
    """

    def __init__(
        self,
        model: "Qwen3VLForConditionalGeneration",
        processor: "AutoProcessor",
        device: str = "cuda",
        model_variant: str = "instruct",
    ):
        """
        Initialize with pre-loaded model and processor.

        Use `from_pretrained()` factory method instead.

        Args:
            model: Loaded Qwen3VL model
            processor: AutoProcessor for tokenization
            device: Device the model is on
            model_variant: "instruct" or "thinking" (auto-detected by from_pretrained)
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.model_variant = model_variant
        self._is_thinking_model = model_variant == "thinking"
        self._loaded = True

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "VLEmbeddingExtractor":
        """
        Load Qwen3-VL model from path or HuggingFace ID.

        The model variant (instruct vs thinking) is auto-detected from the
        chat_template. Thinking models have native <think> block support.

        Args:
            model_path: Path to model directory or HuggingFace model ID
            device: Device to load model on (cuda, cpu, auto)
            torch_dtype: Model precision (default: bfloat16)

        Returns:
            VLEmbeddingExtractor instance with model_variant set
        """
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        logger.info(f"Loading Qwen3-VL from {model_path}...")

        processor = AutoProcessor.from_pretrained(model_path)

        # Qwen3 uses left padding (consistent with text-only model)
        processor.tokenizer.padding_side = "left"

        # Auto-detect model variant from chat_template
        # Thinking model template ends with: <|im_start|>assistant\n<think>\n
        # Instruct model template ends with: <|im_start|>assistant\n
        chat_template = processor.tokenizer.chat_template or ""
        is_thinking = "assistant\\n<think>" in chat_template
        model_variant = "thinking" if is_thinking else "instruct"

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
        )

        hidden_size = model.config.text_config.hidden_size
        logger.info(f"Qwen3-VL loaded. Hidden size: {hidden_size}, variant: {model_variant}")

        if hidden_size != 2560:
            logger.warning(
                f"Hidden size {hidden_size} != 2560 (Z-Image expected). "
                f"Embeddings may not be compatible without projection."
            )

        return cls(model, processor, device, model_variant)

    @classmethod
    def find_model_path(
        cls,
        prefer_variant: str | None = None,
    ) -> tuple[str | None, str | None]:
        """
        Auto-detect Qwen3-VL model in common locations.

        Args:
            prefer_variant: If specified ("thinking" or "instruct"), prioritize
                that variant. Otherwise, checks Thinking first, then Instruct.

        Returns:
            Tuple of (model_path, variant) if found, (None, None) otherwise.
            variant is "thinking" or "instruct".
        """
        # Candidates: (path, variant)
        candidates = [
            # Thinking variants (prioritized for experiments)
            (Path.home() / "Storage" / "Qwen3-VL-4B-Thinking", "thinking"),
            (Path.home() / "models" / "Qwen3-VL-4B-Thinking", "thinking"),
            # Instruct variants
            (Path.home() / "Storage" / "Qwen3-VL-4B-Instruct", "instruct"),
            (Path.home() / "models" / "Qwen3-VL-4B-Instruct", "instruct"),
            (Path("/models/Qwen3-VL-4B-Instruct"), "instruct"),
            (Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-VL-4B-Instruct", "instruct"),
        ]

        # If prefer_variant specified, sort to prioritize that variant
        if prefer_variant:
            candidates.sort(key=lambda x: x[1] != prefer_variant)

        for path, variant in candidates:
            if path.exists():
                logger.info(f"Found Qwen3-VL at {path} (variant: {variant})")
                return str(path), variant

        return None, None

    def extract(
        self,
        image: Image.Image,
        text: str | None = None,
        hidden_layer: int = -8,
        image_tokens_only: bool = False,
        image_tokens_no_markers: bool = False,
        text_tokens_only: bool = True,
        scale_to_text: bool = True,
        target_std: float = DEFAULT_TARGET_STD,
        normalization_mode: str = "global",
        system_prompt: str | None = None,
        outlier_masking: str = "none",
        outlier_threshold: float = 10.0,
    ) -> VLExtractionResult:
        """
        Extract vision-conditioned embeddings from an image.

        This uses Qwen3-VL-4B-Instruct, which is a NON-THINKING model.
        No think block tokens are injected (unlike the separate Thinking model variant).

        Args:
            image: Input image (PIL Image)
            text: Optional text description to include with image
            hidden_layer: Which hidden layer to extract (-8 recommended, produces cleaner results than -2)
            image_tokens_only: If True, only extract image token hidden states (includes marker tokens)
            image_tokens_no_markers: If True, only extract image tokens WITHOUT the special markers
            text_tokens_only: If True (default), only extract text token hidden states (recommended - image tokens cause artifacts)
            scale_to_text: If True, scale embeddings to match text statistics
            target_std: Target standard deviation for global scaling
            normalization_mode: How to normalize embeddings:
                - "global": Scale by global std only (original behavior)
                - "per_dim": Per-dimension normalization to match Qwen3-4B stats
                  (CRITICAL for image tokens which have 600x+ per-dim outliers)
                - "hybrid": 50/50 blend of global and per-dim normalization
            system_prompt: Optional system prompt to prepend. If provided, adds a system
                message before the user message.
            outlier_masking: How to handle dimensions with extreme std ratios vs Qwen3-4B:
                - "none": No masking (default)
                - "zero": Zero out outlier dimensions entirely
                - "clamp": Scale outlier dimensions to threshold level
                - "scale": Proportionally reduce outlier dimension values
            outlier_threshold: Std ratio threshold for outlier detection (default: 10.0).
                Dimension 396 has 617x ratio, dimension 4 has 42x ratio.

        Returns:
            VLExtractionResult with embeddings and metadata

        Note:
            Token selection options are mutually exclusive.
            After transformer self-attention, ALL positions carry mixed information
            from the entire sequence. This filtering tests whether token position
            affects quality, not whether information is "pure" image or text.

        Key Finding (2025-12-12):
            VL text tokens have 0.999 correlation with Qwen3-4B per-dimension stats.
            VL image tokens have only 0.737 correlation with extreme outliers.
            For image tokens, use normalization_mode="per_dim" for best results.
        """
        if not self._loaded:
            raise RuntimeError("Model has been unloaded. Create new extractor.")

        # Check mutual exclusivity
        selections = [image_tokens_only, image_tokens_no_markers, text_tokens_only]
        if sum(selections) > 1:
            raise ValueError("Token selection options are mutually exclusive")

        # Build message content
        messages = []

        # Add system message if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Image is always in the user message
        content = [{"type": "image", "image": image}]
        if text:
            content.append({"type": "text", "text": text})

        messages.append({"role": "user", "content": content})

        # Process inputs using Qwen3-VL's chat template
        # - Instruct model: add_generation_prompt=True produces `assistant\n` (no think block)
        # - Thinking model: add_generation_prompt=True produces `assistant\n<think>\n` (native)
        # No manual think token injection needed - the template handles it based on model variant.
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        logger.debug(f"Input sequence length: {inputs['input_ids'].shape[1]}")

        # Capture the full prompt with special tokens for debugging/metadata
        # Decode the full input sequence (including image placeholders)
        input_token_ids = inputs["input_ids"][0].tolist()
        full_prompt_with_tokens = self.processor.tokenizer.decode(
            input_token_ids,
            skip_special_tokens=False,  # Keep ALL special tokens visible
        )
        logger.debug(f"Full prompt with tokens: {repr(full_prompt_with_tokens[:200])}...")
        # Log the prompt ending to verify think block presence for different model variants
        prompt_ending = full_prompt_with_tokens[-100:] if len(full_prompt_with_tokens) > 100 else full_prompt_with_tokens
        logger.info(f"Prompt template ending ({self.model_variant}): ...{repr(prompt_ending)}")

        # Move to device
        device = next(self.model.parameters()).device
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        logger.debug(f"Input tokens: {inputs['input_ids'].shape[1]}")

        # Forward pass to get hidden states
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # Extract specified hidden layer
        hidden_states = outputs.hidden_states[hidden_layer][0]  # Remove batch dim
        logger.debug(
            f"Hidden states: shape={hidden_states.shape}, "
            f"mean={hidden_states.mean():.4f}, std={hidden_states.std():.4f}"
        )

        # Apply outlier masking to IMAGE TOKENS ONLY before token selection.
        # This is critical because image tokens have extreme per-dimension outliers
        # (dim 396: 617x, dim 4: 42x) that get diluted when averaged with text tokens.
        masked_dims = None
        masked_ratios = None
        if outlier_masking != "none":
            start_idx, end_idx = self._find_vision_token_range(inputs["input_ids"][0])
            if start_idx is not None and end_idx is not None:
                # Extract image tokens (excluding markers for cleaner stats)
                image_tokens = hidden_states[start_idx + 1 : end_idx]

                if image_tokens.shape[0] > 0:
                    # Apply masking to image tokens only
                    masked_image_tokens, mask_info = mask_outlier_dimensions(
                        image_tokens,
                        threshold=outlier_threshold,
                        mode=outlier_masking,
                    )
                    masked_dims = mask_info.get("masked_dimensions", [])
                    masked_ratios = mask_info.get("ratios", {})

                    if masked_dims:
                        # Replace image tokens in the full sequence
                        hidden_states = hidden_states.clone()
                        hidden_states[start_idx + 1 : end_idx] = masked_image_tokens
                        logger.debug(
                            f"Masked {len(masked_dims)} outlier dimensions in image tokens "
                            f"with mode={outlier_masking}: {masked_dims[:5]}"
                            f"{'...' if len(masked_dims) > 5 else ''}"
                        )
                    else:
                        logger.debug(
                            f"No outlier dimensions found in image tokens above threshold {outlier_threshold}"
                        )
            else:
                logger.warning("Could not find vision token markers for outlier masking")

        # Determine token selection mode
        token_selection = "all"
        if image_tokens_only:
            hidden_states = self._filter_image_tokens(
                hidden_states, inputs["input_ids"][0], include_markers=True
            )
            token_selection = "image_only"
        elif image_tokens_no_markers:
            hidden_states = self._filter_image_tokens(
                hidden_states, inputs["input_ids"][0], include_markers=False
            )
            token_selection = "image_only_no_markers"
        elif text_tokens_only:
            hidden_states = self._filter_text_tokens(
                hidden_states, inputs["input_ids"][0]
            )
            token_selection = "text_only"

        # Scale to match text embedding statistics
        original_std = hidden_states.std().item()
        if scale_to_text and original_std > 0:
            if normalization_mode == "per_dim":
                hidden_states = normalize_per_dimension(hidden_states)
                # Scale factor is approximate for per-dim normalization
                scale_factor = DEFAULT_TARGET_STD / original_std
                logger.debug("Applied per-dimension normalization")
            elif normalization_mode == "hybrid":
                hidden_states = normalize_hybrid(hidden_states)
                scale_factor = DEFAULT_TARGET_STD / original_std
                logger.debug("Applied hybrid normalization")
            else:  # "global" or default
                hidden_states = scale_embeddings(hidden_states, target_std)
                scale_factor = target_std / original_std
        else:
            scale_factor = 1.0

        return VLExtractionResult(
            embeddings=hidden_states.cpu(),
            num_tokens=hidden_states.shape[0],
            hidden_layer=hidden_layer,
            original_std=original_std,
            scaled_std=hidden_states.std().item(),
            scale_factor=scale_factor,
            token_selection=token_selection,
            text_description=text,
            full_prompt_with_tokens=full_prompt_with_tokens,
            input_token_ids=input_token_ids,
            normalization_mode=normalization_mode,
            system_prompt=system_prompt,
            outlier_masking=outlier_masking,
            outlier_threshold=outlier_threshold,
            masked_dimensions=masked_dims,
            masked_dim_ratios=masked_ratios,
            model_variant=self.model_variant,
        )

    def generate(
        self,
        prompt: str | None = None,
        image: Image.Image | None = None,
        system_prompt: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
        presence_penalty: float = 1.5,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text using Qwen3-VL with optional image input.

        This method enables using Qwen3-VL for prompt rewriting with vision support.
        It can process text-only, image-only, or combined image+text inputs.

        Sampling Defaults (from Qwen3-VL-4B-Instruct model card):
            temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5

            This is a NON-THINKING model (Instruct variant). For the Thinking variant,
            use Qwen3-VL-4B-Thinking with different parameters.

        Qwen3 Best Practices:
        - DO NOT use greedy decoding (causes repetition loops)
        - presence_penalty=1.5 (model card default) helps reduce repetitions

        Args:
            prompt: User prompt/message (optional if image provided)
            image: Input image for vision-language processing (optional if prompt provided)
            system_prompt: Optional system prompt (rewriter template content)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (default: 0.7, VL Instruct)
            top_p: Nucleus sampling threshold (default: 0.8, VL Instruct)
            top_k: Top-k sampling (default: 20)
            min_p: Minimum probability threshold (default: 0.0, disabled)
            presence_penalty: Penalty for token presence (default: 1.5, VL Instruct)
            do_sample: Whether to use sampling (False = greedy, NOT recommended for Qwen3)

        Returns:
            Generated text (assistant response)

        Raises:
            RuntimeError: If model has been unloaded
            ValueError: If neither prompt nor image is provided

        Example:
            # Image-only (describe what's in the image)
            result = extractor.generate(
                image=pil_image,
                system_prompt="Describe this image for use as an image generation prompt.",
            )

            # Image + text (rewrite prompt based on image style)
            result = extractor.generate(
                prompt="A cat sleeping in sunlight",
                image=style_reference,
                system_prompt="Rewrite this prompt to match the style of the image.",
            )
        """
        if not self._loaded:
            raise RuntimeError("Model has been unloaded. Create new extractor.")

        if prompt is None and image is None:
            raise ValueError("At least one of 'prompt' or 'image' must be provided")

        # Build messages for chat template
        # Note: Qwen3-VL processor expects content to be a list of dicts for all messages
        # when processing vision inputs, so we wrap system prompt in the same format
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

        # Build user message content (can have image, text, or both)
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        if prompt is not None:
            content.append({"type": "text", "text": prompt})

        messages.append({"role": "user", "content": content})

        # Process inputs using Qwen3-VL's chat template
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_length = inputs["input_ids"].shape[1]
        logger.debug(f"[VLEmbeddingExtractor.generate] Input tokens: {input_length}")

        # Move to device
        device = next(self.model.parameters()).device
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # Get proper termination tokens for Qwen3
        tokenizer = self.processor.tokenizer
        eos_token_ids = []
        if tokenizer.eos_token_id is not None:
            eos_token_ids.append(tokenizer.eos_token_id)

        # Add Qwen3-specific stop tokens
        for stop_token in ["<|im_end|>", "<|endoftext|>"]:
            try:
                token_id = tokenizer.convert_tokens_to_ids(stop_token)
                if token_id is not None and token_id not in eos_token_ids:
                    eos_token_ids.append(token_id)
            except Exception:
                pass

        # Use single token or list
        eos_token_id = eos_token_ids if len(eos_token_ids) > 1 else (eos_token_ids[0] if eos_token_ids else None)
        pad_token_id = tokenizer.pad_token_id or (eos_token_ids[0] if eos_token_ids else 0)

        # Build generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
        }

        if do_sample and temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["top_k"] = top_k
            if min_p > 0.0:
                gen_kwargs["min_p"] = min_p
            if presence_penalty > 0.0:
                # transformers uses repetition_penalty (multiplicative, >1.0 = more penalty)
                gen_kwargs["repetition_penalty"] = 1.0 + (presence_penalty * 0.15)
        else:
            gen_kwargs["do_sample"] = False

        logger.debug(f"[VLEmbeddingExtractor.generate] Generation kwargs: {gen_kwargs}")
        logger.info(f"[VLEmbeddingExtractor.generate] Starting generation (max_new_tokens={max_new_tokens})...")

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
                use_cache=True,
            )

        logger.info("[VLEmbeddingExtractor.generate] Generation completed")

        # Decode only the generated part (skip input tokens)
        generated_ids = outputs[0, input_length:].tolist()

        # Parse thinking content at token level (most reliable)
        thinking_content = None
        content_ids = generated_ids

        try:
            # Find </think> token from the end (in case of multiple)
            think_end_idx = len(generated_ids) - generated_ids[::-1].index(THINK_END_TOKEN_ID)
            # Everything before </think> is thinking (may include <think> token at start)
            thinking_ids = generated_ids[:think_end_idx]
            content_ids = generated_ids[think_end_idx:]

            # Remove <think> token from start if present
            if thinking_ids and thinking_ids[0] == THINK_START_TOKEN_ID:
                thinking_ids = thinking_ids[1:]

            # Decode thinking
            thinking_content = tokenizer.decode(thinking_ids, skip_special_tokens=True).strip()
            thinking_content = thinking_content.removeprefix("<think>").removesuffix("</think>").strip()

            logger.info(f"[VLEmbeddingExtractor.generate] Extracted thinking ({len(thinking_content)} chars)")
        except ValueError:
            # No </think> token found - model didn't use thinking format
            logger.debug("[VLEmbeddingExtractor.generate] No </think> token found, using full output")

        # Decode the content
        generated_text = tokenizer.decode(content_ids, skip_special_tokens=False)

        # Clean up end tokens
        for end_token in ["<|im_end|>", "<|endoftext|>"]:
            if generated_text.endswith(end_token):
                generated_text = generated_text[:-len(end_token)]

        generated_text = generated_text.strip()

        # If we extracted thinking, prepend it with <think> tags for downstream parsing
        if thinking_content:
            generated_text = f"<think>\n{thinking_content}\n</think>\n\n{generated_text}"

        logger.debug(f"[VLEmbeddingExtractor.generate] Generated {len(generated_ids)} tokens")

        return generated_text

    def _filter_image_tokens(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        include_markers: bool = True,
    ) -> torch.Tensor:
        """Filter hidden states to only include image token positions.

        Args:
            hidden_states: Full sequence hidden states
            input_ids: Token IDs to find marker positions
            include_markers: If True, include the special marker tokens (151652, 151653).
                           If False, exclude them (only actual image content).
        """
        start_idx, end_idx = self._find_vision_token_range(input_ids)

        if start_idx is not None and end_idx is not None:
            if include_markers:
                filtered = hidden_states[start_idx : end_idx + 1]
            else:
                # Exclude the marker tokens themselves
                filtered = hidden_states[start_idx + 1 : end_idx]
            logger.debug(
                f"Filtered to image tokens (markers={'incl' if include_markers else 'excl'}): "
                f"{hidden_states.shape[0]} -> {filtered.shape[0]}"
            )
            return filtered
        else:
            logger.warning("Could not find vision token markers, using all tokens")
            return hidden_states

    def _filter_text_tokens(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Filter hidden states to EXCLUDE image token positions (text only)."""
        start_idx, end_idx = self._find_vision_token_range(input_ids)

        if start_idx is not None and end_idx is not None:
            # Concatenate tokens before and after image region
            before_image = hidden_states[:start_idx]
            after_image = hidden_states[end_idx + 1:]
            filtered = torch.cat([before_image, after_image], dim=0)
            logger.debug(
                f"Filtered to text tokens: {hidden_states.shape[0]} -> {filtered.shape[0]} "
                f"(removed {start_idx} to {end_idx})"
            )
            return filtered
        else:
            logger.warning("Could not find vision token markers, using all tokens")
            return hidden_states

    def _find_vision_token_range(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[int | None, int | None]:
        """Find the start and end indices of vision tokens."""
        input_ids_list = input_ids.cpu().tolist()

        start_idx = None
        end_idx = None

        for i, tid in enumerate(input_ids_list):
            if tid == VISION_START_TOKEN_ID:
                start_idx = i
            if tid == VISION_END_TOKEN_ID:
                end_idx = i
                break

        return start_idx, end_idx

    def unload(self) -> None:
        """
        Unload model from memory to free VRAM.

        After calling this, the extractor cannot be used until recreated.
        """
        if self._loaded:
            del self.model
            del self.processor
            self._loaded = False

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Qwen3-VL model unloaded from memory")

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "_loaded") and self._loaded:
            self.unload()


def estimate_token_count(image: Image.Image) -> int:
    """
    Estimate the number of vision tokens for an image.

    Qwen3-VL uses spatial merge (2x2 patches -> 1 token), so:
    - 256x256 image -> 64 tokens
    - 512x512 image -> 256 tokens
    - 1024x1024 image -> 1024 tokens
    - 2048x2048 image -> 4096 tokens (exceeds Z-Image 1504 limit!)

    Args:
        image: Input image

    Returns:
        Estimated number of vision tokens
    """
    # Qwen3-VL uses 16x16 patch size with 2x2 spatial merge
    # This gives an effective 32x32 pixel per token
    # See config.json: vision_config.patch_size=16, vision_config.spatial_merge_size=2
    patch_size = 16
    spatial_merge = 2

    w, h = image.size
    # Effective merged patch size
    merged_patch_size = patch_size * spatial_merge  # 32 pixels

    tokens = (w // merged_patch_size) * (h // merged_patch_size)

    return tokens
