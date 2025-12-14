"""
Qwen3-VL embedding extraction for vision conditioning.

This module provides the VLEmbeddingExtractor class for extracting vision-conditioned
embeddings from Qwen3-VL that can be used to condition Z-Image generation.

Architecture Insight:
    Qwen3-VL-4B's text model shares architecture with Qwen3-4B (hidden_size=2560).
    The vision encoder projects image features into this 2560-dim space via
    PatchMerger.linear_fc2, enabling zero-shot vision conditioning without training.

Chat Template Format:
    Z-Image was trained with Qwen3-4B using the format that includes empty think blocks:
        <|im_start|>user
        {prompt}<|im_end|>
        <|im_start|>assistant
        <think>

        </think>

        {content}

    To match this, we use apply_chat_template(enable_thinking=False) which
    ADDS the empty think block. Qwen3's naming is counterintuitive:
        enable_thinking=True  -> NO think block (model CAN think)
        enable_thinking=False -> ADD empty think block (skip thinking)
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

if TYPE_CHECKING:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

logger = logging.getLogger(__name__)

# Qwen3 vision token markers
VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653

# Qwen3 think block token IDs (for manual injection since Qwen3-VL doesn't support enable_thinking)
THINK_START_TOKEN_ID = 151667  # <think>
THINK_END_TOKEN_ID = 151668    # </think>
DOUBLE_NEWLINE_TOKEN_ID = 271  # \n\n (Qwen3 tokenizes double newlines as single token)


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
    chat_template_format: str  # "with_think_block" or "no_think_block"
    full_prompt_with_tokens: str  # Full decoded prompt including all special tokens
    input_token_ids: list[int]  # Raw token IDs for debugging
    normalization_mode: str = "global"  # "global", "per_dim", or "hybrid"
    system_prompt: str | None = None  # System prompt used (if any)
    # Outlier masking fields
    outlier_masking: str = "none"  # "none", "zero", "clamp", or "scale"
    outlier_threshold: float = 10.0  # Std ratio threshold for masking
    masked_dimensions: list[int] | None = None  # Dimensions that were masked
    masked_dim_ratios: dict[int, float] | None = None  # Dimension index -> std ratio


class VLEmbeddingExtractor:
    """
    Extract vision-conditioned embeddings from Qwen3-VL.

    This class manages the Qwen3-VL model lifecycle and provides methods
    for extracting embeddings suitable for Z-Image conditioning.

    Example:
        >>> extractor = VLEmbeddingExtractor.from_pretrained("/path/to/Qwen3-VL-4B")
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
    ):
        """
        Initialize with pre-loaded model and processor.

        Use `from_pretrained()` or `from_path()` factory methods instead.
        """
        self.model = model
        self.processor = processor
        self.device = device
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

        Args:
            model_path: Path to model directory or HuggingFace model ID
            device: Device to load model on (cuda, cpu, auto)
            torch_dtype: Model precision (default: bfloat16)

        Returns:
            VLEmbeddingExtractor instance
        """
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        logger.info(f"Loading Qwen3-VL from {model_path}...")

        processor = AutoProcessor.from_pretrained(model_path)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
        )

        hidden_size = model.config.text_config.hidden_size
        logger.info(f"Qwen3-VL loaded. Hidden size: {hidden_size}")

        if hidden_size != 2560:
            logger.warning(
                f"Hidden size {hidden_size} != 2560 (Z-Image expected). "
                f"Embeddings may not be compatible without projection."
            )

        return cls(model, processor, device)

    @classmethod
    def find_model_path(cls) -> str | None:
        """
        Auto-detect Qwen3-VL model in common locations.

        Returns:
            Path to model directory if found, None otherwise
        """
        candidates = [
            Path.home() / "Storage" / "Qwen3-VL-4B-Instruct",
            Path.home() / "models" / "Qwen3-VL-4B-Instruct",
            Path("/models/Qwen3-VL-4B-Instruct"),
            Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-VL-4B-Instruct",
        ]

        for candidate in candidates:
            if candidate.exists():
                logger.info(f"Found Qwen3-VL at {candidate}")
                return str(candidate)

        return None

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
        force_think_block: bool = True,
        system_prompt: str | None = None,
        outlier_masking: str = "none",
        outlier_threshold: float = 10.0,
    ) -> VLExtractionResult:
        """
        Extract vision-conditioned embeddings from an image.

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
            force_think_block: If True (default), inject empty think block tokens to match
                Qwen3-4B training format. If False, no think block is added.
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
        # NOTE: Qwen3-VL does NOT support enable_thinking parameter (it's ignored with a warning).
        # We can optionally inject think block tokens to match Z-Image's training format.
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Optionally inject think block tokens to match Qwen3-4B training format.
        # Z-Image was trained with Qwen3-4B using empty think blocks:
        #   <|im_start|>assistant\n<think>\n\n</think>\n\n
        # Qwen3-VL produces:
        #   <|im_start|>assistant\n
        # When force_think_block=True: append <think>\n\n</think>\n\n (4 tokens)
        chat_template_format = "no_think_block"
        if force_think_block:
            think_block_tokens = torch.tensor([[
                THINK_START_TOKEN_ID,    # <think>
                DOUBLE_NEWLINE_TOKEN_ID, # \n\n
                THINK_END_TOKEN_ID,      # </think>
                DOUBLE_NEWLINE_TOKEN_ID, # \n\n
            ]], dtype=inputs["input_ids"].dtype)

            # Append think block tokens to input_ids
            inputs["input_ids"] = torch.cat([inputs["input_ids"], think_block_tokens], dim=1)

            # Also extend attention_mask if present
            if "attention_mask" in inputs:
                think_block_mask = torch.ones((1, 4), dtype=inputs["attention_mask"].dtype)
                inputs["attention_mask"] = torch.cat([inputs["attention_mask"], think_block_mask], dim=1)

            chat_template_format = "with_think_block"
            logger.debug(f"Injected think block tokens, new sequence length: {inputs['input_ids'].shape[1]}")
        else:
            logger.debug(f"No think block injection, sequence length: {inputs['input_ids'].shape[1]}")

        # Capture the full prompt with special tokens for debugging/metadata
        # Decode the full input sequence (including image placeholders and think block)
        input_token_ids = inputs["input_ids"][0].tolist()
        full_prompt_with_tokens = self.processor.tokenizer.decode(
            input_token_ids,
            skip_special_tokens=False,  # Keep ALL special tokens visible
        )
        logger.debug(f"Full prompt with tokens: {repr(full_prompt_with_tokens[:200])}...")

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
            chat_template_format=chat_template_format,
            full_prompt_with_tokens=full_prompt_with_tokens,
            input_token_ids=input_token_ids,
            normalization_mode=normalization_mode,
            system_prompt=system_prompt,
            outlier_masking=outlier_masking,
            outlier_threshold=outlier_threshold,
            masked_dimensions=masked_dims,
            masked_dim_ratios=masked_ratios,
        )

    def generate(
        self,
        prompt: str | None = None,
        image: Image.Image | None = None,
        system_prompt: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text using Qwen3-VL with optional image input.

        This method enables using Qwen3-VL for prompt rewriting with vision support.
        It can process text-only, image-only, or combined image+text inputs.

        Qwen3 Best Practices (thinking mode):
        - temperature=0.6, top_p=0.95, top_k=20, min_p=0 (default)
        - DO NOT use greedy decoding (causes repetition)
        - presence_penalty=0-2 helps reduce endless repetitions

        Args:
            prompt: User prompt/message (optional if image provided)
            image: Input image for vision-language processing (optional if prompt provided)
            system_prompt: Optional system prompt (rewriter template content)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (Qwen3 thinking: 0.6)
            top_p: Nucleus sampling threshold (Qwen3 thinking: 0.95)
            top_k: Top-k sampling (Qwen3: 20)
            min_p: Minimum probability threshold (Qwen3: 0.0)
            presence_penalty: Penalty for token presence (0-2, helps reduce repetition)
            do_sample: Whether to use sampling (False = greedy, NOT recommended for Qwen3)

        Returns:
            Generated text (assistant response, may include <think> tags if model used them)

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
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

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
