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

from .blending import DEFAULT_TARGET_STD, scale_embeddings

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
            target_std: Target standard deviation for scaling

        Returns:
            VLExtractionResult with embeddings and metadata

        Note:
            Token selection options are mutually exclusive.
            After transformer self-attention, ALL positions carry mixed information
            from the entire sequence. This filtering tests whether token position
            affects quality, not whether information is "pure" image or text.
        """
        if not self._loaded:
            raise RuntimeError("Model has been unloaded. Create new extractor.")

        # Check mutual exclusivity
        selections = [image_tokens_only, image_tokens_no_markers, text_tokens_only]
        if sum(selections) > 1:
            raise ValueError("Token selection options are mutually exclusive")

        # Build message content
        # Image is always in the user message
        content = [{"type": "image", "image": image}]
        if text:
            content.append({"type": "text", "text": text})

        messages = [{"role": "user", "content": content}]

        # Process inputs using Qwen3-VL's chat template
        # NOTE: Qwen3-VL does NOT support enable_thinking parameter (it's ignored with a warning).
        # We must MANUALLY inject the think block tokens to match Z-Image's training format.
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        # CRITICAL: Manually inject think block tokens to match Qwen3-4B training format.
        # Z-Image was trained with Qwen3-4B using empty think blocks:
        #   <|im_start|>assistant\n<think>\n\n</think>\n\n
        # Qwen3-VL produces:
        #   <|im_start|>assistant\n
        # We need to append: <think>\n\n</think>\n\n (4 tokens - Qwen3 uses token 271 for \n\n)
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

        logger.debug(f"Injected think block tokens, new sequence length: {inputs['input_ids'].shape[1]}")

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
            chat_template_format="with_think_block",  # Manually injected <think>\n\n</think>\n\n
            full_prompt_with_tokens=full_prompt_with_tokens,
            input_token_ids=input_token_ids,
        )

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
    - 512x512 image -> ~256 tokens
    - 1024x1024 image -> ~1024 tokens
    - 2048x2048 image -> ~4096 tokens (exceeds Z-Image 1504 limit!)

    Args:
        image: Input image

    Returns:
        Estimated number of vision tokens
    """
    # Qwen3-VL uses 14x14 patch size with 2x2 spatial merge
    patch_size = 14
    spatial_merge = 2

    w, h = image.size
    patches_w = w // patch_size
    patches_h = h // patch_size

    # After spatial merge
    tokens = (patches_w // spatial_merge) * (patches_h // spatial_merge)

    return tokens
