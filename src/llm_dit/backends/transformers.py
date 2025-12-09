"""
HuggingFace transformers backend for text encoding and generation.

This is the reference implementation, matching the behavior of:
- DiffSynth-Studio z_image.py encode_prompt()
- diffusers ZImagePipeline._encode_prompt()

Key implementation details:
- Uses AutoModelForCausalLM to support both embedding extraction and text generation
- Extracts hidden_states[-2] (penultimate layer) for embeddings
- Filters by attention mask to get variable-length outputs
- Tokenizer padding_side="left" as per Qwen3 convention
- Supports generate() for prompt rewriting using the same loaded model
- Optional embedding cache for repeated prompts (DiffSynth optimization)

Transformers v5 Migration:
- Uses quantization_config parameter instead of deprecated load_in_8bit/load_in_4bit
- BitsAndBytesConfig is passed directly to from_pretrained()
"""

import logging
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_dit.backends.config import BackendConfig
from llm_dit.backends.protocol import EncodingOutput
from llm_dit.utils.embedding_cache import CacheStats, EmbeddingCache

logger = logging.getLogger(__name__)


class TransformersBackend:
    """
    HuggingFace transformers backend for Qwen3-4B text encoding.

    This backend loads the text encoder portion of Z-Image and provides
    embeddings suitable for the DiT context refiner.

    Example:
        backend = TransformersBackend.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            subfolder="text_encoder",
        )
        output = backend.encode(["<|im_start|>user\\nA cat<|im_end|>"])
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: BackendConfig,
        cache: Optional[EmbeddingCache] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self._embedding_dim = model.config.hidden_size
        self._device = next(model.parameters()).device
        self._dtype = next(model.parameters()).dtype
        self._supports_generation = True
        self._cache = cache

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        model_subfolder: str = "text_encoder",
        tokenizer_subfolder: str = "tokenizer",
        config: BackendConfig | None = None,
        quantization_config: "BitsAndBytesConfig | None" = None,
        cache: Optional[EmbeddingCache] = None,
        enable_cache: bool = False,
        cache_size: int = 100,
        **kwargs,
    ) -> "TransformersBackend":
        """
        Load backend from pretrained model.

        Args:
            model_path: Path to model directory or HuggingFace ID
            model_subfolder: Subfolder containing text encoder model (default: "text_encoder")
            tokenizer_subfolder: Subfolder containing tokenizer (default: "tokenizer")
                For diffusers pipelines, tokenizer is typically in a separate folder.
                Set to None to load from model_subfolder.
            config: Optional BackendConfig, created from defaults if not provided
            quantization_config: Optional BitsAndBytesConfig for quantization (v5 API).
                Use this instead of the deprecated load_in_8bit/load_in_4bit flags.
            cache: Optional pre-configured EmbeddingCache instance
            enable_cache: If True and cache is None, create a new cache (default: False)
            cache_size: Maximum number of cached embeddings (default: 100)
            **kwargs: Additional arguments passed to from_pretrained

        Returns:
            Initialized TransformersBackend

        Example:
            # Load from HuggingFace
            backend = TransformersBackend.from_pretrained("Tongyi-MAI/Z-Image-Turbo")

            # Load from local path
            backend = TransformersBackend.from_pretrained("/path/to/model")

            # With 8-bit quantization (v5 API)
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            backend = TransformersBackend.from_pretrained(
                "/path/to/model",
                quantization_config=quantization_config,
            )

            # Custom subfolder layout
            backend = TransformersBackend.from_pretrained(
                "/path/to/model",
                model_subfolder="text_encoder",
                tokenizer_subfolder="tokenizer",
            )
        """
        if config is None:
            config = BackendConfig.for_z_image(model_path, subfolder=model_subfolder)

        # Merge kwargs with config
        torch_dtype = kwargs.pop("torch_dtype", config.get_torch_dtype())
        device_map = kwargs.pop("device_map", config.device)
        trust_remote_code = kwargs.pop("trust_remote_code", config.trust_remote_code)

        # Determine paths based on whether this is a local path or HuggingFace ID
        is_local = Path(model_path).exists()

        # Resolve tokenizer path
        if tokenizer_subfolder is None:
            tokenizer_subfolder = model_subfolder

        if is_local:
            tokenizer_path = str(Path(model_path) / tokenizer_subfolder) if tokenizer_subfolder else model_path
            model_load_path = str(Path(model_path) / model_subfolder) if model_subfolder else model_path
            hf_subfolder = None  # Don't use subfolder param for local paths
        else:
            tokenizer_path = model_path
            model_load_path = model_path
            hf_subfolder = model_subfolder  # Let transformers handle subfolder for HF

        logger.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer_kwargs = {
            "trust_remote_code": trust_remote_code,
            **kwargs,
        }
        # Only add subfolder if it's not None (transformers bugs on subfolder=None)
        if not is_local and tokenizer_subfolder:
            tokenizer_kwargs["subfolder"] = tokenizer_subfolder
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)

        # Qwen3 uses left padding
        tokenizer.padding_side = "left"

        # Build model kwargs
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
            **kwargs,
        }

        # Add quantization_config if provided (v5 API)
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            logger.info(f"Loading model with quantization: {quantization_config}")
        else:
            logger.info(f"Loading model from {model_load_path} (dtype={torch_dtype})")

        # Only add subfolder if it's not None (transformers bugs on subfolder=None)
        if hf_subfolder and not is_local:
            model_kwargs["subfolder"] = hf_subfolder
        model = AutoModelForCausalLM.from_pretrained(model_load_path, **model_kwargs)
        model.eval()

        logger.info(
            f"Loaded {model.config.model_type} with "
            f"hidden_size={model.config.hidden_size}, "
            f"num_layers={model.config.num_hidden_layers}"
        )

        # Set up embedding cache
        if cache is None and enable_cache:
            cache = EmbeddingCache(max_size=cache_size, enabled=True)
            logger.info(f"Embedding cache enabled (max_size={cache_size})")

        return cls(model, tokenizer, config, cache=cache)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension (2560 for Qwen3-4B)."""
        return self._embedding_dim

    @property
    def max_sequence_length(self) -> int:
        """Return max sequence length from config."""
        return self.config.max_length

    @property
    def device(self) -> torch.device:
        """Return model device."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Return model dtype."""
        return self._dtype

    def encode(
        self,
        texts: List[str],
        return_padded: bool = False,
        layer_index: int = -2,
        use_cache: bool = True,
    ) -> EncodingOutput:
        """
        Encode pre-formatted text to embeddings.

        This method expects text that already has the chat template applied.
        It does NOT apply any additional formatting.

        Args:
            texts: List of pre-formatted text strings
            return_padded: If True, also return padded batch tensors
            layer_index: Which hidden layer to extract (default: -2, penultimate).
                        Useful values: -1 (last), -2 (penultimate), -3, -4.
                        Z-Image uses -2 by default.
            use_cache: Whether to use embedding cache (default: True)

        Returns:
            EncodingOutput with variable-length embeddings per input

        Implementation matches DiffSynth z_image.py lines 174-196:
        1. Tokenize with padding="max_length"
        2. Forward through model with output_hidden_states=True
        3. Extract hidden_states[layer_index] (default: penultimate layer)
        4. Filter by attention_mask to get valid tokens only
        """
        # Check cache for single-text input (most common case)
        if use_cache and self._cache is not None and len(texts) == 1:
            cache_key = EmbeddingCache.make_key(texts[0], layer_index, return_padded)
            cached = self._cache.get(cache_key, device=self.device)
            if cached is not None:
                return cached

        # Tokenize - use padding=True to only pad to longest in batch (not max_length)
        # This is MUCH faster for short prompts (21 tokens vs 2048 tokens)
        inputs = self.tokenizer(
            texts,
            padding=True,  # Pad to longest in batch, not max_length
            truncation=True,
            max_length=self.config.max_length,  # Only used for truncation
            return_tensors="pt",
        )

        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device).bool()

        # Debug: log tokenization details
        seq_length = input_ids.shape[1]
        valid_tokens = attention_mask[0].sum().item()
        logger.debug(f"[TransformersBackend] Tokenized: {valid_tokens} valid tokens, seq_length={seq_length}")
        logger.debug(f"[TransformersBackend] Token IDs (first 20): {input_ids[0][:20].tolist()}")

        # Encode
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Extract specified layer (default: hidden_states[-2] for Z-Image)
        # Negative indices: -1 = last, -2 = penultimate, etc.
        hidden_states = outputs.hidden_states[layer_index]
        logger.debug(f"[TransformersBackend] Extracting layer {layer_index} of {len(outputs.hidden_states)} layers")

        # Filter by attention mask to get variable-length outputs
        # This matches diffusers behavior
        embeddings_list = []
        masks_list = []
        token_counts = []
        for i in range(len(texts)):
            mask = attention_mask[i]
            valid_embeds = hidden_states[i][mask]
            embeddings_list.append(valid_embeds)
            masks_list.append(mask[mask])  # All True for valid positions
            token_counts.append(valid_embeds.shape[0])

            # Debug: log embedding stats for comparison with other backends
            logger.debug(f"[TransformersBackend] Embedding [{i}]: shape={valid_embeds.shape}, dtype={valid_embeds.dtype}")
            logger.debug(f"[TransformersBackend] Embedding [{i}] stats: min={valid_embeds.min().item():.4f}, max={valid_embeds.max().item():.4f}, mean={valid_embeds.mean().item():.4f}, std={valid_embeds.std().item():.4f}")
            logger.debug(f"[TransformersBackend] Embedding [{i}] first 5 values: {valid_embeds[0, :5].tolist()}")
            logger.debug(f"[TransformersBackend] Embedding [{i}] last 5 values: {valid_embeds[-1, -5:].tolist()}")

        result = EncodingOutput(
            embeddings=embeddings_list,
            attention_masks=masks_list,
            token_counts=token_counts,
        )

        if return_padded:
            result.padded_embeddings = hidden_states
            result.padded_mask = attention_mask

        # Store in cache for single-text input
        if use_cache and self._cache is not None and len(texts) == 1:
            self._cache.put(cache_key, result)

        return result

    def encode_blended(
        self,
        texts: List[str],
        layer_weights: dict[int, float],
        return_padded: bool = False,
    ) -> EncodingOutput:
        """
        Encode text using a weighted blend of multiple hidden layers.

        This allows combining semantic information from different depths of the
        transformer. For example, deeper layers (-5, -6) may capture more
        structural/syntactic information while shallower layers (-1, -2) capture
        more semantic/conceptual information.

        Args:
            texts: List of pre-formatted text strings
            layer_weights: Dict mapping layer indices to weights, e.g.:
                {-2: 0.7, -5: 0.3} blends 70% penultimate + 30% layer -5
                Weights are normalized to sum to 1.0
            return_padded: If True, also return padded batch tensors

        Returns:
            EncodingOutput with blended embeddings

        Example:
            # Blend semantic (-2) and structural (-5) information
            output = backend.encode_blended(
                texts=["A cat sleeping"],
                layer_weights={-2: 0.7, -5: 0.3}
            )
        """
        if not layer_weights:
            raise ValueError("layer_weights must not be empty")

        # Normalize weights to sum to 1.0
        total_weight = sum(layer_weights.values())
        if total_weight <= 0:
            raise ValueError("layer_weights must sum to a positive value")
        normalized_weights = {k: v / total_weight for k, v in layer_weights.items()}

        logger.debug(f"[TransformersBackend] Blending layers: {normalized_weights}")

        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device).bool()

        # Encode once, get all hidden states
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Blend hidden states from specified layers
        blended_hidden = None
        for layer_idx, weight in normalized_weights.items():
            layer_hidden = outputs.hidden_states[layer_idx]
            if blended_hidden is None:
                blended_hidden = weight * layer_hidden
            else:
                blended_hidden = blended_hidden + weight * layer_hidden

        logger.debug(f"[TransformersBackend] Blended {len(normalized_weights)} layers")

        # Filter by attention mask to get variable-length outputs
        embeddings_list = []
        masks_list = []
        token_counts = []
        for i in range(len(texts)):
            mask = attention_mask[i]
            valid_embeds = blended_hidden[i][mask]
            embeddings_list.append(valid_embeds)
            masks_list.append(mask[mask])
            token_counts.append(valid_embeds.shape[0])

            logger.debug(f"[TransformersBackend] Blended embedding [{i}]: shape={valid_embeds.shape}")
            logger.debug(f"[TransformersBackend] Blended embedding [{i}] stats: min={valid_embeds.min().item():.4f}, max={valid_embeds.max().item():.4f}, mean={valid_embeds.mean().item():.4f}")

        result = EncodingOutput(
            embeddings=embeddings_list,
            attention_masks=masks_list,
            token_counts=token_counts,
        )

        if return_padded:
            result.padded_embeddings = blended_hidden
            result.padded_mask = attention_mask

        return result

    def to(self, device: torch.device) -> "TransformersBackend":
        """Move model to device."""
        self.model = self.model.to(device)
        self._device = device
        return self

    @property
    def supports_generation(self) -> bool:
        """Whether this backend supports text generation."""
        return self._supports_generation

    def generate(
        self,
        prompt: str,
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
        Generate text using the loaded model.

        This method enables using the same Qwen3 model for prompt rewriting
        in addition to embedding extraction.

        Qwen3 Best Practices (thinking mode):
        - temperature=0.6, top_p=0.95, top_k=20, min_p=0 (default)
        - DO NOT use greedy decoding (causes repetition)
        - presence_penalty=0-2 helps reduce endless repetitions

        Args:
            prompt: User prompt/message
            system_prompt: Optional system prompt (rewriter template content)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (Qwen3 thinking: 0.6)
            top_p: Nucleus sampling threshold (Qwen3 thinking: 0.95)
            top_k: Top-k sampling (Qwen3: 20)
            min_p: Minimum probability threshold (Qwen3: 0.0)
            presence_penalty: Penalty for token presence (0-2, helps reduce repetition)
            do_sample: Whether to use sampling (False = greedy, NOT recommended for Qwen3)

        Returns:
            Generated text (assistant response only, no special tokens)

        Example:
            backend = TransformersBackend.from_pretrained(...)
            rewritten = backend.generate(
                prompt="A cat sleeping",
                system_prompt="You are an expert at writing image prompts...",
                temperature=0.6,  # Qwen3 thinking mode
                top_k=20,
            )
        """
        # Build messages for chat template
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template with generation prompt
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        logger.debug(f"[TransformersBackend.generate] Formatted prompt length: {len(formatted)} chars")

        # Tokenize
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.device)

        input_length = inputs.input_ids.shape[1]
        logger.debug(f"[TransformersBackend.generate] Input tokens: {input_length}")

        # Get proper termination tokens for Qwen3
        # Qwen3 uses multiple stop tokens: <|im_end|> (151645), <|endoftext|> (151643)
        eos_token_ids = []
        if self.tokenizer.eos_token_id is not None:
            eos_token_ids.append(self.tokenizer.eos_token_id)

        # Add Qwen3-specific stop tokens
        for stop_token in ["<|im_end|>", "<|endoftext|>"]:
            try:
                token_id = self.tokenizer.convert_tokens_to_ids(stop_token)
                if token_id is not None and token_id not in eos_token_ids:
                    eos_token_ids.append(token_id)
            except Exception:
                pass

        # Use single token or list
        eos_token_id = eos_token_ids if len(eos_token_ids) > 1 else (eos_token_ids[0] if eos_token_ids else None)
        pad_token_id = self.tokenizer.pad_token_id or (eos_token_ids[0] if eos_token_ids else 0)

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
                # presence_penalty is additive (0-2 range), convert approximately
                # presence_penalty of 1.5 roughly maps to repetition_penalty of ~1.2
                gen_kwargs["repetition_penalty"] = 1.0 + (presence_penalty * 0.15)
        else:
            gen_kwargs["do_sample"] = False

        logger.debug(f"[TransformersBackend.generate] Generation kwargs: {gen_kwargs}")
        logger.debug(f"[TransformersBackend.generate] eos_token_ids: {eos_token_ids}")
        logger.info(f"[TransformersBackend.generate] Starting generation (max_new_tokens={max_new_tokens})...")

        # Generate - use use_cache=False to avoid KV cache issues after encoding
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
                use_cache=True,  # Keep generation fast
            )

        logger.info(f"[TransformersBackend.generate] Generation completed")

        # Decode only the generated part (skip input tokens)
        generated_ids = outputs[0, input_length:].tolist()

        # Parse thinking content at token level (most reliable)
        # Qwen3 uses token 151667 for <think> and 151668 for </think>
        THINK_START_TOKEN = 151667
        THINK_END_TOKEN = 151668
        thinking_content = None
        content_ids = generated_ids

        try:
            # Find </think> token from the end (in case of multiple)
            think_end_idx = len(generated_ids) - generated_ids[::-1].index(THINK_END_TOKEN)
            # Everything before </think> is thinking (may include <think> token at start)
            thinking_ids = generated_ids[:think_end_idx]
            content_ids = generated_ids[think_end_idx:]

            # Remove <think> token from start if present
            if thinking_ids and thinking_ids[0] == THINK_START_TOKEN:
                thinking_ids = thinking_ids[1:]

            # Decode thinking (skip special tokens to clean it up)
            thinking_content = self.tokenizer.decode(thinking_ids, skip_special_tokens=True).strip()

            # Extra safety: strip any remaining <think>/<think> tags that might have slipped through
            # (in case these tokens aren't marked as special in the tokenizer)
            thinking_content = thinking_content.removeprefix("<think>").removesuffix("</think>").strip()

            logger.info(f"[TransformersBackend.generate] Extracted thinking via token parsing ({len(thinking_content)} chars)")
        except ValueError:
            # No </think> token found - model didn't use thinking format
            logger.debug("[TransformersBackend.generate] No </think> token found, using full output")

        # Decode the content (after </think> or full output if no thinking)
        # Don't skip special tokens initially so we can clean up properly
        generated_text = self.tokenizer.decode(content_ids, skip_special_tokens=False)

        # Clean up end tokens
        for end_token in ["<|im_end|>", "<|endoftext|>"]:
            if generated_text.endswith(end_token):
                generated_text = generated_text[:-len(end_token)]

        generated_text = generated_text.strip()

        # If we extracted thinking, prepend it with <think> tags for downstream parsing
        # This maintains compatibility with text-level parsing in the server
        if thinking_content:
            generated_text = f"<think>\n{thinking_content}\n</think>\n\n{generated_text}"

        logger.debug(f"[TransformersBackend.generate] Generated {len(generated_ids)} tokens")

        return generated_text

    # Cache management methods

    @property
    def cache(self) -> Optional[EmbeddingCache]:
        """Get the embedding cache, if configured."""
        return self._cache

    @property
    def cache_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self._cache is not None and self._cache.enabled

    def set_cache(self, cache: Optional[EmbeddingCache]) -> None:
        """
        Set or replace the embedding cache.

        Args:
            cache: EmbeddingCache instance, or None to disable caching
        """
        self._cache = cache
        if cache is not None:
            logger.info(f"Embedding cache set (max_size={cache.max_size})")
        else:
            logger.info("Embedding cache disabled")

    def enable_cache(self, max_size: int = 100) -> EmbeddingCache:
        """
        Enable embedding caching.

        Args:
            max_size: Maximum number of cached embeddings

        Returns:
            The newly created cache
        """
        self._cache = EmbeddingCache(max_size=max_size, enabled=True)
        logger.info(f"Embedding cache enabled (max_size={max_size})")
        return self._cache

    def disable_cache(self) -> None:
        """Disable embedding caching."""
        if self._cache is not None:
            self._cache.enabled = False
            logger.info("Embedding cache disabled")

    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        if self._cache is not None:
            self._cache.clear()

    @property
    def cache_stats(self) -> Optional[CacheStats]:
        """Get cache statistics, if cache is enabled."""
        if self._cache is not None:
            return self._cache.stats
        return None
