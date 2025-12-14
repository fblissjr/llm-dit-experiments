"""
API-based text encoder backend for remote LLM inference.

Connects to an OpenAI-compatible embeddings API that returns raw hidden states
(not pooled embeddings). This enables distributed inference where the LLM runs
on a different machine than the DiT.

For Z-Image, we need hidden_states[-2] from Qwen3-4B with NO pooling.

This can work with:
1. Custom heylookitsanllm endpoint with raw hidden states support
2. vLLM serving endpoint with hidden states output
3. Any API that returns [seq_len, embed_dim] tensors

Example:
    backend = APIBackend(
        base_url="http://localhost:8000",
        model_id="qwen3-4b",
    )
    output = backend.encode(["A beautiful sunset"])
"""

import base64
import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from llm_dit.backends.protocol import EncodingOutput

logger = logging.getLogger(__name__)


@dataclass
class APIBackendConfig:
    """Configuration for API backend."""

    base_url: str = "http://localhost:8000"
    model_id: str = "qwen3-4b"
    api_key: str | None = None
    timeout: float = 60.0
    max_length: int = 512
    # Z-Image specific: which hidden layer to use
    hidden_layer: int = -2  # Second to last layer
    # Whether the API returns raw hidden states or pooled
    returns_raw_hidden_states: bool = True
    # Request base64 encoding for smaller, faster responses
    encoding_format: str = "base64"  # "float" or "base64"


class APIBackend:
    """
    API-based text encoder backend.

    Connects to a remote LLM server that exposes hidden states via API.
    This enables running the LLM on a different machine than the DiT.

    For Z-Image, we need:
    - hidden_states[-2] (second-to-last layer)
    - Raw sequence embeddings (no pooling)
    - Shape: [seq_len, 2560]
    """

    def __init__(self, config: APIBackendConfig):
        """Initialize API backend."""
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx required for API backend. Install with: pip install httpx")

        self.config = config
        self._client: httpx.Client | None = None
        self._embedding_dim = 2560  # Qwen3-4B hidden size
        self._device = torch.device("cpu")  # API returns CPU tensors
        self._dtype = torch.bfloat16

    @classmethod
    def from_url(
        cls,
        base_url: str,
        model_id: str = "qwen3-4b",
        api_key: str | None = None,
        **kwargs,
    ) -> "APIBackend":
        """Create backend from URL."""
        config = APIBackendConfig(
            base_url=base_url,
            model_id=model_id,
            api_key=api_key,
            **kwargs,
        )
        return cls(config)

    @property
    def client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            self._client = httpx.Client(
                base_url=self.config.base_url,
                headers=headers,
                timeout=self.config.timeout,
            )
        return self._client

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension (2560 for Qwen3-4B)."""
        return self._embedding_dim

    @property
    def max_sequence_length(self) -> int:
        """Return max sequence length."""
        return self.config.max_length

    @property
    def device(self) -> torch.device:
        """Return device (always CPU for API backend)."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Return dtype."""
        return self._dtype

    def encode(
        self,
        texts: List[str],
        return_padded: bool = False,
    ) -> EncodingOutput:
        """
        Encode texts via API.

        Args:
            texts: List of formatted prompt strings
            return_padded: Also return padded batch tensors

        Returns:
            EncodingOutput with variable-length embeddings
        """
        embeddings_list = []
        attention_masks_list = []

        for text in texts:
            embedding = self._encode_single(text)
            embeddings_list.append(embedding)
            # Create attention mask (all 1s since API returns only valid tokens)
            attention_masks_list.append(torch.ones(embedding.shape[0], dtype=torch.bool))

        # Build output
        if return_padded:
            # Pad to max length
            max_len = max(e.shape[0] for e in embeddings_list)
            padded = torch.zeros(
                len(embeddings_list), max_len, self.embedding_dim, dtype=self._dtype
            )
            padded_mask = torch.zeros(len(embeddings_list), max_len, dtype=torch.bool)

            for i, emb in enumerate(embeddings_list):
                seq_len = emb.shape[0]
                padded[i, :seq_len] = emb
                padded_mask[i, :seq_len] = True

            return EncodingOutput(
                embeddings=embeddings_list,
                attention_masks=attention_masks_list,
                padded_embeddings=padded,
                padded_mask=padded_mask,
            )

        return EncodingOutput(
            embeddings=embeddings_list,
            attention_masks=attention_masks_list,
        )

    def _encode_single(self, text: str) -> torch.Tensor:
        """Encode a single text via API."""
        if self.config.returns_raw_hidden_states:
            return self._encode_raw_hidden_states(text)
        else:
            return self._encode_pooled(text)

    def _encode_raw_hidden_states(self, text: str) -> torch.Tensor:
        """
        Request raw hidden states from API.

        Expects API endpoint: POST /v1/hidden_states
        Request: {"input": str, "model": str, "layer": int, "encoding_format": str}
        Response: {"hidden_states": [[float, ...], ...] or base64_str, "shape": [seq_len, dim]}
        """
        try:
            url = f"{self.config.base_url}/v1/hidden_states"
            logger.info(f"[APIBackend] Requesting hidden states from: {url}")
            logger.info(f"[APIBackend]   model: {self.config.model_id}")
            logger.info(f"[APIBackend]   layer: {self.config.hidden_layer}")
            logger.info(f"[APIBackend]   max_length: {self.config.max_length}")
            logger.info(f"[APIBackend]   encoding_format: {self.config.encoding_format}")
            logger.info(f"[APIBackend]   text length: {len(text)} chars")
            logger.info(f"[APIBackend]   text preview: {text[:100]}...")

            response = self.client.post(
                "/v1/hidden_states",
                json={
                    "input": text,
                    "model": self.config.model_id,
                    "layer": self.config.hidden_layer,
                    "max_length": self.config.max_length,
                    "encoding_format": self.config.encoding_format,
                },
            )
            response.raise_for_status()
            data = response.json()

            logger.info(f"[APIBackend] Response received: encoding_format={data.get('encoding_format')}, shape={data.get('shape')}")

            # Handle base64 or float encoding
            if data.get("encoding_format") == "base64":
                tensor = self._decode_base64_hidden_states(
                    data["hidden_states"],
                    data["shape"],
                )
            else:
                # Standard float array
                hidden_states = data["hidden_states"]
                tensor = torch.tensor(hidden_states, dtype=self._dtype)

            logger.info(f"[APIBackend] Decoded tensor: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")

            # Debug: log embedding stats for comparison with local backends
            logger.debug(f"[APIBackend] Embedding stats: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}")
            logger.debug(f"[APIBackend] Embedding first 5 values: {tensor[0, :5].tolist()}")
            logger.debug(f"[APIBackend] Embedding last 5 values: {tensor[-1, -5:].tolist()}")

            return tensor

        except httpx.HTTPStatusError as e:
            logger.error(f"[APIBackend] HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"[APIBackend] Failed to encode via API: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _decode_base64_hidden_states(
        self,
        data: str,
        shape: list[int],
    ) -> torch.Tensor:
        """Decode base64-encoded hidden states to torch tensor."""
        raw = base64.b64decode(data)
        arr = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()  # copy for writability
        return torch.from_numpy(arr).to(dtype=self._dtype)

    def _encode_pooled(self, text: str) -> torch.Tensor:
        """
        Request pooled embeddings from standard /v1/embeddings endpoint.

        Note: This returns a single vector per text, not a sequence.
        For Z-Image, you should use raw hidden states mode instead.
        """
        logger.warning(
            "Using pooled embeddings from /v1/embeddings. "
            "This may not work correctly for Z-Image which needs sequence embeddings."
        )

        try:
            response = self.client.post(
                "/v1/embeddings",
                json={
                    "input": text,
                    "model": self.config.model_id,
                },
            )
            response.raise_for_status()
            data = response.json()

            # Extract embedding from OpenAI-style response
            embedding = data["data"][0]["embedding"]
            tensor = torch.tensor(embedding, dtype=self._dtype)

            # Pooled embedding is 1D, reshape to [1, dim] to match expected format
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)

            return tensor

        except Exception as e:
            logger.error(f"Failed to get embeddings from API: {e}")
            raise

    def to(self, device: torch.device) -> "APIBackend":
        """API backend always returns CPU tensors (move after receiving)."""
        logger.info(
            f"APIBackend returns CPU tensors. Move embeddings to {device} after receiving."
        )
        return self

    @property
    def supports_generation(self) -> bool:
        """Whether this backend supports text generation."""
        return True  # API backends typically support chat completions

    def generate(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        image: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text via API chat completions endpoint.

        Supports vision-language models when image is provided.

        Qwen3 Best Practices (thinking mode):
        - temperature=0.6, top_p=0.95, top_k=20, min_p=0 (default)
        - DO NOT use greedy decoding (causes repetition)
        - presence_penalty=0-2 helps reduce endless repetitions

        Args:
            prompt: User prompt/message (optional if image provided)
            system_prompt: Optional system prompt (rewriter template content)
            image: Optional base64-encoded image (data:image/...;base64,...)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (Qwen3 thinking: 0.6)
            top_p: Nucleus sampling threshold (Qwen3 thinking: 0.95)
            top_k: Top-k sampling (Qwen3: 20)
            min_p: Minimum probability threshold (Qwen3: 0.0)
            presence_penalty: Penalty for token presence (0-2, helps reduce repetition)
            do_sample: Whether to use sampling (ignored for API, always samples)

        Returns:
            Generated text (assistant response)

        Example:
            backend = APIBackend.from_url("http://localhost:8000", "qwen3-4b")
            rewritten = backend.generate(
                prompt="A cat sleeping",
                system_prompt="You are an expert at writing image prompts...",
                temperature=0.6,  # Qwen3 thinking mode
                top_k=20,
            )

            # With image (VL model)
            backend = APIBackend.from_url("http://localhost:8000", "qwen2.5-vl-72b-mlx")
            description = backend.generate(
                prompt="Describe this image",
                image="data:image/png;base64,...",
            )
        """
        # Validate inputs
        if not prompt and not image:
            raise ValueError("Either prompt or image must be provided")

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Build user content (text, image, or both)
        if image:
            # VL model: use content array with text and image parts
            user_content = []
            if prompt:
                user_content.append({"type": "text", "text": prompt})
            user_content.append({
                "type": "image_url",
                "image_url": {"url": image}
            })
            messages.append({"role": "user", "content": user_content})
        else:
            # Text-only: simple string content
            messages.append({"role": "user", "content": prompt})

        try:
            logger.info(f"[APIBackend.generate] Requesting chat completion from API")
            logger.debug(f"[APIBackend.generate] Messages: {len(messages)}, max_tokens: {max_new_tokens}, has_image: {image is not None}")

            # Build request payload
            payload = {
                "model": self.config.model_id,
                "messages": messages,
                "max_tokens": max_new_tokens,
                "temperature": temperature if do_sample else 0.0,
                "top_p": top_p,
                "top_k": top_k,
            }
            # Only include min_p if non-zero (not all APIs support it)
            if min_p > 0.0:
                payload["min_p"] = min_p
            # Only include presence_penalty if non-zero
            if presence_penalty > 0.0:
                payload["presence_penalty"] = presence_penalty

            response = self.client.post(
                "/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            # Extract assistant message from OpenAI-compatible response
            generated_text = data["choices"][0]["message"]["content"]
            logger.debug(f"[APIBackend.generate] Generated {len(generated_text)} chars")

            return generated_text.strip()

        except httpx.HTTPStatusError as e:
            logger.error(f"[APIBackend.generate] HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"[APIBackend.generate] Failed to generate via API: {e}")
            raise

    def close(self):
        """Close HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
