"""
LLM backend abstraction layer.

Provides a Protocol-based interface for pluggable text encoders,
starting with transformers and designed for vLLM, SGLang, mlx.
"""

from llm_dit.backends.protocol import TextEncoderBackend, EncodingOutput
from llm_dit.backends.config import BackendConfig

__all__ = ["TextEncoderBackend", "EncodingOutput", "BackendConfig"]
