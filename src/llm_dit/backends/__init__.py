"""
LLM backend abstraction layer.

Provides a Protocol-based interface for pluggable text encoders:
- TransformersBackend: Local HuggingFace models (default)
- APIBackend: Remote LLM servers (heylookitsanllm, vLLM, etc.)

Future backends: vLLM native, SGLang, mlx-lm
"""

from llm_dit.backends.protocol import TextEncoderBackend, EncodingOutput
from llm_dit.backends.config import BackendConfig

__all__ = [
    "TextEncoderBackend",
    "EncodingOutput",
    "BackendConfig",
]
