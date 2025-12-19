"""
LLM backend abstraction layer.

Provides a Protocol-based interface for pluggable text encoders:
- TransformersBackend: Local HuggingFace models (Qwen3-4B for Z-Image)
- QwenImageTextEncoderBackend: Qwen2.5-VL-7B for Qwen-Image-Layered
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


# Lazy imports for heavy backends (avoid loading torch/transformers unless needed)
def __getattr__(name: str):
    if name == "QwenImageTextEncoderBackend":
        from llm_dit.backends.qwen_image import QwenImageTextEncoderBackend
        return QwenImageTextEncoderBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
