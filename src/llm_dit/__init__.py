"""
llm-dit-experiments: LLM-DiT experimentation platform with pluggable backends.

Main exports:
- ZImageTextEncoder: High-level text encoder with templates
- ZImagePipeline: End-to-end text-to-image generation
- Conversation, format_prompt: Prompt building utilities
- TransformersBackend: HuggingFace text encoder backend
- FlowMatchScheduler: Pure PyTorch scheduler for Z-Image
- ContextRefiner: Pure PyTorch context refiner module
"""

__version__ = "0.1.0"

# Core conversation and formatting
from llm_dit.conversation import Conversation, format_prompt, Qwen3Formatter

# Backends
from llm_dit.backends import BackendConfig, EncodingOutput, TextEncoderBackend
from llm_dit.backends.transformers import TransformersBackend

# High-level encoders
from llm_dit.encoders import ZImageTextEncoder

# Pipelines
from llm_dit.pipelines import ZImagePipeline, setup_attention_backend, MAX_TEXT_SEQ_LEN

# Schedulers (pure PyTorch)
from llm_dit.schedulers import FlowMatchScheduler

# Models (pure PyTorch components)
from llm_dit.models import ContextRefiner

# Templates
from llm_dit.templates import Template, TemplateRegistry

__all__ = [
    # Version
    "__version__",
    # Conversation
    "Conversation",
    "format_prompt",
    "Qwen3Formatter",
    # Backends
    "BackendConfig",
    "EncodingOutput",
    "TextEncoderBackend",
    "TransformersBackend",
    # Encoders
    "ZImageTextEncoder",
    # Pipelines
    "ZImagePipeline",
    "setup_attention_backend",
    "MAX_TEXT_SEQ_LEN",
    # Schedulers
    "FlowMatchScheduler",
    # Models
    "ContextRefiner",
    # Templates
    "Template",
    "TemplateRegistry",
]
