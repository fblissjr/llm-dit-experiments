"""
Distributed inference support for split LLM/DiT execution.

This module enables running the text encoder (LLM) on one machine
and the DiT/VAE pipeline on another, useful when:
- Mac has fast MPS for LLM but limited VRAM for DiT
- CUDA server has GPU memory for DiT but slow LLM loading

Workflow:
1. Mac: Run `encode_and_save()` to generate embeddings
2. CUDA: Run `load_and_generate()` to generate images from embeddings
"""

from llm_dit.distributed.embeddings import (
    encode_and_save,
    load_embeddings,
    save_embeddings,
    EmbeddingFile,
)

__all__ = [
    "encode_and_save",
    "load_embeddings",
    "save_embeddings",
    "EmbeddingFile",
]
