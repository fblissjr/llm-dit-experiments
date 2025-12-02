"""
Embedding save/load utilities for distributed inference.

Enables running LLM encoder on one machine and DiT/VAE on another.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingMetadata:
    """Metadata for saved embeddings."""

    prompt: str
    template: str | None
    enable_thinking: bool
    system_prompt: str | None
    thinking_content: str | None
    sequence_length: int
    embedding_dim: int
    dtype: str
    created_at: str
    encoder_device: str
    model_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingMetadata":
        return cls(**data)


@dataclass
class EmbeddingFile:
    """Container for embeddings with metadata."""

    embeddings: torch.Tensor  # [seq_len, embed_dim]
    metadata: EmbeddingMetadata

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingFile":
        """Load embeddings from file."""
        path = Path(path)

        # Load tensor
        tensors = load_file(path)
        embeddings = tensors["embeddings"]

        # Load metadata from sidecar JSON
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta_dict = json.load(f)
            metadata = EmbeddingMetadata.from_dict(meta_dict)
        else:
            # Minimal metadata if JSON missing
            metadata = EmbeddingMetadata(
                prompt="<unknown>",
                template=None,
                enable_thinking=True,
                system_prompt=None,
                thinking_content=None,
                sequence_length=embeddings.shape[0],
                embedding_dim=embeddings.shape[1],
                dtype=str(embeddings.dtype),
                created_at="<unknown>",
                encoder_device="<unknown>",
                model_path="<unknown>",
            )

        return cls(embeddings=embeddings, metadata=metadata)

    def save(self, path: str | Path) -> None:
        """Save embeddings to file with metadata sidecar."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save tensor as safetensors
        save_file({"embeddings": self.embeddings}, path)

        # Save metadata as JSON sidecar
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)

        logger.info(f"Saved embeddings to {path}")
        logger.info(f"Saved metadata to {meta_path}")


def save_embeddings(
    embeddings: torch.Tensor,
    path: str | Path,
    prompt: str,
    model_path: str = "",
    template: str | None = None,
    enable_thinking: bool = True,
    system_prompt: str | None = None,
    thinking_content: str | None = None,
    encoder_device: str = "",
) -> Path:
    """
    Save embeddings with metadata for distributed inference.

    Args:
        embeddings: Embedding tensor [seq_len, embed_dim]
        path: Output path (.safetensors)
        prompt: Original prompt text
        model_path: Path to model used
        template: Template name if used
        enable_thinking: Whether thinking was enabled
        system_prompt: System prompt if used
        thinking_content: Thinking content if provided
        encoder_device: Device where encoding was done

    Returns:
        Path to saved file
    """
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".safetensors")

    # Move to CPU for saving
    embeddings_cpu = embeddings.cpu()

    metadata = EmbeddingMetadata(
        prompt=prompt,
        template=template,
        enable_thinking=enable_thinking,
        system_prompt=system_prompt,
        thinking_content=thinking_content,
        sequence_length=embeddings_cpu.shape[0],
        embedding_dim=embeddings_cpu.shape[1],
        dtype=str(embeddings_cpu.dtype),
        created_at=datetime.now().isoformat(),
        encoder_device=encoder_device,
        model_path=model_path,
    )

    emb_file = EmbeddingFile(embeddings=embeddings_cpu, metadata=metadata)
    emb_file.save(path)

    return path


def load_embeddings(path: str | Path, device: str | None = None) -> EmbeddingFile:
    """
    Load embeddings from file.

    Args:
        path: Path to .safetensors file
        device: Optional device to move embeddings to

    Returns:
        EmbeddingFile with embeddings and metadata
    """
    emb_file = EmbeddingFile.load(path)

    if device is not None:
        emb_file.embeddings = emb_file.embeddings.to(device)

    logger.info(f"Loaded embeddings: {emb_file.metadata.prompt[:50]}...")
    logger.info(f"  Shape: {emb_file.embeddings.shape}")
    logger.info(f"  Device: {emb_file.embeddings.device}")

    return emb_file


def encode_and_save(
    prompt: str,
    output_path: str | Path,
    model_path: str,
    templates_dir: str | Path | None = None,
    template: str | None = None,
    enable_thinking: bool = True,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Path:
    """
    Encode a prompt and save embeddings for later use.

    This is the "encoder side" of distributed inference.
    Run this on the machine with the LLM (e.g., Mac with MPS).

    Args:
        prompt: Text prompt to encode
        output_path: Where to save embeddings
        model_path: Path to Z-Image model
        templates_dir: Path to templates
        template: Template name to use
        enable_thinking: Whether to enable thinking tags
        torch_dtype: Model dtype

    Returns:
        Path to saved embeddings file

    Example:
        # On Mac (fast LLM)
        path = encode_and_save(
            "A beautiful sunset",
            "embeddings/sunset.safetensors",
            "/path/to/z-image",
            template="photorealistic",
        )

        # Transfer file to CUDA server, then:
        # load_and_generate("embeddings/sunset.safetensors", ...)
    """
    from llm_dit.encoders import ZImageTextEncoder

    logger.info(f"Loading encoder from {model_path}...")
    encoder = ZImageTextEncoder.from_pretrained(
        model_path,
        templates_dir=templates_dir,
        torch_dtype=torch_dtype,
    )

    logger.info(f"Encoding: {prompt[:50]}...")
    output = encoder.encode(
        prompt,
        template=template,
        enable_thinking=enable_thinking,
    )

    embeddings = output.embeddings[0]

    # Get template content if used
    system_prompt = None
    thinking_content = None
    if template and encoder.templates:
        tpl = encoder.get_template(template)
        if tpl:
            system_prompt = tpl.content
            thinking_content = tpl.thinking_content

    return save_embeddings(
        embeddings=embeddings,
        path=output_path,
        prompt=prompt,
        model_path=model_path,
        template=template,
        enable_thinking=enable_thinking,
        system_prompt=system_prompt,
        thinking_content=thinking_content,
        encoder_device=str(encoder.device),
    )
