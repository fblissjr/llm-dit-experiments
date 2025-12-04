"""
Embedding analysis utilities for experimentation.

Provides tools for:
- Computing similarity between embeddings
- Extracting and applying steering vectors
- Saving/loading embeddings with metadata
- Basic statistics and visualization helpers

Usage:
    uv sync --extra analysis  # Install visualization deps
    uv run scripts/embeddings.py --help
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

import torch
from safetensors.torch import save_file, load_file


@dataclass
class EmbeddingStats:
    """Statistics for an embedding tensor."""

    shape: tuple
    dtype: str
    min: float
    max: float
    mean: float
    std: float
    norm: float  # L2 norm of mean vector


def compute_stats(embeddings: torch.Tensor) -> EmbeddingStats:
    """
    Compute basic statistics for embeddings.

    Args:
        embeddings: Tensor of shape [seq_len, hidden_dim] or [batch, seq_len, hidden_dim]

    Returns:
        EmbeddingStats with shape, dtype, and value statistics
    """
    flat = embeddings.float()  # Convert to float for stats
    mean_vec = flat.mean(dim=tuple(range(flat.dim() - 1)))  # Mean over all but last dim

    return EmbeddingStats(
        shape=tuple(embeddings.shape),
        dtype=str(embeddings.dtype),
        min=flat.min().item(),
        max=flat.max().item(),
        mean=flat.mean().item(),
        std=flat.std().item(),
        norm=mean_vec.norm().item(),
    )


def compute_cosine_similarity(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    reduce: str = "mean",
) -> float:
    """
    Compute cosine similarity between two embeddings.

    Args:
        emb1: First embedding [seq_len, hidden_dim] or [hidden_dim]
        emb2: Second embedding [seq_len, hidden_dim] or [hidden_dim]
        reduce: How to reduce sequence dimension ("mean", "max", "last")

    Returns:
        Cosine similarity in range [-1, 1]
    """
    # Reduce to single vector if needed
    if emb1.dim() == 2:
        if reduce == "mean":
            emb1 = emb1.mean(dim=0)
        elif reduce == "max":
            emb1 = emb1.max(dim=0).values
        elif reduce == "last":
            emb1 = emb1[-1]
        else:
            raise ValueError(f"Unknown reduce method: {reduce}")

    if emb2.dim() == 2:
        if reduce == "mean":
            emb2 = emb2.mean(dim=0)
        elif reduce == "max":
            emb2 = emb2.max(dim=0).values
        elif reduce == "last":
            emb2 = emb2[-1]

    # Compute cosine similarity
    emb1 = emb1.float()
    emb2 = emb2.float()

    similarity = torch.nn.functional.cosine_similarity(
        emb1.unsqueeze(0), emb2.unsqueeze(0)
    ).item()

    return similarity


def compute_mse(emb1: torch.Tensor, emb2: torch.Tensor, reduce: str = "mean") -> float:
    """
    Compute mean squared error between two embeddings.

    Args:
        emb1: First embedding [seq_len, hidden_dim] or [hidden_dim]
        emb2: Second embedding [seq_len, hidden_dim] or [hidden_dim]
        reduce: How to reduce sequence dimension ("mean", "max", "last")

    Returns:
        MSE value (lower = more similar)
    """
    # Reduce to single vector if needed
    if emb1.dim() == 2:
        if reduce == "mean":
            emb1 = emb1.mean(dim=0)
        elif reduce == "max":
            emb1 = emb1.max(dim=0).values
        elif reduce == "last":
            emb1 = emb1[-1]

    if emb2.dim() == 2:
        if reduce == "mean":
            emb2 = emb2.mean(dim=0)
        elif reduce == "max":
            emb2 = emb2.max(dim=0).values
        elif reduce == "last":
            emb2 = emb2[-1]

    emb1 = emb1.float()
    emb2 = emb2.float()

    return torch.nn.functional.mse_loss(emb1, emb2).item()


def extract_steering_vector(
    positive_emb: torch.Tensor,
    negative_emb: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Extract a steering vector from paired embeddings.

    The steering vector points from negative toward positive concept.
    Adding this vector to embeddings shifts them toward the positive concept.

    Args:
        positive_emb: Embedding for positive concept [seq_len, hidden_dim]
        negative_emb: Embedding for negative concept [seq_len, hidden_dim]
        normalize: Whether to L2-normalize the resulting vector

    Returns:
        Steering vector [hidden_dim]

    Example:
        # Extract "photorealistic" direction
        photo_emb = encode("A photorealistic cat")
        cartoon_emb = encode("A cartoon cat")
        photo_direction = extract_steering_vector(photo_emb, cartoon_emb)
    """
    # Reduce to mean vectors
    pos_mean = positive_emb.float().mean(dim=0)
    neg_mean = negative_emb.float().mean(dim=0)

    # Compute direction vector
    direction = pos_mean - neg_mean

    if normalize:
        direction = direction / direction.norm()

    return direction


def apply_steering(
    embeddings: torch.Tensor,
    vector: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Apply a steering vector to embeddings.

    Args:
        embeddings: Input embeddings [seq_len, hidden_dim]
        vector: Steering vector [hidden_dim]
        scale: Scaling factor for the steering effect

    Returns:
        Steered embeddings [seq_len, hidden_dim]

    Example:
        steered = apply_steering(base_emb, photo_direction, scale=2.0)
    """
    # Ensure same dtype
    vector = vector.to(embeddings.dtype)

    # Add scaled steering vector to each position
    return embeddings + scale * vector.unsqueeze(0)


def save_embeddings(
    embeddings: torch.Tensor | List[torch.Tensor],
    path: str | Path,
    metadata: Dict[str, Any] | None = None,
    prompts: List[str] | None = None,
) -> Path:
    """
    Save embeddings to a safetensors file with metadata.

    Args:
        embeddings: Single tensor or list of tensors to save
        path: Output file path
        metadata: Optional dict of metadata (strings only, will be JSON-serialized)
        prompts: Optional list of prompt strings (saved as metadata)

    Returns:
        Path to saved file

    Format:
        - Single tensor: saved as "embeddings"
        - List: saved as "embeddings_0", "embeddings_1", etc.
        - Metadata stored in safetensors header
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build tensors dict
    tensors = {}
    if isinstance(embeddings, list):
        for i, emb in enumerate(embeddings):
            tensors[f"embeddings_{i}"] = emb.contiguous()
    else:
        tensors["embeddings"] = embeddings.contiguous()

    # Build metadata dict (all values must be strings)
    meta = {}
    if metadata:
        meta["custom_metadata"] = json.dumps(metadata)
    if prompts:
        meta["prompts"] = json.dumps(prompts)
    meta["count"] = str(len(tensors))

    save_file(tensors, str(path), metadata=meta)
    return path


def load_embeddings(
    path: str | Path,
) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Load embeddings from a safetensors file.

    Args:
        path: Path to safetensors file

    Returns:
        Tuple of (tensors_dict, metadata_dict)

    Note:
        - Single embedding: tensors["embeddings"]
        - Multiple: tensors["embeddings_0"], tensors["embeddings_1"], etc.
        - metadata["prompts"] contains JSON list of prompts if saved
        - metadata["custom_metadata"] contains any custom data
    """
    import json

    from safetensors import safe_open

    path = Path(path)

    tensors = load_file(str(path))

    # Extract metadata
    metadata = {}
    with safe_open(str(path), framework="pt") as f:
        raw_meta = f.metadata() or {}

    if "prompts" in raw_meta:
        metadata["prompts"] = json.loads(raw_meta["prompts"])
    if "custom_metadata" in raw_meta:
        metadata["custom_metadata"] = json.loads(raw_meta["custom_metadata"])
    if "count" in raw_meta:
        metadata["count"] = int(raw_meta["count"])

    return tensors, metadata


def reduce_embeddings(
    embeddings: torch.Tensor,
    method: str = "mean",
) -> torch.Tensor:
    """
    Reduce sequence of embeddings to a single vector.

    Args:
        embeddings: Tensor [seq_len, hidden_dim]
        method: Reduction method ("mean", "max", "last", "first")

    Returns:
        Reduced tensor [hidden_dim]
    """
    if method == "mean":
        return embeddings.mean(dim=0)
    elif method == "max":
        return embeddings.max(dim=0).values
    elif method == "last":
        return embeddings[-1]
    elif method == "first":
        return embeddings[0]
    else:
        raise ValueError(f"Unknown reduction method: {method}")


def prepare_for_visualization(
    embeddings_list: List[torch.Tensor],
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Prepare multiple embeddings for visualization.

    Reduces each embedding to a single vector and stacks them.

    Args:
        embeddings_list: List of embeddings [seq_len_i, hidden_dim]
        reduction: How to reduce each embedding ("mean", "last", etc.)

    Returns:
        Stacked tensor [n_embeddings, hidden_dim] suitable for t-SNE/UMAP
    """
    reduced = [reduce_embeddings(emb, method=reduction) for emb in embeddings_list]
    return torch.stack(reduced)
