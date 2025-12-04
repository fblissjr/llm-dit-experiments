"""Utility modules for llm-dit-experiments."""

from llm_dit.utils.lora import (
    LoRALoader,
    load_lora,
    clear_lora,
    fuse_lora,
)

from llm_dit.utils.embeddings import (
    EmbeddingStats,
    compute_stats,
    compute_cosine_similarity,
    compute_mse,
    extract_steering_vector,
    apply_steering,
    save_embeddings,
    load_embeddings,
    reduce_embeddings,
    prepare_for_visualization,
)

from llm_dit.utils.model_compat import (
    CompatibilityResult,
    validate_model_config,
    validate_model_path,
    check_compatibility,
    ZIMAGE_REFERENCE,
)

__all__ = [
    # LoRA
    "LoRALoader",
    "load_lora",
    "clear_lora",
    "fuse_lora",
    # Embeddings
    "EmbeddingStats",
    "compute_stats",
    "compute_cosine_similarity",
    "compute_mse",
    "extract_steering_vector",
    "apply_steering",
    "save_embeddings",
    "load_embeddings",
    "reduce_embeddings",
    "prepare_for_visualization",
    # Model compatibility
    "CompatibilityResult",
    "validate_model_config",
    "validate_model_path",
    "check_compatibility",
    "ZIMAGE_REFERENCE",
]
