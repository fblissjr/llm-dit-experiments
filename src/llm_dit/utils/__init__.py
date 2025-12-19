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

from llm_dit.utils.attention import (
    AttentionBackend,
    get_available_backends,
    get_attention_backend,
    set_attention_backend,
    reset_attention_backend,
    attention_forward,
    log_attention_info,
)

from llm_dit.utils.tiled_vae import (
    TiledVAEDecoder,
    decode_latents,
    estimate_vae_memory,
)

from llm_dit.utils.embedding_cache import (
    EmbeddingCache,
    CacheStats,
    get_embedding_cache,
    set_embedding_cache,
    clear_embedding_cache,
)

from llm_dit.utils.long_prompt import (
    LongPromptMode,
    compress_embeddings,
    estimate_quality_loss,
)

from llm_dit.utils.latent_packing import (
    pack_latents_2x2,
    unpack_latents_2x2,
    pack_multi_layer_latents,
    unpack_multi_layer_latents,
    compute_packed_sequence_length,
    get_img_shapes_for_rope,
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
    # Attention
    "AttentionBackend",
    "get_available_backends",
    "get_attention_backend",
    "set_attention_backend",
    "reset_attention_backend",
    "attention_forward",
    "log_attention_info",
    # Tiled VAE
    "TiledVAEDecoder",
    "decode_latents",
    "estimate_vae_memory",
    # Embedding cache
    "EmbeddingCache",
    "CacheStats",
    "get_embedding_cache",
    "set_embedding_cache",
    "clear_embedding_cache",
    # Long prompt handling
    "LongPromptMode",
    "compress_embeddings",
    "estimate_quality_loss",
    # Latent packing (Qwen-Image)
    "pack_latents_2x2",
    "unpack_latents_2x2",
    "pack_multi_layer_latents",
    "unpack_multi_layer_latents",
    "compute_packed_sequence_length",
    "get_img_shapes_for_rope",
]
