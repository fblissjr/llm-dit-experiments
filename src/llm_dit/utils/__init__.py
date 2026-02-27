"""Utility modules for llm-dit-experiments."""

from llm_dit.utils.lora import (
    LoRALoader,
    LoRAFusionRecord,
    FusedLoRAState,
    get_fused_state,
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

from llm_dit.utils.dype import (
    DyPEConfig,
    DyPEPosEmbed,
    ZImageDyPERoPE,
    compute_dype_shift,
    compute_k_t,
    compute_mscale,
    axis_token_span,
    patch_zimage_rope,
    set_zimage_timestep,
)

from llm_dit.utils.vision_yarn import (
    get_1d_vision_yarn_pos_embed,
    get_1d_yarn_pos_embed,
    get_1d_ntk_pos_embed,
    find_correction_range,
    linear_ramp_mask,
    find_newbase_ntk,
)

from llm_dit.utils.logging_config import (
    setup_logging,
    JSONFormatter,
)

from llm_dit.utils.prompt_rewriter import (
    PromptRewriter,
    detect_language,
    create_rewriter_from_config,
    ENGLISH_SYSTEM_PROMPT,
    CHINESE_SYSTEM_PROMPT,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NEGATIVE_PROMPT_EN,
)

from llm_dit.utils.cfg import (
    CFGNormMode,
    apply_cfg_normalization,
    apply_cfg_truncation,
    get_cfg_scale_with_truncation,
    calculate_dynamic_shift,
    calculate_dynamic_shift_simple,
)

from llm_dit.utils.vae_ops import (
    encode_image_to_latents,
    prepare_differential_masks,
    blend_differential_latents,
    scale_noise_for_timestep,
    get_vae_scale_factor,
)

from llm_dit.utils.memory import (
    cleanup_memory,
    get_gpu_memory,
    get_gpu_memory_reserved,
    get_gpu_memory_stats,
    log_memory_usage,
    reset_peak_memory_stats,
    estimate_vram_usage,
    estimate_ltx2_vram,
    MemoryTracker,
)

from llm_dit.utils.gpu_memory import (
    cleanup_gpu_memory,
    free_gpu_memory_context,
    free_gpu_memory_after,
    verify_pure_pytorch,
    staged_gpu_loading,
    FreeGPUMemoryContext,
    get_peak_memory_gb,
    reset_peak_memory,
)

from llm_dit.utils.progress import (
    SamplingProgress,
    StepTracker,
    StageProgress,
    create_denoising_callback,
)

from llm_dit.utils.metrics import (
    SigLIPScorer,
    compute_siglip_score,
    compute_video_siglip_score,
)

from llm_dit.utils.availability import (
    is_torchao_available,
    is_flash_attn_available,
    is_diffusers_available,
    is_xformers_available,
    is_sage_attn_available,
    get_diffusers_version,
    get_flash_attn_version,
    get_cuda_capability,
    check_diffusers_version,
    check_fp8_support,
    log_availability_status,
)

__all__ = [
    # LoRA
    "LoRALoader",
    "LoRAFusionRecord",
    "FusedLoRAState",
    "get_fused_state",
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
    # DyPE (Dynamic Position Extrapolation)
    "DyPEConfig",
    "DyPEPosEmbed",
    "compute_dype_shift",
    "compute_k_t",
    "compute_mscale",
    "axis_token_span",
    # Vision YaRN
    "get_1d_vision_yarn_pos_embed",
    "get_1d_yarn_pos_embed",
    "get_1d_ntk_pos_embed",
    "find_correction_range",
    "linear_ramp_mask",
    "find_newbase_ntk",
    # Logging
    "setup_logging",
    "JSONFormatter",
    # Prompt rewriting
    "PromptRewriter",
    "detect_language",
    "create_rewriter_from_config",
    "ENGLISH_SYSTEM_PROMPT",
    "CHINESE_SYSTEM_PROMPT",
    "DEFAULT_NEGATIVE_PROMPT",
    "DEFAULT_NEGATIVE_PROMPT_EN",
    # CFG utilities
    "CFGNormMode",
    "apply_cfg_normalization",
    "apply_cfg_truncation",
    "get_cfg_scale_with_truncation",
    "calculate_dynamic_shift",
    "calculate_dynamic_shift_simple",
    # VAE operations
    "encode_image_to_latents",
    "prepare_differential_masks",
    "blend_differential_latents",
    "scale_noise_for_timestep",
    "get_vae_scale_factor",
    # Memory utilities
    "cleanup_memory",
    "get_gpu_memory",
    "get_gpu_memory_reserved",
    "get_gpu_memory_stats",
    "log_memory_usage",
    "reset_peak_memory_stats",
    "estimate_vram_usage",
    "estimate_ltx2_vram",
    "MemoryTracker",
    # GPU Memory (pure PyTorch)
    "cleanup_gpu_memory",
    "free_gpu_memory_context",
    "free_gpu_memory_after",
    "verify_pure_pytorch",
    "staged_gpu_loading",
    "FreeGPUMemoryContext",
    "get_peak_memory_gb",
    "reset_peak_memory",
    # Progress tracking
    "SamplingProgress",
    "StepTracker",
    "StageProgress",
    "create_denoising_callback",
    # Metrics
    "SigLIPScorer",
    "compute_siglip_score",
    "compute_video_siglip_score",
    # Availability checks
    "is_torchao_available",
    "is_flash_attn_available",
    "is_diffusers_available",
    "is_xformers_available",
    "is_sage_attn_available",
    "get_diffusers_version",
    "get_flash_attn_version",
    "get_cuda_capability",
    "check_diffusers_version",
    "check_fp8_support",
    "log_availability_status",
]
