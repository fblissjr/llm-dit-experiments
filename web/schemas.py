"""Pydantic request/response models for the generation server.

Request models use plain BaseModel (clients send snake_case).
Response models use CamelModel (server returns camelCase JSON via aliases).

All API schemas live here for shared use across routers.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel


# =============================================================================
# CamelCase Response Base
# =============================================================================


class CamelModel(BaseModel):
    """Base for all response models. Serializes fields as camelCase."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


class SuccessResponse(CamelModel):
    """Standard success/failure response."""

    success: bool
    message: str = ""


class SuccessVramResponse(SuccessResponse):
    """Success response with optional VRAM snapshot."""

    vram: Optional[Dict[str, Any]] = None


# =============================================================================
# Response Models -- Priority 1 (frontend-facing)
# =============================================================================


class LoRAInfo(CamelModel):
    """LoRA fusion state for a single LoRA applied to the active model."""

    name: str
    path: str
    scale: float
    layers_updated: int


class GenerationContextResponse(CamelModel):
    """Composite status snapshot for the frontend status bar.

    Aggregates model variant, LoRA state, VRAM, quantization, compile,
    and session state into a single response to avoid multi-endpoint polling.
    """

    uptime_seconds: Optional[int] = None
    profile: str = "default"
    active_pipeline: Optional[str] = None
    pipeline_display_name: Optional[str] = None
    model_variant: Optional[str] = None
    loras: List[LoRAInfo] = Field(default_factory=list)
    lora_summary: Optional[str] = None
    quantization: Dict[str, str] = Field(default_factory=dict)
    compile_enabled: bool = False
    compile_mode: Optional[str] = None
    block_offload: bool = False
    vram_used_gb: Optional[float] = None
    vram_total_gb: Optional[float] = None
    vram_percent: Optional[float] = None
    pending_restart_fields: List[str] = Field(default_factory=list)
    session_modified_fields: List[str] = Field(default_factory=list)
    fmtt_cached: bool = False
    history_count: int = 0


class VRAMStatusResponse(CamelModel):
    """VRAM usage from /api/vram/status (shape matches ModelManager.get_vram_status)."""

    used_mb: float = 0
    total_mb: float = 24576
    free_mb: float = 24576
    utilization_percent: float = 0
    breakdown: List[Dict[str, Any]] = Field(default_factory=list)


class ConfigTag(CamelModel):
    key: str
    label: str
    color: str


class ConfigWarning(CamelModel):
    severity: str  # "error" | "warning"
    message: str


class ModelStatusResponse(CamelModel):
    """Status for a single pipeline model from /api/models/{id}/status."""

    pipeline_id: str
    status: str  # "loaded" | "unloaded"
    components: List[Dict[str, Any]] = Field(default_factory=list)
    total_vram_mb: int = 0
    vram_mb: int = 0
    model_variant: Optional[str] = None
    display_name: Optional[str] = None
    loras: List[LoRAInfo] = Field(default_factory=list)
    lora_summary: Optional[str] = None
    config_tags: List[ConfigTag] = Field(default_factory=list)
    config_warnings: List[ConfigWarning] = Field(default_factory=list)


class LoRAFileInfo(CamelModel):
    """A single LoRA file on disk."""

    path: str
    name: str
    directory: str
    size_mb: float


class LoRAListResponse(CamelModel):
    """Available LoRA files from /api/loras."""

    loras: List[LoRAFileInfo] = Field(default_factory=list)
    directories: List[str] = Field(default_factory=list)
    count: int = 0
    pipeline_id: Optional[str] = None


class ImageGenerationResult(CamelModel):
    """Standardized response for image generation endpoints."""

    id: Union[int, str]
    pipeline_id: str
    output_type: str = "image"
    url: str
    urls: List[str]
    thumbnail_url: str
    seed: int = -1
    generation_time: float = 0.0


class ClearCacheResponse(CamelModel):
    success: bool
    freed_gb: float = 0.0
    message: str = ""


class RestartResponse(CamelModel):
    success: bool
    message: str = ""
    new_profile: Optional[str] = None


class PresetListResponse(CamelModel):
    """Presets for a specific pipeline."""

    presets: List[Dict[str, Any]] = Field(default_factory=list)
    default_preset: str = ""


class AllPresetsResponse(CamelModel):
    """All presets across pipelines."""

    presets: List[Dict[str, Any]] = Field(default_factory=list)


# =============================================================================
# Response Models -- Priority 2 (non-frontend-facing)
# =============================================================================


class HealthResponse(CamelModel):
    status: str = "ok"
    pipeline_loaded: bool = False
    encoder_loaded: bool = False
    encoder_only_mode: bool = False
    qwen_image_available: bool = False


class EncodeResult(CamelModel):
    shape: List[int]
    dtype: str
    encode_time: float
    token_count: int
    prompt: str
    formatted_prompt: Optional[str] = None


class FormatPromptResult(CamelModel):
    formatted_prompt: str
    char_count: int
    token_count: Optional[int] = None
    max_tokens: int = 1504
    prompt: str
    system_prompt: Optional[str] = None
    thinking_content: Optional[str] = None
    assistant_content: Optional[str] = None
    template: Optional[str] = None
    force_think_block: bool = False
    strip_quotes: bool = False


class TemplateInfo(CamelModel):
    name: str
    description: str = ""
    category: str = "general"
    system_prompt: str = ""
    thinking_content: str = ""
    assistant_content: str = ""
    add_think_block: bool = False


class TemplateListResponse(CamelModel):
    templates: List[TemplateInfo] = Field(default_factory=list)


class RewriterInfo(CamelModel):
    name: str
    description: Optional[str] = None


class RewriterListResponse(CamelModel):
    rewriters: List[RewriterInfo] = Field(default_factory=list)


class RewriteResult(CamelModel):
    original_prompt: str
    rewritten_prompt: str
    thinking_content: Optional[str] = None
    rewriter: Optional[str] = None
    backend: str = "local"
    gen_time: float = 0.0


class SaveEmbeddingsResult(CamelModel):
    path: str
    shape: List[int]
    encode_time: float


class SessionConfigResponse(CamelModel):
    values: Dict[str, Any] = Field(default_factory=dict)
    profile: str = "default"
    modified: List[str] = Field(default_factory=list)
    config_file: Optional[str] = None


class SessionConfigUpdateResponse(CamelModel):
    success: bool = True
    updated: List[str] = Field(default_factory=list)
    pending_restart: List[str] = Field(default_factory=list)
    rejected: List[str] = Field(default_factory=list)


class ProfileListResponse(CamelModel):
    profiles: List[str] = Field(default_factory=list)
    current: str = "default"
    config_file: Optional[str] = None
    error: Optional[str] = None


class PipelineStatusResponse(CamelModel):
    """Status for pipeline-specific status endpoints (flux2, ltx2, etc.)."""

    available: bool = False
    loaded: bool = False


class Flux2StatusResponse(PipelineStatusResponse):
    compile_enabled: bool = False
    compile_dynamic: bool = False
    compile_vae_enabled: bool = False
    supported_models: List[str] = Field(default_factory=list)


class LTX2StatusResponse(PipelineStatusResponse):
    vram_used_gb: Optional[float] = None


class QwenImageEditStatusResponse(CamelModel):
    available: bool = False
    edit_model_loaded: bool = False
    edit_model_path: Optional[str] = None
    supports_multi_image: bool = False


class QwenImageT2IStatusResponse(CamelModel):
    available: bool = False
    configured: bool = False
    model_path: Optional[str] = None
    quantize_transformer: Optional[str] = None
    quantize_text_encoder: Optional[str] = None


class QwenImageT2IConfigResponse(CamelModel):
    model_path: Optional[str] = None
    steps: int = 40
    cfg_scale: float = 4.0
    quantize_transformer: Optional[str] = None
    quantize_text_encoder: Optional[str] = None
    default_width: int = 1024
    default_height: int = 1024
    max_sequence_length: int = 512


class HistoryResponse(CamelModel):
    history: List[Dict[str, Any]] = Field(default_factory=list)


class HistoryDeleteResponse(CamelModel):
    deleted: Dict[str, Any] = Field(default_factory=dict)
    remaining: int = 0


class HistoryClearResponse(CamelModel):
    cleared: int = 0


class UnloadFmttResponse(CamelModel):
    success: bool
    message: str = ""
    free_gb: Optional[float] = None


class DyPEConfigResponse(CamelModel):
    enabled: bool = False
    method: str = "vision_yarn"
    dype_scale: float = 2.0
    dype_exponent: float = 2.0
    dype_start_sigma: float = 1.0
    base_shift: float = 0.5
    max_shift: float = 1.15
    base_resolution: int = 1024
    anisotropic: bool = False
    multipass_recommended_threshold: int = 2048


class DyPEStatusResponse(CamelModel):
    """DyPE feature availability and recommendations."""

    available: bool = False
    supported_methods: List[str] = Field(default_factory=list)
    recommended_for_resolutions: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)


class ParamSchemaResponse(CamelModel):
    """A single UI control definition from the pipeline schema system.

    Uses extra="allow" because ParamSchema.to_dict() dynamically filters
    None values -- the set of keys varies per param.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="allow",
    )

    id: str
    type: str
    label: str
    default: Optional[Any] = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[str]] = None
    options_endpoint: Optional[str] = None
    group: str = "basic"
    tooltip: Optional[str] = None
    conditional: Optional[Dict[str, Any]] = None
    placeholder: Optional[str] = None
    rows: Optional[int] = None
    required: bool = False
    max_count: Optional[int] = None
    scale_min: Optional[float] = None
    scale_max: Optional[float] = None
    dependent_defaults: Optional[Dict[str, Dict[str, Any]]] = None


class PipelineSchemaResponse(CamelModel):
    """Complete pipeline schema for frontend form generation."""

    id: str
    name: str
    description: str
    output_type: str
    color: str
    icon: Optional[str] = None
    params: List[ParamSchemaResponse] = Field(default_factory=list)
    supports_history: bool = True
    supports_img2img: bool = False
    supports_reference_images: bool = False
    supports_streaming: bool = False
    endpoint: Optional[str] = None
    category: str = "image"


class PresetDetailResponse(CamelModel):
    """Full detail for a single generation preset."""

    name: str
    description: str = ""
    category: str = ""
    pipelines: List[str] = Field(default_factory=list)
    variant: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class PipelinesResponse(CamelModel):
    """All pipeline schemas for frontend form generation."""

    pipelines: Dict[str, Any] = Field(default_factory=dict)
    defaults: Dict[str, Any] = Field(default_factory=dict)
    loaded_pipeline: Optional[str] = None


class PipelineDefaultsResponse(CamelModel):
    """Merged schema + config defaults for a specific pipeline.

    Keys are dynamic (parameter IDs vary by pipeline), so we use a
    Dict[str, Any] with an extra _variant field for conditional visibility.
    """

    model_config = ConfigDict(extra="allow")


class ResolutionConfigResponse(CamelModel):
    """Resolution constraints for client-side validation."""

    current_model: Optional[str] = None
    model_constraints: Dict[str, Any] = Field(default_factory=dict)
    active_constraints: Dict[str, Any] = Field(default_factory=dict)
    vae_multiple: int = 16
    vae_scale_factor: int = 8
    min_resolution: int = 256
    max_resolution: int = 4096
    default_resolution: int = 1024
    default_width: int = 1024
    default_height: int = 1024
    dype_base_resolution: int = 1024
    aspect_ratios: Dict[str, Any] = Field(default_factory=dict)
    presets: List[Dict[str, Any]] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    supports_dype: bool = False
    supports_slg: bool = False
    supports_fmtt: bool = False


class RewriterConfigResponse(CamelModel):
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    max_tokens: int
    use_api: bool
    models: List[Dict[str, Any]] = Field(default_factory=list)
    default_model: str = "qwen3-4b"


# =============================================================================
# Request Models (unchanged, plain BaseModel)
# =============================================================================


class DyPEConfigRequest(BaseModel):
    """DyPE configuration for high-resolution generation."""

    enabled: bool = False
    method: str = "vision_yarn"  # vision_yarn, yarn, ntk
    multipass: str = "twopass"  # single, twopass, threepass
    dype_scale: float = 2.0  # Magnitude of DyPE effect
    dype_exponent: float = 2.0  # Decay speed (2.0 = quadratic)
    base_shift: float = 0.5  # Shift at base resolution
    max_shift: float = 1.15  # Shift at max resolution
    pass2_strength: float = 0.5  # img2img strength for pass 2
    pass3_strength: float = 0.4  # img2img strength for pass 3
    frequency_modulation: bool = False  # Timestep-based RoPE frequency scaling (experimental)


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    thinking_content: Optional[str] = None
    assistant_content: Optional[str] = None
    force_think_block: bool = False
    strip_quotes: bool = False
    width: int = 1024
    height: int = 1024
    steps: int = 9
    seed: Optional[int] = None
    template: Optional[str] = None
    guidance_scale: float = 0.0
    cfg_normalization: float = 0.0
    cfg_truncation: float = 1.0
    shift: float = 3.0
    dynamic_shift: bool = False
    d_noise: float = 1.0
    long_prompt_mode: str = "interpolate"
    hidden_layer: int = -2
    layer_weights: Optional[Dict[int, float]] = None
    dype: Optional[DyPEConfigRequest] = None
    slg_scale: Optional[float] = None
    slg_layers: Optional[List[int]] = None
    slg_start: Optional[float] = None
    slg_stop: Optional[float] = None
    fmtt_enabled: bool = False
    fmtt_scale: Optional[float] = None
    fmtt_start: Optional[float] = None
    fmtt_stop: Optional[float] = None
    fmtt_normalize: Optional[str] = None
    fmtt_decode_scale: Optional[float] = None
    fmtt_siglip_model: Optional[str] = None
    fmtt_siglip_device: Optional[str] = None
    fbcache: bool = False
    fbcache_threshold: Optional[float] = None
    fbcache_log: bool = False


class Img2ImgRequest(BaseModel):
    """Request for image-to-image generation with optional differential mask."""

    prompt: str
    negative_prompt: Optional[str] = None
    image: str
    mask_image: Optional[str] = None
    strength: float = Field(
        0.75, ge=0.0, le=1.0, description="Denoising strength (0=no change, 1=full generation)"
    )
    system_prompt: Optional[str] = None
    thinking_content: Optional[str] = None
    assistant_content: Optional[str] = None
    force_think_block: bool = False
    strip_quotes: bool = False
    width: Optional[int] = Field(None, ge=64, le=4096)
    height: Optional[int] = Field(None, ge=64, le=4096)
    steps: int = Field(9, ge=1, le=500)
    seed: Optional[int] = None
    template: Optional[str] = None
    guidance_scale: float = Field(0.0, ge=0.0, le=30.0)
    cfg_normalization: float = Field(0.0, ge=0.0, le=10.0)
    cfg_truncation: float = Field(1.0, ge=0.0, le=1.0)
    cfg_norm_mode: str = "clamp"
    shift: float = Field(3.0, ge=0.0, le=10.0)
    dynamic_shift: bool = False
    d_noise: float = Field(1.0, ge=0.5, le=2.0)
    long_prompt_mode: str = "interpolate"
    hidden_layer: int = Field(-2, ge=-35, le=-1)
    fbcache: bool = False
    fbcache_threshold: Optional[float] = None
    fbcache_log: bool = False


class EncodeRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    thinking_content: Optional[str] = None
    assistant_content: Optional[str] = None
    force_think_block: bool = False
    strip_quotes: bool = False
    template: Optional[str] = None


class RewriteRequest(BaseModel):
    prompt: Optional[str] = None
    rewriter: Optional[str] = None
    custom_system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    model: str = "qwen3-4b"
    image: Optional[str] = None


class QwenImageEditLayerRequest(BaseModel):
    """Request for Qwen-Image layer editing (single image)."""

    layer_image: str
    instruction: str
    steps: int = 40
    cfg_scale: float = 4.0
    seed: Optional[int] = None


class QwenImageEditMultiRequest(BaseModel):
    """Request for Qwen-Image multi-image editing (2511 feature)."""

    images: List[str]
    instruction: str
    steps: int = 40
    cfg_scale: float = 4.0
    seed: Optional[int] = None


class QwenImage2512GenerateRequest(BaseModel):
    """Request for Qwen-Image T2I text-to-image generation."""

    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 1024
    height: int = 1024
    steps: int = 40
    cfg_scale: float = 4.0
    seed: Optional[int] = None
    max_sequence_length: int = 512


class LTX2GenerateRequest(BaseModel):
    """Request for LTX-2 video generation."""

    prompt: str
    negative_prompt: str = "worst quality, blurry, distorted, inconsistent motion"
    width: int = Field(768, ge=256, le=1280)
    height: int = Field(512, ge=256, le=1280)

    @field_validator("width", "height")
    @classmethod
    def snap_to_32(cls, v: int) -> int:
        """Snap to nearest multiple of 32 (LTX-2 VAE requirement)."""
        snapped = round(v / 32) * 32
        return max(256, min(1280, snapped))
    num_frames: int = 33
    fps: float = 24.0
    num_inference_steps: int = 12
    guidance_scale: float = 3.5
    seed: Optional[int] = None
    enable_audio: bool = False
    lora_path: Optional[str] = None
    lora_scale: Optional[float] = None


class Flux2GenerateRequest(BaseModel):
    """Request for FLUX.2 Klein image generation."""

    prompt: str
    model_name: str = "klein-9b-fp8"
    width: int = Field(1024, ge=256, le=2048)
    height: int = Field(1024, ge=256, le=2048)

    @field_validator("width", "height")
    @classmethod
    def snap_to_16(cls, v: int) -> int:
        """Snap to nearest multiple of 16 (FLUX.2 VAE requirement)."""
        snapped = round(v / 16) * 16
        return max(256, min(2048, snapped))
    num_steps: Optional[int] = None
    guidance: Optional[float] = None
    seed: Optional[int] = None
    block_offload: bool = False
    model_path: Optional[str] = None
    vae_path: Optional[str] = None
    reference_images: Optional[List[str]] = None
    match_image_size: Optional[str] = "none"
    loras: Optional[List[str]] = None
    max_text_length: int = 512
    pad_to_max: bool = True
    output_layers: Optional[List[int]] = None

    @field_validator("output_layers")
    @classmethod
    def validate_output_layers(cls, v):
        if v is not None:
            if len(v) != 3:
                raise ValueError("output_layers must have exactly 3 layers")
            for layer in v:
                if not isinstance(layer, int) or layer < 0:
                    raise ValueError(f"Invalid layer index: {layer}")
        return v

    @field_validator("max_text_length")
    @classmethod
    def validate_max_text_length(cls, v):
        if v < 16 or v > 8192:
            raise ValueError("max_text_length must be between 16 and 8192")
        return v
