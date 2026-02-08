"""Configuration management endpoints: pipeline schemas, generation config,
presets, resolution config, rewriter config, session config, and profiles.

Serves the UI with all configuration data needed for dynamic form rendering,
resolution constraints, preset management, and live config editing.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from web.dependencies import ConfigDep

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Pipeline Schema Endpoints
# =============================================================================


@router.get("/api/pipelines")
async def get_pipeline_schemas(config: ConfigDep):
    """Return all pipeline schemas for frontend form generation.

    The frontend uses these schemas to dynamically render forms without
    hardcoding pipeline-specific UI. Each schema describes:
    - Pipeline metadata (name, description, output type)
    - Form parameters with types, defaults, and constraints
    - Feature flags (img2img, streaming, reference images)

    Returns:
        dict with:
        - pipelines: Dict of pipeline_id -> PipelineSchema
        - defaults: Current RuntimeConfig values (if loaded)
        - loaded_pipeline: Currently loaded pipeline type (if any)
    """
    import web.server as srv
    from llm_dit.pipelines.schemas import get_all_pipelines

    # Get all registered pipeline schemas
    pipelines = get_all_pipelines()
    pipeline_dicts = {pid: schema.to_dict() for pid, schema in pipelines.items()}

    # Get current defaults from RuntimeConfig if available
    defaults = {}
    try:
        defaults = config.to_dict()
    except Exception as e:
        logger.warning(f"Failed to serialize RuntimeConfig: {e}")

    # Determine which pipeline is currently loaded
    loaded_pipeline = None
    if srv.pipeline is not None:
        loaded_pipeline = "zimage"
    elif srv.qwen_image_pipeline is not None:
        loaded_pipeline = "qwenimage-edit"
    elif srv.qwen_image_t2i_pipeline is not None:
        loaded_pipeline = "qwenimage-t2i"
    elif srv.ltx2_pipeline is not None:
        loaded_pipeline = "ltx2"
    elif srv.flux2_pipeline is not None:
        loaded_pipeline = "flux2"

    return {
        "pipelines": pipeline_dicts,
        "defaults": defaults,
        "loaded_pipeline": loaded_pipeline,
    }


@router.get("/api/pipelines/{pipeline_id}")
async def get_pipeline_schema(pipeline_id: str):
    """Get schema for a specific pipeline.

    Args:
        pipeline_id: Pipeline identifier (e.g., "zimage", "ltx2")

    Returns:
        PipelineSchema dict for the requested pipeline

    Raises:
        404 if pipeline not found
    """
    from llm_dit.pipelines.schemas import get_all_pipelines, get_pipeline

    schema = get_pipeline(pipeline_id)
    if schema is None:
        raise HTTPException(
            status_code=404,
            detail=f"Pipeline '{pipeline_id}' not found. Available: {list(get_all_pipelines().keys())}",
        )

    return schema.to_dict()


@router.get("/api/pipelines/{pipeline_id}/defaults")
async def get_pipeline_defaults(pipeline_id: str, config: ConfigDep):
    """Get default values for a specific pipeline.

    Merges schema defaults with RuntimeConfig values for the pipeline.

    Args:
        pipeline_id: Pipeline identifier

    Returns:
        Dict of parameter_id -> default_value
    """
    from llm_dit.pipelines.schemas import get_pipeline

    schema = get_pipeline(pipeline_id)
    if schema is None:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")

    # Start with schema defaults
    defaults = schema.get_defaults()

    # Overlay RuntimeConfig values if available
    config_dict = config.to_dict()

    # Map schema param IDs to RuntimeConfig field names
    # Schema uses clean names, config uses prefixed names for clarity
    param_to_config_map = {
        # FLUX.2 mappings
        "block_offload": "flux2_block_offload",
        "compile": "compile",
        # LTX-2 mappings
        "cpu_offload": "cpu_offload",
        # Z-Image mappings
        "shift": "shift",
        "guidance_scale": "guidance_scale",
        "steps": "steps",
        "width": "width",
        "height": "height",
    }

    for param in schema.params:
        # First check direct match (param.id == config field name)
        if param.id in config_dict:
            defaults[param.id] = config_dict[param.id]
        # Then check mapped name
        elif param.id in param_to_config_map:
            config_key = param_to_config_map[param.id]
            if config_key in config_dict:
                defaults[param.id] = config_dict[config_key]

    # Add special _variant field for conditional visibility
    # This tells the UI which variant is configured (base/turbo)
    if pipeline_id == "zimage":
        defaults["_variant"] = getattr(config, "zimage_variant", "turbo")

    return defaults


# =============================================================================
# Generation Config Endpoint
# =============================================================================


@router.get("/api/generation-config")
async def get_generation_config(config: ConfigDep):
    """Get generation configuration defaults from server config.

    Returns default values for width, height, steps, shift, long_prompt_mode, hidden_layer, SLG, and FMTT.
    Also returns feature flags indicating which advanced features are enabled in config.
    The UI should call this on load to sync with server config.

    If a default_preset is configured for the current pipeline (e.g., zimage.default_preset),
    the preset's generation params (steps, guidance_scale, shift, negative_prompt) are used.
    """
    # Get feature flags from config
    dype_enabled = getattr(config, "dype_enabled", False)
    slg_enabled = (
        getattr(config, "slg_enabled", False)
        or getattr(config, "slg_scale", 0.0) > 0
    )
    fmtt_enabled = (
        getattr(config, "fmtt_enabled", False)
        or getattr(config, "fmtt_scale", 0.0) > 0
    )

    # Default generation params from RuntimeConfig (turbo defaults)
    steps = config.steps
    guidance_scale = config.guidance_scale
    shift = config.shift
    d_noise = getattr(config, "d_noise", 1.0)
    default_preset = ""
    negative_prompt = ""

    # Try to load default_preset from zimage config and use its values
    # This makes the API preset-aware: config.toml holds infrastructure,
    # presets hold generation params (steps, guidance_scale, shift, negative_prompt)
    try:
        config_dict = config.to_dict() if hasattr(config, 'to_dict') else {}
        zimage_config = config_dict.get("zimage", {})
        if isinstance(zimage_config, dict):
            default_preset = zimage_config.get("default_preset", "")

        if default_preset:
            from llm_dit.presets import get_preset_registry
            presets_dir = config_dict.get("presets_dir", "presets")
            registry = get_preset_registry(presets_dir)
            preset = registry.get(default_preset)
            if preset:
                # Use preset values for generation params
                if preset.steps is not None:
                    steps = preset.steps
                if preset.guidance_scale is not None:
                    guidance_scale = preset.guidance_scale
                if preset.shift is not None:
                    shift = preset.shift
                if preset.d_noise is not None:
                    d_noise = preset.d_noise
                if preset.negative_prompt:
                    negative_prompt = preset.negative_prompt
    except Exception as e:
        # If preset loading fails, fall back to RuntimeConfig values
        logger.warning(f"Failed to load default preset '{default_preset}': {e}")

    return {
        # Z-Image variant (turbo/base) - determines default parameter values
        "zimage_variant": getattr(config, "zimage_variant", "turbo"),
        "width": config.width,
        "height": config.height,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "shift": shift,
        "dynamic_shift": getattr(config, "dynamic_shift", False),
        "d_noise": d_noise,
        "long_prompt_mode": config.long_prompt_mode,
        "hidden_layer": config.hidden_layer,
        "layer_weights": getattr(config, "layer_weights", None),
        # Preset info
        "default_preset": default_preset,
        "negative_prompt": negative_prompt,
        # SLG settings
        "slg_scale": getattr(config, "slg_scale", 0.0),
        "slg_layers": getattr(config, "slg_layers", None),
        "slg_start": getattr(config, "slg_start", 0.05),
        "slg_stop": getattr(config, "slg_stop", 0.5),
        # FMTT settings
        "fmtt_scale": getattr(config, "fmtt_scale", 0.0),
        "fmtt_start": getattr(config, "fmtt_start", 0.0),
        "fmtt_stop": getattr(config, "fmtt_stop", 0.5),
        "fmtt_normalize": getattr(config, "fmtt_normalize", "unit"),
        "fmtt_decode_scale": getattr(config, "fmtt_decode_scale", 0.5),
        "fmtt_siglip_model": getattr(
            config, "fmtt_siglip_model", "google/siglip2-giant-opt-patch16-384"
        ),
        "fmtt_siglip_device": getattr(config, "fmtt_siglip_device", "cuda"),
        # Feature flags based on config
        "features": {
            "dype_enabled": dype_enabled,
            "slg_enabled": slg_enabled,
            "fmtt_enabled": fmtt_enabled,
            "differential_img2img_enabled": True,  # Always available for Z-Image
        },
    }


# =============================================================================
# Preset Endpoints
# =============================================================================


def _get_preset_registry(config: ConfigDep):
    """Helper to get or initialize the preset registry."""
    from llm_dit.presets import get_preset_registry

    try:
        return get_preset_registry()
    except ValueError:
        # Registry not initialized - initialize with default path
        presets_dir = "presets"
        config_dict = config.to_dict() if hasattr(config, 'to_dict') else {}
        presets_dir = config_dict.get("presets_dir", "presets")
        return get_preset_registry(presets_dir)


@router.get("/api/presets")
async def get_all_presets(config: ConfigDep):
    """Return all generation presets with metadata.

    Returns:
        dict with:
        - presets: List of preset objects with name, description, category, params
    """
    registry = _get_preset_registry(config)
    all_presets = registry.get_all()
    return {
        "presets": [p.to_api_response() for p in all_presets.values()],
    }


@router.get("/api/presets/{pipeline_id}")
async def get_presets_for_pipeline(pipeline_id: str, config: ConfigDep, variant: Optional[str] = None):
    """Return presets that apply to a specific pipeline.

    Args:
        pipeline_id: Pipeline identifier (e.g., "zimage", "ltx2")
        variant: Optional variant filter (e.g., "base", "turbo")

    Returns:
        dict with:
        - presets: List of preset objects applicable to this pipeline
        - default_preset: Name of the default preset (from config or schema)
    """
    registry = _get_preset_registry(config)

    # Get presets for this pipeline (and optional variant filter)
    presets = registry.list_for_pipeline(pipeline_id, variant=variant)

    # Determine default preset from config
    default_preset = ""
    config_dict = config.to_dict() if hasattr(config, 'to_dict') else {}
    # Check pipeline-specific default_preset
    pipeline_config = config_dict.get(pipeline_id, {})
    if isinstance(pipeline_config, dict):
        default_preset = pipeline_config.get("default_preset", "")

    return {
        "presets": [p.to_api_response() for p in presets],
        "default_preset": default_preset,
    }


@router.get("/api/presets/preset/{name}")
async def get_preset_by_name(name: str, config: ConfigDep):
    """Get full details for a specific preset.

    Args:
        name: Preset name

    Returns:
        Full preset object with all parameters

    Raises:
        404 if preset not found
    """
    registry = _get_preset_registry(config)

    preset = registry.get(name)
    if preset is None:
        raise HTTPException(
            status_code=404,
            detail=f"Preset '{name}' not found. Available: {registry.list_names()}",
        )

    return preset.to_api_response()


# =============================================================================
# Resolution Config Endpoint
# =============================================================================


@router.get("/api/resolution-config")
async def get_resolution_config(config: ConfigDep, model: Optional[str] = None):
    """Get resolution constraints for client-side validation.

    Returns VAE multiple, min/max limits, categorized presets, and DyPE config.
    Presets are filtered based on the active model type.

    Args:
        model: Optional model filter ("zimage", "qwenimage-edit", "qwenimage-t2i")
               If not provided, returns presets for all models.

    Model-specific constraints:
    - Z-Image: Flexible resolutions, must be divisible by 16
    - Qwen-Image Edit: Fixed 640x640 or 1024x1024 only
    - Qwen-Image T2I: Default 1328x1328, flexible with VAE constraints
    """
    import web.server as srv
    from llm_dit.constants import (
        ASPECT_RATIOS,
        DEFAULT_RESOLUTION,
        MAX_RESOLUTION,
        MIN_RESOLUTION,
        VAE_MULTIPLE,
        VAE_SCALE_FACTOR,
    )

    # Detect currently loaded model if not specified
    current_model = model
    if current_model is None:
        if srv.pipeline is not None:
            from llm_dit.pipelines import ZImagePipeline
            from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

            if isinstance(srv.pipeline, ZImagePipeline):
                current_model = "zimage"
            elif isinstance(srv.pipeline, QwenImageDiffusersPipeline):
                current_model = "qwenimage-edit"
        if srv.qwen_image_t2i_pipeline is not None:
            current_model = "qwenimage-t2i"
        if srv.qwen_image_pipeline is not None and current_model is None:
            current_model = "qwenimage-edit"

    # DyPE configuration (Z-Image only)
    DYPE_BASE_RESOLUTION = 1024  # Z-Image training resolution

    def get_dype_recommendation(width: int, height: int) -> dict:
        """Get DyPE recommendation based on resolution."""
        max_dim = max(width, height)
        scale = max_dim / DYPE_BASE_RESOLUTION
        if scale <= 1.0:
            return {"recommended": False, "exponent": None}
        if scale >= 3.0:
            exponent = 2.0
        elif scale >= 1.5:
            exponent = 1.0
        else:
            exponent = 0.5
        return {"recommended": True, "exponent": exponent}

    # Model-specific resolution constraints
    model_constraints = {
        "zimage": {
            "vae_multiple": 16,
            "min_resolution": 256,
            "max_resolution": 4096,
            "default_width": 1024,
            "default_height": 1024,
            "flexible": True,
            "supports_dype": True,
            "supports_slg": True,
            "supports_fmtt": True,
        },
        "qwenimage-edit": {
            "vae_multiple": 16,
            "min_resolution": 640,
            "max_resolution": 1024,
            "default_width": 640,
            "default_height": 640,
            "flexible": False,  # Only 640 or 1024
            "fixed_sizes": [640, 1024],
            "supports_dype": False,
            "supports_slg": False,
            "supports_fmtt": False,
        },
        "qwenimage-t2i": {
            "vae_multiple": 16,
            "min_resolution": 256,
            "max_resolution": 2048,
            "default_width": 1328,
            "default_height": 1328,
            "flexible": True,
            "supports_dype": False,
            "supports_slg": False,
            "supports_fmtt": False,
        },
    }

    # Helper to determine aspect category for filter buttons
    def get_aspect_category(width: int, height: int) -> str:
        """Determine aspect category based on ratio for UI filtering."""
        ratio = width / height
        if 0.95 <= ratio <= 1.05:
            return "square"
        elif ratio > 1.05:
            if ratio >= 2.0:  # 19.5:9 = 2.17, 21:9 = 2.33
                return "mobile-landscape"
            return "landscape"
        else:  # ratio < 0.95
            if ratio <= 0.5:  # 9:19.5 = 0.46, 9:20 = 0.45
                return "mobile-portrait"
            return "portrait"

    # Z-Image presets (flexible, all divisible by 16)
    zimage_presets = [
        # Square (1:1)
        {
            "value": "512x512",
            "label": "512",
            "width": 512,
            "height": 512,
            "category": "square",
            "ratio": "1:1",
        },
        {
            "value": "768x768",
            "label": "768",
            "width": 768,
            "height": 768,
            "category": "square",
            "ratio": "1:1",
        },
        {
            "value": "1024x1024",
            "label": "1024",
            "width": 1024,
            "height": 1024,
            "category": "square",
            "ratio": "1:1",
            "default": True,
        },
        {
            "value": "1280x1280",
            "label": "1280",
            "width": 1280,
            "height": 1280,
            "category": "square",
            "ratio": "1:1",
        },
        {
            "value": "1536x1536",
            "label": "1536",
            "width": 1536,
            "height": 1536,
            "category": "square",
            "ratio": "1:1",
        },
        {
            "value": "1920x1920",
            "label": "1920",
            "width": 1920,
            "height": 1920,
            "category": "square",
            "ratio": "1:1",
        },
        {
            "value": "2048x2048",
            "label": "2K",
            "width": 2048,
            "height": 2048,
            "category": "square",
            "ratio": "1:1",
        },
        # Landscape - 16:9
        {
            "value": "1280x720",
            "label": "720p",
            "width": 1280,
            "height": 720,
            "category": "landscape",
            "ratio": "16:9",
        },
        {
            "value": "1920x1088",
            "label": "1080p",
            "width": 1920,
            "height": 1088,
            "category": "landscape",
            "ratio": "16:9",
        },
        {
            "value": "2560x1440",
            "label": "1440p",
            "width": 2560,
            "height": 1440,
            "category": "landscape",
            "ratio": "16:9",
        },
        # Landscape - 3:2
        {
            "value": "1536x1024",
            "label": "1536x1024",
            "width": 1536,
            "height": 1024,
            "category": "landscape",
            "ratio": "3:2",
        },
        {
            "value": "1920x1280",
            "label": "1920x1280",
            "width": 1920,
            "height": 1280,
            "category": "landscape",
            "ratio": "3:2",
        },
        # Landscape - 4:3
        {
            "value": "1024x768",
            "label": "1024x768",
            "width": 1024,
            "height": 768,
            "category": "landscape",
            "ratio": "4:3",
        },
        {
            "value": "1280x960",
            "label": "1280x960",
            "width": 1280,
            "height": 960,
            "category": "landscape",
            "ratio": "4:3",
        },
        {
            "value": "1600x1200",
            "label": "1600x1200",
            "width": 1600,
            "height": 1200,
            "category": "landscape",
            "ratio": "4:3",
        },
        # Mobile Landscape - 21:9, 19.5:9 (phone screens rotated)
        {
            "value": "1792x768",
            "label": "Ultrawide",
            "width": 1792,
            "height": 768,
            "category": "landscape",
            "ratio": "21:9",
        },
        {
            "value": "2560x1088",
            "label": "UW 1080",
            "width": 2560,
            "height": 1088,
            "category": "landscape",
            "ratio": "21:9",
        },
        {
            "value": "2340x1080",
            "label": "Phone HD",
            "width": 2340,
            "height": 1080,
            "category": "landscape",
            "ratio": "19.5:9",
        },
        # Portrait - 9:16
        {
            "value": "720x1280",
            "label": "720p",
            "width": 720,
            "height": 1280,
            "category": "portrait",
            "ratio": "9:16",
        },
        {
            "value": "1088x1920",
            "label": "1080p",
            "width": 1088,
            "height": 1920,
            "category": "portrait",
            "ratio": "9:16",
        },
        {
            "value": "1440x2560",
            "label": "1440p",
            "width": 1440,
            "height": 2560,
            "category": "portrait",
            "ratio": "9:16",
        },
        # Portrait - 2:3
        {
            "value": "1024x1536",
            "label": "1024x1536",
            "width": 1024,
            "height": 1536,
            "category": "portrait",
            "ratio": "2:3",
        },
        {
            "value": "1280x1920",
            "label": "1280x1920",
            "width": 1280,
            "height": 1920,
            "category": "portrait",
            "ratio": "2:3",
        },
        # Portrait - 3:4
        {
            "value": "768x1024",
            "label": "768x1024",
            "width": 768,
            "height": 1024,
            "category": "portrait",
            "ratio": "3:4",
        },
        {
            "value": "960x1280",
            "label": "960x1280",
            "width": 960,
            "height": 1280,
            "category": "portrait",
            "ratio": "3:4",
        },
        {
            "value": "1200x1600",
            "label": "1200x1600",
            "width": 1200,
            "height": 1600,
            "category": "portrait",
            "ratio": "3:4",
        },
        # Mobile Portrait - 9:19.5, 9:20 (phone screens)
        {
            "value": "1080x2340",
            "label": "Phone HD",
            "width": 1080,
            "height": 2340,
            "category": "portrait",
            "ratio": "9:19.5",
        },
        {
            "value": "1284x2778",
            "label": "iPhone Pro",
            "width": 1284,
            "height": 2778,
            "category": "portrait",
            "ratio": "9:19.5",
        },
    ]

    # Qwen-Image Edit presets (FIXED: only 640 or 1024 square)
    qwenimage_edit_presets = [
        {
            "value": "640x640",
            "label": "640 (Fast)",
            "width": 640,
            "height": 640,
            "category": "square",
            "ratio": "1:1",
            "default": True,
        },
        {
            "value": "1024x1024",
            "label": "1024 (Quality)",
            "width": 1024,
            "height": 1024,
            "category": "square",
            "ratio": "1:1",
        },
    ]

    # Qwen-Image T2I presets (flexible, default 1328)
    qwenimage_t2i_presets = [
        {
            "value": "512x512",
            "label": "512",
            "width": 512,
            "height": 512,
            "category": "square",
            "ratio": "1:1",
        },
        {
            "value": "768x768",
            "label": "768",
            "width": 768,
            "height": 768,
            "category": "square",
            "ratio": "1:1",
        },
        {
            "value": "1024x1024",
            "label": "1024",
            "width": 1024,
            "height": 1024,
            "category": "square",
            "ratio": "1:1",
        },
        {
            "value": "1328x1328",
            "label": "1328 (Default)",
            "width": 1328,
            "height": 1328,
            "category": "square",
            "ratio": "1:1",
            "default": True,
        },
        {
            "value": "1536x1536",
            "label": "1536",
            "width": 1536,
            "height": 1536,
            "category": "square",
            "ratio": "1:1",
        },
        # Landscape
        {
            "value": "1328x1024",
            "label": "1328x1024",
            "width": 1328,
            "height": 1024,
            "category": "landscape",
            "ratio": "4:3",
        },
        {
            "value": "1536x1024",
            "label": "1536x1024",
            "width": 1536,
            "height": 1024,
            "category": "landscape",
            "ratio": "3:2",
        },
        # Portrait
        {
            "value": "1024x1328",
            "label": "1024x1328",
            "width": 1024,
            "height": 1328,
            "category": "portrait",
            "ratio": "3:4",
        },
        {
            "value": "1024x1536",
            "label": "1024x1536",
            "width": 1024,
            "height": 1536,
            "category": "portrait",
            "ratio": "2:3",
        },
    ]

    # Select presets based on model
    if current_model == "qwenimage-edit":
        presets = qwenimage_edit_presets
        constraints = model_constraints["qwenimage-edit"]
    elif current_model == "qwenimage-t2i":
        presets = qwenimage_t2i_presets
        constraints = model_constraints["qwenimage-t2i"]
    else:
        # Default to Z-Image
        presets = zimage_presets
        constraints = model_constraints["zimage"]

    # Add aspect_category and DyPE recommendations to presets
    for preset in presets:
        # Add aspect_category for UI filtering
        preset["aspect_category"] = get_aspect_category(preset["width"], preset["height"])

        # Add DyPE recommendations (Z-Image only)
        if current_model in (None, "zimage"):
            preset["dype"] = get_dype_recommendation(preset["width"], preset["height"])
        else:
            preset["dype"] = {"recommended": False, "exponent": None}

    # Determine available categories
    categories = list(set(p["category"] for p in presets))

    # Use config.toml values if available, otherwise fall back to model defaults
    # This ensures the UI respects user's configured resolution while still
    # providing sensible defaults (1024x1024 for Z-Image)
    if current_model == "zimage":
        default_width = getattr(config, "width", None) or constraints.get(
            "default_width", 1024
        )
        default_height = getattr(config, "height", None) or constraints.get(
            "default_height", 1024
        )
    else:
        default_width = constraints.get("default_width", 1024)
        default_height = constraints.get("default_height", 1024)

    return {
        "current_model": current_model,
        "model_constraints": model_constraints,
        "active_constraints": constraints,
        "vae_multiple": VAE_MULTIPLE,
        "vae_scale_factor": VAE_SCALE_FACTOR,
        "min_resolution": constraints.get("min_resolution", MIN_RESOLUTION),
        "max_resolution": constraints.get("max_resolution", MAX_RESOLUTION),
        "default_resolution": DEFAULT_RESOLUTION,
        "default_width": default_width,
        "default_height": default_height,
        "dype_base_resolution": DYPE_BASE_RESOLUTION,
        "aspect_ratios": ASPECT_RATIOS,
        "presets": presets,
        "categories": categories,
        "supports_dype": constraints.get("supports_dype", False),
        "supports_slg": constraints.get("supports_slg", False),
        "supports_fmtt": constraints.get("supports_fmtt", False),
    }


# =============================================================================
# Rewriter Config Endpoints
# =============================================================================


@router.get("/api/rewriter-config")
async def get_rewriter_config(config: ConfigDep):
    """Get rewriter configuration defaults from server config.

    Qwen3 Best Practices (thinking mode):
    - temperature=0.6, top_p=0.95, top_k=20, min_p=0
    - DO NOT use greedy decoding (causes repetition)
    - presence_penalty=0-2 helps reduce endless repetitions
    """
    return {
        "temperature": config.rewriter_temperature,
        "top_p": config.rewriter_top_p,
        "top_k": config.rewriter_top_k,
        "min_p": config.rewriter_min_p,
        "presence_penalty": config.rewriter_presence_penalty,
        "max_tokens": config.rewriter_max_tokens,
        "use_api": config.rewriter_use_api,
    }


@router.get("/api/rewriter-models")
async def get_rewriter_models():
    """Return available rewriter models."""
    models = [
        {
            "id": "qwen3-4b",
            "name": "Qwen3-4B (Text)",
            "supports_image": False,
            "loaded": True,
        }
    ]

    return {
        "models": models,
        "default": "qwen3-4b",
    }


# =============================================================================
# Config Availability & Loading
# =============================================================================


@router.get("/api/configs/available")
async def get_available_configs(config: ConfigDep):
    """List all available configs from modular config file.

    Returns configs grouped by model type with descriptions.
    """
    # Config listing is not supported with the current config system
    # Use --config config.toml --profile <profile_name> at server startup
    return {
        "configs": [],
        "config_path": getattr(config, "config_path", None),
        "current_profile": getattr(config, "current_profile", None),
        "message": "Dynamic config listing not supported. Use --profile at startup.",
    }


@router.post("/api/configs/load")
async def load_config_dynamic(request: dict):
    """Load a config dynamically without server restart.

    NOTE: Dynamic config loading is not supported with the current config system.
    Use --config config.toml --profile <profile_name> at server startup.
    """
    # Dynamic config loading is not supported with current config system
    return {
        "success": False,
        "error": "Dynamic config loading not supported. Restart server with --profile to change config.",
    }


# =============================================================================
# Config Management Endpoints (Phase 1-3)
# =============================================================================


@router.get("/api/config/session")
async def get_session_config(config: ConfigDep):
    """Get current session configuration values.

    Returns the current runtime_config values, the loaded profile name,
    and which fields have been modified during this session.
    """
    from llm_dit.model_manager import HOT_RELOAD_SAFE
    from web.server import session_modified_fields

    # Get all config values
    values = config.to_dict()

    # Filter to just hot-reload safe fields for the config UI
    ui_values = {k: v for k, v in values.items() if k in HOT_RELOAD_SAFE}

    return {
        "values": ui_values,
        "profile": getattr(config, "current_profile", "default"),
        "modified": list(session_modified_fields),
        "config_file": getattr(config, "config_path", None),
    }


@router.put("/api/config/session")
async def update_session_config(request: dict, config: ConfigDep):
    """Update session defaults (hot-reload safe fields only).

    These changes apply immediately but don't persist to file.
    They last until server restart.
    """
    from llm_dit.model_manager import HOT_RELOAD_SAFE, REQUIRES_RESTART
    from web.server import pending_restart_changes, session_modified_fields

    updated = []
    rejected = []
    pending_restart = []

    for field, value in request.items():
        if field in HOT_RELOAD_SAFE:
            # Hot-reload: apply immediately
            old_value = getattr(config, field, None)
            setattr(config, field, value)
            session_modified_fields.add(field)
            updated.append(field)
            logger.info(f"Session config updated: {field} = {value} (was {old_value})")
        elif field in REQUIRES_RESTART:
            # Requires restart: track for later
            pending_restart_changes[field] = value
            pending_restart.append(field)
            logger.info(f"Config change pending restart: {field} = {value}")
        else:
            rejected.append(field)
            logger.warning(f"Unknown config field rejected: {field}")

    return {
        "success": True,
        "updated": updated,
        "pending_restart": pending_restart,
        "rejected": rejected,
    }


@router.get("/api/config/profiles")
async def list_profiles(config: ConfigDep):
    """List available profiles from config.toml."""
    from pathlib import Path

    import tomllib

    config_path = getattr(config, "config_path", None)
    if not config_path:
        return {
            "profiles": [],
            "current": getattr(config, "current_profile", "default"),
            "config_file": None,
            "error": "No config file loaded",
        }

    try:
        config_file = Path(config_path)
        if not config_file.exists():
            return {
                "profiles": [],
                "current": getattr(config, "current_profile", "default"),
                "config_file": str(config_path),
                "error": f"Config file not found: {config_path}",
            }

        with open(config_file, "rb") as f:
            toml_data = tomllib.load(f)

        # Extract profile names (top-level keys that aren't _metadata)
        profiles = [k for k in toml_data.keys() if not k.startswith("_")]

        return {
            "profiles": profiles,
            "current": getattr(config, "current_profile", "default"),
            "config_file": str(config_path),
        }
    except Exception as e:
        logger.error(f"Error listing profiles: {e}")
        return {
            "profiles": [],
            "current": getattr(config, "current_profile", "default"),
            "config_file": str(config_path) if config_path else None,
            "error": str(e),
        }
