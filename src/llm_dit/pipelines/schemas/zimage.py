"""
Z-Image Pipeline Schema

last updated: 2026-02-27

Z-Image (S3-DiT 6B) is the primary image generation pipeline with two variants:
- Turbo: Fast 9-step distilled generation (CFG baked in)
- Base: Quality 40-step generation with full CFG and negative prompt support

Advanced features include:
- DyPE (Dynamic Position Extrapolation) for high-resolution generation
- LoRA weight loading with per-request selection
- VL conditioning via Qwen3-VL vision-language model
- Image-to-image editing with mask support

Note: SLG and FMTT are supported at the API level but not exposed in the
frontend schema. Pass slg_* and fmtt_* parameters directly via API requests.

ZImageConfig only holds infrastructure (model_path, text_encoder_path, variant,
default_preset). All generation params come from presets, not config -- so every
schema param here is config_mapped=False.
"""

from . import register_pipeline, PipelineSchema, ParamSchema


# Common dimension presets for quick selection
DIMENSION_PRESETS = [
    "1024x1024",  # Square (default)
    "1152x896",   # Landscape 4:3
    "1216x832",   # Landscape 3:2
    "1344x768",   # Landscape 16:9
    "768x1344",   # Portrait 9:16
    "832x1216",   # Portrait 2:3
    "896x1152",   # Portrait 3:4
]


register_pipeline(PipelineSchema(
    id="zimage",
    name="Z-Image",
    description="Fast high-quality image generation with S3-DiT 6B (Turbo/Base variants)",
    output_type="image",
    color="blue",
    icon="🎨",
    category="image",
    supports_history=True,
    supports_img2img=True,
    supports_streaming=True,  # Enable SSE progress streaming
    endpoint="/api/generate/stream",  # Stream endpoint for progress updates
    params=[
        # === Basic Parameters ===
        ParamSchema(
            id="preset",
            type="select",
            label="Preset",
            default="photorealistic",
            options=[],  # Populated dynamically from API
            options_endpoint="/api/presets/zimage",
            group="basic",
            tooltip="Load a generation preset with pre-configured settings. Presets provide negative prompts, CFG, and steps optimized for different use cases.",
            config_mapped=False,
        ),
        ParamSchema(
            id="prompt",
            type="textarea",
            label="Prompt",
            placeholder="Describe the image you want to generate...",
            rows=4,
            group="basic",
            required=True,
            tooltip="Detailed description of the image. Supports long prompts up to 1504 tokens.",
            config_mapped=False,
        ),
        ParamSchema(
            id="negative_prompt",
            type="textarea",
            label="Negative Prompt",
            default="",  # Populated when preset is selected
            placeholder="Elements to avoid (blur, artifacts, etc.)...",
            rows=2,
            group="basic",
            conditional={"_variant": "base"},  # Only show for base variant
            tooltip="What to avoid. Preset populates this - edit to customize.",
            config_mapped=False,
        ),
        ParamSchema(
            id="width",
            type="number",
            label="Width",
            default=1024,
            min=256,
            max=4096,
            step=64,
            group="basic",
            tooltip="Image width in pixels. Must be divisible by 16. Higher values use more VRAM.",
            config_mapped=False,
        ),
        ParamSchema(
            id="height",
            type="number",
            label="Height",
            default=1024,
            min=256,
            max=4096,
            step=64,
            group="basic",
            tooltip="Image height in pixels. Must be divisible by 16. Higher values use more VRAM.",
            config_mapped=False,
        ),
        ParamSchema(
            id="dimension_preset",
            type="select",
            label="Preset",
            default="1024x1024",
            options=DIMENSION_PRESETS,
            group="basic",
            tooltip="Quick dimension presets. Selecting a preset updates width and height.",
            config_mapped=False,
        ),
        ParamSchema(
            id="steps",
            type="slider",
            label="Steps",
            default=9,
            min=1,
            max=50,
            step=1,
            group="basic",
            tooltip="Number of denoising steps. 4-12 for turbo mode, 20-50 for quality.",
            config_mapped=False,
        ),
        ParamSchema(
            id="guidance_scale",
            type="slider",
            label="CFG Scale",
            default=0.0,
            min=0.0,
            max=30.0,
            step=0.5,
            group="basic",
            tooltip="Classifier-free guidance scale. 0.0 for turbo mode, 3.5-7.5 for standard.",
            config_mapped=False,
        ),
        ParamSchema(
            id="seed",
            type="number",
            label="Seed",
            default=-1,
            min=-1,
            max=2147483647,
            step=1,
            group="basic",
            tooltip="Random seed for reproducibility. -1 for random.",
            config_mapped=False,
        ),

        # === Scheduler Parameters ===
        ParamSchema(
            id="shift",
            type="slider",
            label="Shift",
            default=3.0,
            min=0.0,
            max=15.0,
            step=0.1,
            group="scheduler",
            tooltip="Flow matching shift parameter. Higher = more denoising in early steps.",
            config_mapped=False,
        ),
        ParamSchema(
            id="dynamic_shift",
            type="checkbox",
            label="Dynamic Shift",
            default=False,
            group="scheduler",
            tooltip="Enable resolution-adaptive shift (recommended for non-1024x1024).",
            config_mapped=False,
        ),
        ParamSchema(
            id="d_noise",
            type="slider",
            label="D-Noise",
            default=1.0,
            min=0.5,
            max=2.0,
            step=0.01,
            group="scheduler",
            tooltip="Sigma schedule scaling. <1.0 = sharper (more denoising), >1.0 = softer. Default 1.0.",
            config_mapped=False,
        ),

        # === DyPE (Dynamic Position Extrapolation) ===
        ParamSchema(
            id="dype_enabled",
            type="checkbox",
            label="Enable DyPE",
            default=False,
            group="advanced",
            tooltip="Enable Dynamic Position Extrapolation for high-resolution (>1024px) generation.",
            config_mapped=False,
        ),
        ParamSchema(
            id="dype_base_resolution",
            type="number",
            label="DyPE Base Resolution",
            default=1024,
            min=512,
            max=2048,
            step=64,
            group="advanced",
            conditional={"dype_enabled": True},
            tooltip="Resolution the model was trained at. Usually 1024.",
            config_mapped=False,
        ),
        ParamSchema(
            id="dype_ntk_factor",
            type="slider",
            label="DyPE NTK Factor",
            default=0.0,
            min=0.0,
            max=8.0,
            step=0.1,
            group="advanced",
            conditional={"dype_enabled": True},
            tooltip="NTK-aware scaling factor. 0 = auto-calculate from resolution.",
            config_mapped=False,
        ),

        # === FBCache (Forward Block Cache) ===
        ParamSchema(
            id="fbcache_enabled",
            type="checkbox",
            label="Enable FBCache",
            default=False,
            group="optimization",
            tooltip="Forward Block Cache for 30-50% speedup with slight quality tradeoff.",
            config_mapped=False,
        ),
        ParamSchema(
            id="fbcache_start_step",
            type="slider",
            label="FBCache Start Step",
            default=1,
            min=1,
            max=10,
            step=1,
            group="optimization",
            conditional={"fbcache_enabled": True},
            tooltip="Step to start caching (skip first N steps for quality).",
            config_mapped=False,
        ),
        ParamSchema(
            id="fbcache_threshold",
            type="slider",
            label="FBCache Threshold",
            default=0.05,
            min=0.01,
            max=0.2,
            step=0.01,
            group="optimization",
            conditional={"fbcache_enabled": True},
            tooltip="Similarity threshold for cache reuse. Lower = more caching.",
            config_mapped=False,
        ),

        # === CPU Offload ===
        ParamSchema(
            id="cpu_offload",
            type="checkbox",
            label="CPU Offload",
            default=False,
            group="optimization",
            tooltip="Move encoder to CPU to fit in 24GB VRAM. Slower but uses less GPU memory.",
            config_mapped=False,
        ),

        # === Torch Compile ===
        ParamSchema(
            id="compile",
            type="checkbox",
            label="Torch Compile",
            default=False,
            group="optimization",
            tooltip="Use torch.compile for faster inference (slow first run, then 20-40% speedup).",
            config_mapped=False,
        ),

        # === Hidden Layer Selection ===
        ParamSchema(
            id="hidden_layer",
            type="select",
            label="Hidden Layer",
            default="-2",
            options=[str(i) for i in range(-1, -33, -1)],  # "-1" to "-32"
            group="advanced",
            tooltip="Which Qwen3-4B layer to extract embeddings from. "
                    "-1 = last layer (most semantic), -2 = penultimate (default), "
                    "deeper negative numbers = earlier layers (more syntactic).",
            config_mapped=False,
        ),

        # === LoRA Enhancement ===
        ParamSchema(
            id="loras",
            type="lora_list",
            label="LoRA Weights",
            default=[],
            group="enhancement",
            tooltip="LoRA files with strength (path:scale format, e.g. style.safetensors:0.8).",
            scale_min=-2.0,
            scale_max=2.0,
            max_count=5,
            config_mapped=False,
        ),

    ],
))
