"""
Z-Image Pipeline Schema

last updated: 2026-01-27

Z-Image (S3-DiT 6B) is the primary image generation pipeline with two variants:
- Turbo: Fast 9-step distilled generation (CFG baked in)
- Base: Quality 40-step generation with full CFG and negative prompt support

Advanced features include:
- DyPE (Dynamic Position Extrapolation) for high-resolution generation
- SLG (Skip Layer Guidance) for improved anatomy
- FMTT (Flow Map Trajectory Tilting) with SigLIP for prompt adherence
- VL conditioning via Qwen3-VL vision-language model
- Image-to-image editing with mask support
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
    endpoint="/api/generate",  # Legacy endpoint for backwards compatibility
    params=[
        # === Basic Parameters ===
        ParamSchema(
            id="prompt",
            type="textarea",
            label="Prompt",
            placeholder="Describe the image you want to generate...",
            rows=4,
            group="basic",
            required=True,
            tooltip="Detailed description of the image. Supports long prompts up to 1504 tokens.",
        ),
        ParamSchema(
            id="negative_prompt",
            type="textarea",
            label="Negative Prompt",
            placeholder="Elements to avoid (blur, artifacts, etc.)...",
            rows=2,
            group="basic",
            conditional={"_variant": "base"},  # Only show for base variant
            tooltip="What to avoid in the image. Uses CFG to steer away from unwanted elements.",
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
        ),
        ParamSchema(
            id="dimension_preset",
            type="select",
            label="Preset",
            default="1024x1024",
            options=DIMENSION_PRESETS,
            group="basic",
            tooltip="Quick dimension presets. Selecting a preset updates width and height.",
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
        ),
        ParamSchema(
            id="dynamic_shift",
            type="checkbox",
            label="Dynamic Shift",
            default=False,
            group="scheduler",
            tooltip="Enable resolution-adaptive shift (recommended for non-1024x1024).",
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
        ),

        # === DyPE (Dynamic Position Extrapolation) ===
        ParamSchema(
            id="dype_enabled",
            type="checkbox",
            label="Enable DyPE",
            default=False,
            group="advanced",
            tooltip="Enable Dynamic Position Extrapolation for high-resolution (>1024px) generation.",
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
        ),

        # === SLG (Skip Layer Guidance) ===
        ParamSchema(
            id="slg_enabled",
            type="checkbox",
            label="Enable SLG",
            default=False,
            group="advanced",
            tooltip="Skip Layer Guidance for improved anatomy and composition.",
        ),
        ParamSchema(
            id="slg_scale",
            type="slider",
            label="SLG Scale",
            default=2.5,
            min=0.0,
            max=10.0,
            step=0.1,
            group="advanced",
            conditional={"slg_enabled": True},
            tooltip="Skip layer guidance strength. Higher = stronger effect.",
        ),
        ParamSchema(
            id="slg_start_step",
            type="slider",
            label="SLG Start Step",
            default=0.15,
            min=0.0,
            max=1.0,
            step=0.01,
            group="advanced",
            conditional={"slg_enabled": True},
            tooltip="When to start SLG (fraction of total steps).",
        ),
        ParamSchema(
            id="slg_end_step",
            type="slider",
            label="SLG End Step",
            default=0.7,
            min=0.0,
            max=1.0,
            step=0.01,
            group="advanced",
            conditional={"slg_enabled": True},
            tooltip="When to end SLG (fraction of total steps).",
        ),
        ParamSchema(
            id="slg_skip_layers",
            type="select",
            label="SLG Skip Layers",
            default="7,8,9",
            options=["5,6,7", "7,8,9", "8,9,10", "10,11,12", "all_middle"],
            group="advanced",
            conditional={"slg_enabled": True},
            tooltip="Which DiT layers to skip. Middle layers (7-10) work best.",
        ),

        # === FMTT (Flow Map Trajectory Tilting) ===
        ParamSchema(
            id="fmtt_enabled",
            type="checkbox",
            label="Enable FMTT",
            default=False,
            group="expert",
            tooltip="Flow Map Trajectory Tilting with SigLIP for better prompt adherence.",
        ),
        ParamSchema(
            id="fmtt_scale",
            type="slider",
            label="FMTT Scale",
            default=1.0,
            min=0.0,
            max=5.0,
            step=0.1,
            group="expert",
            conditional={"fmtt_enabled": True},
            tooltip="FMTT guidance strength.",
        ),
        ParamSchema(
            id="fmtt_start_step",
            type="slider",
            label="FMTT Start Step",
            default=0.0,
            min=0.0,
            max=1.0,
            step=0.01,
            group="expert",
            conditional={"fmtt_enabled": True},
            tooltip="When to start FMTT (fraction of total steps).",
        ),
        ParamSchema(
            id="fmtt_end_step",
            type="slider",
            label="FMTT End Step",
            default=0.5,
            min=0.0,
            max=1.0,
            step=0.01,
            group="expert",
            conditional={"fmtt_enabled": True},
            tooltip="When to end FMTT (fraction of total steps).",
        ),

        # === FBCache (Forward Block Cache) ===
        ParamSchema(
            id="fbcache_enabled",
            type="checkbox",
            label="Enable FBCache",
            default=False,
            group="optimization",
            tooltip="Forward Block Cache for 30-50% speedup with slight quality tradeoff.",
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
        ),

        # === VL Conditioning ===
        ParamSchema(
            id="vl_enabled",
            type="checkbox",
            label="Enable VL Conditioning",
            default=False,
            group="advanced",
            tooltip="Use Qwen3-VL vision-language model for image-guided generation.",
        ),
        ParamSchema(
            id="vl_image",
            type="image",
            label="Reference Image",
            group="advanced",
            conditional={"vl_enabled": True},
            tooltip="Upload an image to use as visual reference.",
        ),
        ParamSchema(
            id="vl_strength",
            type="slider",
            label="VL Strength",
            default=0.8,
            min=0.0,
            max=1.0,
            step=0.05,
            group="advanced",
            conditional={"vl_enabled": True},
            tooltip="How strongly to use the reference image.",
        ),
    ],
))
