"""
LTX-2 Pipeline Schema

last updated: 2026-02-15

LTX-2 is a video generation pipeline with:
- Pure PyTorch implementation (recommended) or diffusers wrapper
- SSE streaming for real-time progress updates
- Native FP8 support on RTX 4090
- Frame formula: 8n+1 (9, 17, 25, 33, ..., 121, ...)
- Enhancement features: NAG, FETA, TeaCache, Latent Norm
"""

from . import register_pipeline, PipelineSchema, ParamSchema


# Video dimension presets (width x height)
VIDEO_DIMENSION_PRESETS = [
    "768x512",    # Landscape 3:2 (default)
    "512x768",    # Portrait 2:3
    "768x768",    # Square
    "1024x576",   # Widescreen 16:9
    "576x1024",   # Vertical video
]

# Offload strategies
OFFLOAD_OPTIONS = ["none", "model", "sequential", "group"]


register_pipeline(PipelineSchema(
    id="ltx2",
    name="LTX-2",
    description="High-quality video generation with pure PyTorch DiT",
    output_type="video",
    color="purple",
    icon="🎬",
    category="video",
    supports_history=True,
    supports_img2img=False,  # TODO: Add img2vid support
    supports_streaming=True,
    endpoint="/api/ltx2/generate/stream",
    params=[
        # === Basic Parameters ===
        ParamSchema(
            id="prompt",
            type="textarea",
            label="Prompt",
            placeholder="Describe the video you want to generate...",
            rows=4,
            group="basic",
            required=True,
            tooltip="Detailed description of the video. Be specific about motion and camera movement.",
        ),
        ParamSchema(
            id="negative_prompt",
            type="textarea",
            label="Negative Prompt",
            placeholder="Elements to avoid...",
            rows=2,
            default="worst quality, inconsistent motion, blurry, jittery, distorted",
            group="basic",
            tooltip="What to avoid in the generation.",
        ),
        ParamSchema(
            id="enhance_prompt",
            type="checkbox",
            label="Enhance Prompt",
            default=False,
            group="basic",
            tooltip="Use Gemma3 to expand your prompt into a detailed video description with motion, lighting, and audio cues.",
        ),
        ParamSchema(
            id="width",
            type="number",
            label="Width",
            default=768,
            min=256,
            max=1280,
            step=64,
            group="basic",
            tooltip="Video width in pixels. Must be divisible by 32.",
        ),
        ParamSchema(
            id="height",
            type="number",
            label="Height",
            default=512,
            min=256,
            max=1280,
            step=64,
            group="basic",
            tooltip="Video height in pixels. Must be divisible by 32.",
        ),
        ParamSchema(
            id="dimension_preset",
            type="select",
            label="Preset",
            default="768x512",
            options=VIDEO_DIMENSION_PRESETS,
            group="basic",
            tooltip="Quick dimension presets for common video formats.",
        ),
        ParamSchema(
            id="num_frames",
            type="number",
            label="Frames",
            default=33,
            min=9,
            step=8,
            group="basic",
            tooltip="Number of frames. Must follow 8n+1 formula (9, 17, 25, 33, ..., 121, ...). Values are snapped to nearest valid count. 33 = ~1.3s, 65 = ~2.7s, 121 = ~5s at 24fps.",
        ),
        ParamSchema(
            id="fps",
            type="slider",
            label="FPS",
            default=24,
            min=12,
            max=60,
            step=1,
            group="basic",
            tooltip="Frames per second for output video.",
        ),
        ParamSchema(
            id="use_two_stage",
            type="checkbox",
            label="Two-Stage",
            default=True,
            group="basic",
            tooltip="Two-stage generation: coarse pass + refinement. Higher quality, slightly slower.",
        ),
        ParamSchema(
            id="stage1_steps",
            type="slider",
            label="Stage 1 Steps",
            default=40,
            min=4,
            max=80,
            step=1,
            group="basic",
            tooltip="Denoising steps for stage 1 (coarse generation). 40 recommended for two-stage, 12 for single-stage distilled.",
        ),
        ParamSchema(
            id="stage2_steps",
            type="slider",
            label="Stage 2 Steps",
            default=3,
            min=1,
            max=10,
            step=1,
            group="basic",
            conditional={"use_two_stage": True},
            tooltip="Refinement steps for stage 2. 3 recommended.",
        ),
        ParamSchema(
            id="guidance_scale",
            type="slider",
            label="CFG Scale",
            default=3.5,
            min=1.0,
            max=10.0,
            step=0.1,
            group="basic",
            tooltip="Classifier-free guidance scale. 3.0-4.0 recommended.",
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

        # === Memory & Performance ===
        ParamSchema(
            id="offload_type",
            type="select",
            label="Offload Strategy",
            default="group",
            options=OFFLOAD_OPTIONS,
            group="optimization",
            tooltip="Memory management: none (fastest), model (balanced), sequential/group (low VRAM).",
        ),
        ParamSchema(
            id="use_fp8",
            type="checkbox",
            label="Use FP8",
            default=True,
            group="optimization",
            tooltip="Enable FP8 quantization (native on RTX 4090, saves ~40% VRAM).",
        ),
        ParamSchema(
            id="compile",
            type="checkbox",
            label="Torch Compile",
            default=False,
            group="optimization",
            tooltip="Use torch.compile for faster inference (slow first run).",
        ),

        # === Enhancement Features ===
        ParamSchema(
            id="latent_norm_enabled",
            type="checkbox",
            label="Latent Normalization",
            default=True,
            group="enhancement",
            tooltip="Normalize latents during generation (recommended).",
        ),
        ParamSchema(
            id="nag_enabled",
            type="checkbox",
            label="Enable NAG",
            default=False,
            group="enhancement",
            tooltip="Normalized Attention Guidance for improved coherence.",
        ),
        ParamSchema(
            id="nag_scale",
            type="slider",
            label="NAG Scale",
            default=1.0,
            min=0.1,
            max=3.0,
            step=0.1,
            group="enhancement",
            conditional={"nag_enabled": True},
            tooltip="NAG strength multiplier.",
        ),
        ParamSchema(
            id="feta_enabled",
            type="checkbox",
            label="Enable FETA",
            default=False,
            group="enhancement",
            tooltip="Feature-Enhanced Temporal Attention for smoother motion.",
        ),
        ParamSchema(
            id="feta_scale",
            type="slider",
            label="FETA Scale",
            default=1.0,
            min=0.1,
            max=3.0,
            step=0.1,
            group="enhancement",
            conditional={"feta_enabled": True},
            tooltip="FETA strength multiplier.",
        ),
        ParamSchema(
            id="teacache_enabled",
            type="checkbox",
            label="Enable TeaCache",
            default=False,
            group="enhancement",
            tooltip="Temporal-Efficient Attention Cache for faster generation.",
        ),
        ParamSchema(
            id="teacache_threshold",
            type="slider",
            label="TeaCache Threshold",
            default=0.15,
            min=0.05,
            max=0.5,
            step=0.01,
            group="enhancement",
            conditional={"teacache_enabled": True},
            tooltip="Cache similarity threshold. Lower = more caching.",
        ),

        # === Advanced ===
        ParamSchema(
            id="stg_enabled",
            type="checkbox",
            label="Enable STG",
            default=False,
            group="advanced",
            tooltip="Spatio-Temporal Guidance for better motion consistency.",
        ),
        ParamSchema(
            id="stg_scale",
            type="slider",
            label="STG Scale",
            default=1.0,
            min=0.0,
            max=3.0,
            step=0.1,
            group="advanced",
            conditional={"stg_enabled": True},
            tooltip="STG strength.",
        ),
        ParamSchema(
            id="stg_start_step",
            type="slider",
            label="STG Start",
            default=0.0,
            min=0.0,
            max=1.0,
            step=0.05,
            group="advanced",
            conditional={"stg_enabled": True},
            tooltip="When to start STG (fraction of steps).",
        ),
        ParamSchema(
            id="stg_end_step",
            type="slider",
            label="STG End",
            default=0.5,
            min=0.0,
            max=1.0,
            step=0.05,
            group="advanced",
            conditional={"stg_enabled": True},
            tooltip="When to end STG (fraction of steps).",
        ),
        ParamSchema(
            id="rescale_scale",
            type="slider",
            label="CFG Rescale",
            default=0.7,
            min=0.0,
            max=1.0,
            step=0.05,
            group="advanced",
            conditional={"use_two_stage": True},
            tooltip="CFG rescaling factor for two-stage generation. 0.7 recommended.",
        ),
        ParamSchema(
            id="ge_gamma",
            type="slider",
            label="GE Gamma",
            default=0.0,
            min=0.0,
            max=1.0,
            step=0.05,
            group="advanced",
            conditional={"use_two_stage": True},
            tooltip="Gradient estimation gamma. 0.0 disables. Used for quality refinement in two-stage.",
        ),
    ],
))
