"""
FLUX.2 Pipeline Schema

last updated: 2026-01-31

FLUX.2 is an image generation pipeline with:
- Klein models: 4 distilled + 4 base variants
- Reference image support for style/subject transfer
- Block-by-block offloading for low VRAM
- FP8 variants for reduced memory usage

NOTE: Model names MUST match keys in FLUX2_MODEL_INFO (constants.py)
"""

from . import register_pipeline, PipelineSchema, ParamSchema


# FLUX.2 model variants - must match keys in FLUX2_MODEL_INFO
# Note: Only 9B variants included (4B not in use)
FLUX2_MODELS = [
    # Distilled models (fast, 4 steps)
    "klein-9b",
    "klein-9b-fp8",
    # Base models (quality, 50 steps)
    "klein-base-9b",
    "klein-base-9b-fp8",
]

# Dimension presets (all multiples of 16)
DIMENSION_PRESETS = [
    "1024x1024",
    "1152x896",
    "1216x832",
    "1344x768",
    "768x1344",
    "832x1216",
    "896x1152",
    "Custom",
]


register_pipeline(PipelineSchema(
    id="flux2",
    name="FLUX.2",
    description="High-fidelity image generation with reference image support",
    output_type="image",
    color="orange",
    icon="⚡",
    category="image",
    supports_history=True,
    supports_img2img=False,
    supports_reference_images=True,
    supports_streaming=True,  # Enable SSE progress streaming
    endpoint="/api/flux2/generate/stream",  # Stream endpoint for progress updates
    params=[
        # === Prompt ===
        ParamSchema(
            id="prompt",
            type="textarea",
            label="Prompt",
            placeholder="Describe the image you want to generate...",
            rows=4,
            group="basic",
            required=True,
            tooltip="Detailed description of the image.",
        ),

        # === Reference Images (key FLUX.2 feature - right after prompt) ===
        ParamSchema(
            id="reference_images",
            type="image",
            label="Reference Images",
            group="basic",
            tooltip="Upload reference images for style/subject transfer (up to 4 images).",
            max_count=4,
        ),
        ParamSchema(
            id="match_image_size",
            type="select",
            label="Match Output to Reference",
            default="0 (First Image)",
            options=[
                "none",
                "0 (First Image)",
                "1 (Second Image)",
                "2 (Third Image)",
                "3 (Fourth Image)",
            ],
            group="basic",
            tooltip="Match output dimensions to a reference image. Prevents squishing when ref has different aspect ratio.",
        ),
        # === Model & Dimensions ===
        ParamSchema(
            id="model_name",
            type="select",
            label="Model",
            default="klein-9b-fp8",
            options=FLUX2_MODELS,
            group="basic",
            tooltip="Model variant. Distilled = fast (4 steps), Base = quality (50 steps). FP8 saves VRAM.",
        ),
        ParamSchema(
            id="width",
            type="number",
            label="Width",
            default=1024,
            min=256,
            max=2048,
            step=16,
            group="basic",
            tooltip="Image width in pixels (multiple of 16).",
        ),
        ParamSchema(
            id="height",
            type="number",
            label="Height",
            default=1024,
            min=256,
            max=2048,
            step=16,
            group="basic",
            tooltip="Image height in pixels (multiple of 16).",
        ),
        ParamSchema(
            id="dimension_preset",
            type="select",
            label="Preset",
            default="1024x1024",
            options=DIMENSION_PRESETS,
            group="basic",
            tooltip="Quick dimension presets.",
        ),

        # === Generation Settings ===
        ParamSchema(
            id="num_steps",
            type="slider",
            label="Steps",
            default=4,
            min=1,
            max=50,
            step=1,
            group="basic",
            tooltip="Inference steps. 4 for distilled, 50 for base models.",
        ),
        ParamSchema(
            id="guidance",
            type="slider",
            label="Guidance",
            default=1.0,
            min=0.0,
            max=10.0,
            step=0.1,
            group="basic",
            tooltip="Guidance strength. 1.0 for distilled, 3.5-4.0 for base.",
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
            id="block_offload",
            type="checkbox",
            label="Block Offload",
            default=True,
            group="optimization",
            tooltip="Enable block-by-block CPU offloading for low VRAM systems.",
        ),
        ParamSchema(
            id="compile",
            type="checkbox",
            label="Torch Compile",
            default=False,
            group="optimization",
            tooltip="Use torch.compile for faster inference (slow first run).",
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
        ),
    ],
))
