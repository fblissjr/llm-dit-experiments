"""
FLUX.2 Pipeline Schema

last updated: 2026-01-25

FLUX.2 is an image generation pipeline with:
- Klein models: 4 distilled + 4 base variants
- Reference image support for style/subject transfer
- Block-by-block offloading for low VRAM
- FP8 variants for reduced memory usage
"""

from . import register_pipeline, PipelineSchema, ParamSchema


# FLUX.2 model variants
FLUX2_MODELS = [
    # Distilled models (fast, 4 steps)
    "flux2-klein-4b-distilled",
    "flux2-klein-4b-distilled-fp8",
    "flux2-klein-9b-distilled",
    "flux2-klein-9b-distilled-fp8",
    # Base models (quality, 50 steps)
    "flux2-klein-4b-base",
    "flux2-klein-4b-base-fp8",
    "flux2-klein-9b-base",
    "flux2-klein-9b-base-fp8",
]

# Dimension presets
DIMENSION_PRESETS = [
    "1024x1024",
    "1152x896",
    "1216x832",
    "1344x768",
    "768x1344",
    "832x1216",
    "896x1152",
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
    endpoint="/api/flux2/generate",
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
            tooltip="Detailed description of the image.",
        ),
        ParamSchema(
            id="model_name",
            type="select",
            label="Model",
            default="flux2-klein-4b-distilled",
            options=FLUX2_MODELS,
            group="basic",
            tooltip="Model variant. Distilled = fast (4 steps), Base = quality (50 steps). FP8 saves VRAM.",
        ),
        ParamSchema(
            id="width",
            type="number",
            label="Width",
            default=1024,
            min=512,
            max=2048,
            step=64,
            group="basic",
            tooltip="Image width in pixels.",
        ),
        ParamSchema(
            id="height",
            type="number",
            label="Height",
            default=1024,
            min=512,
            max=2048,
            step=64,
            group="basic",
            tooltip="Image height in pixels.",
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

        # === Reference Images ===
        ParamSchema(
            id="reference_images",
            type="image",
            label="Reference Images",
            group="advanced",
            tooltip="Upload reference images for style/subject transfer (up to 4 images).",
        ),
        ParamSchema(
            id="reference_strength",
            type="slider",
            label="Reference Strength",
            default=0.8,
            min=0.0,
            max=1.0,
            step=0.05,
            group="advanced",
            tooltip="How strongly to use reference images.",
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
    ],
))
