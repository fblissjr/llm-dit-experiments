"""
Qwen-Image Pipeline Schemas

last updated: 2026-02-08

Qwen-Image has two distinct variants:
- qwenimage-t2i: Text-to-image generation (40 steps, 1024 resolution)
- qwenimage-edit: Image editing with instructions (25 steps, 640 resolution)

Both variants share the 60-layer DiT architecture with 2x2 latent packing.
"""

from . import register_pipeline, PipelineSchema, ParamSchema


# Common quantization options for all Qwen variants (unified torchao methods)
QUANTIZATION_OPTIONS = ["none", "fp8-dynamic", "fp8-weight-only", "int8", "int4"]

# Resolution options (Qwen models are trained on specific resolutions)
RESOLUTION_OPTIONS_640 = ["640x640", "512x768", "768x512"]
RESOLUTION_OPTIONS_1024 = ["1024x1024", "768x1280", "1280x768"]

# Offload type options
OFFLOAD_OPTIONS = ["none", "model", "sequential"]


# === Qwen-Image T2I (Text-to-Image) ===
register_pipeline(PipelineSchema(
    id="qwenimage-t2i",
    name="Qwen-Image T2I",
    description="High-quality text-to-image generation at 1024px resolution",
    output_type="image",
    color="teal",
    icon="🖼️",
    category="image",
    supports_history=True,
    endpoint="/api/qwen-image/t2i/generate",
    params=[
        # === Basic Parameters ===
        ParamSchema(
            id="prompt",
            type="textarea",
            label="Prompt",
            placeholder="Describe the image...",
            rows=4,
            group="basic",
            required=True,
            tooltip="Detailed description of the image to generate.",
            config_mapped=False,
        ),
        ParamSchema(
            id="negative_prompt",
            type="textarea",
            label="Negative Prompt",
            placeholder="Elements to avoid...",
            rows=2,
            default="worst quality, blurry, distorted",
            group="basic",
            tooltip="What to avoid in the generation.",
            config_mapped=False,
        ),
        ParamSchema(
            id="width",
            type="number",
            label="Width",
            default=1024,
            min=512,
            max=1280,
            step=64,
            group="basic",
            tooltip="Image width (1024 recommended).",
            config_mapped=False,
        ),
        ParamSchema(
            id="height",
            type="number",
            label="Height",
            default=1024,
            min=512,
            max=1280,
            step=64,
            group="basic",
            tooltip="Image height (1024 recommended).",
            config_mapped=False,
        ),
        ParamSchema(
            id="steps",
            type="slider",
            label="Steps",
            default=40,
            min=10,
            max=100,
            step=1,
            group="basic",
            tooltip="Denoising steps. 40 is the default for T2I.",
        ),
        ParamSchema(
            id="cfg_scale",
            type="slider",
            label="CFG Scale",
            default=4.0,
            min=1.0,
            max=15.0,
            step=0.5,
            group="basic",
            tooltip="Classifier-free guidance scale.",
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
            tooltip="Random seed. -1 for random.",
            config_mapped=False,
        ),

        # === Memory & Performance ===
        ParamSchema(
            id="quantization",
            type="select",
            label="Quantization",
            default="none",
            options=QUANTIZATION_OPTIONS,
            group="optimization",
            tooltip="Quantization for reduced VRAM. FP8 recommended on RTX 4090.",
            config_mapped=False,
        ),
        ParamSchema(
            id="offload_type",
            type="select",
            label="Offload",
            default="none",
            options=OFFLOAD_OPTIONS,
            group="optimization",
            tooltip="CPU offloading strategy.",
        ),
    ],
))


# === Qwen-Image Edit ===
register_pipeline(PipelineSchema(
    id="qwenimage-edit",
    name="Qwen-Image Edit",
    description="Instruction-based image editing at 640px resolution",
    output_type="image",
    color="green",
    icon="✏️",
    category="image",
    supports_history=True,
    supports_img2img=True,
    endpoint="/api/qwen-image/edit/generate",
    params=[
        # === Basic Parameters ===
        ParamSchema(
            id="image",
            type="image",
            label="Input Image",
            group="basic",
            required=True,
            tooltip="The image to edit.",
            config_mapped=False,
        ),
        ParamSchema(
            id="instruction",
            type="textarea",
            label="Edit Instruction",
            placeholder="Describe how to modify the image...",
            rows=3,
            group="basic",
            required=True,
            tooltip="Natural language instruction for editing (e.g., 'make the sky sunset colors').",
            config_mapped=False,
        ),
        ParamSchema(
            id="resolution",
            type="select",
            label="Resolution",
            default="640x640",
            options=RESOLUTION_OPTIONS_640,
            group="basic",
            tooltip="Output resolution. 640x640 recommended for editing.",
        ),
        ParamSchema(
            id="steps",
            type="slider",
            label="Steps",
            default=25,
            min=10,
            max=50,
            step=1,
            group="basic",
            tooltip="Denoising steps. 25 is default for editing.",
        ),
        ParamSchema(
            id="cfg_scale",
            type="slider",
            label="CFG Scale",
            default=4.0,
            min=1.0,
            max=10.0,
            step=0.5,
            group="basic",
            tooltip="Guidance strength.",
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
            tooltip="Random seed. -1 for random.",
            config_mapped=False,
        ),

        # === Memory ===
        ParamSchema(
            id="quantization",
            type="select",
            label="Quantization",
            default="none",
            options=QUANTIZATION_OPTIONS,
            group="optimization",
            tooltip="Quantization for reduced VRAM.",
            config_mapped=False,
        ),
    ],
))
