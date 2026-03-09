"""
LTX-2.3 Diagnostic Generation Script.

Runs a minimal generation with diagnostic logging enabled to identify
where the signal breaks down in the pipeline.

Usage:
    uv run python scripts/diagnose_ltx23.py
"""

import logging
import sys
from pathlib import Path

# Enable diagnostic logging BEFORE importing anything else
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
    stream=sys.stdout,
)
# Enable the diagnostic logger at INFO level
logging.getLogger("llm_dit.pipelines.generate.diagnostics").setLevel(logging.INFO)
# Also enable debug on the main generate logger for sigma details
logging.getLogger("llm_dit.pipelines.generate").setLevel(logging.DEBUG)
# Enable encoder debug logging
logging.getLogger("llm_dit.encoders.gemma3").setLevel(logging.DEBUG)
logging.getLogger("llm_dit.encoders.embeddings_connector").setLevel(logging.DEBUG)

import torch
from llm_dit.config import load_config
from llm_dit.pipelines.generate import (
    GenerationConfig,
    TwoStageConfig,
    generate_video_two_stage,
)


def main():
    config = load_config("config.toml")
    ltx2_cfg = config.ltx2

    model_path = Path(ltx2_cfg.model_path)
    print(f"\n{'='*60}")
    print(f"LTX-2.3 Diagnostic Generation")
    print(f"Model path: {model_path}")
    print(f"{'='*60}\n")

    # Production-like config: 30 non-distilled steps, CFG=3.0
    gen_config = GenerationConfig(
        num_frames=9,       # Minimum viable (1 latent frame + 1)
        height=512,
        width=512,
        seed=42,
        fps=ltx2_cfg.fps,
    )

    two_stage = TwoStageConfig(
        stage1_steps=30,    # Production: 30 non-distilled steps
        guidance_scale=3.0,
        stage2_steps=3,
        distilled_lora_path=ltx2_cfg.distilled_lora_path,
        distilled_lora_scale=1.0,
        spatial_upsampler_file=ltx2_cfg.spatial_upsampler_file,
        stg_scale=0.0,      # Disable STG for simplicity
        modality_scale=1.0,  # Disable modality guidance
        rescale_scale=0.0,   # Disable rescaling
        ge_gamma=0.0,
    )

    print(f"Config: {gen_config.num_frames}f @ {gen_config.height}x{gen_config.width}")
    print(f"Stage1: {two_stage.stage1_steps} steps, cfg={two_stage.guidance_scale}")
    print(f"Stage2: {two_stage.stage2_steps} steps (distilled)")
    print(f"LoRA: {two_stage.distilled_lora_path}")
    print(f"Upsampler: {two_stage.spatial_upsampler_file}")
    print()

    video = generate_video_two_stage(
        prompt="A cat sitting on a windowsill",
        config=gen_config,
        two_stage=two_stage,
        model_path=str(model_path),
        text_encoder_path=str(model_path / "text_encoder"),
        dtype=torch.bfloat16,
        gemma_variant=ltx2_cfg.gemma_variant,
        transformer_file=ltx2_cfg.transformer_file,
        transformer_device="cuda",
        vae_device="cuda",
        text_encoder_device="cpu",
        quantize="none",  # No quantization for cleaner diagnostics
        video_only=True,
    )

    if isinstance(video, tuple):
        video = video[0]

    print(f"\n{'='*60}")
    print(f"OUTPUT: shape={list(video.shape)}, dtype={video.dtype}")
    print(f"  mean={video.float().mean():.1f}, min={video.min()}, max={video.max()}")

    # Check if output is noise-like (uniform mean ~127, high variance across frames)
    per_frame_means = video.float().mean(dim=(1, 2, 3))
    print(f"  per_frame_means={[f'{m:.1f}' for m in per_frame_means.tolist()]}")

    # Save first frame for visual inspection
    from PIL import Image
    frame = video[0].cpu().numpy()
    img = Image.fromarray(frame)
    out_path = Path("outputs/diagnostic_frame.png")
    out_path.parent.mkdir(exist_ok=True)
    img.save(str(out_path))
    print(f"  First frame saved to: {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
