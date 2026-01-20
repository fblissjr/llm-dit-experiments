#!/usr/bin/env python3
"""
Run reference LTX-2 pipeline with 8-bit Gemma quantization.

Uses the same approach as ltx-trainer's gemma_8bit.py to load Gemma
with bitsandbytes 8-bit quantization and device_map="auto".
"""

import logging
import sys
from pathlib import Path

# Add reference code paths
sys.path.insert(0, str(Path(__file__).parent.parent / "coderef/LTX-2/packages/ltx-core/src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "coderef/LTX-2/packages/ltx-pipelines/src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "coderef/LTX-2/packages/ltx-trainer/src"))

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 70)
    logger.info("REFERENCE PIPELINE TEST WITH 8-BIT GEMMA")
    logger.info("=" * 70)

    # Paths
    checkpoint_path = "models/LTX-2/ltx-2-19b-dev-fp8.safetensors"
    gemma_path = Path.home() / ".cache/huggingface/hub/models--google--gemma-3-12b-it-qat-q4_0-unquantized/snapshots/68f7ee4fbd59087436ada77ed2d62f373fdd4482"

    # Import after path setup
    from ltx_trainer.gemma_8bit import load_8bit_gemma

    # Load text encoder with 8-bit quantization
    logger.info("\nLoading 8-bit Gemma text encoder...")
    text_encoder = load_8bit_gemma(
        checkpoint_path=checkpoint_path,
        gemma_model_path=str(gemma_path),
    )
    # Set to evaluation mode
    text_encoder.train(False)

    # Test encoding
    # Import pipeline components
    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.components.guiders import CFGGuider
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.model.video_vae import decode_video as vae_decode_video
    from ltx_core.text_encoders.gemma import encode_text
    from ltx_core.types import VideoPixelShape
    from ltx_pipelines.utils.helpers import (
        cleanup_memory,
        denoise_audio_video,
        euler_denoising_loop,
        guider_denoising_func,
        image_conditionings_by_replacing_latent,
    )
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils import ModelLedger

    # Parameters
    prompt = "A fluffy orange cat sleeping peacefully on a soft red couch"
    negative_prompt = ""
    seed = 42
    height = 512
    width = 768
    num_frames = 65  # Smaller for memory
    frame_rate = 25.0
    num_inference_steps = 30
    cfg_guidance_scale = 4.0
    output_path = "outputs/reference_output.mp4"

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Encode both prompts at once using already-loaded text encoder
    logger.info(f"\nEncoding prompts: '{prompt}' and '{negative_prompt}'")
    with torch.inference_mode():
        context_p, context_n = encode_text(text_encoder, prompts=[prompt, negative_prompt])
        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = context_n

    logger.info(f"Positive video context shape: {v_context_p.shape}")
    logger.info(f"Positive video context mean: {v_context_p.float().mean():.4f}")
    logger.info(f"Positive video context std: {v_context_p.float().std():.4f}")

    # Per-dim analysis
    v_flat = v_context_p[0].float()
    dim_means = v_flat.mean(dim=0)
    logger.info(f"Per-dim mean range: [{dim_means.min():.2f}, {dim_means.max():.2f}]")
    logger.info(f"Dims with |mean| > 5: {(dim_means.abs() > 5).sum().item()}")

    # Free text encoder
    logger.info("\nFreeing text encoder memory...")
    del text_encoder
    cleanup_memory()

    # Load other models via ModelLedger (uses the checkpoint for everything else)
    logger.info("\nLoading transformer and VAE via ModelLedger...")
    model_ledger = ModelLedger(
        dtype=dtype,
        device=device,
        checkpoint_path=checkpoint_path,
        gemma_root_path=None,  # We already have text embeddings
        fp8transformer=True,
    )

    # Load video encoder and transformer
    video_encoder = model_ledger.video_encoder()
    transformer = model_ledger.transformer()

    # Set up diffusion
    generator = torch.Generator(device=device).manual_seed(seed)
    noiser = GaussianNoiser(generator=generator)
    stepper = EulerDiffusionStep()
    cfg_guider = CFGGuider(cfg_guidance_scale)
    sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=device)

    # Set up denoising loop
    def denoising_loop(sigmas, video_state, audio_state, stepper):
        return euler_denoising_loop(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper,
            denoise_fn=guider_denoising_func(
                cfg_guider,
                v_context_p,
                v_context_n,
                a_context_p,
                a_context_n,
                transformer=transformer,
            ),
        )

    # Set up output shape
    output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)

    # Image conditionings (none for pure text-to-video)
    conditionings = image_conditionings_by_replacing_latent(
        images=[],
        height=height,
        width=width,
        video_encoder=video_encoder,
        dtype=dtype,
        device=device,
    )

    # Import pipeline components
    from ltx_pipelines.utils.types import PipelineComponents
    pipeline_components = PipelineComponents(dtype=dtype, device=device)

    # Run denoising
    logger.info(f"\nRunning diffusion ({num_inference_steps} steps)...")
    video_state, audio_state = denoise_audio_video(
        output_shape=output_shape,
        conditionings=conditionings,
        noiser=noiser,
        sigmas=sigmas,
        stepper=stepper,
        denoising_loop_fn=denoising_loop,
        components=pipeline_components,
        dtype=dtype,
        device=device,
    )

    # Clean up transformer
    del transformer
    cleanup_memory()

    # Decode video
    logger.info("\nDecoding video...")
    video_decoder = model_ledger.video_decoder()
    decoded_video = vae_decode_video(video_state.latent, video_decoder, generator=generator)

    # Save video
    logger.info(f"\nSaving to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    encode_video(
        video=decoded_video,
        fps=frame_rate,
        audio=None,
        audio_sample_rate=None,
        output_path=output_path,
        video_chunks_number=1,
    )

    logger.info("\n" + "=" * 70)
    logger.info(f"SUCCESS: Video saved to {output_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
