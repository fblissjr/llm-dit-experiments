"""
Test video generation using reference LTX-2 coderef implementation with debug statements.

This script uses the reference code from coderef/LTX-2/packages/ to generate a video
with the same settings as test_quality.py, adding debug statements at key points
to compare behavior with our implementation.

Last Updated: 2026-01-20
"""

import sys
from pathlib import Path

# Add coderef packages to path
coderef_root = Path(__file__).parent.parent / "coderef" / "LTX-2" / "packages"
sys.path.insert(0, str(coderef_root / "ltx-pipelines" / "src"))
sys.path.insert(0, str(coderef_root / "ltx-core" / "src"))

import os

import torch
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import CFGGuider
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.model.video_vae import decode_video
from ltx_core.text_encoders.gemma import encode_text
from ltx_core.types import VideoPixelShape
from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.helpers import (
    cleanup_memory,
    denoise_audio_video,
    euler_denoising_loop,
    get_device,
    guider_denoising_func,
)
from ltx_pipelines.utils.types import PipelineComponents


def print_tensor_stats(name: str, tensor: torch.Tensor) -> None:
    """Print detailed statistics about a tensor."""
    print(f"\n{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Min: {tensor.min().item():.6f}")
    print(f"  Max: {tensor.max().item():.6f}")
    print(f"  Mean: {tensor.mean().item():.6f}")
    print(f"  Std: {tensor.std().item():.6f}")


def monkeypatch_video_decoder_debug():
    """Monkeypatch VideoDecoder.forward to add debug statements."""
    from ltx_core.model.video_vae import VideoDecoder

    original_forward = VideoDecoder.forward

    def debug_forward(self, sample, timestep=None, generator=None):
        print("\n" + "="*80)
        print("VideoDecoder.forward() ENTRY")
        print("="*80)
        print_tensor_stats("Input latent (normalized)", sample)

        # Call un_normalize and print result
        batch_size = sample.shape[0]
        if self.timestep_conditioning:
            noise = (
                torch.randn(
                    sample.size(),
                    generator=generator,
                    dtype=sample.dtype,
                    device=sample.device,
                )
                * self.decode_noise_scale
            )
            sample = noise + (1.0 - self.decode_noise_scale) * sample
            print_tensor_stats("After noise injection", sample)

        # Denormalize
        sample_before_denorm = sample.clone()
        sample = self.per_channel_statistics.un_normalize(sample)
        print_tensor_stats("After un_normalize", sample)

        # Print the per-channel statistics used
        print(f"\nPer-channel statistics:")
        print(f"  Mean shape: {self.per_channel_statistics.mean.shape}")
        print(f"  Mean range: [{self.per_channel_statistics.mean.min():.6f}, {self.per_channel_statistics.mean.max():.6f}]")
        print(f"  Std shape: {self.per_channel_statistics.std.shape}")
        print(f"  Std range: [{self.per_channel_statistics.std.min():.6f}, {self.per_channel_statistics.std.max():.6f}]")

        # Continue with rest of forward pass
        if timestep is None and self.timestep_conditioning:
            timestep = torch.full((batch_size,), self.decode_timestep, device=sample.device, dtype=sample.dtype)

        sample = self.conv_in(sample, causal=self.causal)
        print_tensor_stats("After conv_in", sample)

        scaled_timestep = None
        if self.timestep_conditioning:
            if timestep is None:
                raise ValueError("'timestep' parameter must be provided when 'timestep_conditioning' is True")
            scaled_timestep = timestep * self.timestep_scale_multiplier.to(sample)

        for i, up_block in enumerate(self.up_blocks):
            from ltx_core.model.video_vae.resnet import UNetMidBlock3D, ResnetBlock3D
            if isinstance(up_block, UNetMidBlock3D):
                block_kwargs = {
                    "causal": self.causal,
                    "timestep": scaled_timestep if self.timestep_conditioning else None,
                    "generator": generator,
                }
                sample = up_block(sample, **block_kwargs)
            elif isinstance(up_block, ResnetBlock3D):
                sample = up_block(sample, causal=self.causal, generator=generator)
            else:
                sample = up_block(sample, causal=self.causal)

        sample = self.conv_norm_out(sample)

        if self.timestep_conditioning:
            embedded_timestep = self.last_time_embedder(
                timestep=scaled_timestep.flatten(),
                hidden_dtype=sample.dtype,
            )
            embedded_timestep = embedded_timestep.view(batch_size, embedded_timestep.shape[-1], 1, 1, 1)
            ada_values = self.last_scale_shift_table[None, ..., None, None, None].to(
                device=sample.device, dtype=sample.dtype
            ) + embedded_timestep.reshape(
                batch_size,
                2,
                -1,
                embedded_timestep.shape[-3],
                embedded_timestep.shape[-2],
                embedded_timestep.shape[-1],
            )
            shift, scale = ada_values.unbind(dim=1)
            sample = sample * (1 + scale) + shift

        sample = self.conv_act(sample)
        sample = self.conv_out(sample, causal=self.causal)
        print_tensor_stats("After conv_out (before unpatchify)", sample)

        from ltx_core.model.video_vae.ops import unpatchify
        sample = unpatchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)

        print_tensor_stats("After unpatchify (decoder output)", sample)
        print("="*80)
        print("VideoDecoder.forward() EXIT")
        print("="*80 + "\n")

        return sample

    VideoDecoder.forward = debug_forward


def monkeypatch_decode_video_debug():
    """Monkeypatch decode_video to add debug statements."""
    import ltx_core.model.video_vae as vae_module

    original_decode_video = vae_module.decode_video

    def debug_decode_video(latent, video_decoder, tiling_config=None, generator=None):
        """Decode with debug statements."""
        from einops import rearrange

        def convert_to_uint8(frames):
            print_tensor_stats("Decoder output (pre-conversion)", frames)

            # Show intermediate steps
            frames_shifted = (frames + 1.0) / 2.0
            print_tensor_stats("After (x + 1.0) / 2.0", frames_shifted)

            frames_clamped = frames_shifted.clamp(0.0, 1.0)
            print_tensor_stats("After clamp(0.0, 1.0)", frames_clamped)

            frames_scaled = frames_clamped * 255.0
            print_tensor_stats("After * 255.0", frames_scaled)

            frames_uint8 = frames_scaled.to(torch.uint8)
            print_tensor_stats("After to(uint8)", frames_uint8)

            frames = rearrange(frames_uint8[0], "c f h w -> f h w c")
            print_tensor_stats("After rearrange (final)", frames)

            return frames

        if tiling_config is not None:
            for frames in video_decoder.tiled_decode(latent, tiling_config, generator=generator):
                yield convert_to_uint8(frames)
        else:
            decoded_video = video_decoder(latent, generator=generator)
            yield convert_to_uint8(decoded_video)

    vae_module.decode_video = debug_decode_video


def main():
    # Apply debug monkeypatches
    print("Applying debug monkeypatches...")
    monkeypatch_video_decoder_debug()
    monkeypatch_decode_video_debug()

    # Configuration matching test_quality.py
    prompt = "A cat walking through a sunny garden"
    negative_prompt = ""
    seed = 10
    height = 256
    width = 384
    num_frames = 9
    frame_rate = 25.0
    num_inference_steps = 40
    cfg_guidance_scale = 4.0

    print("\n" + "="*80)
    print("REFERENCE LTX-2 CODE TEST")
    print("="*80)
    print(f"Prompt: {prompt}")
    print(f"Seed: {seed}")
    print(f"Resolution: {num_frames} frames @ {height}x{width}")
    print(f"Steps: {num_inference_steps}, CFG: {cfg_guidance_scale}")
    print("="*80 + "\n")

    device = get_device()
    dtype = torch.bfloat16

    # Model paths
    model_path = "models/LTX-2"
    checkpoint_path = os.path.join(model_path, "ltx_video_2_transformer.safetensors")
    gemma_root = os.path.join(model_path, "gemma-2-2b-it")

    # Initialize model ledger
    model_ledger = ModelLedger(
        dtype=dtype,
        device=device,
        checkpoint_path=checkpoint_path,
        gemma_root_path=gemma_root,
        loras=[],
        fp8transformer=False,
    )

    pipeline_components = PipelineComponents(
        dtype=dtype,
        device=device,
    )

    # Generator and pipeline components
    generator = torch.Generator(device=device).manual_seed(seed)
    noiser = GaussianNoiser(generator=generator)
    stepper = EulerDiffusionStep()
    cfg_guider = CFGGuider(cfg_guidance_scale)

    # Encode text
    print("\n--- Text Encoding ---")
    text_encoder = model_ledger.text_encoder()
    context_p, context_n = encode_text(text_encoder, prompts=[prompt, negative_prompt])
    v_context_p, a_context_p = context_p
    v_context_n, a_context_n = context_n
    print(f"Video context positive: {v_context_p.shape}")
    print(f"Audio context positive: {a_context_p.shape}")

    torch.cuda.synchronize()
    del text_encoder
    cleanup_memory()

    # Denoising
    print("\n--- Denoising ---")
    video_encoder = model_ledger.video_encoder()
    transformer = model_ledger.transformer()
    sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=device)
    print(f"Sigmas: {sigmas.shape}, range: [{sigmas.min():.6f}, {sigmas.max():.6f}]")

    def denoising_loop_fn(sigmas, video_state, audio_state, stepper):
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

    output_shape = VideoPixelShape(
        batch=1,
        frames=num_frames,
        width=width,
        height=height,
        fps=frame_rate
    )

    video_state, audio_state = denoise_audio_video(
        output_shape=output_shape,
        conditionings=[],
        noiser=noiser,
        sigmas=sigmas,
        stepper=stepper,
        denoising_loop_fn=denoising_loop_fn,
        components=pipeline_components,
        dtype=dtype,
        device=device,
    )

    print_tensor_stats("Final video latent", video_state.latent)

    torch.cuda.synchronize()
    del transformer
    cleanup_memory()

    # Decode video
    print("\n--- Video Decoding ---")
    video_decoder = model_ledger.video_decoder()

    # Decode (this will trigger our debug statements)
    frames_list = list(decode_video(video_state.latent, video_decoder, generator=generator))
    frames = frames_list[0]  # Single chunk for non-tiled decoding

    print(f"\n--- Final Output ---")
    print_tensor_stats("Final frames (uint8)", frames)

    # Save frames
    from PIL import Image
    output_dir = Path("outputs/test_quality_reference")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(frames.shape[0]):
        frame = frames[i].cpu().numpy()
        img = Image.fromarray(frame)
        img.save(output_dir / f"frame_{i:03d}.png")

    print(f"\nSaved {frames.shape[0]} frames to {output_dir}/")
    print("\n" + "="*80)
    print("REFERENCE TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
