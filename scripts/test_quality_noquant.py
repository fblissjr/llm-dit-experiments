"""
Test without FP8 quantization to rule out quantization effects.

Last Updated: 2026-01-20

This is identical to test_quality.py but with quantize=False.
Memory requirement: ~23GB (bf16 transformer ~19GB + VAE ~2GB + overhead)
"""
import torch


def main():
    from llm_dit.pipelines.generate import GenerationConfig, generate_video_with_offloading

    # Official LTX-2 defaults
    config = GenerationConfig(
        num_frames=9,
        height=256,
        width=384,
        num_inference_steps=40,
        guidance_scale=4.0,
        seed=10,
    )

    print(f"Config: {config.num_frames} frames @ {config.height}x{config.width}")
    print(f"Steps: {config.num_inference_steps}, CFG: {config.guidance_scale}")
    print("Running WITHOUT FP8 quantization (bf16 transformer)")

    video = generate_video_with_offloading(
        prompt="A cat walking through a sunny garden",
        config=config,
        model_path="models/LTX-2",
        quantize=False,  # <-- No FP8 quantization
        precision="bf16",
        dtype=torch.bfloat16,
    )

    print(f"\nOutput: {video.shape}")
    print(f"Min: {video.min()}, Max: {video.max()}")

    # Save frames
    from PIL import Image
    import os
    os.makedirs("outputs/test_quality_noquant", exist_ok=True)
    for i in range(video.shape[0]):
        frame = video[i].cpu().numpy()
        img = Image.fromarray(frame)
        img.save(f"outputs/test_quality_noquant/frame_{i:03d}.png")
    print("Saved frames to outputs/test_quality_noquant/")


if __name__ == "__main__":
    main()
