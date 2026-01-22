"""
Quick test with more inference steps to check quality.

Last Updated: 2026-01-19
"""
import torch


def main():
    from llm_dit.pipelines.generate import GenerationConfig, generate_video_with_offloading

    # Official LTX-2 defaults from ltx_pipelines/utils/constants.py
    config = GenerationConfig(
        num_frames=9,  # Keep short for testing
        height=256,
        width=384,
        num_inference_steps=40,  # Official default
        guidance_scale=4.0,      # Official default
        seed=10,                 # Official default seed
    )

    print(f"Config: {config.num_frames} frames @ {config.height}x{config.width}")
    print(f"Steps: {config.num_inference_steps}, CFG: {config.guidance_scale}")

    video = generate_video_with_offloading(
        prompt="A cat walking through a sunny garden",
        config=config,
        model_path="models/LTX-2",
        quantize=True,
        precision="fp8-native",
        dtype=torch.bfloat16,
    )

    print(f"\nOutput: {video.shape}")
    print(f"Min: {video.min()}, Max: {video.max()}")

    # Save frames as images
    from PIL import Image
    import os
    os.makedirs("outputs/test_quality_40steps", exist_ok=True)
    for i in range(video.shape[0]):
        frame = video[i].cpu().numpy()
        img = Image.fromarray(frame)
        img.save(f"outputs/test_quality_40steps/frame_{i:03d}.png")
    print("Saved frames to outputs/test_quality_40steps/")


if __name__ == "__main__":
    main()
