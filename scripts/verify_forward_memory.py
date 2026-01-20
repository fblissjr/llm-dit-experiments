"""
Verify memory usage during forward pass with FP8.

Last Updated: 2026-01-19

Tests memory consumption at each stage to identify OOM bottleneck.
Uses proper Modality API matching LTX-2 reference implementation.
"""
import gc
import torch


def get_memory():
    """Get current GPU memory in GB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0, 0


def log_memory(stage: str):
    """Log memory at a specific stage."""
    allocated, reserved = get_memory()
    print(f"[{stage}] allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")


def main():
    print("=== Forward Pass Memory Test ===\n")

    log_memory("Initial")

    # Load model
    from llm_dit.models.ltx2.loader import load_ltx2_transformer_fp8_native

    print("\nLoading transformer with fp8-native...")
    model = load_ltx2_transformer_fp8_native(
        "models/LTX-2/transformer",
        dtype=torch.bfloat16,
        device="cuda"
    )
    model.eval()
    log_memory("After model load")

    # Create minimal test inputs matching smoke test config
    # 9 frames, 256x384 -> latent: [1, 192, 128] in token format (T=2*8*12)
    batch_size = 1
    num_frames = 9
    height = 256
    width = 384

    # Latent dimensions
    t_latent = (num_frames - 1) // 8 + 1  # 2
    h_latent = height // 32  # 8
    w_latent = width // 32  # 12
    num_tokens = t_latent * h_latent * w_latent  # 192
    latent_dim = 128

    print(f"\nLatent dims: t={t_latent}, h={h_latent}, w={w_latent}, num_tokens={num_tokens}")

    # Create inputs using Modality format
    from llm_dit.models.ltx2.components import Modality
    from llm_dit.pipelines.generate import create_position_indices

    # Latent in token format [B, T, D]
    latent = torch.randn(
        batch_size, num_tokens, latent_dim,
        device="cuda", dtype=torch.bfloat16
    )

    # Timestep as [B, T] (per-token for LTX-2)
    sigma = 0.5
    timestep = (torch.ones(batch_size, num_tokens, device="cuda", dtype=torch.bfloat16) * sigma * 1000)

    # Position indices [B, 3, T, 2]
    positions = create_position_indices(
        batch_size, num_frames, height, width, torch.device("cuda")
    )

    # Text embeddings (3840 dim for Gemma3 output)
    context_len = 128  # Typical context length
    context = torch.randn(
        batch_size, context_len, 3840,
        device="cuda", dtype=torch.bfloat16
    )

    # Create Modality object
    video_modality = Modality(
        latent=latent,
        timesteps=timestep,
        positions=positions,
        context=context,
        enabled=True,
        context_mask=None,
    )

    log_memory("After input creation")

    # Forward pass
    print("\nRunning forward pass...")
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        try:
            output, _ = model(video=video_modality)
            log_memory("After forward pass")
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak memory during forward: {peak:.2f}GB")
            print(f"Output shape: {output.shape}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                peak = torch.cuda.max_memory_allocated() / 1024**3
                print(f"OOM! Peak memory: {peak:.2f}GB")
                print(f"Error: {e}")
            else:
                raise

    # Test CFG scenario (2 forward passes)
    print("\n\nTesting CFG scenario (2 forward passes)...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    log_memory("Before CFG test")

    with torch.no_grad():
        try:
            # First forward (conditional)
            output1, _ = model(video=video_modality)
            log_memory("After cond forward")

            # Free intermediate memory
            del output1
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            log_memory("After cleanup cond")

            # Second forward (unconditional - zero context)
            uncond_modality = Modality(
                latent=latent,
                timesteps=timestep,
                positions=positions,
                context=torch.zeros_like(context),
                enabled=True,
                context_mask=None,
            )
            output2, _ = model(video=uncond_modality)
            log_memory("After uncond forward")

            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak memory during CFG: {peak:.2f}GB")

        except RuntimeError as e:
            if "out of memory" in str(e):
                peak = torch.cuda.max_memory_allocated() / 1024**3
                print(f"OOM during CFG! Peak memory: {peak:.2f}GB")
            else:
                raise

    # Cleanup
    del model, latent, context, positions
    gc.collect()
    torch.cuda.empty_cache()
    log_memory("After cleanup")


if __name__ == "__main__":
    main()
