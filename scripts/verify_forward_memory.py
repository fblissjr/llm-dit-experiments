"""
Verify memory usage during forward pass with FP8.

Last Updated: 2026-01-19

Tests memory consumption at each stage to identify OOM bottleneck.
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
    # 9 frames, 256x384 -> latent: [1, 128, 2, 8, 12] (F/8+1, H/32, W/32)
    batch_size = 1
    channels = 128
    num_frames_latent = 2  # (9-1)/8 + 1 = 2
    height_latent = 8  # 256/32
    width_latent = 12  # 384/32

    # Create inputs
    hidden_states = torch.randn(
        batch_size, channels, num_frames_latent, height_latent, width_latent,
        device="cuda", dtype=torch.bfloat16
    )

    # Timestep
    timestep = torch.tensor([0.5], device="cuda", dtype=torch.bfloat16)

    # Text embeddings (minimal)
    encoder_hidden_states = torch.randn(
        batch_size, 4, 3840,  # 4 tokens, 3840 dim
        device="cuda", dtype=torch.bfloat16
    )

    # Position indices
    seq_len = num_frames_latent * height_latent * width_latent
    indices = torch.arange(seq_len, device="cuda", dtype=torch.float32)
    frame_indices = indices // (height_latent * width_latent)
    height_indices = (indices % (height_latent * width_latent)) // width_latent
    width_indices = indices % width_latent

    indices_grid = torch.stack([
        frame_indices,
        height_indices,
        width_indices
    ], dim=0).unsqueeze(0)  # [1, 3, seq_len]

    log_memory("After input creation")

    # Forward pass
    print("\nRunning forward pass...")
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        try:
            output = model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                indices_grid=indices_grid,
            )
            log_memory("After forward pass")
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak memory during forward: {peak:.2f}GB")
            print(f"Output shape: {output.shape if hasattr(output, 'shape') else type(output)}")
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
            output1 = model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                indices_grid=indices_grid,
            )
            log_memory("After cond forward")

            # Second forward (unconditional)
            uncond_embeddings = torch.zeros_like(encoder_hidden_states)
            output2 = model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=uncond_embeddings,
                indices_grid=indices_grid,
            )
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
    del model, hidden_states, encoder_hidden_states
    gc.collect()
    torch.cuda.empty_cache()
    log_memory("After cleanup")


if __name__ == "__main__":
    main()
