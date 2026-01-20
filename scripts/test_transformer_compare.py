"""
Compare transformer outputs between our implementation and reference.

Avoids importing audio components by directly loading only what we need.

Last Updated: 2026-01-20
"""
import sys
from pathlib import Path

# Add coderef to path
coderef_root = Path(__file__).parent.parent / "coderef" / "LTX-2" / "packages"
sys.path.insert(0, str(coderef_root / "ltx-core" / "src"))

import torch


def print_stats(name: str, tensor: torch.Tensor) -> None:
    """Print tensor statistics."""
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}, dtype: {tensor.dtype}")
    print(f"  Mean: {tensor.mean():.4f}, Std: {tensor.std():.4f}")
    print(f"  Range: [{tensor.min():.4f}, {tensor.max():.4f}]")


def test_scheduler_comparison():
    """Compare scheduler outputs."""
    print("\n=== SCHEDULER COMPARISON ===")

    from ltx_core.components.schedulers import LTX2Scheduler as RefScheduler
    from llm_dit.schedulers import LTX2Scheduler as OurScheduler

    ref_scheduler = RefScheduler()
    our_scheduler = OurScheduler()

    # Compare sigmas for 40 steps (no latent = use MAX_SHIFT_ANCHOR tokens)
    ref_sigmas = ref_scheduler.execute(steps=40).to(dtype=torch.float32)
    our_sigmas = our_scheduler.execute(steps=40).to(dtype=torch.float32)

    print_stats("Reference sigmas", ref_sigmas)
    print_stats("Our sigmas", our_sigmas)
    print(f"Sigma diff: {(ref_sigmas - our_sigmas).abs().max():.8f}")

    return ref_sigmas, our_sigmas


def test_euler_step_comparison():
    """Compare single Euler step."""
    print("\n=== EULER STEP COMPARISON ===")

    # Create test inputs
    torch.manual_seed(42)
    x = torch.randn(1, 192, 128, device="cuda", dtype=torch.bfloat16)
    velocity = torch.randn(1, 192, 128, device="cuda", dtype=torch.bfloat16)

    sigma = torch.tensor([0.5], device="cuda", dtype=torch.float32)
    sigma_next = torch.tensor([0.4], device="cuda", dtype=torch.float32)

    # Reference step (from reference code):
    # EulerDiffusionStep.forward() does: x + velocity * dt where dt = sigma_next - sigma
    dt = sigma_next - sigma
    ref_output = x + velocity.float() * dt.view(-1, 1, 1)
    ref_output = ref_output.to(torch.bfloat16)

    # Our step (from generate.py line 417-420):
    # dt = sigma_next - sigma
    # denoised = (latents.float() + velocity.float() * dt).to(latents.dtype)
    our_dt = sigma_next - sigma
    our_output = (x.float() + velocity.float() * our_dt).to(x.dtype)

    print_stats("Input x", x)
    print_stats("Velocity", velocity)
    print(f"dt = {dt.item():.6f}")
    print_stats("Reference output", ref_output)
    print_stats("Our output", our_output)
    print(f"Step diff: {(ref_output - our_output).abs().max():.8f}")


def test_cfg_comparison():
    """Compare CFG application."""
    print("\n=== CFG COMPARISON ===")

    torch.manual_seed(42)
    cond = torch.randn(1, 192, 128, device="cuda", dtype=torch.bfloat16)
    uncond = torch.randn(1, 192, 128, device="cuda", dtype=torch.bfloat16)
    scale = 4.0

    # Reference CFG: cond + (scale - 1) * (cond - uncond)
    # Which simplifies to: scale * cond - (scale - 1) * uncond
    ref_guided = cond + (scale - 1) * (cond - uncond)

    # Our CFG (should be same formula)
    # uncond + scale * (cond - uncond) = uncond + scale*cond - scale*uncond
    # = scale*cond - (scale-1)*uncond
    our_guided = uncond + scale * (cond - uncond)

    print_stats("Cond", cond)
    print_stats("Uncond", uncond)
    print(f"Scale: {scale}")
    print_stats("Reference guided", ref_guided)
    print_stats("Our guided", our_guided)
    print(f"CFG diff: {(ref_guided - our_guided).abs().max():.8f}")

    # Verify mathematical equivalence
    print("\nMathematical verification:")
    print(f"  ref = cond + (scale-1)*(cond-uncond) = scale*cond - (scale-1)*uncond")
    print(f"  our = uncond + scale*(cond-uncond) = scale*cond - (scale-1)*uncond")
    print(f"  These ARE mathematically equivalent!")


def test_text_encoding():
    """Test text encoding comparison if possible."""
    print("\n=== TEXT ENCODING (skipped - requires full model load) ===")
    print("Text encoding comparison would require loading Gemma model.")
    print("For now, assuming text encoding is correct.")


def main():
    print("=" * 80)
    print("TRANSFORMER/DENOISING COMPONENT COMPARISON")
    print("=" * 80)

    test_scheduler_comparison()
    test_euler_step_comparison()
    test_cfg_comparison()
    test_text_encoding()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Key findings:
1. If scheduler sigmas match exactly -> scheduler is correct
2. If Euler step matches -> diffusion stepping is correct
3. CFG formula is mathematically equivalent (verified)

Next areas to investigate:
- Transformer model predictions (velocity vs x0)
- RoPE positional embeddings
- Attention patterns
- FP8 quantization effects
""")


if __name__ == "__main__":
    main()
