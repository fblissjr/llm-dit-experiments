"""Z-Image canonical test parameters.

last updated: 2026-02-12

Z-Image has two model variants with fundamentally different parameters:
  - Turbo: 9 steps, no CFG (guidance=0.0), shift=3.0
  - Base: 30+ steps, CFG=4.0, shift=6.0

Both variants share resolution and seed defaults.
"""

# =============================================================================
# Reference values
# =============================================================================

REFERENCE_SEED = 42
REFERENCE_HEIGHT = 1024
REFERENCE_WIDTH = 1024

# Turbo variant defaults
TURBO_STEPS = 9
TURBO_GUIDANCE = 0.0
TURBO_SHIFT = 3.0

# Base variant defaults
BASE_STEPS = 30
BASE_GUIDANCE = 4.0
BASE_SHIFT = 6.0


# =============================================================================
# Test tiers
# =============================================================================

SMOKE = {
    "height": 256,
    "width": 256,
    "num_inference_steps": TURBO_STEPS,    # 9
    "guidance_scale": TURBO_GUIDANCE,      # 0.0
    "shift": TURBO_SHIFT,                  # 3.0
    "seed": REFERENCE_SEED,                # 42
    "variant": "turbo",
}

STANDARD_TURBO = {
    "height": 512,
    "width": 512,
    "num_inference_steps": TURBO_STEPS,    # 9
    "guidance_scale": TURBO_GUIDANCE,      # 0.0
    "shift": TURBO_SHIFT,                  # 3.0
    "seed": REFERENCE_SEED,                # 42
    "variant": "turbo",
}

STANDARD_BASE = {
    "height": 512,
    "width": 512,
    "num_inference_steps": BASE_STEPS,     # 30
    "guidance_scale": BASE_GUIDANCE,       # 4.0
    "shift": BASE_SHIFT,                   # 6.0
    "seed": REFERENCE_SEED,                # 42
    "variant": "base",
}

REFERENCE_BASE = {
    "height": REFERENCE_HEIGHT,            # 1024
    "width": REFERENCE_WIDTH,              # 1024
    "num_inference_steps": BASE_STEPS,     # 30
    "guidance_scale": BASE_GUIDANCE,       # 4.0
    "shift": BASE_SHIFT,                   # 6.0
    "seed": REFERENCE_SEED,                # 42
    "variant": "base",
}


# =============================================================================
# Test prompts
# =============================================================================

SMOKE_PROMPT = "A serene mountain landscape at dawn"

PROMPTS = {
    "smoke": SMOKE_PROMPT,
}
