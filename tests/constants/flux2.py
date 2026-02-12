"""FLUX.2 canonical test parameters.

last updated: 2026-02-12

No external reference repo for FLUX.2 -- values derived from our validated
smoke/standard configs and the model variant defaults in config.py.

FLUX.2 Klein distilled (9B FP8): 4 steps, guidance=1.0
FLUX.2 Klein base: 50 steps, guidance=4.0
"""

# =============================================================================
# Reference values (from our config.py Flux2Config defaults)
# =============================================================================

REFERENCE_SEED = 42
REFERENCE_HEIGHT = 1024
REFERENCE_WIDTH = 1024
REFERENCE_STEPS = 20                   # Reasonable quality for non-distilled
REFERENCE_GUIDANCE = 3.5               # Standard CFG for base model

# Distilled model defaults
DISTILLED_STEPS = 4
DISTILLED_GUIDANCE = 1.0


# =============================================================================
# Test tiers
# =============================================================================

SMOKE = {
    "height": 256,
    "width": 256,
    "num_inference_steps": 2,
    "guidance_scale": DISTILLED_GUIDANCE,  # 1.0
    "seed": REFERENCE_SEED,                # 42
    "quantization": "fp8-weight-only",
    "default_model": "klein-9b-fp8",
}

STANDARD = {
    "height": 512,
    "width": 512,
    "num_inference_steps": 4,
    "guidance_scale": DISTILLED_GUIDANCE,  # 1.0
    "seed": REFERENCE_SEED,                # 42
    "quantization": "fp8-weight-only",
    "default_model": "klein-9b-fp8",
}

REFERENCE = {
    "height": REFERENCE_HEIGHT,            # 1024
    "width": REFERENCE_WIDTH,              # 1024
    "num_inference_steps": REFERENCE_STEPS,  # 20
    "guidance_scale": REFERENCE_GUIDANCE,  # 3.5
    "seed": REFERENCE_SEED,                # 42
    "quantization": "none",
    "default_model": "klein-9b",
}


# =============================================================================
# Test prompts
# =============================================================================

SMOKE_PROMPT = "A red balloon floating in a clear blue sky"

PROMPTS = {
    "smoke": SMOKE_PROMPT,
}
