"""LTX-2 canonical test parameters.

last updated: 2026-02-16

All values traced from the official LTX-2 reference implementation:
  coderef/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py

Two tier families:
  - Two-stage tiers (SMOKE, STANDARD): Used by TOML overlays / E2E API tests.
    These run the two-stage pipeline with distilled LoRA refinement.
  - Full model tiers (FULL_SMOKE, FULL_REFERENCE): Used by protocol.py / backend
    comparison tests. These run the full (non-distilled) model for 1:1 cross-
    implementation comparison.

Both families share resolution, CFG, seed, and FPS from the reference repo.
"""

# =============================================================================
# Reference values (from official LTX-2 constants.py)
# =============================================================================

REFERENCE_SEED = 10                    # DEFAULT_SEED
REFERENCE_HEIGHT = 512                 # DEFAULT_1_STAGE_HEIGHT
REFERENCE_WIDTH = 768                  # DEFAULT_1_STAGE_WIDTH
REFERENCE_FRAMES = 121                 # DEFAULT_NUM_FRAMES
REFERENCE_STEPS = 30                   # DEFAULT_NUM_INFERENCE_STEPS (official V2.3)
REFERENCE_CFG = 3.0                    # DEFAULT_VIDEO_GUIDER_PARAMS.cfg_scale
REFERENCE_STG = 1.0                    # stg_scale
REFERENCE_RESCALE = 0.7                # rescale_scale
REFERENCE_FPS = 24.0                   # DEFAULT_FRAME_RATE

# Noise schedule for distilled pipeline (verbatim from reference)
DISTILLED_SIGMA_VALUES = [
    1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0,
]

# Reduced schedule for super-resolution stage 2
STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]


# =============================================================================
# Two-stage tiers (TOML overlays / E2E API tests)
# =============================================================================

SMOKE = {
    "height": 256,                     # Scaled down for speed (landscape like reference)
    "width": 384,
    "num_frames": 9,                   # Minimum valid (8k+1)
    "guidance_scale": REFERENCE_CFG,   # 3.0
    "seed": REFERENCE_SEED,            # 10
    "fps": REFERENCE_FPS,              # 24.0
    "fp8": True,
    "use_two_stage": True,
    "stage1_num_inference_steps": 4,
}

STANDARD = {
    "height": REFERENCE_HEIGHT,        # 512
    "width": REFERENCE_WIDTH,          # 768
    "num_frames": 33,
    "guidance_scale": REFERENCE_CFG,   # 3.0
    "seed": REFERENCE_SEED,            # 10
    "fps": REFERENCE_FPS,              # 24.0
    "fp8": True,
    "use_two_stage": True,
    "stage1_num_inference_steps": 8,
}


# =============================================================================
# Full model tiers (protocol.py / backend comparison tests)
# =============================================================================
# These match the protocol.py values that integration tests rely on.
# Key difference: 30-40 steps (non-distilled) vs 4-12 steps (distilled).

FULL_SMOKE = {
    "height": REFERENCE_HEIGHT,        # 512
    "width": REFERENCE_WIDTH,          # 768
    "num_frames": 33,
    "num_inference_steps": 30,
    "guidance_scale": REFERENCE_CFG,   # 3.0
    "seed": REFERENCE_SEED,            # 10
    "fps": REFERENCE_FPS,              # 24.0
    "fp8": True,
    "use_two_stage": False,
}

FULL_REFERENCE = {
    "height": REFERENCE_HEIGHT,        # 512
    "width": REFERENCE_WIDTH,          # 768
    "num_frames": REFERENCE_FRAMES,    # 121
    "num_inference_steps": REFERENCE_STEPS,  # 30
    "guidance_scale": REFERENCE_CFG,   # 3.0
    "seed": REFERENCE_SEED,            # 10
    "fps": REFERENCE_FPS,              # 24.0
    "fp8": False,
    "use_two_stage": False,
}


# =============================================================================
# Test prompts
# =============================================================================

SMOKE_PROMPT = "A cat walking"

REFERENCE_PROMPTS = {
    "cat_walking": "A cat walking",
    "cat_playing": "A cat playing with a ball",
    "sunset": "A beautiful sunset over the ocean",
}

PROMPTS = {
    "smoke": SMOKE_PROMPT,
    "reference": REFERENCE_PROMPTS,
}
