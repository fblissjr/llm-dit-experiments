last updated: 2026-02-02

# ltx-2 prompting guide

This guide describes how to write effective prompts for LTX-2 video generation. LTX-2 was trained on CogVLM2-Video captions using a specific narrative style that produces the best results.

## core principles

### 1. narrative screenplay format

Write prompts as flowing narrative paragraphs, not structured templates. LTX-2 responds best to screenplay-style descriptions that read like a scene breakdown.

**Do:**
```
The sun rises over a misty mountain range, casting golden light across
alpine meadows dotted with wildflowers. A lone hiker emerges from a
forest path, pausing to take in the breathtaking vista.
```

**Do not:**
```
[VISUAL]: sunrise, mountains, mist
[SUBJECT]: hiker, backpack
[ACTION]: walking, stopping
```

### 2. present tense

Always describe actions in present tense to create a sense of immediacy.

**Do:** "The car speeds down the highway"
**Do not:** "The car sped down the highway"

### 3. six key aspects

Structure your prompts to address these six aspects of the scene:

| Aspect | What to Include | Example |
|--------|-----------------|---------|
| **Shot** | Camera distance and framing | "tight cinematic close-up", "wide establishing shot" |
| **Scene** | Location, lighting, atmosphere | "warm sunny backyard", "dimly lit jazz club" |
| **Action** | What happens, in temporal order | "walks through the door, pauses, looks around" |
| **Characters** | Physical descriptions, not personality | "a woman in her 30s with curly red hair" |
| **Camera** | Movement and technique | "the camera slowly pans right", "handheld tracking shot" |
| **Audio** | Ambient sounds, dialogue | "birds chirping", "she says softly, 'Hello'" |

### 4. physical cues for emotion

Express emotion through observable physical actions, not abstract labels.

**Do:** "tears streaming down her face", "clenched fists at his sides", "trembling hands"
**Do not:** "she looks sad", "he feels angry", "nervous expression"

### 5. dialogue formatting

Write dialogue in quotation marks with optional accent/language specifications.

**Examples:**
- She whispers, "I knew you'd come back."
- He replies with a slight French accent, "Perhaps we should discuss this elsewhere."
- The child exclaims, "Look, a puppy!"

### 6. camera language

Use professional cinematography terms for camera movement:

| Term | Meaning |
|------|---------|
| dolly | Camera moves toward or away from subject |
| pan | Camera rotates horizontally |
| tilt | Camera rotates vertically |
| track | Camera moves parallel to subject |
| crane | Camera moves up or down on boom |
| handheld | Naturalistic, slightly shaky movement |
| steadicam | Smooth tracking without rails |
| zoom | Focal length changes (in/out) |

## what to avoid

### content that doesn't work well

| Issue | Why It Fails | Alternative |
|-------|--------------|-------------|
| Text/logo generation | Model not trained for text | Describe signs conceptually |
| Complex physics | Juggling, gymnastics, etc. | Simpler actions |
| Many characters | Coherence issues | 1-3 subjects maximum |
| Conflicting lighting | Multiple light sources | Single dominant light |
| Scene overload | Too many concurrent actions | Focus on one action arc |
| Abstract concepts | "Love", "freedom", etc. | Physical manifestations |

### format mistakes

- **Templates:** Do not use `[VISUAL]:`, `[SPEECH]:`, or similar markers
- **Bullet lists:** Write as prose, not bulleted points
- **Minimal prompts:** "Cat walking" is too sparse; add scene context
- **Run-on complexity:** Keep sentences digestible

## example prompts

### simple (good for testing)

```
A cat walking across a sunlit wooden floor, its shadow stretching
behind it. The camera follows from a low angle as afternoon light
streams through nearby windows.
```

### moderate (balanced complexity)

```
A golden retriever puppy runs through a sun-dappled forest trail,
kicking up fallen autumn leaves. The camera follows low to the ground,
capturing the dog's joyful expression as it bounds forward. Warm
afternoon light filters through the canopy above, creating dynamic
shadows that dance across the scene.
```

### complex (multi-scene narrative)

```
A warm sunny backyard. The camera starts in a tight cinematic close-up
of a woman and a man in their 30s, facing each other with serious
expressions. The woman, emotional and dramatic, says softly, "That's
it... Dad's lost it." The man exhales, slightly annoyed: "Stop being
so dramatic." The camera slowly pans right, revealing the grandfather
in the garden wearing enormous butterfly wings, dancing among the
tomato plants with a watering can.
```

### cinematic (technical precision)

```
Establishing wide shot of Manhattan at dusk. The camera cranes down
toward street level as city lights begin to flicker on. Traffic flows
like rivers of light through the grid of avenues. A slow dolly follows
a yellow cab as it weaves between lanes, the ambient hum of the city
mixing with distant sirens. The shot settles on a corner diner, its
neon sign buzzing to life in the gathering twilight.
```

## recommended parameters

Based on official LTX-2 guidelines:

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| guidance_scale | 3.0 | 2.0-5.0 | Higher = more prompt adherence |
| steps | 40 | 30-50 | 40 is official default |
| stg_scale | 1.0 | 0.0-2.0 | Spatio-temporal guidance |
| rescale_scale | 0.7 | 0.0-1.0 | Reduces oversaturation |
| num_frames | 121 | 9-161 | Must be (8*k)+1 |
| frame_rate | 24.0 | 24.0 | Standard film rate |

### frame count guidelines

LTX-2 requires frame counts in the form `(8 * k) + 1`:

| Frames | Duration @24fps | Use Case |
|--------|-----------------|----------|
| 9 | 0.375s | Smoke tests |
| 33 | 1.375s | Quick validation |
| 65 | 2.7s | Short clips |
| 121 | 5.0s | Standard videos |
| 161 | 6.7s | Extended scenes |

## testing your prompts

Use the smoke test preset to quickly validate prompt quality:

```bash
uv run pytest tests/integration/pipeline/test_ltx2_baselines.py::TestLTX2Baselines::test_smoke_baseline_generation -v -s
```

For custom prompts:

```bash
uv run python -m llm_dit.pipelines.generate \
    --prompt "Your prompt here" \
    --output outputs/test.mp4 \
    --num-frames 33 --height 512 --width 768
```

## related documentation

- [ltx2_generate.md](ltx2_generate.md) - Full generation guide
- [internal/docs/ltx2/quickstart.md](../../../internal/docs/ltx2/quickstart.md) - Quick reference
- [presets/testing/ltx2_smoke_test.md](../../../presets/testing/ltx2_smoke_test.md) - Test presets
