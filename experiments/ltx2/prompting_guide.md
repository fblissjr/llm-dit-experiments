# LTX-2 Prompting Guide for Experiments

Last updated: 2026-01-16

> **CRITICAL**: This prompting style reflects LTX-2's training data distribution. Deviating from it creates **out-of-distribution inputs** that invalidate experiment results. All experiments MUST use properly structured prompts.

---

## Quick Reference: The 6 Key Aspects

Every prompt must include:

1. **Shot** - Cinematography terms (close-up, wide, medium, over-the-shoulder)
2. **Scene** - Lighting, color palette, textures, atmosphere
3. **Action** - Natural sequence flowing start to finish
4. **Character** - Age, clothing, physical cues (NOT abstract emotions)
5. **Camera** - Movement relative to subject, what appears after motion
6. **Audio** - Ambient sounds, dialogue in "quotation marks"

---

## Format Rules

| Rule | Correct | Wrong |
|------|---------|-------|
| Structure | Single flowing paragraph, 4-8 sentences | Bullet points, fragments |
| Tense | Present ("The car speeds") | Past ("The car sped") |
| Emotions | Physical cues ("slumped shoulders") | Abstract ("sad", "confused") |
| Dialogue | "In quotation marks" | Without quotes |
| Detail | Match to shot scale | Same detail for all shots |

---

## What Works vs What Breaks

| Works Well | Avoid (Causes Artifacts) |
|------------|--------------------------|
| Cinematic compositions with lighting | Internal states ("sad", "angry") |
| Single-subject emotional expressions | Text and logos (unreadable) |
| Atmosphere (fog, rain, golden hour) | Complex physics (jumping, juggling) |
| Clear camera language | Scene overload (too many characters) |
| Stylized aesthetics (noir, pixel art) | Inconsistent lighting logic |
| Backlighting, rim light, color palettes | Overcomplicated multi-action prompts |

---

## Experiment Prompts

### For Layer Analysis Experiments

Use these prompts that cover different visual categories:

**Simple Object (Static)**
> "A medium shot of a bright red rubber ball resting on a pristine white surface. The lighting is soft and even, creating gentle shadows beneath the ball. The camera holds steady as dust particles drift through a warm beam of afternoon sunlight. The ball's glossy surface catches subtle reflections from the ambient light."

**Animal (Motion)**
> "A golden retriever runs joyfully through a sun-dappled park, its fur gleaming in the warm afternoon light. The camera tracks alongside as the dog bounds across lush green grass, tongue out and tail wagging energetically. Birds chirp softly in the background as leaves rustle in a gentle breeze. The scene captures natural motion with shallow depth of field."

**Person (Emotion via Physical Cues)**
> "A lone figure walks slowly through heavy rain on a city sidewalk, holding a bright red umbrella. Their shoulders are hunched against the cold, coat collar pulled up. The camera follows from behind as streetlights create golden halos in the downpour. Raindrops splash against the pavement creating small ripples in growing puddles."

**Complex Scene (Multiple Elements)**
> "A bustling city street at night comes alive with neon signs reflecting off rain-slicked pavement. Crowds of people in dark coats hurry past storefronts while taxis honk in the distance. The camera slowly pans across the scene, capturing the vibrant energy of urban nightlife. Steam rises from a nearby food cart as colored lights dance across wet surfaces."

**Abstract/Stylized**
> "A dreamlike surreal landscape unfolds with floating islands suspended in a pink and purple sky. Ethereal mist swirls around ancient stone structures as bioluminescent plants pulse with soft rhythmic light. The camera drifts slowly through this otherworldly realm with smooth, floating movement. Crystalline formations catch and refract light in rainbow patterns."

**Dramatic Lighting**
> "A person's face emerges from complete darkness, dramatically illuminated by a single flickering candle held at chest level. The warm orange glow dances across their features, creating deep shadows in the eye sockets and under the chin. The camera holds a tight close-up as the flame gently sways. Wisps of smoke curl upward, catching the light before dissolving into shadow."

---

## Vocabulary Reference

### Camera Movement
follows, tracks, pans across, circles around, tilts upward, pushes in, pulls back, overhead view, handheld movement, over-the-shoulder, wide establishing shot, static frame, dolly, crane

### Lighting
flickering candles, neon glow, natural sunlight, dramatic shadows, rim light, backlighting, golden hour, soft diffused, harsh contrast

### Atmosphere
fog, mist, rain, dust, smoke, particles, steam, haze

### Pacing
slow motion, time-lapse, lingering shot, continuous shot, freeze-frame, fade-in, seamless transition

### Style Keywords
cinematic, film noir, documentary, painterly, pixel art, surreal, minimalist, handheld, shallow depth of field

---

## Common Mistakes in Experiments

**BAD**: "A cat sitting on a box"
- Too short, no cinematography, no lighting, no atmosphere

**BAD**: "A sad person walking in rain feeling depressed"
- Uses internal states instead of physical cues

**BAD**: "Show the word HELLO on screen"
- Text generation doesn't work reliably

**GOOD**: Full paragraph with all 6 aspects covered, 4-8 sentences, present tense, physical descriptions only.

---

## Source

Based on the official LTX-2 prompting guide from [ltx.io](https://ltx.io/model/model-blog/prompting-guide-for-ltx-2).

See `ltx2_prompting_guide_written_by_gemini.md` for the complete reference with extensive examples.
