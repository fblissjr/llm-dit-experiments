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

## Official Example Prompts (VERBATIM)

These are the official LTX-2 example prompts from the prompting guide. **Use these verbatim** in experiments - do not summarize or simplify. They match LTX-2's training data distribution.

### Action / Cinematic

> "An action packed, cinematic shot of a monster truck driving fast towards the camera, the truck passes the cameras it pans left to follow the trucks reckless drive. dust and motion blur is around the truck, hand held feel to the camera as it tries to track its ride into the distance. the truck then drifts and turns around, then drives back towards the camera until seen in extreme close up."

### News Broadcast (Live Event)

> "EXT. SMALL TOWN STREET – MORNING – LIVE NEWS BROADCAST. The shot opens on a news reporter standing in front of a row of cordoned-off cars, yellow caution tape fluttering behind him. The light is warm, early sun reflecting off the camera lens. The faint hum of chatter and distant drilling fills the air. The reporter, composed but visibly excited, looks directly into the camera, microphone in hand.
> Reporter (live): "Thank you, Sylvia. And yes — this is a sentence I never thought I'd say on live television — but this morning, here in the quiet town of New Castle, Vermont… black gold has been found!"
> He gestures slightly toward the field behind him. Reporter (grinning): "If my cameraman can pan over, you'll see what all the excitement's about."
> The camera pans right, slowly revealing a construction site surrounded by workers in hard hats. A beat of silence — then, with a sudden roar, a geyser of oil erupts from the ground, blasting upward in a violent plume. Workers cheer and scramble, the black stream glistening in the morning light. The camera shakes slightly, trying to stay focused through the chaos.
> Reporter (off-screen, shouting over the noise): "There it is, folks — the moment New Castle will never forget!"
> The camera catches the sunlight gleaming off the oil mist before pulling back, revealing the entire scene — the small-town skyline silhouetted against the wild fountain of oil."

### Comedy / Dialogue (Script Style)

> "A warm sunny backyard. The camera starts in a tight cinematic close-up of a woman and a man in their 30s, facing each other with serious expressions. The woman, emotional and dramatic, says softly, "That's it... Dad's lost it. And we've lost Dad."
> The man exhales, slightly annoyed: "Stop being so dramatic, Jess." A beat. He glances aside, then mutters defensively, "He's just having fun."
> The camera slowly pans right, revealing the grandfather in the garden wearing enormous butterfly wings, waving his arms in the air like he's trying to take off. He shouts, "Wheeeew!" as he flaps his wings with full commitment.
> The woman covers her face, on the verge of tears. The tone is deadpan, absurd, and quietly tragic."

### Animation (Pixar Style - Baker)

> "INT. OVEN – DAY. Static camera from inside the oven, looking outward through the slightly fogged glass door. Warm golden light glows around freshly baked cookies. The baker's face fills the frame, eyes wide with focus, his breath fogging the glass as he leans in. Subtle reflections move across the glass as steam rises.
> Baker (whispering dramatically): "Today… I achieve perfection." He leans even closer, nose nearly touching the glass. "Golden edges. Soft center. The gods themselves will smell these cookies and weep."
> Baker: "Wait—" (beat) "Did I… forget the chocolate chips?"
> Cut to side view — coworker pops into frame, chewing casually. Coworker (mouth full): "Nope. You forgot the sugar."
> Quick zoom back to the baker's horrified face, pressed against the oven door, as cookies deflate behind the glass. Steam drifts upward in slow motion. pixar style acting and timing"

### Animation (Frog Yoga)

> "The camera opens in a calm, sunlit frog yoga studio. Warm morning light washes over the wooden floor as incense smoke drifts lazily in the air. The senior frog instructor sits cross-legged at the center, eyes closed, voice deep and calm. "We are one with the pond." All the frogs answer softly: "Ommm…" "We are one with the mud." "Ommm…" He smiles faintly. "We are one with the flies." A pause.
> The camera pans to the side towards one frog who twitches, eyes darting. Suddenly its tongue snaps out, catching a fly mid-air and pulling it into its mouth.
> The master exhales slowly, still serene. "But we do not chase the flies…" Beat. "not during class."
> The guilty frog lowers its head in shame, folding its hands back into a meditative pose. The other frogs resume their chant: "Ommm…" Camera holds for a moment on the embarrassed frog, eyes closed too tightly, pretending nothing happened."

### Documentary / Drama

> "INT. DAYTIME TALK SHOW SET – AFTERNOON Soft studio lighting glows across a warm-toned set. The audience murmurs faintly as the camera pans to reveal three guests seated on a couch — a middle-aged couple and the show's host sitting across from them.
> The host leans forward, voice steady but probing: Host: "When did you first notice that your daughter, Missy, started to spiral?"
> The woman's face crumples; she takes a shaky breath and begins to cry. Her husband places a comforting hand on her shoulder, looking down before turning back toward the host. Father (quietly, with guilt): "We… we don't know what we did wrong."
> The studio falls silent for a moment. The camera cuts to the host, who looks gravely into the lens. Host (to camera): "Let's take a look at a short piece our team prepared — chronicling Missy's downward path."
> The lights dim slightly as the camera pushes in on the mother's tear-streaked face. The studio monitors flicker to life, beginning to play the segment as the audience holds its breath."

### Sci-Fi / Stylized

> "The young african american woman wearing a futuristic transparent visor and a bodysuit with a tube attached to her neck. she is soldering a robotic arm. she stops and looks to her right as she hears a suspicious strong hit sound from a distance. she gets up slowly from her chair and says with an angry african american accent: "Rick I told you to close that goddamn door after you!".
> then, a futuristic blue alien explorer with dreadlocks wearing a rugged outfit walks into the scene excitedly holding a futuristic device and says with a low robotic voice: "Fuck the door look what I found!".
> the alien hands the woman the device, she looks down at it excitedly as the camera zooms in on her intrigued illuminated face. she then says: "is this what I think it is?" she smiles excitedly. sci-fi style cinematic scene"

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

## Related Documentation

- **Complete Reference**: [ltx2_official_prompting_guide.md](./ltx2_official_prompting_guide.md) - Full official guide with vocabulary lists and expanded examples
- **Code Implementation**: [prompts.py](./prompts.py) - Centralized prompt module (import from here in experiments)
- **Standardization Summary**: [prompting_fix_summary.md](./prompting_fix_summary.md) - Why prompts were standardized
- **Research Context**: [internal/research/ltx2/ltx2_captioning_research.md](../../internal/research/ltx2/ltx2_captioning_research.md) - Training data caption format analysis

## Source

Based on the official LTX-2 prompting guide from [ltx.io](https://ltx.io/model/model-blog/prompting-guide-for-ltx-2).
