# LTX-2 Prompting Guide

Last updated: 2026-01-16

This is the complete official LTX-2 prompting guide with extensive vocabulary and examples.

**For experiments**: Import prompts from [prompts.py](./prompts.py) instead of defining inline.
**Quick reference**: See [prompting_guide.md](./prompting_guide.md) for condensed format rules.

---

To get the most out of LTX-2, the goal is to **paint a complete picture**. A successful prompt flows naturally from beginning to end, covering all the elements the model needs to bring your vision to life.

## 1. The 6 Key Aspects of a Prompt

When constructing your prompt, ensure you cover these core elements:

* **Establish the Shot:** Use cinematography terms (e.g., *wide establishing shot*, *close-up*) and genre characteristics to set the scale and style.
* **Set the Scene:** Describe lighting, color palette, surface textures, and atmosphere to shape the mood.
* **Describe the Action:** Write the core action as a natural sequence, flowing logically from start to finish.
* **Define Character(s):** Include age, hairstyle, clothing, and distinguishing details. **Crucially, express emotions through physical cues** rather than abstract feelings (e.g., *"slumped shoulders"* instead of *"sad"*).
* **Identify Camera Movement:** Specify how and when the camera moves. Describing how subjects appear *after* the movement helps the model complete the motion accurately.
* **Describe the Audio:** Use clear descriptions for ambient sounds, music, and speech.
* Place spoken dialogue in **"quotation marks"**.
* Specify language and accent if needed.



## 2. Best Practices for Results

* **Structure:** Keep your prompt in a **single flowing paragraph** (approx. 4–8 sentences).
* **Tense:** Use **present tense verbs** for movement and action (e.g., *"The car speeds"* not *"sped"*).
* **Detail vs. Scale:** Match your detail to the shot scale (e.g., close-ups need precise detail regarding texture/eyes; wide shots focus on composition).
* **Camera Logic:** Describe movement relative to the subject (e.g., *"Camera tracks the runner from the side"*).
* **Iteration:** LTX-2 is designed for fast experimentation—refine your prompt freely.

---

## 3. What Works & What to Avoid

| **What Works Well** | **What to Avoid** |
| --- | --- |
| **Cinematic Compositions:** Wide/medium/close-ups with thoughtful lighting, shallow depth of field, and natural motion. | **Internal States:** Avoid "sad" or "confused." Use visual cues (posture, expressions) instead. |
| **Emotive Moments:** Strong single-subject emotional expressions, subtle gestures, and facial nuance. | **Text & Logos:** The model does not currently generate readable text or consistent branding. |
| **Atmosphere:** Fog, mist, golden hour, rain, reflections, and ambient textures ground the scene. | **Complex Physics:** Non-linear or fast-twisting motion (juggling, jumping) causes artifacts. Dancing is OK. |
| **Clear Camera Language:** Explicit instructions like "slow dolly in," "handheld tracking," or "over-the-shoulder." | **Scene Overload:** Too many characters or layered actions reduce clarity. |
| **Stylized Aesthetics:** Noir, pixel art, surrealism, painterly styles, or analog film (name them early). | **Inconsistent Lighting:** Do not mix conflicting sources (e.g., warm sunset + cold neon) unless motivated. |
| **Lighting & Mood:** Backlighting, color palettes, rim light, flickering lamps. | **Overcomplicated Prompts:** Start simple and layer complexity gradually. |

### **Voice Capabilities**

* **Performance:** Characters can both **talk and sing**.
* **Versatility:** The model supports **multiple languages**.

---

## 4. Expanded Vocabulary Reference

*Use these lists to refine the style and specificity of your prompts.*

### **Categories & Genre**

* **Animation:** Stop-motion, 2D / 3D animation, Claymation, Hand-drawn.
* **Stylized:** Comic book, Cyberpunk, 8-bit pixel, Surreal, Minimalist, Painterly, Illustrated.
* **Cinematic:** Period drama, Film noir, Fantasy, Epic space opera, Thriller, Modern romance, Experimental film, Arthouse, Documentary.

### **Visual Details**

* **Lighting:** Flickering candles, Neon glow, Natural sunlight, Dramatic shadows, Rim light.
* **Textures:** Rough stone, Smooth metal, Worn fabric, Glossy surfaces.
* **Color Palette:** Vibrant, Muted, Monochromatic, High contrast.
* **Atmosphere:** Fog, Rain, Dust, Smoke, Particles.

### **Camera & Technical Style**

* **Movement:** Follows, Tracks, Pans across, Circles around, Tilts upward, Pushes in / pulls back, Overhead view, Handheld movement, Over-the-shoulder, Wide establishing shot, Static frame.
* **Pacing & Time:** Slow motion, Time-lapse, Rapid cuts, Lingering shot, Continuous shot, Freeze-frame, Fade-in / fade-out, Seamless transition, Sudden stop.
* **Film Characteristics:** Film grain, Lens flares, Pixelated edges, Jittery stop-motion.
* **Scale Indicators:** Expansive, Epic, Intimate, Claustrophobic.
* **Visual Effects:** Particle systems, Motion blur, Depth of field.

### **Sound and Voice**

* **Dialogue Style:** Energetic announcer, Resonant voice with gravitas, Distorted radio-style, Robotic monotone, Childlike curiosity.
* **Volume:** Whisper, Mutter, Shout, Scream.
* **Ambient Settings:** Coffeeshop noise, Wind and rain, Forest ambience with birds.

---

## 5. Comprehensive Example Library

### **Action / Cinematic**

> "An action packed, cinematic shot of a monster truck driving fast towards the camera, the truck passes the cameras it pans left to follow the trucks reckless drive. dust and motion blur is around the truck, hand held feel to the camera as it tries to track its ride into the distance. the truck then drifts and turns around, then drives back towards the camera until seen in extreme close up."

### **News Broadcast (Live Event)**

> "EXT. SMALL TOWN STREET – MORNING – LIVE NEWS BROADCAST. The shot opens on a news reporter standing in front of a row of cordoned-off cars, yellow caution tape fluttering behind him. The light is warm, early sun reflecting off the camera lens. The faint hum of chatter and distant drilling fills the air. The reporter, composed but visibly excited, looks directly into the camera, microphone in hand.
> Reporter (live): “Thank you, Sylvia. And yes — this is a sentence I never thought I’d say on live television — but this morning, here in the quiet town of New Castle, Vermont… black gold has been found!”
> He gestures slightly toward the field behind him. Reporter (grinning): “If my cameraman can pan over, you’ll see what all the excitement’s about.”
> The camera pans right, slowly revealing a construction site surrounded by workers in hard hats. A beat of silence — then, with a sudden roar, a geyser of oil erupts from the ground, blasting upward in a violent plume. Workers cheer and scramble, the black stream glistening in the morning light. The camera shakes slightly, trying to stay focused through the chaos.
> Reporter (off-screen, shouting over the noise): “There it is, folks — the moment New Castle will never forget!”
> The camera catches the sunlight gleaming off the oil mist before pulling back, revealing the entire scene — the small-town skyline silhouetted against the wild fountain of oil."

### **Comedy / Dialogue (Script Style)**

> "A warm sunny backyard. The camera starts in a tight cinematic close-up of a woman and a man in their 30s, facing each other with serious expressions. The woman, emotional and dramatic, says softly, “That’s it... Dad’s lost it. And we’ve lost Dad.”
> The man exhales, slightly annoyed: “Stop being so dramatic, Jess.” A beat. He glances aside, then mutters defensively, “He’s just having fun.”
> The camera slowly pans right, revealing the grandfather in the garden wearing enormous butterfly wings, waving his arms in the air like he’s trying to take off. He shouts, “Wheeeew!” as he flaps his wings with full commitment.
> The woman covers her face, on the verge of tears. The tone is deadpan, absurd, and quietly tragic."

### **Animation (Pixar Style - Baker)**

> "INT. OVEN – DAY. Static camera from inside the oven, looking outward through the slightly fogged glass door. Warm golden light glows around freshly baked cookies. The baker’s face fills the frame, eyes wide with focus, his breath fogging the glass as he leans in. Subtle reflections move across the glass as steam rises.
> Baker (whispering dramatically): “Today… I achieve perfection.” He leans even closer, nose nearly touching the glass. “Golden edges. Soft center. The gods themselves will smell these cookies and weep.”
> Baker: “Wait—” (beat) “Did I… forget the chocolate chips?”
> Cut to side view — coworker pops into frame, chewing casually. Coworker (mouth full): “Nope. You forgot the sugar.”
> Quick zoom back to the baker’s horrified face, pressed against the oven door, as cookies deflate behind the glass. Steam drifts upward in slow motion. pixar style acting and timing"

### **Animation (Frog Yoga)**

> "The camera opens in a calm, sunlit frog yoga studio. Warm morning light washes over the wooden floor as incense smoke drifts lazily in the air. The senior frog instructor sits cross-legged at the center, eyes closed, voice deep and calm. “We are one with the pond.” All the frogs answer softly: “Ommm…” “We are one with the mud.” “Ommm…” He smiles faintly. “We are one with the flies.” A pause.
> The camera pans to the side towards one frog who twitches, eyes darting. Suddenly its tongue snaps out, catching a fly mid-air and pulling it into its mouth.
> The master exhales slowly, still serene. “But we do not chase the flies…” Beat. “not during class.”
> The guilty frog lowers its head in shame, folding its hands back into a meditative pose. The other frogs resume their chant: “Ommm…” Camera holds for a moment on the embarrassed frog, eyes closed too tightly, pretending nothing happened."

### **Documentary / Drama**

> "INT. DAYTIME TALK SHOW SET – AFTERNOON Soft studio lighting glows across a warm-toned set. The audience murmurs faintly as the camera pans to reveal three guests seated on a couch — a middle-aged couple and the show’s host sitting across from them.
> The host leans forward, voice steady but probing: Host: “When did you first notice that your daughter, Missy, started to spiral?”
> The woman’s face crumples; she takes a shaky breath and begins to cry. Her husband places a comforting hand on her shoulder, looking down before turning back toward the host. Father (quietly, with guilt): “We… we don’t know what we did wrong.”
> The studio falls silent for a moment. The camera cuts to the host, who looks gravely into the lens. Host (to camera): “Let’s take a look at a short piece our team prepared — chronicling Missy’s downward path.”
> The lights dim slightly as the camera pushes in on the mother’s tear-streaked face. The studio monitors flicker to life, beginning to play the segment as the audience holds its breath."

### **Sci-Fi / Stylized**

> "The young african american woman wearing a futuristic transparent visor and a bodysuit with a tube attached to her neck. she is soldering a robotic arm. she stops and looks to her right as she hears a suspicious strong hit sound from a distance. she gets up slowly from her chair and says with an angry african american accent: "Rick I told you to close that goddamn door after you!".
> then, a futuristic blue alien explorer with dreadlocks wearing a rugged outfit walks into the scene excitedly holding a futuristic device and says with a low robotic voice: "Fuck the door look what I found!".
> the alien hands the woman the device, she looks down at it excitedly as the camera zooms in on her intrigued illuminated face. she then says: "is this what I think it is?" she smiles excitedly. sci-fi style cinematic scene"
