"""
LTX-2 Official Prompts

Last updated: 2026-01-16

Verbatim examples from official LTX-2 prompting guide and properly formatted
category prompts for experiments. These match the training data distribution
and should be used for all experiments unless testing specific format variations.

See: experiments/ltx2/prompting_guide.md for usage guidelines.

Key format requirements:
- 100-300+ words, multi-paragraph structure
- Present tense verbs
- Physical cues (not abstract emotions)
- Dialogue in "quotation marks" with character names
- Scene headings (INT./EXT.) where appropriate
- Camera movement described inline
- Style tags at end where appropriate
"""

# =============================================================================
# OFFICIAL PROMPTS - Verbatim from official LTX-2 prompting guide
# =============================================================================
# VERBATIM - do not modify these prompts

OFFICIAL_PROMPTS = {
    "action_cinematic": (
        "An action packed, cinematic shot of a monster truck driving fast towards the camera, "
        "the truck passes the cameras it pans left to follow the trucks reckless drive. "
        "dust and motion blur is around the truck, hand held feel to the camera as it tries "
        "to track its ride into the distance. the truck then drifts and turns around, then "
        "drives back towards the camera until seen in extreme close up."
    ),
    "news_broadcast": (
        "EXT. SMALL TOWN STREET – MORNING – LIVE NEWS BROADCAST. The shot opens on a news "
        "reporter standing in front of a row of cordoned-off cars, yellow caution tape fluttering "
        "behind him. The light is warm, early sun reflecting off the camera lens. The faint hum "
        "of chatter and distant drilling fills the air. The reporter, composed but visibly excited, "
        "looks directly into the camera, microphone in hand.\n"
        "Reporter (live): \"Thank you, Sylvia. And yes — this is a sentence I never thought I'd say "
        "on live television — but this morning, here in the quiet town of New Castle, Vermont… "
        "black gold has been found!\"\n"
        "He gestures slightly toward the field behind him. Reporter (grinning): \"If my cameraman "
        "can pan over, you'll see what all the excitement's about.\"\n"
        "The camera pans right, slowly revealing a construction site surrounded by workers in hard "
        "hats. A beat of silence — then, with a sudden roar, a geyser of oil erupts from the ground, "
        "blasting upward in a violent plume. Workers cheer and scramble, the black stream glistening "
        "in the morning light. The camera shakes slightly, trying to stay focused through the chaos.\n"
        "Reporter (off-screen, shouting over the noise): \"There it is, folks — the moment New Castle "
        "will never forget!\"\n"
        "The camera catches the sunlight gleaming off the oil mist before pulling back, revealing the "
        "entire scene — the small-town skyline silhouetted against the wild fountain of oil."
    ),
    "comedy_dialogue": (
        "A warm sunny backyard. The camera starts in a tight cinematic close-up of a woman and a "
        "man in their 30s, facing each other with serious expressions. The woman, emotional and "
        "dramatic, says softly, \"That's it... Dad's lost it. And we've lost Dad.\"\n"
        "The man exhales, slightly annoyed: \"Stop being so dramatic, Jess.\" A beat. He glances "
        "aside, then mutters defensively, \"He's just having fun.\"\n"
        "The camera slowly pans right, revealing the grandfather in the garden wearing enormous "
        "butterfly wings, waving his arms in the air like he's trying to take off. He shouts, "
        "\"Wheeeew!\" as he flaps his wings with full commitment.\n"
        "The woman covers her face, on the verge of tears. The tone is deadpan, absurd, and "
        "quietly tragic."
    ),
    "animation_pixar": (
        "INT. OVEN – DAY. Static camera from inside the oven, looking outward through the slightly "
        "fogged glass door. Warm golden light glows around freshly baked cookies. The baker's face "
        "fills the frame, eyes wide with focus, his breath fogging the glass as he leans in. Subtle "
        "reflections move across the glass as steam rises.\n"
        "Baker (whispering dramatically): \"Today… I achieve perfection.\" He leans even closer, nose "
        "nearly touching the glass. \"Golden edges. Soft center. The gods themselves will smell "
        "these cookies and weep.\"\n"
        "Baker: \"Wait—\" (beat) \"Did I… forget the chocolate chips?\"\n"
        "Cut to side view — coworker pops into frame, chewing casually. Coworker (mouth full): "
        "\"Nope. You forgot the sugar.\"\n"
        "Quick zoom back to the baker's horrified face, pressed against the oven door, as cookies "
        "deflate behind the glass. Steam drifts upward in slow motion. pixar style acting and timing"
    ),
    "animation_frog": (
        "The camera opens in a calm, sunlit frog yoga studio. Warm morning light washes over the "
        "wooden floor as incense smoke drifts lazily in the air. The senior frog instructor sits "
        "cross-legged at the center, eyes closed, voice deep and calm. \"We are one with the pond.\" "
        "All the frogs answer softly: \"Ommm…\" \"We are one with the mud.\" \"Ommm…\" He smiles "
        "faintly. \"We are one with the flies.\" A pause.\n"
        "The camera pans to the side towards one frog who twitches, eyes darting. Suddenly its "
        "tongue snaps out, catching a fly mid-air and pulling it into its mouth.\n"
        "The master exhales slowly, still serene. \"But we do not chase the flies…\" Beat. "
        "\"not during class.\"\n"
        "The guilty frog lowers its head in shame, folding its hands back into a meditative pose. "
        "The other frogs resume their chant: \"Ommm…\" Camera holds for a moment on the embarrassed "
        "frog, eyes closed too tightly, pretending nothing happened."
    ),
    "documentary": (
        "INT. DAYTIME TALK SHOW SET – AFTERNOON Soft studio lighting glows across a warm-toned set. "
        "The audience murmurs faintly as the camera pans to reveal three guests seated on a couch — "
        "a middle-aged couple and the show's host sitting across from them.\n"
        "The host leans forward, voice steady but probing: Host: \"When did you first notice that "
        "your daughter, Missy, started to spiral?\"\n"
        "The woman's face crumples; she takes a shaky breath and begins to cry. Her husband places "
        "a comforting hand on her shoulder, looking down before turning back toward the host. "
        "Father (quietly, with guilt): \"We… we don't know what we did wrong.\"\n"
        "The studio falls silent for a moment. The camera cuts to the host, who looks gravely into "
        "the lens. Host (to camera): \"Let's take a look at a short piece our team prepared — "
        "chronicling Missy's downward path.\"\n"
        "The lights dim slightly as the camera pushes in on the mother's tear-streaked face. The "
        "studio monitors flicker to life, beginning to play the segment as the audience holds its breath."
    ),
    "scifi_stylized": (
        "The young african american woman wearing a futuristic transparent visor and a bodysuit "
        "with a tube attached to her neck. she is soldering a robotic arm. she stops and looks to "
        "her right as she hears a suspicious strong hit sound from a distance. she gets up slowly "
        "from her chair and says with an angry african american accent: \"Rick I told you to close "
        "that goddamn door after you!\".\n"
        "then, a futuristic blue alien explorer with dreadlocks wearing a rugged outfit walks into "
        "the scene excitedly holding a futuristic device and says with a low robotic voice: "
        "\"Fuck the door look what I found!\".\n"
        "the alien hands the woman the device, she looks down at it excitedly as the camera zooms "
        "in on her intrigued illuminated face. she then says: \"is this what I think it is?\" she "
        "smiles excitedly. sci-fi style cinematic scene"
    ),
}


# =============================================================================
# CATEGORY PROMPTS - Rewritten experiment prompts in proper LTX-2 format
# =============================================================================
# These are the original experiment category prompts rewritten to match
# the official format: 100+ words, dialogue, scene headings, all 6 aspects

CATEGORY_PROMPTS = {
    "animal": (
        "EXT. SUBURBAN PARK – GOLDEN HOUR. A golden retriever bounds across sun-dappled grass, "
        "its honey-colored fur catching the warm afternoon light as it runs toward the camera. "
        "The dog's tongue lolls happily, ears flopping with each energetic stride. The camera "
        "tracks alongside at a low angle, capturing the athletic grace of the dog's movement.\n"
        "A young woman's voice calls out from off-screen: \"Max! Come here, boy!\" The retriever's "
        "ears perk up and it changes direction, kicking up small tufts of grass as it pivots.\n"
        "The camera follows as Max races toward his owner, tail wagging furiously. Birds chirp "
        "softly in the background, leaves rustling in a gentle breeze. The woman kneels down, "
        "arms open, as Max barrels toward her. She laughs: \"Good boy! Such a good boy!\"\n"
        "The dog leaps into her embrace, nearly knocking her over. Shallow depth of field keeps "
        "the reunion in sharp focus while the park blurs into warm bokeh behind them."
    ),
    "urban": (
        "EXT. DOWNTOWN TOKYO – NIGHT – RAIN. The camera opens on a wide shot of a rain-slicked "
        "city street, neon signs reflecting in shimmering pools on the asphalt. Crowds of people "
        "in dark coats hurry past illuminated storefronts, umbrellas bobbing like a sea of "
        "multicolored jellyfish.\n"
        "The camera slowly dollies forward through the crowd, capturing the vibrant chaos of "
        "urban nightlife. Steam rises from a nearby ramen cart where the vendor calls out: "
        "\"Irasshaimase! Fresh noodles!\"\n"
        "A businessman pauses under an awning, checking his phone. His face illuminates blue "
        "in the screen's glow. Behind him, a massive LED billboard cycles through advertisements "
        "– the light washing over passing pedestrians in waves of color.\n"
        "The camera tilts upward to reveal the towering skyline, rain visible as silver streaks "
        "against the bright signage. A taxi honks in the distance. The wet pavement creates "
        "perfect mirror reflections of the entire scene above."
    ),
    "nature": (
        "EXT. MOUNTAIN LAKE – DAWN. The camera holds steady on a serene alpine lake, its surface "
        "perfectly still like polished glass. Snow-capped peaks reflect in the mirror-like water, "
        "creating a symmetrical composition of breathtaking beauty. Wisps of morning mist drift "
        "lazily across the scene.\n"
        "A lone kayaker paddles silently into frame from the left, the soft splash of her oar "
        "breaking the silence. She pauses mid-stroke, taking in the view. Her voice is barely "
        "above a whisper: \"I've never seen anything like this.\"\n"
        "The camera slowly pans right, revealing wildflowers blooming along the shoreline – "
        "purple lupines and yellow buttercups nodding in a gentle breeze. A fish breaks the "
        "surface with a soft plop, sending concentric ripples across the reflection.\n"
        "An eagle's cry echoes off the distant cliffs. The camera tilts up to catch the bird "
        "soaring overhead, dark silhouette against the pink and gold sunrise. documentary style"
    ),
    "abstract": (
        "The camera drifts through a dreamlike surreal landscape where physics bends to imagination. "
        "Floating islands suspended in a pink and purple twilight sky rotate slowly, ancient stone "
        "temples perched impossibly on their edges. Bioluminescent plants pulse with soft, rhythmic "
        "light – azure and magenta, like breathing neon.\n"
        "A narrator speaks in a calm, ethereal voice: \"In this place, gravity is merely a "
        "suggestion.\" Crystal formations grow from nothing, catching light and refracting it "
        "into rainbow prisms that dance across the clouds below.\n"
        "A waterfall cascades upward from one floating island to another, its water defying "
        "expectation. The camera follows a single droplet as it rises, catching the light. "
        "Tiny glowing creatures – somewhere between fireflies and stars – swirl around the "
        "ascending stream.\n"
        "The world pulses gently, as if breathing. Each exhale brings new impossible geometries "
        "into view. surreal fantasy style, ethereal atmosphere"
    ),
    "human": (
        "INT. POTTERY STUDIO – AFTERNOON. Warm golden sunlight streams through tall warehouse "
        "windows, dust motes floating in the shafts of light. An artisan in a clay-stained apron "
        "hunches over a pottery wheel, hands wet with slip as she shapes a vessel emerging from "
        "spinning clay.\n"
        "The camera holds a medium shot, capturing her focused expression – brow slightly "
        "furrowed, lips pressed together in concentration. Her hands move with practiced precision, "
        "thumbs pressing into the clay to form the bowl's interior.\n"
        "She speaks softly, almost to herself: \"A little more here... yes, that's it.\" The wheel "
        "hums quietly as she works. Behind her, shelves lined with finished pieces catch the light – "
        "some glazed in deep ocean blues, others in earthy terracotta.\n"
        "A student approaches hesitantly. Student: \"Sensei, can you show me that technique again?\" "
        "The artisan smiles without looking up: \"Watch my thumbs. Feel the clay. Let it tell you "
        "what it wants to become.\" documentary style cinematography"
    ),
    "interior": (
        "INT. ANTIQUE STUDY – LATE AFTERNOON. The camera opens on a close-up of an antique wooden "
        "desk bathed in warm amber light. Dust motes drift lazily through sunbeams streaming from "
        "tall windows hung with heavy velvet curtains. An old leather-bound book lies open next "
        "to a brass inkwell and feather quill.\n"
        "The camera slowly pushes in as golden rays cast intricate shadow patterns from lace "
        "curtains across the worn wood grain. A cup of tea steams gently, wisps curling upward "
        "into the light.\n"
        "An elderly scholar enters frame, settling into a worn leather chair that creaks with "
        "familiarity. He adjusts his reading glasses and peers at the yellowed pages. Scholar "
        "(muttering to himself): \"After all these years... finally, the missing passage.\"\n"
        "His weathered finger traces a line of faded text. Outside, a grandfather clock chimes "
        "the hour. The scholar looks up with wonder in his eyes, the discovery reflected in his "
        "expression. period drama style, rich warm tones"
    ),
    "lighting": (
        "INT. DARK CHAMBER – NIGHT. A person's face emerges from complete darkness, dramatically "
        "illuminated by a single flickering candle held at chest level. The warm orange glow "
        "dances across weathered features, creating deep shadows in the eye sockets and beneath "
        "the chin.\n"
        "The camera holds a tight close-up as the flame gently sways, casting shifting patterns "
        "of light and shadow. The subject's eyes catch the reflection of the fire, glistening "
        "with an emotion that remains unspoken.\n"
        "A breath. Then, barely above a whisper: \"It's been seven years since I've seen this "
        "place.\" The hand holding the candle trembles slightly, causing the shadows to shiver.\n"
        "The camera pulls back slowly, revealing more of the dark space – shapes of furniture "
        "covered in white sheets, emerging ghostlike from the blackness. Wisps of smoke curl "
        "upward from the candle, catching the light before dissolving into shadow above. "
        "film noir style, chiaroscuro lighting"
    ),
    "motion": (
        "EXT. DESERT HIGHWAY – HIGH NOON. A sleek sports car races down an empty stretch of "
        "asphalt, its red paint gleaming under the harsh midday sun. The camera tracks alongside "
        "in a dynamic side shot, capturing the blur of motion as the landscape streaks past.\n"
        "Heat waves shimmer off the road surface. The engine's roar fills the soundtrack as "
        "the driver downshifts, the car's rear end kicking out slightly as it takes a curve. "
        "Driver (to passenger): \"Hold on – I'm gonna push it.\"\n"
        "The speedometer climbs. Motion blur streaks the background into abstract lines of "
        "tan and blue. Dust kicks up behind the wheels, creating a rooster tail against the "
        "desert sky.\n"
        "The passenger grips the door handle, knuckles white. Passenger: \"This is insane!\" "
        "A grin spreads across the driver's face as the car accelerates further, chrome details "
        "catching the sunlight, tinted windows reflecting the endless road ahead. action cinema style"
    ),
}


# =============================================================================
# STRUCTURED FORMAT PROMPTS - For format ablation experiments
# =============================================================================
# Same semantic content in different structural formats
# Expected: Prose performs best (matches training distribution)

STRUCTURED_PROMPTS = {
    "prose_baseline": (
        "EXT. SUBURBAN PARK – GOLDEN HOUR. A golden retriever bounds across sun-dappled grass, "
        "its honey-colored fur catching the warm afternoon light as it runs toward the camera. "
        "The dog's tongue lolls happily, ears flopping with each energetic stride. The camera "
        "tracks alongside at a low angle, capturing the athletic grace of the dog's movement. "
        "A young woman's voice calls out from off-screen: \"Max! Come here, boy!\" The retriever's "
        "ears perk up and it changes direction, kicking up small tufts of grass as it pivots."
    ),
    "markdown": (
        "# Scene: Park at Golden Hour\n\n"
        "## Setting\n"
        "- Location: Suburban park\n"
        "- Time: Golden hour, late afternoon\n"
        "- Lighting: Warm, sun-dappled\n\n"
        "## Subject\n"
        "- A golden retriever with honey-colored fur\n"
        "- Running energetically toward camera\n"
        "- Tongue out, ears flopping, tail wagging\n\n"
        "## Camera\n"
        "- Low angle tracking shot\n"
        "- Following alongside the dog\n\n"
        "## Audio\n"
        "- Woman's voice: \"Max! Come here, boy!\"\n"
        "- Birds chirping, leaves rustling"
    ),
    "json": (
        '{"scene": {"location": "suburban park", "time": "golden hour"}, '
        '"subject": {"type": "golden retriever", "appearance": "honey-colored fur", '
        '"action": "running energetically toward camera", '
        '"details": "tongue out, ears flopping, tail wagging"}, '
        '"camera": {"angle": "low", "movement": "tracking alongside"}, '
        '"lighting": "warm sun-dappled afternoon light", '
        '"audio": {"dialogue": "Max! Come here, boy!", "ambient": "birds chirping, leaves rustling"}}'
    ),
    "xml": (
        "<scene>\n"
        "  <setting>\n"
        "    <location>suburban park</location>\n"
        "    <time>golden hour</time>\n"
        "    <lighting>warm sun-dappled afternoon light</lighting>\n"
        "  </setting>\n"
        "  <subject>\n"
        "    <type>golden retriever</type>\n"
        "    <appearance>honey-colored fur</appearance>\n"
        "    <action>running energetically toward camera</action>\n"
        "    <details>tongue out, ears flopping, tail wagging</details>\n"
        "  </subject>\n"
        "  <camera angle=\"low\" movement=\"tracking alongside\"/>\n"
        "  <audio>\n"
        "    <dialogue speaker=\"woman\">Max! Come here, boy!</dialogue>\n"
        "    <ambient>birds chirping, leaves rustling</ambient>\n"
        "  </audio>\n"
        "</scene>"
    ),
    "yaml": (
        "scene:\n"
        "  setting:\n"
        "    location: suburban park\n"
        "    time: golden hour\n"
        "    lighting: warm sun-dappled afternoon light\n"
        "  subject:\n"
        "    type: golden retriever\n"
        "    appearance: honey-colored fur\n"
        "    action: running energetically toward camera\n"
        "    details: tongue out, ears flopping, tail wagging\n"
        "  camera:\n"
        "    angle: low\n"
        "    movement: tracking alongside\n"
        "  audio:\n"
        "    dialogue: \"Max! Come here, boy!\"\n"
        "    ambient: birds chirping, leaves rustling"
    ),
}


# =============================================================================
# QUICK SUBSETS - For --quick mode in experiments
# =============================================================================

# Three diverse official examples for quick testing
QUICK_OFFICIAL = ["action_cinematic", "comedy_dialogue", "documentary"]

# Two category prompts for quick testing
QUICK_CATEGORY = ["animal", "urban"]

# Minimal format ablation set
QUICK_STRUCTURED = ["prose_baseline", "json"]


# =============================================================================
# LEGACY PROMPTS - Short prompts from old experiments (for reference only)
# =============================================================================
# DO NOT use these for new experiments - they are out-of-distribution
# Kept here for backward compatibility and comparison studies

LEGACY_SHORT_PROMPTS = {
    "animal_short": "A golden retriever runs through a sun-dappled park",
    "urban_short": "A bustling city street at night with neon signs",
    "nature_short": "A mountain lake reflects snow-capped peaks",
    "human_short": "An artisan shapes clay on a pottery wheel",
    "abstract_short": "Floating islands in a pink and purple sky",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_official_prompts(quick: bool = False) -> dict[str, str]:
    """Get official prompts, optionally filtered for quick mode."""
    if quick:
        return {k: OFFICIAL_PROMPTS[k] for k in QUICK_OFFICIAL}
    return OFFICIAL_PROMPTS.copy()


def get_category_prompts(quick: bool = False) -> dict[str, str]:
    """Get category prompts, optionally filtered for quick mode."""
    if quick:
        return {k: CATEGORY_PROMPTS[k] for k in QUICK_CATEGORY}
    return CATEGORY_PROMPTS.copy()


def get_all_prompts(quick: bool = False) -> dict[str, str]:
    """Get all prompts (official + category), optionally filtered for quick mode."""
    prompts = {}
    prompts.update(get_official_prompts(quick))
    prompts.update(get_category_prompts(quick))
    return prompts


def get_structured_prompts(quick: bool = False) -> dict[str, str]:
    """Get structured format prompts for ablation, optionally filtered."""
    if quick:
        return {k: STRUCTURED_PROMPTS[k] for k in QUICK_STRUCTURED}
    return STRUCTURED_PROMPTS.copy()


def word_count(text: str) -> int:
    """Count words in a prompt."""
    return len(text.split())


def validate_prompts():
    """Validate that all prompts meet format requirements."""
    issues = []

    # Check official prompts
    for name, prompt in OFFICIAL_PROMPTS.items():
        wc = word_count(prompt)
        if wc < 50:
            issues.append(f"OFFICIAL[{name}]: Only {wc} words (expected 50+)")

    # Check category prompts
    for name, prompt in CATEGORY_PROMPTS.items():
        wc = word_count(prompt)
        if wc < 100:
            issues.append(f"CATEGORY[{name}]: Only {wc} words (expected 100+)")
        if '"' not in prompt:
            issues.append(f"CATEGORY[{name}]: Missing dialogue (no quotation marks)")

    if issues:
        print("Validation issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print(f"All prompts validated successfully:")
    print(f"  - Official: {len(OFFICIAL_PROMPTS)} prompts")
    print(f"  - Category: {len(CATEGORY_PROMPTS)} prompts")
    print(f"  - Structured: {len(STRUCTURED_PROMPTS)} formats")
    return True


if __name__ == "__main__":
    validate_prompts()

    print("\n" + "=" * 60)
    print("OFFICIAL PROMPTS")
    print("=" * 60)
    for name, prompt in OFFICIAL_PROMPTS.items():
        print(f"\n{name} ({word_count(prompt)} words):")
        print(f"  {prompt[:100]}...")

    print("\n" + "=" * 60)
    print("CATEGORY PROMPTS")
    print("=" * 60)
    for name, prompt in CATEGORY_PROMPTS.items():
        print(f"\n{name} ({word_count(prompt)} words):")
        print(f"  {prompt[:100]}...")
