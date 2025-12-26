last updated: 2025-12-26

# qwen-image-edit-2511 prompting guide

Comprehensive guide for writing effective prompts with Qwen-Image-Edit-2511. Covers edit categories, prompt patterns, Chinese/English examples, and best practices.

## table of contents

- [overview](#overview)
- [edit categories](#edit-categories)
- [prompt examples](#prompt-examples)
- [prompt patterns and templates](#prompt-patterns-and-templates)
- [best practices](#best-practices)
- [special capabilities](#special-capabilities)
- [technical parameters](#technical-parameters)
- [common mistakes to avoid](#common-mistakes-to-avoid)

---

## overview

Qwen-Image-Edit-2511 is an instruction-based image editing model that accepts natural language prompts to modify images. The model supports:

- Single-image editing with text instructions
- Multi-image composition (2-4 images combined)
- Both Chinese and English prompts
- Complex text rendering in edited images
- Identity/character preservation across edits

**Key insight:** The model is trained to understand descriptive prompts about the desired output state, not step-by-step editing commands. Describe what you want the result to look like, not the editing process.

---

## edit categories

### 1. character and identity editing

Modify portraits while preserving the subject's identity and visual characteristics.

| Use Case | Description |
|----------|-------------|
| Style transfer | Convert photo to artistic style (anime, Ghibli, oil painting) |
| Pose changes | Modify body position via keypoint guidance |
| Expression changes | Alter facial expressions |
| Clothing changes | Modify outfit while keeping identity |
| Background replacement | New scene while preserving subject |

### 2. multi-image composition

Combine multiple source images into coherent outputs. Optimal with 2-3 images.

| Composition Type | Description |
|------------------|-------------|
| Person + Person | Group photos from individual portraits |
| Person + Scene | Place subject in new environment |
| Person + Product | Product photos with human elements |
| Subject + Style | Apply style from reference image |
| Multiple Objects | Arrange objects in new composition |

### 3. appearance editing

Modify visual attributes while preserving structure.

| Edit Type | Description |
|-----------|-------------|
| Color changes | Recolor objects, backgrounds, elements |
| Material replacement | Change textures and surfaces |
| Lighting adjustment | Modify light direction, intensity, color |
| Detail enhancement | Add or refine visual details |
| Element insertion | Add objects with proper reflections/shadows |
| Element removal | Remove unwanted objects cleanly |

### 4. text and typography editing

Modify text within images (exceptional Chinese support).

| Edit Type | Description |
|-----------|-------------|
| Text content | Change what text says |
| Font style | Modify typography style |
| Text color | Recolor text elements |
| Text size | Adjust text scale |
| Text addition | Add new text to images |
| Calligraphy correction | Fix handwritten characters |

### 5. geometric and structural editing

Changes involving spatial reasoning.

| Edit Type | Description |
|-----------|-------------|
| View synthesis | Generate new camera angles (90, 180 degree rotations) |
| Perspective changes | Modify viewpoint |
| Construction lines | Add auxiliary annotation lines |
| Layout restructuring | Reorganize spatial arrangement |

### 6. industrial and product design

Specialized edits for product and design applications.

| Edit Type | Description |
|-----------|-------------|
| Batch material replacement | Change surface materials across components |
| Product colorways | Generate color variants |
| Design iteration | Modify product designs systematically |
| Technical annotation | Add design specifications |

---

## prompt examples

### single-image editing examples

#### english prompts

| Prompt | Edit Type |
|--------|-----------|
| "Change the rabbit's color to purple, with a flash light background" | Color + lighting |
| "Make the house blue with a red roof" | Color change |
| "Convert to Studio Ghibli animation style" | Style transfer |
| "Add warm sunset lighting from the left" | Lighting |
| "Remove the person in the background" | Element removal |
| "Change the text to say 'Welcome Home'" | Text editing |
| "Make it look more photorealistic" | Style enhancement |
| "Add a vintage film grain effect" | Style effect |
| "Rotate the object 90 degrees to show the side view" | View synthesis |

#### chinese prompts with translations

| Chinese | English Translation | Edit Type |
|---------|---------------------|-----------|
| "把兔子的颜色改成紫色，加上闪光灯背景" | "Change the rabbit's color to purple, with flash light background" | Color + lighting |
| "这个女生看着面前的电视屏幕，屏幕上面写着'阿里巴巴'" | "This girl is watching a TV screen displaying 'Alibaba'" | Text rendering |
| "改成宫崎骏动漫风格" | "Convert to Miyazaki/Ghibli animation style" | Style transfer |
| "增加温暖的夕阳光线" | "Add warm sunset lighting" | Lighting |
| "删除背景中的人物" | "Remove the person in the background" | Element removal |
| "让画面更加真实" | "Make the image more realistic" | Enhancement |
| "添加复古胶片颗粒效果" | "Add vintage film grain effect" | Style effect |

### multi-image composition examples

#### english prompts

| Prompt | Images | Result |
|--------|--------|--------|
| "The magician bear is on the left, the alchemist bear is on the right, facing each other in the central park square" | 2 character images | Combined scene with positioned characters |
| "Both people standing together in a park, natural lighting" | 2 portrait images | Group photo |
| "Place the subject in the landscape with the artistic style" | Subject + landscape + style reference | Styled composition |
| "The person from the first image sitting in the room from the second" | Portrait + interior | Subject in new environment |
| "All items arranged on the table" | Multiple product images | Product composition |

#### chinese prompts with translations

| Chinese | English Translation | Use Case |
|---------|---------------------|----------|
| "魔法师熊在左边，炼金术师熊在右边，面向中央公园广场" | "Magician bear on left, alchemist bear on right, facing the central park square" | Character positioning |
| "两个人一起站在公园里，自然光线" | "Both people standing together in a park, natural lighting" | Group photo |
| "把第一张图的人物放在第二张图的场景中" | "Place the person from the first image in the scene from the second" | Subject + background |
| "所有物品摆放在桌子上" | "All items arranged on the table" | Product arrangement |

### text rendering examples

#### complex text prompts (english)

```
A coffee shop entrance features a chalkboard sign reading "Qwen Coffee $2 per cup,"
with a neon light beside it displaying "Open 24/7"
```

```
A movie poster with the title "Imagination Unleashed" at the top,
subtitle "Enter a world beyond your imagination" below,
director and cast credits at the bottom
```

```
A man in a suit holding a yellowed paper with handwritten words:
"In the garden of dreams, time stands still"
```

#### complex text prompts (chinese)

```
宫崎骏的动漫风格。平视角拍摄，阳光下的古街热闹非凡。
沿街有"云存储"、"云计算"、"云模型"的招牌店铺
```
Translation: "Miyazaki animation style. Eye-level shot, bustling ancient street in sunlight. Street shops with signs reading 'Cloud Storage', 'Cloud Computing', 'Cloud Model'"

```
一副典雅庄重的对联悬挂于厅堂之中，红底金字，
上联"先天下之忧而忧"，下联"后天下之乐而乐"
```
Translation: "An elegant couplet hanging in the hall, gold text on red, upper line 'Worry before the world worries', lower line 'Rejoice after the world rejoices'"

```
一张企业级高质量PPT页面图像，整体采用科技感十足的星空蓝配色
```
Translation: "A high-quality enterprise PPT slide image, using tech-inspired starry sky blue color scheme"

---

## prompt patterns and templates

### pattern 1: descriptive state

Describe the desired end state, not the process.

**Template:** `[Subject] [in state/condition] [with details] [style modifiers]`

**Good:** "The cat sitting on a blue velvet cushion, warm afternoon lighting"
**Bad:** "Take the cat and put it on a cushion and make it blue"

### pattern 2: spatial positioning (multi-image)

Specify relative positions of combined elements.

**Template:** `[Subject A] is [position], [Subject B] is [position], [context/setting]`

**Example:** "The magician bear is on the left, the alchemist bear is on the right, facing each other in the central park square"

### pattern 3: style transfer

Reference specific artistic styles by name.

**Template:** `[Subject/scene] in [style name] style`

**Examples:**
- "Portrait in Studio Ghibli animation style"
- "Landscape in impressionist oil painting style"
- "Product in minimalist design style"

### pattern 4: text specification

Describe text content, placement, and styling.

**Template:** `[Object with text] displaying/reading "[exact text]" [text styling] [placement]`

**Example:** "A neon sign displaying 'OPEN' in red glowing letters above the door"

### pattern 5: lighting modification

Specify light source, direction, and quality.

**Template:** `[Subject] with [lighting type] from [direction] [time/mood]`

**Examples:**
- "Portrait with soft golden light from the left, sunset mood"
- "Product with dramatic rim lighting from behind, dark background"

### pattern 6: material and color changes

Specify the target material or color precisely.

**Template:** `Change [element] to [material/color] [additional details]`

**Examples:**
- "Change the chair from wood to brushed steel with matte finish"
- "Change the dress to deep burgundy velvet"

---

## best practices

### 1. use prompt enhancement

The Qwen team strongly recommends using prompt enhancement for stable editing results.

> "Editing results may become unstable if prompt rewriting is not used."

For programmatic use:
```python
from tools.prompt_utils import polish_edit_prompt
enhanced_prompt = polish_edit_prompt(prompt, pil_image)
```

### 2. add magic suffixes for quality

Append quality modifiers to improve output:

| Language | Magic Suffix |
|----------|--------------|
| English | `, Ultra HD, 4K, cinematic composition` |
| Chinese | `, 超清，4K，电影级构图` |

### 3. be specific about text content

When editing text in images:
- Quote the exact text you want: `reading "Exact Text Here"`
- Specify font style: "in elegant serif font" or "用楷书字体"
- Note text size and placement: "large title at the top"

### 4. specify spatial relationships

For multi-image composition:
- Use clear positional terms: left, right, center, foreground, background
- Describe facing/orientation: "facing each other", "looking toward camera"
- Include context: "in a park setting", "on a beach at sunset"

### 5. use progressive refinement

For complex edits:
1. Start with a basic edit
2. Review the result
3. Refine with more specific instructions
4. Use region marking (bounding boxes) for precision when needed

### 6. match prompt language to content

- Use Chinese prompts for Chinese text rendering
- Use English prompts for English text rendering
- Can use bilingual prompts for mixed content

### 7. keep prompts focused

- Single-image edits: Focus on one or two changes per prompt
- Multi-image: Focus on composition and relationship between images
- Complex edits: Break into multiple steps if needed

---

## special capabilities

### integrated lora capabilities

The 2511 model includes built-in LoRA capabilities that activate through natural language prompts:

| Capability | Description | Prompt Pattern |
|------------|-------------|----------------|
| Lighting enhancement | Realistic lighting control | Describe lighting naturally - no special syntax |
| Viewpoint generation | New camera angles | "from a different angle", "side view", "top-down view" |

No explicit LoRA specification is needed - the model responds to natural descriptions.

### geometric reasoning

Enhanced ability to:
- Generate construction lines for design/annotation
- Maintain geometric consistency during edits
- Handle perspective transformations

### character consistency

Improved preservation of:
- Facial identity across style changes
- Body proportions during pose modifications
- Visual characteristics during artistic transformation

### multi-person consistency

When combining multiple people:
- Maintains individual identities
- Creates coherent group compositions
- Handles natural interactions between subjects

---

## technical parameters

### recommended settings

```python
inputs = {
    "image": [image1, image2],              # Single image or list
    "prompt": prompt,                        # Natural language instruction
    "negative_prompt": " ",                  # Empty string (recommended default)
    "num_inference_steps": 40,               # Diffusion steps
    "true_cfg_scale": 4.0,                   # Consistency control
    "guidance_scale": 1.0,                   # Standard guidance
    "generator": torch.manual_seed(0),       # For reproducibility
    "num_images_per_prompt": 1,              # Output count
}
```

### parameter notes

| Parameter | Value | Notes |
|-----------|-------|-------|
| `num_inference_steps` | 40 | Default for editing (was 50 in 2509) |
| `true_cfg_scale` | 4.0 | Controls edit strength vs preservation |
| `guidance_scale` | 1.0 | Standard for editing tasks |
| `negative_prompt` | " " | Empty string recommended |

### true_cfg_scale effects

| Value | Effect |
|-------|--------|
| 2.0 | Subtle edits, strong source preservation |
| 4.0 | Balanced (recommended default) |
| 6.0+ | Stronger edits, may reduce consistency |

---

## common mistakes to avoid

### mistake 1: command-style prompts

**Wrong:** "First remove the background, then add a sunset, then change colors"
**Right:** "The subject against a vivid sunset background, warm orange tones"

### mistake 2: vague instructions

**Wrong:** "Make it better"
**Right:** "Increase contrast, add warm color grading, sharpen details"

### mistake 3: conflicting instructions

**Wrong:** "Make it brighter and more moody with dramatic shadows"
**Right:** "Add dramatic side lighting with deep shadows, bright highlights"

### mistake 4: ignoring source image context

**Wrong:** "Put them in space" (when source has no relevant elements)
**Right:** "Both subjects floating in zero gravity against a starfield background"

### mistake 5: over-specification

**Wrong:** 200-word detailed technical specification
**Right:** Focused 1-2 sentence description of key changes

### mistake 6: wrong language for text rendering

**Wrong:** Using English prompt for Chinese character rendering
**Right:** Use Chinese prompts when the target text is Chinese

---

## see also

- [Qwen-Image-Edit-2511 Documentation](qwen_image_edit_2511.md) - Model integration and API
- [Testing Guide](qwen_image_edit_2511_testing.md) - Testing procedures
- [Qwen-Image Guide](../qwen_image_guide.md) - Complete workflow guide
- [HuggingFace Model](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) - Official model page
- [Qwen-Image GitHub](https://github.com/QwenLM/Qwen-Image) - Official repository
