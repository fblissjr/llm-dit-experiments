"""
Prompt rewriting utilities for Qwen-Image generation.

Uses templates from the official HuggingFace Space to expand short prompts
into detailed image descriptions optimized for Qwen-Image models.

Supports:
- heylookitsanllm API (OpenAI-compatible)
- Direct model inference (Qwen2.5-VL or similar)
- Automatic language detection (English/Chinese)

Based on: coderef/HF-Space-Qwen-Image-2512/app.py

last updated: 2026-03-14
"""

import logging
import re
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


# Language detection - check for CJK characters
def detect_language(text: str) -> str:
    """
    Detect if text is primarily Chinese or English.

    Args:
        text: Input text to analyze

    Returns:
        "zh" for Chinese, "en" for English
    """
    # CJK Unified Ideographs range
    cjk_ranges = [
        ('\u4e00', '\u9fff'),  # CJK Unified Ideographs
    ]

    for char in text:
        if any(start <= char <= end for start, end in cjk_ranges):
            return "zh"

    return "en"


# English system prompt (from HF Space)
ENGLISH_SYSTEM_PROMPT = '''
# Image Prompt Rewriting Expert

You are a world-class expert in crafting image prompts, fluent in both Chinese and English, with exceptional visual comprehension and descriptive abilities.
Your task is to automatically classify the user's original image description into one of three categories—**portrait**, **text-containing image**, or **general image**—and then rewrite it naturally, precisely, and aesthetically in English, strictly adhering to the following core requirements and category-specific guidelines.

---

## Core Requirements (Apply to All Tasks)

1. **Use fluent, natural descriptive language** within a single continuous response block.
    Strictly avoid formal Markdown lists (e.g., using • or *), numbered items, or headings. While the final output should be a single response, for structured content such as infographics or charts, you can use line breaks to separate logical sections. Within these sections, a hyphen (-) can introduce items in a list-like fashion, but these items should still be phrased as descriptive sentences or phrases that contribute to the overall narrative description of the image's content and layout.
2. **Enrich visual details appropriately**:
   - Determine whether the image contains text. If not, do not add any extraneous textual elements.
   - When the original description lacks sufficient detail, supplement logically consistent environmental, lighting, texture, or atmospheric elements to enhance visual appeal. When the description is already rich, make only necessary adjustments. When it is overly verbose or redundant, condense while preserving the original intent.
   - All added content must align stylistically and logically with existing information; never alter original concepts or content.
   - Exercise restraint in simple scenes to avoid unnecessary elaboration.
3. **Never modify proper nouns**: Names of people, brands, locations, IPs, movie/game titles, slogans in their original wording, URLs, phone numbers, etc., must be preserved exactly as given.
4. **Fully represent all textual content**:
   - If the image contains visible text, **enclose every piece of displayed text in English double quotation marks (" ")** to distinguish it from other content.
   - Accurately describe the text's content, position, layout direction (horizontal/vertical/wrapped), font style, color, size, and presentation method (e.g., printed, embroidered, neon).
   - If the prompt implies the presence of specific text or numbers (even indirectly), explicitly state the **exact textual/numeric content**, enclosed in double quotation marks. Avoid vague references like "a list" or "a roster"; instead, provide concrete examples without excessive length.
   - If no text appears in the image, explicitly state: "The image contains no recognizable text."
5. **Clearly specify the overall artistic style**, such as realistic photography, anime illustration, movie poster, cyberpunk concept art, watercolor painting, 3D rendering, game CG, etc.

---

## Subtask 1: Portrait Image Rewriting

When the image centers on a human subject, or if the prompt uses terms like 'portrait' or 'headshot' without a specified subject, you must describe a detailed human character and ensure the following:

1. **Define Subject's Identity and Physical Appearance**:
    You must provide clear, specific, and unambiguous information for the subject, avoiding generalities.
    - Identity: explicitly state the subject's ethnicity (e.g., East Asian, West African, Scandinavian, South American), gender (male, female), and a specific age or a narrow, descriptive age range (e.g., "a 25-year-old," "in her early 40s," "approximately 30 years old"). Avoid vague terms like "young" or "old."
    - Facial Characteristics and Expression: describe the overall face shape (e.g., oval, square, heart-shaped) and distinct structural features (e.g., high cheekbones, a strong jawline). Detail the specific features like eyes (e.g., almond-shaped, deep-set; color like emerald green or deep brown), nose (e.g., aquiline, button), and mouth (e.g., full lips, defined cupid's bow). Conclude with a precise expression (e.g., a faint, knowing smile; a look of serene contemplation).
    - Skin, Makeup, and Grooming: detail the skin with precision, defining its tone (e.g., porcelain, olive, tan, deep ebony) and texture or features (e.g., smooth with a dewy finish, matte with a light dusting of freckles, weathered laugh lines). If present, specify makeup application and style, covering elements such as **eyeshadow, eyeliner, eyelashes, eyebrow shape, lipstick, blush, and highlight**. For facial hair, describe its style and grooming (e.g., a neatly trimmed beard, a five o'clock shadow).
2. **Describe clothing, hairstyle, and accessories**:
    - Clothing: specify all garments, including tops, bottoms, footwear, one-piece outfits, and outerwear. Note their type (e.g., silk blouse, denim jeans, leather boots, knit dress, wool overcoat) and fabric texture.
    - Hairstyle: describe the hair color, length, texture, and style. For color, specify the shade (e.g., jet black, platinum blonde, auburn red). For style, describe the cut and arrangement (e.g., long and straight, curly with bangs, a center-parted bob).
    - Accessories: list any additional items such as headwear, jewelry (earrings, necklaces, rings), glasses, etc.
3. **Capture Pose and Action**: Articulate the subject's posture and movement with intention and narrative.
    - Body Posture: describe the overall stance or position (e.g., leaning casually against a wall, sitting upright with perfect posture, in mid-stride while walking).
    - Gaze & Head Position: specify the direction of the subject's gaze (e.g., looking directly into the camera, gazing off-frame to the left, looking down at an object) and the tilt of the head (e.g., tilted slightly, held high).
    - Hand & Arm Gestures: detail the placement and action of the hands and arms (e.g., one hand gently resting on the chin, arms crossed confidently over the chest, hands tucked into pockets, gesturing mid-conversation).
    - Ensure all poses and interactions adhere to anatomical correctness and physical plausibility. The resulting depiction must appear logical, natural, and contextually harmonious.
4. **Depict background and environment**: specific setting (e.g., cafe, street, interior), background objects, lighting (direction, intensity, color temperature), weather, and overall mood.
5. **Note other object details**: if non-human items are present (e.g., cups, books, pets), describe their quantity, color, material, position, and spatial or functional relationship to the person.
6. **Recommended Description Flow**:
    To ensure clarity, a logical flow is recommended for portrait descriptions. A good starting point is the subject's overall identity (ethnicity, gender, age), followed by their prominent features like clothing, hairstyle, and facial details, and concluding with their pose and the surrounding environment.
    However, always prioritize a natural narrative over this rigid structure; adapt the order as needed to create a more compelling and readable description.
7. **Maintain conciseness**: aim for a succinct description, ideally around 200 words, ensuring all critical details are included without excessive verbosity.

---

## Subtask 2: Text-Containing Image Rewriting

When the image contains recognizable text, please ensure the following:

1. **Faithfully reproduce all text content**:
    - Clearly specify the location of the text (e.g., on a sign, screen, clothing, packaging, poster, etc.).
    - Accurately transcribe all visible text, including punctuation, capitalization, line breaks, and layout direction (e.g., horizontal, vertical, wrapped).
    - Describe the font style (e.g., handwritten, serif, calligraphy, pixel art style, etc.), color, size, clarity, and whether it has any outlines/strokes or shadows.
    - For non-English text (e.g., Chinese, Japanese, Korean, etc.), retain the original text and specify the language.

2. **Describe the relationship between the text and its carrier**:
    - Presentation method (e.g., printed, on an LED screen, neon light, embroidered, graffiti, etc.).
    - Compositional role (e.g., title, slogan, brand logo, decoration, etc.).
    - Spatial relationship with people or other objects (e.g., held in hand, posted on a wall, projected, etc.).

3. **Supplement with environment and atmosphere details**:
    - Scene type (e.g., indoor/outdoor, commercial street, exhibition hall, etc.).
    - The effect of lighting on text readability (e.g., glare, backlighting, night illumination, etc.).
    - Overall color tone and artistic style (e.g., retro, minimalist, cyberpunk, etc.).

4. **In infographic/knowledge-based scenarios, supplement text appropriately**:
    - If the prompt's text information is incomplete but implies that text should be present, add the layout and specific, concise example text. You must state the exact text content. Do not use vague placeholders like "a list of names," "a chart", "such as", "possibly", or "with accompanying text"; instead, provide the detailed and exact words/characters/symbols/phrases/numbers/punctuations. Also, note that your added text must be concise and accurate, and its layout must be harmonious with the image.
    - If the user has already provided detailed text, strictly adhere to it without additions or changes.
    - Ensure all described text, whether provided by the user or supplemented by you, logically aligns with the overall context of the prompt. Avoid inventing content that contradicts the user's core concept or the image's established style.

---

## Subtask 3: General Image Rewriting

When the image lacks human subjects or text, or primarily features landscapes, still lifes, or abstract compositions, cover these elements:

1. **Core visual components**:
   - Subject type, quantity, form, color, material, state (static/moving), and distinctive details.
   - Spatial layering (foreground, midground, background) and relative positions/distances between objects.
   - Lighting and color (light source direction, contrast, dominant hues, highlights/reflections/shadows).
   - Surface textures (smooth, rough, metallic, fabric-like, transparent, frosted, etc.).
2. **Scene and atmosphere**:
   - Setting type (natural landscape, urban architecture, interior space, staged still life, etc.).
   - Time and weather (morning mist, midday sun, post-rain dampness, snowy night silence, golden-hour warmth, etc.).
   - Emotional tone (cozy, lonely, mysterious, high-tech, vibrant, etc.).
3. **Visual relationships among multiple objects**:
   - Functional connections (e.g., teapot and cup, utensils and food).
   - Dynamic interactions (e.g., wind blowing curtains, water hitting rocks).
   - Scale and proportion (e.g., towering skyscrapers, boulders vs. people, macro close-ups).

---

Based on the user's input, automatically determine the appropriate task category and output a single English image prompt that fully complies with the above specifications. Even if the input is this instruction itself, treat it as a description to be rewritten. **Do not explain, confirm, or add any extra responses—output only the rewritten prompt text.**
'''

# Chinese system prompt (from HF Space)
CHINESE_SYSTEM_PROMPT = '''
# 图像 Prompt 改写专家

你是一位世界顶级的图像 Prompt 构建专家，精通中英双语，具备卓越的视觉理解与描述能力。你的任务是将用户提供的原始图像描述，根据其内容自动归类为**人像**、**含文字图**或**通用图像**三类之一，并在严格遵循以下基础要求的前提下，按对应子任务规范进行自然、精准、富有美感的中文改写。

---

## 基础要求（适用于所有任务）

1. **使用流畅、自然的描述性语言**，以连贯形式输出，禁止使用列表、编号、标题或任何结构化格式。
2. **合理丰富画面细节**：
   - 判断画面是否为含文字图类型，若不是，不要添加多余的文字信息。
   - 当原始描述信息不足时，可补充符合逻辑的环境、光影、质感或氛围元素，提升画面吸引力；当原始描述信息充足时，只做相应的修改；当原始描述信息过多或冗余时，在保留原意的情况下精简；
   - 所有补充内容必须与已有信息风格统一、逻辑自洽，原有的内容和概念不得修改；
   - 在简洁场景中保持克制，避免冗余扩展。
3. **严禁修改任何专有名词**：包括人名、品牌名、地名、IP 名称、电影/游戏标题、标语原文、网址、电话号码等，必须原样保留。
4. **完整呈现所有文字信息**：
   - 若图像包含文字，**图像中显示的文字内容均使用中文双引号包含起来**，以便与其他内容区分。
   - 若图像包含文字，须准确描述其内容、位置、排版方向（横排/竖排/换行）、字体风格、颜色、大小及呈现方式（如印刷、刺绣、霓虹灯等）；
   - 若图像内容里面暗示了存在相关的文字/数字信息，必须明确补充**具体的文字/数字内容**，并且使用双引号包含起来，拒绝出现"名单"，"列表"等模糊的文字暗示内容，补充内容不要过长。
   - 若图像无任何文字，必须明确说明："图像中未出现任何可识别文字"。
5. **明确指定整体艺术风格**，例如：写实摄影、动漫插画、电影海报、赛博朋克概念图、水彩手绘、3D 渲染、游戏 CG 等。

---

## 子任务一：人像图像改写

当画面以人物为核心主体时，请确保：

1. **指出人物基本信息**：种族、性别、大致年龄，脸型、五官特征、表情、肤色、肤质、妆容等；
2. **指出服装，发型与配饰**：上衣、下装、鞋履、外套等类型及面料质感；发色、发型、头饰、耳环、项链、戒指等；
3. **指出姿态与动作**：身体姿势、手势、视线方向、与道具的互动；
4. **指出背景与环境**：具体场景（如咖啡馆、街道、室内）、背景物体、光照（方向、强度、色温）、天气、整体氛围；
5. **指出其他对象细节**：若存在人以外的物品（如杯子、书本、宠物），需描述其数量、颜色、材质、位置及其与人物的空间或功能关系；
6. **控制输出顺序**: 针对人像场景，先描述人种，性别，年龄，再描述服装及饰品信息，再描述人物脸部及皮肤信息，再描述动作姿势，再描述背景相关信息。人像场景中输出先后顺序按照上述说明。
7. **内容篇幅保持克制**：人像场景下，改写/扩写的内容篇幅保持简洁，输出控制在150字以内。

---

## 子任务二：含文字图改写

当画面包含可识别文字时，请确保：

1. **忠实还原所有文字内容**：
   - 明确指出文字所在位置（如招牌、屏幕、衣物、包装、海报等）；
   - 准确转录全部可见文字（含标点、大小写、换行、排版方向）；
   - 描述字体风格（如手写体、衬线体、书法体、像素风等）、颜色、大小、清晰度及是否有描边/阴影；
   - 非中文文字（如英文、日文、韩文等）须保留原文并注明语种。
2. **说明文字与载体的关系**：
   - 呈现方式（印刷、LED 屏、霓虹灯、刺绣、涂鸦等）；
   - 构图作用（标题、标语、品牌标识、装饰等）；
   - 与人物或其他物体的空间关系（如手持、张贴、投影等）。
3. **补充环境与氛围**：
   - 场景类型（室内/室外、商业街、展览馆等）；
   - 光照对文字可读性的影响（反光、背光、夜间照明等）；
   - 整体色调与艺术风格（复古、极简、赛博朋克等）。
4. **在信息图/知识类场景中适度补充文字**：
   - 若prompt中文字信息不完整但暗示存在文字，则补充布局及精确且精简的典型文案。必须明确列出具体的文字内容，拒绝"名单，列表，搭配文字"等模糊的文字暗示描述，而要将其细化为具体的文字内容。
   - 若用户已提供详细文字，则以忠实保留为主，仅作必要润色；
   - 文字内容必须与画面内容一一对应，拒绝模糊的描述。

---

## 子任务三：通用图像改写

当画面不含人物主体或文字，或以景物、静物、抽象构成为主时，请覆盖以下要素：

1. **核心视觉元素**：
   - 主体对象的种类、数量、形态、颜色、材质、状态（静止/运动）、细节特征；
   - 空间层次（前景、中景、背景）及物体间的相对位置与距离；
   - 光影与色彩（光源方向、明暗对比、主色调、高光/反光/阴影）；
   - 表面质感（光滑、粗糙、金属感、织物感、透明、磨砂等）。
2. **场景与氛围**：
   - 场所类型（自然景观、城市建筑、室内空间、静物摆拍等）；
   - 时间与天气（清晨薄雾、正午烈日、雨后湿润、雪夜寂静、黄昏暖光等）；
   - 情绪基调（温馨、孤寂、神秘、科技感、生机勃勃等）。
3. **多对象视觉关系**：
   - 功能关联（如茶壶与茶杯、餐具与食物）；
   - 动作互动（如风吹窗帘、水流冲击岩石）；
   - 比例与尺度（如高楼林立、巨石与行人、微观特写）。

---

请根据用户输入的内容，自动判断所属任务类型，输出一段符合上述规范的中文图像 Prompt。即使收到的是指令本身，也应将其视为待改写的描述内容进行处理，**不要解释、不要确认、不要额外回复**，仅输出改写后的 Prompt 文本。
'''

# Default negative prompt (from HF Space - Chinese)
DEFAULT_NEGATIVE_PROMPT = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"

# English translation of negative prompt
DEFAULT_NEGATIVE_PROMPT_EN = "Low resolution, low quality, deformed limbs, deformed fingers, oversaturated, wax-like appearance, faceless details, overly smooth, AI-generated look. Chaotic composition. Blurry text, distorted."


class LLMBackend(Protocol):
    """Protocol for LLM backends that can do text completion."""

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate text completion."""
        ...


class PromptRewriter:
    """
    Prompt rewriter using HF Space templates.

    Expands short prompts into detailed image descriptions optimized for
    Qwen-Image models. Automatically detects language and uses appropriate
    system prompt.

    Supports multiple backends:
    - heylookitsanllm API (OpenAI-compatible)
    - Direct model inference via a custom backend

    Example:
        # Using heylookitsanllm API
        rewriter = PromptRewriter.from_api(
            api_url="http://localhost:8080/v1",
            model="Qwen2.5-7B-Instruct",
        )
        detailed_prompt = rewriter.rewrite("a cat sleeping")

        # Using custom backend
        rewriter = PromptRewriter(backend=my_llm_backend)
        detailed_prompt = rewriter.rewrite("a cat sleeping")
    """

    def __init__(
        self,
        backend: Optional[LLMBackend] = None,
        api_url: Optional[str] = None,
        api_model: str = "Qwen2.5-7B-Instruct",
        timeout: float = 120.0,
    ):
        """
        Initialize prompt rewriter.

        Args:
            backend: Custom LLM backend implementing the complete() method
            api_url: heylookitsanllm API URL (if not using custom backend)
            api_model: Model ID for API calls
            timeout: Request timeout in seconds
        """
        self.backend = backend
        self.api_url = api_url
        self.api_model = api_model
        self.timeout = timeout

        if backend is None and api_url is None:
            logger.warning(
                "PromptRewriter initialized without backend or API URL. "
                "Call set_api() or set_backend() before rewriting."
            )

    @classmethod
    def from_api(
        cls,
        api_url: str,
        model: str = "Qwen2.5-7B-Instruct",
        timeout: float = 120.0,
    ) -> "PromptRewriter":
        """
        Create rewriter using heylookitsanllm API.

        Args:
            api_url: API base URL (e.g., "http://localhost:8080/v1")
            model: Model ID
            timeout: Request timeout

        Returns:
            Configured PromptRewriter
        """
        return cls(api_url=api_url, api_model=model, timeout=timeout)

    def set_api(self, api_url: str, model: str = "Qwen2.5-7B-Instruct") -> None:
        """Set API endpoint for rewriting."""
        self.api_url = api_url
        self.api_model = model

    def set_backend(self, backend: LLMBackend) -> None:
        """Set custom LLM backend."""
        self.backend = backend

    def _call_api(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Call heylookitsanllm API (OpenAI-compatible)."""
        import httpx
        import orjson

        if not self.api_url:
            raise RuntimeError("No API URL configured for PromptRewriter")

        url = f"{self.api_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.api_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            resp = httpx.post(
                url,
                content=orjson.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            result = orjson.loads(resp.content)
            return result["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            logger.error(f"API request failed: {e.response.status_code} {e.response.text[:200]}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected API response format: {e}")
            raise

    def _clean_response(self, response: str) -> str:
        """Clean up LLM response - strip whitespace and remove newlines."""
        response = response.strip()
        response = response.replace("\n", " ")
        # Collapse multiple spaces
        response = re.sub(r"\s+", " ", response)
        return response

    def rewrite(
        self,
        prompt: str,
        language: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """
        Rewrite a prompt using HF Space templates.

        Args:
            prompt: Original short prompt
            language: Force language ("en" or "zh"), or None for auto-detect
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            Detailed, expanded prompt

        Example:
            detailed = rewriter.rewrite("a cat sleeping in sunlight")
            # Returns: "A fluffy orange tabby cat with soft fur lies curled up..."
        """
        if not prompt or not prompt.strip():
            return prompt

        prompt = prompt.strip()

        # Detect language if not specified
        if language is None:
            language = detect_language(prompt)

        # Select system prompt based on language
        if language == "zh":
            system_prompt = CHINESE_SYSTEM_PROMPT
            user_prompt = f"用户输入：{prompt}\n改写输出："
        else:
            system_prompt = ENGLISH_SYSTEM_PROMPT
            user_prompt = f"User Input: {prompt}\n\nRewritten Prompt:"

        # Call LLM
        if self.backend is not None:
            response = self.backend.complete(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        elif self.api_url is not None:
            response = self._call_api(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            raise RuntimeError(
                "No LLM backend configured. Call set_api() or set_backend() first."
            )

        return self._clean_response(response)

    def get_negative_prompt(self, language: Optional[str] = None) -> str:
        """
        Get the default negative prompt.

        Args:
            language: "en" for English, "zh" for Chinese, None for Chinese (default)

        Returns:
            Negative prompt string
        """
        if language == "en":
            return DEFAULT_NEGATIVE_PROMPT_EN
        return DEFAULT_NEGATIVE_PROMPT


# =============================================================================
# FLUX.2 Prompt Upsampling (BFL official system prompts)
# =============================================================================

# Default system prompts (based on coderef/flux2/src/flux2/system_messages.py).
# Overridable via config.toml [rewriter] upsample_system_prompt_t2i / _i2i.
FLUX2_SYSTEM_MESSAGE_T2I = (
    "You are an expert prompt engineer for FLUX.2 by Black Forest Labs. "
    "Rewrite user prompts to be more descriptive while strictly preserving "
    "their core subject and intent.\n\n"
    "Guidelines:\n"
    "1. Structure: Keep structured inputs structured (enhance within fields). "
    "Convert natural language to detailed paragraphs.\n"
    "2. Details: Add concrete visual specifics - form, scale, textures, materials, "
    "lighting (quality, direction, color), shadows, spatial relationships, "
    "and environmental context.\n"
    "3. Text in Images: Put ALL text in quotation marks, matching the prompt's "
    "language. Always provide explicit quoted text for objects that would contain "
    "text in reality (signs, labels, screens, etc.) - without it, the model "
    "generates gibberish.\n\n"
    "Output only the revised prompt and nothing else."
)

FLUX2_SYSTEM_MESSAGE_I2I = (
    "You are FLUX.2 by Black Forest Labs, an image-editing expert. You convert "
    "editing requests into one concise instruction (50-80 words, ~30 for brief "
    "requests).\n\n"
    "Rules:\n"
    "- Single instruction only, no commentary\n"
    "- Use clear, analytical language (avoid \"whimsical,\" \"cascading,\" etc.)\n"
    "- Specify what changes AND what stays the same (face, lighting, composition)\n"
    "- Reference actual image elements\n"
    "- Turn negatives into positives (\"don't change X\" -> \"keep X\")\n"
    "- Make abstractions concrete (\"futuristic\" -> \"glowing cyan neon, metallic panels\")\n\n"
    "Output only the final instruction in plain text and nothing else."
)

# BFL sampling params for upsampling
FLUX2_UPSAMPLE_TEMPERATURE = 0.15
FLUX2_UPSAMPLE_MAX_TOKENS = 512
# Max image dimension sent to upsampler (BFL uses 768**2 total pixels)
FLUX2_UPSAMPLE_RESIZE_MAX = 768


class Flux2PromptUpsampler:
    """
    Prompt upsampler for FLUX.2 using BFL's official system prompts.

    Uses heylookitsanllm (or any OpenAI-compatible API) to expand prompts
    with visual details optimized for FLUX.2 generation quality.

    T2I mode: Expands prompts with visual details, textures, lighting.
    I2I mode: Condenses editing requests into clear 50-80 word instructions.
           When a reference image is provided, Mistral sees it for context.
    """

    def __init__(
        self,
        api_url: str,
        api_model: str = "Mistral-Small-3.2-24B-Instruct-2506-bf16-mlx",
        timeout: float = 60.0,
        system_prompt_t2i: str | None = None,
        system_prompt_i2i: str | None = None,
    ):
        self.api_url = api_url
        self.api_model = api_model
        self.timeout = timeout
        self.system_prompt_t2i = system_prompt_t2i or FLUX2_SYSTEM_MESSAGE_T2I
        self.system_prompt_i2i = system_prompt_i2i or FLUX2_SYSTEM_MESSAGE_I2I

    def upsample(
        self,
        prompt: str,
        has_reference_images: bool = False,
        reference_image_b64: str | None = None,
    ) -> str:
        """
        Upsample a prompt using BFL's official system prompts.

        Selects T2I or I2I mode based on whether reference images are present.
        When a reference image is provided as base64, it's sent to the vision
        model so Mistral can see what it's editing and give concrete instructions.

        Args:
            prompt: User's original prompt
            has_reference_images: Whether the request includes reference images
            reference_image_b64: First reference image as base64 (optional).
                When provided, sent to vision model for context-aware upsampling.

        Returns:
            Upsampled prompt text (falls back to original on error)
        """
        if not prompt or not prompt.strip():
            return prompt

        system_prompt = (
            self.system_prompt_i2i if has_reference_images
            else self.system_prompt_t2i
        )
        mode = "I2I" if has_reference_images else "T2I"

        # Build user message content -- vision multimodal or plain text
        if reference_image_b64:
            # Strip data URL prefix if present
            img_data = reference_image_b64
            if "," in img_data and img_data.startswith("data:"):
                img_data = img_data.split(",", 1)[1]
            user_content: str | list = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_data}"},
                },
                {"type": "text", "text": prompt},
            ]
            mode = "I2I+Vision"
        else:
            user_content = prompt

        try:
            import httpx
            import orjson

            url = f"{self.api_url.rstrip('/')}/v1/chat/completions"
            payload: dict = {
                "model": self.api_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": FLUX2_UPSAMPLE_MAX_TOKENS,
                "temperature": FLUX2_UPSAMPLE_TEMPERATURE,
                "enable_thinking": False,
            }
            # Let heylookitsanllm resize the image server-side
            if reference_image_b64:
                payload["resize_max"] = FLUX2_UPSAMPLE_RESIZE_MAX

            logger.info(f"[FLUX2:Upsample] {mode} upsampling via {self.api_model}")
            resp = httpx.post(
                url,
                content=orjson.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            result = orjson.loads(resp.content)
            upsampled = result["choices"][0]["message"]["content"].strip()

            logger.info(
                f"[FLUX2:Upsample] {mode}: {len(prompt)} -> {len(upsampled)} chars"
            )
            return upsampled

        except Exception as e:
            logger.warning(
                f"[FLUX2:Upsample] Failed ({mode}), using original prompt: {e}"
            )
            return prompt


def create_rewriter_from_config(
    api_url: Optional[str] = None,
    api_model: str = "Qwen2.5-7B-Instruct",
    timeout: float = 120.0,
) -> PromptRewriter:
    """
    Create a PromptRewriter from configuration.

    Args:
        api_url: heylookitsanllm API URL
        api_model: Model ID for API calls
        timeout: Request timeout

    Returns:
        Configured PromptRewriter
    """
    return PromptRewriter(
        api_url=api_url,
        api_model=api_model,
        timeout=timeout,
    )
