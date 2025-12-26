import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, ".")

from llm_dit.cli import load_runtime_config
from llm_dit.startup import PipelineLoader
from llm_dit.vl import VLEmbeddingExtractor, blend_embeddings

output_dir = Path("experiments/results/vl_correct_img2img")
output_dir.mkdir(parents=True, exist_ok=True)

sunset = Image.open("experiments/inputs/sunset.png").convert("RGB")
house = Image.open("experiments/inputs/test_scene.png").convert("RGB")
prompt = "Homer Simpson standing on the grass"

sunset.save(output_dir / "vl_source_sunset.png")
house.save(output_dir / "img2img_source_house.png")

# extract vl from SUNSET with FULL tokens
import os
vl_model_path = os.environ.get("QWEN3_VL_PATH")
if not vl_model_path:
    raise ValueError("Set QWEN3_VL_PATH environment variable")
vl_extractor = VLEmbeddingExtractor.from_pretrained(
    vl_model_path, device="cpu", torch_dtype=torch.bfloat16
)
vl_emb = vl_extractor.extract(
    sunset,
    text=prompt,
    hidden_layer=-6,
    text_tokens_only=False,  # FULL image tokens
    scale_to_text=True,
).embeddings
print(f"vl from sunset: {vl_emb.shape}")
vl_extractor.unload()
torch.cuda.empty_cache()


class C:
    config = "config.toml"
    profile = "default"


config = load_runtime_config(C())
pipe = PipelineLoader(config).load_pipeline().pipeline
text_emb = pipe.encoder.encode(prompt).embeddings[0]

# test matrix
for alpha in [0.0, 0.3, 0.5]:
    for strength in [0.7, 0.8, 0.9]:
        blended = text_emb if alpha == 0 else blend_embeddings(vl_emb, text_emb, alpha)
        gen = torch.Generator(device="cuda").manual_seed(42)
        # img2img from HOUSE, vl from SUNSET
        result = pipe.img2img(
            prompt_embeds=blended,
            image=house,
            strength=strength,
            num_inference_steps=config.steps,
            guidance_scale=config.guidance_scale,
            generator=gen,
        )
        result.save(output_dir / f"a{int(alpha * 10)}_s{int(strength * 10)}.png")
        print(f"a={alpha} s={strength}")

print(f"done: {output_dir}")

print(f"done: {output_dir}")
