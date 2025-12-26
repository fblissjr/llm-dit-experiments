#!/usr/bin/env python3
"""
last updated: 2025-12-19

Sweep test for Z-Image Omni with Bagel connector weights.
Generates comparison grids with different parameters.
"""

import sys
from pathlib import Path
from datetime import datetime

DIFFUSERS_PR_PATH = Path(__file__).parent.parent.parent / "coderef/diffusers/src"
sys.path.insert(0, str(DIFFUSERS_PR_PATH))

import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors import safe_open

# Paths
BAGEL_PATH = Path.home() / "Storage/ByteDance-Seed_BAGEL-7B-MoT/ema.safetensors"
ZIMAGE_PATH = Path.home() / "Storage/Tongyi-MAI_Z-Image-Turbo"
SIGLIP_PATH = Path.home() / "Storage/google_siglip2-so400m-patch14-384"
QWEN3_PATH = Path.home() / "Storage/Qwen3-4B"

INPUT_DIR = Path(__file__).parent.parent / "inputs"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "bagel_connector"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_and_adapt_bagel_connector(target_dim=3840):
    """Load Bagel's connector and adapt to Z-Image dimensions."""
    with safe_open(str(BAGEL_PATH), framework='pt', device='cpu') as f:
        fc1_weight = f.get_tensor('connector.fc1.weight')
        fc1_bias = f.get_tensor('connector.fc1.bias')

    current_out = fc1_weight.shape[0]
    pad_size = target_dim - current_out

    std = fc1_weight.float().std().item()
    pad_weight = torch.randn(pad_size, fc1_weight.shape[1], dtype=fc1_weight.dtype) * (std * 0.1)
    pad_bias = torch.zeros(pad_size, dtype=fc1_bias.dtype)

    adapted_weight = torch.cat([fc1_weight, pad_weight], dim=0)
    adapted_bias = torch.cat([fc1_bias, pad_bias], dim=0)

    return adapted_weight, adapted_bias


def load_pipeline():
    """Load Z-Image pipeline with Bagel connector weights."""
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
    from transformers import AutoModel, AutoTokenizer, AutoProcessor

    print("Loading pipeline...")

    vae = AutoencoderKL.from_pretrained(ZIMAGE_PATH, subfolder="vae", torch_dtype=torch.bfloat16)
    text_encoder = AutoModel.from_pretrained(QWEN3_PATH, dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_PATH, trust_remote_code=True)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(ZIMAGE_PATH, subfolder="scheduler")

    siglip_full = AutoModel.from_pretrained(SIGLIP_PATH)
    siglip = siglip_full.vision_model
    siglip_processor = AutoProcessor.from_pretrained(SIGLIP_PATH)

    base_transformer = ZImageTransformer2DModel.from_pretrained(
        ZIMAGE_PATH, subfolder="transformer", torch_dtype=torch.bfloat16
    )

    config = dict(base_transformer.config)
    config["siglip_feat_dim"] = 1152
    transformer = ZImageTransformer2DModel(**config)
    transformer.load_state_dict(base_transformer.state_dict(), strict=False)

    # Inject Bagel weights
    adapted_weight, adapted_bias = load_and_adapt_bagel_connector(3840)
    siglip_linear = transformer.siglip_embedder[1]
    with torch.no_grad():
        siglip_linear.weight.copy_(adapted_weight.to(siglip_linear.weight.dtype))
        siglip_linear.bias.copy_(adapted_bias.to(siglip_linear.bias.dtype))

    # Init siglip_refiner
    for name, param in transformer.named_parameters():
        if "siglip_refiner" in name:
            if param.dim() >= 2:
                torch.nn.init.xavier_uniform_(param, gain=0.1)
            else:
                torch.nn.init.zeros_(param)

    transformer = transformer.to(torch.bfloat16)
    print("Pipeline loaded with Bagel connector!")

    return {
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
        "transformer": transformer,
        "siglip": siglip,
        "siglip_processor": siglip_processor,
    }


def encode_prompt(tokenizer, text_encoder, prompt, device):
    formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(formatted, padding="max_length", max_length=512, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = text_encoder(
            input_ids=inputs.input_ids.to(device),
            attention_mask=inputs.attention_mask.to(device),
            output_hidden_states=True,
        )
        embeddings = outputs.hidden_states[-2]

    mask = inputs.attention_mask.bool()
    return embeddings[0][mask[0]]


def encode_siglip(siglip, processor, image, device, scale_factor=1.0):
    inputs = processor(images=[image], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = siglip(pixel_values=pixel_values)
        hidden_state = outputs.last_hidden_state

    B, N, C = hidden_state.shape
    H = W = int(N ** 0.5)
    hidden_state = hidden_state.squeeze(0).view(H, W, C)
    hidden_state = hidden_state * scale_factor

    return hidden_state


def generate_single(
    components,
    prompt,
    reference_image=None,
    scale_factor=1.0,
    height=512,
    width=512,
    num_steps=4,
    device="cuda",
):
    vae = components["vae"].to(device)
    text_encoder = components["text_encoder"].to(device)
    transformer = components["transformer"].to(device)
    siglip = components["siglip"].to(device)
    scheduler = components["scheduler"]
    tokenizer = components["tokenizer"]
    siglip_processor = components["siglip_processor"]

    dtype = torch.bfloat16

    cap_feats = encode_prompt(tokenizer, text_encoder, prompt, device).to(dtype)

    latent_h, latent_w = height // 16, width // 16
    latents = torch.randn(16, 1, latent_h, latent_w, device=device, dtype=dtype)

    if reference_image is None:
        cap_feats_list = [cap_feats]
        siglip_feats = None
    else:
        siglip_emb = encode_siglip(siglip, siglip_processor, reference_image, device, scale_factor).to(dtype)
        siglip_feats = [[siglip_emb]]
        cap_feats_list = [cap_feats]

    image_seq_len = latent_h * latent_w // 4
    mu = 0.5 + (1.15 - 0.5) * (image_seq_len - 256) / (4096 - 256)
    mu = max(0.5, min(1.15, mu))
    scheduler.set_timesteps(num_steps, device=device, mu=mu)

    for t in scheduler.timesteps:
        timestep = t.expand(1)
        timestep_norm = (1000 - timestep) / 1000
        x = [latents]

        with torch.no_grad():
            if siglip_feats is not None:
                output = transformer(x=x, t=timestep_norm, cap_feats=cap_feats_list, siglip_feats=siglip_feats, return_dict=False)
            else:
                output = transformer(x=x, t=timestep_norm, cap_feats=cap_feats_list, return_dict=False)

        noise_pred = -output[0][0].squeeze(1)
        latents_for_step = latents.squeeze(1)
        latents = scheduler.step(noise_pred.float(), t, latents_for_step.float(), return_dict=False)[0]
        latents = latents.unsqueeze(1).to(dtype)

    latents_for_decode = latents.squeeze(1).unsqueeze(0)
    latents_for_decode = (latents_for_decode / vae.config.scaling_factor) + vae.config.shift_factor

    with torch.no_grad():
        image = vae.decode(latents_for_decode, return_dict=False)[0]

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image[0] * 255).astype("uint8")
    return Image.fromarray(image)


def create_grid(images, labels, title, cols=4, cell_size=256):
    """Create a labeled grid of images."""
    n = len(images)
    rows = (n + cols - 1) // cols

    # Add space for labels
    label_height = 30
    grid_w = cols * cell_size
    grid_h = rows * (cell_size + label_height) + 40  # Extra for title

    grid = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    # Try to get a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()
        title_font = font

    # Draw title
    draw.text((10, 10), title, fill=(0, 0, 0), font=title_font)

    for i, (img, label) in enumerate(zip(images, labels)):
        row = i // cols
        col = i % cols

        x = col * cell_size
        y = 40 + row * (cell_size + label_height)

        # Resize image
        img_resized = img.resize((cell_size, cell_size), Image.LANCZOS)
        grid.paste(img_resized, (x, y))

        # Draw label
        draw.text((x + 5, y + cell_size + 5), label, fill=(0, 0, 0), font=font)

    return grid


def sweep_scale_factors(components, reference_image, prompt, output_path):
    """Sweep different scale factors for SigLIP embeddings."""
    print("\n=== Sweeping scale factors ===")

    scale_factors = [1.0, 3.0, 5.0, 8.0, 11.0, 15.0, 20.0, 30.0]
    images = []
    labels = []

    # First add the reference image
    ref_resized = reference_image.resize((512, 512), Image.LANCZOS)

    # Add text-only baseline
    print("  Generating text-only baseline...")
    baseline = generate_single(components, prompt, None, 1.0, 512, 512, 4, DEVICE)
    images.append(baseline)
    labels.append("Text only")

    # Add reference
    images.append(ref_resized)
    labels.append("Reference")

    for scale in scale_factors:
        print(f"  Generating with scale={scale}...")
        img = generate_single(components, prompt, reference_image, scale, 512, 512, 4, DEVICE)
        images.append(img)
        labels.append(f"scale={scale}")

    grid = create_grid(images, labels, f"Scale Factor Sweep | Prompt: {prompt[:50]}...", cols=5)
    grid.save(output_path)
    print(f"  Saved: {output_path}")


def sweep_styles(components, prompt, output_path):
    """Test different style reference images."""
    print("\n=== Sweeping style images ===")

    style_images = [
        ("style_anime_girl.png", "Anime"),
        ("style_oil_painting.png", "Oil Paint"),
        ("style_cyberpunk_city.png", "Cyberpunk"),
        ("style_watercolor_forest.png", "Watercolor"),
        ("style_pixel_art.png", "Pixel Art"),
        ("style_ukiyo_wave.png", "Ukiyo-e"),
    ]

    images = []
    labels = []

    # Text-only baseline
    print("  Generating text-only baseline...")
    baseline = generate_single(components, prompt, None, 1.0, 512, 512, 4, DEVICE)
    images.append(baseline)
    labels.append("Text only")

    scale_factor = 11.0  # Use default scale

    for filename, style_name in style_images:
        filepath = INPUT_DIR / filename
        if filepath.exists():
            print(f"  Generating with {style_name} style...")
            ref_img = Image.open(filepath).convert("RGB")
            img = generate_single(components, prompt, ref_img, scale_factor, 512, 512, 4, DEVICE)
            images.append(img)
            labels.append(style_name)

            # Also add the reference thumbnail
            # images.append(ref_img.resize((512, 512), Image.LANCZOS))
            # labels.append(f"{style_name} ref")

    grid = create_grid(images, labels, f"Style Transfer | Prompt: {prompt[:50]}... | scale={scale_factor}", cols=4)
    grid.save(output_path)
    print(f"  Saved: {output_path}")


def sweep_prompts(components, reference_image, ref_name, output_path):
    """Test different prompts with same reference."""
    print("\n=== Sweeping prompts ===")

    prompts = [
        "A cat sitting on a windowsill",
        "A mountain landscape at sunset",
        "A futuristic city skyline",
        "A portrait of an old man",
        "A bowl of fruit on a table",
        "An astronaut floating in space",
    ]

    images = []
    labels = []

    # Add reference
    ref_resized = reference_image.resize((512, 512), Image.LANCZOS)
    images.append(ref_resized)
    labels.append("Reference")

    scale_factor = 11.0

    for prompt in prompts:
        print(f"  Generating: {prompt[:30]}...")
        img = generate_single(components, prompt, reference_image, scale_factor, 512, 512, 4, DEVICE)
        images.append(img)
        labels.append(prompt[:25] + "...")

    grid = create_grid(images, labels, f"Prompt Sweep | Reference: {ref_name} | scale={scale_factor}", cols=4)
    grid.save(output_path)
    print(f"  Saved: {output_path}")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("="*60)
    print("Z-Image Omni + Bagel Connector - Parameter Sweep")
    print("="*60)

    # Load pipeline once
    components = load_pipeline()

    # Test 1: Scale factor sweep with anime style
    ref_path = INPUT_DIR / "style_anime_girl.png"
    if ref_path.exists():
        ref_img = Image.open(ref_path).convert("RGB")
        sweep_scale_factors(
            components,
            ref_img,
            "A portrait of a young woman with flowing hair",
            OUTPUT_DIR / f"sweep_scale_{timestamp}.png"
        )

    # Test 2: Style sweep with a fixed prompt
    sweep_styles(
        components,
        "A serene mountain landscape with a lake",
        OUTPUT_DIR / f"sweep_styles_{timestamp}.png"
    )

    # Test 3: Prompt sweep with oil painting reference
    ref_path = INPUT_DIR / "style_oil_painting.png"
    if ref_path.exists():
        ref_img = Image.open(ref_path).convert("RGB")
        sweep_prompts(
            components,
            ref_img,
            "Oil Painting",
            OUTPUT_DIR / f"sweep_prompts_{timestamp}.png"
        )

    print("\n" + "="*60)
    print(f"All results saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
