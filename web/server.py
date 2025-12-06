#!/usr/bin/env python3
"""
Simple web server for Z-Image generation.

Usage:
    uv run web/server.py
    uv run web/server.py --port 8000
    uv run web/server.py --config config.toml --profile default
    uv run web/server.py --encoder-only  # Fast mode, no DiT/VAE
"""

import argparse
import asyncio
import io
import logging
import time
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Z-Image Generator")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline/encoder (loaded on startup)
pipeline = None
encoder = None  # For encoder-only mode
config = None
encoder_only_mode = False

# In-memory history (cleared on server restart)
generation_history = []
MAX_HISTORY = 50


class GenerateRequest(BaseModel):
    prompt: str  # User prompt
    system_prompt: Optional[str] = None  # System prompt (optional)
    thinking_content: Optional[str] = None  # Content inside <think>...</think> (triggers think block)
    assistant_content: Optional[str] = None  # Content after </think> (optional)
    force_think_block: bool = False  # If True, add empty think block even without content
    width: int = 1024
    height: int = 1024
    steps: int = 9
    seed: Optional[int] = None
    template: Optional[str] = None
    guidance_scale: float = 0.0
    shift: float = 3.0  # Scheduler shift parameter


class EncodeRequest(BaseModel):
    prompt: str  # User prompt
    system_prompt: Optional[str] = None  # System prompt (optional)
    thinking_content: Optional[str] = None  # Content inside <think>...</think> (triggers think block)
    assistant_content: Optional[str] = None  # Content after </think> (optional)
    force_think_block: bool = False  # If True, add empty think block even without content
    template: Optional[str] = None


class RewriteRequest(BaseModel):
    prompt: str  # User prompt to rewrite/expand
    rewriter: str  # Name of rewriter template to use
    max_tokens: int = 512  # Maximum tokens to generate
    temperature: float = 0.7  # Sampling temperature


@app.get("/")
async def index():
    """Serve the main page."""
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "pipeline_loaded": pipeline is not None,
        "encoder_loaded": encoder is not None,
        "encoder_only_mode": encoder_only_mode,
    }


@app.post("/api/encode")
async def encode(request: EncodeRequest):
    """Encode a prompt to embeddings (for distributed inference)."""
    # Use encoder from pipeline or standalone encoder
    enc = encoder if encoder is not None else (pipeline.encoder if pipeline else None)
    if enc is None:
        raise HTTPException(status_code=503, detail="Encoder not loaded")

    try:
        start = time.time()
        output = enc.encode(
            request.prompt,
            template=request.template,
            system_prompt=request.system_prompt,
            thinking_content=request.thinking_content,
            assistant_content=request.assistant_content,
            force_think_block=request.force_think_block,
        )
        encode_time = time.time() - start

        embeddings = output.embeddings[0]

        # Get formatted prompt if available
        formatted_prompt = None
        if output.formatted_prompts:
            formatted_prompt = output.formatted_prompts[0]
            logger.info(f"Formatted prompt ({len(formatted_prompt)} chars):")
            logger.info(f"---BEGIN FORMATTED PROMPT---")
            logger.info(formatted_prompt)
            logger.info(f"---END FORMATTED PROMPT---")

        return {
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
            "encode_time": encode_time,
            "prompt": request.prompt,
            "formatted_prompt": formatted_prompt,
        }
    except Exception as e:
        logger.error(f"Encoding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate")
async def generate(request: GenerateRequest):
    """Generate an image from a prompt."""
    if encoder_only_mode:
        raise HTTPException(
            status_code=400,
            detail="Server running in encoder-only mode. Use /api/encode instead."
        )
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    try:
        logger.info("=" * 60)
        logger.info("GENERATION REQUEST")
        logger.info("=" * 60)
        logger.info(f"  Prompt: {request.prompt[:80]}...")
        logger.info(f"  Size: {request.width}x{request.height}")
        logger.info(f"  Steps: {request.steps}")
        logger.info(f"  Seed: {request.seed}")
        logger.info(f"  Template: {request.template}")
        logger.info(f"  Force think block: {request.force_think_block}")
        logger.info(f"  Guidance: {request.guidance_scale}")
        logger.info("-" * 60)
        logger.info("Pipeline state:")
        logger.info(f"  pipeline.device: {pipeline.device}")
        logger.info(f"  pipeline.dtype: {pipeline.dtype}")
        logger.info(f"  pipeline.encoder: {type(pipeline.encoder).__name__ if pipeline.encoder else 'None'}")
        logger.info(f"  pipeline.transformer: {pipeline.transformer is not None}")
        logger.info(f"  pipeline.vae: {pipeline.vae is not None}")
        if pipeline.encoder:
            backend = getattr(pipeline.encoder, 'backend', None)
            logger.info(f"  encoder.backend: {type(backend).__name__ if backend else 'None'}")
        logger.info("-" * 60)

        # Set up generator for reproducibility
        generator = None
        if request.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(request.seed)

        start = time.time()

        # Generate image
        logger.info("Calling pipeline()...")
        image = pipeline(
            request.prompt,
            height=request.height,
            width=request.width,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
            template=request.template,
            system_prompt=request.system_prompt,
            thinking_content=request.thinking_content,
            assistant_content=request.assistant_content,
            force_think_block=request.force_think_block,
        )

        gen_time = time.time() - start
        logger.info(f"Generated in {gen_time:.1f}s")
        logger.info("=" * 60)

        # Convert to PNG bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        # Convert to base64 for history storage
        import base64
        img_bytes_copy = io.BytesIO()
        image.save(img_bytes_copy, format="PNG")
        img_b64 = base64.b64encode(img_bytes_copy.getvalue()).decode("ascii")

        # Get formatted prompt for history
        formatted_prompt = None
        enc = encoder if encoder is not None else (pipeline.encoder if pipeline else None)
        if enc:
            try:
                from llm_dit.conversation import Conversation
                conv = enc._build_conversation(
                    prompt=request.prompt,
                    template=request.template,
                    system_prompt=request.system_prompt,
                    thinking_content=request.thinking_content,
                    assistant_content=request.assistant_content,
                    force_think_block=request.force_think_block,
                )
                formatted_prompt = enc.formatter.format(conv)
            except Exception as e:
                logger.warning(f"Failed to get formatted prompt: {e}")

        # Store in history
        history_entry = {
            "id": len(generation_history),
            "timestamp": time.time(),
            "prompt": request.prompt,
            "system_prompt": request.system_prompt,
            "thinking_content": request.thinking_content,
            "assistant_content": request.assistant_content,
            "force_think_block": request.force_think_block,
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "seed": request.seed,
            "template": request.template,
            "guidance_scale": request.guidance_scale,
            "gen_time": gen_time,
            "image_b64": img_b64,
            "formatted_prompt": formatted_prompt,
        }
        generation_history.insert(0, history_entry)
        # Trim history
        if len(generation_history) > MAX_HISTORY:
            generation_history.pop()

        return StreamingResponse(
            img_bytes,
            media_type="image/png",
            headers={
                "X-Generation-Time": str(gen_time),
                "X-Seed": str(request.seed) if request.seed else "random",
                "X-History-Id": str(history_entry["id"]),
            },
        )

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/format-prompt")
async def format_prompt_endpoint(request: EncodeRequest):
    """Preview the formatted prompt without encoding (fast, no GPU needed)."""
    # Use encoder from pipeline or standalone encoder
    enc = encoder if encoder is not None else (pipeline.encoder if pipeline else None)
    if enc is None:
        raise HTTPException(status_code=503, detail="Encoder not loaded")

    try:
        # Build conversation and format without encoding
        from llm_dit.conversation import Conversation

        conv = enc._build_conversation(
            prompt=request.prompt,
            template=request.template,
            system_prompt=request.system_prompt,
            thinking_content=request.thinking_content,
            assistant_content=request.assistant_content,
            force_think_block=request.force_think_block,
        )
        formatted = enc.formatter.format(conv)

        return {
            "formatted_prompt": formatted,
            "char_count": len(formatted),
            "prompt": request.prompt,
            "system_prompt": request.system_prompt,
            "thinking_content": request.thinking_content,
            "assistant_content": request.assistant_content,
            "template": request.template,
            "force_think_block": request.force_think_block,
        }
    except Exception as e:
        logger.error(f"Format failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/templates")
async def list_templates():
    """List available templates."""
    # Use encoder from pipeline or standalone encoder
    enc = encoder if encoder is not None else (pipeline.encoder if pipeline else None)
    if enc is None or enc.templates is None:
        return {"templates": []}

    templates = []
    for name in enc.templates:
        tpl = enc.templates.get(name)
        templates.append({
            "name": name,
            "has_thinking": bool(tpl.thinking_content) if tpl else False,
        })

    return {"templates": templates}


@app.get("/api/rewriters")
async def list_rewriters():
    """List available rewriter templates."""
    # Use encoder from pipeline or standalone encoder
    enc = encoder if encoder is not None else (pipeline.encoder if pipeline else None)
    if enc is None or enc.templates is None:
        return {"rewriters": []}

    # Get rewriter templates (category == "rewriter")
    rewriters = []
    for tpl in enc.templates.list_by_category("rewriter"):
        rewriters.append({
            "name": tpl.name,
            "description": tpl.description,
        })

    return {"rewriters": rewriters}


@app.post("/api/rewrite")
async def rewrite_prompt(request: RewriteRequest):
    """
    Rewrite/expand a prompt using a rewriter template.

    Uses the same Qwen3 model loaded for text encoding to generate expanded prompts.
    This enables prompt enhancement without loading additional models.
    """
    # Use encoder from pipeline or standalone encoder
    enc = encoder if encoder is not None else (pipeline.encoder if pipeline else None)
    if enc is None:
        raise HTTPException(status_code=503, detail="Encoder not loaded")

    # Check if backend supports generation
    backend = getattr(enc, 'backend', None)
    if backend is None:
        raise HTTPException(status_code=503, detail="No backend available for generation")

    if not getattr(backend, 'supports_generation', False):
        raise HTTPException(
            status_code=400,
            detail="Backend does not support text generation"
        )

    # Get the rewriter template
    if enc.templates is None:
        raise HTTPException(status_code=400, detail="No templates loaded")

    rewriter_template = enc.templates.get(request.rewriter)
    if rewriter_template is None:
        raise HTTPException(
            status_code=404,
            detail=f"Rewriter template not found: {request.rewriter}"
        )

    if rewriter_template.category != "rewriter":
        raise HTTPException(
            status_code=400,
            detail=f"Template '{request.rewriter}' is not a rewriter template"
        )

    try:
        start = time.time()
        logger.info(f"[Rewrite] Using template: {request.rewriter}")
        logger.info(f"[Rewrite] Input prompt: {request.prompt[:100]}...")

        # Generate using the backend
        generated = backend.generate(
            prompt=request.prompt,
            system_prompt=rewriter_template.content,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        gen_time = time.time() - start
        logger.info(f"[Rewrite] Generated {len(generated)} chars in {gen_time:.2f}s")

        return {
            "original_prompt": request.prompt,
            "rewritten_prompt": generated,
            "rewriter": request.rewriter,
            "gen_time": gen_time,
        }

    except Exception as e:
        logger.error(f"Rewrite failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
async def get_history():
    """Get generation history."""
    return {"history": generation_history}


@app.delete("/api/history/{index}")
async def delete_history_item(index: int):
    """Delete a history item."""
    if 0 <= index < len(generation_history):
        deleted = generation_history.pop(index)
        return {"deleted": deleted, "remaining": len(generation_history)}
    raise HTTPException(status_code=404, detail="History item not found")


@app.delete("/api/history")
async def clear_history():
    """Clear all history."""
    global generation_history
    count = len(generation_history)
    generation_history = []
    return {"cleared": count}


@app.post("/api/save-embeddings")
async def save_embeddings_endpoint(request: EncodeRequest):
    """Encode and save embeddings to file for distributed inference."""
    # Use encoder from pipeline or standalone encoder
    enc = encoder if encoder is not None else (pipeline.encoder if pipeline else None)
    if enc is None:
        raise HTTPException(status_code=503, detail="Encoder not loaded")

    try:
        from llm_dit.distributed import save_embeddings as save_emb

        start = time.time()
        output = enc.encode(
            request.prompt,
            template=request.template,
            force_think_block=request.force_think_block,
        )
        encode_time = time.time() - start

        embeddings = output.embeddings[0]

        # Generate filename from prompt
        import hashlib
        prompt_hash = hashlib.md5(request.prompt.encode()).hexdigest()[:8]
        filename = f"embeddings_{prompt_hash}.safetensors"
        output_dir = Path(__file__).parent.parent / "embeddings"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / filename

        # Get device from encoder or pipeline
        device = str(enc.device) if hasattr(enc, 'device') else 'unknown'

        save_path = save_emb(
            embeddings=embeddings,
            path=str(output_path),
            prompt=request.prompt,
            model_path='unknown',  # Not stored in encoder
            template=request.template,
            force_think_block=request.force_think_block,
            encoder_device=device,
        )

        return {
            "path": str(save_path),
            "shape": list(embeddings.shape),
            "encode_time": encode_time,
        }

    except Exception as e:
        logger.error(f"Save embeddings failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def load_pipeline(
    model_path: str,
    templates_dir: Optional[str] = None,
    encoder_device: str = "auto",
    dit_device: str = "auto",
    vae_device: str = "auto",
    lora_paths: Optional[list] = None,
    lora_scales: Optional[list] = None,
):
    """Load the full generation pipeline."""
    global pipeline

    from llm_dit.pipelines import ZImagePipeline

    logger.info(f"Loading pipeline from {model_path}...")
    logger.info(f"  Encoder device: {encoder_device}")
    logger.info(f"  DiT device: {dit_device}")
    logger.info(f"  VAE device: {vae_device}")
    start = time.time()

    pipeline = ZImagePipeline.from_pretrained(
        model_path,
        templates_dir=templates_dir,
        torch_dtype=torch.bfloat16,
        encoder_device=encoder_device,
        dit_device=dit_device,
        vae_device=vae_device,
    )

    load_time = time.time() - start
    logger.info(f"Pipeline loaded in {load_time:.1f}s")
    logger.info(f"Device: {pipeline.device}")

    # Load LoRAs if configured
    if lora_paths:
        logger.info(f"Loading {len(lora_paths)} LoRA(s)...")
        scales = lora_scales if lora_scales else [1.0] * len(lora_paths)
        try:
            updated = pipeline.load_lora(lora_paths, scale=scales)
            logger.info(f"  {updated} layers updated by LoRA")
        except Exception as e:
            logger.error(f"  Failed to load LoRA: {e}")


def load_encoder_only(
    model_path: str,
    templates_dir: Optional[str] = None,
    encoder_device: str = "auto",
):
    """Load only the encoder (fast mode for testing on Mac)."""
    global encoder, encoder_only_mode

    from llm_dit.encoders import ZImageTextEncoder

    logger.info(f"Loading encoder only from {model_path}...")
    logger.info(f"  Encoder device: {encoder_device}")
    start = time.time()

    encoder = ZImageTextEncoder.from_pretrained(
        model_path,
        templates_dir=templates_dir,
        torch_dtype=torch.bfloat16,
        device_map=encoder_device,
    )

    encoder_only_mode = True
    load_time = time.time() - start
    logger.info(f"Encoder loaded in {load_time:.1f}s (encoder-only mode)")
    logger.info(f"Device: {encoder.device}")


def load_api_encoder(
    api_url: str,
    model_id: str,
    templates_dir: Optional[str] = None,
):
    """Load encoder that uses heylookitsanllm API backend (encoder-only mode)."""
    global encoder, encoder_only_mode

    from llm_dit.backends.api import APIBackend, APIBackendConfig
    from llm_dit.encoders import ZImageTextEncoder
    from llm_dit.templates import TemplateRegistry

    logger.info(f"Connecting to API backend at {api_url}...")

    # Create API backend
    api_config = APIBackendConfig(
        base_url=api_url,
        model_id=model_id,
        encoding_format="base64",
    )
    backend = APIBackend(api_config)

    # Load templates if provided
    templates = None
    if templates_dir:
        templates = TemplateRegistry.from_directory(templates_dir)
        logger.info(f"Loaded {len(templates)} templates")

    # Create encoder with API backend
    encoder = ZImageTextEncoder(
        backend=backend,
        templates=templates,
    )

    encoder_only_mode = True
    logger.info(f"API encoder ready (model: {model_id})")


def load_hybrid_pipeline(
    model_path: str,
    templates_dir: Optional[str] = None,
    enable_cpu_offload: bool = False,
    enable_flash_attn: bool = False,
    enable_compile: bool = False,
    encoder_device: str = "cpu",
    dit_device: str = "cuda",
    vae_device: str = "cuda",
    lora_paths: Optional[list] = None,
    lora_scales: Optional[list] = None,
):
    """Load full pipeline with local encoder + DiT/VAE (for A/B testing vs API)."""
    global pipeline, encoder_only_mode

    from llm_dit.pipelines import ZImagePipeline

    logger.info("=" * 60)
    logger.info("HYBRID MODE SETUP (local encoder + local DiT/VAE)")
    logger.info("=" * 60)
    logger.info(f"  Model path: {model_path}")
    logger.info(f"  Templates: {templates_dir}")
    logger.info(f"  Encoder device: {encoder_device}")
    logger.info(f"  DiT device: {dit_device}")
    logger.info(f"  VAE device: {vae_device}")
    logger.info(f"  CPU Offload: {enable_cpu_offload}")
    logger.info(f"  Flash Attention: {enable_flash_attn}")
    logger.info(f"  Torch Compile: {enable_compile}")
    logger.info("-" * 60)

    start = time.time()

    # Load full pipeline with device placement
    pipeline = ZImagePipeline.from_pretrained(
        model_path,
        templates_dir=templates_dir,
        torch_dtype=torch.bfloat16,
        encoder_device=encoder_device,
        dit_device=dit_device,
        vae_device=vae_device,
    )

    load_time = time.time() - start
    logger.info(f"Pipeline loaded in {load_time:.1f}s")

    # Apply optimizations
    if enable_flash_attn:
        logger.info("Enabling Flash Attention...")
        try:
            pipeline.transformer.set_attention_backend("flash")
            logger.info("  Flash Attention enabled")
        except Exception as e:
            logger.warning(f"  Failed to enable Flash Attention: {e}")
            logger.warning("  Install with: pip install flash-attn --no-build-isolation")

    if enable_compile:
        logger.info("Compiling transformer with torch.compile...")
        try:
            pipeline.transformer = torch.compile(pipeline.transformer, mode="reduce-overhead")
            logger.info("  Transformer compiled (first run will be slow)")
        except Exception as e:
            logger.warning(f"  Failed to compile: {e}")

    # Load LoRAs if configured
    if lora_paths:
        logger.info(f"Loading {len(lora_paths)} LoRA(s)...")
        scales = lora_scales if lora_scales else [1.0] * len(lora_paths)
        try:
            updated = pipeline.load_lora(lora_paths, scale=scales)
            logger.info(f"  {updated} layers updated by LoRA")
        except Exception as e:
            logger.error(f"  Failed to load LoRA: {e}")

    encoder_only_mode = False
    logger.info("-" * 60)
    logger.info(f"Hybrid pipeline ready (local encoder on {encoder_device})")
    logger.info(f"  Encoder device: {pipeline.encoder.device}")
    logger.info(f"  DiT device: {next(pipeline.transformer.parameters()).device}")
    logger.info(f"  VAE device: {next(pipeline.vae.parameters()).device}")
    logger.info("=" * 60)


def load_api_pipeline(
    api_url: str,
    model_id: str,
    model_path: str,
    templates_dir: Optional[str] = None,
    enable_cpu_offload: bool = False,
    enable_flash_attn: bool = False,
    enable_compile: bool = False,
    dit_device: str = "auto",
    vae_device: str = "auto",
    lora_paths: Optional[list] = None,
    lora_scales: Optional[list] = None,
):
    """Load full pipeline with API backend for encoding + local DiT/VAE for generation."""
    global pipeline, encoder_only_mode

    from llm_dit.backends.api import APIBackend, APIBackendConfig
    from llm_dit.encoders import ZImageTextEncoder
    from llm_dit.pipelines import ZImagePipeline
    from llm_dit.templates import TemplateRegistry

    logger.info("=" * 60)
    logger.info("DISTRIBUTED MODE SETUP")
    logger.info("=" * 60)
    logger.info(f"  API URL: {api_url}")
    logger.info(f"  API Model: {model_id}")
    logger.info(f"  Local Model: {model_path}")
    logger.info(f"  Templates: {templates_dir}")
    logger.info(f"  CPU Offload: {enable_cpu_offload}")
    logger.info(f"  Flash Attention: {enable_flash_attn}")
    logger.info(f"  Torch Compile: {enable_compile}")
    logger.info("-" * 60)

    # Create API backend for encoding
    logger.info("Creating API backend...")
    api_config = APIBackendConfig(
        base_url=api_url,
        model_id=model_id,
        encoding_format="base64",
    )
    backend = APIBackend(api_config)
    logger.info(f"  Backend created: {backend}")

    # Load templates if provided
    templates = None
    if templates_dir:
        templates = TemplateRegistry.from_directory(templates_dir)
        logger.info(f"  Loaded {len(templates)} templates")

    # Create encoder with API backend
    logger.info("Creating API-backed encoder...")
    api_encoder = ZImageTextEncoder(
        backend=backend,
        templates=templates,
    )
    logger.info(f"  Encoder created: {api_encoder}")
    logger.info(f"  Encoder device: {getattr(api_encoder, 'device', 'N/A')}")

    logger.info("-" * 60)
    logger.info(f"Loading DiT/VAE from {model_path}...")
    logger.info(f"  DiT device: {dit_device}")
    logger.info(f"  VAE device: {vae_device}")
    start = time.time()

    # Load generator-only pipeline, then attach our API encoder
    pipeline = ZImagePipeline.from_pretrained_generator_only(
        model_path,
        torch_dtype=torch.bfloat16,
        enable_cpu_offload=enable_cpu_offload,
        dit_device=dit_device,
        vae_device=vae_device,
    )

    load_time = time.time() - start
    logger.info(f"  DiT/VAE loaded in {load_time:.1f}s")
    logger.info(f"  Transformer device: {pipeline.transformer.device if pipeline.transformer else 'None'}")
    logger.info(f"  Transformer dtype: {next(pipeline.transformer.parameters()).dtype if pipeline.transformer else 'None'}")
    logger.info(f"  VAE device: {next(pipeline.vae.parameters()).device if pipeline.vae else 'None'}")

    # Apply optimizations
    if enable_flash_attn:
        logger.info("Enabling Flash Attention...")
        try:
            pipeline.transformer.set_attention_backend("flash")
            logger.info("  Flash Attention enabled")
        except Exception as e:
            logger.warning(f"  Failed to enable Flash Attention: {e}")
            logger.warning("  Install with: pip install flash-attn --no-build-isolation")

    if enable_compile:
        logger.info("Compiling transformer with torch.compile...")
        try:
            pipeline.transformer = torch.compile(pipeline.transformer, mode="reduce-overhead")
            logger.info("  Transformer compiled (first run will be slow)")
        except Exception as e:
            logger.warning(f"  Failed to compile: {e}")

    # Replace the encoder with our API-backed one
    logger.info("Attaching API encoder to pipeline...")
    pipeline.encoder = api_encoder

    # Load LoRAs if configured
    if lora_paths:
        logger.info(f"Loading {len(lora_paths)} LoRA(s)...")
        scales = lora_scales if lora_scales else [1.0] * len(lora_paths)
        try:
            updated = pipeline.load_lora(lora_paths, scale=scales)
            logger.info(f"  {updated} layers updated by LoRA")
        except Exception as e:
            logger.error(f"  Failed to load LoRA: {e}")

    logger.info("-" * 60)
    encoder_only_mode = False
    opts = []
    if enable_cpu_offload:
        opts.append("CPU offload")
    if enable_flash_attn:
        opts.append("Flash Attn")
    if enable_compile:
        opts.append("compiled")
    opts_str = f" ({', '.join(opts)})" if opts else ""
    logger.info(f"Pipeline ready (API encoder + local DiT/VAE{opts_str})")
    logger.info(f"  pipeline.device: {pipeline.device}")
    logger.info(f"  pipeline.dtype: {pipeline.dtype}")
    logger.info(f"  pipeline.encoder: {pipeline.encoder}")
    logger.info(f"  pipeline.transformer: {pipeline.transformer}")
    logger.info(f"  pipeline.vae: {pipeline.vae}")
    logger.info("=" * 60)


def main():
    # Use shared CLI argument parser
    from llm_dit.cli import create_base_parser, load_runtime_config, setup_logging

    parser = create_base_parser(
        description="Z-Image web server",
        include_server_args=True,
        include_generation_args=True,
    )

    # Add server-specific arguments
    parser.add_argument(
        "--encoder-only",
        action="store_true",
        help="Load only encoder (fast mode for Mac, no image generation)",
    )

    args = parser.parse_args()

    # Load unified config (handles TOML + CLI overrides)
    runtime_config = load_runtime_config(args)
    setup_logging(runtime_config)

    # Extract values from runtime config
    host = runtime_config.host
    port = runtime_config.port
    model_path = runtime_config.model_path
    templates_dir = runtime_config.templates_dir

    # Find default templates if not specified
    if templates_dir is None:
        default_templates = Path(__file__).parent.parent / "templates" / "z_image"
        if default_templates.exists():
            templates_dir = str(default_templates)

    # Determine which mode to use
    if runtime_config.api_url and model_path and runtime_config.local_encoder:
        # Hybrid mode: Local encoder (for A/B testing) + local DiT/VAE
        load_hybrid_pipeline(
            model_path,
            templates_dir,
            enable_cpu_offload=runtime_config.cpu_offload,
            enable_flash_attn=runtime_config.flash_attn,
            enable_compile=runtime_config.compile,
            encoder_device=runtime_config.encoder_device,
            dit_device=runtime_config.dit_device,
            vae_device=runtime_config.vae_device,
            lora_paths=runtime_config.lora_paths,
            lora_scales=runtime_config.lora_scales,
        )
        mode = f"hybrid (local encoder on {runtime_config.encoder_device}, for A/B testing vs API)"
    elif runtime_config.api_url and model_path:
        # Distributed mode: API encoding + local DiT/VAE generation
        load_api_pipeline(
            runtime_config.api_url,
            runtime_config.api_model,
            model_path,
            templates_dir,
            enable_cpu_offload=runtime_config.cpu_offload,
            enable_flash_attn=runtime_config.flash_attn,
            enable_compile=runtime_config.compile,
            dit_device=runtime_config.dit_device,
            vae_device=runtime_config.vae_device,
            lora_paths=runtime_config.lora_paths,
            lora_scales=runtime_config.lora_scales,
        )
        opts = []
        if runtime_config.cpu_offload:
            opts.append("CPU offload")
        if runtime_config.flash_attn:
            opts.append("Flash Attn")
        if runtime_config.compile:
            opts.append("compiled")
        opts_str = f" + {', '.join(opts)}" if opts else ""
        mode = f"distributed (API encoder + local DiT{opts_str})"
    elif runtime_config.api_url:
        # API backend mode - encoder only (no model path)
        load_api_encoder(runtime_config.api_url, runtime_config.api_model, templates_dir)
        mode = f"API encoder-only ({runtime_config.api_model})"
    elif args.encoder_only:
        # Local encoder only
        if model_path is None:
            logger.error("No model path specified. Use --model-path or --config.")
            return 1
        load_encoder_only(model_path, templates_dir, encoder_device=runtime_config.encoder_device)
        mode = "encoder-only"
    else:
        # Full pipeline
        if model_path is None:
            logger.error("No model path specified. Use --model-path or --config.")
            return 1
        load_pipeline(
            model_path,
            templates_dir,
            encoder_device=runtime_config.encoder_device,
            dit_device=runtime_config.dit_device,
            vae_device=runtime_config.vae_device,
            lora_paths=runtime_config.lora_paths,
            lora_scales=runtime_config.lora_scales,
        )
        mode = "full"

    # Run server
    import uvicorn
    logger.info(f"Starting server at http://{host}:{port} ({mode} mode)")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
