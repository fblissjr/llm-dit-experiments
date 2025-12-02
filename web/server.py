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


class GenerateRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    steps: int = 9
    seed: Optional[int] = None
    template: Optional[str] = None
    enable_thinking: bool = True
    guidance_scale: float = 0.0


class EncodeRequest(BaseModel):
    prompt: str
    template: Optional[str] = None
    enable_thinking: bool = True


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
            enable_thinking=request.enable_thinking,
        )
        encode_time = time.time() - start

        embeddings = output.embeddings[0]

        return {
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
            "encode_time": encode_time,
            "prompt": request.prompt,
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
        logger.info(f"Generating: {request.prompt[:50]}...")

        # Set up generator for reproducibility
        generator = None
        if request.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(request.seed)

        start = time.time()

        # Generate image
        image = pipeline(
            request.prompt,
            height=request.height,
            width=request.width,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
            template=request.template,
            enable_thinking=request.enable_thinking,
        )

        gen_time = time.time() - start
        logger.info(f"Generated in {gen_time:.1f}s")

        # Convert to PNG bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return StreamingResponse(
            img_bytes,
            media_type="image/png",
            headers={
                "X-Generation-Time": str(gen_time),
                "X-Seed": str(request.seed) if request.seed else "random",
            },
        )

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/templates")
async def list_templates():
    """List available templates."""
    if pipeline is None or pipeline.encoder.templates is None:
        return {"templates": []}

    templates = []
    for name, tpl in pipeline.encoder.templates.items():
        templates.append({
            "name": name,
            "has_thinking": bool(tpl.thinking_content),
        })

    return {"templates": templates}


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
            enable_thinking=request.enable_thinking,
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
            enable_thinking=request.enable_thinking,
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


def load_pipeline(model_path: str, templates_dir: Optional[str] = None):
    """Load the full generation pipeline."""
    global pipeline

    from llm_dit.pipelines import ZImagePipeline

    logger.info(f"Loading pipeline from {model_path}...")
    start = time.time()

    pipeline = ZImagePipeline.from_pretrained(
        model_path,
        templates_dir=templates_dir,
        torch_dtype=torch.bfloat16,
    )

    load_time = time.time() - start
    logger.info(f"Pipeline loaded in {load_time:.1f}s")
    logger.info(f"Device: {pipeline.device}")


def load_encoder_only(model_path: str, templates_dir: Optional[str] = None):
    """Load only the encoder (fast mode for testing on Mac)."""
    global encoder, encoder_only_mode

    from llm_dit.encoders import ZImageTextEncoder

    logger.info(f"Loading encoder only from {model_path}...")
    start = time.time()

    encoder = ZImageTextEncoder.from_pretrained(
        model_path,
        templates_dir=templates_dir,
        torch_dtype=torch.bfloat16,
    )

    encoder_only_mode = True
    load_time = time.time() - start
    logger.info(f"Encoder loaded in {load_time:.1f}s (encoder-only mode)")
    logger.info(f"Device: {encoder.device}")


def main():
    parser = argparse.ArgumentParser(description="Z-Image web server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--config", type=str, help="Path to config.toml")
    parser.add_argument("--profile", type=str, default="default", help="Config profile")
    parser.add_argument("--model-path", type=str, help="Path to Z-Image model")
    parser.add_argument("--templates-dir", type=str, help="Path to templates")
    parser.add_argument(
        "--encoder-only",
        action="store_true",
        help="Load only encoder (fast mode for Mac, no image generation)",
    )
    args = parser.parse_args()

    # Load config or use CLI args
    model_path = args.model_path
    templates_dir = args.templates_dir

    if args.config:
        from llm_dit.config import Config
        cfg = Config.from_toml(args.config, args.profile)
        if model_path is None:
            model_path = cfg.model_path
        if templates_dir is None:
            templates_dir = cfg.templates_dir

    if model_path is None:
        logger.error("No model path specified. Use --model-path or --config.")
        return 1

    # Load pipeline or encoder only
    if args.encoder_only:
        load_encoder_only(model_path, templates_dir)
    else:
        load_pipeline(model_path, templates_dir)

    # Run server
    import uvicorn
    mode = "encoder-only" if args.encoder_only else "full"
    logger.info(f"Starting server at http://{args.host}:{args.port} ({mode} mode)")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
