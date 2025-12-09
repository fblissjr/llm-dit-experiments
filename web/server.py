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
import re
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
rewriter_backend = None  # API backend for rewriting (if configured)
runtime_config = None  # RuntimeConfig from CLI/TOML
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
    strip_quotes: bool = False  # If True, remove " characters (for JSON-type prompts)
    width: int = 1024
    height: int = 1024
    steps: int = 9
    seed: Optional[int] = None
    template: Optional[str] = None
    guidance_scale: float = 0.0
    shift: float = 3.0  # Scheduler shift parameter
    long_prompt_mode: str = "interpolate"  # truncate/interpolate/pool/attention_pool
    hidden_layer: int = -2  # Which hidden layer to extract (-1 to -6)


class EncodeRequest(BaseModel):
    prompt: str  # User prompt
    system_prompt: Optional[str] = None  # System prompt (optional)
    thinking_content: Optional[str] = None  # Content inside <think>...</think> (triggers think block)
    assistant_content: Optional[str] = None  # Content after </think> (optional)
    force_think_block: bool = False  # If True, add empty think block even without content
    strip_quotes: bool = False  # If True, remove " characters (for JSON-type prompts)
    template: Optional[str] = None


class RewriteRequest(BaseModel):
    prompt: str  # User prompt to rewrite/expand
    rewriter: Optional[str] = None  # Name of rewriter template (optional if custom_system_prompt provided)
    custom_system_prompt: Optional[str] = None  # Ad-hoc system prompt for rewriting
    max_tokens: Optional[int] = None  # Maximum tokens to generate (default from config: 512)
    temperature: Optional[float] = None  # Sampling temperature (default: 0.6 for Qwen3 thinking)
    top_p: Optional[float] = None  # Nucleus sampling (default: 0.95)
    top_k: Optional[int] = None  # Top-k sampling (default: 20 for Qwen3)
    min_p: Optional[float] = None  # Minimum probability (default: 0.0)
    presence_penalty: Optional[float] = None  # Presence penalty (0-2, default: 0.0)


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


@app.get("/api/rewriter-config")
async def get_rewriter_config():
    """Get rewriter configuration defaults from server config.

    Qwen3 Best Practices (thinking mode):
    - temperature=0.6, top_p=0.95, top_k=20, min_p=0
    - DO NOT use greedy decoding (causes repetition)
    - presence_penalty=0-2 helps reduce endless repetitions
    """
    if runtime_config is None:
        # Return hardcoded defaults matching Qwen3 thinking mode
        return {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "max_tokens": 512,
            "use_api": False,
        }
    return {
        "temperature": runtime_config.rewriter_temperature,
        "top_p": runtime_config.rewriter_top_p,
        "top_k": runtime_config.rewriter_top_k,
        "min_p": runtime_config.rewriter_min_p,
        "presence_penalty": runtime_config.rewriter_presence_penalty,
        "max_tokens": runtime_config.rewriter_max_tokens,
        "use_api": runtime_config.rewriter_use_api,
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
            remove_quotes=request.strip_quotes,
        )
        encode_time = time.time() - start

        embeddings = output.embeddings[0]
        token_count = output.token_counts[0] if output.token_counts else embeddings.shape[0]

        # Get formatted prompt if available
        formatted_prompt = None
        if output.formatted_prompts:
            formatted_prompt = output.formatted_prompts[0]
            logger.info(f"Formatted prompt ({len(formatted_prompt)} chars, {token_count} tokens):")
            logger.info(f"---BEGIN FORMATTED PROMPT---")
            logger.info(formatted_prompt)
            logger.info(f"---END FORMATTED PROMPT---")

        return {
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
            "encode_time": encode_time,
            "token_count": token_count,
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
        logger.info(f"  Long prompt mode: {request.long_prompt_mode}")
        logger.info(f"  Hidden layer: {request.hidden_layer}")
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
        logger.info(f"Calling pipeline() with long_prompt_mode={request.long_prompt_mode}, hidden_layer={request.hidden_layer}...")
        image = pipeline(
            request.prompt,
            height=request.height,
            width=request.width,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            shift=request.shift,
            generator=generator,
            template=request.template,
            system_prompt=request.system_prompt,
            thinking_content=request.thinking_content,
            assistant_content=request.assistant_content,
            force_think_block=request.force_think_block,
            remove_quotes=request.strip_quotes,
            long_prompt_mode=request.long_prompt_mode,
            hidden_layer=request.hidden_layer,
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
                    remove_quotes=request.strip_quotes,
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
            "strip_quotes": request.strip_quotes,
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "seed": request.seed,
            "template": request.template,
            "guidance_scale": request.guidance_scale,
            "shift": request.shift,
            "long_prompt_mode": request.long_prompt_mode,
            "hidden_layer": request.hidden_layer,
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
            remove_quotes=request.strip_quotes,
        )
        formatted = enc.formatter.format(conv)

        # Get token count if tokenizer is available
        token_count = None
        if hasattr(enc, 'backend') and hasattr(enc.backend, 'tokenizer'):
            tokens = enc.backend.tokenizer.encode(formatted, add_special_tokens=False)
            token_count = len(tokens)

        return {
            "formatted_prompt": formatted,
            "char_count": len(formatted),
            "token_count": token_count,
            "max_tokens": 1504,
            "prompt": request.prompt,
            "system_prompt": request.system_prompt,
            "thinking_content": request.thinking_content,
            "assistant_content": request.assistant_content,
            "template": request.template,
            "force_think_block": request.force_think_block,
            "strip_quotes": request.strip_quotes,
        }
    except Exception as e:
        logger.error(f"Format failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/templates")
async def list_templates():
    """List available templates with full data for UI population."""
    # Use encoder from pipeline or standalone encoder
    enc = encoder if encoder is not None else (pipeline.encoder if pipeline else None)
    if enc is None or enc.templates is None:
        return {"templates": []}

    templates = []
    for name in enc.templates:
        tpl = enc.templates.get(name)
        if tpl and tpl.category != "rewriter":  # Exclude rewriter templates
            templates.append({
                "name": name,
                "description": tpl.description or "",
                "category": tpl.category or "general",
                "system_prompt": tpl.content or "",
                "thinking_content": tpl.thinking_content or "",
                "assistant_content": tpl.assistant_content or "",
                "add_think_block": tpl.add_think_block,
            })

    # Sort by category then name
    templates.sort(key=lambda x: (x["category"], x["name"]))
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
    Rewrite/expand a prompt using a rewriter template or custom system prompt.

    Uses the same Qwen3 model loaded for text encoding to generate expanded prompts,
    or a separate API backend if configured.

    Supports two modes:
    1. Template mode: Use `rewriter` to specify a rewriter template
    2. Ad-hoc mode: Use `custom_system_prompt` for custom rewriting instructions

    Backend selection:
    - If rewriter_use_api is True and rewriter_backend is configured, uses API backend
    - Otherwise, uses the local encoder's backend
    """
    # Determine which backend to use for generation
    # Priority: rewriter_backend (if API mode), encoder's backend, pipeline's encoder backend
    backend = None
    backend_name = "local"

    if rewriter_backend is not None:
        backend = rewriter_backend
        backend_name = "api"
        logger.info("[Rewrite] Using API backend for rewriting")
    else:
        # Use encoder from pipeline or standalone encoder
        enc = encoder if encoder is not None else (pipeline.encoder if pipeline else None)
        if enc is not None:
            backend = getattr(enc, 'backend', None)
            backend_name = "local"

    if backend is None:
        raise HTTPException(status_code=503, detail="No backend available for generation")

    if not getattr(backend, 'supports_generation', False):
        raise HTTPException(
            status_code=400,
            detail="Backend does not support text generation"
        )

    # Get template loader from encoder (for template lookup)
    enc = encoder if encoder is not None else (pipeline.encoder if pipeline else None)

    # Determine system prompt: custom takes precedence, then template
    system_prompt = None
    rewriter_name = "custom"

    if request.custom_system_prompt:
        # Ad-hoc mode: use custom system prompt directly
        system_prompt = request.custom_system_prompt.strip()
        rewriter_name = "custom"
        logger.info(f"[Rewrite] Using custom system prompt ({len(system_prompt)} chars)")
    elif request.rewriter:
        # Template mode: get system prompt from template
        if enc is None or enc.templates is None:
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

        system_prompt = rewriter_template.content
        rewriter_name = request.rewriter
    else:
        raise HTTPException(
            status_code=400,
            detail="Either 'rewriter' or 'custom_system_prompt' must be provided"
        )

    # Get generation parameters from request or config defaults
    # Qwen3 Best Practices (thinking mode): temperature=0.6, top_p=0.95, top_k=20
    max_tokens = request.max_tokens
    temperature = request.temperature
    top_p = request.top_p
    top_k = request.top_k
    min_p = request.min_p
    presence_penalty = request.presence_penalty

    if runtime_config is not None:
        if max_tokens is None:
            max_tokens = runtime_config.rewriter_max_tokens
        if temperature is None:
            temperature = runtime_config.rewriter_temperature
        if top_p is None:
            top_p = runtime_config.rewriter_top_p
        if top_k is None:
            top_k = runtime_config.rewriter_top_k
        if min_p is None:
            min_p = runtime_config.rewriter_min_p
        if presence_penalty is None:
            presence_penalty = runtime_config.rewriter_presence_penalty
    else:
        # Fallback defaults (Qwen3 thinking mode)
        if max_tokens is None:
            max_tokens = 512
        if temperature is None:
            temperature = 0.6
        if top_p is None:
            top_p = 0.95
        if top_k is None:
            top_k = 20
        if min_p is None:
            min_p = 0.0
        if presence_penalty is None:
            presence_penalty = 0.0

    try:
        start = time.time()
        logger.info(f"[Rewrite] Using: {rewriter_name} (backend: {backend_name})")
        logger.info(f"[Rewrite] Input prompt: {request.prompt[:100]}...")
        logger.info(f"[Rewrite] Params: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}, top_k={top_k}, min_p={min_p}, presence_penalty={presence_penalty}")

        # Generate using the backend
        generated = backend.generate(
            prompt=request.prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
        )

        gen_time = time.time() - start
        logger.info(f"[Rewrite] Generated {len(generated)} chars in {gen_time:.2f}s")

        # Parse the generated output to separate thinking from the prompt
        # The model may output in several formats:
        # 1. <think>...</think> followed by the prompt
        # 2. Plain reasoning text followed by JSON/structured output
        # 3. Just the rewritten prompt
        thinking_content = None
        rewritten_prompt = generated

        # Try to find <think>...</think> tags first
        think_match = re.search(r'<think>\s*(.*?)\s*</think>', generated, re.DOTALL)
        if think_match:
            thinking_content = think_match.group(1).strip()
            # Remove the think block from the rewritten prompt
            rewritten_prompt = re.sub(r'<think>.*?</think>\s*', '', generated, flags=re.DOTALL).strip()
            logger.info(f"[Rewrite] Extracted thinking via <think> tags ({len(thinking_content)} chars), prompt ({len(rewritten_prompt)} chars)")
        else:
            # No think tags - try to find JSON at the end and treat preceding text as thinking
            # Look for a JSON object (starts with { and ends with })
            json_match = re.search(r'(\{[\s\S]*\})\s*$', generated)
            if json_match:
                json_text = json_match.group(1)
                # Everything before the JSON is reasoning/thinking
                pre_json = generated[:json_match.start()].strip()
                if pre_json:
                    thinking_content = pre_json
                    rewritten_prompt = json_text
                    logger.info(f"[Rewrite] Extracted thinking via JSON detection ({len(thinking_content)} chars), JSON prompt ({len(rewritten_prompt)} chars)")
            # If output starts with reasoning patterns like "Okay," "Let me", etc. and has a clear break
            elif re.match(r'^(Okay|Let me|I need|First|The user|Looking)', generated):
                # Look for double newline as separator between thinking and output
                parts = re.split(r'\n\n+', generated, maxsplit=1)
                if len(parts) == 2 and len(parts[1]) > 50:
                    # If second part is substantial, treat first as thinking
                    # But only if second part looks like a prompt (not more reasoning)
                    if not re.match(r'^(Okay|Let me|I need|First|The user|Looking|Now)', parts[1]):
                        thinking_content = parts[0].strip()
                        rewritten_prompt = parts[1].strip()
                        logger.info(f"[Rewrite] Extracted thinking via paragraph split ({len(thinking_content)} chars), prompt ({len(rewritten_prompt)} chars)")

        # Defense in depth: strip any remaining <think>/<think> tags from both outputs
        # This handles edge cases where tags might be nested or malformed
        if thinking_content:
            thinking_content = re.sub(r'</?think>', '', thinking_content).strip()
        if rewritten_prompt:
            rewritten_prompt = re.sub(r'</?think>', '', rewritten_prompt).strip()

        # Clear CUDA cache to prevent memory issues when switching back to encoding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("[Rewrite] Cleared CUDA cache after generation")

        return {
            "original_prompt": request.prompt,
            "rewritten_prompt": rewritten_prompt,
            "thinking_content": thinking_content,
            "rewriter": request.rewriter,
            "backend": backend_name,
            "gen_time": gen_time,
        }

    except Exception as e:
        logger.error(f"Rewrite failed: {e}")
        # Clear CUDA cache even on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
    text_encoder_path: Optional[str] = None,
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
    if text_encoder_path:
        logger.info(f"  Text encoder: {text_encoder_path}")
    logger.info(f"  Encoder device: {encoder_device}")
    logger.info(f"  DiT device: {dit_device}")
    logger.info(f"  VAE device: {vae_device}")
    start = time.time()

    pipeline = ZImagePipeline.from_pretrained(
        model_path,
        text_encoder_path=text_encoder_path,
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
    from llm_dit.startup import PipelineLoader

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
    parser.add_argument(
        "--use-api-encoder",
        action="store_true",
        help="Use API backend for encoding (default: local encoder)",
    )

    args = parser.parse_args()

    # Load unified config (handles TOML + CLI overrides)
    global runtime_config, pipeline, encoder, rewriter_backend, encoder_only_mode
    runtime_config = load_runtime_config(args)
    setup_logging(runtime_config)

    # Validate model path (unless using API-only mode)
    if not runtime_config.model_path and not runtime_config.api_url:
        logger.error("No model path specified. Use --model-path or --config.")
        return 1

    # Use PipelineLoader for unified loading
    loader = PipelineLoader(runtime_config)
    use_api = getattr(args, "use_api_encoder", False)

    # Load using PipelineLoader.auto_load()
    result = loader.auto_load(
        encoder_only=args.encoder_only,
        use_api=use_api,
    )

    # Set global state from load result
    pipeline = result.pipeline
    encoder = result.encoder
    encoder_only_mode = result.mode in ("encoder_only", "api_encoder")
    mode = result.mode

    # Initialize rewriter API backend if configured
    if runtime_config.rewriter_use_api:
        # Determine API URL: rewriter-specific or fall back to main API URL
        rewriter_url = runtime_config.rewriter_api_url or runtime_config.api_url
        if rewriter_url:
            from llm_dit.backends.api import APIBackend, APIBackendConfig

            rewriter_api_config = APIBackendConfig(
                base_url=rewriter_url,
                model_id=runtime_config.rewriter_api_model,
            )
            rewriter_backend = APIBackend(rewriter_api_config)
            logger.info(f"[Rewriter] API backend configured: {rewriter_url} (model: {runtime_config.rewriter_api_model})")
            logger.info(f"[Rewriter] Defaults: temperature={runtime_config.rewriter_temperature}, top_p={runtime_config.rewriter_top_p}, max_tokens={runtime_config.rewriter_max_tokens}")
        else:
            logger.warning("[Rewriter] use_api=True but no API URL configured. Using local model.")

    # Run server
    import uvicorn
    host = runtime_config.host
    port = runtime_config.port
    logger.info(f"Starting server at http://{host}:{port} ({mode} mode)")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
