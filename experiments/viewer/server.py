#!/usr/bin/env python3
"""
Standalone web viewer for experiment comparison.

Lightweight server that serves experiment images and generates comparisons
without loading heavy generation models.

Usage:
    uv run experiments/viewer/server.py
    uv run experiments/viewer/server.py --port 7861
    uv run experiments/viewer/server.py --host 0.0.0.0  # Allow external access
"""

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.compare.discovery import discover_experiments, get_experiment_by_name
from experiments.compare.models import ComparisonSpec
from experiments.compare.grid import generate_grid, generate_side_by_side
from experiments.compare.diff import compute_diff

app = FastAPI(
    title="Experiment Viewer",
    description="View and compare Z-Image experiment results",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
VIEWER_DIR = Path(__file__).parent
STATIC_DIR = VIEWER_DIR / "static"

# Mount static files if directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Cache for experiments (refreshed on demand)
_experiments_cache = None


def get_experiments(refresh: bool = False):
    """Get cached experiments or refresh from disk."""
    global _experiments_cache
    if _experiments_cache is None or refresh:
        _experiments_cache = discover_experiments()
    return _experiments_cache


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main viewer page."""
    html_path = VIEWER_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Viewer not found</h1><p>index.html missing</p>")
    return HTMLResponse(html_path.read_text())


@app.get("/api/experiments")
async def list_experiments(refresh: bool = False):
    """List all available experiments."""
    experiments = get_experiments(refresh)
    return {
        "experiments": [
            {
                "name": exp.name,
                "type": exp.experiment_type,
                "timestamp": exp.timestamp,
                "variable_name": exp.variable_name,
                "prompt_count": len(exp.prompt_ids),
                "prompts": exp.prompt_ids,
                "variable_values": [str(v) for v in exp.variable_values],
                "seeds": exp.seeds,
                "image_count": len(exp.images),
                "summary": exp.summary,
            }
            for exp in experiments
        ]
    }


@app.get("/api/experiments/{name}")
async def get_experiment(name: str):
    """Get detailed experiment data including all images."""
    exp = get_experiment_by_name(name)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {name}")

    return {
        "name": exp.name,
        "type": exp.experiment_type,
        "timestamp": exp.timestamp,
        "variable_name": exp.variable_name,
        "base_path": str(exp.base_path),
        "prompts": exp.prompt_ids,
        "variables": [str(v) for v in exp.variable_values],
        "seeds": exp.seeds,
        "summary": exp.summary,
        "images": [
            {
                "filename": img.path.name,
                "prompt_id": img.prompt_id,
                "variable_name": img.variable_name,
                "variable_value": str(img.variable_value),
                "seed": img.seed,
                "siglip_score": img.siglip_score,
                "image_reward": img.image_reward,
                "generation_time": img.generation_time,
            }
            for img in exp.images
        ],
    }


@app.get("/api/image/{experiment}/{filename}")
async def serve_image(experiment: str, filename: str):
    """Serve an experiment image by filename."""
    exp = get_experiment_by_name(experiment)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment}")

    # Find image in experiment
    for img in exp.images:
        if img.path.name == filename:
            if img.path.exists():
                return FileResponse(img.path, media_type="image/png")
            break

    # Try direct path in images folder
    image_path = exp.base_path / "images" / filename
    if image_path.exists():
        return FileResponse(image_path, media_type="image/png")

    raise HTTPException(status_code=404, detail=f"Image not found: {filename}")


class GridRequest(BaseModel):
    """Request body for grid generation."""
    experiment: str
    prompts: Optional[list[str]] = None
    seed: Optional[int] = None
    thumbnail_size: int = 256
    show_metrics: bool = True


@app.post("/api/grid")
async def generate_grid_endpoint(request: GridRequest):
    """Generate a grid image on-demand."""
    exp = get_experiment_by_name(request.experiment)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {request.experiment}")

    spec = ComparisonSpec(
        experiment=exp,
        prompts=request.prompts,
        seeds=[request.seed] if request.seed else None,
    )

    img = generate_grid(
        spec,
        thumbnail_size=(request.thumbnail_size, request.thumbnail_size),
        show_metrics=request.show_metrics,
    )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")


class DiffRequest(BaseModel):
    """Request body for diff generation."""
    experiment: str
    prompt_id: str
    value_a: str
    value_b: str
    seed: int = 42
    mode: str = "highlight"


@app.post("/api/diff")
async def generate_diff_endpoint(request: DiffRequest):
    """Generate a difference image between two variable values."""
    exp = get_experiment_by_name(request.experiment)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {request.experiment}")

    # Find images
    img_a = None
    img_b = None
    for img in exp.images:
        if img.prompt_id == request.prompt_id and img.seed == request.seed:
            if str(img.variable_value) == request.value_a:
                img_a = img
            elif str(img.variable_value) == request.value_b:
                img_b = img

    if not img_a:
        raise HTTPException(status_code=404, detail=f"Image not found for value: {request.value_a}")
    if not img_b:
        raise HTTPException(status_code=404, detail=f"Image not found for value: {request.value_b}")

    diff_img = compute_diff(img_a.path, img_b.path, mode=request.mode)

    buf = io.BytesIO()
    diff_img.save(buf, format="PNG")
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")


class SideBySideRequest(BaseModel):
    """Request body for side-by-side generation."""
    experiment: str
    prompt_id: str
    value_a: str
    value_b: str
    seed: int = 42


@app.post("/api/side-by-side")
async def generate_side_by_side_endpoint(request: SideBySideRequest):
    """Generate a side-by-side comparison image."""
    exp = get_experiment_by_name(request.experiment)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {request.experiment}")

    spec = ComparisonSpec(experiment=exp)

    try:
        img = generate_side_by_side(
            spec,
            value_a=request.value_a,
            value_b=request.value_b,
            prompt_id=request.prompt_id,
            seed=request.seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "experiments": len(get_experiments())}


def main():
    parser = argparse.ArgumentParser(description="Experiment comparison viewer")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7861, help="Port to listen on")
    args = parser.parse_args()

    import uvicorn

    print(f"Starting experiment viewer at http://{args.host}:{args.port}")
    print(f"Scanning experiments from: {Path(__file__).parent.parent / 'results'}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
