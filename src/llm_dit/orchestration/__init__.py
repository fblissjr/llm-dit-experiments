"""
Orchestration - Multi-pipeline composition and execution.

Last Updated: 2026-01-12

Core components:
- ModelPool: Central model registry with lazy loading
- PipelineStep: Base class for composable steps
- Orchestrator: Sequential execution engine

Usage:
    from llm_dit.orchestration import ModelPool, ModelSpec, Orchestrator
    from llm_dit.orchestration.steps import PipelineStep, StepInput, StepOutput

    # Setup model pool
    pool = ModelPool(vram_budget_gb=24.0)
    pool.register("encoder", ModelSpec(model_class=MyEncoder, path="..."))

    # Create orchestrator
    orchestrator = Orchestrator(pool)
    orchestrator.add_step(MyStep())

    # Run
    result = orchestrator.run({"prompt": "Hello"})
"""

from .model_pool import ModelPool, ModelSpec, ModelHandle, ModelState
from .orchestrator import Orchestrator, ExecutionContext, compose
from .steps import (
    PipelineStep,
    StepInput,
    StepOutput,
    StepConfig,
    FunctionStep,
    ConditionalStep,
    LoopStep,
)
from .outputs import (
    TextEmbeddings,
    ImageOutput,
    VideoOutput,
    TranscriptionOutput,
    ScenePrompt,
    ScenePromptsOutput,
    AudioFeatures,
)

__all__ = [
    # Model pool
    "ModelPool",
    "ModelSpec",
    "ModelHandle",
    "ModelState",
    # Orchestrator
    "Orchestrator",
    "ExecutionContext",
    "compose",
    # Steps
    "PipelineStep",
    "StepInput",
    "StepOutput",
    "StepConfig",
    "FunctionStep",
    "ConditionalStep",
    "LoopStep",
    # Outputs
    "TextEmbeddings",
    "ImageOutput",
    "VideoOutput",
    "TranscriptionOutput",
    "ScenePrompt",
    "ScenePromptsOutput",
    "AudioFeatures",
]
