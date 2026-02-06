"""FastAPI dependency injection for shared server state.

Provides typed dependencies that routers use to access RuntimeConfig
and ModelManager without globals. Uses FastAPI's Depends() pattern.

Usage in routers:
    from web.dependencies import ConfigDep, ManagerDep

    @router.post("/generate")
    async def generate(config: ConfigDep, manager: ManagerDep):
        pipeline = manager.get_pipeline("flux2")
        model_path = config.flux2.model_path
"""

from typing import Annotated

from fastapi import Depends, Request

from llm_dit.config import RuntimeConfig
from llm_dit.model_manager import ModelManager


def get_runtime_config(request: Request) -> RuntimeConfig:
    return request.app.state.runtime_config


def get_model_manager(request: Request) -> ModelManager:
    return request.app.state.model_manager


ConfigDep = Annotated[RuntimeConfig, Depends(get_runtime_config)]
ManagerDep = Annotated[ModelManager, Depends(get_model_manager)]
