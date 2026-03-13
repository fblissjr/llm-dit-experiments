"""
Model Pool - Central registry for model lifecycle management.

Last Updated: 2026-01-12

Provides:
- Lazy loading of models on first use
- Automatic offloading to CPU when not in use
- Context manager interface for scoped model access
- VRAM budget management with LRU eviction
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelState:
    """Model lifecycle states."""

    UNLOADED = "unloaded"  # Not in memory
    CPU = "cpu"  # Offloaded to CPU RAM
    GPU = "gpu"  # Active on GPU
    LOADING = "loading"  # Currently loading


@dataclass
class ModelSpec:
    """
    Specification for a loadable model.

    Attributes:
        model_class: The model class to instantiate (e.g., LTX2Transformer)
        path: Path to model weights (file or directory)
        dtype: Data type when loaded (default bfloat16)
        device: Target device when active (default cuda)
        offload_device: Device for offloading (default cpu)
        quantization: Quantization mode (none, 4bit, 8bit, fp8)
        config: Model-specific configuration dict
        loader: Optional custom loader function
    """

    model_class: Optional[Type[nn.Module]] = None
    path: Optional[str] = None
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"
    offload_device: str = "cpu"
    quantization: str = "none"
    config: Dict[str, Any] = field(default_factory=dict)
    loader: Optional[Callable[["ModelSpec"], nn.Module]] = None

    def __post_init__(self):
        if self.path is not None:
            self.path = str(Path(self.path).expanduser())


class ModelHandle:
    """
    Handle to a model instance with lifecycle management.

    Provides context manager interface for automatic offloading:

        with model_pool.use("umt5-xxl") as encoder:
            embeddings = encoder.encode(prompt)
        # Model automatically offloaded after context exits
    """

    def __init__(self, pool: "ModelPool", name: str):
        self._pool = pool
        self._name = name
        self._model: Optional[nn.Module] = None

    def __enter__(self) -> nn.Module:
        self._model = self._pool._load_to_device(self._name)
        return self._model

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._pool.auto_offload:
            self._pool.offload(self._name)
        return False  # Don't suppress exceptions


class ModelPool:
    """
    Central registry for model instances with lazy loading and offloading.

    Features:
    - Register models by name with specs
    - Load on-demand with .use() context manager
    - Automatic offloading after use (configurable)
    - VRAM budget with LRU eviction

    Example:
        pool = ModelPool(vram_budget_gb=24.0)

        # Register models
        pool.register("gemma3", ModelSpec(
            model_class=Gemma3Encoder,
            path="models/LTX-2.3/text_encoder/",
            dtype=torch.bfloat16,
        ))

        # Use model with automatic lifecycle
        with pool.use("umt5-xxl") as encoder:
            embeddings = encoder.encode("A cat sleeping")
        # Model offloaded after context exits

        # Or manual control
        encoder = pool.get("umt5-xxl", load=True)
        embeddings = encoder.encode("A dog running")
        pool.offload("umt5-xxl")
    """

    def __init__(
        self,
        vram_budget_gb: float = 24.0,
        auto_offload: bool = True,
        offload_strategy: str = "lru",  # lru, fifo, priority
    ):
        """
        Initialize model pool.

        Args:
            vram_budget_gb: Maximum VRAM to use (triggers eviction)
            auto_offload: Whether to offload after context manager exit
            offload_strategy: How to choose which model to offload
        """
        self.vram_budget_gb = vram_budget_gb
        self.auto_offload = auto_offload
        self.offload_strategy = offload_strategy

        self._specs: Dict[str, ModelSpec] = {}
        self._instances: Dict[str, nn.Module] = {}
        self._states: Dict[str, str] = {}
        self._usage_order: List[str] = []  # For LRU tracking
        self._load_callbacks: List[Callable[[str, nn.Module], None]] = []
        self._offload_callbacks: List[Callable[[str], None]] = []

    def register(
        self,
        name: str,
        spec: Optional[ModelSpec] = None,
        *,
        model_class: Optional[Type[nn.Module]] = None,
        path: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        **kwargs,
    ) -> "ModelPool":
        """
        Register a model specification.

        Can pass a ModelSpec or individual parameters.

        Args:
            name: Unique name for this model
            spec: ModelSpec instance, or None to use kwargs
            model_class: Model class if not using spec
            path: Model path if not using spec
            dtype: Data type if not using spec
            device: Target device if not using spec
            **kwargs: Additional ModelSpec fields

        Returns:
            self for chaining
        """
        if spec is None:
            spec = ModelSpec(
                model_class=model_class,
                path=path,
                dtype=dtype,
                device=device,
                **kwargs,
            )

        self._specs[name] = spec
        self._states[name] = ModelState.UNLOADED
        logger.debug(f"Registered model: {name}")
        return self

    def register_instance(
        self,
        name: str,
        model: nn.Module,
        device: Optional[str] = None,
    ) -> "ModelPool":
        """
        Register an already-loaded model instance.

        Useful for sharing models between orchestrator and standalone pipelines.

        Args:
            name: Unique name for this model
            model: Already-loaded model instance
            device: Device the model is on (auto-detected if None)

        Returns:
            self for chaining
        """
        if device is None:
            try:
                device = str(next(model.parameters()).device)
            except StopIteration:
                device = "cpu"

        self._instances[name] = model
        self._states[name] = ModelState.GPU if "cuda" in device else ModelState.CPU
        self._specs[name] = ModelSpec(device=device)

        # Track in usage order
        if name not in self._usage_order:
            self._usage_order.append(name)

        logger.debug(f"Registered instance: {name} on {device}")
        return self

    def use(self, name: str) -> ModelHandle:
        """
        Get context manager for using a model.

        Model is loaded to GPU on enter, optionally offloaded on exit.

        Args:
            name: Model name to use

        Returns:
            ModelHandle context manager
        """
        if name not in self._specs and name not in self._instances:
            raise KeyError(f"Model not registered: {name}")
        return ModelHandle(self, name)

    def get(
        self,
        name: str,
        load: bool = False,
    ) -> Optional[nn.Module]:
        """
        Get model instance if loaded.

        Args:
            name: Model name
            load: If True, load to GPU if not already

        Returns:
            Model instance or None if not loaded
        """
        if load:
            return self._load_to_device(name)
        return self._instances.get(name)

    def is_loaded(self, name: str) -> bool:
        """Check if model is loaded (on CPU or GPU)."""
        return name in self._instances

    def is_on_device(self, name: str) -> bool:
        """Check if model is on GPU."""
        return self._states.get(name) == ModelState.GPU

    def offload(self, name: str) -> None:
        """
        Offload model to CPU.

        Args:
            name: Model name to offload
        """
        if name not in self._instances:
            return

        if self._states[name] == ModelState.GPU:
            spec = self._specs.get(name, ModelSpec())
            self._instances[name].to(spec.offload_device)
            self._states[name] = ModelState.CPU
            torch.cuda.empty_cache()

            for callback in self._offload_callbacks:
                callback(name)

            logger.debug(f"Offloaded {name} to {spec.offload_device}")

    def unload(self, name: str) -> None:
        """
        Completely unload model from memory.

        Args:
            name: Model name to unload
        """
        if name in self._instances:
            del self._instances[name]
            self._states[name] = ModelState.UNLOADED
            if name in self._usage_order:
                self._usage_order.remove(name)
            torch.cuda.empty_cache()
            logger.debug(f"Unloaded {name}")

    def offload_all(self) -> None:
        """Offload all models to CPU."""
        for name in list(self._instances.keys()):
            self.offload(name)

    def unload_all(self) -> None:
        """Unload all models from memory."""
        for name in list(self._instances.keys()):
            self.unload(name)

    def _load_to_device(self, name: str) -> nn.Module:
        """
        Load model to GPU, offloading others if needed.

        Args:
            name: Model name to load

        Returns:
            Loaded model instance
        """
        spec = self._specs.get(name)

        # Already loaded and on GPU
        if name in self._instances and self._states[name] == ModelState.GPU:
            self._touch_usage(name)
            return self._instances[name]

        # Loaded but on CPU - move to GPU
        if name in self._instances and self._states[name] == ModelState.CPU:
            self._ensure_vram_available(spec)
            device = spec.device if spec else "cuda"
            self._instances[name].to(device)
            self._states[name] = ModelState.GPU
            self._touch_usage(name)
            logger.debug(f"Moved {name} to {device}")
            return self._instances[name]

        # Not loaded - load fresh
        if spec is None:
            raise KeyError(f"No spec registered for: {name}")

        self._ensure_vram_available(spec)
        self._states[name] = ModelState.LOADING

        instance = self._load_model(spec)
        self._instances[name] = instance
        self._states[name] = ModelState.GPU
        self._touch_usage(name)

        for callback in self._load_callbacks:
            callback(name, instance)

        logger.info(f"Loaded {name} to {spec.device}")
        return instance

    def _load_model(self, spec: ModelSpec) -> nn.Module:
        """
        Load model from disk.

        Args:
            spec: Model specification

        Returns:
            Loaded model instance
        """
        # Custom loader takes precedence
        if spec.loader is not None:
            return spec.loader(spec)

        if spec.model_class is None:
            raise ValueError("ModelSpec must have model_class or loader")

        # Instantiate model
        model = spec.model_class(**spec.config)

        # Load weights if path provided
        if spec.path:
            path = Path(spec.path)

            # Try to find weights file
            if path.is_dir():
                # Look for common weight files
                candidates = [
                    path / "model.safetensors",
                    path / "pytorch_model.safetensors",
                    path / "diffusion_pytorch_model.safetensors",
                ]
                for candidate in candidates:
                    if candidate.exists():
                        path = candidate
                        break
                else:
                    # Check for sharded weights
                    shards = list(path.glob("*.safetensors"))
                    if shards:
                        # Load sharded model
                        self._load_sharded_weights(model, shards)
                        return model.to(device=spec.device, dtype=spec.dtype)

            if path.exists() and path.is_file():
                from safetensors.torch import load_file as load_safetensors

                state_dict = load_safetensors(str(path))
                model.load_state_dict(state_dict)

        return model.to(device=spec.device, dtype=spec.dtype)

    def _load_sharded_weights(
        self,
        model: nn.Module,
        shards: List[Path],
    ) -> None:
        """Load sharded safetensors weights."""
        from safetensors.torch import load_file as load_safetensors

        for shard in sorted(shards):
            state_dict = load_safetensors(str(shard))
            model.load_state_dict(state_dict, strict=False)
            logger.debug(f"Loaded shard: {shard.name}")

    def _ensure_vram_available(self, spec: Optional[ModelSpec]) -> None:
        """
        Ensure enough VRAM is available, offloading if needed.

        Uses LRU strategy to choose which models to offload.
        """
        if not torch.cuda.is_available():
            return

        # Check current usage
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)

        # Offload LRU models until under budget
        while allocated_gb > self.vram_budget_gb * 0.9:  # 90% threshold
            if not self._usage_order:
                break

            # Find oldest GPU model
            for name in self._usage_order:
                if self._states.get(name) == ModelState.GPU:
                    self.offload(name)
                    allocated_gb = torch.cuda.memory_allocated() / (1024**3)
                    break
            else:
                break  # No GPU models to offload

    def _touch_usage(self, name: str) -> None:
        """Update usage order for LRU tracking."""
        if name in self._usage_order:
            self._usage_order.remove(name)
        self._usage_order.append(name)

    def on_load(self, callback: Callable[[str, nn.Module], None]) -> None:
        """Register callback for model load events."""
        self._load_callbacks.append(callback)

    def on_offload(self, callback: Callable[[str], None]) -> None:
        """Register callback for model offload events."""
        self._offload_callbacks.append(callback)

    def status(self) -> Dict[str, str]:
        """Get current status of all registered models."""
        return {name: self._states.get(name, ModelState.UNLOADED) for name in self._specs}

    def __repr__(self) -> str:
        loaded = sum(1 for s in self._states.values() if s != ModelState.UNLOADED)
        gpu = sum(1 for s in self._states.values() if s == ModelState.GPU)
        return f"ModelPool({len(self._specs)} registered, {loaded} loaded, {gpu} on GPU)"
