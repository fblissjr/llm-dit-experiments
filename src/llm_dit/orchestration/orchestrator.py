"""
Orchestrator - Pipeline composition and execution engine.

Last Updated: 2026-01-12

The orchestrator:
- Executes steps in sequence with shared context
- Manages model loading/offloading between steps
- Supports progress callbacks for UI integration
- Handles input/output mapping between steps
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union

from .model_pool import ModelPool
from .steps import PipelineStep, StepConfig

logger = logging.getLogger(__name__)


class ExecutionContext:
    """
    Shared context during orchestration execution.

    Stores all inputs and intermediate outputs.
    Steps read from and write to this context.
    """

    def __init__(self, initial: Optional[Dict[str, Any]] = None):
        self._data: Dict[str, Any] = dict(initial or {})
        self._history: List[Dict[str, Any]] = []  # Step outputs for debugging

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from context."""
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value in context."""
        self._data[key] = value

    def update(self, data: Dict[str, Any]) -> None:
        """Update context with multiple values."""
        self._data.update(data)

    def snapshot(self) -> Dict[str, Any]:
        """Get copy of current context."""
        return dict(self._data)

    def record_step(self, step_name: str, outputs: Dict[str, Any]) -> None:
        """Record step outputs for debugging."""
        self._history.append({
            "step": step_name,
            "outputs": outputs,
            "timestamp": time.time(),
        })

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self._history

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value


class Orchestrator:
    """
    Orchestrates multi-step pipeline execution with shared model pool.

    Features:
    - Sequential step execution with shared context
    - Automatic model loading/offloading between steps
    - Input/output mapping between steps
    - Progress callbacks for UI integration
    - Error handling with configurable strategies

    Example - Simple chain:
        orchestrator = Orchestrator(model_pool=pool)
        orchestrator.add_step(TextEncoderStep())
        orchestrator.add_step(GenerateStep())
        orchestrator.add_step(DecodeStep())

        result = orchestrator.run({"prompt": "A cat sleeping"})
        image = result["image"]

    Example - With mappings:
        orchestrator = Orchestrator(pool)
        orchestrator.add_step(
            ZImageAdapter(),
            output_mapping={"image": "reference_frame"},  # Rename output
        )
        orchestrator.add_step(
            LTX2VideoAdapter(),
            input_mapping={"reference_image": "reference_frame"},  # Wire to previous
        )

    Example - Builder pattern:
        result = (
            Orchestrator(pool)
            .add_step(EncodeStep())
            .add_step(GenerateStep())
            .add_step(DecodeStep())
            .run({"prompt": "A dog running"})
        )
    """

    def __init__(
        self,
        model_pool: Optional[ModelPool] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ):
        """
        Initialize orchestrator.

        Args:
            model_pool: Shared model pool for all steps
            progress_callback: Called with (step_index, total_steps, step_name)
        """
        self.model_pool = model_pool or ModelPool()
        self.progress_callback = progress_callback
        self._steps: List[StepConfig] = []

    def add_step(
        self,
        step: PipelineStep,
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        enabled: bool = True,
        on_error: str = "raise",
    ) -> "Orchestrator":
        """
        Add a step to the orchestration.

        Args:
            step: PipelineStep instance
            input_mapping: Map step input names to context keys
                           e.g., {"prompt": "user_prompt"} means step's "prompt"
                           input reads from context["user_prompt"]
            output_mapping: Map step output names to context keys
                            e.g., {"image": "generated_image"} means step's "image"
                            output is stored as context["generated_image"]
            enabled: Whether step is enabled (disabled steps are skipped)
            on_error: Error handling: "raise" (default), "skip", "default"

        Returns:
            self for chaining
        """
        self._steps.append(StepConfig(
            step=step,
            input_mapping=input_mapping or {},
            output_mapping=output_mapping or {},
            enabled=enabled,
            on_error=on_error,
        ))
        return self

    def run(
        self,
        initial_inputs: Optional[Dict[str, Any]] = None,
        context: Optional[ExecutionContext] = None,
    ) -> Dict[str, Any]:
        """
        Execute all steps in sequence.

        Args:
            initial_inputs: Initial context values
            context: Existing context to continue (for resumption)

        Returns:
            Final context as dict
        """
        if context is None:
            context = ExecutionContext(initial_inputs)
        elif initial_inputs:
            context.update(initial_inputs)

        total_steps = len([s for s in self._steps if s.enabled])
        executed = 0

        for i, step_config in enumerate(self._steps):
            if not step_config.enabled:
                logger.debug(f"Skipping disabled step: {step_config.step.name}")
                continue

            step = step_config.step

            # Report progress
            if self.progress_callback:
                self.progress_callback(executed, total_steps, step.name)

            try:
                # Gather inputs from context
                step_inputs = self._gather_inputs(step, step_config.input_mapping, context)

                # Validate inputs
                step_inputs = step.validate_inputs(step_inputs)

                # Load required models
                models = self._load_models(step.required_models)

                # Execute step
                logger.info(f"Executing step {executed + 1}/{total_steps}: {step.name}")
                start_time = time.time()
                outputs = step.execute(step_inputs, models)
                elapsed = time.time() - start_time
                logger.debug(f"Step {step.name} completed in {elapsed:.2f}s")

                # Validate outputs
                step.validate_outputs(outputs)

                # Store outputs in context
                for output_name, value in outputs.items():
                    context_key = step_config.output_mapping.get(output_name, output_name)
                    context.set(context_key, value)

                # Record for debugging
                context.record_step(step.name, outputs)

            except Exception as e:
                logger.error(f"Step {step.name} failed: {e}")

                if step_config.on_error == "raise":
                    raise
                elif step_config.on_error == "skip":
                    logger.warning(f"Skipping failed step: {step.name}")
                elif step_config.on_error == "default":
                    # Use default outputs
                    for out in step.outputs:
                        context.set(out.name, None)
                else:
                    raise

            finally:
                # Offload models after step (if auto_offload enabled)
                if self.model_pool.auto_offload:
                    for model_name in step.required_models:
                        self.model_pool.offload(model_name)

            executed += 1

        # Final progress callback
        if self.progress_callback:
            self.progress_callback(total_steps, total_steps, "complete")

        return context.snapshot()

    def _gather_inputs(
        self,
        step: PipelineStep,
        mapping: Dict[str, str],
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """
        Gather step inputs from context using mapping.

        Args:
            step: The step to gather inputs for
            mapping: Input name -> context key mapping
            context: Execution context

        Returns:
            Dict of input_name -> value
        """
        inputs = {}

        for inp in step.inputs:
            # Use mapping if provided, otherwise use input name directly
            context_key = mapping.get(inp.name, inp.name)

            if context_key in context:
                inputs[inp.name] = context.get(context_key)
            elif inp.default is not None:
                inputs[inp.name] = inp.default
            # If required and not found, validate_inputs will raise

        return inputs

    def _load_models(self, model_names: List[str]) -> Dict[str, Any]:
        """
        Load required models from pool.

        Args:
            model_names: List of model names to load

        Returns:
            Dict of model_name -> model_instance
        """
        models = {}
        for name in model_names:
            models[name] = self.model_pool.get(name, load=True)
        return models

    def clear(self) -> "Orchestrator":
        """Clear all steps."""
        self._steps.clear()
        return self

    def enable_step(self, index: int) -> "Orchestrator":
        """Enable step at index."""
        if 0 <= index < len(self._steps):
            self._steps[index].enabled = True
        return self

    def disable_step(self, index: int) -> "Orchestrator":
        """Disable step at index."""
        if 0 <= index < len(self._steps):
            self._steps[index].enabled = False
        return self

    @property
    def steps(self) -> List[PipelineStep]:
        """Get list of steps."""
        return [s.step for s in self._steps]

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        step_names = [s.step.name for s in self._steps]
        return f"Orchestrator({' -> '.join(step_names)})"


def compose(*steps: PipelineStep, pool: Optional[ModelPool] = None) -> Orchestrator:
    """
    Convenience function to compose steps into an orchestrator.

    Example:
        orchestrator = compose(
            EncodeStep(),
            GenerateStep(),
            DecodeStep(),
            pool=my_pool,
        )
        result = orchestrator.run({"prompt": "Hello"})
    """
    orchestrator = Orchestrator(model_pool=pool)
    for step in steps:
        orchestrator.add_step(step)
    return orchestrator
