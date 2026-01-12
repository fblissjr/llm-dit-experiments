"""
Pipeline Steps - Composable units for orchestration.

Last Updated: 2026-01-12

Steps are the fundamental building blocks:
- Declare inputs/outputs for automatic wiring
- Declare required models from the pool
- Execute with validated inputs and model instances

Granularity is at the MODEL level, not pipeline level.
Write custom steps that use any subset of models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, List, Optional, Type


@dataclass
class StepInput:
    """
    Declaration of a step input.

    Attributes:
        name: Parameter name in execute()
        type: Expected type for validation
        required: Whether input must be provided
        default: Default value if not provided
        description: Human-readable description
    """

    name: str
    type: Type
    required: bool = True
    default: Any = None
    description: str = ""

    def __post_init__(self):
        if self.default is not None:
            self.required = False


@dataclass
class StepOutput:
    """
    Declaration of a step output.

    Attributes:
        name: Key in the returned dict
        type: Type of the output value
        description: Human-readable description
    """

    name: str
    type: Type
    description: str = ""


class PipelineStep(ABC):
    """
    Base class for composable pipeline steps.

    Steps declare their inputs, outputs, and required models.
    The orchestrator uses these declarations for:
    - Input validation
    - Automatic wiring between steps
    - Model loading/offloading

    Example - Full pipeline adapter:
        class ZImageAdapter(PipelineStep):
            inputs = [
                StepInput("prompt", str),
                StepInput("width", int, default=1024),
                StepInput("height", int, default=1024),
            ]
            outputs = [StepOutput("image", ImageOutput)]
            required_models = ["z-image-encoder", "z-image-dit", "z-image-vae"]

            def execute(self, inputs, models):
                # Use all three models together
                ...

    Example - Single model step:
        class TextEncoderStep(PipelineStep):
            inputs = [StepInput("prompt", str)]
            outputs = [StepOutput("embeddings", TextEmbeddings)]
            required_models = ["text-encoder"]  # Just one model

            def execute(self, inputs, models):
                encoder = models["text-encoder"]
                return {"embeddings": encoder.encode(inputs["prompt"])}

    Example - No models (pure computation):
        class ResizeStep(PipelineStep):
            inputs = [StepInput("image", ImageOutput), StepInput("scale", float)]
            outputs = [StepOutput("image", ImageOutput)]
            required_models = []  # No ML models needed

            def execute(self, inputs, models):
                img = inputs["image"].image.resize(...)
                return {"image": ImageOutput(image=img)}
    """

    # Subclasses override these class attributes (ClassVar ensures no accidental mutation)
    inputs: ClassVar[List[StepInput]] = []
    outputs: ClassVar[List[StepOutput]] = []
    required_models: ClassVar[List[str]] = []

    # Optional metadata (can be overridden per-instance)
    name: str = ""  # Display name (defaults to class name)
    description: str = ""

    def __init__(self, **config):
        """
        Initialize step with optional configuration.

        Config is stored and can be used in execute().
        """
        self.config = config
        if not self.name:
            self.name = self.__class__.__name__

    @abstractmethod
    def execute(
        self,
        inputs: Dict[str, Any],
        models: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute the step.

        Args:
            inputs: Validated input dict (keys match self.inputs names)
            models: Dict of model_name -> model_instance for required_models

        Returns:
            Dict of output_name -> value (keys match self.outputs names)
        """
        pass

    def validate_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and fill defaults for inputs.

        Args:
            inputs: Raw input dict

        Returns:
            Validated input dict with defaults filled

        Raises:
            ValueError: If required input missing
            TypeError: If input has wrong type
        """
        validated = {}

        for inp in self.inputs:
            if inp.name in inputs:
                value = inputs[inp.name]
                # Type check (allow None for optional)
                if value is not None and not isinstance(value, inp.type):
                    # Skip check for generic types (Union, Optional, etc.)
                    if not hasattr(inp.type, "__origin__"):
                        raise TypeError(
                            f"Input '{inp.name}' must be {inp.type.__name__}, "
                            f"got {type(value).__name__}"
                        )
                validated[inp.name] = value
            elif inp.required:
                raise ValueError(f"Missing required input: {inp.name}")
            else:
                validated[inp.name] = inp.default

        return validated

    def validate_outputs(self, outputs: Dict[str, Any]) -> None:
        """
        Validate that execute() returned expected outputs.

        Args:
            outputs: Output dict from execute()

        Raises:
            ValueError: If required output missing
        """
        for out in self.outputs:
            if out.name not in outputs:
                raise ValueError(
                    f"Step {self.name} missing output: {out.name}"
                )

    def __repr__(self) -> str:
        return f"{self.name}(inputs={[i.name for i in self.inputs]}, models={self.required_models})"


class FunctionStep(PipelineStep):
    """
    Create a step from a function.

    For quick one-off steps without defining a class.

    Example:
        def my_transform(inputs, models):
            return {"result": inputs["data"] * 2}

        step = FunctionStep(
            fn=my_transform,
            inputs=[StepInput("data", float)],
            outputs=[StepOutput("result", float)],
        )
    """

    def __init__(
        self,
        fn: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
        inputs: Optional[List[StepInput]] = None,
        outputs: Optional[List[StepOutput]] = None,
        required_models: Optional[List[str]] = None,
        name: Optional[str] = None,
        **config,
    ):
        super().__init__(**config)
        self._fn = fn
        if inputs is not None:
            self.inputs = inputs
        if outputs is not None:
            self.outputs = outputs
        if required_models is not None:
            self.required_models = required_models
        if name is not None:
            self.name = name
        else:
            self.name = fn.__name__

    def execute(
        self,
        inputs: Dict[str, Any],
        models: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self._fn(inputs, models)


class ConditionalStep(PipelineStep):
    """
    Step that conditionally executes one of multiple branches.

    NOTE: Models from BOTH branches are loaded, even though only one runs.
    This is a known limitation. For VRAM-sensitive cases where branches use
    different models, consider using separate orchestrator configurations
    or pre-evaluating the condition.

    Example:
        step = ConditionalStep(
            condition=lambda inputs: inputs.get("mode") == "fast",
            if_true=FastGenerateStep(),
            if_false=QualityGenerateStep(),
        )
    """

    def __init__(
        self,
        condition: Callable[[Dict[str, Any]], bool],
        if_true: PipelineStep,
        if_false: PipelineStep,
        **config,
    ):
        super().__init__(**config)
        self._condition = condition
        self._if_true = if_true
        self._if_false = if_false

        # Merge inputs/outputs from both branches
        self.inputs = list(set(if_true.inputs + if_false.inputs))
        self.outputs = list(set(if_true.outputs + if_false.outputs))
        # NOTE: All models from both branches are loaded - limitation of current design
        self.required_models = list(set(if_true.required_models + if_false.required_models))

    def execute(
        self,
        inputs: Dict[str, Any],
        models: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self._condition(inputs):
            return self._if_true.execute(inputs, models)
        else:
            return self._if_false.execute(inputs, models)


class LoopStep(PipelineStep):
    """
    Step that loops over a list input.

    Example:
        step = LoopStep(
            inner_step=ProcessFrameStep(),
            list_input="frames",
            output_name="processed_frames",
        )
    """

    def __init__(
        self,
        inner_step: PipelineStep,
        list_input: str,
        output_name: str,
        **config,
    ):
        super().__init__(**config)
        self._inner = inner_step
        self._list_input = list_input
        self._output_name = output_name

        # Inherit from inner step
        self.inputs = inner_step.inputs
        self.outputs = [StepOutput(output_name, list)]
        self.required_models = inner_step.required_models

    def execute(
        self,
        inputs: Dict[str, Any],
        models: Dict[str, Any],
    ) -> Dict[str, Any]:
        items = inputs.get(self._list_input, [])
        results = []

        for item in items:
            # Create input dict with current item
            item_inputs = dict(inputs)
            item_inputs[self._list_input] = item

            # Execute inner step
            output = self._inner.execute(item_inputs, models)
            results.append(output)

        return {self._output_name: results}


@dataclass
class StepConfig:
    """
    Configuration for a step in an orchestration.

    Attributes:
        step: The PipelineStep instance
        input_mapping: Map step input names to context keys
        output_mapping: Map step output names to context keys
        enabled: Whether step is enabled
        on_error: Error handling strategy
    """

    step: PipelineStep
    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_mapping: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    on_error: str = "raise"  # raise, skip, default
