"""
Base training module for diffusion models.

Provides common functionality for model training, including:
- LoRA injection using PEFT
- Gradient checkpointing
- Trainable parameter management
- State dict export

Based on DiffSynth-Studio implementation (Apache 2.0 license).
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Set

import torch
import torch.nn as nn

from llm_dit.training.config import LoRAConfig, TrainingConfig
from llm_dit.training.gradient_checkpoint import (
    enable_gradient_checkpointing,
    disable_gradient_checkpointing,
)

logger = logging.getLogger(__name__)


class BaseTrainingModule(nn.Module, ABC):
    """
    Base class for training diffusion models.

    Provides common functionality for model training including LoRA
    injection, gradient checkpointing, and state dict management.

    Subclasses must implement:
        - get_pipeline_inputs(): Convert dataset items to pipeline inputs
        - forward(): Compute training loss

    Attributes:
        pipe: The underlying pipeline (set by subclass)
        config: Training configuration
        lora_config: LoRA configuration (if using LoRA)
        trainable_model_names: Set of model names being trained
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ):
        """
        Initialize the training module.

        Args:
            config: Training configuration
            lora_config: Optional LoRA configuration
        """
        super().__init__()

        self.pipe = None  # Set by subclass
        self.config = config or TrainingConfig()
        self.lora_config = lora_config
        self.trainable_model_names: Set[str] = set()
        self._lora_layers: Dict[str, Any] = {}

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """
        Iterate over trainable parameters.

        Yields:
            Parameters that require gradients
        """
        for name, param in self.named_parameters():
            if param.requires_grad:
                yield param

    def trainable_param_names(self) -> Set[str]:
        """
        Get names of trainable parameters.

        Returns:
            Set of parameter names that require gradients
        """
        return {name for name, param in self.named_parameters() if param.requires_grad}

    def num_trainable_params(self) -> int:
        """
        Count trainable parameters.

        Returns:
            Total number of trainable parameters
        """
        return sum(p.numel() for p in self.trainable_parameters())

    def freeze_all(self) -> None:
        """Freeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def freeze_except(self, trainable_models: List[str]) -> None:
        """
        Freeze all models except specified ones.

        Args:
            trainable_models: List of model names to keep trainable
        """
        if self.pipe is None:
            raise RuntimeError("Pipeline not initialized")

        # First freeze everything
        self.freeze_all()

        # Then unfreeze specified models
        for model_name in trainable_models:
            model = getattr(self.pipe, model_name, None)
            if model is None:
                logger.warning(f"Model '{model_name}' not found in pipeline")
                continue

            model.requires_grad_(True)
            self.trainable_model_names.add(model_name)

            # Enable gradient checkpointing if configured
            if self.config.use_gradient_checkpointing:
                enable_gradient_checkpointing(model)

            logger.info(f"Enabled training for: {model_name}")

    def add_lora_to_model(
        self,
        model: nn.Module,
        target_modules: List[str],
        lora_rank: int,
        lora_alpha: Optional[int] = None,
        lora_dropout: float = 0.0,
    ) -> nn.Module:
        """
        Add LoRA adapters to a model using PEFT.

        Args:
            model: Model to add LoRA to
            target_modules: Names of modules to apply LoRA
            lora_rank: LoRA rank (r in the paper)
            lora_alpha: LoRA alpha scaling (defaults to rank)
            lora_dropout: Dropout probability for LoRA layers

        Returns:
            Model with LoRA adapters
        """
        try:
            from peft import LoraConfig, get_peft_model, inject_adapter_in_model
        except ImportError:
            raise ImportError(
                "PEFT is required for LoRA training. "
                "Install with: pip install peft"
            )

        if lora_alpha is None:
            lora_alpha = lora_rank

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

        # Inject LoRA adapters
        model = inject_adapter_in_model(lora_config, model)

        # Track LoRA layers
        self._lora_layers[id(model)] = lora_config

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        logger.info(
            f"Added LoRA (rank={lora_rank}, alpha={lora_alpha}): "
            f"{trainable:,} trainable / {total:,} total params "
            f"({100*trainable/total:.2f}%)"
        )

        return model

    def load_lora_checkpoint(
        self,
        model: nn.Module,
        checkpoint_path: str,
    ) -> None:
        """
        Load LoRA weights from checkpoint.

        Args:
            model: Model with LoRA adapters
            checkpoint_path: Path to LoRA checkpoint
        """
        from safetensors.torch import load_file

        state_dict = load_file(checkpoint_path)

        # Filter to LoRA keys only
        lora_state_dict = {
            k: v for k, v in state_dict.items()
            if "lora" in k.lower()
        }

        model.load_state_dict(lora_state_dict, strict=False)
        logger.info(f"Loaded LoRA checkpoint from: {checkpoint_path}")

    def export_trainable_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        remove_prefix: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Export only trainable parameters from state dict.

        Args:
            state_dict: Full state dict
            remove_prefix: Prefix to remove from keys

        Returns:
            Filtered state dict with only trainable params
        """
        trainable_names = self.trainable_param_names()

        filtered = {}
        for name, value in state_dict.items():
            if name not in trainable_names:
                continue

            # Remove prefix if specified
            export_name = name
            if remove_prefix and export_name.startswith(remove_prefix):
                export_name = export_name[len(remove_prefix):]

            filtered[export_name] = value

        return filtered

    def to(self, *args, **kwargs) -> "BaseTrainingModule":
        """Move module to device/dtype."""
        super().to(*args, **kwargs)
        if self.pipe is not None:
            self.pipe.to(*args, **kwargs)
        return self

    @abstractmethod
    def get_pipeline_inputs(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert dataset item to pipeline inputs.

        Args:
            data: Dataset item dictionary

        Returns:
            Dictionary of inputs for the pipeline/loss function
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        """
        Forward pass computing training loss.

        Args:
            data: Dataset item

        Returns:
            Loss tensor
        """
        raise NotImplementedError
