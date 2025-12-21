"""
Model checkpoint logging and saving.

Usage:
    from llm_dit.training import ModelLogger

    logger = ModelLogger(output_dir="./checkpoints")
    logger.on_step_end(accelerator, model, step=100)
    logger.on_epoch_end(accelerator, model, epoch=1)
"""

import logging
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from accelerate import Accelerator
    from llm_dit.training.base import BaseTrainingModule

logger = logging.getLogger(__name__)


class ModelLogger:
    """
    Checkpoint saving and logging.

    Handles:
        - Per-step checkpointing
        - Per-epoch checkpointing
        - State dict filtering (trainable params only)
        - Prefix removal for clean exports
        - Optional optimizer state saving

    Attributes:
        output_dir: Directory for checkpoints
        remove_prefix: Prefix to remove from state dict keys
        save_optimizer: Whether to save optimizer state
        num_steps: Total steps processed
    """

    def __init__(
        self,
        output_dir: str,
        remove_prefix: Optional[str] = None,
        save_optimizer: bool = False,
    ):
        """
        Initialize the logger.

        Args:
            output_dir: Directory to save checkpoints
            remove_prefix: Prefix to remove from state dict keys
            save_optimizer: Whether to save optimizer state
        """
        self.output_dir = Path(output_dir)
        self.remove_prefix = remove_prefix
        self.save_optimizer = save_optimizer
        self.num_steps = 0

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ModelLogger initialized: {self.output_dir}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """
        Log training metrics.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number
        """
        step_str = f"[Step {step}]" if step is not None else ""
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info(f"{step_str} {metrics_str}")

    def on_step_end(
        self,
        accelerator: "Accelerator",
        model: "BaseTrainingModule",
        save_steps: Optional[int] = None,
        step: Optional[int] = None,
    ) -> None:
        """
        Called after each training step.

        Args:
            accelerator: Accelerator instance
            model: Training module
            save_steps: Save checkpoint every N steps
            step: Current step number (uses internal counter if None)
        """
        if step is not None:
            self.num_steps = step
        else:
            self.num_steps += 1

        if save_steps and self.num_steps % save_steps == 0:
            self._save_checkpoint(
                accelerator,
                model,
                f"step-{self.num_steps}.safetensors",
            )

    def on_epoch_end(
        self,
        accelerator: "Accelerator",
        model: "BaseTrainingModule",
        epoch: int,
    ) -> None:
        """
        Called after each epoch.

        Args:
            accelerator: Accelerator instance
            model: Training module
            epoch: Completed epoch number
        """
        self._save_checkpoint(
            accelerator,
            model,
            f"epoch-{epoch}.safetensors",
        )

    def save_final(
        self,
        accelerator: "Accelerator",
        model: "BaseTrainingModule",
    ) -> None:
        """
        Save final checkpoint.

        Args:
            accelerator: Accelerator instance
            model: Training module
        """
        self._save_checkpoint(
            accelerator,
            model,
            "final.safetensors",
        )

    def _save_checkpoint(
        self,
        accelerator: "Accelerator",
        model: "BaseTrainingModule",
        filename: str,
    ) -> None:
        """
        Save model checkpoint.

        Args:
            accelerator: Accelerator instance
            model: Training module
            filename: Checkpoint filename
        """
        accelerator.wait_for_everyone()

        if not accelerator.is_main_process:
            return

        try:
            from safetensors.torch import save_file

            # Get state dict from accelerator
            state_dict = accelerator.get_state_dict(model)

            # Export only trainable parameters
            model_unwrapped = accelerator.unwrap_model(model)
            state_dict = model_unwrapped.export_trainable_state_dict(
                state_dict,
                remove_prefix=self.remove_prefix,
            )

            # Save with safetensors
            save_path = self.output_dir / filename
            save_file(state_dict, save_path)

            logger.info(f"Saved checkpoint: {save_path}")

        except ImportError:
            # Fall back to torch.save
            state_dict = accelerator.get_state_dict(model)
            model_unwrapped = accelerator.unwrap_model(model)
            state_dict = model_unwrapped.export_trainable_state_dict(
                state_dict,
                remove_prefix=self.remove_prefix,
            )

            save_path = self.output_dir / filename.replace(".safetensors", ".pt")
            torch.save(state_dict, save_path)

            logger.info(f"Saved checkpoint (pt format): {save_path}")


class TensorBoardLogger:
    """
    TensorBoard logging for training metrics.

    Optional wrapper around SummaryWriter for consistent logging.
    """

    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Directory for TensorBoard logs
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
            logger.info(f"TensorBoard logging enabled: {log_dir}")
        except ImportError:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")
            self.writer = None
            self.enabled = False

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value."""
        if self.enabled and self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int) -> None:
        """Log multiple scalars."""
        if self.enabled and self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_image(self, tag: str, img_tensor: torch.Tensor, step: int) -> None:
        """Log an image."""
        if self.enabled and self.writer is not None:
            self.writer.add_image(tag, img_tensor, step)

    def close(self) -> None:
        """Close the writer."""
        if self.enabled and self.writer is not None:
            self.writer.close()
