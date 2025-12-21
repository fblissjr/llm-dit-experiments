"""
Training runner using HuggingFace Accelerate.

Provides the main training loop with:
- Distributed training support
- Mixed precision training
- Gradient accumulation
- Learning rate scheduling
- Checkpoint saving

Usage:
    from llm_dit.training import launch_training_task

    launch_training_task(
        accelerator=accelerator,
        dataset=dataset,
        model=model,
        logger=logger,
        config=config,
    )
"""

import logging
from typing import TYPE_CHECKING, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if TYPE_CHECKING:
    from accelerate import Accelerator
    from llm_dit.training.base import BaseTrainingModule
    from llm_dit.training.config import TrainingConfig
    from llm_dit.training.logger import ModelLogger
    from llm_dit.training.data import TrainingDataset

logger = logging.getLogger(__name__)


def create_optimizer(
    model: "BaseTrainingModule",
    config: "TrainingConfig",
) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer for training.

    Args:
        model: Training module
        config: Training configuration

    Returns:
        Configured optimizer
    """
    return torch.optim.AdamW(
        model.trainable_parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
    )


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: "TrainingConfig",
    num_training_steps: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer instance
        config: Training configuration
        num_training_steps: Total number of training steps

    Returns:
        Learning rate scheduler
    """
    if config.lr_scheduler == "constant":
        return torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=num_training_steps,
        )

    elif config.lr_scheduler == "constant_with_warmup":
        from transformers import get_constant_schedule_with_warmup
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
        )

    elif config.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
        )

    elif config.lr_scheduler == "cosine_with_warmup":
        from transformers import get_cosine_schedule_with_warmup
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps,
        )

    elif config.lr_scheduler == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=num_training_steps,
        )

    elif config.lr_scheduler == "linear_with_warmup":
        from transformers import get_linear_schedule_with_warmup
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps,
        )

    else:
        raise ValueError(f"Unknown lr_scheduler: {config.lr_scheduler}")


def create_dataloader(
    dataset: "TrainingDataset",
    config: "TrainingConfig",
) -> DataLoader:
    """
    Create training dataloader.

    Args:
        dataset: Training dataset
        config: Training configuration

    Returns:
        DataLoader instance
    """
    def collate_fn(batch):
        """Collate function for single-item batches."""
        if len(batch) == 1:
            return batch[0]
        return batch

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def launch_training_task(
    accelerator: "Accelerator",
    dataset: "TrainingDataset",
    model: "BaseTrainingModule",
    model_logger: "ModelLogger",
    config: "TrainingConfig",
    tb_logger: Optional["TensorBoardLogger"] = None,
) -> None:
    """
    Main training loop.

    Runs the complete training process with:
    - Distributed training via Accelerate
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Checkpoint saving
    - Progress logging

    Args:
        accelerator: HuggingFace Accelerator instance
        dataset: Training dataset
        model: Training module
        model_logger: Checkpoint logger
        config: Training configuration
        tb_logger: Optional TensorBoard logger
    """
    from llm_dit.training.logger import TensorBoardLogger

    # Create dataloader
    dataloader = create_dataloader(dataset, config)

    # Calculate total steps
    num_update_steps_per_epoch = len(dataloader) // config.gradient_accumulation_steps
    num_training_steps = config.num_epochs * num_update_steps_per_epoch

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    lr_scheduler = create_lr_scheduler(optimizer, config, num_training_steps)

    # Prepare for distributed training
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    # Log training info
    if accelerator.is_main_process:
        logger.info("=" * 60)
        logger.info("Training Configuration")
        logger.info("=" * 60)
        logger.info(f"  Model path: {config.model_path}")
        logger.info(f"  Task: {config.task}")
        logger.info(f"  Epochs: {config.num_epochs}")
        logger.info(f"  Batch size: {config.batch_size}")
        logger.info(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
        logger.info(f"  Learning rate: {config.learning_rate}")
        logger.info(f"  Total steps: {num_training_steps}")
        logger.info(f"  Mixed precision: {config.mixed_precision}")
        logger.info(f"  Gradient checkpointing: {config.use_gradient_checkpointing}")
        logger.info("=" * 60)

    # Set seed for reproducibility
    if config.seed is not None:
        torch.manual_seed(config.seed)

    # Training loop
    global_step = 0

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{config.num_epochs}",
            disable=not accelerator.is_local_main_process,
        )

        for step, data in enumerate(progress_bar):
            with accelerator.accumulate(model):
                # Forward pass
                loss = model(data)

                # Backward pass
                accelerator.backward(loss)

                # Gradient clipping
                if config.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        config.max_grad_norm,
                    )

                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Track loss
                epoch_loss += loss.detach().float()
                num_batches += 1

            # Update global step
            if accelerator.sync_gradients:
                global_step += 1

                # Logging
                if global_step % config.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches

                    if accelerator.is_main_process:
                        model_logger.log_metrics(
                            {"loss": avg_loss.item(), "lr": lr_scheduler.get_last_lr()[0]},
                            step=global_step,
                        )

                        if tb_logger is not None:
                            tb_logger.log_scalar("train/loss", avg_loss.item(), global_step)
                            tb_logger.log_scalar("train/lr", lr_scheduler.get_last_lr()[0], global_step)

                # Step checkpoint
                model_logger.on_step_end(
                    accelerator,
                    model,
                    save_steps=config.save_steps,
                    step=global_step,
                )

            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
            })

        # End of epoch
        avg_epoch_loss = epoch_loss / num_batches
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch + 1} complete. Average loss: {avg_epoch_loss:.4f}")

        # Epoch checkpoint
        if (epoch + 1) % config.save_epochs == 0:
            model_logger.on_epoch_end(accelerator, model, epoch + 1)

    # Save final checkpoint
    model_logger.save_final(accelerator, model)

    if accelerator.is_main_process:
        logger.info("Training complete!")

    # Cleanup
    if tb_logger is not None:
        tb_logger.close()


def launch_validation_task(
    accelerator: "Accelerator",
    dataset: "TrainingDataset",
    model: "BaseTrainingModule",
    config: "TrainingConfig",
) -> float:
    """
    Run validation on a dataset.

    Args:
        accelerator: Accelerator instance
        dataset: Validation dataset
        model: Training module
        config: Training configuration

    Returns:
        Average validation loss
    """
    dataloader = create_dataloader(dataset, config)
    dataloader = accelerator.prepare(dataloader)

    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Validation", disable=not accelerator.is_local_main_process):
            loss = model(data)
            total_loss += loss.float()
            num_batches += 1

    avg_loss = accelerator.gather(total_loss).mean() / num_batches

    if accelerator.is_main_process:
        logger.info(f"Validation loss: {avg_loss:.4f}")

    return avg_loss.item()
