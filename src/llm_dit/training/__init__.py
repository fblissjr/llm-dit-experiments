"""
Training infrastructure for llm-dit-experiments.

This module provides everything needed to train Z-Image models:
- Training modules (BaseTrainingModule, ZImageTrainingModule)
- Loss functions (FlowMatchSFTLoss, DirectDistillLoss)
- Dataset utilities (TrainingDataset, CachedEmbeddingDataset)
- Training runner (launch_training_task)
- Checkpoint logging (ModelLogger)

Quick Start:
    ```python
    from accelerate import Accelerator
    from llm_dit.training import (
        ZImageTrainingModule,
        TrainingDataset,
        ModelLogger,
        TrainingConfig,
        launch_training_task,
    )

    # Configure training
    config = TrainingConfig(
        model_path="/path/to/z-image",
        use_lora=True,
        lora_rank=32,
        num_epochs=10,
        learning_rate=1e-4,
    )

    # Initialize
    accelerator = Accelerator(mixed_precision="bf16")
    dataset = TrainingDataset("data/train.jsonl", "data/images")
    model = ZImageTrainingModule(config)
    logger = ModelLogger("./checkpoints")

    # Train
    launch_training_task(accelerator, dataset, model, logger, config)
    ```

See Also:
    - scripts/train.py for complete training script
    - internal/research/training_infrastructure_design.md for architecture
"""

from llm_dit.training.config import (
    TrainingConfig,
    LoRAConfig,
)

from llm_dit.training.base import (
    BaseTrainingModule,
)

from llm_dit.training.z_image import (
    ZImageTrainingModule,
)

from llm_dit.training.losses import (
    FlowMatchSFTLoss,
    DirectDistillLoss,
    ConsistencyLoss,
)

from llm_dit.training.gradient_checkpoint import (
    gradient_checkpoint_forward,
    enable_gradient_checkpointing,
    disable_gradient_checkpointing,
)

from llm_dit.training.logger import (
    ModelLogger,
    TensorBoardLogger,
)

from llm_dit.training.runner import (
    launch_training_task,
    launch_validation_task,
    create_optimizer,
    create_lr_scheduler,
    create_dataloader,
)

from llm_dit.training.data import (
    TrainingDataset,
    CachedEmbeddingDataset,
)

__all__ = [
    # Config
    "TrainingConfig",
    "LoRAConfig",
    # Training modules
    "BaseTrainingModule",
    "ZImageTrainingModule",
    # Loss functions
    "FlowMatchSFTLoss",
    "DirectDistillLoss",
    "ConsistencyLoss",
    # Gradient checkpointing
    "gradient_checkpoint_forward",
    "enable_gradient_checkpointing",
    "disable_gradient_checkpointing",
    # Logging
    "ModelLogger",
    "TensorBoardLogger",
    # Runner
    "launch_training_task",
    "launch_validation_task",
    "create_optimizer",
    "create_lr_scheduler",
    "create_dataloader",
    # Data
    "TrainingDataset",
    "CachedEmbeddingDataset",
]
