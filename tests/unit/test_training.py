"""
Unit tests for llm_dit.training module.

Tests cover:
- Loss functions (FlowMatchSFTLoss, DirectDistillLoss, ConsistencyLoss)
- Gradient checkpointing utilities
- Training configuration dataclasses
- Dataset loading
- Training runner utilities
"""

from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.nn as nn


# ============================================================================
# Training Configuration Tests
# ============================================================================

class TestLoRAConfig:
    """Test LoRAConfig dataclass."""

    def test_default_values(self):
        """Test LoRAConfig default values."""
        from llm_dit.training.config import LoRAConfig

        config = LoRAConfig()
        assert config.base_model == "transformer"
        assert config.rank == 32
        assert config.alpha == 32
        assert config.dropout == 0.0
        assert config.checkpoint is None
        assert "to_q" in config.target_modules
        assert "to_k" in config.target_modules
        assert "to_v" in config.target_modules

    def test_custom_values(self):
        """Test LoRAConfig with custom values."""
        from llm_dit.training.config import LoRAConfig

        config = LoRAConfig(
            base_model="dit",
            target_modules=["q_proj", "v_proj"],
            rank=16,
            alpha=32,
            dropout=0.1,
            checkpoint="/path/to/lora",
        )
        assert config.base_model == "dit"
        assert config.rank == 16
        assert config.alpha == 32
        assert config.dropout == 0.1
        assert config.checkpoint == "/path/to/lora"
        assert config.target_modules == ["q_proj", "v_proj"]

    def test_from_dict(self):
        """Test LoRAConfig.from_dict()."""
        from llm_dit.training.config import LoRAConfig

        data = {
            "base_model": "transformer",
            "rank": 64,
            "alpha": 128,
            "extra_field": "ignored",  # Should be filtered out
        }
        config = LoRAConfig.from_dict(data)
        assert config.rank == 64
        assert config.alpha == 128


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_default_values(self):
        """Test TrainingConfig default values."""
        from llm_dit.training.config import TrainingConfig

        config = TrainingConfig()
        assert config.model_path == ""
        assert config.task == "sft"
        assert config.use_lora is False
        assert config.num_epochs == 1
        assert config.batch_size == 1
        assert config.learning_rate == 1e-5
        assert config.mixed_precision == "bf16"
        assert config.use_gradient_checkpointing is True
        assert config.lr_scheduler == "constant"
        assert config.max_grad_norm == 1.0

    def test_custom_values(self):
        """Test TrainingConfig with custom values."""
        from llm_dit.training.config import TrainingConfig

        config = TrainingConfig(
            model_path="/path/to/model",
            task="distill",
            use_lora=True,
            lora_rank=16,
            num_epochs=10,
            learning_rate=5e-6,
        )
        assert config.model_path == "/path/to/model"
        assert config.task == "distill"
        assert config.use_lora is True
        assert config.lora_rank == 16
        assert config.num_epochs == 10
        assert config.learning_rate == 5e-6

    def test_from_dict(self):
        """Test TrainingConfig.from_dict()."""
        from llm_dit.training.config import TrainingConfig

        data = {
            "model_path": "/path/to/model",
            "num_epochs": 5,
            "batch_size": 4,
            "extra_field": "ignored",
        }
        config = TrainingConfig.from_dict(data)
        assert config.model_path == "/path/to/model"
        assert config.num_epochs == 5
        assert config.batch_size == 4

    def test_to_dict(self):
        """Test TrainingConfig.to_dict()."""
        from llm_dit.training.config import TrainingConfig

        config = TrainingConfig(model_path="/test", num_epochs=3)
        data = config.to_dict()
        assert data["model_path"] == "/test"
        assert data["num_epochs"] == 3
        assert "learning_rate" in data

    def test_get_lora_config_disabled(self):
        """Test get_lora_config when LoRA is disabled."""
        from llm_dit.training.config import TrainingConfig

        config = TrainingConfig(use_lora=False)
        lora_config = config.get_lora_config()
        assert lora_config is None

    def test_get_lora_config_enabled(self):
        """Test get_lora_config when LoRA is enabled."""
        from llm_dit.training.config import TrainingConfig

        config = TrainingConfig(
            use_lora=True,
            lora_rank=64,
            lora_alpha=128,
            lora_dropout=0.05,
        )
        lora_config = config.get_lora_config()
        assert lora_config is not None
        assert lora_config.rank == 64
        assert lora_config.alpha == 128
        assert lora_config.dropout == 0.05


class TestTrainingConfigFromTOML:
    """Test TrainingConfig TOML loading."""

    def test_from_toml(self, tmp_path):
        """Test loading TrainingConfig from TOML file."""
        from llm_dit.training.config import TrainingConfig

        toml_content = """
[training]
model_path = "/path/to/model"
num_epochs = 10
batch_size = 2
learning_rate = 1e-5
use_lora = true
lora_rank = 32
"""
        config_path = tmp_path / "train_config.toml"
        config_path.write_text(toml_content)

        config = TrainingConfig.from_toml(config_path)
        assert config.model_path == "/path/to/model"
        assert config.num_epochs == 10
        assert config.batch_size == 2
        assert config.use_lora is True
        assert config.lora_rank == 32


# ============================================================================
# Gradient Checkpointing Tests
# ============================================================================

class TestGradientCheckpointing:
    """Test gradient checkpointing utilities."""

    def test_create_custom_forward(self):
        """Test create_custom_forward creates callable wrapper."""
        from llm_dit.training.gradient_checkpoint import create_custom_forward

        class SimpleModule(nn.Module):
            def forward(self, x):
                return x * 2

        module = SimpleModule()
        custom_forward = create_custom_forward(module)

        x = torch.randn(4, 4)
        result = custom_forward(x)
        expected = module(x)
        assert torch.allclose(result, expected)

    def test_gradient_checkpoint_forward_no_checkpoint(self):
        """Test forward without checkpointing."""
        from llm_dit.training.gradient_checkpoint import gradient_checkpoint_forward

        class SimpleModule(nn.Module):
            def forward(self, x):
                return x * 2

        module = SimpleModule()
        x = torch.randn(4, 4)

        result = gradient_checkpoint_forward(
            module,
            False,  # use_gradient_checkpointing
            x,
        )
        expected = module(x)
        assert torch.allclose(result, expected)

    def test_gradient_checkpoint_forward_with_checkpoint(self):
        """Test forward with checkpointing enabled."""
        from llm_dit.training.gradient_checkpoint import gradient_checkpoint_forward

        class SimpleModule(nn.Module):
            def forward(self, x):
                return x * 2 + 1

        module = SimpleModule()
        x = torch.randn(4, 4, requires_grad=True)

        result = gradient_checkpoint_forward(
            module,
            True,  # use_gradient_checkpointing
            x,
        )
        expected = module(x)
        assert torch.allclose(result, expected)

    def test_enable_gradient_checkpointing(self):
        """Test enable_gradient_checkpointing utility."""
        from llm_dit.training.gradient_checkpoint import enable_gradient_checkpointing

        # Test with HuggingFace-style method
        mock_model = MagicMock()
        mock_model.gradient_checkpointing_enable = MagicMock()
        enable_gradient_checkpointing(mock_model)
        mock_model.gradient_checkpointing_enable.assert_called_once()

    def test_disable_gradient_checkpointing(self):
        """Test disable_gradient_checkpointing utility."""
        from llm_dit.training.gradient_checkpoint import disable_gradient_checkpointing

        mock_model = MagicMock()
        mock_model.gradient_checkpointing_disable = MagicMock()
        disable_gradient_checkpointing(mock_model)
        mock_model.gradient_checkpointing_disable.assert_called_once()


# ============================================================================
# Loss Function Tests
# ============================================================================

class TestFlowMatchSFTLoss:
    """Test FlowMatchSFTLoss function."""

    @pytest.fixture
    def mock_pipe(self):
        """Create mock pipeline for loss testing."""
        pipe = MagicMock()

        # Mock scheduler
        pipe.scheduler = MagicMock()
        pipe.scheduler.timesteps = torch.linspace(1.0, 0.0, 10)

        def add_noise(sample, noise, timestep):
            # Simple linear interpolation
            t = timestep / 1000.0 if timestep.max() > 1 else timestep
            return sample * (1 - t) + noise * t

        pipe.scheduler.add_noise = add_noise

        def training_target(sample, noise, timestep):
            # Flow matching velocity: v = noise - sample
            return noise - sample

        pipe.scheduler.training_target = training_target

        def training_weight(timestep):
            # Uniform weight
            return torch.ones_like(timestep)

        pipe.scheduler.training_weight = training_weight

        # Mock transformer
        pipe.transformer = MagicMock()

        def mock_forward(hidden_states, timestep, encoder_hidden_states, **kwargs):
            # Return tensor matching input shape
            return MagicMock(sample=torch.randn_like(hidden_states))

        pipe.transformer.side_effect = mock_forward
        pipe.transformer.return_value = MagicMock(sample=torch.randn(1, 4, 64, 64))

        return pipe

    def test_flow_match_sft_loss_runs(self, mock_pipe):
        """Test FlowMatchSFTLoss executes without error."""
        from llm_dit.training.losses import FlowMatchSFTLoss

        input_latents = torch.randn(1, 4, 64, 64)
        prompt_embeds = torch.randn(1, 77, 2560)

        loss = FlowMatchSFTLoss(
            mock_pipe,
            input_latents=input_latents,
            prompt_embeds=prompt_embeds,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # MSE is non-negative

    def test_flow_match_sft_loss_timestep_bounds(self, mock_pipe):
        """Test FlowMatchSFTLoss with custom timestep boundaries."""
        from llm_dit.training.losses import FlowMatchSFTLoss

        input_latents = torch.randn(1, 4, 64, 64)
        prompt_embeds = torch.randn(1, 77, 2560)

        loss = FlowMatchSFTLoss(
            mock_pipe,
            input_latents=input_latents,
            prompt_embeds=prompt_embeds,
            min_timestep_boundary=0.2,
            max_timestep_boundary=0.8,
        )

        assert isinstance(loss, torch.Tensor)


class TestDirectDistillLoss:
    """Test DirectDistillLoss function."""

    @pytest.fixture
    def mock_distill_pipe(self):
        """Create mock pipeline for distillation loss testing."""
        pipe = MagicMock()

        # Mock scheduler
        pipe.scheduler = MagicMock()
        pipe.scheduler.timesteps = torch.linspace(1.0, 0.0, 8)
        pipe.scheduler.set_timesteps = MagicMock()

        def step(model_output, timestep, sample):
            return MagicMock(prev_sample=sample - model_output * 0.1)

        pipe.scheduler.step = step

        # Mock transformer
        pipe.transformer = MagicMock()
        pipe.transformer.return_value = MagicMock(sample=torch.randn(1, 4, 64, 64))

        return pipe

    def test_direct_distill_loss_runs(self, mock_distill_pipe):
        """Test DirectDistillLoss executes without error."""
        from llm_dit.training.losses import DirectDistillLoss

        input_latents = torch.randn(1, 4, 64, 64)
        prompt_embeds = torch.randn(1, 77, 2560)

        loss = DirectDistillLoss(
            mock_distill_pipe,
            input_latents=input_latents,
            prompt_embeds=prompt_embeds,
            num_inference_steps=8,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0


class TestConsistencyLoss:
    """Test ConsistencyLoss function."""

    @pytest.fixture
    def mock_consistency_pipe(self):
        """Create mock pipeline for consistency loss testing."""
        pipe = MagicMock()

        # Mock scheduler
        pipe.scheduler = MagicMock()
        pipe.scheduler.timesteps = torch.linspace(1.0, 0.0, 10)

        def add_noise(sample, noise, timestep):
            t = timestep / 1000.0 if timestep.max() > 1 else timestep
            return sample * (1 - t) + noise * t

        pipe.scheduler.add_noise = add_noise

        # Mock transformer
        pipe.transformer = MagicMock()
        pipe.transformer.return_value = MagicMock(sample=torch.randn(1, 4, 64, 64))

        return pipe

    def test_consistency_loss_runs(self, mock_consistency_pipe):
        """Test ConsistencyLoss executes without error."""
        from llm_dit.training.losses import ConsistencyLoss

        input_latents = torch.randn(1, 4, 64, 64)
        prompt_embeds = torch.randn(1, 77, 2560)

        loss = ConsistencyLoss(
            mock_consistency_pipe,
            input_latents=input_latents,
            prompt_embeds=prompt_embeds,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_consistency_loss_with_timestep_pairs(self, mock_consistency_pipe):
        """Test ConsistencyLoss with explicit timestep pairs."""
        from llm_dit.training.losses import ConsistencyLoss

        input_latents = torch.randn(1, 4, 64, 64)
        prompt_embeds = torch.randn(1, 77, 2560)
        timestep_pairs = torch.tensor([0.3, 0.4])

        loss = ConsistencyLoss(
            mock_consistency_pipe,
            input_latents=input_latents,
            prompt_embeds=prompt_embeds,
            timestep_pairs=timestep_pairs,
        )

        assert isinstance(loss, torch.Tensor)


# ============================================================================
# Training Runner Tests
# ============================================================================

class TestTrainingRunnerUtilities:
    """Test training runner utility functions."""

    def test_create_optimizer(self):
        """Test create_optimizer creates AdamW optimizer."""
        from llm_dit.training.runner import create_optimizer
        from llm_dit.training.config import TrainingConfig

        # Simple mock model
        model = MagicMock()
        model.trainable_parameters = lambda: iter([torch.nn.Parameter(torch.randn(10, 10))])

        config = TrainingConfig(learning_rate=1e-4, weight_decay=0.01)
        optimizer = create_optimizer(model, config)

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults["lr"] == 1e-4
        assert optimizer.defaults["weight_decay"] == 0.01

    @pytest.mark.parametrize("scheduler_name,expected_type", [
        ("constant", torch.optim.lr_scheduler.ConstantLR),
        ("cosine", torch.optim.lr_scheduler.CosineAnnealingLR),
        ("linear", torch.optim.lr_scheduler.LinearLR),
    ])
    def test_create_lr_scheduler(self, scheduler_name, expected_type):
        """Test create_lr_scheduler creates correct scheduler type."""
        from llm_dit.training.runner import create_lr_scheduler
        from llm_dit.training.config import TrainingConfig

        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(10))], lr=1e-4)
        config = TrainingConfig(lr_scheduler=scheduler_name, warmup_steps=0)

        scheduler = create_lr_scheduler(optimizer, config, num_training_steps=100)
        assert isinstance(scheduler, expected_type)

    def test_create_lr_scheduler_invalid(self):
        """Test create_lr_scheduler raises on invalid scheduler."""
        from llm_dit.training.runner import create_lr_scheduler
        from llm_dit.training.config import TrainingConfig

        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(10))], lr=1e-4)
        config = TrainingConfig(lr_scheduler="invalid_scheduler")

        with pytest.raises(ValueError, match="Unknown lr_scheduler"):
            create_lr_scheduler(optimizer, config, num_training_steps=100)

    def test_create_dataloader(self):
        """Test create_dataloader creates DataLoader with correct settings."""
        from llm_dit.training.runner import create_dataloader
        from llm_dit.training.config import TrainingConfig
        from torch.utils.data import Dataset

        class DummyDataset(Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {"data": torch.randn(4)}

        dataset = DummyDataset()
        config = TrainingConfig(batch_size=2, num_workers=0)

        dataloader = create_dataloader(dataset, config)
        assert len(dataloader) == 5  # 10 items / batch_size 2


# ============================================================================
# Base Training Module Tests
# ============================================================================

class TestBaseTrainingModule:
    """Test BaseTrainingModule class."""

    def test_trainable_parameters(self):
        """Test trainable_parameters yields only trainable params."""
        from llm_dit.training.base import BaseTrainingModule
        from llm_dit.training.config import TrainingConfig

        class DummyModule(BaseTrainingModule):
            def __init__(self):
                super().__init__(TrainingConfig())
                self.trainable = nn.Parameter(torch.randn(10))
                self.frozen = nn.Parameter(torch.randn(10))
                self.frozen.requires_grad = False

            def get_pipeline_inputs(self, data):
                return {}

            def forward(self, data):
                return torch.tensor(0.0)

        module = DummyModule()
        trainable_params = list(module.trainable_parameters())

        assert len(trainable_params) == 1
        assert trainable_params[0] is module.trainable

    def test_num_trainable_params(self):
        """Test num_trainable_params counts correctly."""
        from llm_dit.training.base import BaseTrainingModule
        from llm_dit.training.config import TrainingConfig

        class DummyModule(BaseTrainingModule):
            def __init__(self):
                super().__init__(TrainingConfig())
                self.layer = nn.Linear(10, 5)
                self.layer.weight.requires_grad = True
                self.layer.bias.requires_grad = True

            def get_pipeline_inputs(self, data):
                return {}

            def forward(self, data):
                return torch.tensor(0.0)

        module = DummyModule()
        num_params = module.num_trainable_params()

        # Linear(10, 5) has 10*5 + 5 = 55 params
        assert num_params == 55

    def test_freeze_all(self):
        """Test freeze_all freezes all parameters."""
        from llm_dit.training.base import BaseTrainingModule
        from llm_dit.training.config import TrainingConfig

        class DummyModule(BaseTrainingModule):
            def __init__(self):
                super().__init__(TrainingConfig())
                self.layer1 = nn.Linear(10, 5)
                self.layer2 = nn.Linear(5, 3)

            def get_pipeline_inputs(self, data):
                return {}

            def forward(self, data):
                return torch.tensor(0.0)

        module = DummyModule()
        module.freeze_all()

        for param in module.parameters():
            assert param.requires_grad is False

    def test_export_trainable_state_dict(self):
        """Test export_trainable_state_dict filters correctly."""
        from llm_dit.training.base import BaseTrainingModule
        from llm_dit.training.config import TrainingConfig

        class DummyModule(BaseTrainingModule):
            def __init__(self):
                super().__init__(TrainingConfig())
                self.trainable = nn.Linear(10, 5)
                self.frozen = nn.Linear(5, 3)
                self.frozen.requires_grad_(False)

            def get_pipeline_inputs(self, data):
                return {}

            def forward(self, data):
                return torch.tensor(0.0)

        module = DummyModule()
        full_state = module.state_dict()
        filtered_state = module.export_trainable_state_dict(full_state)

        # Should only contain trainable layer params
        assert "trainable.weight" in filtered_state
        assert "trainable.bias" in filtered_state
        assert "frozen.weight" not in filtered_state
        assert "frozen.bias" not in filtered_state

    def test_export_trainable_state_dict_remove_prefix(self):
        """Test export_trainable_state_dict removes prefix."""
        from llm_dit.training.base import BaseTrainingModule
        from llm_dit.training.config import TrainingConfig

        class DummyModule(BaseTrainingModule):
            def __init__(self):
                super().__init__(TrainingConfig())
                self.pipe = MagicMock()
                self.pipe.transformer = nn.Linear(10, 5)

            def trainable_param_names(self):
                return {"pipe.transformer.weight", "pipe.transformer.bias"}

            def get_pipeline_inputs(self, data):
                return {}

            def forward(self, data):
                return torch.tensor(0.0)

        module = DummyModule()
        state_dict = {
            "pipe.transformer.weight": torch.randn(5, 10),
            "pipe.transformer.bias": torch.randn(5),
        }
        filtered = module.export_trainable_state_dict(
            state_dict,
            remove_prefix="pipe.transformer.",
        )

        assert "weight" in filtered
        assert "bias" in filtered
        assert "pipe.transformer.weight" not in filtered
