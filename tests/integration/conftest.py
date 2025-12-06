"""
Fixtures for integration tests.

Provides test client with mocked encoder for format-prompt tests.
"""

import pytest
import torch
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, Mock

from llm_dit.backends.protocol import EncodingOutput
from llm_dit.templates.loader import Template


@pytest.fixture
def mock_encoder():
    """Mock encoder with encode and format methods."""
    encoder = MagicMock()
    encoder.device = torch.device("cpu")

    # Mock templates
    encoder.templates = {
        "default": Template(
            name="default",
            content="Generate natural images",
            thinking_content="",
        ),
    }

    # Mock encode method
    def mock_encode(
        prompt,
        template=None,
        system_prompt=None,
        thinking_content=None,
        assistant_content=None,
        force_think_block=False,
        remove_quotes=False,
        return_padded=False,
    ):
        embeddings = torch.randn(77, 2560)
        attention_mask = torch.ones(77, dtype=torch.bool)
        return EncodingOutput(
            embeddings=[embeddings],
            attention_masks=[attention_mask],
            token_counts=[77],
        )

    encoder.encode = Mock(side_effect=mock_encode)

    # Mock _build_conversation for format-prompt endpoint
    def mock_build_conversation(
        prompt,
        template=None,
        system_prompt=None,
        thinking_content=None,
        assistant_content=None,
        force_think_block=False,
        remove_quotes=False,
    ):
        from llm_dit.conversation import Conversation
        return Conversation.simple(
            user_prompt=prompt,
            system_prompt=system_prompt or "",
            thinking_content=thinking_content or "",
            assistant_content=assistant_content or "",
            force_think_block=force_think_block,
            remove_quotes=remove_quotes,
        )

    encoder._build_conversation = Mock(side_effect=mock_build_conversation)

    # Mock formatter
    from llm_dit.conversation import Qwen3Formatter
    encoder.formatter = Qwen3Formatter()

    # Mock backend with tokenizer for token counting
    mock_backend = MagicMock()
    mock_backend.tokenizer = Mock()
    mock_backend.tokenizer.return_value = {"input_ids": torch.zeros(1, 42)}
    encoder.backend = mock_backend

    return encoder


@pytest.fixture
def client(mock_encoder):
    """Test client with mocked encoder for format-prompt tests."""
    from web.server import app
    import web.server as server_module

    # Store original values
    original_pipeline = server_module.pipeline
    original_encoder = server_module.encoder
    original_encoder_only_mode = server_module.encoder_only_mode

    # Inject mocked encoder
    server_module.pipeline = None
    server_module.encoder = mock_encoder
    server_module.encoder_only_mode = True

    yield TestClient(app)

    # Restore original values
    server_module.pipeline = original_pipeline
    server_module.encoder = original_encoder
    server_module.encoder_only_mode = original_encoder_only_mode
