"""
Wan UMT5-XXL Text Encoder.

Last Updated: 2026-01-12

Custom T5 encoder implementation matching Wan's weight format.
Based on Wan2GP/models/wan/modules/t5.py
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def fp16_clamp(x: torch.Tensor) -> torch.Tensor:
    """Clamp tensor to avoid fp16 overflow."""
    if x.dtype == torch.float16 and torch.isinf(x).any():
        clamp = torch.finfo(x.dtype).max - 1000
        x = torch.clamp(x, min=-clamp, max=clamp)
    return x


class T5LayerNorm(nn.Module):
    """T5-style RMSNorm."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.type_as(self.weight)
        return self.weight * x


class GELU(nn.Module):
    """GELU activation (T5 uses tanh approximation)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class T5Attention(nn.Module):
    """T5-style multi-head attention."""

    def __init__(self, dim: int, dim_attn: int, num_heads: int, dropout: float = 0.1):
        assert dim_attn % num_heads == 0
        super().__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads

        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        pos_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [B, L1, C]
        context: [B, L2, C] or None
        mask: [B, L2] or [B, L1, L2] or None
        """
        context = x if context is None else context
        b, n, c = x.size(0), self.num_heads, self.head_dim

        q = self.q(x).view(b, -1, n, c)
        k = self.k(context).view(b, -1, n, c)
        v = self.v(context).view(b, -1, n, c)

        attn_bias = x.new_zeros(b, n, q.size(1), k.size(1))
        if pos_bias is not None:
            attn_bias += pos_bias
        if mask is not None:
            assert mask.ndim in [2, 3]
            mask = mask.view(b, 1, 1, -1) if mask.ndim == 2 else mask.unsqueeze(1)
            attn_bias.masked_fill_(mask == 0, torch.finfo(x.dtype).min)

        # T5 does not use scaling
        attn = torch.einsum('binc,bjnc->bnij', q, k) + attn_bias
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum('bnij,bjnc->binc', attn, v)

        x = x.reshape(b, -1, n * c)
        x = self.o(x)
        x = self.dropout(x)
        return x


class T5FeedForward(nn.Module):
    """T5-style gated feed-forward."""

    def __init__(self, dim: int, dim_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.dim_ffn = dim_ffn

        self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), GELU())
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x) * self.gate(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class T5RelativeEmbedding(nn.Module):
    """T5-style relative position embedding."""

    def __init__(
        self,
        num_buckets: int,
        num_heads: int,
        bidirectional: bool = True,
        max_dist: int = 128,
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist

        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, lq: int, lk: int) -> torch.Tensor:
        device = self.embedding.weight.device
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - \
            torch.arange(lq, device=device).unsqueeze(1)
        rel_pos = self._relative_position_bucket(rel_pos)
        rel_pos_embeds = self.embedding(rel_pos)
        rel_pos_embeds = rel_pos_embeds.permute(2, 0, 1).unsqueeze(0)  # [1, N, Lq, Lk]
        return rel_pos_embeds.contiguous()

    def _relative_position_bucket(self, rel_pos: torch.Tensor) -> torch.Tensor:
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).long() * num_buckets
            rel_pos = torch.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))

        max_exact = num_buckets // 2
        rel_pos_large = max_exact + (torch.log(rel_pos.float() / max_exact) /
                                     math.log(self.max_dist / max_exact) *
                                     (num_buckets - max_exact)).long()
        rel_pos_large = torch.min(
            rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1))
        rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets


class T5SelfAttentionBlock(nn.Module):
    """T5 encoder self-attention block."""

    def __init__(
        self,
        dim: int,
        dim_attn: int,
        dim_ffn: int,
        num_heads: int,
        num_buckets: int,
        shared_pos: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.shared_pos = shared_pos

        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pos_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        e = pos_bias if self.shared_pos else self.pos_embedding(x.size(1), x.size(1))
        x = fp16_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(x + self.ffn(self.norm2(x)))
        return x


class WanT5Encoder(nn.Module):
    """
    UMT5-XXL Encoder matching Wan's weight format.

    Weight keys: token_embedding.weight, blocks.X.{norm1,attn,norm2,ffn,pos_embedding}
    """

    def __init__(
        self,
        vocab_size: int = 256384,
        dim: int = 4096,
        dim_attn: int = 4096,
        dim_ffn: int = 10240,
        num_heads: int = 64,
        num_layers: int = 24,
        num_buckets: int = 32,
        shared_pos: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True) if shared_pos else None
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            T5SelfAttentionBlock(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                                 shared_pos, dropout) for _ in range(num_layers)
        ])
        self.norm = T5LayerNorm(dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        input_ids: [B, L] token IDs
        attention_mask: [B, L] mask (1 for valid, 0 for padding)

        Returns: [B, L, dim] hidden states
        """
        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        e = self.pos_embedding(x.size(1), x.size(1)) if self.pos_embedding else None

        for block in self.blocks:
            x = block(x, attention_mask, pos_bias=e)

        x = self.norm(x)
        x = self.dropout(x)
        return x


class WanTextEncoder(nn.Module):
    """
    High-level wrapper for Wan UMT5-XXL text encoder.

    Provides a simple interface for text encoding with tokenization.
    """

    def __init__(
        self,
        max_length: int = 512,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.max_length = max_length
        self.dtype = dtype
        self.model = WanT5Encoder()
        self.tokenizer = None

    def load_tokenizer(self, tokenizer_path: str):
        """Load tokenizer from path."""
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def load_weights(self, checkpoint_path: str):
        """Load model weights from .safetensors file."""
        from pathlib import Path
        from safetensors.torch import load_file as load_safetensors

        path = Path(checkpoint_path)
        if path.suffix != ".safetensors":
            raise ValueError(
                f"Expected .safetensors file, got {path.suffix}. "
                f"Convert with: uv run python scripts/convert_to_safetensors.py {path}"
            )
        state_dict = load_safetensors(str(path))
        self.model.load_state_dict(state_dict)

    def encode(
        self,
        text: str,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text to embeddings.

        Args:
            text: Input text string
            device: Target device

        Returns:
            embeddings: [1, seq_len, 4096] text embeddings
            mask: [1, seq_len] attention mask
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_tokenizer() first.")

        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # Move to model device (ignore device param - always use model's device)
        model_device = next(self.model.parameters()).device
        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)

        # Encode with autocast (only for CUDA devices)
        with torch.no_grad():
            device_type = model_device.type
            autocast_enabled = device_type == 'cuda'
            with torch.amp.autocast(device_type=device_type, dtype=self.dtype, enabled=autocast_enabled):
                embeddings = self.model(input_ids, attention_mask)

        return embeddings, attention_mask

    def forward(self, text: str) -> torch.Tensor:
        """Encode text and return embeddings."""
        embeddings, _ = self.encode(text)
        return embeddings


def load_wan_text_encoder(
    checkpoint_path: str,
    tokenizer_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = 'cuda',
) -> WanTextEncoder:
    """
    Load Wan UMT5-XXL text encoder.

    Args:
        checkpoint_path: Path to models_t5_umt5-xxl-enc-bf16.safetensors
        tokenizer_path: Path to tokenizer directory (google/umt5-xxl)
        dtype: Model dtype
        device: Target device

    Returns:
        Loaded WanTextEncoder
    """
    encoder = WanTextEncoder(dtype=dtype)
    encoder.load_tokenizer(tokenizer_path)
    encoder.load_weights(checkpoint_path)
    encoder.model = encoder.model.to(dtype=dtype, device=device)
    return encoder
