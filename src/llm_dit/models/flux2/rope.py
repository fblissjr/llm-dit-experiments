"""
4D Rotary Position Embeddings (RoPE) for FLUX.2.

Last Updated: 2026-01-24

Implements 4-dimensional rotary position embeddings for FLUX.2 Klein models.
Unlike LTX-2's 3D RoPE (t, h, w), FLUX.2 uses 4D coordinates (t, h, w, l)
where l is a linear position index.

Key Differences from LTX-2:
- 4D positional encoding instead of 3D
- Different theta (2000 vs 10000)
- Text uses only the 'l' coordinate (t=h=w=0, l=sequence_position)
- Images use h,w for spatial position with t=0 and l=0 (NOT linear index)

Ported from: coderef/flux2/src/flux2/model.py

Usage:
    from llm_dit.models.flux2.rope import EmbedND, apply_rope, create_image_ids, create_text_ids

    # Create position embedder
    pe_embedder = EmbedND(dim=128, theta=2000, axes_dim=[32, 32, 32, 32])

    # Create position IDs for image (64x64 latent = 4096 tokens)
    img_ids = create_image_ids(batch_size=1, height=64, width=64)  # [1, 4096, 4]

    # Create position IDs for text (512 tokens)
    txt_ids = create_text_ids(batch_size=1, seq_len=512)  # [1, 512, 4]

    # Get positional embeddings
    pe_img = pe_embedder(img_ids)  # [1, 1, 4096, 128//2, 2, 2]
    pe_txt = pe_embedder(txt_ids)  # [1, 1, 512, 128//2, 2, 2]
"""

import logging
import torch
from einops import rearrange
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class EmbedND(nn.Module):
    """
    N-dimensional positional embedding using rotary embeddings.

    Applies RoPE independently to each coordinate axis and concatenates results.
    For FLUX.2, this is 4D: (t, h, w, l).

    Args:
        dim: Total positional embedding dimension (typically head_dim)
        theta: Base frequency for RoPE (2000 for FLUX.2)
        axes_dim: List of dimensions for each axis (e.g., [32, 32, 32, 32])
    """

    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        """
        Compute positional embeddings from position IDs.

        Args:
            ids: Position coordinates [B, seq_len, n_axes] where n_axes=4 for FLUX.2

        Returns:
            Rotary embeddings [B, 1, seq_len, dim//2, 2, 2] containing cos/sin matrices
        """
        # Apply rope to each axis independently and concatenate
        # ids shape: [B, seq_len, 4] for (t, h, w, l)
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(len(self.axes_dim))],
            dim=-3,  # Concatenate along the frequency dimension
        )
        # Output shape: [B, seq_len, dim//2, 2, 2] -> [B, 1, seq_len, dim//2, 2, 2]
        return emb.unsqueeze(1)


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    """
    Compute rotary position embeddings for a single axis.

    Creates cos/sin embeddings in a 2x2 rotation matrix format:
    [cos(θ), -sin(θ)]
    [sin(θ),  cos(θ)]

    Args:
        pos: Position indices for one axis [B, seq_len]
        dim: Embedding dimension for this axis (must be even)
        theta: Base frequency

    Returns:
        Rotation matrices [B, seq_len, dim//2, 2, 2]
    """
    assert dim % 2 == 0, f"Dimension must be even, got {dim}"

    # Create frequency scale: theta^(-2i/dim) for i in [0, dim/2)
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)  # [dim/2]

    # Compute angles: pos * omega
    # pos: [B, seq_len], omega: [dim/2] -> out: [B, seq_len, dim/2]
    out = torch.einsum("...n,d->...nd", pos, omega)

    # Create rotation matrix components
    # Stack as [cos, -sin, sin, cos] then reshape to 2x2
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)],
        dim=-1,
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)

    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Uses matrix multiplication with the 2x2 rotation matrices to apply
    the rotation in the complex plane.

    Args:
        xq: Query tensor [B, H, seq_len, head_dim]
        xk: Key tensor [B, H, seq_len, head_dim]
        freqs_cis: Rotation matrices [B, 1, seq_len, head_dim//2, 2, 2]

    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as inputs
    """
    # Reshape to [B, H, seq_len, head_dim//2, 2] for matrix multiplication
    # (pair adjacent dimensions)
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)

    # Apply rotation: [cos, -sin; sin, cos] @ [x1; x2]
    # freqs_cis[..., 0] = [cos, -sin], freqs_cis[..., 1] = [sin, cos]
    # Result: [cos*x1 - sin*x2, sin*x1 + cos*x2]
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]

    # Reshape back to [B, H, seq_len, head_dim]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def create_image_ids(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Create 4D position IDs for image latents.

    For images:
    - t (temporal) = 0 (single image, no temporal dimension)
    - h (height) = row index [0, height)
    - w (width) = column index [0, width)
    - l (linear) = 0 (images use h,w for spatial position, NOT l)

    Args:
        batch_size: Batch size
        height: Latent height (after patchify, e.g., 64 for 1024px)
        width: Latent width (after patchify, e.g., 64 for 1024px)
        device: Target device
        dtype: Target dtype

    Returns:
        Position IDs [B, height*width, 4]
    """
    logger.info(f"[PosIDs:Image] Creating position IDs for height={height}, width={width}")

    # Create coordinate grids
    img_ids = torch.zeros(height, width, 4, device=device, dtype=dtype)

    # t = 0 for all positions (images have no temporal dimension)
    img_ids[..., 0] = 0

    # h = row index (broadcast across width)
    img_ids[..., 1] = torch.arange(height, device=device, dtype=dtype)[:, None]

    # w = column index (broadcast across height)
    img_ids[..., 2] = torch.arange(width, device=device, dtype=dtype)[None, :]

    # l = 0 for all image tokens (images use h,w for spatial, text uses l for sequence)
    img_ids[..., 3] = 0

    # Flatten spatial dimensions and add batch dimension
    img_ids = img_ids.view(-1, 4)  # [H*W, 4]
    img_ids = img_ids.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H*W, 4]

    total_tokens = height * width
    logger.info(f"[PosIDs:Image] Created {total_tokens} tokens")
    logger.info(f"[PosIDs:Image] First 3 IDs: {img_ids[0, :3].tolist()}")
    logger.info(f"[PosIDs:Image] Last 3 IDs: {img_ids[0, -3:].tolist()}")

    return img_ids


def create_text_ids(
    batch_size: int,
    seq_len: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Create 4D position IDs for text tokens.

    For text:
    - t (temporal) = 0
    - h (height) = 0
    - w (width) = 0
    - l (linear) = token position [0, seq_len)

    Text only uses the linear coordinate, with all spatial coordinates set to 0.

    Args:
        batch_size: Batch size
        seq_len: Text sequence length
        device: Target device
        dtype: Target dtype

    Returns:
        Position IDs [B, seq_len, 4]
    """
    txt_ids = torch.zeros(batch_size, seq_len, 4, device=device, dtype=dtype)

    # Only the linear coordinate is non-zero for text
    txt_ids[..., 3] = torch.arange(seq_len, device=device, dtype=dtype)

    return txt_ids


def create_reference_ids(
    batch_size: int,
    ref_heights: list[int],
    ref_widths: list[int],
    t_scale: int = 10,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Create 4D position IDs for reference images (for image conditioning).

    Reference images get different t coordinates to distinguish them from
    the generation target. Each reference image gets t = t_scale * (1 + idx).
    Like main images, l=0 for all reference tokens (spatial info via h,w only).

    Args:
        batch_size: Batch size
        ref_heights: List of reference image heights (in latent space)
        ref_widths: List of reference image widths (in latent space)
        t_scale: Scale factor for temporal separation (default 10)
        device: Target device
        dtype: Target dtype

    Returns:
        Position IDs [B, total_ref_tokens, 4]
    """
    all_ref_ids = []

    for idx, (h, w) in enumerate(zip(ref_heights, ref_widths)):
        t_offset = t_scale * (1 + idx)  # t = 10, 20, 30, ... for refs

        ref_ids = torch.zeros(h, w, 4, device=device, dtype=dtype)
        ref_ids[..., 0] = t_offset  # Different t for each reference
        ref_ids[..., 1] = torch.arange(h, device=device, dtype=dtype)[:, None]
        ref_ids[..., 2] = torch.arange(w, device=device, dtype=dtype)[None, :]
        ref_ids[..., 3] = 0  # l=0 for reference images (spatial info via h,w only)

        all_ref_ids.append(ref_ids.view(-1, 4))

    # Concatenate all reference IDs
    if all_ref_ids:
        ref_ids = torch.cat(all_ref_ids, dim=0)  # [total_ref_tokens, 4]
        ref_ids = ref_ids.unsqueeze(0).expand(batch_size, -1, -1)  # [B, total_ref_tokens, 4]
        return ref_ids
    else:
        return torch.zeros(batch_size, 0, 4, device=device, dtype=dtype)


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    """
    Scaled dot-product attention with rotary position embeddings.

    This is the core attention computation used in FLUX.2. It applies RoPE
    to Q and K before computing attention.

    Args:
        q: Query tensor [B, H, seq_len, head_dim]
        k: Key tensor [B, H, seq_len, head_dim]
        v: Value tensor [B, H, seq_len, head_dim]
        pe: Positional embeddings from EmbedND

    Returns:
        Attention output [B, seq_len, H*head_dim]
    """
    # Apply rotary position embeddings
    q, k = apply_rope(q, k, pe)

    # Ensure tensors are contiguous for Flash Attention 2 dispatch
    # SDPA only uses FA2 when tensors are contiguous in memory
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Compute scaled dot-product attention (dispatches to FA2 when available)
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    # Reshape from [B, H, L, D] to [B, L, H*D]
    x = rearrange(x, "B H L D -> B L (H D)")

    return x
