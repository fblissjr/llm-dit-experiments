"""
Anti-aliased blur downsampling for latent upsampler.

Last Updated: 2026-01-18

BlurDownsample applies a separable binomial filter (Gaussian approximation)
before strided downsampling to prevent aliasing artifacts. Uses Pascal's
triangle coefficients for the kernel.
"""

import math

import torch
import torch.nn.functional as F
from einops import rearrange


class BlurDownsample(torch.nn.Module):
    """
    Anti-aliased spatial downsampling by integer stride using a fixed separable binomial kernel.

    Applies only on H,W dimensions. For dims=3, operates per-frame.

    The binomial kernel uses coefficients from Pascal's triangle (e.g., [1,4,6,4,1]
    for kernel_size=5) to approximate a Gaussian blur, providing smooth anti-aliasing.

    Args:
        dims: Number of spatial dimensions (2 or 3). For dims=3, blur is applied
            per-frame on H,W only (temporal dimension unchanged).
        stride: Downsampling stride factor. Must be >= 1.
        kernel_size: Size of the blur kernel. Must be odd and >= 3. Defaults to 5.
    """

    kernel: torch.Tensor  # Type hint for registered buffer

    def __init__(self, dims: int, stride: int, kernel_size: int = 5) -> None:
        super().__init__()
        assert dims in (2, 3)
        assert isinstance(stride, int)
        assert stride >= 1
        assert kernel_size >= 3
        assert kernel_size % 2 == 1
        self.dims = dims
        self.stride = stride
        self.kernel_size = kernel_size

        # Build separable binomial kernel using binomial coefficients from Pascal's triangle.
        # For kernel_size=5: [1, 4, 6, 4, 1] (row 4 of Pascal's triangle)
        # The 2D kernel is the outer product, normalized to sum to 1.
        k = torch.tensor([math.comb(kernel_size - 1, i) for i in range(kernel_size)])
        k2d = k[:, None] @ k[None, :]
        k2d = (k2d / k2d.sum()).float()  # shape (kernel_size, kernel_size)
        self.register_buffer("kernel", k2d[None, None, :, :])  # (1, 1, kernel_size, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return x

        if self.dims == 2:
            return self._apply_2d(x)
        else:
            # dims == 3: apply per-frame on H,W
            b, c, f, h, w = x.shape
            x = rearrange(x, "b c f h w -> (b f) c h w")
            x = self._apply_2d(x)
            h2, w2 = x.shape[-2:]
            x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)
            return x

    def _apply_2d(self, x2d: torch.Tensor) -> torch.Tensor:
        c = x2d.shape[1]
        # Expand kernel for depthwise convolution (same kernel per channel)
        weight = self.kernel.expand(c, 1, self.kernel_size, self.kernel_size)
        # Depthwise conv with groups=c applies same kernel independently per channel
        x2d = F.conv2d(
            x2d,
            weight=weight.to(x2d.dtype),
            bias=None,
            stride=self.stride,
            padding=self.kernel_size // 2,
            groups=c,
        )
        return x2d
